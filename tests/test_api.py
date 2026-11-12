"""
Tests for the service layer `multical.api`.

Equivalence tests compare the new layer with the existing numerical code:
- pretreatments against the stored output of `apply_pretreatment`;
- per-analyte PLS validation against `calculate_rmsecv_fast` run on one analyte;
- the final model against `train_and_save_model_pls` + `load_and_predict_pls`.
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ._cases import ANALYTES, KMAX, PRETREATMENT, PRETREATMENT_STEPS, make_dataset

from multical import api
from multical.api import plots
from multical.core.saving import load_and_predict_pls, train_and_save_model_pls
from multical.io.txt_pair import read_txt_pairs
from multical.preprocessing.pipeline import apply_pretreatment
from multical.var_selection import calculate_rmsecv_fast, select_k_ftest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden", "reference.npz")
SRC = os.path.join(os.path.dirname(HERE), "src")
RTOL = 1e-7


def _quiet(func, *args, **kwargs):
    """Call legacy code that prints progress, discarding the output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _config(order="venetian", seed=0, **extra):
    return api.RunConfig(
        pretreatment=api.recipe_from_legacy(PRETREATMENT),
        model={"id": "pls", "params": {"max_k": KMAX}},
        validation={"id": "kfold", "params": {"folds": 5, "order": order}},
        seed=seed, **extra,
    )


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, cls.X, cls.Y = make_dataset()
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L")


class PretreatmentEquivalence(ApiTestCase):
    def test_every_legacy_step_gives_the_same_spectra(self):
        with np.load(GOLDEN_PATH) as golden:
            for label, step in PRETREATMENT_STEPS.items():
                with self.subTest(step=label):
                    recipe = api.recipe_from_legacy([step])
                    Xp, wlp, _ = api.fit_recipe(recipe, self.X[:8], self.wl)
                    np.testing.assert_allclose(Xp, golden[f"pre_{label}_X"], rtol=1e-9, atol=1e-12)
                    np.testing.assert_allclose(wlp, golden[f"pre_{label}_wl"], rtol=1e-12)

    def test_state_from_calibration_is_reused_on_new_samples(self):
        cal, new = self.X[:20], self.X[20:]
        for step_id, expected in (
            ("autoscale", (new - cal.mean(axis=0)) / cal.std(axis=0, ddof=1)),
            ("meancenter", new - cal.mean(axis=0)),
        ):
            with self.subTest(step=step_id):
                recipe = [{"id": step_id, "params": {}}]
                _, _, states = api.fit_recipe(recipe, cal, self.wl)
                out, _ = api.apply_recipe(recipe, states, new, self.wl)
                np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

        recipe = [{"id": "msc", "params": {}}]
        _, _, states = api.fit_recipe(recipe, cal, self.wl)
        np.testing.assert_allclose(states[0]["reference"], cal.mean(axis=0))
        one, _ = api.apply_recipe(recipe, states, new[:1], self.wl)
        many, _ = api.apply_recipe(recipe, states, new, self.wl)
        np.testing.assert_allclose(one[0], many[0], rtol=1e-10)  # a sample is corrected the same way alone or in a batch

    def test_every_registered_pretreatment_runs_with_defaults(self):
        for item in api.PRETREATMENTS:
            with self.subTest(pretreatment=item.id):
                params = {"start": 7500.0, "end": 7900.0} if item.id == "blctr" else {}
                Xp, wlp, _ = api.fit_recipe([{"id": item.id, "params": params}], self.X, self.wl)
                self.assertEqual(Xp.shape, (self.X.shape[0], wlp.size))
                self.assertTrue(np.all(np.isfinite(Xp)))


class CalibrationEquivalence(ApiTestCase):
    def test_validation_error_matches_legacy_pls_per_analyte(self):
        Xp, _ = _quiet(apply_pretreatment, PRETREATMENT, self.X.copy(), self.wl.copy(), plot=False)
        for order in ("venetian", "consecutive", "random"):
            result = api.run_calibration(self.dataset, _config(order=order, seed=7))
            for j, name in enumerate(ANALYTES):
                with self.subTest(order=order, analyte=name):
                    np.random.seed(7)
                    _, _, curve = calculate_rmsecv_fast(Xp, self.Y[:, [j]], KMAX, folds=5, cv_type=order)
                    res = result.per_analyte[name]
                    np.testing.assert_allclose(res.rmsecv, curve, rtol=RTOL)
                    self.assertEqual(res.best_k, select_k_ftest(curve, Xp.shape[0]))

    def test_final_model_matches_saved_legacy_model(self):
        cal = self.dataset.subset(np.arange(22))
        result = api.run_calibration(cal, _config())
        new_pred = api.predict(result, self.X[22:], self.wl)

        Xp, wlp = _quiet(apply_pretreatment, PRETREATMENT, self.X.copy(), self.wl.copy(), plot=False)
        best_k = [result.per_analyte[a].best_k for a in ANALYTES]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pkl")
            _quiet(train_and_save_model_pls, Xp[:22], self.Y[:22], wlp, best_k, path)
            legacy_pred, _ = load_and_predict_pls(Xp[22:], wlp, path)
        np.testing.assert_allclose(new_pred, legacy_pred, rtol=RTOL)

    def test_calibration_error_and_shapes(self):
        result = api.run_calibration(self.dataset, _config())
        n = self.X.shape[0]
        self.assertTrue(result.validated.all())
        for name in ANALYTES:
            res = result.per_analyte[name]
            self.assertEqual(res.y_cv.shape, (KMAX, n))
            self.assertEqual(res.y_cal.shape, (KMAX, n))
            self.assertTrue(np.all(np.diff(res.rmsec) <= 1e-12))       # calibration error never grows with k
            self.assertTrue(np.all(res.rmsec <= res.rmsecv + 1e-12))
            # the kept model is the one whose calibration predictions were reported
            fitted = api.predict(result, self.X, self.wl)[:, ANALYTES.index(name)]
            np.testing.assert_allclose(fitted, res.y_cal[res.best_index], rtol=1e-6)

    def test_min_rule_picks_lowest_validation_error(self):
        result = api.run_calibration(self.dataset, _config(selection="min"))
        for res in result.per_analyte.values():
            self.assertEqual(res.best_index, int(np.argmin(res.rmsecv)))

    def test_leave_one_out_equals_kfold_with_n_parts(self):
        n = self.X.shape[0]
        base = dict(pretreatment=api.recipe_from_legacy(PRETREATMENT), model={"id": "pls", "params": {"max_k": 5}})
        loo = api.run_calibration(self.dataset, api.RunConfig(validation={"id": "loo", "params": {}}, **base))
        kfold = api.run_calibration(
            self.dataset,
            api.RunConfig(validation={"id": "kfold", "params": {"folds": n, "order": "consecutive"}}, **base),
        )
        for name in ANALYTES:
            np.testing.assert_allclose(loo.per_analyte[name].rmsecv, kfold.per_analyte[name].rmsecv, rtol=1e-12)

    def test_holdout_is_reproducible_and_validates_only_the_held_part(self):
        def run(seed):
            cfg = api.RunConfig(
                pretreatment=api.recipe_from_legacy(PRETREATMENT), model={"id": "pls", "params": {"max_k": 5}},
                validation={"id": "holdout", "params": {"fraction": 0.25}}, seed=seed,
            )
            return api.run_calibration(self.dataset, cfg)

        a, b, c = run(3), run(3), run(4)
        self.assertEqual(int(a.validated.sum()), int(np.floor(self.X.shape[0] * 0.25)))
        np.testing.assert_array_equal(a.validated, b.validated)
        np.testing.assert_array_equal(a.per_analyte["gl"].rmsecv, b.per_analyte["gl"].rmsecv)
        self.assertFalse(np.array_equal(a.validated, c.validated))
        self.assertTrue(np.all(np.isnan(a.per_analyte["gl"].y_cv[:, ~a.validated])))

    def test_subset_of_analytes(self):
        result = api.run_calibration(self.dataset, _config(analytes=["gl"]))
        self.assertEqual(result.analytes, ["gl"])
        self.assertEqual(api.predict(result, self.X[:3], self.wl).shape, (3, 1))


class ErrorsAndControl(ApiTestCase):
    def test_clear_errors(self):
        cases = {
            "max_k too large": lambda: api.run_calibration(
                self.dataset, api.RunConfig(model={"id": "pls", "params": {"max_k": 40}})),
            "constant analyte": lambda: api.run_calibration(
                api.Dataset(self.X, self.wl, np.ones((self.X.shape[0], 1)), analytes=["x"])),
            "missing reference": lambda: api.run_calibration(api.Dataset(self.X, self.wl)),
            "unknown analyte": lambda: api.run_calibration(self.dataset, _config(analytes=["zz"])),
            "bad parameter value": lambda: api.RunConfig(
                validation={"id": "kfold", "params": {"folds": 1}}).resolved(),
            "unknown parameter": lambda: api.RunConfig(model={"id": "pls", "params": {"nope": 1}}).resolved(),
            "empty cut": lambda: api.fit_recipe([{"id": "cut", "params": {"ranges": [[1, 2]]}}], self.X, self.wl),
        }
        for label, call in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    call()
        with self.assertRaises(KeyError):
            api.RunConfig(model={"id": "does_not_exist"}).resolved()

    def test_prediction_refuses_a_different_spectral_grid(self):
        result = api.run_calibration(self.dataset, _config())
        with self.assertRaises(ValueError):
            api.predict(result, self.X[:2, :-1], self.wl[:-1])
        with self.assertRaises(ValueError):
            api.predict(result, self.X[:2], self.wl + 5.0)

    def test_progress_and_cancel(self):
        seen = []
        api.run_calibration(self.dataset, _config(), progress=lambda frac, msg: seen.append(frac))
        self.assertEqual(len(seen), len(ANALYTES) * (5 + 1))
        self.assertEqual(seen, sorted(seen))
        self.assertAlmostEqual(seen[-1], 1.0)

        calls = {"n": 0}

        def cancel_after_three():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(api.CalibrationCancelled):
            api.run_calibration(self.dataset, _config(), cancel=cancel_after_three)


class ConfigAndRegistry(ApiTestCase):
    def test_config_survives_json(self):
        cfg = _config(order="random", seed=11, analytes=["gl", "xy"]).resolved()
        again = api.RunConfig.from_json(cfg.to_json()).resolved()
        self.assertEqual(again.to_dict(), cfg.to_dict())
        a = api.run_calibration(self.dataset, cfg)
        b = api.run_calibration(self.dataset, again)
        np.testing.assert_array_equal(a.per_analyte["gl"].rmsecv, b.per_analyte["gl"].rmsecv)

    def test_defaults_are_filled(self):
        cfg = api.RunConfig().resolved()
        self.assertEqual(cfg.model, {"id": "pls", "params": {"max_k": 15}})
        self.assertEqual(cfg.validation, {"id": "kfold", "params": {"folds": 5, "order": "venetian"}})

    def test_new_methods_plug_in_without_touching_the_engine(self):
        @api.pretreatment(id="test_double", label="Dobrar", params=[api.Param("factor", "Fator", "float", 2.0, min=0)])
        def double(X, wl, factor):
            return X * factor, wl

        @api.register(api.MODELS)
        class MeanModel(api.Model):
            id = "test_mean"
            label = "Média"

            def fit(self, X, y):
                return {"mean": float(np.mean(y))}

            def predict(self, state, X):
                return np.full(X.shape[0], state["mean"])

        self.addCleanup(api.PRETREATMENTS.unregister, "test_double")
        self.addCleanup(api.MODELS.unregister, "test_mean")

        cfg = api.RunConfig(
            pretreatment=[{"id": "test_double", "params": {}}],
            model={"id": "test_mean", "params": {}},
            validation={"id": "kfold", "params": {"folds": 5}},
        )
        result = api.run_calibration(self.dataset, cfg)
        np.testing.assert_allclose(result.X_pre, self.X * 2.0)
        res = result.per_analyte["gl"]
        self.assertEqual(res.rmsecv.shape, (1,))
        self.assertIsNone(res.best_k)
        self.assertLess(res.best_r2cv, 0.0)  # predicting the mean is the no-skill baseline
        np.testing.assert_allclose(api.predict(result, self.X[:2], self.wl)[:, 1], self.Y[:, 1].mean())

        with self.assertRaises(ValueError):
            api.MODELS.register(MeanModel())

    def test_importing_the_api_does_not_load_pyplot(self):
        code = "import sys, multical.api, multical.api.plots; sys.exit('matplotlib.pyplot' in sys.modules)"
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=dict(os.environ, PYTHONPATH=SRC),
                                  capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TxtPairReader(ApiTestCase):
    def _write_pair(self, folder, name, rows, drop_last_reference=False):
        times = np.arange(len(rows)) * 30.0
        ref = np.column_stack([times, self.Y[rows]])
        if drop_last_reference:
            ref = ref[:-1]
        ref_path, spec_path = os.path.join(folder, f"{name}_refe.txt"), os.path.join(folder, f"{name}_nonda.txt")
        np.savetxt(ref_path, ref, fmt="%.10g", delimiter="\t")
        header = "Time\t" + "\t".join(f"{w:.10g}" for w in self.wl)
        np.savetxt(spec_path, np.column_stack([times, self.X[rows]]), fmt="%.12g", delimiter="\t",
                   header=header, comments="")
        return ref_path, spec_path

    def test_reads_the_legacy_text_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [self._write_pair(tmp, "exp1", np.arange(0, 12)), self._write_pair(tmp, "exp2", np.arange(12, 30))]
            data = read_txt_pairs(pairs, analytes=ANALYTES, units="g/L")
        np.testing.assert_allclose(data.X, self.X, rtol=1e-10)
        np.testing.assert_allclose(data.Y, self.Y, rtol=1e-9)
        np.testing.assert_allclose(data.wavelengths, self.wl, rtol=1e-9)
        self.assertEqual(list(np.unique(data.groups)), ["exp1_nonda", "exp2_nonda"])
        self.assertEqual(data.sample_ids[0], "exp1_nonda_1")
        self.assertIn("txt_pair", api.IMPORTERS)

    def test_row_count_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair = self._write_pair(tmp, "exp1", np.arange(0, 12), drop_last_reference=True)
            with self.assertRaises(ValueError):
                read_txt_pairs([pair])


class Plots(ApiTestCase):
    def test_plot_functions_return_figures(self):
        from matplotlib.figure import Figure

        result = api.run_calibration(self.dataset, _config())
        figures = [
            plots.plot_spectra(self.wl, self.X),
            plots.plot_error_curve(result, "gl"),
            plots.plot_predicted_vs_measured(result, "gl"),
        ]
        for fig in figures:
            self.assertIsInstance(fig, Figure)
            with tempfile.TemporaryDirectory() as tmp:
                fig.savefig(os.path.join(tmp, "fig.png"))


if __name__ == "__main__":
    unittest.main()
