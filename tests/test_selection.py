"""
Variable selection per analyte: the built-in methods against the original code
(`multical.var_selection`), what the engine keeps, the model file and the
checks on methods written by users.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

from ._cases import ANALYTES, make_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api  # noqa: E402
from multical.api import export  # noqa: E402
from multical.api.calibrate import selection_seed  # noqa: E402
from multical.api.contract import ContractError  # noqa: E402
from multical.api.registry import Model  # noqa: E402
from multical.api.selection import vip_scores  # noqa: E402
from multical.models.pls import PLS  # noqa: E402
from multical.var_selection import (  # noqa: E402
    calculate_rmsecv_fast, calculate_vip, run_pso, run_simulated_annealing, select_k_ftest,
)

KMAX = 6
FOLDS = 5


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class _SerialPool:
    """Stands in for multiprocessing.Pool in the original PSO: same results, no worker processes."""

    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, func, items):
        return [func(item) for item in items]


class SelectionCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, cls.X, cls.Y = make_dataset()
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L")

    def tearDown(self):
        for item in ("t_sonda", "t_posicoes", "t_curto", "t_vazio", "t_texto", "t_fora", "t_tudo"):
            api.SELECTORS.unregister(item)

    def _config(self, selector=None, params=None, analytes=("gl",), model=None, **extra):
        return api.RunConfig(
            pretreatment=[], model=model or {"id": "pls", "params": {"max_k": KMAX}},
            analytes=None if analytes is None else list(analytes),
            variable_selection={"id": selector, "params": params or {}} if selector else None, **extra)

    def _run(self, *args, **kwargs):
        progress = kwargs.pop("progress", None)
        cancel = kwargs.pop("cancel", None)
        return api.run_calibration(self.dataset, self._config(*args, **kwargs), progress=progress, cancel=cancel)

    def _y(self, name):
        return self.Y[:, [ANALYTES.index(name)]]


class LegacyEquivalence(SelectionCase):
    """Same numbers and same choices as the original code when it is given the values of ONE analyte."""

    def test_error_of_a_choice_is_the_rmsecv_of_the_original_code(self):
        rng = np.random.RandomState(4)
        masks = [rng.rand(self.wl.size) < share for share in (0.7, 0.4, 0.1)] + [np.ones(self.wl.size, dtype=bool)]
        seen = {}

        @api.register(api.SELECTORS)
        class Probe(api.Selector):
            id = "t_sonda"
            label = "Sonda"

            def select(self, X, y, error, rng, progress):
                seen["errors"] = [error(mask) for mask in masks]
                seen["curve"] = error.curve(masks[0])
                seen["folds"], seen["axis"], seen["X"] = error.folds, error.wavelengths, X
                return masks[1]

        result = self._run("t_sonda")
        y = self._y("gl")
        for mask, value in zip(masks, seen["errors"]):
            legacy = calculate_rmsecv_fast(self.X[:, mask], y, KMAX, FOLDS, "venetian")[0]
            self.assertAlmostEqual(value / legacy, 1.0, places=9)
        np.testing.assert_allclose(seen["curve"], calculate_rmsecv_fast(self.X[:, masks[0]], y, KMAX, FOLDS, "venetian")[2],
                                   rtol=1e-9)
        self.assertEqual(len(seen["folds"]), FOLDS)
        np.testing.assert_array_equal(seen["axis"], self.wl)
        self.assertEqual(seen["X"].shape, self.X.shape)
        # What is reported afterwards is the regular validation of the chosen points, not the quick one.
        res = result.per_analyte["gl"]
        np.testing.assert_array_equal(res.selected, masks[1])
        np.testing.assert_allclose(res.rmsecv, calculate_rmsecv_fast(self.X[:, masks[1]], y, KMAX, FOLDS, "venetian")[2],
                                   rtol=1e-12)
        self.assertAlmostEqual(res.full_rmsecv, self._run().per_analyte["gl"].best_rmsecv, places=12)

    def test_vip_scores_and_search_for_the_limit(self):
        y = self._y("xy")
        curve = calculate_rmsecv_fast(self.X, y, KMAX, FOLDS, "venetian")[2]
        k = select_k_ftest(curve, self.X.shape[0])
        legacy_scores = _quiet(calculate_vip, PLS(), self.X, y, k)
        np.testing.assert_allclose(vip_scores(self.X, y.ravel(), k), legacy_scores, rtol=1e-10)

        # The search of scripts/calibration/run_variable_selection.py, with the values of one analyte.
        best_mask, best_error = np.ones(self.wl.size, dtype=bool), 1e9
        for limit in np.arange(0.1, 1.5, 0.1):
            mask = legacy_scores >= limit
            if mask.sum() < 2:
                continue
            value = calculate_rmsecv_fast(self.X[:, mask], y, KMAX, FOLDS, "venetian")[0]
            if value < best_error:
                best_error, best_mask = value, mask
        res = self._run("vip", {"max_k": KMAX}, analytes=("xy",)).per_analyte["xy"]
        np.testing.assert_array_equal(res.selected, best_mask)
        self.assertLess(res.n_selected, self.wl.size)

        fixed = self._run("vip", {"max_k": KMAX, "auto": False, "threshold": 1.0}, analytes=("xy",)).per_analyte["xy"]
        np.testing.assert_array_equal(fixed.selected, legacy_scores >= 1.0)
        with self.assertRaisesRegex(ValueError, "Reduza o limite"):
            self._run("vip", {"max_k": KMAX, "auto": False, "threshold": 50.0}, analytes=("xy",))

    def test_pso_makes_the_same_choice(self):
        y = self._y("gl")
        for params in ({"n_particles": 8, "max_iter": 6},
                       {"n_particles": 6, "max_iter": 5, "lambda_sparsity": 0.5, "p_init": 0.3, "w": 0.8, "c1": 1.2, "c2": 1.7}):
            with self.subTest(params=params):
                legacy = dict(params)
                with mock.patch("multical.var_selection.pso.Pool", _SerialPool):
                    mask, _ = run_pso(self.X, y, KMAX, FOLDS, "venetian", seed=selection_seed(3, 0), verbose=False, **legacy)
                res = self._run("pso", params, seed=3).per_analyte["gl"]
                np.testing.assert_array_equal(res.selected, mask)
                self.assertTrue(2 <= res.n_selected < self.wl.size)

    def test_simulated_annealing_makes_the_same_choice(self):
        y = self._y("cb")
        params = {"max_iter": 60, "initial_temp": 0.3, "alpha": 0.95}
        np.random.seed(selection_seed(11, 0))
        mask, _ = _quiet(run_simulated_annealing, self.X, y, KMAX, FOLDS, "venetian", **params)
        res = self._run("sa", params, analytes=("cb",), seed=11).per_analyte["cb"]
        np.testing.assert_array_equal(res.selected, mask)

    def test_quick_validation_of_the_pls_gives_the_regular_numbers(self):
        rng = np.random.RandomState(1)
        model = api.MODELS.get("pls")
        folds = list(api.SPLITTERS.get("kfold").split(30, folds=5, order="venetian"))
        y = self.Y[:, 1]
        for X in (self.X, rng.normal(size=(30, 2000)) + np.outer(y, rng.normal(size=2000)), self.X[:, :20]):
            Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
            quick, regular = model.cv_folds(Xs, y, folds, KMAX), Model.cv_folds(model, Xs, y, folds, KMAX)
            self.assertEqual(quick.shape, (KMAX, 30))
            np.testing.assert_allclose(quick, regular, rtol=1e-7, atol=1e-9)


class EngineWithSelection(SelectionCase):
    def test_each_analyte_gets_its_own_points(self):
        result = self._run("vip", {"max_k": KMAX}, analytes=None)
        masks = [result.per_analyte[name].selected for name in ANALYTES]
        for mask in masks:
            self.assertEqual((mask.dtype, mask.shape), (np.dtype(bool), (self.wl.size,)))
        self.assertFalse(np.array_equal(masks[0], masks[1]) and np.array_equal(masks[1], masks[2]))
        # Each analyte's band is where its own model looks: the three bands are centred at 4800, 5900 and 7000.
        for mask, centre in zip(masks, (4800.0, 5900.0, 7000.0)):
            self.assertTrue(mask[np.argmin(np.abs(self.wl - centre))])
        plain = self._run(analytes=None)
        for name in ANALYTES:
            res = result.per_analyte[name]
            self.assertIsNone(plain.per_analyte[name].selected)
            self.assertIsNone(plain.per_analyte[name].n_selected)
            self.assertAlmostEqual(res.full_rmsecv, plain.per_analyte[name].best_rmsecv, places=12)
            self.assertEqual(len(res.candidates), min(KMAX, res.n_selected))

    def test_predictions_use_the_chosen_points_and_survive_the_model_file(self):
        result = self._run("pso", {"n_particles": 6, "max_iter": 4}, analytes=("cb", "xy"))
        new = self.X[:6] * 1.02 + 0.001
        direct = api.predict(result, new, self.wl)
        for j, name in enumerate(("cb", "xy")):
            res = result.per_analyte[name]
            np.testing.assert_allclose(api.predict(result, self.X, self.wl)[:, j], res.y_cal[res.best_index], rtol=1e-9)
            # The same estimate from the coefficients and the chosen columns alone.
            Xs = ((new - result.x_mean) / result.x_std)[:, res.selected]
            by_hand = (Xs @ res.model_state["B"]).ravel() * res.model_state["y_std"] + res.model_state["y_mean"]
            np.testing.assert_allclose(direct[:, j], by_hand, rtol=1e-12)
        with tempfile.TemporaryDirectory() as tmp:
            path = api.save_model(result, os.path.join(tmp, "modelo"))
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            model = api.load_model(path)
            plain_path = api.save_model(self._run(), os.path.join(tmp, "simples"))
            with zipfile.ZipFile(plain_path) as archive:
                plain_manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertIsNone(api.load_model(plain_path).per_analyte["gl"].selected)
        self.assertEqual((manifest["format"], plain_manifest["format"]), (2, 1))     # older programs refuse what they would misread
        self.assertEqual(manifest["config"]["variable_selection"]["id"], "pso")
        self.assertEqual(manifest["validation"]["cb"]["n_points"], result.per_analyte["cb"].n_selected)
        for name in ("cb", "xy"):
            np.testing.assert_array_equal(model.per_analyte[name].selected, result.per_analyte[name].selected)
        np.testing.assert_allclose(api.predict(model, new, self.wl), direct, rtol=1e-12)
        report = api.predict_report(model, api.Dataset(new, self.wl))
        np.testing.assert_allclose(report.Y, direct, rtol=1e-12)

    def test_same_seed_repeats_the_choice_and_another_seed_changes_it(self):
        params = {"n_particles": 6, "max_iter": 4}
        first = self._run("pso", params, seed=5).per_analyte["gl"].selected
        np.testing.assert_array_equal(first, self._run("pso", params, seed=5).per_analyte["gl"].selected)
        self.assertFalse(np.array_equal(first, self._run("pso", params, seed=6).per_analyte["gl"].selected))
        self.assertNotEqual(selection_seed(5, 0), selection_seed(5, 1))
        self.assertIsNone(selection_seed(None, 0))

    def test_progress_and_cancel(self):
        seen = []
        self._run("sa", {"max_iter": 25}, analytes=("cb", "gl"), progress=lambda f, m: seen.append((f, m)))
        fractions = [f for f, _ in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(0.0 <= fractions[0] and fractions[-1] == 1.0)
        self.assertTrue(any("escolhendo os pontos do espectro – tentativa 3 de 25" in m for _, m in seen))
        self.assertTrue(any(m.startswith("gl: validação") for _, m in seen))

        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 40

        with self.assertRaises(api.CalibrationCancelled):
            self._run("pso", {"n_particles": 10, "max_iter": 50}, cancel=cancel)
        self.assertLess(calls["n"], 60)      # stopped at the next evaluation, not at the end of the search

    def test_works_with_models_that_have_no_complexity_and_with_other_validations(self):
        svr = self._run("vip", {"max_k": KMAX}, model={"id": "svr", "params": {"tune": False}}).per_analyte["gl"]
        self.assertTrue(2 <= svr.n_selected <= self.wl.size)
        self.assertEqual(len(svr.rmsecv), 1)
        spa = self._run("sa", {"max_iter": 10}, model={"id": "spa", "params": {"max_k": 4}}).per_analyte["gl"]
        self.assertEqual(spa.model_state["selected"].max() < spa.n_selected, True)      # positions inside the chosen points
        groups = np.array(["exp1"] * 10 + ["exp2"] * 12 + ["exp3"] * 8)
        data = api.Dataset(self.X, self.wl, self.Y, analytes=list(ANALYTES), groups=groups)
        config = self._config("vip", {"max_k": KMAX}, validation={"id": "loexo", "params": {}})
        self.assertIsNotNone(api.run_calibration(data, config).per_analyte["gl"].selected)
        holdout = self._config("vip", {"max_k": 4}, model={"id": "pls", "params": {"max_k": 4}},
                               validation={"id": "holdout", "params": {"fraction": 0.3}})
        self.assertIsNotNone(api.run_calibration(self.dataset, holdout).per_analyte["gl"].selected)

    def test_configuration_round_trip(self):
        config = self._config("pso", {"max_iter": 7})
        resolved = config.resolved()
        self.assertEqual(resolved.variable_selection["params"]["n_particles"], 30)      # defaults filled in
        self.assertEqual(resolved.variable_selection["params"]["max_iter"], 7)
        data = config.to_dict()
        self.assertEqual((data["format"], data["variable_selection"]["id"]), (2, "pso"))
        again = api.RunConfig.from_json(config.to_json())
        self.assertEqual(again.to_dict(), data)
        plain = self._config().to_dict()
        self.assertEqual(plain["format"], 1)
        self.assertNotIn("variable_selection", plain)
        self.assertIsNone(api.RunConfig.from_dict(plain).variable_selection)
        with self.assertRaisesRegex(ValueError, "mais novo"):
            api.RunConfig.from_dict({**data, "format": 3})
        with self.assertRaisesRegex(KeyError, "não existe"):
            self._config("inexistente").resolved()
        with self.assertRaisesRegex(ValueError, "desconhecido"):
            self._config("pso", {"particulas": 3}).resolved()

    def test_assessment_and_exported_files(self):
        result = self._run("vip", {"max_k": KMAX}, analytes=("cb", "gl"))
        items = api.assess(result)
        for item in items:
            n_points = result.per_analyte[item.analyte].n_selected
            self.assertIn(f"{n_points} de {self.wl.size} pontos do espectro", item.summary)
        notes = api.selection_notes(result)
        self.assertEqual(len(notes), 3)
        self.assertIn("com todos os 150 pontos era ", notes[0])
        self.assertIn("tende a parecer menor", notes[-1])
        self.assertEqual(api.selection_notes(self._run()), [])
        with tempfile.TemporaryDirectory() as tmp:
            written = export.export_run(result, os.path.join(tmp, "execucao"))
            names = {os.path.relpath(p, os.path.join(tmp, "execucao")).replace("\\", "/") for p in written}
            self.assertTrue({"pontos_escolhidos.csv", "figuras/pontos_escolhidos_cb.png",
                             "figuras/pontos_escolhidos_gl.png", "relatorio.pdf"} <= names)
            for path in written:
                self.assertGreater(os.path.getsize(path), 0)
            with open(os.path.join(tmp, "execucao", "pontos_escolhidos.csv"), encoding="utf-8-sig") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines[0], "posicao_no_eixo;cb_usa;gl_usa")
            self.assertEqual(len(lines), 1 + self.wl.size)
            self.assertEqual(sum(line.split(";")[1] == "sim" for line in lines[1:]), result.per_analyte["cb"].n_selected)
            with open(os.path.join(tmp, "execucao", "resumo.csv"), encoding="utf-8-sig") as handle:
                header, first = [line.split(";") for line in handle.read().splitlines()[:2]]
            self.assertEqual(header[-3:], ["pontos_usados", "pontos_do_espectro", "erro_validacao_todos_os_pontos"])
            self.assertEqual(first[-3:-1], [str(result.per_analyte["cb"].n_selected), "150"])
            self.assertIn(",", first[-1])


class SelectionsWrittenByUsers(SelectionCase):
    def _selector(self, item_id, returns):
        @api.register(api.SELECTORS)
        class Custom(api.Selector):
            id = item_id
            label = f"Teste {item_id}"

            def select(self, X, y, error, rng, progress):
                return returns(X)

        return item_id

    def test_positions_are_accepted_like_a_mask(self):
        res = self._run(self._selector("t_posicoes", lambda X: np.array([3, 10, 11, 40]))).per_analyte["gl"]
        np.testing.assert_array_equal(np.flatnonzero(res.selected), [3, 10, 11, 40])
        self.assertEqual(list(res.candidates), [1, 2, 3, 4])        # never more complexity than points in use
        everything = self._run(self._selector("t_tudo", lambda X: np.ones(X.shape[1], dtype=bool))).per_analyte["gl"]
        np.testing.assert_allclose(everything.rmsecv, self._run().per_analyte["gl"].rmsecv, rtol=1e-12)

    def test_wrong_answers_are_explained(self):
        cases = (
            ("t_curto", lambda X: np.ones(X.shape[1] - 1, dtype=bool), "formato \\(149,\\); esperado um item por ponto"),
            ("t_vazio", lambda X: np.zeros(X.shape[1], dtype=bool), "nenhum ponto do espectro foi escolhido"),
            ("t_texto", lambda X: "todos", "vetor de verdadeiro/falso"),
            ("t_fora", lambda X: np.array([0, 150]), "entre 0 e 149"),
        )
        for item_id, returns, message in cases:
            with self.subTest(case=item_id):
                with self.assertRaisesRegex(ContractError, f"“Teste {item_id}”.*{message}"):
                    self._run(self._selector(item_id, returns))


if __name__ == "__main__":
    unittest.main()
