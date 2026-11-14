"""
Q × T² of the calibration samples and of estimated spectra, and the LOESS
smoothing computed for all spectra at once, both against the original code.
"""
import os
import sys
import tempfile
import time
import unittest
import warnings

import numpy as np

from ._cases import ANALYTES, make_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api  # noqa: E402
from multical.api import export, plots  # noqa: E402
from multical.api.domain import check_domain  # noqa: E402
from multical.api.pretreatments import loess_rows  # noqa: E402
from multical.models.pls import PLS  # noqa: E402
from multical.preprocessing.loess import loess  # noqa: E402
from multical.utils import zscore_matlab_style  # noqa: E402

KMAX = 6


class InfluenceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, X, cls.Y = make_dataset()
        cls.X = X.copy()
        cls.X[7, 60:70] += 0.02                 # one spectrum with a band the others do not have
        ids = [f"amostra_{i + 1}" for i in range(30)]
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L", sample_ids=ids)
        cls.recipe = [{"id": "sg", "params": {"radius": 3, "poly_order": 2, "deriv": 1}}]

    def _run(self, model=None, **extra):
        config = api.RunConfig(pretreatment=self.recipe, model=model or {"id": "pls", "params": {"max_k": KMAX}}, **extra)
        return api.run_calibration(self.dataset, config)


class CalibrationSamples(InfluenceCase):
    def test_pls_gives_the_influence_plot_of_the_scripts(self):
        result = self._run()
        for j, name in enumerate(ANALYTES):
            data = api.influence(result, name)
            # multical.core.outliers.evaluate_and_plot_outliers, line by line, for this analyte
            Xnorm, _, _ = zscore_matlab_style(result.X_pre)
            Ynorm, _, _ = zscore_matlab_style(self.Y[:, [j]])
            k = result.per_analyte[name].best_k
            pls = PLS()
            _, T, P, _, _, _, _, _ = pls.nipals(Xnorm, Ynorm, k)
            T2, t2_limit, Q, q_limit = pls.compute_outlier_metrics(Xnorm, T, P, conf_level=0.95)
            np.testing.assert_allclose(data.t2, T2, rtol=1e-12)
            np.testing.assert_allclose(data.q, Q, rtol=1e-12)
            self.assertEqual((data.t2_limit, data.q_limit), (t2_limit, q_limit))
            self.assertEqual((data.source, data.components, data.confidence), ("pls", k, 0.95))
            np.testing.assert_array_equal(data.outside, (T2 > t2_limit) | (Q > q_limit))
            self.assertEqual(data.sample_ids, self.dataset.sample_ids)

    def test_the_odd_spectrum_is_found_and_named(self):
        result = self._run()
        data = api.influence(result, "xy")
        self.assertTrue(data.outside[7])
        self.assertGreater(data.q[7], data.q_limit)                 # a band nobody else has: residual, not T²
        self.assertLess(data.t2[7], data.t2_limit)
        flagged = data.flagged()
        self.assertEqual(flagged[0], "amostra_8")                   # the farthest comes first
        self.assertEqual(set(flagged), {self.dataset.sample_ids[i] for i in np.flatnonzero(data.outside)})
        figure = plots.plot_influence(result, "xy")
        axis = figure.axes[0]
        self.assertIn(f"xy: {int(data.outside.sum())} de 30 amostras além dos limites", axis.get_title())
        self.assertIn("variáve", axis.get_title())
        self.assertIn("amostra_8", [text.get_text() for text in axis.texts])
        self.assertEqual([line.get_label() for line in axis.lines], ["Limite de T² (95%)", "Limite de Q (95%)"])

    def test_models_without_latent_variables_use_the_summary_of_the_spectra(self):
        result = self._run(model={"id": "svr", "params": {"tune": False}})
        data = api.influence(result, "cb")
        t2, q, inside = check_domain(result.domain, result.X_pre)      # SVR receives the spectra unscaled
        np.testing.assert_allclose(data.t2, t2)
        np.testing.assert_allclose(data.q, q)
        np.testing.assert_array_equal(data.outside, ~inside)
        self.assertEqual((data.source, data.confidence), ("pca", 0.99))
        self.assertIn("componente", plots.plot_influence(result, "cb").axes[0].get_title())
        np.testing.assert_array_equal(api.influence(result, "gl").t2, data.t2)     # one summary for every analyte

    def test_with_a_variable_selection_only_the_chosen_points_count(self):
        result = self._run(analytes=["cb"], variable_selection={"id": "vip", "params": {"max_k": KMAX}})
        res = result.per_analyte["cb"]
        self.assertLess(res.n_selected, result.wavelengths.size)
        data = api.influence(result, "cb")
        Xnorm, _, _ = zscore_matlab_style(result.X_pre[:, res.selected])
        Ynorm, _, _ = zscore_matlab_style(self.Y[:, [0]])
        pls = PLS()
        _, T, P, _, _, _, _, _ = pls.nipals(Xnorm, Ynorm, res.best_k)
        np.testing.assert_allclose(data.q, pls.compute_outlier_metrics(Xnorm, T, P, conf_level=0.95)[2], rtol=1e-12)

    def test_exported_folder_has_the_figures_and_the_table(self):
        result = self._run(analytes=["cb", "gl"])
        with tempfile.TemporaryDirectory() as tmp:
            written = export.export_run(result, os.path.join(tmp, "execucao"))
            names = {os.path.relpath(p, os.path.join(tmp, "execucao")).replace("\\", "/") for p in written}
            self.assertTrue({"amostras_q_t2.csv", "figuras/q_x_t2_cb.png", "figuras/q_x_t2_gl.png"} <= names)
            with open(os.path.join(tmp, "execucao", "amostras_q_t2.csv"), encoding="utf-8-sig") as handle:
                rows = [line.split(";") for line in handle.read().splitlines()]
        self.assertEqual(rows[0], ["amostra", "cb_T2", "cb_Q", "cb_alem_dos_limites", "gl_T2", "gl_Q", "gl_alem_dos_limites"])
        self.assertEqual((rows[1][0], len(rows)), ("limite", 2 + 30))
        data = api.influence(result, "gl")
        self.assertAlmostEqual(float(rows[1][5].replace(",", ".")), data.q_limit, delta=1e-5 * data.q_limit)
        self.assertEqual([row[6] for row in rows[2:]], ["sim" if out else "" for out in data.outside])


class EstimatedSpectra(InfluenceCase):
    def test_estimates_carry_the_limits_and_are_drawn_against_them(self):
        result = self._run()
        new = np.vstack([self.X[:5], self.X[5:8] + 0.05 * np.sin(self.wl / 40.0)])     # three spectra with a new feature
        groups = np.array(["lote A"] * 5 + ["lote B"] * 3)
        report = api.predict_report(result, api.Dataset(new, self.wl, groups=groups))
        self.assertEqual((report.t2_limit, report.q_limit), (result.domain["t2_limit"], result.domain["q_limit"]))
        np.testing.assert_array_equal(report.inside, (report.t2 <= report.t2_limit) & (report.q <= report.q_limit))
        self.assertTrue(report.inside[:5].all() and not report.inside[5:].any())
        axis = plots.plot_prediction_domain(report).axes[0]
        self.assertIn("3 de 8 espectros além dos limites", axis.get_title())
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        self.assertTrue({"lote A", "lote B", "Limite de T²", "Limite de Q"} <= set(labels))

        with tempfile.TemporaryDirectory() as tmp:                                    # the same from a model file
            saved = api.load_model(api.save_model(result, os.path.join(tmp, "m")))
        again = api.predict_report(saved, api.Dataset(new, self.wl, groups=groups))
        self.assertEqual((again.t2_limit, again.q_limit), (report.t2_limit, report.q_limit))
        saved.domain = None                                                           # a model file without the summary
        blind = api.predict_report(saved, api.Dataset(new, self.wl))
        self.assertIsNone(blind.t2_limit)
        self.assertEqual(plots.plot_prediction_domain(blind).axes, [])                # a sentence instead of a plot


class QuickLoess(unittest.TestCase):
    def test_same_numbers_as_the_original_function(self):
        rng = np.random.RandomState(0)
        regular = np.linspace(4000.0, 8000.0, 90)
        axes = {"regular": regular, "irregular": np.sort(regular + rng.uniform(-10, 10, 90)),
                "decrescente": regular[::-1].copy(), "fora de ordem": rng.permutation(regular), "curto": regular[:6]}
        for label, x in axes.items():
            Y = np.sin(np.outer(rng.uniform(1, 3, 3), x / 700.0)) + 0.05 * rng.normal(size=(3, x.size)) + 2.0
            for alpha in (0.05, 0.2, 1.0):
                for order in (1, 2):
                    with self.subTest(axis=label, alpha=alpha, order=order):
                        expected = np.vstack([loess(x, row, alpha, order) for row in Y])
                        np.testing.assert_allclose(loess_rows(x, Y, alpha, order), expected, rtol=1e-9, atol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # the original function warns about repeated positions
            x = np.repeat(regular[:15], 4)
            Y = rng.normal(size=(2, 60))
            np.testing.assert_allclose(loess_rows(x, Y, 0.4, 2), np.vstack([loess(x, row, 0.4, 2) for row in Y]),
                                       rtol=1e-8, atol=1e-10)
        with self.assertRaises(ValueError):
            loess_rows(regular, np.ones((1, 90)), 0.0, 2)
        np.testing.assert_array_equal(loess_rows(np.full(5, 3.0), np.arange(5.0)[None, :], 0.5, 2), np.arange(5.0)[None, :])

    def test_recipe_step_is_the_original_and_keeps_the_input(self):
        wl, X, _ = make_dataset()
        before = X.copy()
        out, axis, _ = api.fit_recipe([{"id": "loess", "params": {"alpha": 0.2, "order": 2}}], X, wl)
        np.testing.assert_allclose(out, np.vstack([loess(wl, row, 0.2, 2) for row in X]), rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(X, before)
        np.testing.assert_array_equal(axis, wl)

    def test_spectra_of_the_real_size_take_seconds_not_minutes(self):
        rng = np.random.RandomState(1)
        x = np.linspace(1200.0, 10000.0, 8801)
        Y = np.sin(np.outer(rng.uniform(1, 3, 40), x / 1500.0)) + 0.01 * rng.normal(size=(40, 8801))
        start = time.perf_counter()
        smooth = loess_rows(x, Y, 0.2, 2)
        took = time.perf_counter() - start
        # The original function needs about 3.6 s for each of these spectra (over 2 minutes for the 40).
        self.assertLess(took, 60.0)
        self.assertLess(np.abs(np.diff(smooth, axis=1)).mean(), np.abs(np.diff(Y, axis=1)).mean())      # it did smooth
        np.testing.assert_allclose(smooth[3, 4000:4003], loess(x[3000:5000], Y[3, 3000:5000], 0.2 * 8801 / 2000, 2)[1000:1003],
                                   rtol=1e-9)        # a window cut out of the long axis gives the same values


if __name__ == "__main__":
    unittest.main()
