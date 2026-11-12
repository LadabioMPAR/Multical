"""Tests for estimating new samples: the report, the inside/outside check and models saved by the old scripts."""
import contextlib
import io
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

from ._cases import ANALYTES, KMAX, PRETREATMENT, make_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from multical import api  # noqa: E402
from multical.api.domain import check_domain, fit_domain  # noqa: E402
from multical.core.saving import load_and_predict_pls, train_and_save_model_pls  # noqa: E402
from multical.io.txt_pair import read_txt_spectra  # noqa: E402
from multical.preprocessing.pipeline import apply_pretreatment  # noqa: E402


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class PredictionCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wl, cls.X, cls.Y = make_dataset()
        cls.dataset = api.Dataset(cls.X, cls.wl, cls.Y, analytes=list(ANALYTES), units="g/L")
        cls.config = api.RunConfig(pretreatment=api.recipe_from_legacy(PRETREATMENT),
                                   model={"id": "pls", "params": {"max_k": KMAX}})
        cls.result = api.run_calibration(cls.dataset, cls.config)


class Domain(PredictionCase):
    def test_calibration_samples_are_inside_and_strange_spectra_are_not(self):
        domain = self.result.domain
        self.assertGreaterEqual(domain["loadings"].shape[0], 1)
        Xm = api.calibrate.model_input(self.result, self.X, self.wl)
        _, _, inside = check_domain(domain, Xm)
        self.assertTrue(inside.all())

        report = api.predict_report(self.result, self.dataset)
        self.assertEqual(report.n_outside, 0)

        strange = self.X[:4].copy()
        strange[:, 60:75] += 0.5 * np.ptp(self.X)           # a band that no calibration spectrum has
        far = api.predict_report(self.result, api.Dataset(strange, self.wl))
        self.assertEqual(far.n_outside, 4)
        self.assertTrue(np.all(far.q > domain["q_limit"]))

        extreme = self.X.mean(axis=0) + 6 * (self.X[:3] - self.X.mean(axis=0))   # same shape, far from the centre
        self.assertEqual(api.predict_report(self.result, api.Dataset(extreme, self.wl)).n_outside, 3)

    def test_fit_domain_on_tiny_sets(self):
        rng = np.random.RandomState(0)
        for n in (3, 6, 12):
            domain = fit_domain(rng.normal(size=(n, 20)))
            self.assertTrue(np.isfinite(domain["q_limit"]) and domain["q_limit"] > 0)
            self.assertEqual(domain["loadings"].shape[1], 20)


class Report(PredictionCase):
    def test_report_matches_predict_whatever_the_chunk(self):
        new = api.Dataset(self.X[::2] * 1.02, self.wl, sample_ids=[f"n{i}" for i in range(15)], times=np.arange(15.0))
        expected = api.predict(self.result, new.X, self.wl)
        calls = []
        report = api.predict_report(self.result, new, progress=lambda f, m: calls.append((f, m)), chunk=4)
        np.testing.assert_allclose(report.Y, expected, rtol=1e-12)
        np.testing.assert_allclose(api.predict_report(self.result, new, chunk=500).Y, expected, rtol=1e-12)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1], (1.0, "15 de 15 espectros"))
        self.assertEqual((report.analytes, report.units, report.sample_ids[3]), (list(ANALYTES), "g/L", "n3"))
        np.testing.assert_allclose(report.typical_error, [self.result.per_analyte[a].best_rmsecv for a in ANALYTES])
        with self.assertRaises(api.CalibrationCancelled):
            api.predict_report(self.result, new, cancel=lambda: True)
        with self.assertRaises(ValueError):
            api.predict_report(self.result, api.Dataset(self.X[:, :-2], self.wl[:-2]))

    def test_saved_model_gives_the_same_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = api.load_model(api.save_model(self.result, os.path.join(tmp, "m")))
            strange = self.X[:6].copy()
            strange[3:, 60:75] += 0.5 * np.ptp(self.X)
            data = api.Dataset(strange, self.wl, times=np.arange(6.0), groups=np.array(["lote"] * 6))
            live, saved = api.predict_report(self.result, data), api.predict_report(model, data)
            np.testing.assert_allclose(saved.Y, live.Y, rtol=1e-12)
            np.testing.assert_allclose(saved.t2, live.t2, rtol=1e-10)
            np.testing.assert_array_equal(saved.inside, [True, True, True, False, False, False])
            np.testing.assert_allclose(saved.typical_error, live.typical_error)

            path = api.export_predictions(saved, os.path.join(tmp, "previsoes.csv"))
            with open(path, encoding="utf-8-sig") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(len(lines), 7)
            self.assertTrue(lines[0].startswith("amostra;experimento;tempo_min;cb_g/L;cb_erro_tipico;gl_g/L"))
            self.assertEqual(lines[1].split(";")[:3], ["amostra_1", "lote", "0"])
            self.assertEqual(lines[1].split(";")[-3], "dentro do que o modelo conhece")
            self.assertIn("espectro diferente", lines[6].split(";")[-3])
            self.assertNotIn(".", lines[1].split(";")[3])

            model.domain = None                               # as a model file without the summary
            plain = api.predict_report(model, data)
            self.assertIsNone(plain.inside)
            self.assertEqual(plain.n_outside, int((~plain.in_range.all(axis=1)).sum()))     # only the range check is left
            api.export_predictions(plain, path)
            with open(path, encoding="utf-8-sig") as handle:
                self.assertIn("não avaliado", handle.read())

    def test_estimates_outside_the_calibrated_range_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = api.load_model(api.save_model(self.result, os.path.join(tmp, "m")))
        j = list(ANALYTES).index("gl")
        self.assertEqual((model.per_analyte["gl"].y_min, model.per_analyte["gl"].y_max),
                         (float(self.Y[:, j].min()), float(self.Y[:, j].max())))
        for source in (self.result, model):
            report = api.predict_report(source, self.dataset)
            self.assertTrue(report.in_range.all() and report.ok.all())
            self.assertEqual(report.notes(0), [])
            report.Y[2, j] = self.Y[:, j].max() + 0.5 * np.ptp(self.Y[:, j])      # as if the model had extrapolated
            flagged = api.prediction.Prediction(**{**report.__dict__, "in_range": api.prediction.predict_report(
                source, self.dataset).in_range.copy()})
            flagged.in_range[2, j] = False
            self.assertEqual(flagged.n_outside, 1)
            self.assertIn("gl fora da faixa calibrada", flagged.notes(2)[0])
        # a spectrum scaled far beyond the calibration gives estimates outside the range
        big = api.Dataset(self.X.mean(axis=0) + 8 * (self.X[:1] - self.X.mean(axis=0)), self.wl)
        report = api.predict_report(model, big)
        self.assertFalse(report.in_range.all())
        self.assertTrue(any("fora da faixa calibrada" in note for note in report.notes(0)))


class LegacyModel(PredictionCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "model_calibration.pkl")
        absor, lam = _quiet(apply_pretreatment, [list(s) for s in PRETREATMENT], self.X.copy(), self.wl.copy(), plot=False)
        self.X_pre, self.wl_pre = absor, lam
        _quiet(train_and_save_model_pls, absor, self.Y, lam, [3, 4, 2], self.path, rmsecv_list=[0.1, 0.2, 0.3])

    def test_old_model_file_predicts_as_the_script(self):
        model = api.load_legacy_pls(self.path, api.recipe_from_legacy(PRETREATMENT), analytes=ANALYTES, units="g/L")
        self.assertEqual([model.per_analyte[a].k for a in ANALYTES], [3, 4, 2])
        expected, rmsecv = _quiet(load_and_predict_pls, self.X_pre, self.wl_pre, self.path)
        report = api.predict_report(model, self.dataset)
        np.testing.assert_allclose(report.Y, expected, rtol=1e-10)
        np.testing.assert_allclose(report.typical_error, rmsecv)
        self.assertIsNone(report.inside)                      # old files have no summary of the calibration spectra
        self.assertEqual(list(api.load_legacy_pls(self.path, api.recipe_from_legacy(PRETREATMENT)).per_analyte),
                         ["analito_1", "analito_2", "analito_3"])

    def test_wrong_recipe_is_detected(self):
        model = api.load_legacy_pls(self.path, [])            # the recipe was forgotten
        with self.assertRaisesRegex(ValueError, "Confira se a receita"):
            api.predict(model, self.X, self.wl)
        with self.assertRaisesRegex(ValueError, "depende dos dados de calibração"):
            api.load_legacy_pls(self.path, [{"id": "msc", "params": {}}])
        with self.assertRaisesRegex(ValueError, "foram dados 2 nomes"):
            api.load_legacy_pls(self.path, [], analytes=["a", "b"])

    def test_files_with_code_are_refused_without_running_it(self):
        flag = os.path.join(self._tmp.name, "executou.txt")

        class Payload:
            def __reduce__(self):
                return (open, (flag, "w"))

        bad = os.path.join(self._tmp.name, "malicioso.pkl")
        with open(bad, "wb") as handle:
            pickle.dump({"type": "PLS", "wavelengths": self.wl_pre, "analytes": [Payload()]}, handle)
        with self.assertRaisesRegex(ValueError, "não é um modelo PLS antigo"):
            api.load_legacy_pls(bad, [])
        self.assertFalse(os.path.exists(flag))
        with open(bad, "wb") as handle:
            handle.write(b"isto nao e um pickle")
        with self.assertRaises(ValueError):
            api.load_legacy_pls(bad, [])


class TextSpectra(PredictionCase):
    def test_reads_spectra_in_either_axis_order(self):
        times = np.array([0.0, 3.0, 6.0])
        with tempfile.TemporaryDirectory() as tmp:
            for name, order in (("crescente.txt", slice(None)), ("decrescente.txt", slice(None, None, -1))):
                path = os.path.join(tmp, name)
                header = "Time\t" + "\t".join(f"{w:.10g}" for w in self.wl[order])
                np.savetxt(path, np.column_stack([times, self.X[:3][:, order]]), fmt="%.12g", delimiter="\t",
                           header=header, comments="")
                data = read_txt_spectra(path)
                np.testing.assert_allclose(data.wavelengths, self.wl, rtol=1e-9)
                np.testing.assert_allclose(data.X, self.X[:3], rtol=1e-10)
                np.testing.assert_array_equal(data.times, times)
                self.assertIsNone(data.Y)
                self.assertEqual(data.sample_ids[1], f"{name[:-4]}_3min")
            bad = os.path.join(tmp, "ruim.txt")
            with open(bad, "w") as handle:
                handle.write("Time\t1\t2\t3\n0\t1.0\t2.0\n")
            with self.assertRaises(ValueError):
                read_txt_spectra(bad)


if __name__ == "__main__":
    unittest.main()
