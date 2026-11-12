"""
Drives the interface from the first screen to the exported folder, without
showing a window (Qt "offscreen"). Skipped when PySide6 is not installed.
"""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

from ._spp_writer import write_spp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError:          # pragma: no cover
    QApplication = None

from multical import api  # noqa: E402

POINTS = 300
ANALYTES = ["gl", "xy"]


def _write_experiment(folder, name, seed, with_header=True):
    """A .spp time series whose bands follow two concentrations, and its table of measured values."""
    rng = np.random.RandomState(seed)
    seconds = 2.0 + 120.0 * np.arange(45)
    minutes = seconds / 60.0
    x = np.linspace(0, 1, POINTS)

    def conc(t):
        return np.column_stack([8 + 0.08 * t + seed, 1 + 5 * (1 - np.exp(-t / (25.0 + 5 * seed)))])

    def band(c, w):
        return np.exp(-((x - c) / w) ** 2)

    bands = np.vstack([band(0.35, 0.06), band(0.65, 0.09)])
    spectra = conc(minutes) @ bands + 0.4 * band(0.5, 0.4) + 0.002 * rng.normal(size=(45, POINTS))
    spp = os.path.join(folder, f"{name}.spp")
    write_spp(spp, spectra[:, ::-1], seconds, start=4000.0 + POINTS - 1, step=-1.0, allocated=60)
    ref_times = np.arange(11.0, 91.0, 10.0)      # about one minute away from the closest spectrum
    table = np.column_stack([ref_times, conc(ref_times) + 0.01 * rng.normal(size=(ref_times.size, 2))])
    ref = os.path.join(folder, f"{name}_refe.csv")
    with open(ref, "w", encoding="utf-8") as handle:
        if with_header:
            handle.write("tempo;gl;xy\n")
        for row in table:
            handle.write(";".join(f"{v:.4f}".replace(".", ",") for v in row) + "\n")
    return spp, ref


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class WizardEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.files = [_write_experiment(cls.tmp, f"exp{i}", i) for i in (1, 2, 3)]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        from multical_gui.wizard import MainWindow
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.close)
        self.project = self.window.project
        self.start, self.imp, self.inspect, self.pretreat, self.model, self.results, self.export = self.window.pages

    def _wait(self, condition, timeout=60):
        end = time.time() + timeout
        while not condition() and time.time() < end:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(condition(), "a interface não chegou ao estado esperado a tempo")

    def _import_all(self):
        self.assertTrue(self.window.go_to(1))
        for spp, ref in self.files:
            self.imp.add_experiment(spp, ref)

    def test_steps_are_locked_until_the_previous_one_is_done(self):
        self.start.create.click()
        self.assertEqual(self.window.current, 1)
        self.assertFalse(self.window.go_to(2))
        self.assertIn("Adicione pelo menos um experimento", self.window.message.text())
        self.assertFalse(self.window.next.isEnabled())
        self.assertFalse(self.window.steps.item(3).flags() & Qt.ItemIsEnabled)
        self._import_all()
        self.assertTrue(self.window.next.isEnabled())
        self.assertFalse(self.window.go_to(5))        # no model yet
        self.assertIn("Treinar modelo", self.window.message.text())

    def test_import_links_samples_by_time(self):
        self._import_all()
        data = self.project.dataset
        self.assertEqual((data.n_samples, data.analytes, data.units), (24, ANALYTES, "g/L"))
        self.assertEqual(self.imp.table.rowCount(), 3)
        self.assertEqual(self.imp.table.item(0, 4).text(), "8")
        self.assertIn("24 amostras", self.imp.summary.text())
        exact = data.X.copy()

        self.imp.radius.setValue(4.0)
        self.imp.window_mode.setChecked(True)
        self.assertEqual((self.project.match_mode, self.project.radius_min), ("window", 4.0))
        self.assertEqual(self.project.dataset.n_samples, 24)
        self.assertFalse(np.allclose(self.project.dataset.X, exact))

        self.imp.radius.setValue(0.3)                 # too narrow: no spectrum falls inside any window
        self.assertIsNone(self.project.dataset)
        self.assertEqual(self.imp.summary.level, "error")
        self.assertIn("nenhum espectro", self.imp.problems.toPlainText())
        self.imp.exact.setChecked(True)
        np.testing.assert_array_equal(self.project.dataset.X, exact)

        self.imp.analytes.setText("glicose, xilose")
        self.imp.analytes.editingFinished.emit()
        self.assertEqual(self.project.dataset.analytes, ["glicose", "xilose"])
        self.imp.units.setText("mg/mL")
        self.imp.units.editingFinished.emit()
        self.assertEqual(self.project.dataset.units, "mg/mL")

        broken = os.path.join(self.tmp, "ruim.csv")
        with open(broken, "w") as handle:
            handle.write("10;1,0\n20;2,0\n")
        self.imp.add_experiment(self.files[0][0], broken, "uma coluna")
        self.assertIn("colunas de valores medidos", self.project.experiments[-1].error)
        self.assertEqual(self.project.dataset.n_samples, 24)
        self.imp.table.selectRow(3)
        self.imp.remove.click()
        self.assertEqual(len(self.project.experiments), 3)

    def test_from_import_to_exported_model(self):
        self._import_all()

        self.assertTrue(self.window.go_to(2))
        self.assertEqual(self.inspect.samples.count(), 24)
        self.inspect.samples.item(5).setCheckState(Qt.Unchecked)
        self.assertEqual(self.project.excluded, {self.project.dataset.sample_ids[5]})
        self.assertIn("23 de 24", self.inspect.count.text())
        self.inspect.samples.setCurrentRow(2)

        self.assertTrue(self.window.go_to(3))
        self.assertEqual(self.pretreat.steps.count(), 1)
        self.assertIn("Savitzky-Golay", self.pretreat.steps.item(0).text())
        self.assertFalse(self.pretreat.box.isEnabled())            # locked outside the advanced mode
        self.window.advanced.setChecked(True)
        self.assertTrue(self.pretreat.box.isEnabled())
        self.pretreat.choices.setCurrentIndex(self.pretreat.choices.findData("cut"))
        self.pretreat.add.click()
        self.pretreat.form._editors["ranges"].setText("4000-4005")
        self.pretreat.form._editors["ranges"].editingFinished.emit()
        self.pretreat.up.click()                                   # cut first, then a 15-point filter on 6 points
        self.assertIn("maior que o espectro", self.pretreat.validate())
        self.assertFalse(self.window.go_to(4))
        self.pretreat.restore.click()
        self.assertIsNone(self.pretreat.validate())
        self.assertEqual(self.project.config.pretreatment, api.default_pretreatment())
        self.window.advanced.setChecked(False)

        self.assertTrue(self.window.go_to(4))
        self.assertEqual([c.text() for c in self.model.analyte_checks], ANALYTES)
        self.assertFalse(self.model.advanced_box.isVisible())
        self.assertIsNotNone(self.model.validate())
        self.model.train_button.click()
        self.assertFalse(self.window.steps.isEnabled())            # busy: no leaving the screen
        self._wait(lambda: self.project.result is not None and self.model.worker is None)
        self.assertEqual(self.window.current, 5)
        result = self.project.result
        self.assertEqual((result.analytes, len(result.sample_ids)), (ANALYTES, 23))
        self.assertGreater(result.per_analyte["gl"].best_r2cv, 0.9)

        self.assertEqual(self.results.analytes.count(), 2)
        self.assertIn("erro típico ±", self.results.analytes.item(0).text())
        self.assertFalse(self.results.tabs.isTabVisible(2))
        self.results.analytes.setCurrentRow(1)
        self.assertEqual(self.results.table.rowCount(), len(result.per_analyte["xy"].candidates))

        self.assertTrue(self.window.go_to(6))
        self.assertTrue(self.export.folder.text())
        folder = os.path.join(self.tmp, "execucao")
        self.export.folder.setText(folder)
        self.export.checks["report"].setChecked(False)
        written = self.export.export()
        self.assertEqual(self.export.banner.level, "ok")
        names = {os.path.basename(p) for p in written}
        self.assertTrue({"modelo.multical", "resumo.csv", "predicoes.csv", "config.json"} <= names)
        self.assertNotIn("relatorio.pdf", names)
        self.assertIsNone(self.export.export())                    # same folder again: refused
        self.assertEqual(self.export.banner.level, "error")

        model = api.load_model(os.path.join(folder, "modelo.multical"))
        data = self.project.active_dataset()
        np.testing.assert_allclose(api.predict(model, data.X, data.wavelengths),
                                   api.predict(result, data.X, data.wavelengths), rtol=1e-12)

        # changing the data afterwards discards the model and locks the later steps again
        self.window.go_to(2)
        self.inspect.include_all.click()
        self.assertIsNone(self.project.result)
        self.assertFalse(self.window.steps.item(5).flags() & Qt.ItemIsEnabled)

    def test_training_can_be_cancelled_and_errors_are_shown(self):
        self._import_all()
        self.window.go_to(4)
        self.model.train()
        self.model.cancel()
        self._wait(lambda: self.model.worker is None)
        self.assertIsNone(self.project.result)
        self.assertEqual(self.model.status.text(), "Treino cancelado.")
        self.assertTrue(self.model.train_button.isEnabled())

        self.window.advanced.setChecked(True)
        self.assertTrue(self.model.advanced_box.isVisible())
        self.model.model_form._editors["max_k"].setValue(40)       # more than 24 samples allow
        self.model.train()
        self._wait(lambda: self.model.worker is None)
        self.assertIsNone(self.project.result)
        self.assertEqual(self.model.banner.level, "error")
        self.assertIn("maior do que os dados permitem", self.model.banner.text())

        for check in self.model.analyte_checks[1:]:
            check.setChecked(False)
        self.model.validation.setCurrentIndex(self.model.validation.findData("loo"))
        self.window.go_to(3)
        self.window.go_to(4)                                       # coming back lowers the maximum to what fits
        self.assertLessEqual(self.project.config.resolved().model["params"]["max_k"], 22)
        self.model.train()
        self._wait(lambda: self.project.result is not None and self.model.worker is None)
        self.assertEqual(self.project.result.analytes, ["gl"])
        self.assertEqual(self.project.result.config.validation["id"], "loo")


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class UseAModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from multical_gui.state import Project
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.files = [_write_experiment(cls.tmp, f"exp{i}", i) for i in (1, 2, 3)]
        project = Project()
        for spp, ref in cls.files:
            project.add_experiment(spp, ref)
        cls.result = api.run_calibration(project.active_dataset(), project.config)
        cls.model_path = api.save_model(cls.result, os.path.join(cls.tmp, "modelo"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        from multical_gui.wizard import MainWindow
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.close)
        self.page = self.window.predict_page

    def _wait(self, condition, timeout=60):
        end = time.time() + timeout
        while not condition() and time.time() < end:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(condition(), "a interface não chegou ao estado esperado a tempo")

    def _run(self):
        self.page.run_button.click()
        self._wait(lambda: self.page.worker is None)

    def test_estimates_a_time_series_and_saves_the_table(self):
        self.assertEqual(self.window.steps.count(), 7)
        self.window.pages[0].use.click()
        self.assertEqual((self.window.flow, self.window.steps.count()), ("predict", 2))
        self.assertIs(self.window.stack.currentWidget(), self.page)
        self.assertFalse(self.page.run_button.isEnabled())
        self.assertFalse(self.window.next.isVisible())

        self.assertTrue(self.page.load_model(self.model_path))
        self.assertIn("gl, xy · g/L · criado com 24 amostras", self.page.model_info.text())
        self.assertIn("gl: erro típico ±", self.page.model_info.text())
        self.assertFalse(self.page.run_button.isEnabled())           # still no spectra
        self.page.add_spectra(self.files[0][0])
        self.page.add_spectra(self.files[0][0])                        # the same file twice counts once
        self.assertEqual(self.page.files.count(), 1)
        self.page.every.setValue(3)
        self._run()

        prediction = self.page.prediction
        self.assertEqual(len(prediction.sample_ids), 15)               # 45 spectra, one in every three
        np.testing.assert_allclose(prediction.times[:3], [2 / 60, 362 / 60, 722 / 60])
        self.assertEqual(self.page.table.rowCount(), 15)
        self.assertEqual(self.page.table.columnCount(), 5)
        self.assertEqual(self.page.table.horizontalHeaderItem(2).text(), "gl (g/L)")
        self.assertEqual(self.page.analyte.count(), 2)
        self.assertIn("espectros estimados", self.page.status.text())
        self.assertEqual(self.page.banner.level, "warning" if prediction.n_outside else "ok")
        middle = np.flatnonzero((prediction.times > 11) & (prediction.times < 81))
        self.assertTrue(prediction.ok[middle].all())                   # the calibrated part of the experiment

        from multical.io.spp import read_spp
        whole = read_spp(self.files[0][0])
        np.testing.assert_allclose(prediction.Y, api.predict(self.result, whole.X[::3], whole.wavelengths), rtol=1e-12)

        csv_path = self.page.save_table(os.path.join(self.tmp, "estimativas.csv"))
        with open(csv_path, encoding="utf-8-sig") as handle:
            self.assertEqual(len(handle.read().splitlines()), 16)
        self.assertEqual(self.page.banner.level, "ok")

        self.page.every.setValue(1)                                   # changing an input clears the old estimates
        self.assertIsNone(self.page.prediction)
        self.assertEqual(self.page.table.rowCount(), 0)
        self.assertFalse(self.page.save_button.isEnabled())

    def test_strange_spectra_are_flagged_and_bad_inputs_explained(self):
        self.window.start_flow("predict")
        self.assertFalse(self.page.load_model(os.path.join(self.tmp, "antigo.pkl")))
        self.assertIn("Modelos antigos", self.page.banner.text())
        not_a_model = os.path.join(self.tmp, "falso.multical")
        with open(not_a_model, "w") as handle:
            handle.write("nada")
        self.assertFalse(self.page.load_model(not_a_model))
        self.assertEqual(self.page.banner.level, "error")
        self.assertTrue(self.page.load_model(self.model_path))

        from multical.io.spp import read_spp
        whole = read_spp(self.files[1][0])
        strange = whole.X[10:16].copy()
        strange[:, 120:150] += 3.0                                    # a band the calibration never had
        text = os.path.join(self.tmp, "estranhos.txt")
        header = "Time\t" + "\t".join(f"{w:.10g}" for w in whole.wavelengths)
        np.savetxt(text, np.column_stack([np.arange(6) * 3.0, strange]), fmt="%.12g", delimiter="\t", header=header, comments="")
        self.page.add_spectra(text)
        self._run()
        self.assertEqual(self.page.prediction.n_outside, 6)
        self.assertEqual(self.page.banner.level, "warning")
        self.assertIn("espectro diferente", self.page.table.item(0, 4).text())

        other_grid = os.path.join(self.tmp, "outra_faixa.txt")
        np.savetxt(other_grid, np.column_stack([np.arange(6) * 3.0, strange[:, :200]]), fmt="%.12g", delimiter="\t",
                   header="Time\t" + "\t".join(f"{w:.10g}" for w in whole.wavelengths[:200]), comments="")
        self.page.files.setCurrentRow(0)
        self.page.remove_file.click()
        self.page.add_spectra(other_grid)
        self._run()
        self.assertIsNone(self.page.prediction)
        self.assertEqual(self.page.banner.level, "error")
        self.assertIn("mesma faixa", self.page.banner.text())

        self.page.add_spectra(self.files[2][0])
        self.page.run()
        self.page.cancel()
        self._wait(lambda: self.page.worker is None)
        self.assertIsNone(self.page.prediction)

        self.window.go_to(0)                                          # back to the first screen, other path
        self.window.pages[0].create.click()
        self.assertEqual((self.window.flow, self.window.steps.count(), self.window.current), ("create", 7, 1))


if __name__ == "__main__":
    unittest.main()
