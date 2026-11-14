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
if os.path.isdir(r"C:\Windows\Fonts"):      # without fonts the offscreen platform draws wide placeholder boxes
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

try:
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication
except ImportError:          # pragma: no cover
    QApplication = None

from multical import api  # noqa: E402

_SETTINGS = None


def setUpModule():
    """Keep the tests away from the user's own settings and from dialogs that wait for an answer."""
    global _SETTINGS
    if QApplication is None:
        return
    _SETTINGS = tempfile.TemporaryDirectory()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, _SETTINGS.name)
    from multical_gui.wizard import MainWindow
    MainWindow.ask_about_unsaved = lambda self: "discard"


def tearDownModule():
    if _SETTINGS is not None:
        _SETTINGS.cleanup()

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

        self.assertEqual(self.imp.analyte_table.rowCount(), 2)
        self.assertEqual([self.imp.analyte_table.item(r, 1).text() for r in range(2)], ANALYTES)   # from the file header
        self.imp.analyte_table.item(0, 1).setText("glicose")
        self.imp.analyte_table.item(1, 1).setText("xilose")
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

    def test_side_bar_cannot_change_the_screen_while_one_is_being_shown(self):
        # Hiding the first screen can hand the keyboard focus to the side bar, whose list then picks a
        # current row by itself; that used to send the program back to the first screen without showing it.
        start = self.window.pages[0]
        self.window.steps.blockSignals(True)
        self.window.steps.setCurrentRow(-1)
        self.window.steps.blockSignals(False)
        emitted = []
        original = self.window.stack.setCurrentWidget

        def swap(widget):                                             # what Qt does when the list gains focus
            self.window.steps.setCurrentRow(0)
            emitted.append(self.window.steps.signalsBlocked())
            original(widget)

        self.window.stack.setCurrentWidget = swap
        start.use.click()
        self.window.stack.setCurrentWidget = original
        self.assertEqual(emitted, [True])
        self.assertEqual((self.window.flow, self.window.current, self.window.steps.currentRow()), ("predict", 1, 1))
        self.assertIs(self.window.stack.currentWidget(), self.page)
        self.assertTrue(self.window.back.isVisible())

    def test_window_fits_a_small_laptop_screen(self):
        for flow in ("create", "predict"):
            self.window.set_flow(flow)
            hint = self.window.minimumSizeHint()
            self.assertLessEqual(hint.width(), 1150, flow)
            self.assertLessEqual(hint.height(), 700, flow)


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class MethodsFromTheRegistries(unittest.TestCase):
    """The screens were not written for these methods: they only read the registries."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.files = [_write_experiment(cls._tmp.name, f"exp{i}", i) for i in (1, 2, 3)]
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        api.PRETREATMENTS.unregister("log10")
        api.load_plugins([os.path.join(repo, "plugins")], reload=True)

    @classmethod
    def tearDownClass(cls):
        api.PRETREATMENTS.unregister("log10")
        cls._tmp.cleanup()

    def setUp(self):
        from multical_gui.wizard import MainWindow
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.close)
        self.project = self.window.project
        _, self.imp, _, self.pretreat, self.model, self.results, _ = self.window.pages
        self.window.go_to(1)
        for spp, ref in self.files:
            self.imp.add_experiment(spp, ref)
        self.window.advanced.setChecked(True)

    def _train(self):
        self.model.train_button.click()
        end = time.time() + 120
        while self.model.worker is not None and time.time() < end:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertIsNone(self.model.worker)
        self.assertEqual(self.model.banner.text(), "")
        self.assertIsNotNone(self.project.result)

    def test_plugin_pretreatment_appears_and_works(self):
        self.window.go_to(3)
        position = self.pretreat.choices.findData("log10")
        self.assertGreaterEqual(position, 0)
        self.assertIn("exemplo de plugin", self.pretreat.choices.itemText(position))
        self.pretreat.choices.setCurrentIndex(position)
        self.pretreat.add.click()
        self.assertIsNone(self.pretreat.validate())
        self.assertEqual([s["id"] for s in self.project.config.pretreatment], ["sg", "log10"])
        self.pretreat.form._editors["offset"].setValue(2.5)
        self.assertEqual(self.project.config.pretreatment[1]["params"], {"offset": 2.5})
        self.window.go_to(4)
        self._train()
        self.assertEqual(self.project.result.config.pretreatment[1], {"id": "log10", "params": {"offset": 2.5}})

    def test_every_model_and_validation_scheme_trains_from_the_screen(self):
        self.window.go_to(4)
        self.assertEqual([self.model.model.itemData(i) for i in range(self.model.model.count())], api.MODELS.ids())
        words = {"pls": "variáve", "pcr": "componente", "spa": "ponto", "svr": "ajuste escolhido"}
        for model_id in api.MODELS.ids():
            with self.subTest(model=model_id):
                self.model.model.setCurrentIndex(self.model.model.findData(model_id))
                self.assertEqual(self.project.config.model["id"], model_id)
                self._train()
                self.assertEqual(self.window.current, 5)
                self.assertEqual(self.project.result.config.model["id"], model_id)
                self.assertEqual(self.results.analytes.count(), 2)
                self.assertIn(words[model_id], self.results.analytes.item(0).text())
                self.results.analytes.setCurrentRow(1)
                first = next(iter(self.project.result.per_analyte.values()))
                self.assertEqual(self.results.table.rowCount(), len(first.candidates))
                self.window.go_to(4)
        self.model.model.setCurrentIndex(self.model.model.findData("pls"))
        for scheme in api.SPLITTERS.ids():
            with self.subTest(validation=scheme):
                self.model.validation.setCurrentIndex(self.model.validation.findData(scheme))
                self._train()
                self.assertEqual(self.project.result.config.validation["id"], scheme)
                self.window.go_to(4)

    def test_variable_selection_from_the_screen_to_the_model_file(self):
        from multical_gui.pages.model_page import duration
        from multical_gui.pages.predict_page import describe_model
        from multical_gui.state import Project
        self.window.go_to(4)
        combo = self.model.variables
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], [None] + api.SELECTORS.ids())
        self.assertIsNone(self.project.config.variable_selection)
        self.assertFalse(self.model.variables_hint.isVisible())
        self.assertNotIn("Seleção de variáveis", self.model.summary.text())

        for selector in api.SELECTORS.ids():                       # every method shows its own settings
            combo.setCurrentIndex(combo.findData(selector))
            self.assertEqual(self.project.config.variable_selection, {"id": selector, "params": {}})
            self.assertEqual(set(self.model.variables_form._editors), {p.name for p in api.SELECTORS.get(selector).params})
        combo.setCurrentIndex(combo.findData("pso"))
        self.assertTrue(self.model.variables_hint.isVisible())
        for _ in range(3):
            self.app.processEvents()
        form, area = self.model.variables_form, self.model.advanced_scroll      # many settings: scrolled, never squeezed
        self.assertGreaterEqual(form.height(), form.sizeHint().height())
        self.assertGreaterEqual(self.model.advanced_box.height(), self.model.advanced_box.sizeHint().height())
        if self.model.advanced_box.height() > area.viewport().height():
            self.assertTrue(area.verticalScrollBar().isVisible())
            self.assertGreater(area.verticalScrollBar().maximum(), 0)
        self.model.variables_form._editors["n_particles"].setValue(6)
        self.model.variables_form._editors["max_iter"].setValue(3)
        chosen = self.project.config.variable_selection
        self.assertEqual((chosen["id"], chosen["params"]["n_particles"], chosen["params"]["max_iter"]), ("pso", 6, 3))
        self.assertIn("Seleção de variáveis: PSO", self.model.summary.text())
        self.assertTrue(self.project.dirty)

        self._train()
        self.assertEqual(self.window.current, 5)
        result = self.project.result
        points = result.per_analyte["gl"].n_selected
        self.assertTrue(2 <= points < result.wavelengths.size)
        self.assertIn(f"{points} de {result.wavelengths.size} pontos do espectro", self.results.analytes.item(0).text())
        self.assertTrue(self.results.tabs.isTabVisible(3))
        self.assertIn("tende a parecer menor", self.results.warnings.text())
        self.assertIn(f"gl: {points} de", self.results.points.figure.axes[0].get_title())
        self.results.analytes.setCurrentRow(1)
        self.assertIn("xy:", self.results.points.figure.axes[0].get_title())

        with tempfile.TemporaryDirectory() as tmp:
            self.window.go_to(6)
            page = self.window.pages[6]
            page.folder.setText(os.path.join(tmp, "execucao"))
            names = {os.path.basename(p) for p in page.export()}
            self.assertTrue({"pontos_escolhidos.csv", "pontos_escolhidos_gl.png", "modelo.multical"} <= names)
            model = api.load_model(os.path.join(tmp, "execucao", "modelo.multical"))
            self.assertIn(f"usa {points} de {result.wavelengths.size} pontos do espectro", describe_model(model))
            data = self.project.active_dataset()
            np.testing.assert_allclose(api.predict(model, data.X, data.wavelengths),
                                       api.predict(result, data.X, data.wavelengths), rtol=1e-12)

            path = self.project.save(os.path.join(tmp, "com selecao"))
            reopened = Project()
            reopened.open(path)
            self.assertEqual(reopened.config.variable_selection, self.project.config.variable_selection)

        self.window.go_to(4)                                        # the choice is still on the screen, and can be undone
        self.assertEqual(combo.currentData(), "pso")
        self.assertEqual(self.model.variables_form._editors["max_iter"].value(), 3)
        combo.setCurrentIndex(0)
        self.assertIsNone(self.project.config.variable_selection)
        self.assertIsNone(self.model.variables_form)
        self.assertIsNone(self.project.result)
        self._train()
        self.assertFalse(self.results.tabs.isTabVisible(3))
        self.assertNotIn("pontos do espectro", self.results.analytes.item(0).text())

        self.assertEqual([duration(s) for s in (2, 23, 49, 50, 100, 400)], ["5 s", "25 s", "50 s", "1 min", "2 min", "7 min"])


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class InstalledProgram(unittest.TestCase):
    """What the packaged program relies on: its self-check and the guide next to it."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_self_check_passes_and_writes_its_report(self):
        from multical_gui.app import main
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "relatorio.txt")
            self.assertEqual(main(["Multical", "--autoteste", report]), 0)
            with open(report, encoding="utf-8") as handle:
                text = handle.read()
        self.assertEqual(text.count("OK  "), 7, text)
        self.assertIn("Seleção de variáveis: vip (", text)
        self.assertIn("Pré-tratamentos: cut, sg", text)
        self.assertIn("Tudo certo.", text)
        self.assertIn("modelos pls, pcr, spa, svr", text)

    def test_self_check_reports_a_failure(self):
        from multical_gui import selftest
        original = selftest.CHECKS
        selftest.CHECKS = (("Verificação quebrada", lambda: 1 / 0),)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = os.path.join(tmp, "relatorio.txt")
                self.assertEqual(selftest.run(report), 1)
                with open(report, encoding="utf-8") as handle:
                    text = handle.read()
        finally:
            selftest.CHECKS = original
        self.assertIn("FALHA  Verificação quebrada: ZeroDivisionError", text)
        self.assertIn("1 verificação(ões) falharam", text)

    def test_guide_is_reachable_from_the_first_screen(self):
        from multical_gui.app import guide_path
        from multical_gui.wizard import MainWindow
        path = guide_path()
        self.assertTrue(path and path.endswith(os.path.join("docs", "guia_do_usuario.html")))
        with open(path, encoding="utf-8") as handle:
            html = handle.read()
        folder = os.path.dirname(path)
        import re
        images = re.findall(r'src="([^"]+)"', html)
        self.assertGreaterEqual(len(images), 3)
        for image in images:
            self.assertTrue(os.path.isfile(os.path.join(folder, image)), image)
        window = MainWindow()
        window.show()
        self.addCleanup(window.close)
        self.assertTrue(window.pages[0].guide.isVisible())


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class TextSpectraAnalytesAndProjects(unittest.TestCase):
    """Spectra in text files, telling the analytes apart, and saving/opening projects."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.files = [_write_experiment(cls.tmp, f"exp{i}", i) for i in (1, 2, 3)]
        from multical.io.spp import read_spp
        cls.series = read_spp(cls.files[0][0])
        cls.table = np.loadtxt(cls.files[0][1], delimiter=";", skiprows=1,
                               converters=lambda v: float(v.replace(",", ".")), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        from multical_gui.wizard import MainWindow
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.close)
        self.project = self.window.project
        self.imp = self.window.pages[1]
        self.window.go_to(1)

    def _text_spectra(self, name, rows, minutes):
        """Write spectra as the '_nonda.txt' files: header with the axis, then time (min) and one spectrum per row."""
        path = os.path.join(self.tmp, name)
        header = "Time\t" + "\t".join(f"{w:.10g}" for w in self.series.wavelengths)
        np.savetxt(path, np.column_stack([minutes, self.series.X[rows]]), fmt="%.12g", delimiter="\t", header=header, comments="")
        return path

    def _table(self, name, rows, header="tempo;gl;xy"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            if header:
                handle.write(header + "\n")
            for row in rows:
                handle.write(";".join(f"{v:.4f}".replace(".", ",") for v in row) + "\n")
        return path

    # ---- spectra in text files ---------------------------------------------
    def test_text_file_with_one_spectrum_per_sample_is_linked_by_row(self):
        times = self.table[:, 0]
        rows = [int(np.argmin(np.abs(self.series.times / 60.0 - t))) for t in times]
        spectra = self._text_spectra("exp1_nonda.txt", rows, times)
        self.imp.add_experiment(spectra, self.files[0][1])
        exp = self.project.experiments[0]
        self.assertEqual((exp.kind, exp.linked_by, exp.n_spectra, exp.n_linked, exp.error), ("txt", "order", 8, 8, ""))
        np.testing.assert_allclose(self.project.dataset.X, self.series.X[rows], rtol=1e-10)
        np.testing.assert_allclose(self.project.dataset.Y, self.table[:, 1:], rtol=1e-12)
        self.assertEqual(self.project.dataset.sample_ids[0], "exp1_nonda_11min")
        self.assertIn("ligado pela ordem das linhas", self.imp.table.item(0, 5).text())

        shifted = self._text_spectra("exp1_outros_tempos.txt", rows, times + 5.0)
        self.imp.add_experiment(shifted, self.files[0][1], "tempos diferentes")
        self.assertEqual(self.project.experiments[1].linked_by, "order")
        self.assertIn("não coincidem", self.project.experiments[1].note)
        self.assertIn("atenção", self.imp.table.item(1, 5).text())

    def test_text_file_with_a_whole_series_is_linked_by_time_like_spp(self):
        minutes = self.series.times / 60.0
        spectra = self._text_spectra("exp1_serie.txt", np.arange(self.series.n_samples), minutes)
        self.imp.add_experiment(spectra, self.files[0][1], "texto")
        self.imp.add_experiment(self.files[0][0], self.files[0][1], "spp")
        text, spp = self.project.experiments
        self.assertEqual((text.kind, text.linked_by, text.n_spectra, text.n_linked), ("txt", "time", 45, 8))
        self.assertEqual((spp.kind, spp.linked_by, spp.n_linked), ("spp", "time", 8))
        data = self.project.dataset
        np.testing.assert_allclose(data.X[:8], data.X[8:], rtol=1e-9)          # same spectra chosen from both files
        self.assertIn("ligado pelo tempo", self.imp.table.item(0, 5).text())

        self.imp.radius.setValue(4.0)
        self.imp.window_mode.setChecked(True)                                   # the window applies to the text series too
        np.testing.assert_allclose(self.project.dataset.X[:8], self.project.dataset.X[8:], rtol=1e-9)
        self.assertFalse(np.allclose(self.project.dataset.X[:8], data.X[:8]))

        with open(os.path.join(self.tmp, "nao_e_espectro.txt"), "w") as handle:
            handle.write("isto não é um arquivo de espectros\n1 2 3\n")
        self.imp.add_experiment(os.path.join(self.tmp, "nao_e_espectro.txt"), self.files[0][1], "ruim")
        self.assertIn("não está no formato esperado", self.project.experiments[-1].error)
        self.assertEqual(self.project.dataset.n_samples, 16)

    # ---- analytes ----------------------------------------------------------
    def test_analytes_follow_the_table_and_can_be_named(self):
        three = self._table("tres.csv", np.column_stack([self.table, 100 + self.table[:, 1]]), header="")
        self.imp.add_experiment(self.files[0][0], three)
        table = self.imp.analyte_table
        self.assertEqual(table.rowCount(), 3)                                   # as many analytes as the file has
        self.assertEqual([table.item(r, 1).text() for r in range(3)], ["analito_1", "analito_2", "analito_3"])
        self.assertEqual([table.item(r, 0).text() for r in range(3)], ["2ª", "3ª", "4ª"])
        low, high = self.table[:, 1].min(), self.table[:, 1].max()
        self.assertEqual(table.item(0, 2).text(), f"{low:.4g}".replace(".", ",") + " g/L")
        self.assertEqual(table.item(2, 3).text(), f"{100 + high:.4g}".replace(".", ",") + " g/L")
        self.assertFalse(table.item(0, 2).flags() & Qt.ItemIsEditable)

        table.item(0, 1).setText("glicose")
        table.item(2, 1).setText("etanol")
        self.assertEqual(self.project.dataset.analytes, ["glicose", "analito_2", "etanol"])
        table.item(1, 1).setText("Glicose")                                     # repeated name: refused and shown again
        self.assertEqual(self.project.dataset.analytes, ["glicose", "analito_2", "etanol"])
        self.assertEqual(table.item(1, 1).text(), "analito_2")
        self.assertIn("não podem se repetir", self.window.message.text())
        table.item(1, 1).setText("  ")
        self.assertEqual(self.project.dataset.analytes[1], "analito_2")

        self.project.config.analytes = ["glicose", "etanol"]
        table.item(2, 1).setText("EtOH")                                        # the selection follows the new name
        self.assertEqual(self.project.config.analytes, ["glicose", "EtOH"])
        self.window.go_to(2)
        self.window.go_to(3)
        self.window.go_to(4)
        model = self.window.pages[4]
        self.assertEqual([c.text() for c in model.analyte_checks], ["glicose", "analito_2", "EtOH"])
        self.assertEqual([c.isChecked() for c in model.analyte_checks], [True, False, True])

    def test_tables_with_columns_in_another_order_are_refused(self):
        self.imp.add_experiment(self.files[0][0], self.files[0][1])
        swapped = self._table("trocada.csv", self.table[:, [0, 2, 1]], header="tempo;xy;gl")
        self.imp.add_experiment(self.files[1][0], swapped, "trocado")
        self.assertIn("não são as mesmas, na mesma ordem", self.project.experiments[1].error)
        self.assertEqual(self.project.dataset.n_samples, 8)
        unnamed = self._table("sem_nomes.csv", self.table, header="")
        self.imp.add_experiment(self.files[1][0], unnamed, "sem nomes")           # nothing to compare: accepted
        self.assertEqual(self.project.experiments[2].error, "")

    # ---- projects ----------------------------------------------------------
    def _fill_project(self):
        for spp, ref in self.files:
            self.imp.add_experiment(spp, ref)
        self.imp.radius.setValue(4.0)
        self.imp.window_mode.setChecked(True)
        self.imp.analyte_table.item(0, 1).setText("glicose")
        self.imp.units.setText("mg/mL")
        self.imp.units.editingFinished.emit()
        self.project.set_excluded([self.project.dataset.sample_ids[3], self.project.dataset.sample_ids[10]])
        self.project.set_recipe([{"id": "snv", "params": {}}, {"id": "sg", "params": {"radius": 5, "poly_order": 2, "deriv": 2}}])
        self.project.config.model = {"id": "pcr", "params": {"max_k": 4}}
        self.project.config.validation = {"id": "loexo", "params": {}}
        self.project.config.analytes = ["glicose"]
        self.window.advanced.setChecked(True)

    def test_project_is_saved_and_opened_with_everything_the_user_chose(self):
        self.assertEqual(self.window.windowTitle(), "Multical")
        self._fill_project()
        self.window.refresh_nav()
        self.assertEqual(self.window.windowTitle(), "Multical — projeto sem nome*")
        before = self.project.to_dict()
        X, ids = self.project.dataset.X.copy(), list(self.project.dataset.sample_ids)

        path = os.path.join(self.tmp, "meu projeto")
        self.assertTrue(self.window.save_project_as(path))
        self.assertTrue(os.path.isfile(path + ".mcproj"))
        self.assertFalse(self.project.dirty)
        self.assertEqual(self.window.windowTitle(), "Multical — meu projeto.mcproj")
        self.project.set_excluded([])
        self.window.refresh_nav()
        self.assertTrue(self.window.windowTitle().endswith("*"))
        self.assertTrue(self.window.save_project())
        self.project.set_excluded([ids[3], ids[10]])
        self.assertTrue(self.window.save_project())

        self.assertTrue(self.window.new_project())
        self.assertTrue(self.project.is_empty)
        self.assertEqual((self.window.current, self.window.windowTitle()), (0, "Multical"))
        self.assertFalse(self.window.advanced.isChecked())

        self.assertTrue(self.window.open_project(path + ".mcproj"))
        self.assertEqual(self.project.to_dict(), before)
        self.assertEqual((self.window.flow, self.window.current), ("create", 1))
        self.assertTrue(self.window.advanced.isChecked())
        np.testing.assert_array_equal(self.project.dataset.X, X)
        self.assertEqual(self.project.dataset.analytes, ["glicose", "xy"])
        self.assertEqual(self.project.dataset.units, "mg/mL")
        self.assertEqual(self.project.excluded, {ids[3], ids[10]})
        self.assertEqual(self.imp.table.rowCount(), 3)
        self.assertTrue(self.imp.window_mode.isChecked())
        self.assertIsNone(self.project.result)                                  # the model is not part of the project
        self.assertFalse(self.project.dirty)

        # the choices reach the other screens
        self.window.go_to(2)
        self.assertIn("22 de 24", self.window.pages[2].count.text())
        self.window.go_to(3)
        self.assertEqual(self.window.pages[3].steps.count(), 2)
        self.window.go_to(4)
        model = self.window.pages[4]
        self.assertEqual((model.model.currentData(), model.validation.currentData()), ("pcr", "loexo"))
        self.assertEqual([c.isChecked() for c in model.analyte_checks], [True, False])

        from multical_gui.recent import recent_projects
        self.assertEqual(recent_projects()[0], os.path.abspath(path + ".mcproj"))
        self.window.go_to(0)
        start = self.window.pages[0]
        self.assertTrue(start.recent.isVisible())
        self.assertIn("meu projeto.mcproj", start.recent.item(0).text())

    def test_project_follows_its_folder_and_reports_missing_files(self):
        import shutil
        folder = os.path.join(self.tmp, "pasta do projeto")
        os.makedirs(os.path.join(folder, "dados"))
        copies = []
        for spp, ref in self.files[:2]:
            copies.append(tuple(shutil.copy(p, os.path.join(folder, "dados")) for p in (spp, ref)))
        for spp, ref in copies:
            self.imp.add_experiment(spp, ref)
        self.assertTrue(self.window.save_project_as(os.path.join(folder, "p.mcproj")))

        moved = os.path.join(self.tmp, "pasta levada para outro lugar")
        shutil.move(folder, moved)                                              # project and data moved together
        self.assertTrue(self.window.open_project(os.path.join(moved, "p.mcproj")))
        self.assertEqual([e.error for e in self.project.experiments], ["", ""])
        self.assertTrue(self.project.experiments[0].spectra_path.startswith(moved))
        self.assertEqual(self.project.dataset.n_samples, 16)

        os.remove(self.project.experiments[1].spectra_path)                     # one data file is gone
        self.assertTrue(self.window.open_project(os.path.join(moved, "p.mcproj")))
        self.assertIn("não encontrado", self.project.experiments[1].error)
        self.assertEqual(self.project.dataset.n_samples, 8)
        self.assertIn("não puderam ser lidos", self.window.message.text())

    def test_unsaved_changes_are_asked_about_and_bad_files_are_refused(self):
        from PySide6.QtWidgets import QMessageBox
        answers = []
        self.window.ask_about_unsaved = lambda: answers.append("asked") or self.choice
        self.choice = "cancel"
        self.assertTrue(self.window.new_project())                               # empty project: nothing to ask
        self.assertEqual(answers, [])
        self.imp.add_experiment(*self.files[0])
        self.assertFalse(self.window.new_project())                              # asked, and the user stays
        self.assertEqual((answers, len(self.project.experiments)), (["asked"], 1))
        self.assertFalse(self.window.close())
        self.assertTrue(self.window.isVisible())

        saved = []
        self.window.save_project = lambda: saved.append(True) or True
        self.choice = "save"
        self.assertTrue(self.window.new_project())
        self.assertEqual(saved, [True])
        del self.window.save_project

        self.choice = "discard"
        self.imp.add_experiment(*self.files[0])
        warnings = []
        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(lambda *args: warnings.append(args[2]))
        try:
            bad = os.path.join(self.tmp, "nao_e_projeto.mcproj")
            with open(bad, "w") as handle:
                handle.write("{ isto não é json")
            self.assertFalse(self.window.open_project(bad))
            newer = os.path.join(self.tmp, "futuro.mcproj")
            with open(newer, "w") as handle:
                handle.write('{"format": 99, "experiments": []}')
            self.assertFalse(self.window.open_project(newer))
            missing_plugin = os.path.join(self.tmp, "plugin.mcproj")
            with open(missing_plugin, "w") as handle:
                handle.write('{"format": 1, "experiments": [], "config": {"model": {"id": "modelo_de_plugin", "params": {}}}}')
            self.assertFalse(self.window.open_project(missing_plugin))
        finally:
            QMessageBox.warning = original
        self.assertEqual(len(warnings), 3)
        self.assertIn("versão mais nova", warnings[1])
        self.assertIn("modelo_de_plugin", warnings[2])
        self.assertEqual(len(self.project.experiments), 1)                        # a refused file leaves the current project as it was
        self.assertIsNotNone(self.project.dataset)

    def test_plugins_that_failed_are_listed_on_the_first_screen(self):
        start = self.window.pages[0]
        self.window.go_to(0)
        self.assertFalse(start.plugin_errors.isVisible())
        api.PLUGIN_ERRORS.append((os.path.join(self.tmp, "meu_plugin.py"), "NameError: name 'np' is not defined\nTraceback..."))
        try:
            start.enter()
            self.assertTrue(start.plugin_errors.isVisible())
            self.assertIn("meu_plugin.py: NameError: name 'np' is not defined", start.plugin_errors.text())
            self.assertNotIn("Traceback", start.plugin_errors.text())
        finally:
            del api.PLUGIN_ERRORS[:]


@unittest.skipIf(QApplication is None, "PySide6 não está instalado")
class SeveralFilesAtOnce(unittest.TestCase):
    """Spectra files and tables of measured values chosen together are sorted into experiments."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        cls.files = {name: _write_experiment(cls.tmp, name, seed) for seed, name in enumerate(("exp1", "exp2", "exp10"), start=1)}
        from multical.io.spp import read_spp
        series = read_spp(cls.files["exp1"][0])
        cls.text = os.path.join(cls.tmp, "lote7_nonda.txt")             # spectra in text, no table with a similar name
        header = "Time\t" + "\t".join(f"{w:.10g}" for w in series.wavelengths)
        np.savetxt(cls.text, np.column_stack([np.arange(8) * 10.0 + 11.0, series.X[:8]]), fmt="%.12g", delimiter="\t",
                   header=header, comments="")
        cls.loose_table = os.path.join(cls.tmp, "medidas_do_laboratorio.csv")
        with open(cls.files["exp2"][1], encoding="utf-8") as source, open(cls.loose_table, "w", encoding="utf-8") as target:
            target.write(source.read())
        cls.notes = os.path.join(cls.tmp, "anotacoes.txt")
        with open(cls.notes, "w", encoding="utf-8") as handle:
            handle.write("Experimento 1 feito na terca\nexp2 repetido\n")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        from multical_gui.wizard import MainWindow
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.window.close)
        self.project = self.window.project
        self.imp = self.window.pages[1]
        self.window.go_to(1)

    def _all(self):
        return [path for pair in self.files.values() for path in pair]

    def test_files_are_told_apart_and_paired_by_name(self):
        from multical_gui.file_pairs import file_kind, sort_files
        self.assertEqual([file_kind(p) for p in (self.files["exp1"][0], self.files["exp1"][1], self.text, self.notes)],
                         ["spectra", "table", "spectra", ""])
        self.assertEqual(file_kind(os.path.join(self.tmp, "nao_existe.txt")), "")

        shuffled = list(reversed(self._all())) + [self.notes, self.text, self.loose_table]
        pairs, spare, ignored = sort_files(shuffled)
        names = [(os.path.basename(s), t and os.path.basename(t)) for s, t in pairs]
        self.assertEqual(names, [("exp1.spp", "exp1_refe.csv"), ("exp2.spp", "exp2_refe.csv"),      # exp2 before exp10,
                                 ("exp10.spp", "exp10_refe.csv"), ("lote7_nonda.txt", None)])      # and 1 is not 10
        self.assertEqual(([os.path.basename(p) for p in spare], [os.path.basename(p) for p in ignored]),
                         (["medidas_do_laboratorio.csv"], ["anotacoes.txt"]))

        # Two spectra files that fit the same table equally well: none is guessed.
        copies = {}
        for name, source in (("lote3_nonda.spp", 0), ("lote3_inf.spp", 0), ("lote3_refe.csv", 1), ("exp2_inf.spp", 0)):
            copies[name] = os.path.join(self.tmp, name)
            with open(self.files["exp2"][source], "rb") as origin, open(copies[name], "wb") as target:
                target.write(origin.read())
            self.addCleanup(os.remove, copies[name])
        pairs, spare, _ = sort_files([copies["lote3_nonda.spp"], copies["lote3_inf.spp"], copies["lote3_refe.csv"]])
        self.assertEqual([t for _, t in pairs], [None, None])
        self.assertEqual(spare, [copies["lote3_refe.csv"]])
        # The name that differs least from the table wins: exp2 goes with exp2_refe, exp2_inf stays without one.
        pairs, spare, _ = sort_files([copies["exp2_inf.spp"], *self.files["exp2"]])
        self.assertEqual([(os.path.basename(s), t and os.path.basename(t)) for s, t in pairs],
                         [("exp2.spp", "exp2_refe.csv"), ("exp2_inf.spp", None)])
        self.assertEqual(spare, [])
        # One spectra file and one table: taken as a pair whatever the names.
        self.assertEqual(sort_files([self.loose_table, self.text]), ([(self.text, self.loose_table)], [], []))

    def test_pairs_are_shown_confirmed_and_added_in_one_go(self):
        from multical_gui.state import Project
        seen, rebuilds = {}, []
        original = Project.rebuild

        def counting(project):
            rebuilds.append(1)
            return original(project)

        def confirm(dialog):
            seen["rows"] = [(dialog.table.item(r, 0).text(), dialog.choices[r].currentText(), dialog.names[r].text())
                            for r in range(dialog.table.rowCount())]
            seen["button"] = dialog.buttons.button(dialog.buttons.StandardButton.Ok).text()
            seen["status"], seen["notes"] = dialog.status.text(), dialog.notes.text()
            if len(dialog.names) > 1:
                dialog.names[1].setText("segundo lote")
            return True

        Project.rebuild = counting
        self.addCleanup(setattr, Project, "rebuild", original)
        added = self.imp.add_files(self._all() + [self.notes], confirm=confirm)
        self.assertEqual(added, 3)
        self.assertEqual(len(rebuilds), 1)                                       # the files are read once, not once per experiment
        self.assertEqual(seen["rows"], [("exp1.spp", "exp1_refe.csv", "exp1"), ("exp2.spp", "exp2_refe.csv", "exp2"),
                                        ("exp10.spp", "exp10_refe.csv", "exp10")])
        self.assertEqual(seen["button"], "Adicionar 3 experimentos")
        self.assertEqual(seen["status"], "")
        self.assertIn("anotacoes.txt", seen["notes"])
        self.assertEqual([e.name for e in self.project.experiments], ["exp1", "segundo lote", "exp10"])
        self.assertEqual([(e.error, e.n_linked) for e in self.project.experiments], [("", 8)] * 3)
        self.assertEqual(self.imp.table.rowCount(), 3)
        self.assertEqual(self.project.dataset.n_samples, 24)
        self.assertIn("3 experimento(s) adicionado(s)", self.window.message.text())
        self.assertIsNone(self.imp.validate())

        # The same files again: nothing to add; with a new one among them, only the new one is offered.
        self.assertEqual(self.imp.add_files(self._all(), confirm=lambda dialog: self.fail("nothing to confirm")), 0)
        self.assertIn("já estavam no projeto", self.window.message.text())
        self.assertEqual(self.imp.add_files(self._all() + [self.text, self.loose_table], confirm=confirm), 0)   # no table chosen
        self.assertEqual(seen["rows"], [("lote7_nonda.txt", "— não adicionar —", "lote7_nonda")])
        self.assertIn("exp1.spp", seen["notes"])
        self.assertEqual(len(self.project.experiments), 3)

    def test_a_pair_can_be_corrected_and_mistakes_are_refused(self):
        def choose(dialog, row, name):
            dialog.choices[row].setCurrentIndex(dialog.choices[row].findText(name))

        def correct(dialog):
            ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
            self.assertEqual(dialog.choices[1].currentText(), "— não adicionar —")      # the text spectra had no match
            self.assertIn("Sem tabela (não serão adicionados): lote7_nonda.txt", dialog.status.text())
            self.assertIn("Tabelas sem uso: medidas_do_laboratorio.csv", dialog.status.text())
            self.assertEqual((ok.text(), ok.isEnabled()), ("Adicionar 1 experimento", True))
            choose(dialog, 1, "exp1_refe.csv")                                          # already used by exp1.spp
            self.assertIn("mesma tabela foi escolhida", dialog.status.text())
            self.assertEqual(dialog.status.level, "error")
            self.assertFalse(ok.isEnabled())
            choose(dialog, 1, "medidas_do_laboratorio.csv")
            self.assertEqual((dialog.status.text(), ok.text(), ok.isEnabled()), ("", "Adicionar 2 experimentos", True))
            return True

        files = [*self.files["exp1"], self.text, self.loose_table]
        self.assertEqual(self.imp.add_files(files, confirm=correct), 2)
        self.assertEqual([(e.name, os.path.basename(e.reference_path), e.kind) for e in self.project.experiments],
                         [("exp1", "exp1_refe.csv", "spp"), ("lote7_nonda", "medidas_do_laboratorio.csv", "txt")])

        def spoil(dialog):
            choose(dialog, 0, "exp10_refe.csv")
            choose(dialog, 1, "exp10_refe.csv")
            return True                                                                 # as if "add" could be pressed anyway

        before = len(self.project.experiments)
        self.assertEqual(self.imp.add_files([*self.files["exp2"], *self.files["exp10"]], confirm=spoil), 0)
        self.assertEqual(self.imp.add_files([*self.files["exp2"], *self.files["exp10"]], confirm=lambda dialog: False), 0)
        self.assertEqual(len(self.project.experiments), before)

    def test_selections_without_spectra_or_without_tables_are_explained(self):
        self.assertEqual(self.imp.add_files([self.files["exp1"][1], self.loose_table, self.notes]), 0)
        self.assertIn("não há nenhum de espectros", self.window.message.text())
        self.assertEqual(self.imp.add_files([self.files["exp1"][0], self.text]), 0)
        self.assertIn("não há nenhuma tabela de valores medidos", self.window.message.text())
        self.assertEqual(self.project.experiments, [])

    def test_button_opens_one_file_dialog_for_all_the_files(self):
        from unittest import mock

        from PySide6.QtWidgets import QDialog, QFileDialog

        from multical_gui.pages.import_page import PairsDialog
        self.assertEqual(self.imp.add_several.text(), "Adicionar vários arquivos…")
        with mock.patch.object(QFileDialog, "getOpenFileNames", return_value=(self._all(), "")) as chooser, \
                mock.patch.object(PairsDialog, "exec", return_value=QDialog.Accepted):
            self.imp.add_several.click()
        self.assertEqual(chooser.call_count, 1)
        self.assertEqual(len(self.project.experiments), 3)
        self.assertEqual(self.imp._last_folder, os.path.dirname(self._all()[0]))
        with mock.patch.object(QFileDialog, "getOpenFileNames", return_value=([], "")):      # cancelled: nothing happens
            self.imp.add_several.click()
        self.assertEqual(len(self.project.experiments), 3)


if __name__ == "__main__":
    unittest.main()
