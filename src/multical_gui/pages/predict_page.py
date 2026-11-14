"""Screen "Usar um modelo pronto": open a model file, choose spectra, estimate and save the table."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from multical import api
from multical.api import plots
from multical.api.assessment import number
from multical.api.prediction import calibrated_ranges, typical_errors

from ..widgets import Banner, PlotCanvas
from ..worker import PredictionWorker
from .base import Page

MODEL_FILTER = "Modelos do Multical (*.multical);;Todos os arquivos (*)"
SPECTRA_FILTER = "Espectros (*.spp *.txt);;Todos os arquivos (*)"


def describe_model(model):
    """A few lines telling what a model file estimates and how well."""
    unit = f" {model.units}" if model.units else ""
    head = [", ".join(model.analytes)]
    if model.units:
        head.append(model.units)
    if model.n_samples:
        head.append(f"criado com {model.n_samples} amostras")
    if model.created:
        date = model.created.split("T")[0].split("-")
        head.append("em " + "/".join(reversed(date)))
    lines = [" · ".join(head)]
    for name, error, (low, high) in zip(model.analytes, typical_errors(model), calibrated_ranges(model)):
        parts = []
        if error == error:      # not NaN
            parts.append(f"erro típico ±{number(error)}{unit}")
        if low == low:
            parts.append(f"calibrado de {number(low)} a {number(high)}{unit}")
        selected = getattr(model.per_analyte[name], "selected", None)
        if selected is not None:
            parts.append(f"usa {int(selected.sum())} de {selected.size} pontos do espectro")
        lines.append(f"{name}: " + (", ".join(parts) or "sem informação de validação"))
    return "\n".join(lines)


class PredictPage(Page):
    title = "Estimar amostras"
    intro = ("Abra um modelo criado no Multical e os espectros das amostras novas. O programa aplica o mesmo "
             "pré-tratamento usado no modelo e avisa quando uma amostra foge do que o modelo conhece.")

    def build(self):
        top = QHBoxLayout()
        model_box = QGroupBox("Modelo")
        model_layout = QVBoxLayout(model_box)
        self.open_model = QPushButton("Abrir modelo…", objectName="primary")
        self.model_info = QLabel("Nenhum modelo aberto.", objectName="hint")
        self.model_info.setWordWrap(True)
        self.model_info.setMinimumWidth(200)
        self.model_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        model_layout.addWidget(self.open_model, 0, Qt.AlignLeft)
        model_layout.addWidget(self.model_info, 1)

        files_box = QGroupBox("Espectros a estimar")
        files_layout = QVBoxLayout(files_box)
        row = QHBoxLayout()
        self.add_file = QPushButton("Adicionar arquivo…")
        self.remove_file = QPushButton("Remover")
        self.every = QSpinBox()
        self.every.setRange(1, 1000)
        self.every.setToolTip("Em séries longas, usar um espectro a cada N deixa o cálculo mais rápido.")
        row.addWidget(self.add_file)
        row.addWidget(self.remove_file)
        row.addStretch(1)
        thinning = QHBoxLayout()
        thinning.addWidget(QLabel("Usar um espectro a cada"))
        thinning.addWidget(self.every)
        thinning.addStretch(1)
        self.files = QListWidget()
        self.files.setMaximumHeight(60)
        files_layout.addLayout(row)
        files_layout.addWidget(self.files)
        files_layout.addLayout(thinning)
        top.addWidget(model_box, 1)
        top.addWidget(files_box, 1)
        self.body.addLayout(top)

        run = QHBoxLayout()
        self.run_button = QPushButton("Estimar", objectName="primary")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.cancel_button = QPushButton("Cancelar")
        self.save_button = QPushButton("Salvar tabela (CSV)…")
        run.addWidget(self.run_button)
        run.addWidget(self.progress, 1)
        run.addWidget(self.cancel_button)
        run.addStretch(0)
        run.addWidget(self.save_button)
        self.body.addLayout(run)
        self.status = QLabel(objectName="hint")
        self.body.addWidget(self.status)
        self.banner = Banner()
        self.body.addWidget(self.banner)

        self.tabs = QTabWidget()
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        chart = QWidget()
        chart_layout = QVBoxLayout(chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.analyte = QComboBox()
        self.canvas = PlotCanvas()
        chart_layout.addWidget(self.analyte, 0, Qt.AlignLeft)
        chart_layout.addWidget(self.canvas, 1)
        self.tabs.addTab(self.table, "Tabela")
        self.tabs.addTab(chart, "Ao longo do tempo")
        self.domain_canvas = PlotCanvas()
        self.tabs.addTab(self.domain_canvas, "Q × T²")
        self.tabs.setTabToolTip(2, (
            "Cada espectro estimado comparado com os que criaram o modelo.\n"
            "T²: quanto ele se afasta dentro do que o modelo usa. Q: quanto dele o modelo não descreve.\n"
            "Além de um dos limites, a estimativa merece desconfiança."))
        self.body.addWidget(self.tabs, 1)

        self.model = None
        self.model_path = ""
        self.sources = []
        self.prediction = None
        self.worker = None
        self._folder = ""
        self.open_model.clicked.connect(self._ask_model)
        self.add_file.clicked.connect(self._ask_files)
        self.remove_file.clicked.connect(self._remove_file)
        self.every.valueChanged.connect(self._inputs_changed)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button.clicked.connect(self._ask_save)
        self.analyte.currentIndexChanged.connect(self._plot)
        self._update()

    # ---- inputs ----------------------------------------------------------
    def load_model(self, path):
        if str(path).lower().endswith(".pkl"):
            self.banner.set_message(
                "Modelos antigos (.pkl) não guardam o pré-tratamento e ainda não podem ser abertos por esta tela. "
                "Crie o modelo de novo pelo Multical para obter um arquivo .multical.", "error")
            return False
        try:
            model = api.load_model(path)
        except (ValueError, OSError) as error:
            self.banner.set_message(str(error), "error")
            return False
        self.model, self.model_path = model, str(path)
        self.model_info.setText(f"{os.path.basename(str(path))}\n{describe_model(model)}")
        self.banner.set_message("")
        self._inputs_changed()
        return True

    def add_spectra(self, path):
        path = os.path.normpath(str(path))
        if path not in self.sources:
            self.sources.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.files.addItem(item)
        self._inputs_changed()

    def _ask_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir modelo", self._folder, MODEL_FILTER)
        if path:
            self._folder = os.path.dirname(path)
            self.load_model(path)

    def _ask_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Escolher espectros", self._folder, SPECTRA_FILTER)
        for path in paths:
            self._folder = os.path.dirname(path)
            self.add_spectra(path)

    def _remove_file(self):
        row = self.files.currentRow()
        if 0 <= row < len(self.sources):
            del self.sources[row]
            self.files.takeItem(row)
            self._inputs_changed()

    def _inputs_changed(self):
        self.prediction = None
        self._show_prediction()
        self._update()

    def _update(self):
        running = self.worker is not None
        self.run_button.setEnabled(not running and self.model is not None and bool(self.sources))
        self.progress.setVisible(running)
        self.cancel_button.setVisible(running)
        self.save_button.setEnabled(not running and self.prediction is not None)
        for widget in (self.open_model, self.add_file, self.remove_file, self.every):
            widget.setEnabled(not running)

    # ---- estimating ------------------------------------------------------
    def run(self):
        if self.worker is not None or self.model is None or not self.sources:
            return
        self.banner.set_message("")
        self.status.setText("Lendo os espectros…")
        self.progress.setValue(0)
        self.worker = PredictionWorker(self.model, list(self.sources), self.every.value(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(lambda: self._finish("Cálculo cancelado."))
        self.window_.set_busy(True)
        self._update()
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelando…")

    def _on_progress(self, fraction, message):
        self.progress.setValue(int(fraction * 1000))
        self.status.setText(message)

    def _finish(self, status):
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self.status.setText(status)
        self.window_.set_busy(False)
        self._update()

    def _on_success(self, prediction):
        self.prediction = prediction
        self._finish(f"{len(prediction.sample_ids)} espectros estimados.")
        self._show_prediction()
        flagged = prediction.n_outside
        if flagged:
            self.banner.set_message(
                f"{flagged} de {len(prediction.sample_ids)} amostras fogem do que o modelo conhece. "
                "As estimativas delas merecem desconfiança; veja a coluna “Situação”.", "warning")
        else:
            self.banner.set_message("Todas as amostras estão dentro do que o modelo conhece.", "ok")

    def _on_failure(self, message):
        self.prediction = None
        self._finish("")
        self._show_prediction()
        self.banner.set_message(message, "error")

    # ---- results ---------------------------------------------------------
    def _show_prediction(self):
        prediction = self.prediction
        self.analyte.blockSignals(True)
        self.analyte.clear()
        if prediction is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.canvas.show_message("As estimativas aparecem aqui.")
            self.domain_canvas.show_message("A comparação com os espectros do modelo aparece aqui.")
            self.analyte.blockSignals(False)
            return
        self.domain_canvas.set_figure(plots.plot_prediction_domain(prediction))
        unit = f" ({prediction.units})" if prediction.units else ""
        headers = ["Amostra", "Tempo (min)"] + [f"{name}{unit}" for name in prediction.analytes] + ["Situação"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(prediction.sample_ids))
        last = len(headers) - 1
        for row, sample in enumerate(prediction.sample_ids):
            notes = prediction.notes(row)
            cells = [sample, "" if prediction.times is None else number(float(prediction.times[row]), 6)]
            cells += [number(float(value), 4) for value in prediction.Y[row]]
            if notes:
                cells.append("; ".join(notes))
            else:
                cells.append("dentro do que o modelo conhece" if prediction.inside is not None else "dentro da faixa calibrada")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if 0 < column < last:
                    item.setTextAlignment(Qt.AlignCenter)
                if column == last:
                    item.setForeground(QColor("#b06a00") if notes else QColor("#1d7a45"))
                self.table.setItem(row, column, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(last, QHeaderView.Stretch)
        self.analyte.addItems(prediction.analytes)
        self.analyte.blockSignals(False)
        self._plot()

    def _plot(self):
        if self.prediction is not None and self.analyte.currentText():
            self.canvas.set_figure(plots.plot_prediction_over_time(self.prediction, self.analyte.currentText()))

    def save_table(self, path):
        if self.prediction is None:
            return None
        try:
            written = api.export_predictions(self.prediction, path)
        except OSError as error:
            self.banner.set_message(f"Não foi possível salvar: {error}", "error")
            return None
        self.banner.set_message(f"Tabela salva em {written}", "ok")
        return written

    def _ask_save(self):
        start = os.path.join(self._folder, "estimativas.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Salvar tabela", start, "CSV (*.csv)")
        if path:
            self.save_table(path if path.lower().endswith(".csv") else path + ".csv")
