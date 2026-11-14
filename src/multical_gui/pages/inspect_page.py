"""Screen 2: look at the spectra and take out the ones that are clearly wrong."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget

from multical import api
from multical.api import plots

from ..widgets import PlotCanvas, nearest_curve
from .base import Page

MIN_SAMPLES = 6


class InspectPage(Page):
    title = "Inspecionar espectros"
    intro = ("Confira se os espectros fazem sentido. Clique em uma amostra na lista, ou em um espectro no gráfico, "
             "para destacá-la, e desmarque as que não devem entrar no modelo. Nada é removido dos arquivos.")

    def build(self):
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        self.count = QLabel()
        self.samples = QListWidget()
        buttons = QHBoxLayout()
        self.include_all = QPushButton("Incluir todas")
        self.drop_flagged = QPushButton("Retirar as com aviso")
        buttons.addWidget(self.include_all)
        buttons.addWidget(self.drop_flagged)
        column.addWidget(self.count)
        column.addWidget(self.samples, 1)
        column.addLayout(buttons)
        left.setMaximumWidth(380)
        self.canvas = PlotCanvas()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 800])
        self.body.addWidget(splitter, 1)

        self.samples.itemChanged.connect(self._checks_changed)
        self.samples.currentRowChanged.connect(lambda _row: self._show_selected())
        self.canvas.clicked.connect(self._plot_clicked)
        self.include_all.clicked.connect(lambda: self._set_all(lambda row: True))
        self.drop_flagged.clicked.connect(lambda: self._set_all(lambda row: row not in self._flags))
        self._flags = {}
        self._shown = None      # samples of the spectra in the plot, to change only the highlight when they are the same

    def enter(self):
        data = self.project.dataset
        self._flags = api.spectrum_warnings(data) if data is not None else {}
        self.samples.blockSignals(True)
        self.samples.clear()
        if data is not None:
            for row, sample in enumerate(data.sample_ids):
                item = QListWidgetItem(sample + (f"   ⚠ {self._flags[row]}" if row in self._flags else ""))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if sample in self.project.excluded else Qt.Checked)
                self.samples.addItem(item)
        self.samples.blockSignals(False)
        self.drop_flagged.setEnabled(bool(self._flags))
        self._update()

    def _set_all(self, keep):
        self.samples.blockSignals(True)
        for row in range(self.samples.count()):
            self.samples.item(row).setCheckState(Qt.Checked if keep(row) else Qt.Unchecked)
        self.samples.blockSignals(False)
        self._checks_changed()

    def _checks_changed(self, _item=None):
        data = self.project.dataset
        excluded = [data.sample_ids[row] for row in range(self.samples.count())
                    if self.samples.item(row).checkState() != Qt.Checked]
        self.project.set_excluded(excluded)
        self._update()

    def _update(self):
        data = self.project.dataset
        total = 0 if data is None else data.n_samples
        used = total - len(self.project.excluded)
        flagged = f" · {len(self._flags)} com aviso" if self._flags else ""
        self.count.setText(f"{used} de {total} amostras em uso{flagged}")
        self._plot()
        self.window_.refresh_nav()

    def _highlighted(self, data):
        """(position in `data`, name) of the sample chosen in the list; (None, "") when it is not in use."""
        row = self.samples.currentRow()
        if row < 0:
            return None, ""
        sample = self.project.dataset.sample_ids[row]
        return (data.sample_ids.index(sample), sample) if sample in data.sample_ids else (None, "")

    def _plot(self):
        data = self.project.active_dataset()
        if data is None or data.n_samples == 0:
            self._shown = None
            self.canvas.show_message("Nenhuma amostra em uso.")
            return
        highlight, name = self._highlighted(data)
        self.canvas.set_figure(plots.plot_spectra_by_group(
            data.wavelengths, data.X, data.groups, highlight, title=f"Espectros em uso ({data.n_samples})",
            highlight_label=f"Selecionada: {name}"))
        self._shown = list(data.sample_ids)

    def _show_selected(self):
        """Move the highlight to the sample chosen in the list, keeping the zoom of the plot."""
        data = self.project.active_dataset()
        if data is None or self._shown != list(data.sample_ids):
            self._plot()
            return
        highlight, name = self._highlighted(data)
        plots.set_highlight(self.canvas.figure, data.wavelengths, data.X, highlight, f"Selecionada: {name}")
        self.canvas.redraw()

    def _plot_clicked(self, event):
        """Choose in the list the spectrum that passes under the mouse."""
        data = self.project.active_dataset()
        if data is None or self._shown != list(data.sample_ids):
            return
        position = nearest_curve(event.inaxes, data.wavelengths, data.X, event.x, event.y)
        if position is None:
            return
        row = self.project.dataset.sample_ids.index(data.sample_ids[position])
        self.samples.setCurrentRow(row)
        self.samples.scrollToItem(self.samples.item(row))

    def validate(self):
        data = self.project.dataset
        if data is None:
            return None
        used = data.n_samples - len(self.project.excluded)
        if used < MIN_SAMPLES:
            return f"São necessárias pelo menos {MIN_SAMPLES} amostras em uso para criar um modelo (há {used})."
        broken = [sample for sample, complete in zip(data.sample_ids, np.isfinite(data.X).all(axis=1))
                  if not complete and sample not in self.project.excluded]
        if broken:
            return f"Retire as amostras com valores ausentes antes de continuar: {', '.join(broken[:5])}."
        return None
