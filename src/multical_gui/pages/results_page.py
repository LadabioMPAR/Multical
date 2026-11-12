"""Screen 5: say in plain language how good each model is, and show why."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from multical import api
from multical.api import plots
from multical.api.assessment import LEVELS, R2_ATTENTION, R2_GOOD, number

from ..widgets import Banner, PlotCanvas
from .base import Page

COLORS = {"bom": "#2e9e5b", "atencao": "#e0a100", "ruim": "#cc3b3b"}


def _dot(color):
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 14, 14)
    painter.end()
    return QIcon(pixmap)


class ResultsPage(Page):
    title = "Resultados"
    intro = ("Cada linha resume um modelo. O “erro típico” é quanto a estimativa costuma se afastar do valor "
             "medido em amostras que o modelo não viu no treino.")

    def build(self):
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        self.analytes = QListWidget()
        self.analytes.setWordWrap(True)
        self.analytes.setMaximumHeight(210)
        column.addWidget(self.analytes)
        self.legend = QLabel(objectName="hint")
        self.legend.setWordWrap(True)
        self.legend.setText(
            f"Verde: R² de validação a partir de {number(R2_GOOD)}. Amarelo: a partir de {number(R2_ATTENTION)}. "
            "Vermelho: abaixo disso. Estes limites são provisórios.")
        column.addWidget(self.legend)
        self.warnings = Banner()
        column.addWidget(self.warnings)
        column.addWidget(QLabel("Amostras com erro grande na validação:"))
        self.suspects = QPlainTextEdit(readOnly=True)
        column.addWidget(self.suspects, 1)

        left.setMaximumWidth(480)
        self.tabs = QTabWidget()
        self.scatter = PlotCanvas()
        self.curve = PlotCanvas()
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.scatter, "Previsto × medido")
        self.tabs.addTab(self.curve, "Erro por variáveis latentes")
        self.tabs.addTab(self.table, "Tabela completa")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 700])
        self.body.addWidget(splitter, 1)
        self.analytes.currentRowChanged.connect(self._show_analyte)
        self._shown = None

    def set_advanced(self, advanced):
        self.tabs.setTabVisible(1, advanced)
        self.tabs.setTabVisible(2, advanced)

    def enter(self):
        result = self.project.result
        self.analytes.blockSignals(True)
        self.analytes.clear()
        self._items = api.assess(result) if result is not None else []
        for item in self._items:
            entry = QListWidgetItem(_dot(COLORS[item.level]), f"{item.analyte} — {LEVELS[item.level]}\n{item.summary}")
            self.analytes.addItem(entry)
        self.analytes.blockSignals(False)
        data = self.project.active_dataset()
        notes = api.dataset_warnings(data) if data is not None else []
        self.warnings.set_message("\n".join(notes), "warning")
        if self._items:
            self.analytes.setCurrentRow(0)
            self._show_analyte(0)

    def _show_analyte(self, row):
        result = self.project.result
        if result is None or not 0 <= row < len(self._items):
            return
        item = self._items[row]
        res = result.per_analyte[item.analyte]
        self.scatter.set_figure(plots.plot_predicted_vs_measured(result, item.analyte))
        if len(res.candidates) > 1:
            self.curve.set_figure(plots.plot_error_curve(result, item.analyte))
        else:
            self.curve.show_message("Este modelo não tem um número de variáveis latentes a escolher.")
        self.suspects.setPlainText("\n".join(item.suspects) or "Nenhuma amostra se destacou.")

        headers = ["Variáveis latentes", "Erro de validação", "R² de validação", "Erro de calibração", "R² de calibração"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(len(res.candidates))
        for i, k in enumerate(res.candidates):
            values = [str(int(k)) + (" (escolhido)" if i == res.best_index else ""),
                      number(res.rmsecv[i], 4), number(res.r2cv[i], 4), number(res.rmsec[i], 4), number(res.r2cal[i], 4)]
            for column, text in enumerate(values):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, column, cell)
