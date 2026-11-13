"""Reusable pieces: a form generated from parameter descriptions, a plot area and a message banner."""
import re

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QVBoxLayout, QWidget,
)

_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:–|-|a|até)\s*(\d+(?:[.,]\d+)?)")


def format_value(spec, value):
    """A parameter value as short text for lists and summaries."""
    if spec.kind == "bool":
        return "sim" if value else "não"
    if spec.kind == "ranges":
        return "; ".join(f"{lo:g}–{hi:g}" for lo, hi in value) or "tudo"
    if spec.kind == "float":
        return f"{value:g}".replace(".", ",")
    return str(value)


def describe_params(specs, values):
    return ", ".join(f"{s.label.lower()}: {format_value(s, values[s.name])}" for s in specs if s.name in values)


class ParamForm(QWidget):
    """
    One field per `Param`. Parameters marked as advanced are hidden unless
    `set_advanced(True)`. `values()` returns {name: value} ready for the API and
    raises ValueError with a readable message when a field cannot be understood.
    """
    changed = Signal()

    def __init__(self, specs, values=None, advanced=False, parent=None):
        super().__init__(parent)
        self._specs = list(specs)
        self._editors = {}
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        values = values or {}
        for spec in self._specs:
            editor = self._editor(spec, values.get(spec.name, spec.default))
            label = QLabel(spec.label)
            if spec.help:
                label.setToolTip(spec.help)
                editor.setToolTip(spec.help)
            self._layout.addRow(label, editor)
            self._editors[spec.name] = editor
        if not self._specs:
            self._layout.addRow(QLabel("Este passo não tem ajustes."))
        self.set_advanced(advanced)

    def _editor(self, spec, value):
        if spec.kind == "bool":
            editor = QCheckBox()
            editor.setChecked(bool(value))
            editor.toggled.connect(self.changed)
        elif spec.kind == "choice":
            editor = QComboBox()
            for choice in spec.choices:
                editor.addItem(str(choice), choice)
            editor.setCurrentIndex(max(list(spec.choices).index(value) if value in spec.choices else 0, 0))
            editor.currentIndexChanged.connect(self.changed)
        elif spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.min) if spec.min is not None else -1_000_000,
                            int(spec.max) if spec.max is not None else 1_000_000)
            editor.setValue(int(value))
            editor.valueChanged.connect(self.changed)
        elif spec.kind == "float":
            editor = QDoubleSpinBox()
            editor.setDecimals(4)
            editor.setRange(spec.min if spec.min is not None else -1e9, spec.max if spec.max is not None else 1e9)
            editor.setSingleStep(0.1)
            editor.setValue(float(value))
            editor.valueChanged.connect(self.changed)
        else:   # ranges
            editor = QLineEdit(format_value(spec, value) if value else "")
            editor.setPlaceholderText("ex.: 4400–7500; 8000–9000 (vazio: tudo)")
            editor.editingFinished.connect(self.changed)
        return editor

    def set_advanced(self, advanced):
        for row, spec in enumerate(self._specs):
            self._layout.setRowVisible(row, advanced or not spec.advanced)

    def values(self):
        out = {}
        for spec in self._specs:
            editor = self._editors[spec.name]
            if spec.kind == "bool":
                out[spec.name] = editor.isChecked()
            elif spec.kind == "choice":
                out[spec.name] = editor.currentData()
            elif spec.kind in ("int", "float"):
                out[spec.name] = editor.value()
            else:
                text = editor.text().strip()
                pairs = [[float(a.replace(",", ".")), float(b.replace(",", "."))] for a, b in _RANGE.findall(text)]
                if text and not pairs:
                    raise ValueError(f"{spec.label}: escreva as faixas como 4400–7500; 8000–9000.")
                out[spec.name] = pairs
            out[spec.name] = spec.coerce(out[spec.name])
        return out


class PlotCanvas(QWidget):
    """Shows a matplotlib Figure with the usual zoom/pan/save toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = self._toolbar = None
        self.setMinimumSize(360, 210)
        self.show_message("")

    def set_figure(self, figure):
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, 1)
        self.figure = figure

    def show_message(self, text):
        figure = Figure(layout="constrained")
        figure.text(0.5, 0.5, text, ha="center", va="center", color="gray", fontsize=11, wrap=True)
        self.set_figure(figure)


class Banner(QLabel):
    """A message strip: 'info', 'ok', 'warning' or 'error'. Hidden while empty."""
    _COLORS = {
        "info": ("#eef3fb", "#1c3d6e"), "ok": ("#e9f6ee", "#1d5c37"),
        "warning": ("#fdf4dc", "#6b4e00"), "error": ("#fbe9e9", "#8a1f1f"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setMinimumWidth(200)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.set_message("")

    def set_message(self, text, level="info"):
        background, color = self._COLORS[level]
        self.setStyleSheet(f"background:{background}; color:{color}; border-radius:6px; padding:8px 10px;")
        self.setText(text)
        self.setVisible(bool(text))
        self.level = level
