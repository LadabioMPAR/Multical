"""Reusable pieces: a form generated from parameter descriptions, a plot area and a message banner."""
import re

import numpy as np
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


def _in_sentence(label):
    """A field label inside a sentence: only a capital that starts an ordinary word is lowered ("C" stays)."""
    words = label.split()
    words = [w if not w[:1].isupper() or not w[1:2].islower() else w[0].lower() + w[1:] for w in words[:1]] + words[1:]
    return " ".join(words)


def describe_params(specs, values):
    return ", ".join(f"{_in_sentence(s.label)}: {format_value(s, values[s.name])}" for s in specs
                     if s.name in values and s.is_active(values))


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
        if any(spec.active_when for spec in self._specs):
            self.changed.connect(self._update_active)
            self._update_active()
        self.set_advanced(advanced)

    def _update_active(self):
        """Grey out the fields that the current values make unused (for instance a value chosen automatically)."""
        current = {}
        for spec in self._specs:
            editor = self._editors[spec.name]
            current[spec.name] = (editor.isChecked() if spec.kind == "bool" else
                                  editor.currentData() if spec.kind == "choice" else None)
        for row, spec in enumerate(self._specs):
            active = spec.is_active(current)
            self._editors[spec.name].setEnabled(active)
            label = self._layout.itemAt(row, QFormLayout.LabelRole)
            if label is not None and label.widget() is not None:
                label.widget().setEnabled(active)

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


def nearest_curve(ax, x, Y, px, py, tolerance=8.0):
    """
    Row of Y (curves drawn against x on `ax`) that passes closest to the point
    (px, py) of the screen, in pixels; None when none passes within `tolerance`
    pixels. x must be in increasing or decreasing order.
    """
    x = np.asarray(x, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    (x0, _), (x1, _) = ax.transData.inverted().transform([[px - tolerance, py], [px + tolerance, py]])
    near = np.flatnonzero((x >= min(x0, x1)) & (x <= max(x0, x1)))
    if near.size:
        start, stop = int(near.min()), int(near.max())
    else:       # zoomed in between two points of the spectra
        start = stop = int(np.argmin(np.abs(x - (x0 + x1) / 2)))
    columns = np.arange(max(start - 1, 0), min(stop + 1, x.size - 1) + 1)
    points = ax.transData.transform(np.column_stack([np.tile(x[columns], Y.shape[0]), Y[:, columns].ravel()]))
    points = points.reshape(Y.shape[0], columns.size, 2)
    click = np.array([px, py], dtype=float)
    if columns.size == 1:
        distance = np.linalg.norm(points[:, 0] - click, axis=1)
    else:       # distance to each segment of each curve
        a, ab = points[:, :-1], np.diff(points, axis=1)
        t = np.clip(((click - a) * ab).sum(axis=-1) / np.maximum((ab * ab).sum(axis=-1), 1e-12), 0.0, 1.0)
        distance = np.linalg.norm(a + t[..., None] * ab - click, axis=-1).min(axis=1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    best = int(np.argmin(distance))
    return best if distance[best] <= tolerance else None


class PlotCanvas(QWidget):
    """
    Shows a matplotlib Figure with the usual zoom/pan/save toolbar. `clicked`
    carries the matplotlib event of a left click inside the plot, when the
    toolbar is not zooming or panning.
    """
    clicked = Signal(object)

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
        self._canvas.mpl_connect("button_press_event", self._pressed)
        self.figure = figure

    def _pressed(self, event):
        mode = getattr(self._toolbar, "mode", "")
        if str(getattr(mode, "value", mode)) or event.button != 1 or event.inaxes is None or event.dblclick:
            return
        self.clicked.emit(event)

    def redraw(self):
        """Draw again after the figure was changed in place (keeps the zoom)."""
        self._canvas.draw_idle()

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
