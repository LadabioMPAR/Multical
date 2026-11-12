"""Screen 3: the pretreatment recipe, with the spectra before and after."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from multical import api
from multical.api import plots

from ..widgets import Banner, ParamForm, PlotCanvas, describe_params
from .base import Page


class PretreatPage(Page):
    title = "Pré-tratamento"
    intro = ("Os espectros são corrigidos antes de criar o modelo. A receita padrão do laboratório já está "
             "escolhida; o gráfico mostra os espectros antes e depois.")

    def build(self):
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(QLabel("Receita (os passos são aplicados nesta ordem):"))
        self.steps = QListWidget()
        self.steps.setMaximumHeight(170)
        self.steps.setWordWrap(True)
        column.addWidget(self.steps)

        self.editing = QWidget()
        edit = QHBoxLayout(self.editing)
        edit.setContentsMargins(0, 0, 0, 0)
        self.choices = QComboBox()
        self.choices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.choices.setMinimumContentsLength(18)
        self.add = QPushButton("Adicionar")
        self.remove = QPushButton("Remover")
        self.up = QPushButton("▲")
        self.down = QPushButton("▼")
        for widget in (self.choices, self.add, self.remove, self.up, self.down):
            edit.addWidget(widget, 1 if widget is self.choices else 0)
        column.addWidget(self.editing)

        self.box = QGroupBox("Ajustes do passo selecionado")
        self.box_layout = QVBoxLayout(self.box)
        self.form = None
        self.locked = QLabel("Marque “Avançado” (no alto da janela) para alterar a receita.", objectName="hint")
        self.locked.setWordWrap(True)
        column.addWidget(self.box)
        column.addWidget(self.locked)
        self.restore = QPushButton("Voltar à receita padrão")
        column.addWidget(self.restore)
        self.error = Banner()
        column.addWidget(self.error)
        column.addStretch(1)

        left.setMaximumWidth(470)
        self.canvas = PlotCanvas()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 720])
        self.body.addWidget(splitter, 1)

        for item in sorted(api.PRETREATMENTS, key=lambda p: (p.group, p.label)):
            self.choices.addItem(f"{item.group} – {item.label}" if item.group else item.label, item.id)
        self.steps.currentRowChanged.connect(self._show_form)
        self.add.clicked.connect(self._add)
        self.remove.clicked.connect(self._remove)
        self.up.clicked.connect(lambda: self._move(-1))
        self.down.clicked.connect(lambda: self._move(1))
        self.restore.clicked.connect(lambda: self._set_recipe(api.default_pretreatment(), 0))
        self._advanced = False
        self._problem = None

    # ---- recipe ----------------------------------------------------------
    def _recipe(self):
        return api.normalize_recipe(self.project.config.pretreatment)

    def _set_recipe(self, steps, select=None):
        self.project.set_recipe(steps)
        self._refresh(select)

    def _add(self):
        steps = self._recipe() + [{"id": self.choices.currentData(), "params": {}}]
        self._set_recipe(steps, len(steps) - 1)

    def _remove(self):
        row, steps = self.steps.currentRow(), self._recipe()
        if 0 <= row < len(steps):
            del steps[row]
            self._set_recipe(steps, min(row, len(steps) - 1))

    def _move(self, offset):
        row, steps = self.steps.currentRow(), self._recipe()
        target = row + offset
        if 0 <= row < len(steps) and 0 <= target < len(steps):
            steps[row], steps[target] = steps[target], steps[row]
            self._set_recipe(steps, target)

    def _form_changed(self):
        row, steps = self.steps.currentRow(), self._recipe()
        try:
            steps[row]["params"] = self.form.values()
        except ValueError as error:
            self._problem = str(error)
            self.error.set_message(self._problem, "error")
            self.window_.refresh_nav()
            return
        self.project.set_recipe(steps)
        self._refresh(row, rebuild_form=False)

    # ---- display ---------------------------------------------------------
    def enter(self):
        self._refresh(max(self.steps.currentRow(), 0))

    def set_advanced(self, advanced):
        self._advanced = advanced
        self.editing.setVisible(advanced)
        self.locked.setVisible(not advanced)
        self.box.setEnabled(advanced)
        if self.form is not None:
            self.form.set_advanced(advanced)

    def _refresh(self, select=None, rebuild_form=True):
        steps = self._recipe()
        self.steps.blockSignals(True)
        self.steps.clear()
        for number, step in enumerate(steps, start=1):
            item = api.PRETREATMENTS.get(step["id"])
            detail = describe_params([p for p in item.params if not p.advanced or self._advanced], step["params"])
            self.steps.addItem(f"{number}. {item.label}" + (f" ({detail})" if detail else ""))
        if not steps:
            self.steps.addItem("Sem pré-tratamento: os espectros são usados como estão.")
        if select is not None and steps:
            self.steps.setCurrentRow(max(min(select, len(steps) - 1), 0))
        self.steps.blockSignals(False)
        if rebuild_form:
            self._show_form(self.steps.currentRow())
        self._preview()
        self.window_.refresh_nav()

    def _show_form(self, row):
        if self.form is not None:
            self.box_layout.removeWidget(self.form)
            self.form.setParent(None)
            self.form.deleteLater()
            self.form = None
        steps = self._recipe()
        if 0 <= row < len(steps):
            item = api.PRETREATMENTS.get(steps[row]["id"])
            self.form = ParamForm(item.params, steps[row]["params"], self._advanced)
            self.form.changed.connect(self._form_changed)
            self.box_layout.addWidget(self.form)
            self.box.setTitle(f"Ajustes de “{item.label}”")
        self.box.setVisible(self.form is not None)

    def _preview(self):
        data = self.project.active_dataset()
        self._problem = None
        if data is None or data.n_samples == 0:
            self.canvas.show_message("Importe os dados para ver o efeito do pré-tratamento.")
            self.error.set_message("")
            return
        try:
            X, wl, _ = api.fit_recipe(self.project.config.pretreatment, data.X, data.wavelengths)
            if X.shape[1] == 0:
                raise ValueError("O corte removeu todos os pontos dos espectros.")
        except (ValueError, KeyError) as error:
            self._problem = str(error)
            self.error.set_message(self._problem, "error")
            self.canvas.show_message("A receita não pôde ser aplicada.")
            return
        self.error.set_message("")
        self.canvas.set_figure(plots.plot_before_after(data.wavelengths, data.X, wl, X))

    def validate(self):
        return self._problem
