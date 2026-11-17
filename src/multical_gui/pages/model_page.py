"""Screen 4: choose the analytes, then train and test the model in the background."""
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout,
)

from multical import api

from ..widgets import Banner, ParamForm, describe_params
from ..worker import CalibrationWorker
from .base import Page

SELECTION_RULES = (("ftest", "Teste F de Osten: o modelo mais simples que não perde precisão (recomendado)"),
                   ("haaland", "Teste F de Haaland e Thomas: o menor modelo tão bom quanto o de menor erro"),
                   ("min", "Menor erro de validação"))
NO_SELECTION = "Não usar: o modelo usa todos os pontos do espectro"
SELECTION_HINT = ("A seleção de variáveis repete a validação centenas ou milhares de vezes para cada analito: "
                  "pode levar de alguns segundos a vários minutos, e dá para cancelar.")


def duration(seconds):
    """A waiting time in words, rounded as people say it."""
    if seconds < 50:
        return f"{max(5, 5 * round(seconds / 5))} s"
    return f"{max(1, round(seconds / 60))} min"


class ModelPage(Page):
    title = "Modelo e validação"
    intro = ("Escolha o que o modelo deve estimar e clique em “Treinar modelo”. O programa cria um modelo por "
             "analito e o testa com amostras que ficaram de fora do treino.")

    def build(self):
        box = QGroupBox("O que estimar")
        self.analyte_row = QHBoxLayout(box)
        self.analyte_checks = []
        self.body.addWidget(box)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(200)
        self.body.addWidget(self.summary)

        self.advanced_box = QGroupBox("Ajustes avançados")
        form = QFormLayout(self.advanced_box)
        self.model = QComboBox()
        self.model_holder = QVBoxLayout()
        self.validation = QComboBox()
        self.validation_holder = QVBoxLayout()
        self.selection = QComboBox()
        for rule, label in SELECTION_RULES:
            self.selection.addItem(label, rule)
        self.variables = QComboBox()
        self.variables_holder = QVBoxLayout()
        self.variables_hint = QLabel(SELECTION_HINT, objectName="hint")
        self.variables_hint.setWordWrap(True)
        self.variables_holder.addWidget(self.variables_hint)
        for combo in (self.model, self.validation, self.selection, self.variables):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(24)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setToolTip("Usada quando a validação sorteia amostras. A mesma semente repete o mesmo resultado.")
        form.addRow("Modelo:", self.model)
        form.addRow("", self.model_holder)
        form.addRow("Validação:", self.validation)
        form.addRow("", self.validation_holder)
        form.addRow("Escolha da complexidade:", self.selection)
        form.addRow("Seleção de variáveis:", self.variables)
        form.addRow("", self.variables_holder)
        form.addRow("Semente do sorteio:", self.seed)
        # The settings of some methods are many: the box scrolls instead of squeezing its rows or the window.
        self.advanced_scroll = QScrollArea()
        self.advanced_scroll.setWidgetResizable(True)
        self.advanced_scroll.setFrameShape(QFrame.NoFrame)
        self.advanced_scroll.setWidget(self.advanced_box)
        self.body.addWidget(self.advanced_scroll, 100)

        run = QHBoxLayout()
        self.train_button = QPushButton("Treinar modelo", objectName="primary")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.cancel_button = QPushButton("Cancelar")
        run.addWidget(self.train_button)
        run.addWidget(self.progress, 1)
        run.addWidget(self.cancel_button)
        self.body.addLayout(run)
        self.status = QLabel(objectName="hint")
        self.body.addWidget(self.status)
        self.banner = Banner()
        self.body.addWidget(self.banner)
        self.body.addStretch(1)

        for item in api.MODELS:
            self.model.addItem(item.label, item.id)
        for item in api.SPLITTERS:
            self.validation.addItem(item.label, item.id)
        self.variables.addItem(NO_SELECTION, None)
        for item in api.SELECTORS:
            self.variables.addItem(item.label, item.id)
        self.model_form = self.validation_form = self.variables_form = None
        self._started = time.monotonic()
        self.worker = None
        self._advanced = False
        self.model.currentIndexChanged.connect(lambda: self._method_changed("model"))
        self.validation.currentIndexChanged.connect(lambda: self._method_changed("validation"))
        self.variables.currentIndexChanged.connect(lambda: self._method_changed("variables"))
        self.selection.currentIndexChanged.connect(self._settings_changed)
        self.seed.valueChanged.connect(self._settings_changed)
        self.train_button.clicked.connect(self.train)
        self.cancel_button.clicked.connect(self.cancel)
        self._set_running(False)

    # ---- display ---------------------------------------------------------
    def enter(self):
        project, config = self.project, self.project.config
        names = list(project.dataset.analytes) if project.dataset is not None else []
        if [c.text() for c in self.analyte_checks] != names:
            for check in self.analyte_checks:
                self.analyte_row.removeWidget(check)
                check.deleteLater()
            self.analyte_checks = []
            for name in names:
                check = QCheckBox(name)
                check.toggled.connect(self._settings_changed)
                self.analyte_row.insertWidget(len(self.analyte_checks), check)
                self.analyte_checks.append(check)
            self.analyte_row.addStretch(1)
        for widget in (self.model, self.validation, self.selection, self.variables, self.seed, *self.analyte_checks):
            widget.blockSignals(True)
        for check in self.analyte_checks:
            check.setChecked(config.analytes is None or check.text() in config.analytes)
        self.model.setCurrentIndex(max(self.model.findData(config.model["id"]), 0))
        self.validation.setCurrentIndex(max(self.validation.findData(config.validation["id"]), 0))
        self.selection.setCurrentIndex(max(self.selection.findData(config.selection), 0))
        self.seed.setValue(config.seed or 0)
        chosen = config.variable_selection["id"] if config.variable_selection else None
        self.variables.setCurrentIndex(max(self.variables.findData(chosen), 0))
        for widget in (self.model, self.validation, self.selection, self.variables, self.seed, *self.analyte_checks):
            widget.blockSignals(False)
        self._build_forms()
        self._fit_complexity()
        self._describe()

    def set_advanced(self, advanced):
        self._advanced = advanced
        self.advanced_scroll.setVisible(advanced)
        for form in (self.model_form, self.validation_form, self.variables_form):
            if form is not None:
                form.set_advanced(True)

    def _build_forms(self):
        config = self.project.config.resolved()
        for name, holder, registry, entry in (
                ("model", self.model_holder, api.MODELS, config.model),
                ("validation", self.validation_holder, api.SPLITTERS, config.validation),
                ("variables", self.variables_holder, api.SELECTORS, config.variable_selection)):
            old = getattr(self, f"{name}_form")
            if old is not None:
                holder.removeWidget(old)
                old.setParent(None)
                old.deleteLater()
            form = None
            if entry is not None:       # only the variable selection can be absent
                form = ParamForm(registry.get(entry["id"]).params, entry["params"], advanced=True)
                form.changed.connect(self._settings_changed)
                holder.insertWidget(0, form)
            setattr(self, f"{name}_form", form)
        self.variables_hint.setVisible(self.variables_form is not None)
        QTimer.singleShot(0, self._fit_box)     # after Qt has shown the new rows: hidden ones count as empty

    def _fit_box(self):
        """Keep every row at its natural height; when they do not fit, the area around the box scrolls."""
        self.advanced_box.setMinimumHeight(0)
        self.advanced_box.layout().activate()
        self.advanced_box.setMinimumHeight(self.advanced_box.sizeHint().height())

    def _method_changed(self, which):
        if which == "variables":
            chosen = self.variables.currentData()
            self.project.config.variable_selection = {"id": chosen, "params": {}} if chosen else None
        else:
            combo = self.model if which == "model" else self.validation
            setattr(self.project.config, which, {"id": combo.currentData(), "params": {}})
        self.project.result = None
        self._build_forms()
        self._fit_complexity()
        self._describe()

    def _settings_changed(self):
        config = self.project.config
        try:
            config.model = {"id": self.model.currentData(), "params": self.model_form.values()}
            config.validation = {"id": self.validation.currentData(), "params": self.validation_form.values()}
            if self.variables_form is not None:
                config.variable_selection = {"id": self.variables.currentData(), "params": self.variables_form.values()}
        except ValueError as error:
            self.banner.set_message(str(error), "error")
            return
        config.selection = self.selection.currentData()
        config.seed = self.seed.value()
        chosen = [c.text() for c in self.analyte_checks if c.isChecked()]
        config.analytes = None if len(chosen) == len(self.analyte_checks) else chosen
        self.project.result = None
        self.banner.set_message("")
        self._describe()

    def _fit_complexity(self):
        """Lower the number of latent variables when there are too few samples for the value asked."""
        config = self.project.config
        model = api.MODELS.get(config.model["id"])
        try:
            limit = self.project.largest_complexity()
        except ValueError:
            return
        if not model.sweep or limit is None or limit < 1:
            return
        current = config.resolved().model["params"][model.sweep]
        if current > limit:
            config.model = {"id": model.id, "params": {**config.resolved().model["params"], model.sweep: limit}}
            self._build_forms()
            label = next(p.label for p in model.params if p.name == model.sweep)
            self.banner.set_message(
                f"“{label}” foi reduzido para {limit}, que é o que a quantidade de amostras permite.", "info")

    def _describe(self):
        config = self.project.config.resolved()
        model, splitter = api.MODELS.get(config.model["id"]), api.SPLITTERS.get(config.validation["id"])
        data = self.project.active_dataset()
        def described(item, values):
            detail = describe_params(item.params, values)
            return f"{item.label} ({detail})." if detail else f"{item.label}."

        parts = [f"Amostras em uso: {0 if data is None else data.n_samples}.",
                 "Modelo: " + described(model, config.model["params"]),
                 "Teste: " + described(splitter, config.validation["params"])]
        if config.variable_selection:
            selector = api.SELECTORS.get(config.variable_selection["id"])
            main = {p.name: config.variable_selection["params"][p.name] for p in selector.params if not p.advanced}
            parts.append("Seleção de variáveis: " + described(selector, main))
        self.summary.setText("  ".join(parts))
        self.train_button.setEnabled(self.worker is None and any(c.isChecked() for c in self.analyte_checks))
        self.window_.refresh_nav()

    # ---- training --------------------------------------------------------
    def _set_running(self, running):
        self.train_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.progress.setVisible(running)
        self.cancel_button.setVisible(running)
        for widget in (self.advanced_box, *getattr(self, "analyte_checks", [])):
            widget.setEnabled(not running)

    def train(self):
        if self.worker is not None:
            return
        data = self.project.active_dataset()
        if data is None or not any(c.isChecked() for c in self.analyte_checks):
            self.banner.set_message("Escolha pelo menos um analito.", "error")
            return
        self.banner.set_message("")
        self.status.setText("Preparando…")
        self.progress.setValue(0)
        self._started = time.monotonic()
        self.worker = CalibrationWorker(data, self.project.config, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(lambda: self._finish("Treino cancelado."))
        self._set_running(True)
        self.window_.set_busy(True)
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelando…")

    def _on_progress(self, fraction, message):
        self.progress.setValue(int(fraction * 1000))
        elapsed = time.monotonic() - self._started
        if elapsed > 5 and 0.02 < fraction < 1:
            message += f"  ·  faltam cerca de {duration(elapsed * (1 - fraction) / fraction)}"
        self.status.setText(message)

    def _finish(self, status):
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self._set_running(False)
        self.status.setText(status)
        self.window_.set_busy(False)
        self._describe()

    def _on_success(self, result):
        self.project.result = result
        self._finish("Modelo treinado.")
        self.window_.go_to(self.window_.pages.index(self) + 1)

    def _on_failure(self, message):
        self.project.result = None
        self._finish("")
        self.banner.set_message(message, "error")

    def validate(self):
        if self.project.result is None:
            return "Clique em “Treinar modelo” para ver os resultados."
        return None
