"""Screen 1: add experiments (spectra + measured values) and link them."""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton, QTableWidget, QTableWidgetItem,
)

from multical import api

from ..state import default_experiment_name
from ..widgets import Banner
from .base import Page

SPECTRA_FILTER = "Espectros (*.spp *.txt);;Todos os arquivos (*)"
REFERENCE_FILTER = "Tabelas (*.txt *.csv *.tsv);;Todos os arquivos (*)"


class AddExperimentDialog(QDialog):
    """Asks for the spectra file, the table of measured values and a name."""

    def __init__(self, parent=None, start_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Adicionar experimento")
        self.setMinimumWidth(640)
        self._folder = start_folder
        self.spectra, self.reference, self.name = QLineEdit(), QLineEdit(), QLineEdit()
        self.name.setPlaceholderText("Nome que aparecerá nos gráficos e tabelas")
        form = QFormLayout(self)
        form.addRow("Espectros (.spp):", self._with_button(self.spectra, SPECTRA_FILTER))
        form.addRow("Valores medidos:", self._with_button(self.reference, REFERENCE_FILTER))
        form.addRow("", QLabel("Tabela de texto ou CSV: tempo da amostra em minutos na primeira coluna\n"
                               "e um analito por coluna.", objectName="hint"))
        form.addRow("Nome do experimento:", self.name)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Adicionar")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.spectra.textChanged.connect(self._update)
        self.reference.textChanged.connect(self._update)
        self._update()

    def _with_button(self, edit, file_filter):
        row = QHBoxLayout()
        button = QPushButton("Procurar…")
        button.clicked.connect(lambda: self._browse(edit, file_filter))
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _browse(self, edit, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, "Escolher arquivo", self._folder, file_filter)
        if path:
            edit.setText(os.path.normpath(path))
            self._folder = os.path.dirname(path)

    def _update(self):
        spectra = self.spectra.text().strip()
        if spectra and not self.name.isModified():
            self.name.setText(default_experiment_name(spectra))
        ready = os.path.isfile(spectra) and os.path.isfile(self.reference.text().strip())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ready)

    def values(self):
        return self.spectra.text().strip(), self.reference.text().strip(), self.name.text().strip()


class ImportPage(Page):
    title = "Importar dados"
    intro = ("Adicione cada experimento: o arquivo de espectros do equipamento e a tabela com os valores "
             "medidos em laboratório. O programa procura, para cada amostra, o espectro registrado no mesmo momento.")

    COLUMNS = ("Experimento", "Arquivo de espectros", "Valores medidos", "Espectros no arquivo", "Amostras ligadas", "Situação")

    def build(self):
        buttons = QHBoxLayout()
        self.add = QPushButton("Adicionar experimento…", objectName="primary")
        self.remove = QPushButton("Remover selecionado")
        buttons.addWidget(self.add)
        buttons.addWidget(self.remove)
        buttons.addStretch(1)
        self.body.addLayout(buttons)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.Stretch)
        self.body.addWidget(self.table, 2)

        box = QGroupBox("Qual espectro representa cada amostra? (arquivos .spp)")
        row = QHBoxLayout(box)
        self.exact = QRadioButton("O espectro mais próximo do tempo da amostra")
        self.window_mode = QRadioButton("A média dos espectros em uma janela de ±")
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 600.0)
        self.radius.setDecimals(1)
        self.radius.setSuffix(" min")
        self.exact.setChecked(True)
        row.addWidget(self.exact)
        row.addSpacing(24)
        row.addWidget(self.window_mode)
        row.addWidget(self.radius)
        row.addStretch(1)
        self.body.addWidget(box)

        names = QFormLayout()
        self.analytes = QLineEdit()
        self.analytes.setPlaceholderText("Nomes separados por vírgula, na ordem das colunas da tabela")
        self.units = QLineEdit()
        self.units.setMaximumWidth(120)
        names.addRow("Nomes dos analitos:", self.analytes)
        names.addRow("Unidade dos valores medidos:", self.units)
        self.body.addLayout(names)

        self.summary = Banner()
        self.body.addWidget(self.summary)
        self.problems = QPlainTextEdit(readOnly=True)
        self.problems.setMaximumHeight(110)
        self.body.addWidget(self.problems)

        self.add.clicked.connect(self._ask_experiment)
        self.remove.clicked.connect(self._remove_selected)
        self.exact.toggled.connect(self._matching_changed)
        self.radius.valueChanged.connect(self._matching_changed)
        self.analytes.editingFinished.connect(self._names_changed)
        self.units.editingFinished.connect(self._units_changed)
        self._last_folder = ""

    # ---- actions ---------------------------------------------------------
    def add_experiment(self, spectra_path, reference_path, name=None):
        self.project.add_experiment(spectra_path, reference_path, name)
        self.refresh()

    def _ask_experiment(self):
        dialog = AddExperimentDialog(self, self._last_folder)
        if dialog.exec() == QDialog.Accepted:
            spectra, reference, name = dialog.values()
            self._last_folder = os.path.dirname(spectra)
            self.add_experiment(spectra, reference, name)

    def _remove_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.project.experiments):
            self.project.remove_experiment(row)
            self.refresh()

    def _matching_changed(self):
        mode = "exact" if self.exact.isChecked() else "window"
        if (mode, self.radius.value()) != (self.project.match_mode, self.project.radius_min):
            self.project.match_mode, self.project.radius_min = mode, self.radius.value()
            self.project.rebuild()
            self.refresh()

    def _names_changed(self):
        names = self.analytes.text().split(",")
        if self.project.dataset is not None and not self.project.set_analyte_names(names):
            count = len(self.project.dataset.analytes)
            self.window_.message.setText(f"Informe {count} nomes diferentes, separados por vírgula.")
        self.refresh()

    def _units_changed(self):
        self.project.set_units(self.units.text())
        self.refresh()

    # ---- display ---------------------------------------------------------
    def enter(self):
        self.refresh()

    def refresh(self):
        project = self.project
        for widget in (self.exact, self.window_mode, self.radius, self.analytes, self.units):
            widget.blockSignals(True)
        (self.exact if project.match_mode == "exact" else self.window_mode).setChecked(True)
        self.radius.setValue(project.radius_min)
        self.radius.setEnabled(project.match_mode == "window")
        self.analytes.setText(", ".join(project.analyte_names))
        self.units.setText(project.units)
        for widget in (self.exact, self.window_mode, self.radius, self.analytes, self.units):
            widget.blockSignals(False)

        self.table.setRowCount(len(project.experiments))
        lines = []
        for row, exp in enumerate(project.experiments):
            status = exp.error or ("Tudo certo" if not exp.problems else f"{len(exp.problems)} amostra(s) sem espectro")
            cells = (exp.name, os.path.basename(exp.spectra_path), os.path.basename(exp.reference_path),
                     str(exp.n_spectra), str(exp.n_linked), status)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 5 and (exp.error or exp.problems):
                    item.setForeground(Qt.red if exp.error else Qt.darkYellow)
                item.setToolTip(exp.spectra_path if column == 1 else exp.reference_path if column == 2 else text)
                self.table.setItem(row, column, item)
            lines += [f"{exp.name}: {exp.error}"] if exp.error else [f"{exp.name}: {p}" for p in exp.problems]
        self.problems.setPlainText("\n".join(lines))
        self.problems.setVisible(bool(lines))
        self.remove.setEnabled(bool(project.experiments))

        data = project.dataset
        if data is None:
            self.summary.set_message("Nenhum experimento adicionado ainda." if not project.experiments
                                     else "Nenhum experimento pôde ser usado. Veja a situação de cada um.",
                                     "info" if not project.experiments else "error")
        else:
            used = sum(1 for e in project.experiments if not e.error)
            text = (f"{data.n_samples} amostras com valor medido · {len(data.analytes)} analito(s) · "
                    f"{used} experimento(s) · espectros de {data.wavelengths[0]:g} a {data.wavelengths[-1]:g} "
                    f"com {data.wavelengths.size} pontos")
            warnings = api.dataset_warnings(data)
            self.summary.set_message(text + ("\n" + "\n".join(warnings) if warnings else ""), "warning" if warnings else "ok")
        self.window_.refresh_nav()

    def validate(self):
        if self.project.dataset is None:
            return "Adicione pelo menos um experimento com espectros e valores medidos."
        return None
