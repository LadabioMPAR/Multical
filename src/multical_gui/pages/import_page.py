"""Screen 1: add experiments (spectra + measured values), link them and name the analytes."""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from multical import api
from multical.api.assessment import number

from ..file_pairs import sort_files
from ..state import LINKED_BY, default_experiment_name
from ..widgets import Banner
from .base import Page

SPECTRA_FILTER = "Espectros (*.spp *.txt);;Arquivos do equipamento (*.spp);;Texto (*.txt);;Todos os arquivos (*)"
REFERENCE_FILTER = "Tabelas (*.txt *.csv *.tsv);;Todos os arquivos (*)"
SEVERAL_FILTER = "Espectros e tabelas (*.spp *.txt *.csv *.tsv);;Todos os arquivos (*)"


class AddExperimentDialog(QDialog):
    """Asks for the spectra file, the table of measured values and a name."""

    def __init__(self, parent=None, start_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Adicionar experimento")
        self.setMinimumWidth(680)
        self._folder = start_folder
        self.spectra, self.reference, self.name = QLineEdit(), QLineEdit(), QLineEdit()
        self.name.setPlaceholderText("Nome que aparecerá nos gráficos e tabelas")
        form = QFormLayout(self)
        form.addRow("Espectros (.spp ou .txt):", self._with_button(self.spectra, SPECTRA_FILTER))
        form.addRow("", QLabel(
            "Arquivo .spp gravado pelo equipamento, ou arquivo de texto com o eixo espectral na primeira linha\n"
            "e um espectro por linha, começando pelo tempo em minutos (como os arquivos “_nonda.txt”).",
            objectName="hint"))
        form.addRow("Valores medidos:", self._with_button(self.reference, REFERENCE_FILTER))
        form.addRow("", QLabel(
            "Tabela de texto ou CSV: tempo da amostra em minutos na primeira coluna e um analito por coluna.\n"
            "Uma primeira linha com os nomes dos analitos é opcional.", objectName="hint"))
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


class PairsDialog(QDialog):
    """
    Shows which table of measured values the program would use with each spectra
    file chosen, so the user confirms or corrects it before the experiments are added.
    """
    SKIP = "— não adicionar —"
    COLUMNS = ("Arquivo de espectros", "Valores medidos", "Nome do experimento")

    def __init__(self, pairs, tables, parent=None, notes=()):
        super().__init__(parent)
        self.setWindowTitle("Conferir os experimentos")
        self.setMinimumSize(760, 380)
        self._spectra = [spectra for spectra, _ in pairs]
        self._tables = list(tables)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "O programa juntou cada arquivo de espectros à tabela de valores medidos de nome mais parecido. "
            "Confira cada linha e corrija a tabela se for preciso.", objectName="hint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(len(pairs), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.choices, self.names = [], []
        for row, (spectra, suggested) in enumerate(pairs):
            item = QTableWidgetItem(os.path.basename(spectra))
            item.setFlags(Qt.ItemIsEnabled)
            item.setToolTip(spectra)
            self.table.setItem(row, 0, item)
            choice = QComboBox()
            choice.addItem(self.SKIP, None)
            for path in self._tables:
                choice.addItem(os.path.basename(path), path)
                choice.setItemData(choice.count() - 1, path, Qt.ToolTipRole)
            choice.setCurrentIndex(self._tables.index(suggested) + 1 if suggested in self._tables else 0)
            choice.currentIndexChanged.connect(self._update)
            self.table.setCellWidget(row, 1, choice)
            name = QLineEdit(default_experiment_name(spectra))
            self.table.setCellWidget(row, 2, name)
            self.choices.append(choice)
            self.names.append(name)
        layout.addWidget(self.table, 1)

        self.status = Banner()
        layout.addWidget(self.status)
        self.notes = QLabel("\n".join(notes), objectName="hint")
        self.notes.setWordWrap(True)
        self.notes.setVisible(bool(notes))
        layout.addWidget(self.notes)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    def values(self):
        """[(spectra_path, table_path, name)] for the rows that have a table."""
        return [(spectra, choice.currentData(), name.text().strip())
                for spectra, choice, name in zip(self._spectra, self.choices, self.names) if choice.currentData()]

    def problem(self):
        """Why the experiments cannot be added as they are (None: they can)."""
        chosen = [table for _, table, _ in self.values()]
        if not chosen:
            return "Escolha a tabela de valores medidos de pelo menos um arquivo de espectros."
        repeated = sorted({os.path.basename(t) for t in chosen if chosen.count(t) > 1})
        if repeated:
            return f"A mesma tabela foi escolhida para mais de um arquivo de espectros: {', '.join(repeated)}."
        return None

    def _update(self):
        problem, chosen = self.problem(), {table for _, table, _ in self.values()}
        lines, level = [], "ok"
        without = [os.path.basename(s) for s, c in zip(self._spectra, self.choices) if not c.currentData()]
        spare = [os.path.basename(t) for t in self._tables if t not in chosen]
        if without:
            lines.append(f"Sem tabela (não serão adicionados): {', '.join(without)}.")
        if spare:
            lines.append(f"Tabelas sem uso: {', '.join(spare)}.")
        if lines:
            level = "warning"
        if problem:
            lines, level = [problem] + lines, "error"
        self.status.set_message("\n".join(lines), level)
        count = len(self.values())
        button = self.buttons.button(QDialogButtonBox.Ok)
        button.setText(f"Adicionar {count} experimento{'s' if count != 1 else ''}")
        button.setEnabled(problem is None)


class ImportPage(Page):
    title = "Importar dados"
    intro = ("Adicione cada experimento: o arquivo de espectros (.spp do equipamento ou .txt) e a tabela com os "
             "valores medidos em laboratório. Com “Adicionar vários arquivos”, escolha de uma vez os espectros e as "
             "tabelas de todos os experimentos. O programa liga cada amostra ao seu espectro e mostra abaixo os "
             "analitos encontrados, para você conferir e dar nome a cada um.")

    COLUMNS = ("Experimento", "Arquivo de espectros", "Valores medidos", "Espectros no arquivo", "Amostras ligadas", "Situação")
    ANALYTE_COLUMNS = ("Coluna", "Nome do analito", "Menor valor", "Maior valor")

    def build(self):
        buttons = QHBoxLayout()
        self.add = QPushButton("Adicionar experimento…", objectName="primary")
        self.add_several = QPushButton("Adicionar vários arquivos…")
        self.add_several.setToolTip(
            "Escolha de uma vez os arquivos de espectros e as tabelas de valores medidos de vários experimentos.\n"
            "O programa forma os pares pelo nome dos arquivos e mostra para você conferir.")
        self.remove = QPushButton("Remover selecionado")
        buttons.addWidget(self.add)
        buttons.addWidget(self.add_several)
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
        self.table.setMinimumWidth(300)
        self.body.addWidget(self.table, 3)

        lower = QHBoxLayout()
        box = QGroupBox("Qual espectro representa cada amostra? (quando a ligação é pelo tempo)")
        choices = QVBoxLayout(box)
        row = QHBoxLayout()
        self.exact = QRadioButton("O espectro mais próximo do tempo da amostra")
        self.window_mode = QRadioButton("A média dos espectros em uma janela de ±")
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 600.0)
        self.radius.setDecimals(1)
        self.radius.setSuffix(" min")
        self.exact.setChecked(True)
        choices.addWidget(self.exact)
        row.addWidget(self.window_mode)
        row.addWidget(self.radius)
        row.addStretch(1)
        choices.addLayout(row)
        choices.addStretch(1)
        lower.addWidget(box, 2)

        analytes = QGroupBox("Analitos (dê duplo clique no nome para alterar)")
        analyte_layout = QVBoxLayout(analytes)
        self.analyte_table = QTableWidget(0, len(self.ANALYTE_COLUMNS))
        self.analyte_table.setHorizontalHeaderLabels(self.ANALYTE_COLUMNS)
        self.analyte_table.verticalHeader().setVisible(False)
        self.analyte_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.analyte_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.analyte_table.horizontalHeader().setMinimumSectionSize(70)
        self.analyte_table.setMinimumWidth(340)      # room for the name, which is what the user reads and edits
        self.analyte_table.horizontalHeaderItem(0).setToolTip("Posição da coluna na tabela de valores medidos")
        self.analyte_table.setMinimumHeight(96)
        self.analyte_table.setMaximumHeight(150)
        units = QFormLayout()
        self.units = QLineEdit()
        self.units.setMaximumWidth(120)
        units.addRow("Unidade dos valores medidos:", self.units)
        analyte_layout.addWidget(self.analyte_table)
        analyte_layout.addLayout(units)
        lower.addWidget(analytes, 3)
        self.body.addLayout(lower)

        self.summary = Banner()
        self.body.addWidget(self.summary)
        self.problems = QPlainTextEdit(readOnly=True)
        self.problems.setMaximumHeight(90)
        self.body.addWidget(self.problems)

        self.add.clicked.connect(self._ask_experiment)
        self.add_several.clicked.connect(self._ask_files)
        self.remove.clicked.connect(self._remove_selected)
        self.exact.toggled.connect(self._matching_changed)
        self.radius.valueChanged.connect(self._matching_changed)
        self.analyte_table.itemChanged.connect(self._name_changed)
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

    def _ask_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Escolher os espectros e as tabelas de valores medidos", self._last_folder, SEVERAL_FILTER)
        if paths:
            self._last_folder = os.path.dirname(paths[0])
            self.add_files(paths)

    def add_files(self, paths, confirm=None):
        """
        Add the experiments found in `paths` (spectra files and tables of measured values, in any order).
        The pairs are shown for the user to confirm; `confirm(dialog)` stands in for that in the tests.
        Returns how many experiments were added.
        """
        pairs, tables, ignored = sort_files(paths)
        notes = []
        known = [spectra for spectra, _ in pairs if self.project.has_spectra_file(spectra)]
        if known:
            notes.append(f"Já estavam no projeto e ficaram de fora: {', '.join(os.path.basename(p) for p in known)}.")
            pairs = [(spectra, table) for spectra, table in pairs if spectra not in known]
        if ignored:
            notes.append(f"Não são espectros nem tabelas: {', '.join(os.path.basename(p) for p in ignored)}.")
        tables = tables + [table for _, table in pairs if table is not None]
        problem = None
        if not pairs:
            problem = ("Todos os arquivos de espectros escolhidos já estavam no projeto." if known
                       else "Entre os arquivos escolhidos não há nenhum de espectros (.spp ou .txt com um espectro por linha).")
        elif not tables:
            problem = ("Entre os arquivos escolhidos não há nenhuma tabela de valores medidos. "
                       "Escolha, junto com os espectros, a tabela de cada experimento.")
        if problem:
            self.window_.message.setText(problem)
            return 0
        dialog = PairsDialog(pairs, sorted(tables, key=lambda p: os.path.basename(p).lower()), self, notes)
        accepted = confirm(dialog) if confirm is not None else dialog.exec() == QDialog.Accepted
        if not accepted or dialog.problem():
            dialog.deleteLater()
            return 0
        items = dialog.values()
        dialog.deleteLater()
        self.project.add_experiments(items)
        self.refresh()
        self.window_.message.setText(f"{len(items)} experimento(s) adicionado(s).")
        return len(items)

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

    def rename_analyte(self, position, name):
        """Give the analyte at `position` (0 = first column of values) a new name. Returns False when refused."""
        accepted = self.project.set_analyte_name(position, name)
        self.refresh()
        if not accepted:
            self.window_.message.setText("Cada analito precisa de um nome, e os nomes não podem se repetir.")
        return accepted

    def _name_changed(self, item):
        if item.column() == 1:
            self.rename_analyte(item.row(), item.text())

    def _units_changed(self):
        self.project.set_units(self.units.text())
        self.refresh()

    # ---- display ---------------------------------------------------------
    def enter(self):
        self.refresh()

    def _fill_analytes(self):
        unit = f" {self.project.units}" if self.project.units else ""
        rows = self.project.analyte_summary()
        self.analyte_table.blockSignals(True)
        self.analyte_table.setRowCount(len(rows))
        for position, row in enumerate(rows):
            cells = (f"{row['column']}ª", row["name"], number(row["low"], 4) + unit, number(row["high"], 4) + unit)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column != 1:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setToolTip("Duplo clique para alterar o nome")
                self.analyte_table.setItem(position, column, item)
        self.analyte_table.blockSignals(False)

    def refresh(self):
        project = self.project
        for widget in (self.exact, self.window_mode, self.radius, self.units):
            widget.blockSignals(True)
        (self.exact if project.match_mode == "exact" else self.window_mode).setChecked(True)
        self.radius.setValue(project.radius_min)
        self.radius.setEnabled(project.match_mode == "window")
        self.units.setText(project.units)
        for widget in (self.exact, self.window_mode, self.radius, self.units):
            widget.blockSignals(False)
        self._fill_analytes()

        self.table.setRowCount(len(project.experiments))
        lines = []
        for row, exp in enumerate(project.experiments):
            if exp.error:
                status = exp.error
            else:
                status = "Tudo certo" if not exp.problems else f"{len(exp.problems)} amostra(s) sem espectro"
                status += f" · {LINKED_BY[exp.linked_by]}"
                if exp.note:
                    status += f" · atenção: {exp.note}"
            flagged = bool(exp.error or exp.problems or exp.note)
            cells = (exp.name, os.path.basename(exp.spectra_path), os.path.basename(exp.reference_path),
                     str(exp.n_spectra), str(exp.n_linked), status)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column in (3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 5 and flagged:
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
