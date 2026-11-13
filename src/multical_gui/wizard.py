"""The main window: numbered steps on the left, one screen at a time, Voltar/Avançar at the bottom."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .pages.export_page import ExportPage
from .pages.import_page import ImportPage
from .pages.inspect_page import InspectPage
from .pages.model_page import ModelPage
from .pages.predict_page import PredictPage
from .pages.pretreat_page import PretreatPage
from .pages.results_page import ResultsPage
from .pages.start_page import StartPage
from .recent import forget_project, recent_projects, remember_project
from .state import PROJECT_EXTENSION, Project

PROJECT_FILTER = f"Projetos do Multical (*{PROJECT_EXTENSION});;Todos os arquivos (*)"

STYLE = """
QWidget { font-size: 10pt; }
QListWidget#steps { background: #1f2a3a; color: #c9d3e2; border: none; padding: 8px 0; outline: 0; }
QListWidget#steps::item { padding: 10px 16px; }
QListWidget#steps::item:selected { background: #34507a; color: white; }
QListWidget#steps::item:disabled { color: #66728a; }
QLabel#title { font-size: 16pt; font-weight: 600; }
QLabel#intro { color: #445; padding-bottom: 6px; }
QLabel#hint { color: #667; }
QPushButton { padding: 6px 14px; }
QPushButton#primary { background: #2f6fd0; color: white; border: none; border-radius: 4px; font-weight: 600; }
QPushButton#primary:disabled { background: #a9bcd9; }
QPushButton#big { font-size: 13pt; padding: 26px; text-align: left; }
"""


class MainWindow(QMainWindow):
    def __init__(self, project=None):
        super().__init__()
        self.project = project or Project()
        self.setWindowTitle("Multical")
        self.resize(1200, 760)
        self.setMinimumSize(960, 620)
        self.setStyleSheet(STYLE)

        self.steps = QListWidget(objectName="steps")
        self.steps.setFixedWidth(250)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stack = QStackedWidget()
        self.title = QLabel(objectName="title")
        self.advanced = QCheckBox("Avançado")
        self.advanced.setToolTip("Mostra os ajustes para quem já conhece calibração multivariada.")
        self.message = QLabel(objectName="hint")
        self.message.setWordWrap(True)
        self.back = QPushButton("◀ Voltar")
        self.next = QPushButton("Avançar ▶", objectName="primary")

        create = [cls(self.project, self) for cls in
                  (StartPage, ImportPage, InspectPage, PretreatPage, ModelPage, ResultsPage, ExportPage)]
        self.predict_page = PredictPage(self.project, self)
        # Two paths share the first screen: creating a model (the numbered steps) and using one.
        self.flows = {"create": create, "predict": [create[0], self.predict_page]}
        for page in create + [self.predict_page]:
            self.stack.addWidget(page)
        self.pages = []
        self.current = 0
        self.set_flow("create")

        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.advanced)
        footer = QHBoxLayout()
        footer.addWidget(self.message, 1)
        footer.addWidget(self.back)
        footer.addWidget(self.next)
        right = QVBoxLayout()
        right.setContentsMargins(18, 14, 18, 14)
        right.addLayout(header)
        right.addWidget(self.stack, 1)
        right.addLayout(footer)
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.steps)
        layout.addLayout(right, 1)
        self.setCentralWidget(central)

        self._build_menu()
        self.steps.currentRowChanged.connect(self._step_clicked)
        self.back.clicked.connect(lambda: self.go_to(self.current - 1))
        self.next.clicked.connect(lambda: self.go_to(self.current + 1))
        self.advanced.toggled.connect(self._set_advanced)
        self._show(0)

    # ---- project files ---------------------------------------------------
    def _build_menu(self):
        menu = self.menuBar().addMenu("&Arquivo")

        def action(text, slot, shortcut=None):
            item = QAction(text, self)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            item.triggered.connect(lambda _checked=False: slot())
            menu.addAction(item)
            return item

        action("&Novo projeto", self.new_project, "Ctrl+N")
        action("&Abrir projeto…", self.open_project, "Ctrl+O")
        self.recent_menu = menu.addMenu("Abrir &recente")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        menu.addSeparator()
        self.save_action = action("&Salvar projeto", self.save_project, "Ctrl+S")
        action("Salvar projeto &como…", self.save_project_as, "Ctrl+Shift+S")
        menu.addSeparator()
        action("Sai&r", self.close)

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        paths = recent_projects()
        for path in paths:
            item = self.recent_menu.addAction(path)
            item.triggered.connect(lambda _checked=False, p=path: self.open_project(p))
        if not paths:
            self.recent_menu.addAction("(nenhum projeto recente)").setEnabled(False)

    def ask_about_unsaved(self):
        """Ask what to do with changes not saved yet: returns "save", "discard" or "cancel"."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Multical")
        box.setText("O projeto tem alterações que ainda não foram salvas.")
        save = box.addButton("Salvar", QMessageBox.AcceptRole)
        discard = box.addButton("Não salvar", QMessageBox.DestructiveRole)
        box.addButton("Cancelar", QMessageBox.RejectRole)
        box.exec()
        return "save" if box.clickedButton() is save else "discard" if box.clickedButton() is discard else "cancel"

    def _may_leave_project(self):
        """Before replacing or closing the project: False when the user wants to stay."""
        if not self.project.dirty or (self.project.is_empty and self.project.path is None):
            return True
        choice = self.ask_about_unsaved()
        if choice == "save":
            return self.save_project()
        return choice == "discard"

    def _show_project(self):
        """After the project changed as a whole: back to the numbered steps, on the import screen."""
        self.advanced.setChecked(self.project.advanced)
        self.set_flow("create")
        self.current = 0
        self.go_to(1 if not self.project.is_empty else 0)

    def new_project(self):
        if not self._may_leave_project():
            return False
        self.project.reset()
        self._show_project()
        return True

    def open_project(self, path=None):
        if not self._may_leave_project():
            return False
        if path is None:
            start = os.path.dirname(self.project.path) if self.project.path else ""
            path, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", start, PROJECT_FILTER)
            if not path:
                return False
        try:
            self.project.open(path)
        except ValueError as error:
            forget_project(path)
            QMessageBox.warning(self, "Multical", str(error))
            self.refresh_nav()
            return False
        remember_project(path)
        self._show_project()
        unusable = [e.name for e in self.project.experiments if e.error]
        if unusable:
            self.message.setText(
                f"O projeto foi aberto, mas {len(unusable)} experimento(s) não puderam ser lidos; veja a coluna “Situação”.")
        return True

    def save_project(self):
        if self.project.path is None:
            return self.save_project_as()
        return self._write_project(self.project.path)

    def save_project_as(self, path=None):
        if path is None:
            start = self.project.path or os.path.join(os.path.expanduser("~"), "Documents", "projeto" + PROJECT_EXTENSION)
            path, _ = QFileDialog.getSaveFileName(self, "Salvar projeto", start, PROJECT_FILTER)
            if not path:
                return False
        return self._write_project(path)

    def _write_project(self, path):
        try:
            saved = self.project.save(path)
        except OSError as error:
            QMessageBox.warning(self, "Multical", f"Não foi possível salvar o projeto: {error}")
            return False
        remember_project(saved)
        self.message.setText(f"Projeto salvo em {saved}")
        self._update_title()
        return True

    def _update_title(self):
        name = os.path.basename(self.project.path) if self.project.path else "projeto sem nome"
        mark = "*" if self.project.dirty else ""
        self.setWindowTitle("Multical" if self.project.is_empty and self.project.path is None else f"Multical — {name}{mark}")

    def closeEvent(self, event):
        if self._may_leave_project():
            event.accept()
        else:
            event.ignore()

    # ---- navigation ------------------------------------------------------
    def set_flow(self, name):
        """Show the steps of one path ("create" or "predict") in the side bar."""
        self.flow = name
        self.pages = self.flows[name]
        self.steps.blockSignals(True)
        self.steps.clear()
        for number, page in enumerate(self.pages):
            self.steps.addItem(QListWidgetItem(page.title if number == 0 else f"{number}. {page.title}"))
        self.steps.setCurrentRow(0)      # never leave the list without a current row (see _show)
        self.steps.blockSignals(False)

    def start_flow(self, name):
        """Leave the first screen along one of the paths."""
        self.set_flow(name)
        self.go_to(1)

    def blocking_message(self, target):
        """Why the screen `target` cannot be opened yet (None when it can)."""
        for page in self.pages[:target]:
            reason = page.validate()
            if reason:
                return reason
        return None

    def go_to(self, target):
        if not 0 <= target < len(self.pages):
            return False
        reason = self.blocking_message(target) if target > self.current else None
        if reason:
            self.message.setText(reason)
            self.refresh_nav()
            return False
        self._show(target)
        return True

    def _show(self, index):
        # Swapping the visible screen can move the keyboard focus to the side bar, and a list that
        # receives focus may change its current row; its signal stays blocked until refresh_nav sets the
        # row, so that cannot send the user to another screen behind their back.
        self.steps.blockSignals(True)
        self.current = index
        page = self.pages[index]
        page.set_advanced(self.advanced.isChecked())
        page.enter()
        self.stack.setCurrentWidget(page)
        self.title.setText(page.title)
        self.message.setText("")
        self.refresh_nav()

    def _step_clicked(self, row):
        if row != self.current and row >= 0:
            self.go_to(row)
            self.refresh_nav()

    def refresh_nav(self):
        """Enable the steps the user may open and the Voltar/Avançar buttons."""
        self.steps.blockSignals(True)
        for row in range(len(self.pages)):
            item = self.steps.item(row)
            open_ = row <= self.current or self.blocking_message(row) is None
            item.setFlags((item.flags() | Qt.ItemIsEnabled) if open_ else (item.flags() & ~Qt.ItemIsEnabled))
        self.steps.setCurrentRow(self.current)
        self.steps.blockSignals(False)
        last = self.current == len(self.pages) - 1
        self.back.setVisible(self.current > 0)
        self.next.setVisible(self.current > 0 and not last)
        self.next.setEnabled(not last and self.pages[self.current].validate() is None)
        self._update_title()

    def set_busy(self, busy):
        """While a calculation runs the user cannot leave the screen."""
        self.steps.setEnabled(not busy)
        self.menuBar().setEnabled(not busy)
        self.back.setEnabled(not busy)
        self.advanced.setEnabled(not busy)
        if busy:
            self.next.setEnabled(False)
        else:
            self.refresh_nav()

    def _set_advanced(self, advanced):
        self.project.advanced = advanced
        self.pages[self.current].set_advanced(advanced)
