"""Screen 6: save the model and the results in one folder."""
import datetime
import os

from PySide6.QtCore import QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout

from multical.api import export

from ..widgets import Banner
from .base import Page


def default_folder(model_id):
    documents = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or os.path.expanduser("~")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    return os.path.normpath(os.path.join(documents, "Multical", "execucoes", f"{stamp}_{model_id.upper()}"))


class ExportPage(Page):
    title = "Exportar"
    intro = "Salve o modelo e os resultados em uma pasta. O arquivo .multical é o que será usado para estimar novas amostras."

    def build(self):
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.browse = QPushButton("Procurar…")
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        box = QGroupBox("Salvar em")
        box.setLayout(row)
        self.body.addWidget(box)

        items = QGroupBox("O que salvar")
        column = QVBoxLayout(items)
        self.checks = {}
        for key, label in export.EXPORT_ITEMS.items():
            check = QCheckBox(label)
            check.setChecked(True)
            column.addWidget(check)
            self.checks[key] = check
        self.body.addWidget(items)

        buttons = QHBoxLayout()
        self.export_button = QPushButton("Exportar", objectName="primary")
        self.open_button = QPushButton("Abrir pasta")
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        self.body.addLayout(buttons)
        self.banner = Banner()
        self.body.addWidget(self.banner)
        self.files = QPlainTextEdit(readOnly=True)
        self.body.addWidget(self.files, 1)

        self.browse.clicked.connect(self._browse)
        self.export_button.clicked.connect(self.export)
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._exported)))
        self._exported = None
        self._for_result = None

    def enter(self):
        result = self.project.result
        if result is not self._for_result:      # a new model: suggest a new folder and forget the last export
            self._for_result = result
            self._exported = None
            self.files.clear()
            self.banner.set_message("")
            if result is not None:
                self.folder.setText(default_folder(result.config.model["id"]))
        self.open_button.setEnabled(self._exported is not None)
        self.export_button.setEnabled(result is not None)

    def _browse(self):
        start = os.path.dirname(self.folder.text()) or ""
        chosen = QFileDialog.getExistingDirectory(self, "Escolher a pasta onde salvar", start)
        if chosen:
            self.folder.setText(os.path.normpath(os.path.join(chosen, os.path.basename(self.folder.text()) or "execucao")))

    def export(self):
        """Write the chosen items. Returns the list of files, or None when nothing was written."""
        result, folder = self.project.result, self.folder.text().strip()
        items = [key for key, check in self.checks.items() if check.isChecked()]
        problem = None
        if result is None:
            problem = "Não há modelo treinado para exportar."
        elif not folder:
            problem = "Escolha a pasta onde salvar."
        elif not items:
            problem = "Marque pelo menos um item para salvar."
        elif os.path.isdir(folder) and os.listdir(folder):
            problem = "Esta pasta já tem arquivos. Escolha outro nome para não sobrescrever uma execução anterior."
        if problem:
            self.banner.set_message(problem, "error")
            return None
        try:
            written = export.export_run(result, folder, items)
        except OSError as error:
            self.banner.set_message(f"Não foi possível salvar nesta pasta: {error}", "error")
            return None
        self._exported = folder
        self.open_button.setEnabled(True)
        self.banner.set_message(f"{len(written)} arquivo(s) salvos em {folder}", "ok")
        self.files.setPlainText("\n".join(os.path.relpath(path, folder) for path in written))
        return written
