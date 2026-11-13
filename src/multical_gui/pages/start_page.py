"""Screen 0: choose what to do, or go back to a saved project."""
import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton

from multical import api

from ..app import guide_path
from ..recent import recent_projects
from ..widgets import Banner
from .base import Page


class StartPage(Page):
    title = "Início"
    intro = "O Multical cria modelos que estimam concentrações a partir de espectros. Escolha o que deseja fazer."

    def build(self):
        row = QHBoxLayout()
        self.create = QPushButton("Criar um modelo\n\nTenho espectros e valores medidos\nem laboratório.", objectName="big")
        self.use = QPushButton("Usar um modelo pronto\n\nQuero estimar novas amostras\ncom um modelo já criado.", objectName="big")
        for button in (self.create, self.use):
            button.setMinimumWidth(280)
        row.addWidget(self.create)
        row.addWidget(self.use)
        self.body.addLayout(row)

        links = QHBoxLayout()
        self.open_project = QPushButton("Abrir um projeto salvo…")
        self.guide = QPushButton("Abrir o guia do usuário")
        self.guide.setVisible(guide_path() is not None)
        links.addWidget(self.open_project)
        links.addWidget(self.guide)
        links.addStretch(1)
        self.body.addLayout(links)

        self.recent_label = QLabel("Projetos recentes (duplo clique para abrir):")
        self.recent = QListWidget()
        self.recent.setMaximumHeight(170)
        self.body.addWidget(self.recent_label)
        self.body.addWidget(self.recent)
        self.plugin_errors = Banner()
        self.body.addWidget(self.plugin_errors)
        self.body.addStretch(1)

        self.create.clicked.connect(lambda: self.window_.start_flow("create"))
        self.use.clicked.connect(lambda: self.window_.start_flow("predict"))
        self.open_project.clicked.connect(lambda: self.window_.open_project())
        self.guide.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(guide_path())))
        self.recent.itemActivated.connect(lambda item: self.window_.open_project(item.data(Qt.UserRole)))

    def enter(self):
        paths = recent_projects()
        self.recent.clear()
        for path in paths:
            item = QListWidgetItem(f"{os.path.basename(path)}   —   {os.path.dirname(path)}")
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.recent.addItem(item)
        self.recent_label.setVisible(bool(paths))
        self.recent.setVisible(bool(paths))

        failed = api.PLUGIN_ERRORS
        if failed:
            lines = [f"{os.path.basename(path)}: {text.splitlines()[0]}" for path, text in failed]
            self.plugin_errors.set_message(
                f"{len(failed)} plugin(s) não puderam ser carregados e seus métodos não aparecem nas listas:\n"
                + "\n".join(lines), "error")
        else:
            self.plugin_errors.set_message("")
