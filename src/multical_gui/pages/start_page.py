"""Screen 0: choose what to do."""
from PySide6.QtWidgets import QHBoxLayout, QPushButton

from .base import Page


class StartPage(Page):
    title = "Início"
    intro = "O Multical cria modelos que estimam concentrações a partir de espectros. Escolha o que deseja fazer."

    def build(self):
        row = QHBoxLayout()
        self.create = QPushButton("Criar um modelo\n\nTenho espectros e valores medidos\nem laboratório.", objectName="big")
        self.use = QPushButton("Usar um modelo pronto\n\nQuero estimar novas amostras\ncom um modelo já criado.", objectName="big")
        row.addWidget(self.create)
        row.addWidget(self.use)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self.create.clicked.connect(lambda: self.window_.start_flow("create"))
        self.use.clicked.connect(lambda: self.window_.start_flow("predict"))
