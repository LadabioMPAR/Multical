"""What every screen of the assistant provides."""
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class Page(QWidget):
    title = ""          # shown in the side bar and at the top of the screen
    intro = ""          # one or two sentences telling the user what to do here

    def __init__(self, project, window):
        super().__init__()
        self.project = project
        self.window_ = window
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        if self.intro:
            label = QLabel(self.intro)
            label.setWordWrap(True)
            label.setObjectName("intro")
            self.body.addWidget(label)
        self.build()

    def build(self):
        """Create the widgets (called once)."""

    def enter(self):
        """Refresh from the project (called every time the screen is shown)."""

    def validate(self):
        """Return a sentence telling why the user cannot go on, or None when the step is complete."""
        return None

    def set_advanced(self, advanced):
        """Show or hide the settings meant for experienced users."""
