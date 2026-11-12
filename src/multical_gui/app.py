"""Program entry point."""
import multiprocessing
import sys


def prepare(app):
    """Same look on every machine: Fusion style, light colours (also when Windows is in dark mode), pt-BR numbers."""
    from PySide6.QtCore import QLocale, Qt

    app.setApplicationName("Multical")
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    QLocale.setDefault(QLocale(QLocale.Portuguese, QLocale.Brazil))


def main(argv=None):
    multiprocessing.freeze_support()      # needed by the packaged .exe when a method uses worker processes
    from PySide6.QtWidgets import QApplication

    from .wizard import MainWindow

    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    prepare(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
