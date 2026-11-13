"""Program entry point."""
import multiprocessing
import os
import sys


def guide_path():
    """The user guide: next to the executable in the installed program, in docs/ in a copy of the repository."""
    if getattr(sys, "frozen", False):
        base = os.path.join(os.path.dirname(sys.executable), "guia")
    else:
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
    path = os.path.join(base, "guia_do_usuario.html")
    return path if os.path.isfile(path) else None


def prepare(app):
    """Same look on every machine: Fusion style, light colours (also when Windows is in dark mode), pt-BR numbers."""
    from PySide6.QtCore import QLocale, Qt

    app.setApplicationName("Multical")
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    QLocale.setDefault(QLocale(QLocale.Portuguese, QLocale.Brazil))


def main(argv=None):
    multiprocessing.freeze_support()      # needed by the packaged .exe when a method uses worker processes
    argv = sys.argv if argv is None else argv
    if "--autoteste" in argv:              # self-check of an installed copy; see selftest.py
        from .selftest import run
        position = argv.index("--autoteste")
        return run(argv[position + 1] if len(argv) > position + 1 else None)
    from PySide6.QtWidgets import QApplication

    from .wizard import MainWindow

    app = QApplication.instance() or QApplication(argv)
    prepare(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
