"""Entry point of the packaged program (the file PyInstaller turns into Multical.exe)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()      # must run first in a packaged program that starts worker processes
    from multical_gui.app import main
    sys.exit(main())
