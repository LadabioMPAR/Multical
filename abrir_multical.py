"""Opens the Multical interface from a copy of the repository: `py abrir_multical.py`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from multical_gui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
