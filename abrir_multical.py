"""Opens the Multical interface from a copy of the repository: `py abrir_multical.py`."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
# Methods added by the user live in the "plugins" folder next to this file.
os.environ.setdefault("MULTICAL_PLUGINS", os.path.join(HERE, "plugins"))

from multical_gui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
