"""
The package must import both ways:

- as `multical` (installed, or with only `src/` on the path) -- needed by the GUI and the .exe;
- as `src.multical` from the repository root -- how the existing scripts import it.
"""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

WALK_PACKAGE = """
import importlib, pkgutil, sys
import {root} as pkg
failed = []
for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        importlib.import_module(mod.name)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":      # optional dependency (ANN model)
            continue
        failed.append(f"{{mod.name}}: {{exc}}")
    except Exception as exc:
        failed.append(f"{{mod.name}}: {{exc!r}}")
if failed:
    sys.exit("\\n".join(failed))
"""


def _run(code, cwd, pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True)


class ImportTest(unittest.TestCase):
    def test_imports_as_multical_without_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run(WALK_PACKAGE.format(root="multical"), cwd=tmp, pythonpath=SRC)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_imports_as_src_multical_from_repo_root(self):
        proc = _run(WALK_PACKAGE.format(root="src.multical"), cwd=REPO, pythonpath=REPO)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
