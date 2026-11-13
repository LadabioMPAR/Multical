"""
Methods added without changing the program: every .py file in a plugin folder
is run once, and whatever it registers (pretreatments, models, validation
schemes, importers) shows up next to the built-in methods.

Folders searched: the ones in the environment variable MULTICAL_PLUGINS
(separated by the system path separator) and, in the packaged program, the
folder "plugins" next to the executable. Nothing is loaded from the current
directory. A plugin is ordinary Python code with full access to the machine:
only use files from a trusted source.
"""
import importlib.util
import os
import sys
import traceback

PLUGIN_ERRORS = []      # (file, message) for plugins that failed to load
_loaded = set()


def plugin_folders():
    folders = [f for f in os.environ.get("MULTICAL_PLUGINS", "").split(os.pathsep) if f.strip()]
    if getattr(sys, "frozen", False):
        folders.append(os.path.join(os.path.dirname(sys.executable), "plugins"))
    return folders


def load_plugins(folders=None, reload=False):
    """
    Run the plugin files not loaded yet (every file when `reload` is True).
    Returns the files loaded now; failures go to PLUGIN_ERRORS.
    """
    loaded = []
    for folder in (plugin_folders() if folders is None else folders):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.abspath(os.path.join(folder, name))
            if not name.endswith(".py") or name.startswith("_") or (path in _loaded and not reload):
                continue
            _loaded.add(path)
            try:
                spec = importlib.util.spec_from_file_location(f"multical_plugin_{os.path.splitext(name)[0]}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded.append(path)
            except Exception as error:      # a broken plugin must not stop the program
                PLUGIN_ERRORS.append((path, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"))
    return loaded
