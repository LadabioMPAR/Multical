"""The project files used most recently, kept in the user's settings."""
import os

from PySide6.QtCore import QSettings

LIMIT = 8
_KEY = "projetos_recentes"


def _settings():
    return QSettings("LadabioMPAR", "Multical")


def _stored():
    value = _settings().value(_KEY, [])
    if isinstance(value, str):          # a single entry comes back as text
        value = [value]
    return [str(path) for path in (value or [])]


def recent_projects():
    """Paths of recent projects that still exist, most recent first."""
    return [path for path in _stored() if os.path.isfile(path)]


def remember_project(path):
    path = os.path.abspath(str(path))
    paths = [path] + [p for p in _stored() if os.path.normcase(p) != os.path.normcase(path)]
    _settings().setValue(_KEY, paths[:LIMIT])


def forget_project(path):
    _settings().setValue(_KEY, [p for p in _stored() if os.path.normcase(p) != os.path.normcase(os.path.abspath(str(path)))])
