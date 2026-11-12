"""
Service layer used by the GUI (and usable from scripts).

Nothing in here opens windows, prints or writes files: functions take data and
return data. Importing this package does not load matplotlib.pyplot.
"""
from .params import Param, resolve_params
from .registry import (
    IMPORTERS, MODELS, PRETREATMENTS, SPLITTERS,
    Importer, Model, Pretreatment, Splitter, pretreatment, register,
)
from .dataset import Dataset
from .pairing import MATCH_MODES, TimeMatch, match_times, pair_with_reference
from .recipe import apply_recipe, fit_recipe, normalize_recipe
from .pretreatments import recipe_from_legacy
from . import models as _models, splitters as _splitters  # noqa: F401  (register the built-ins)
from .config import RunConfig, default_pretreatment
from .results import AnalyteResult, CalibrationResult
from .calibrate import CalibrationCancelled, predict, run_calibration
from .assessment import assess, dataset_warnings, spectrum_warnings
from .bundle import SavedModel, load_model, save_model
from ..io import spp as _spp, txt_pair as _txt_pair  # noqa: F401,E402  (register the built-in importers; keep last)

__all__ = [
    "Param", "resolve_params",
    "IMPORTERS", "MODELS", "PRETREATMENTS", "SPLITTERS",
    "Importer", "Model", "Pretreatment", "Splitter", "pretreatment", "register",
    "Dataset", "apply_recipe", "fit_recipe", "normalize_recipe", "recipe_from_legacy",
    "RunConfig", "default_pretreatment", "AnalyteResult", "CalibrationResult",
    "CalibrationCancelled", "predict", "run_calibration",
    "MATCH_MODES", "TimeMatch", "match_times", "pair_with_reference",
    "assess", "dataset_warnings", "spectrum_warnings", "SavedModel", "load_model", "save_model",
]
