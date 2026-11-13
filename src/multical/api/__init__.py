"""
Service layer used by the GUI (and usable from scripts).

Nothing in here opens windows, prints or writes files: functions take data and
return data. Importing this package does not load matplotlib.pyplot.
"""
from .params import Param, resolve_params
from .registry import (
    IMPORTERS, MODELS, PRETREATMENTS, SELECTORS, SPLITTERS,
    Importer, Model, Pretreatment, Selector, Splitter, pretreatment, register,
)
from .dataset import Dataset
from .pairing import MATCH_MODES, TimeMatch, match_times, pair_with_reference
from .recipe import apply_recipe, fit_recipe, normalize_recipe
from .pretreatments import recipe_from_legacy
from . import models as _models, selection as _selection, splitters as _splitters  # noqa: F401  (register the built-ins)
from .config import RunConfig, default_pretreatment
from .results import AnalyteResult, CalibrationResult
from .calibrate import CalibrationCancelled, predict, run_calibration
from .assessment import assess, dataset_warnings, selection_notes, spectrum_warnings
from .bundle import SavedModel, load_legacy_pls, load_model, save_model
from .prediction import Prediction, export_predictions, predict_report
from ..io import spp as _spp, txt_pair as _txt_pair  # noqa: F401,E402  (register the built-in importers)
from .plugins import PLUGIN_ERRORS, load_plugins  # noqa: E402

load_plugins()      # methods added by the user; keep after every built-in registration

__all__ = [
    "Param", "resolve_params",
    "IMPORTERS", "MODELS", "PRETREATMENTS", "SELECTORS", "SPLITTERS",
    "Importer", "Model", "Pretreatment", "Selector", "Splitter", "pretreatment", "register",
    "Dataset", "apply_recipe", "fit_recipe", "normalize_recipe", "recipe_from_legacy",
    "RunConfig", "default_pretreatment", "AnalyteResult", "CalibrationResult",
    "CalibrationCancelled", "predict", "run_calibration",
    "MATCH_MODES", "TimeMatch", "match_times", "pair_with_reference",
    "assess", "dataset_warnings", "selection_notes", "spectrum_warnings", "SavedModel", "load_model", "save_model",
    "load_legacy_pls", "Prediction", "export_predictions", "predict_report",
    "PLUGIN_ERRORS", "load_plugins",
]
