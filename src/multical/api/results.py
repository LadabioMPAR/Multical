"""Plain data returned by `run_calibration`. Nothing here plots, prints or writes files."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import RunConfig


@dataclass
class AnalyteResult:
    name: str
    candidates: np.ndarray     # complexity values tried (k = 1..max); [0] when the model has no sweep
    rmsecv: np.ndarray         # validation error per candidate, in the analyte's unit
    rmsec: np.ndarray          # calibration error per candidate
    r2cv: np.ndarray
    r2cal: np.ndarray
    y_cv: np.ndarray           # (n_candidates, n_samples); NaN where a sample was never validated
    y_cal: np.ndarray          # (n_candidates, n_samples)
    best_index: int            # position of the chosen candidate
    model_state: dict          # final model, fitted with every calibration sample

    @property
    def best_k(self):
        """Chosen complexity (for PLS: number of latent variables); None without a sweep."""
        value = int(self.candidates[self.best_index])
        return value if value > 0 else None

    @property
    def best_rmsecv(self):
        return float(self.rmsecv[self.best_index])

    @property
    def best_r2cv(self):
        return float(self.r2cv[self.best_index])


@dataclass
class CalibrationResult:
    config: RunConfig               # resolved configuration actually used
    analytes: List[str]
    units: str
    wavelengths_in: np.ndarray      # grid expected from new spectra
    wavelengths: np.ndarray         # grid after pretreatment
    X_pre: np.ndarray               # pretreated calibration spectra
    Y: np.ndarray                   # reference values of the modelled analytes
    sample_ids: List[str]
    validated: np.ndarray           # bool per sample: used at least once for validation
    pretreatment_states: List[dict]
    x_mean: Optional[np.ndarray]    # autoscaling of X (None when the model does not ask for it)
    x_std: Optional[np.ndarray]
    per_analyte: Dict[str, AnalyteResult]
    domain: Optional[dict] = None   # summary of the calibration spectra (see domain.py)
