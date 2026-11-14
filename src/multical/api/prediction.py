"""
Estimating new samples with a model: the values, the validation error of the
model (RMSECV or RMSEP) that goes with them and whether each spectrum is inside
what the model knows.
"""
import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .assessment import error_name
from .calibrate import CalibrationCancelled, analyte_input, model_input
from .contract import check_predictions
from .domain import check_domain
from .registry import MODELS

RANGE_MARGIN = 0.10     # an estimate may leave the calibrated range by this fraction of it before being flagged


@dataclass
class Prediction:
    analytes: List[str]
    units: str
    sample_ids: List[str]
    groups: Optional[np.ndarray]
    times: Optional[np.ndarray]     # as in the dataset given (the interface uses minutes)
    Y: np.ndarray                   # (n_samples, n_analytes)
    typical_error: np.ndarray       # validation error (RMSECV/RMSEP) of each analyte's model (NaN when unknown)
    t2: Optional[np.ndarray]        # distances to the calibration spectra; None when the model has no summary of them
    q: Optional[np.ndarray]
    inside: Optional[np.ndarray]    # bool per sample: the spectrum looks like the calibration ones
    ranges: np.ndarray              # (n_analytes, 2): lowest and highest measured value in the calibration (NaN: unknown)
    in_range: np.ndarray            # (n_samples, n_analytes) bool: the estimate is within the calibrated range
    t2_limit: Optional[float] = None   # limits that `inside` was judged against (None with t2 and q)
    q_limit: Optional[float] = None
    error_name: str = "RMSECV"         # what typical_error is: RMSECV, or RMSEP for a separate test set

    @property
    def ok(self):
        """bool per sample: nothing was flagged."""
        fine = self.in_range.all(axis=1)
        return fine if self.inside is None else fine & self.inside

    @property
    def n_outside(self):
        return int(np.sum(~self.ok))

    def notes(self, row):
        """Sentences explaining why a sample was flagged (empty list: nothing to say)."""
        out = []
        if self.inside is not None and not self.inside[row]:
            out.append("espectro diferente dos usados para criar o modelo")
        for j, name in enumerate(self.analytes):
            if not self.in_range[row, j]:
                lo, hi = (f"{v:.3g}".replace(".", ",") for v in self.ranges[j])
                out.append(f"{name} fora da faixa calibrada ({lo} a {hi})")
        return out


def typical_errors(model):
    """Validation error (RMSECV or RMSEP) of each analyte of a CalibrationResult or of a model read from a file."""
    out = []
    for name in model.analytes:
        entry = model.per_analyte[name]
        out.append(entry.best_rmsecv if hasattr(entry, "best_rmsecv") else entry.rmsecv)
    return np.array(out, dtype=float)


def calibrated_ranges(model):
    """(n_analytes, 2) lowest and highest measured value of each analyte in the calibration; NaN when unknown."""
    out = np.full((len(model.analytes), 2), np.nan)
    Y = getattr(model, "Y", None)
    for j, name in enumerate(model.analytes):
        if Y is not None:
            out[j] = np.min(Y[:, j]), np.max(Y[:, j])
        else:
            entry = model.per_analyte[name]
            if getattr(entry, "y_min", None) is not None and getattr(entry, "y_max", None) is not None:
                out[j] = entry.y_min, entry.y_max
    return out


def predict_report(model, dataset, progress=None, cancel=None, chunk=50):
    """
    Estimate every analyte for the spectra of `dataset`.

    Works through the spectra `chunk` at a time, calling progress(fraction, message)
    and stopping with CalibrationCancelled when cancel() returns True.
    """
    n = dataset.n_samples
    fitted = MODELS.get(model.config.model["id"])
    domain = getattr(model, "domain", None)
    Y = np.empty((n, len(model.analytes)))
    t2 = q = inside = None
    if domain:
        t2, q, inside = np.empty(n), np.empty(n), np.empty(n, dtype=bool)
    for start in range(0, n, chunk):
        if cancel is not None and cancel():
            raise CalibrationCancelled()
        rows = slice(start, min(start + chunk, n))
        X = model_input(model, dataset.X[rows], dataset.wavelengths)
        for j, name in enumerate(model.analytes):
            entry = model.per_analyte[name]
            Y[rows, j] = check_predictions(fitted, fitted.predict(entry.model_state, analyte_input(entry, X)), X.shape[0])
        if domain:
            t2[rows], q[rows], inside[rows] = check_domain(domain, X)
        if progress is not None:
            progress(rows.stop / n, f"{rows.stop} de {n} espectros")
    ranges = calibrated_ranges(model)
    margin = RANGE_MARGIN * (ranges[:, 1] - ranges[:, 0])
    with np.errstate(invalid="ignore"):
        in_range = ~((Y < ranges[:, 0] - margin) | (Y > ranges[:, 1] + margin))     # unknown range: never flagged
    return Prediction(
        analytes=list(model.analytes), units=model.units, sample_ids=list(dataset.sample_ids),
        groups=dataset.groups, times=dataset.times, Y=Y, typical_error=typical_errors(model),
        t2=t2, q=q, inside=inside, ranges=ranges, in_range=in_range,
        t2_limit=float(domain["t2_limit"]) if domain else None, q_limit=float(domain["q_limit"]) if domain else None,
        error_name=error_name(model.config),
    )


def _cell(value):
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.6g}".replace(".", ",")


def export_predictions(prediction, path):
    """Write the estimates as CSV (';' between columns, decimal comma)."""
    unit = f"_{prediction.units}" if prediction.units else ""
    header = ["amostra", "experimento", "tempo_min"]
    for name in prediction.analytes:
        header += [f"{name}{unit}", f"{name}_{prediction.error_name.lower()}"]
    header += ["situacao", "distancia_T2", "residuo_Q"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        for i, sample in enumerate(prediction.sample_ids):
            row = [sample, "" if prediction.groups is None else str(prediction.groups[i]),
                   "" if prediction.times is None else _cell(float(prediction.times[i]))]
            for j in range(len(prediction.analytes)):
                row += [_cell(float(prediction.Y[i, j])), _cell(float(prediction.typical_error[j]))]
            notes = prediction.notes(i)
            if prediction.inside is None:
                row += [" / ".join(notes) or "espectro não avaliado (modelo antigo)", "", ""]
            else:
                row += [" / ".join(notes) or "dentro do que o modelo conhece",
                        _cell(float(prediction.t2[i])), _cell(float(prediction.q[i]))]
            writer.writerow(row)
    return path
