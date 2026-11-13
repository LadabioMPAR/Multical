"""
Plain-language reading of a calibration: is each model good enough, and which
samples deserve a second look.

The limits below are provisional defaults, kept in one place so the laboratory
can set its own.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .registry import MODELS

R2_GOOD = 0.90            # validation R² at or above this: "bom"
R2_ATTENTION = 0.70       # at or above this: "atenção"; below: "ruim"
FEW_SAMPLES = 20          # fewer calibration samples than this raises a warning
LARGE_ERROR_FACTOR = 2.5  # a validation error above this many times the typical error flags the sample

LEVELS = {"bom": "Bom", "atencao": "Atenção", "ruim": "Ruim"}


def number(value, digits=3):
    """Format with significant digits and a decimal comma."""
    return f"{value:.{digits}g}".replace(".", ",")


@dataclass
class AnalyteAssessment:
    analyte: str
    level: str          # "bom" | "atencao" | "ruim"
    summary: str        # one line for the results screen
    suspects: List[str]  # sentences about samples with a large validation error


def assess(result):
    """One AnalyteAssessment per analyte of a CalibrationResult."""
    unit = f" {result.units}" if result.units else ""
    one, many = MODELS.get(result.config.model["id"]).sweep_label
    out = []
    for j, name in enumerate(result.analytes):
        res = result.per_analyte[name]
        r2 = res.best_r2cv
        level = "bom" if r2 >= R2_GOOD else "atencao" if r2 >= R2_ATTENTION else "ruim"
        parts = [f"erro típico ±{number(res.best_rmsecv)}{unit}", f"R² de validação {number(r2)}"]
        if res.best_k is not None:
            parts.append(f"{res.best_k} {one if res.best_k == 1 else many}")
        elif res.best_label is not None:
            parts.append(f"ajuste escolhido: {res.best_label}")
        residual = res.y_cv[res.best_index] - result.Y[:, j]
        suspects = []
        order = np.argsort(-np.abs(np.where(result.validated, residual, 0.0)))
        for i in order:
            if not result.validated[i] or abs(residual[i]) <= LARGE_ERROR_FACTOR * res.best_rmsecv:
                break
            suspects.append(
                f"{result.sample_ids[i]}: medido {number(result.Y[i, j])}{unit}, "
                f"previsto {number(res.y_cv[res.best_index, i])}{unit}"
            )
        out.append(AnalyteAssessment(name, level, " · ".join(parts), suspects))
    return out


def dataset_warnings(dataset):
    """Sentences about a calibration set that is likely to give a weak model."""
    warnings = []
    if dataset.Y is not None and dataset.n_samples < FEW_SAMPLES:
        warnings.append(
            f"Só há {dataset.n_samples} amostras com valor medido; com menos de {FEW_SAMPLES} o modelo tende a ser pouco confiável."
        )
    return warnings


def spectrum_warnings(dataset):
    """{position: sentence} for spectra that look wrong (missing values, flat, repeated)."""
    out, seen = {}, {}
    for i, row in enumerate(dataset.X):
        if not np.all(np.isfinite(row)):
            out[i] = "tem valores ausentes"
        elif np.ptp(row) == 0:
            out[i] = "é uma linha reta"
        else:
            key = row.tobytes()
            if key in seen:
                out[i] = f"é igual a {dataset.sample_ids[seen[key]]}"
            else:
                seen[key] = i
    return out


def candidate_name(result):
    """What the candidates of this calibration are called, capitalised (for headers and axis labels)."""
    first = next(iter(result.per_analyte.values()))
    if first.labels is not None:
        return "Ajuste testado"
    return MODELS.get(result.config.model["id"]).sweep_label[1].capitalize()


def candidate_table(result, analyte):
    """(headers, rows of text) with the error of every candidate tried for one analyte."""
    res = result.per_analyte[analyte]
    headers = [candidate_name(result), "Erro de validação", "R² de validação", "Erro de calibração", "R² de calibração"]
    rows = []
    for i, value in enumerate(res.candidates):
        name = res.labels[i] if res.labels is not None else ("único" if int(value) == 0 else str(int(value)))
        rows.append([name + (" (escolhido)" if i == res.best_index and len(res.candidates) > 1 else ""),
                     number(res.rmsecv[i], 4), number(res.r2cv[i], 4), number(res.rmsec[i], 4), number(res.r2cal[i], 4)])
    return headers, rows
