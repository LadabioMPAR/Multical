"""
Q × T² of the calibration samples: which spectra are unlike the others.

- T² (Hotelling): how far a sample is from the centre along the directions the
  model uses; large means an extreme sample of the usual kind.
- Q (residual): how much of the spectrum those directions do not describe;
  large means something in it that the other samples do not have.

For a PLS model this is the influence plot of the scripts
(`multical.core.outliers.evaluate_and_plot_outliers`): the scores of the PLS of
each analyte, with its chosen number of latent variables, and 95% limits. Other
models have no latent variables of their own, so the principal-component
summary that judges new spectra (`domain.py`) is used instead, with its limits.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from ..models.pls import PLS
from ..utils import zscore_matlab_style
from .domain import CONFIDENCE as DOMAIN_CONFIDENCE, check_domain

PLS_CONFIDENCE = 0.95


@dataclass
class Influence:
    analyte: str
    sample_ids: List[str]
    t2: np.ndarray
    q: np.ndarray
    t2_limit: float
    q_limit: float
    source: str             # "pls": latent variables of this analyte's model; "pca": summary of the spectra
    components: int
    confidence: float

    @property
    def outside(self):
        """bool per sample: beyond one of the limits."""
        return (self.t2 > self.t2_limit) | (self.q > self.q_limit)

    def flagged(self):
        """Ids of the samples beyond a limit, the farthest first."""
        distance = np.maximum(self.t2 / self.t2_limit, self.q / self.q_limit)
        return [self.sample_ids[i] for i in np.argsort(-distance) if self.outside[i]]


def _model_input(result):
    X = np.asarray(result.X_pre, dtype=float)
    return X if result.x_mean is None else (X - result.x_mean) / result.x_std


def influence(result, analyte):
    """Q × T² of every calibration sample for the model of one analyte of a CalibrationResult."""
    res = result.per_analyte[analyte]
    j = result.analytes.index(analyte)
    ids = list(result.sample_ids)
    if result.config.model["id"] == "pls" and res.best_k:
        _, mean, std = zscore_matlab_style(result.X_pre)
        X = (result.X_pre - mean) / np.where(std == 0, 1.0, std)
        if res.selected is not None:
            X = X[:, res.selected]
        yn, _, _ = zscore_matlab_style(result.Y[:, [j]])
        pls = PLS()
        _, T, P, _, _, _, _, _ = pls.nipals(X, yn, res.best_k)
        t2, t2_limit, q, q_limit = pls.compute_outlier_metrics(X, T, P, conf_level=PLS_CONFIDENCE)
        return Influence(analyte, ids, np.asarray(t2), np.asarray(q), float(t2_limit), float(q_limit),
                         "pls", int(res.best_k), PLS_CONFIDENCE)
    if not result.domain:
        raise ValueError("Este resultado não guarda o resumo dos espectros necessário para o gráfico Q × T².")
    t2, q, _ = check_domain(result.domain, _model_input(result))
    return Influence(analyte, ids, t2, q, float(result.domain["t2_limit"]), float(result.domain["q_limit"]),
                     "pca", int(np.atleast_2d(result.domain["loadings"]).shape[0]), DOMAIN_CONFIDENCE)
