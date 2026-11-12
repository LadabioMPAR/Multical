"""
Does a new spectrum look like the ones the model was built with?

A principal-component summary of the calibration spectra (after pretreatment
and scaling, i.e. what the model actually sees) gives two distances for any
spectrum:

- T² (Hotelling): how far it is from the centre along the usual directions of
  variation; large means "the same kind of spectrum, but more extreme";
- Q (residual): how much of it those directions cannot describe; large means
  "something in this spectrum was never seen in the calibration".

A spectrum is "inside" when both are below their limits. The limits come from
the calibration samples themselves, each one judged by a summary built without
it (cross-validation), so they reflect how unseen samples of the same kind
behave. The settings below are provisional defaults.
"""
import numpy as np
from scipy import stats

MAX_COMPONENTS = 10
VARIANCE_EXPLAINED = 0.95
CONFIDENCE = 0.99
FOLDS = 5


def _components(Xc, count):
    """Loadings (count x p) and variance of each score, from centred data."""
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    variance = s[:count] ** 2 / max(Xc.shape[0] - 1, 1)
    return vt[:count], np.where(variance > 0, variance, 1e-12)


def _distances(X, mean, loadings, variance):
    Xc = X - mean
    scores = Xc @ loadings.T
    t2 = np.sum(scores ** 2 / variance, axis=1)
    q = np.sum((Xc - scores @ loadings) ** 2, axis=1)
    return t2, q


def fit_domain(X):
    """Summary of the calibration spectra `X` (as given to the model). Returns a dict of arrays and numbers."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    mean = X.mean(axis=0)
    _, s, _ = np.linalg.svd(X - mean, full_matrices=False)
    explained = np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-300)
    count = int(np.searchsorted(explained, VARIANCE_EXPLAINED) + 1)
    count = max(1, min(count, MAX_COMPONENTS, n // 4 if n >= 8 else 1))
    loadings, variance = _components(X - mean, count)

    # Distances of every calibration sample to a summary built without it.
    t2_cv, q_cv = np.empty(n), np.empty(n)
    folds = min(FOLDS, n)
    for fold in range(folds):
        held = np.arange(fold, n, folds)
        rest = np.setdiff1d(np.arange(n), held)
        m = X[rest].mean(axis=0)
        ld, var = _components(X[rest] - m, min(count, len(rest) - 1))
        t2_cv[held], q_cv[held] = _distances(X[held], m, ld, var)

    if n > count + 1:
        t2_limit = count * (n - 1) * (n + 1) / (n * (n - count)) * stats.f.ppf(CONFIDENCE, count, n - count)
    else:
        t2_limit = np.inf
    t2_limit = max(float(t2_limit), float(np.max(t2_cv)))
    q_mean, q_var = float(np.mean(q_cv)), float(np.var(q_cv, ddof=1)) if n > 1 else 0.0
    if q_mean > 0 and q_var > 0:      # Box approximation: Q ~ g * chi2(h)
        g, h = q_var / (2 * q_mean), 2 * q_mean ** 2 / q_var
        q_limit = g * stats.chi2.ppf(CONFIDENCE, h)
    else:
        q_limit = q_mean
    q_limit = max(float(q_limit), float(np.max(q_cv)))
    return {"mean": mean, "loadings": loadings, "variance": variance, "t2_limit": t2_limit, "q_limit": q_limit}


def check_domain(domain, X):
    """Returns (t2, q, inside) for the spectra `X` (as given to the model)."""
    t2, q = _distances(np.atleast_2d(np.asarray(X, dtype=float)), domain["mean"], domain["loadings"], domain["variance"])
    inside = (t2 <= domain["t2_limit"]) & (q <= domain["q_limit"])
    return t2, q, inside
