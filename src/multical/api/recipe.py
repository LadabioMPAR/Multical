"""
A recipe is an ordered list of pretreatment steps: [{"id": ..., "params": {...}}].

`fit_recipe` is used on the calibration spectra and returns the state learned by
each step; `apply_recipe` reuses that state on any other spectra (validation,
prediction), so new samples are never used to compute the correction.
"""
import numpy as np

from .params import resolve_params
from .registry import PRETREATMENTS


def normalize_recipe(steps):
    """Validate ids and parameters; return a new list with every default filled in."""
    out = []
    for step in steps or []:
        item = PRETREATMENTS.get(step["id"])
        out.append({"id": item.id, "params": resolve_params(item.params, step.get("params"), owner=item.label)})
    return out


def fit_recipe(steps, X, wavelengths):
    """Fit and apply every step in order. Returns (X_new, wavelengths_new, states)."""
    X = np.array(X, dtype=float)
    wavelengths = np.array(wavelengths, dtype=float)
    states = []
    for step in normalize_recipe(steps):
        item = PRETREATMENTS.get(step["id"])
        state = item.fit(X, wavelengths, **step["params"])
        X, wavelengths = item.apply(X, wavelengths, state, **step["params"])
        states.append(state)
    return X, wavelengths, states


def apply_recipe(steps, states, X, wavelengths):
    """Apply an already fitted recipe. Returns (X_new, wavelengths_new)."""
    steps = normalize_recipe(steps)
    if len(steps) != len(states):
        raise ValueError("A receita e o estado guardado têm números de passos diferentes.")
    X = np.array(X, dtype=float)
    wavelengths = np.array(wavelengths, dtype=float)
    for step, state in zip(steps, states):
        item = PRETREATMENTS.get(step["id"])
        X, wavelengths = item.apply(X, wavelengths, state, **step["params"])
    return X, wavelengths
