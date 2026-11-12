"""
Calibration and prediction without plots, prints or files.

Rules fixed for this layer:
- one model per analyte, both in validation and in the final model, so the
  reported error belongs to the model that is actually kept;
- pretreatments and the autoscaling of X are computed from the calibration set
  only and reapplied unchanged to any other spectra.
"""
import numpy as np

from ..utils import zscore_matlab_style
from ..var_selection.utils import select_k_ftest
from .config import RunConfig
from .recipe import apply_recipe, fit_recipe
from .registry import MODELS, SPLITTERS
from .results import AnalyteResult, CalibrationResult


class CalibrationCancelled(Exception):
    """Raised when the `cancel` callback asks to stop."""


def _rmse(pred, y, mask):
    return np.sqrt(np.mean((pred[:, mask] - y[mask]) ** 2, axis=1))


def _check_dataset(dataset, names):
    if dataset.Y is None:
        raise ValueError("Não há valores medidos; eles são necessários para criar um modelo.")
    missing = [a for a in names if a not in dataset.analytes]
    if missing:
        raise ValueError(f"Analito(s) não encontrado(s) nos dados: {', '.join(missing)}.")
    columns = [dataset.analytes.index(a) for a in names]
    Y = dataset.Y[:, columns]
    if not np.all(np.isfinite(dataset.X)):
        raise ValueError("Há espectros com valores ausentes ou infinitos.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Há amostras sem valor medido; retire-as antes de criar o modelo.")
    for j, name in enumerate(names):
        if np.ptp(Y[:, j]) == 0:
            raise ValueError(f"Todos os valores medidos de '{name}' são iguais; não é possível criar um modelo.")
    return Y


def run_calibration(dataset, config=None, progress=None, cancel=None):
    """
    Fit and validate one model per analyte.

    progress(fraction, message) is called as work advances; cancel() returning
    True stops the run with CalibrationCancelled.
    """
    config = (config or RunConfig()).resolved()
    names = list(config.analytes or dataset.analytes)
    Y = _check_dataset(dataset, names)
    n = dataset.n_samples

    model = MODELS.get(config.model["id"])
    splitter = SPLITTERS.get(config.validation["id"])
    if splitter.needs_groups and dataset.groups is None:
        raise ValueError(f"'{splitter.label}' precisa saber a qual experimento cada amostra pertence.")

    X_pre, wl_pre, states = fit_recipe(config.pretreatment, dataset.X, dataset.wavelengths)

    if model.scale_x:
        _, x_mean, x_std = zscore_matlab_style(X_pre)
        x_std = np.where(x_std == 0, 1.0, x_std)
        Xs = (X_pre - x_mean) / x_std
    else:
        x_mean = x_std = None
        Xs = X_pre

    rng = np.random.RandomState(config.seed)
    folds = [
        (np.asarray(tr), np.asarray(va))
        for tr, va in splitter.split(n, groups=dataset.groups, X=X_pre, Y=Y, rng=rng, **config.validation["params"])
    ]
    if not folds:
        raise ValueError("A validação escolhida não gerou nenhuma divisão dos dados.")
    validated = np.zeros(n, dtype=bool)
    for _, va in folds:
        validated[va] = True
    smallest_train = min(len(tr) for tr, _ in folds)

    params = dict(config.model["params"])
    if model.sweep:
        max_k = params.pop(model.sweep)
        limit = min(smallest_train - 1, Xs.shape[1])
        if max_k > limit:
            raise ValueError(
                f"O máximo pedido ({max_k}) é maior do que os dados permitem ({limit}): "
                f"o menor grupo de treino tem {smallest_train} amostras e os espectros têm {Xs.shape[1]} pontos."
            )
        candidates = np.arange(1, max_k + 1)
    else:
        max_k = None
        candidates = np.array([0])

    total = len(names) * (len(folds) + 1)
    done = 0

    def step(message):
        nonlocal done
        done += 1
        if progress is not None:
            progress(done / total, message)

    n_cal_eff = smallest_train if splitter.kind == "holdout" else n
    per_analyte = {}
    for j, name in enumerate(names):
        y = Y[:, j]
        y_cv = np.full((len(candidates), n), np.nan)
        for f, (tr, va) in enumerate(folds):
            if cancel is not None and cancel():
                raise CalibrationCancelled()
            if model.sweep:
                y_cv[:, va] = model.cv_path(Xs[tr], y[tr], Xs[va], max_k, **params)
            else:
                y_cv[0, va] = model.predict(model.fit(Xs[tr], y[tr], **params), Xs[va])
            step(f"{name}: validação {f + 1} de {len(folds)}")

        if cancel is not None and cancel():
            raise CalibrationCancelled()
        if model.sweep:
            y_cal = np.asarray(model.cv_path(Xs, y, Xs, max_k, **params))
        else:
            y_cal = np.asarray(model.predict(model.fit(Xs, y, **params), Xs)).reshape(1, n)

        everyone = np.ones(n, dtype=bool)
        rmsecv = _rmse(y_cv, y, validated)
        rmsec = _rmse(y_cal, y, everyone)
        r2cv = 1.0 - rmsecv ** 2 / np.var(y[validated])
        r2cal = 1.0 - rmsec ** 2 / np.var(y)

        if len(candidates) == 1:
            best_index = 0
        elif config.selection == "ftest":
            best_index = int(select_k_ftest(rmsecv, n_cal_eff)) - 1
        else:
            best_index = int(np.argmin(rmsecv))

        if model.sweep:
            state = model.fit(Xs, y, k=int(candidates[best_index]), **params)
        else:
            state = model.fit(Xs, y, **params)
        step(f"{name}: modelo final")

        per_analyte[name] = AnalyteResult(
            name=name, candidates=candidates, rmsecv=rmsecv, rmsec=rmsec, r2cv=r2cv, r2cal=r2cal,
            y_cv=y_cv, y_cal=y_cal, best_index=best_index, model_state=state,
        )

    return CalibrationResult(
        config=config, analytes=names, units=dataset.units,
        wavelengths_in=dataset.wavelengths.copy(), wavelengths=wl_pre, X_pre=X_pre, Y=Y,
        sample_ids=list(dataset.sample_ids), validated=validated, pretreatment_states=states,
        x_mean=x_mean, x_std=x_std, per_analyte=per_analyte,
    )


def predict(result, X, wavelengths):
    """
    Predict every analyte of a calibration for new spectra.

    The spectra must be on the same grid used in the calibration; the stored
    pretreatment and scaling are applied as they were learned.
    Returns an array (n_samples, n_analytes), columns in the order of `result.analytes`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    wavelengths = np.asarray(wavelengths, dtype=float).ravel()
    expected = result.wavelengths_in
    if wavelengths.size != expected.size or not np.allclose(wavelengths, expected, rtol=0, atol=1e-6 * max(1.0, np.ptp(expected))):
        raise ValueError(
            "Estes espectros não estão na mesma faixa/resolução usada para criar o modelo "
            f"({wavelengths.size} pontos recebidos, {expected.size} esperados)."
        )
    X_pre, _ = apply_recipe(result.config.pretreatment, result.pretreatment_states, X, wavelengths)
    if result.x_mean is not None:
        X_pre = (X_pre - result.x_mean) / result.x_std
    model = MODELS.get(result.config.model["id"])
    out = np.empty((X.shape[0], len(result.analytes)))
    for j, name in enumerate(result.analytes):
        out[:, j] = model.predict(result.per_analyte[name].model_state, X_pre)
    return out
