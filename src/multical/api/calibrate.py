"""
Calibration and prediction without plots, prints or files.

Rules fixed for this layer:
- one model per analyte, both in validation and in the final model, so the
  reported error belongs to the model that is actually kept;
- pretreatments and the autoscaling of X are computed from the calibration set
  only and reapplied unchanged to any other spectra;
- a variable selection, when asked, is run for each analyte with that analyte's
  own model, and its choice of points goes with the model.
"""
import numpy as np

from ..utils import zscore_matlab_style
from ..var_selection.utils import select_k_ftest
from .config import RunConfig
from .contract import (
    check_cv_path, check_predictions, check_selection, check_settings_path, check_split, check_state,
)
from .domain import fit_domain
from .recipe import apply_recipe, fit_recipe
from .registry import MODELS, SELECTORS, SPLITTERS, Model
from .results import AnalyteResult, CalibrationResult

SELECTION_SHARE = 0.9       # part of an analyte's progress taken by the variable selection, when there is one
REFINE_ROUNDS = 2           # rounds of settings near the best one, for models that offer them (Model.refine)


class CalibrationCancelled(Exception):
    """Raised when the `cancel` callback asks to stop."""


class SubsetError:
    """
    What a variable selection calls to know how good a choice of points is:
    `error(mask)` is the validation error of the analyte's model using only the
    points of `mask` (the lowest among the complexities or settings tried).
    """

    def __init__(self, curve, folds, wavelengths, cancel=None):
        self._curve, self._cancel = curve, cancel
        self.folds = folds                  # [(train positions, validation positions), ...]
        self.wavelengths = wavelengths      # axis of the spectra given to the selection
        self.evaluations = 0

    def _mask(self, mask):
        mask = np.asarray(mask)
        if mask.dtype.kind == "b":
            return mask
        out = np.zeros(self.wavelengths.size, dtype=bool)
        out[mask.astype(int)] = True
        return out

    def curve(self, mask):
        """Validation error of every complexity or setting tried, using only the points of `mask`."""
        if self._cancel is not None and self._cancel():
            raise CalibrationCancelled()
        self.evaluations += 1
        return self._curve(self._mask(mask))

    def __call__(self, mask):
        mask = self._mask(mask)
        if not mask.any():
            return float("inf")
        values = self.curve(mask)
        return float(np.min(values)) if np.all(np.isfinite(values)) else float("inf")


def selection_seed(seed, position):
    """Seed of the variable selection of the analyte at `position` (None: not repeatable)."""
    return None if seed is None else (int(seed) + 7919 * (position + 1)) % (2 ** 32)


def _setting_key(extra):
    """A setting {param: value} as something that can be compared and put in a set."""
    return tuple(sorted((name, round(float(value), 12) if isinstance(value, (int, float)) and not isinstance(value, bool)
                         else value) for name, value in extra.items()))


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
    selector = SELECTORS.get(config.variable_selection["id"]) if config.variable_selection else None
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
        check_split(splitter, tr, va, n)
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
        grid = None
    else:
        max_k = None
        grid = model.candidates(**params) or None
    base_settings = list(grid) if grid else [("", {})]
    refines = bool(grid) and type(model).refine is not Model.refine
    n_cal_eff = smallest_train if splitter.kind == "holdout" else n

    def stop_if_cancelled():
        if cancel is not None and cancel():
            raise CalibrationCancelled()

    def candidates_for(n_points, settings=base_settings):
        """Complexities (1..max, never more than the points in use) or settings tried."""
        if model.sweep:
            return np.arange(1, min(max_k, n_points) + 1)
        return np.arange(1, len(settings) + 1) if grid else np.array([0])

    def fit_setting(X_fit, y_fit, extra):
        """Fit the model with one setting {param: value} (models without a sweep)."""
        return check_state(model, model.fit(X_fit, y_fit, **{**params, **extra}))

    def settings_predictions(X_fit, y_fit, X_out, settings):
        """Predictions on X_out of the model fitted with each of `settings`, shape (len(settings), len(X_out))."""
        if type(model).settings_path is Model.settings_path:
            rows = []
            for _, extra in settings:
                stop_if_cancelled()
                rows.append(check_predictions(model, model.predict(fit_setting(X_fit, y_fit, extra), X_out), len(X_out)))
            return np.vstack(rows)
        values = model.settings_path(X_fit, y_fit, X_out, [extra for _, extra in settings], **params)
        return check_settings_path(model, values, len(settings), len(X_out))

    def validate(X, y, settings=base_settings, on_fold=None):
        """Predictions of every candidate for the samples left out of the training, shape (n_candidates, n)."""
        n_candidates = len(candidates_for(X.shape[1], settings))
        y_cv = np.full((n_candidates, n), np.nan)
        for f, (tr, va) in enumerate(folds):
            stop_if_cancelled()
            if model.sweep:
                y_cv[:, va] = check_cv_path(
                    model, model.cv_path(X[tr], y[tr], X[va], n_candidates, **params), n_candidates, len(va))
            else:
                y_cv[:, va] = settings_predictions(X[tr], y[tr], X[va], settings)
            if on_fold is not None:
                on_fold(f)
        return y_cv

    def refined(X, y, y_cv, step):
        """The settings tried and their predictions, after the rounds of settings near the best one."""
        settings = list(base_settings)
        if not refines:
            return settings, y_cv
        for number in range(1, REFINE_ROUNDS + 1):
            stop_if_cancelled()
            known = {_setting_key(extra) for _, extra in settings}
            more = []
            for label, extra in model.refine(settings, _rmse(y_cv, y, validated), number, **params) or []:
                key = _setting_key(extra)
                if key not in known:
                    known.add(key)
                    more.append((label, dict(extra)))
            if not more:
                break
            y_cv = np.vstack([y_cv, validate(X, y, more, on_fold=lambda f: step(number, f))])
            settings += more
        return settings, y_cv

    def search_curve(X, y):
        """Validation error of every candidate with the points of X; the quick path of the variable selection."""
        if not model.sweep:
            return _rmse(validate(X, y), y, validated)
        top = len(candidates_for(X.shape[1]))
        with np.errstate(all="ignore"):
            y_cv = check_cv_path(model, model.cv_folds(X, y, folds, top, **params), top, n)
            return _rmse(y_cv, y, validated)

    def choose(rmsecv):
        if len(rmsecv) == 1:
            return 0
        if config.selection == "ftest" and model.sweep:      # the F-test only makes sense for a growing complexity
            return int(select_k_ftest(rmsecv, n_cal_eff)) - 1
        return int(np.argmin(rmsecv))

    def report(position, fraction, message):
        if progress is not None:
            progress((position + min(max(fraction, 0.0), 1.0)) / len(names), message)

    per_analyte = {}
    for j, name in enumerate(names):
        y = Y[:, j]
        selected = full_rmsecv = None
        start = 0.0
        if selector is not None:
            stop_if_cancelled()
            report(j, 0.0, f"{name}: escolhendo os pontos do espectro")
            full_curve = _rmse(validate(Xs, y), y, validated)
            full_rmsecv = float(full_curve[choose(full_curve)])
            error = SubsetError(lambda mask: search_curve(Xs[:, mask], y), folds, wl_pre, cancel)

            def selection_progress(fraction, message=""):
                text = f"{name}: escolhendo os pontos do espectro" + (f" – {message}" if message else "")
                report(j, SELECTION_SHARE * min(max(float(fraction), 0.0), 1.0), text)

            chosen = selector.select(Xs, y, error, np.random.RandomState(selection_seed(config.seed, j)),
                                     selection_progress, **config.variable_selection["params"])
            selected = check_selection(selector, chosen, Xs.shape[1])
            start = SELECTION_SHARE
        Xj = Xs if selected is None else Xs[:, selected]
        steps = len(folds) * (1 + (REFINE_ROUNDS if refines else 0)) + 1

        def step(done, message):
            report(j, start + (1.0 - start) * done / steps, message)

        y_cv = validate(Xj, y, on_fold=lambda f: step(f + 1, f"{name}: validação {f + 1} de {len(folds)}"))
        settings, y_cv = refined(Xj, y, y_cv, lambda number, f: step(
            len(folds) * number + f + 1, f"{name}: refinando os ajustes perto do melhor ({number} de {REFINE_ROUNDS})"))
        candidates = candidates_for(Xj.shape[1], settings)

        stop_if_cancelled()
        if model.sweep:
            y_cal = check_cv_path(model, model.cv_path(Xj, y, Xj, len(candidates), **params), len(candidates), n)
        else:
            y_cal = settings_predictions(Xj, y, Xj, settings)

        everyone = np.ones(n, dtype=bool)
        rmsecv = _rmse(y_cv, y, validated)
        rmsec = _rmse(y_cal, y, everyone)
        r2cv = 1.0 - rmsecv ** 2 / np.var(y[validated])
        r2cal = 1.0 - rmsec ** 2 / np.var(y)
        best_index = choose(rmsecv)

        if model.sweep:
            state = check_state(model, model.fit(Xj, y, k=int(candidates[best_index]), **params))
        else:
            state = fit_setting(Xj, y, settings[best_index][1])
        step(steps, f"{name}: modelo final")

        per_analyte[name] = AnalyteResult(
            name=name, candidates=candidates, rmsecv=rmsecv, rmsec=rmsec, r2cv=r2cv, r2cal=r2cal,
            y_cv=y_cv, y_cal=y_cal, best_index=best_index, model_state=state,
            labels=[label for label, _ in settings] if grid else None,
            selected=selected, full_rmsecv=full_rmsecv,
        )

    return CalibrationResult(
        config=config, analytes=names, units=dataset.units,
        wavelengths_in=dataset.wavelengths.copy(), wavelengths=wl_pre, X_pre=X_pre, Y=Y,
        sample_ids=list(dataset.sample_ids), validated=validated, pretreatment_states=states,
        x_mean=x_mean, x_std=x_std, per_analyte=per_analyte, domain=fit_domain(Xs),
    )


def _same_grid(a, b):
    return a.size == b.size and np.allclose(a, b, rtol=0, atol=1e-6 * max(1.0, np.ptp(b)))


def model_input(result, X, wavelengths):
    """
    New spectra as the fitted model expects them: the grid is checked, then the
    stored pretreatment and scaling are applied as they were learned.
    `result` is a CalibrationResult or a model read from a file.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    wavelengths = np.asarray(wavelengths, dtype=float).ravel()
    expected = result.wavelengths_in
    if expected is not None and not _same_grid(wavelengths, expected):
        raise ValueError(
            "Estes espectros não estão na mesma faixa/resolução usada para criar o modelo "
            f"({wavelengths.size} pontos recebidos, {expected.size} esperados)."
        )
    X_pre, wl_pre = apply_recipe(result.config.pretreatment, result.pretreatment_states, X, wavelengths)
    expected_pre = getattr(result, "wavelengths_pre", None)
    if expected_pre is not None and not _same_grid(wl_pre, np.asarray(expected_pre, dtype=float).ravel()):
        raise ValueError(
            f"Depois do pré-tratamento os espectros têm {wl_pre.size} pontos e o modelo espera {np.size(expected_pre)}. "
            "Confira se a receita informada é a mesma usada para criar o modelo."
        )
    if result.x_mean is not None:
        X_pre = (X_pre - result.x_mean) / result.x_std
    return X_pre


def analyte_input(entry, X):
    """The columns of the model input (see `model_input`) used by the model of one analyte."""
    selected = getattr(entry, "selected", None)
    if selected is None:
        return X
    selected = np.asarray(selected, dtype=bool)
    if selected.size != X.shape[1]:
        raise ValueError(
            f"O modelo guarda uma escolha de pontos para espectros com {selected.size} pontos, "
            f"mas depois do pré-tratamento estes têm {X.shape[1]}.")
    return X[:, selected]


def predict(result, X, wavelengths):
    """
    Predict every analyte of a calibration for new spectra.

    The spectra must be on the same grid used in the calibration.
    Returns an array (n_samples, n_analytes), columns in the order of `result.analytes`.
    """
    X_pre = model_input(result, X, wavelengths)
    model = MODELS.get(result.config.model["id"])
    out = np.empty((X_pre.shape[0], len(result.analytes)))
    for j, name in enumerate(result.analytes):
        entry = result.per_analyte[name]
        out[:, j] = check_predictions(model, model.predict(entry.model_state, analyte_input(entry, X_pre)), X_pre.shape[0])
    return out
