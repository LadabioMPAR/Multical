"""
Extension points.

Five registries hold everything the GUI can offer: IMPORTERS, PRETREATMENTS,
MODELS, SPLITTERS and SELECTORS. Adding a method means writing one class (or function) and
registering it; no interface code changes.
"""
import numpy as np


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, item):
        if not getattr(item, "id", ""):
            raise ValueError(f"{self.kind}: o item precisa de um 'id'.")
        if item.id in self._items:
            raise ValueError(f"{self.kind}: o id {item.id!r} já está registrado.")
        self._items[item.id] = item
        return item

    def unregister(self, item_id):
        self._items.pop(item_id, None)

    def get(self, item_id):
        try:
            return self._items[item_id]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "nenhum"
            raise KeyError(f"{self.kind} {item_id!r} não existe. Disponíveis: {available}.") from None

    def ids(self):
        return list(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def __contains__(self, item_id):
        return item_id in self._items

    def __len__(self):
        return len(self._items)


IMPORTERS = Registry("Importador")
PRETREATMENTS = Registry("Pré-tratamento")
MODELS = Registry("Modelo")
SPLITTERS = Registry("Validação")
SELECTORS = Registry("Seleção de variáveis")


def register(registry):
    """Class decorator: instantiate the class and add it to `registry`."""
    def decorator(cls):
        registry.register(cls())
        return cls
    return decorator


# --------------------------------------------------------------------------- contracts

class Pretreatment:
    """
    A spectral correction step.

    `fit` learns whatever must be remembered from the calibration spectra (for
    example their mean) and returns it as a dict of arrays; `apply` uses that
    state on any spectra. Stateless steps only implement `apply`.
    """
    id = ""
    label = ""
    group = ""
    params = ()

    def fit(self, X, wavelengths, **params):
        return {}

    def apply(self, X, wavelengths, state, **params):
        """Return (X_new, wavelengths_new)."""
        raise NotImplementedError


class Model:
    """
    A regression model for ONE analyte (y is a 1-D vector).

    If `sweep` names one of the params (for example "max_k"), the model has a
    complexity value k = 1..max: the engine calls `fit(X, y, k=..., **others)`
    and `cv_path(...)`; `sweep_label` says what k counts (singular, plural).

    Otherwise the engine calls `fit(X, y, **params)`. Such a model may offer
    several settings to be tried by validation: `candidates(**params)` returns a
    list of (description, {param: value}) and the engine keeps the setting with
    the lowest validation error.

    `scale_x = True` asks the engine to autoscale X with the mean and standard
    deviation of the whole calibration set before `fit`/`predict`.
    """
    id = ""
    label = ""
    params = ()
    sweep = None
    sweep_label = ("nível de complexidade", "níveis de complexidade")
    scale_x = False

    def candidates(self, **params):
        """Settings to try, as [(description, {param: value}), ...]; None when there is a single setting."""
        return None

    def fit(self, X, y, **params):
        """Return the fitted state (dict of arrays and numbers)."""
        raise NotImplementedError

    def predict(self, state, X):
        """Return predictions as a 1-D array."""
        raise NotImplementedError

    def cv_path(self, X_train, y_train, X_val, max_k, **params):
        """Predictions on X_val for k = 1..max_k, shape (max_k, n_val)."""
        out = np.empty((max_k, X_val.shape[0]))
        for k in range(1, max_k + 1):
            out[k - 1] = self.predict(self.fit(X_train, y_train, k=k, **params), X_val)
        return out

    def cv_folds(self, X, y, folds, max_k, **params):
        """
        `cv_path` for every (train, validation) pair of `folds` at once: shape
        (max_k, n_samples), NaN where a sample is never left out.

        A variable selection calls this thousands of times, so a model may
        override it with a faster calculation that gives the same numbers up to
        rounding. The validation that is reported always uses `cv_path`.
        """
        out = np.full((max_k, X.shape[0]), np.nan)
        for train, val in folds:
            out[:, val] = self.cv_path(X[train], y[train], X[val], max_k, **params)
        return out


class Splitter:
    """
    A validation scheme. `kind` is "cv" (every sample is validated once) or
    "holdout" (one split). `split` yields (train_idx, val_idx) pairs.
    """
    id = ""
    label = ""
    params = ()
    kind = "cv"
    needs_groups = False

    def split(self, n, groups=None, X=None, Y=None, rng=None, **params):
        raise NotImplementedError


class Selector:
    """
    Variable selection: chooses which points of the spectrum the model of ONE
    analyte uses.

    `select(X, y, error, rng, progress, **params)` receives the pretreated
    calibration spectra (n_samples, n_points) and the measured values (1-D), and
    returns a bool vector with one item per point (True: the point is used) or
    the positions of the chosen points.

    `error(mask)` gives the validation error of the chosen model when only the
    points of `mask` are used (the lowest among the complexities or settings
    tried), in the unit of the analyte; it also stops the run when the user
    cancels. `error.folds` holds the (train, validation) positions and
    `error.wavelengths` the axis. `rng` is a numpy RandomState made from the
    seed of the run; `progress(fraction, message="")` reports how far it is.
    """
    id = ""
    label = ""
    params = ()

    def select(self, X, y, error, rng, progress, **params):
        raise NotImplementedError


class Importer:
    """Reads files into a `Dataset`."""
    id = ""
    label = ""
    extensions = ()

    def read(self, sources, **options):
        raise NotImplementedError


class _FunctionPretreatment(Pretreatment):
    def __init__(self, item_id, label, group, params, func):
        self.id, self.label, self.group, self.params = item_id, label, group, tuple(params)
        self._func = func

    def apply(self, X, wavelengths, state, **params):
        return self._func(X, wavelengths, **params)


def pretreatment(id, label, group="", params=()):
    """Register a stateless function `f(X, wavelengths, **params) -> (X, wavelengths)`."""
    def decorator(func):
        PRETREATMENTS.register(_FunctionPretreatment(id, label, group, params, func))
        return func
    return decorator
