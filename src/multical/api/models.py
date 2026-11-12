"""Built-in models. One model is fitted per analyte."""
import numpy as np

from ..models.pls import PLS
from ..utils import zscore_matlab_style
from .params import Param
from .registry import MODELS, Model, register


def _scale_y(y):
    yn, mean, std = zscore_matlab_style(np.asarray(y, dtype=float).reshape(-1, 1))
    return yn, float(mean[0]), float(std[0])


@register(MODELS)
class PLSModel(Model):
    id = "pls"
    label = "PLS (mínimos quadrados parciais)"
    sweep = "max_k"
    scale_x = True
    params = (
        Param("max_k", "Máximo de variáveis latentes", "int", default=15, min=1,
              help="O programa testa de 1 até este número e escolhe o melhor.", advanced=True),
    )

    def fit(self, X, y, k):
        yn, y_mean, y_std = _scale_y(y)
        B = PLS().nipals(X, yn, k)[0]
        return {"B": np.asarray(B, dtype=float), "y_mean": y_mean, "y_std": y_std, "k": int(k)}

    def predict(self, state, X):
        return (X @ state["B"]).ravel() * state["y_std"] + state["y_mean"]

    def cv_path(self, X_train, y_train, X_val, max_k):
        # NIPALS once up to max_k; the coefficients for each k come from the first k components
        # (same incremental scheme as MulticalEngine.run_cv and calculate_rmsecv_fast).
        yn, y_mean, y_std = _scale_y(y_train)
        _, _, P, _, Q, W, _, _ = PLS().nipals(X_train, yn, max_k)
        out = np.empty((max_k, X_val.shape[0]))
        for k in range(1, max_k + 1):
            wk, pk, qk = W[:, :k], P[:, :k], Q[:, :k]
            beta = wk @ np.linalg.pinv(pk.T @ wk) @ qk.T
            out[k - 1] = (X_val @ beta).ravel() * y_std + y_mean
        return out
