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
    sweep_label = ("variável latente", "variáveis latentes")
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


@register(MODELS)
class PCRModel(Model):
    """
    Principal component regression, as `multical.models.pcr.pcr_model` (the
    components are taken from an SVD of X, which gives the same subspace as the
    eigenvectors of X'X without building that matrix).
    """
    id = "pcr"
    label = "PCR (regressão por componentes principais)"
    sweep = "max_k"
    sweep_label = ("componente principal", "componentes principais")
    scale_x = True
    params = (
        Param("max_k", "Máximo de componentes principais", "int", default=15, min=1,
              help="O programa testa de 1 até este número e escolhe o melhor.", advanced=True),
    )

    @staticmethod
    def _path(X, yn, max_k):
        """Coefficients for k = 1..max_k, one column each."""
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        loadings = vt[:max_k].T                                   # (p, max_k)
        scores = X @ loadings
        out = np.empty((X.shape[1], loadings.shape[1]))
        for k in range(1, loadings.shape[1] + 1):
            beta = np.linalg.lstsq(scores[:, :k], yn, rcond=None)[0]
            out[:, k - 1] = (loadings[:, :k] @ beta).ravel()
        return out

    def fit(self, X, y, k):
        yn, y_mean, y_std = _scale_y(y)
        return {"B": self._path(X, yn, k)[:, [-1]], "y_mean": y_mean, "y_std": y_std, "k": int(k)}

    def predict(self, state, X):
        return (X @ state["B"]).ravel() * state["y_std"] + state["y_mean"]

    def cv_path(self, X_train, y_train, X_val, max_k):
        yn, y_mean, y_std = _scale_y(y_train)
        return (X_val @ self._path(X_train, yn, max_k)).T * y_std + y_mean


@register(MODELS)
class SPAModel(Model):
    """
    Successive projections algorithm followed by least squares on the chosen
    points, as `multical.models.spa.spa_model` with `teste_switch=0`: the points
    are chosen on the spectra as they are, then X and y are scaled with the
    training samples only.
    """
    id = "spa"
    label = "SPA (seleção de pontos do espectro + regressão linear)"
    sweep = "max_k"
    sweep_label = ("ponto do espectro", "pontos do espectro")
    params = (
        Param("max_k", "Máximo de pontos do espectro", "int", default=15, min=1,
              help="O programa testa de 1 até este número de pontos e escolhe o melhor.", advanced=True),
        Param("cini", "Ponto inicial (posição no espectro)", "int", default=0, min=0, advanced=True,
              help="Posição, contada a partir de 0, do ponto por onde a seleção começa."),
    )

    @staticmethod
    def _prepare(X, y, max_k, cini):
        from ..models.spa import spa_clean
        if cini >= X.shape[1]:
            raise ValueError(f"O ponto inicial ({cini}) está além do fim do espectro ({X.shape[1]} pontos).")
        chosen = spa_clean(X, cini, max_k)
        Xn, x_mean, x_std = zscore_matlab_style(X)
        x_std = np.where(np.asarray(x_std) == 0, 1.0, x_std)
        yn, y_mean, y_std = _scale_y(y)
        return chosen, (X - x_mean) / x_std, np.ravel(x_mean), np.ravel(x_std), yn, y_mean, y_std

    def fit(self, X, y, k, cini):
        chosen, Xn, x_mean, x_std, yn, y_mean, y_std = self._prepare(X, y, k, cini)
        beta = np.linalg.lstsq(Xn[:, chosen], yn, rcond=None)[0]
        return {"selected": chosen.astype(int), "beta": np.ravel(beta), "x_mean": x_mean[chosen], "x_std": x_std[chosen],
                "y_mean": y_mean, "y_std": y_std, "k": int(k)}

    def predict(self, state, X):
        chosen = np.asarray(state["selected"], dtype=int)
        return ((X[:, chosen] - state["x_mean"]) / state["x_std"]) @ np.ravel(state["beta"]) * state["y_std"] + state["y_mean"]

    def cv_path(self, X_train, y_train, X_val, max_k, cini):
        # The selection is greedy: the points for k are the first k of the points for max_k.
        chosen, Xn, x_mean, x_std, yn, y_mean, y_std = self._prepare(X_train, y_train, max_k, cini)
        Vn = (X_val - x_mean) / x_std
        out = np.empty((max_k, X_val.shape[0]))
        for k in range(1, max_k + 1):
            beta = np.linalg.lstsq(Xn[:, chosen[:k]], yn, rcond=None)[0]
            out[k - 1] = (Vn[:, chosen[:k]] @ beta).ravel() * y_std + y_mean
        return out


_SVR_KERNELS = ("rbf", "linear")
_SVR_GRID = {"C": (1.0, 10.0, 100.0, 1000.0), "epsilon": (0.01, 0.1), "gamma_factor": (0.1, 1.0, 10.0)}


@register(MODELS)
class SVRModel(Model):
    """
    Support vector regression (scikit-learn) with X and y scaled by the training
    samples, as `multical.models.svr.SVRModel`. The fitted model is kept as plain
    numbers (support vectors and coefficients) and predicted here, so the model
    file needs no pickled scikit-learn object.
    """
    id = "svr"
    label = "SVR (máquina de vetores de suporte)"
    params = (
        Param("tune", "Escolher os ajustes automaticamente", "bool", default=True,
              help="Testa várias combinações de C, epsilon e largura e fica com a de menor erro de validação."),
        Param("kernel", "Tipo de curva (kernel)", "choice", default="rbf", choices=_SVR_KERNELS, advanced=True,
              help="rbf: relações curvas; linear: relação proporcional."),
        Param("C", "C (rigidez do ajuste)", "float", default=100.0, min=1e-6, advanced=True,
              help="Usado quando a escolha automática está desligada."),
        Param("epsilon", "Epsilon (tolerância)", "float", default=0.1, min=0.0, advanced=True,
              help="Usado quando a escolha automática está desligada."),
        Param("gamma_factor", "Largura do kernel (vezes o padrão)", "float", default=1.0, min=1e-9, advanced=True,
              help="1 é o valor padrão do scikit-learn ('scale'). Usado quando a escolha automática está desligada."),
    )

    def candidates(self, tune, kernel, C, epsilon, gamma_factor):
        if not tune:
            return None
        widths = _SVR_GRID["gamma_factor"] if kernel == "rbf" else (1.0,)
        out = []
        for c in _SVR_GRID["C"]:
            for eps in _SVR_GRID["epsilon"]:
                for width in widths:
                    text = f"C={c:g}; epsilon={eps:g}" + (f"; largura={width:g}" if kernel == "rbf" else "")
                    text = text.replace(".", ",")      # decimal comma, as everywhere else on screen
                    out.append((text, {"C": c, "epsilon": eps, "gamma_factor": width}))
        return out

    def fit(self, X, y, tune, kernel, C, epsilon, gamma_factor):
        from sklearn.svm import SVR
        y = np.ravel(y)
        x_mean, x_std = X.mean(axis=0), X.std(axis=0)
        x_std = np.where(x_std == 0, 1.0, x_std)
        y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
        Xn = (X - x_mean) / x_std
        gamma = gamma_factor / (Xn.shape[1] * (Xn.var() or 1.0))       # scikit-learn's "scale", times the factor
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma).fit(Xn, (y - y_mean) / y_std)
        state = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                 "intercept": float(svr.intercept_[0]), "gamma": float(gamma), "linear": int(kernel == "linear")}
        if kernel == "linear":
            state["weights"] = (svr.dual_coef_ @ svr.support_vectors_).ravel()
        else:
            state["support"], state["dual"] = np.asarray(svr.support_vectors_), svr.dual_coef_.ravel()
        return state

    def predict(self, state, X):
        Xn = (X - state["x_mean"]) / state["x_std"]
        if state["linear"]:
            scaled = Xn @ state["weights"] + state["intercept"]
        else:
            support = state["support"]
            distance = (Xn ** 2).sum(axis=1)[:, None] + (support ** 2).sum(axis=1)[None, :] - 2.0 * Xn @ support.T
            scaled = np.exp(-state["gamma"] * np.maximum(distance, 0.0)) @ state["dual"] + state["intercept"]
        return scaled * state["y_std"] + state["y_mean"]
