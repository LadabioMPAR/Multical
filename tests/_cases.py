"""
Reference cases for the numerical core.

`compute_all()` runs the current calculation code on a small synthetic data set
and returns a flat dict {name: ndarray}. `tests/make_golden.py` stores that dict
once; `tests/test_golden.py` recomputes it and compares, so any refactor of the
core (decoupling plots, new API layer, new GUI) can prove it did not change the
numbers.

The data are synthetic on purpose: `data/` is not versioned.
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from multical.core.engine import MulticalEngine  # noqa: E402
from multical.core.saving import train_and_save_model_pls, load_and_predict_pls  # noqa: E402
from multical.models.pcr import pcr_model  # noqa: E402
from multical.models.pls import PLS  # noqa: E402
from multical.models.spa import spa_clean, spa_model  # noqa: E402
from multical.models.svr import SVRModel  # noqa: E402
from multical.preprocessing.pipeline import apply_pretreatment  # noqa: E402
from multical.utils import zscore_matlab_style  # noqa: E402
from multical.var_selection import calculate_rmsecv_fast, calculate_vip, select_k_ftest  # noqa: E402

ANALYTES = ["cb", "gl", "xy"]
KMAX = 8
PRETREATMENT = [["Cut", 4400, 7500, 0], ["SG", 3, 2, 1, 0]]

# One entry per pretreatment available in preprocessing/pipeline.py.
PRETREATMENT_STEPS = {
    "ma_same_grid": ["MA", 3, 1, 0],
    "ma_downsample": ["MA", 3, 2, 0],
    "autoscale": ["AutoScale", 0],
    "meancenter": ["MeanCenter", 0],
    "sg_deriv1": ["SG", 3, 2, 1, 0],
    "sg_smooth": ["SG", 4, 2, 0, 0],
    "loess": ["Loess", 0.2, 2, 0],
    "deriv1": ["Deriv", 1, 0],
    "deriv2": ["Deriv", 2, 0],
    "snv": ["SNV", 0],
    "emsc2": ["EMSC", 2, 0],
    "msc": ["MSC", 0],
    "blctr": ["BLCtr", 7500, 7900, 0.0, 0],
    "cut_one": ["Cut", 4400, 7500, 0],
    "cut_two": ["Cut", 4200, 5200, 6000, 7600, 0],
}

# Validation schemes accepted by MulticalEngine.run_cv (OptimModel argument).
VALIDATIONS = {
    "kfold_venetian": ["kfold", 5, "venetian"],
    "kfold_consecutive": ["kfold", 5, "consecutive"],
    "kfold_random": ["kfold", 5, "random"],
    "loo": -1,
    "holdout": ["Val", 0.25],
}


def make_dataset(seed=0, n_samples=30, n_wl=150):
    """Three overlapping Gaussian bands plus offset, slope, scatter and noise."""
    rs = np.random.RandomState(seed)  # legacy generator: stream is frozen across numpy versions
    wl = np.linspace(4000.0, 8000.0, n_wl)
    Y = rs.uniform(0.5, 10.0, size=(n_samples, 3))
    centers, widths = [4800.0, 5900.0, 7000.0], [250.0, 400.0, 300.0]
    pure = np.stack([np.exp(-0.5 * ((wl - c) / w) ** 2) for c, w in zip(centers, widths)])
    X = 0.05 * (Y @ pure)
    X = X * rs.uniform(0.95, 1.05, size=(n_samples, 1))
    X = X + rs.uniform(0.0, 0.05, size=(n_samples, 1))
    X = X + rs.uniform(0.0, 1e-5, size=(n_samples, 1)) * (wl - wl[0])
    X = X + rs.normal(0.0, 2e-3, size=X.shape)
    return wl, X, Y


def _engine_cases(wl, X, Y, out):
    absor0 = np.vstack([wl, X])
    names = ["RMSECV", "RMSECV_conc", "RMSEcal", "RMSEcal_conc", "RMSEtest", "RMSEtest_conc", "R2CV", "R2cal"]
    for label, optim in VALIDATIONS.items():
        np.random.seed(123)  # 'random' and 'Val' draw from the global generator
        res = MulticalEngine().run(
            1, 2, 0, KMAX, len(ANALYTES), list(ANALYTES), "g/L", Y.copy(), absor0.copy(),
            0.0, [], optim, PRETREATMENT, analysis_list=None, output_dir=None, use_ftest=True,
        )
        plt.close("all")
        for name, value in zip(names, res[:8]):
            out[f"engine_{label}_{name}"] = np.asarray(value)
        best_k = res[8]
        out[f"engine_{label}_best_k"] = np.array([best_k[j] for j in range(len(ANALYTES))], dtype=np.int64)


def _pretreatment_cases(wl, X, out):
    for label, step in PRETREATMENT_STEPS.items():
        Xp, wlp = apply_pretreatment([step], X[:8].copy(), wl.copy(), plot=False)
        out[f"pre_{label}_X"] = np.asarray(Xp)
        out[f"pre_{label}_wl"] = np.asarray(wlp)


def _model_cases(Xp, wlp, Y, out):
    Xtr, Xte, Ytr = Xp[:22], Xp[22:], Y[:22]

    for switch in (0, 1):
        Yp, Ytp, _ = PLS().predict_model(Xtr, Ytr, 4, Xt=Xte, teste_switch=switch)
        out[f"pls_switch{switch}_Yp"], out[f"pls_switch{switch}_Ytp"] = np.asarray(Yp), np.asarray(Ytp)

        Yp, Ytp, _ = pcr_model(Xtr, Ytr, 4, Xt=Xte, teste_switch=switch)
        out[f"pcr_switch{switch}_Yp"], out[f"pcr_switch{switch}_Ytp"] = np.asarray(Yp), np.asarray(Ytp)

        Yp, Ytp, info = spa_model(Xtr, Ytr, 4, 0, Xt=Xte, teste_switch=switch)
        out[f"spa_switch{switch}_Yp"], out[f"spa_switch{switch}_Ytp"] = np.asarray(Yp), np.asarray(Ytp)
        out[f"spa_switch{switch}_selected"] = np.asarray(info["SelectedVars"], dtype=np.int64)

    out["spa_clean_selected"] = np.asarray(spa_clean(Xp, 0, 6), dtype=np.int64)

    Xn, _, _ = zscore_matlab_style(Xp)
    Yn, _, _ = zscore_matlab_style(Y)
    pls = PLS()
    _, T, P, _, _, _, _, _ = pls.nipals(Xn, Yn, 3)
    T2, t2_limit, Q, q_limit = pls.compute_outlier_metrics(Xn, T, P, conf_level=0.95)
    out["outlier_T2"], out["outlier_Q"] = np.asarray(T2), np.asarray(Q)
    out["outlier_limits"] = np.array([t2_limit, q_limit], dtype=float)

    out["vip_scores"] = np.asarray(calculate_vip(PLS(), Xp, Y, 4))
    plt.close("all")

    for label, target in (("all", None), ("gl", 1)):
        rmin, kmin, curve = calculate_rmsecv_fast(Xp, Y, KMAX, folds=5, cv_type="venetian", target_idx=target)
        out[f"rmsecv_fast_{label}_curve"] = np.asarray(curve)
        out[f"rmsecv_fast_{label}_min"] = np.array([rmin, kmin], dtype=float)
        out[f"rmsecv_fast_{label}_ftest_k"] = np.array([select_k_ftest(curve, Xp.shape[0])], dtype=np.int64)

    # SVR depends on libsvm through scikit-learn; compared with a looser tolerance.
    out["svr_rbf_Ytp"] = np.asarray(
        SVRModel().fit_predict(Xtr, Ytr, Xte, C=10.0, epsilon=0.01, gamma="scale", kernel="rbf")
    )


def _saved_model_case(Xp, wlp, Y, out):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        train_and_save_model_pls(Xp[:22], Y[:22], wlp, [3, 4, 2], path, rmsecv_list=[0.1, 0.2, 0.3])
        y_pred, rmsecv = load_and_predict_pls(Xp[22:], wlp, path)
    out["saved_pls_y_pred"] = np.asarray(y_pred)
    out["saved_pls_rmsecv"] = np.asarray(rmsecv, dtype=float)


def compute_all():
    """Run every reference case and return {name: ndarray}."""
    wl, X, Y = make_dataset()
    out = {"data_wl": wl, "data_X": X, "data_Y": Y}
    with contextlib.redirect_stdout(io.StringIO()):
        _pretreatment_cases(wl, X, out)
        _engine_cases(wl, X, Y, out)
        Xp, wlp = apply_pretreatment(PRETREATMENT, X.copy(), wl.copy(), plot=False)
        _model_cases(Xp, wlp, Y, out)
        _saved_model_case(Xp, wlp, Y, out)
    plt.close("all")
    return out
