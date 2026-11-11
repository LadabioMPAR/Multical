"""
Model comparison under nested Leave-One-Experiment-Out (LOExO), target = glucose,
robust pipeline (Cut + SG + SNV from run_calibration_svr.py).

Models (all leak-free; any tuning/selection done on TRAIN experiments only):
  - mean         : no-skill baseline (predicts the training mean) -> R2 = 0 reference
  - Ridge        : linear, L2-regularised (alpha by internal CV)
  - ElasticNet   : sparse linear, L1/L2 (alpha & l1_ratio by internal CV)
  - PLS          : partial least squares (k by inner CV + Osten F-test)
  - PCR          : principal component regression (k by inner venetian CV)
  - SPA (MLR)    : successive projections variable selection + MLR (k, cini by inner CV)
  - kNN          : k-nearest-neighbours (n_neighbors by inner CV)
  - Random Forest: tree ensemble (nonlinear)
  - SVR(rbf)     : support vector regression (small rbf grid by inner CV)

Goal: answer "with this much data, what can we do?" — show where each model lands
relative to the no-skill line. (PyTorch ANN is intentionally excluded: with only
6 batches and 3101 features it would massively overfit; see notes printed at end.)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.svr import SVRModel
from src.multical.models.spa import spa_model, spa_clean
from src.multical.var_selection import calculate_rmsecv_fast, select_k_ftest
import scripts.calibration.run_calibration_svr as R
from scripts.calibration.run_loexo_pls import pls_predict_holdout

RESULTS_DIR = "results_loexo_models"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']
GL = 1
MAX_LV = 15
INNER_FOLDS = 5

SVR_INNER_GRID = {'C': [1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.05],
                  'gamma': ['scale', 'auto', 1e-4, 1e-3], 'degree': [2], 'coef0': [1.0]}


def load_pair(refe, nonda):
    xi = np.atleast_2d(np.loadtxt(refe))
    if xi.shape[1] > 1:
        xi = xi[:, 1:]
    with open(nonda) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    absi = np.atleast_2d(np.loadtxt(nonda, skiprows=1))
    return xi, absi[:, 1:], wl


def load_and_pretreat():
    exps = {}
    for e in EXP_IDS:
        x_list, a_list, wl = [], [], None
        for split in ('cal', 'test'):
            refe = os.path.join(SPLITS, f"{e}_refe_{split}.txt")
            nonda = os.path.join(SPLITS, f"{e}_nonda_{split}.txt")
            if os.path.exists(refe) and os.path.exists(nonda):
                xi, ai, wl = load_pair(refe, nonda); x_list.append(xi); a_list.append(ai)
        a_pre, _ = apply_pretreatment(R.PRETREATMENT, np.vstack(a_list), wl, plot=False)
        exps[e] = (np.vstack(x_list)[:, GL], a_pre)   # store glucose vector + spectra
    return exps


# ---------------- per-model leak-free predictors (train -> held-out) -------------
def predict_mean(A_tr, y_tr, A_te):
    return np.full(A_te.shape[0], y_tr.mean())


def predict_ridge(A_tr, y_tr, A_te):
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 7, 25)))
    mdl.fit(A_tr, y_tr)
    return mdl.predict(A_te)


# Efficient PCR: economy-SVD computed ONCE per training subset; all k come for free.
# (Mathematically equivalent to pcr_model: zscore + PCA + least-squares on scores.)
def _pcr_fit(X_tr):
    mu = X_tr.mean(0)
    sig = X_tr.std(0, ddof=1); sig[sig == 0] = 1.0
    Xn = (X_tr - mu) / sig
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    return mu, sig, U * S, Vt              # scores T = U*S, loadings Vt


def _pcr_predict(fit, y_tr, X_te, k):
    mu, sig, T, Vt = fit
    ymu = y_tr.mean(); ysig = y_tr.std(ddof=1) or 1.0
    beta = np.linalg.lstsq(T[:, :k], (y_tr - ymu) / ysig, rcond=None)[0]
    Tte = ((X_te - mu) / sig) @ Vt[:k].T
    return (Tte @ beta) * ysig + ymu


def predict_enet(A_tr, y_tr, A_te):
    # Sparse linear (L1/L2); alpha & l1_ratio by internal CV (train only).
    mdl = make_pipeline(StandardScaler(),
                        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0],
                                     cv=5, max_iter=10000, random_state=0))
    mdl.fit(A_tr, y_tr)
    return mdl.predict(A_te)


def predict_knn(A_tr, y_tr, A_te):
    # Instance-based; n_neighbors by inner CV (train only).
    base = make_pipeline(StandardScaler(), KNeighborsRegressor())
    gs = GridSearchCV(base, {'kneighborsregressor__n_neighbors': [1, 3, 5, 7]}, cv=5)
    gs.fit(A_tr, y_tr)
    return gs.predict(A_te)


def predict_rf(A_tr, y_tr, A_te):
    # Tree ensemble (nonlinear); no scaling needed for trees.
    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                               random_state=0, n_jobs=-1)
    rf.fit(A_tr, y_tr)
    return rf.predict(A_te)


def _inner_k_pcr(A, y, kmax):
    n = len(y); kmax = min(kmax, n - 2)
    sse = np.zeros(kmax)
    for i in range(INNER_FOLDS):
        val = np.arange(i, n, INNER_FOLDS)
        if len(val) == 0:
            continue
        tr = np.setdiff1d(np.arange(n), val)
        fit = _pcr_fit(A[tr])
        for k in range(1, kmax + 1):
            yp = _pcr_predict(fit, y[tr], A[val], k)
            sse[k - 1] += np.sum((y[val] - yp) ** 2)
    return int(np.argmin(sse)) + 1


def predict_pcr(A_tr, y_tr, A_te):
    k = _inner_k_pcr(A_tr, y_tr, MAX_LV)
    return _pcr_predict(_pcr_fit(A_tr), y_tr, A_te, k)


def predict_pls(A_tr, y_tr, A_te):
    _, _, vec = calculate_rmsecv_fast(A_tr, y_tr.reshape(-1, 1), MAX_LV, INNER_FOLDS, 'venetian', target_idx=0)
    k = select_k_ftest(vec, n_cal=A_tr.shape[0])
    return pls_predict_holdout(A_tr, y_tr, A_te, k)


# SPA = Successive Projections Algorithm (variable selection) + MLR.
# Selects 'k' wavelengths starting from 'cini'; k and cini chosen by inner CV (train only).
def _spa_cv_rmse(A, y, cini, kmax):
    n = len(y); kmax = min(kmax, n - 2)
    sse = np.zeros(kmax)
    for i in range(INNER_FOLDS):
        val = np.arange(i, n, INNER_FOLDS)
        if len(val) == 0:
            continue
        tr = np.setdiff1d(np.arange(n), val)
        order = spa_clean(A[tr], int(cini), kmax)               # selection on train only
        mu = A[tr].mean(0); sig = A[tr].std(0, ddof=1); sig[sig == 0] = 1.0
        Xn = (A[tr] - mu) / sig; Xv = (A[val] - mu) / sig
        ymu = y[tr].mean(); ysig = y[tr].std(ddof=1) or 1.0
        yn = (y[tr] - ymu) / ysig
        for k in range(1, kmax + 1):
            cols = order[:k]
            beta = np.linalg.lstsq(Xn[:, cols], yn, rcond=None)[0]
            yp = (Xv[:, cols] @ beta) * ysig + ymu
            sse[k - 1] += np.sum((y[val] - yp) ** 2)
    return np.sqrt(sse / n)


def predict_spa(A_tr, y_tr, A_te):
    p = A_tr.shape[1]
    cini_cands = np.unique(np.linspace(0, p - 1, 8).astype(int))
    best = (np.inf, 0, 1)                                       # (rmse, cini, k)
    for cini in cini_cands:
        rmse = _spa_cv_rmse(A_tr, y_tr, cini, MAX_LV)
        k = int(np.argmin(rmse)) + 1
        if rmse[k - 1] < best[0]:
            best = (rmse[k - 1], int(cini), k)
    _, cini, k = best
    _, ytp, _ = spa_model(A_tr, y_tr, k, cini, Xt=A_te, teste_switch=0)
    return ytp.flatten()


def predict_svr(A_tr, y_tr, A_te):
    combos = R.build_param_combos(SVR_INNER_GRID, ['rbf'])
    bp_list, _, _ = R.grid_search_svr(A_tr, y_tr.reshape(-1, 1), combos, INNER_FOLDS, 'venetian', ['gl'], perm=None)
    bp = bp_list[0]
    m = SVRModel()
    return m.fit_predict(A_tr, y_tr.reshape(-1, 1), A_te,
                         C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
                         kernel='rbf').flatten()


MODELS = {
    'mean (no-skill)': predict_mean,
    'Ridge':           predict_ridge,
    'ElasticNet':      predict_enet,
    'PLS':             predict_pls,
    'PCR':             predict_pcr,
    'SPA (MLR)':       predict_spa,
    'kNN':             predict_knn,
    'Random Forest':   predict_rf,
    'SVR (rbf)':       predict_svr,
}


def loexo(exps, predict_fn):
    all_ref, all_pred = [], []
    for held in EXP_IDS:
        train_e = [e for e in EXP_IDS if e != held]
        A_tr = np.vstack([exps[e][1] for e in train_e])
        y_tr = np.concatenate([exps[e][0] for e in train_e])
        A_te = exps[held][1]
        y_te = exps[held][0]
        y_pred = predict_fn(A_tr, y_tr, A_te)
        all_ref.extend(y_te.tolist()); all_pred.extend(y_pred.tolist())
    all_ref = np.array(all_ref); all_pred = np.array(all_pred)
    rmsep = float(np.sqrt(np.mean((all_ref - all_pred) ** 2)))
    sst = float(np.sum((all_ref - all_ref.mean()) ** 2)) or 1.0
    r2 = 1 - float(np.sum((all_ref - all_pred) ** 2)) / sst
    bias = float(np.mean(all_pred - all_ref))
    return rmsep, r2, bias


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exps = load_and_pretreat()
    print("Samples per experiment:", {e: exps[e][0].shape[0] for e in EXP_IDS})

    res = {}
    for name, fn in MODELS.items():
        rmsep, r2, bias = loexo(exps, fn)
        res[name] = (rmsep, r2, bias)
        print(f"  {name:<16}: RMSEP={rmsep:.3f}  R2={r2:+.3f}  bias={bias:+.3f}")

    print(f"\n{'='*60}")
    print("LOExO MODEL COMPARISON - glucose (Cut+SG+SNV)".center(60))
    print(f"{'='*60}")
    h = f"{'Model':<16}|{'RMSEP':^9}|{'R2_pred':^9}|{'bias':^8}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for name, (rmsep, r2, bias) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<16}|{rmsep:^9.3f}|{r2:^+9.3f}|{bias:^+8.2f}")
    print("-" * len(h))
    print("(units g/L; R2<=0 means not better than predicting the mean)")
    print("\nNOTE: PyTorch ANN excluded on purpose - 6 batches / 3101 features would")
    print("overfit badly. With little data, simpler/regularised models are the right")
    print("choice; complexity does not overcome the lack of batch diversity.")

    names = list(res.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [res[n][0] for n in names]
    ax.bar(range(len(names)), vals, color='slateblue')
    ax.axhline(res['mean (no-skill)'][0], color='red', ls='--', lw=1, label='no-skill (mean)')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel(f"LOExO RMSEP ({R.UNITS})")
    ax.set_title("Glucose: cross-batch RMSEP by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "LOExO_model_comparison.png"), dpi=300)
    print(f"\nSaved chart in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
