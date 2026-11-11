"""
Leave-One-Experiment-Out over exp4..exp7 (full data files), PLS-based.

For each held-out experiment (train on the other 3, test on the held-out one) and for
each method below, fit per-analyte PLS and report metrics for glucose (gl) and xylose (xy):
  - PLS    : plain PLS, all wavelengths (run_calibration style)
  - VIP    : VIP variable selection + PLS (run_variable_selection style)
  - PSO    : PSO variable selection + PLS (run_variable_selection style)

Reported per method/analyte: best held-out batch (by R2 test), #wavenumbers retained,
R2_cal, R2_cv, R2_test, RMSECV, RMSEP, and #latent variables (LVs).

Preprocessing = default of those scripts: Cut(4400-7500) + Savitzky-Golay(7,2,1).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from src.multical.var_selection import calculate_vip, select_k_ftest, run_pso

# =============================================================================
#                              CONFIGURATION
# =============================================================================
DATA_DIR = "data"
EXPERIMENTS = ['exp4', 'exp5', 'exp6', 'exp7']
ANALYTE_NAMES = ['cb', 'gl', 'xy']
TARGETS = {'gl': 1, 'xy': 2}            # analyte name -> column index
UNITS = 'g/L'

PRETREAT = [['Cut', 4400, 7500, 0], ['SG', 7, 2, 1, 0]]
MAX_LV = 15
K_FOLDS = 5
CV_TYPE = 'venetian'

VIP_THRESHOLDS = np.arange(0.2, 1.6, 0.1)
PSO_PARAMS = dict(n_particles=20, max_iter=20, w=0.9, c1=1.49, c2=1.49)


# =============================================================================
#                              Data / PLS helpers
# =============================================================================
def load_full(e):
    xi = np.atleast_2d(np.loadtxt(os.path.join(DATA_DIR, f"{e}_refe.txt")))[:, 1:]
    with open(os.path.join(DATA_DIR, f"{e}_nonda.txt")) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    ai = np.atleast_2d(np.loadtxt(os.path.join(DATA_DIR, f"{e}_nonda.txt"), skiprows=1))[:, 1:]
    a, wl_pre = apply_pretreatment(PRETREAT, ai, wl, plot=False)
    return xi, a, wl_pre


def _beta(P, Q, W, k):
    wk, pk, qk = W[:, :k], P[:, :k], Q[:, :k]
    return wk @ np.linalg.pinv(pk.T @ wk) @ qk.T


def _cv_pred(A, y, kmax, folds=K_FOLDS):
    """Venetian k-fold CV predictions, shape (n, kmax). Combined-X norm, train-y norm."""
    n = len(y)
    kmax = int(min(kmax, A.shape[1], n - 2))
    ycv = np.zeros((n, max(kmax, 1)))
    eng = PLS()
    for i in range(folds):
        val = np.arange(i, n, folds)
        if len(val) == 0:
            continue
        tr = np.setdiff1d(np.arange(n), val)
        Xcn, _, _ = zscore_matlab_style(np.vstack([A[tr], A[val]]))
        ntr = len(tr)
        Xtn, Xvn = Xcn[:ntr], Xcn[ntr:]
        yn, ymu, ysig = zscore_matlab_style(y[tr].reshape(-1, 1))
        _, _, P, _, Q, W, _, _ = eng.nipals(Xtn, yn, kmax)
        for k in range(1, kmax + 1):
            yv = (Xvn @ _beta(P, Q, W, k)) * ysig + ymu
            ycv[val, k - 1] = yv.flatten()
    return ycv, kmax


def _r2(y, yp):
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return 1 - float(np.sum((y - yp) ** 2)) / sst


def pls_fold_metrics(A_tr, y_tr, A_te, y_te):
    """Full PLS metrics on a given (already variable-masked) train/test split."""
    ycv, kmax = _cv_pred(A_tr, y_tr, MAX_LV)
    rmsecv_k = np.sqrt(np.mean((y_tr[:, None] - ycv) ** 2, axis=0))
    k = select_k_ftest(rmsecv_k, n_cal=len(y_tr))
    k = int(min(k, kmax))

    rmsecv = float(rmsecv_k[k - 1])
    r2cv = _r2(y_tr, ycv[:, k - 1])

    # Calibration fit (full train) + test prediction at k
    Xn, Xmu, Xsig = zscore_matlab_style(A_tr)
    yn, ymu, ysig = zscore_matlab_style(y_tr.reshape(-1, 1))
    _, _, P, _, Q, W, _, _ = PLS().nipals(Xn, yn, k)
    beta = _beta(P, Q, W, k)
    ycal = (Xn @ beta) * ysig + ymu
    r2cal = _r2(y_tr, ycal.flatten())

    Xsig_safe = Xsig.copy(); Xsig_safe[Xsig_safe == 0] = 1.0
    ytp = (((A_te - Xmu) / Xsig_safe) @ beta) * ysig + ymu
    rmsep = float(np.sqrt(np.mean((y_te - ytp.flatten()) ** 2)))
    r2test = _r2(y_te, ytp.flatten())

    return dict(k=k, n_vars=A_tr.shape[1], rmsecv=rmsecv, rmsep=rmsep,
                r2cal=r2cal, r2cv=r2cv, r2test=r2test)


# =============================================================================
#                              Variable selection
# =============================================================================
def vip_mask(A_tr, Y_tr, jt):
    ycv, kmax = _cv_pred(A_tr, Y_tr[:, jt], MAX_LV)
    rmsecv_k = np.sqrt(np.mean((Y_tr[:, jt][:, None] - ycv) ** 2, axis=0))
    k_vip = int(min(select_k_ftest(rmsecv_k, n_cal=A_tr.shape[0]), kmax))
    vip = calculate_vip(PLS(), A_tr, Y_tr[:, jt].reshape(-1, 1), k_vip)

    best = (np.inf, np.ones(A_tr.shape[1], dtype=bool))
    for th in VIP_THRESHOLDS:
        mask = vip >= th
        if mask.sum() < 2:
            continue
        ycv2, _ = _cv_pred(A_tr[:, mask], Y_tr[:, jt], MAX_LV)
        rmin = float(np.min(np.sqrt(np.mean((Y_tr[:, jt][:, None] - ycv2) ** 2, axis=0))))
        if rmin < best[0]:
            best = (rmin, mask)
    return best[1]


def pso_mask(A_tr, Y_tr, jt):
    mask, _ = run_pso(A_tr, Y_tr, MAX_LV, K_FOLDS, CV_TYPE, target_idx=jt, **PSO_PARAMS)
    return mask


METHODS = ['PLS', 'VIP', 'PSO']


# =============================================================================
#                              Main
# =============================================================================
def main():
    data = {e: load_full(e) for e in EXPERIMENTS}
    n_feat = data[EXPERIMENTS[0]][1].shape[1]
    print("Loaded:", {e: data[e][0].shape[0] for e in EXPERIMENTS}, f"| {n_feat} wavenumbers after pretreatment\n")

    # results[method][analyte][held_exp] = metrics dict
    results = {m: {a: {} for a in TARGETS} for m in METHODS}

    for held in EXPERIMENTS:
        train_e = [e for e in EXPERIMENTS if e != held]
        A_tr = np.vstack([data[e][1] for e in train_e])
        Y_tr = np.vstack([data[e][0] for e in train_e])
        A_te = data[held][1]
        Y_te = data[held][0]
        print(f"### Held-out = {held} (train on {train_e}, {A_tr.shape[0]} samples)")

        # Precompute VIP/PSO masks per analyte
        masks = {a: {'PLS': np.ones(n_feat, dtype=bool)} for a in TARGETS}
        for a, jt in TARGETS.items():
            print(f"    [{a}] VIP selection...")
            masks[a]['VIP'] = vip_mask(A_tr, Y_tr, jt)
            print(f"    [{a}] PSO selection...")
            masks[a]['PSO'] = pso_mask(A_tr, Y_tr, jt)

        for a, jt in TARGETS.items():
            for m in METHODS:
                mask = masks[a][m]
                met = pls_fold_metrics(A_tr[:, mask], Y_tr[:, jt], A_te[:, mask], Y_te[:, jt])
                results[m][a][held] = met
                print(f"      {m:<4} {a}: vars={met['n_vars']:5d} LV={met['k']:2d} "
                      f"R2cal={met['r2cal']:+.3f} R2cv={met['r2cv']:+.3f} R2test={met['r2test']:+.3f} "
                      f"RMSECV={met['rmsecv']:.3f} RMSEP={met['rmsep']:.3f}")

    # ---------------- Report ----------------
    print(f"\n\n{'#'*78}\n{'FINAL REPORT - LOExO exp4..exp7'.center(78)}\n{'#'*78}")
    for a in TARGETS:
        print(f"\n================  ANALYTE: {a.upper()}  ================")
        for m in METHODS:
            print(f"\n--- Method: {m} ---")
            hdr = f"{'held-out':<9}|{'#wave':^7}|{'LVs':^5}|{'R2cal':^8}|{'R2cv':^8}|{'R2test':^8}|{'RMSECV':^8}|{'RMSEP':^8}"
            print("-" * len(hdr)); print(hdr); print("-" * len(hdr))
            best_e, best_r2 = None, -np.inf
            for held in EXPERIMENTS:
                r = results[m][a][held]
                print(f"{held:<9}|{r['n_vars']:^7}|{r['k']:^5}|{r['r2cal']:^+8.3f}|{r['r2cv']:^+8.3f}|"
                      f"{r['r2test']:^+8.3f}|{r['rmsecv']:^8.3f}|{r['rmsep']:^8.3f}")
                if r['r2test'] > best_r2:
                    best_r2, best_e = r['r2test'], held
            print("-" * len(hdr))
            r = results[m][a][best_e]
            print(f">> BEST batch (by R2_test): {best_e}  |  #wavenumbers={r['n_vars']}  |  LVs={r['k']}")
            print(f"   R2_cal={r['r2cal']:.3f}  R2_cv={r['r2cv']:.3f}  R2_test={r['r2test']:.3f}  "
                  f"RMSECV={r['rmsecv']:.3f}  RMSEP={r['rmsep']:.3f} {UNITS}")


if __name__ == "__main__":
    main()
