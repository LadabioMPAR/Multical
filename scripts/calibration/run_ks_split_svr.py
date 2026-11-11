"""
Kennard-Stone (or SPXY) cal/test split made HERE, then SVR calibration on it.

Self-contained KS implementation (no external dependency). Pipeline:
  1. Load + pretreat all samples (robust default Cut + SG + SNV).
  2. Split into calibration/test by Kennard-Stone or SPXY (pooled or per-experiment).
  3. Train an SVR (rbf) per analyte on the calibration set (hyperparameters tuned by
     inner CV on calibration only), predict the test set, report RMSEP / R2.

IMPORTANT: a KS/SPXY test is WITHIN-BATCH and optimistic (cal & test of the same
experiment / interpolation points). It does NOT measure prediction of a NEW batch.
Always cross-check with run_loexo_model_comparison.py (LOExO = honest cross-batch).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.svr import SVRModel
import scripts.calibration.run_calibration_svr as R
import scripts.calibration.run_loexo_model_comparison as M

# =============================================================================
#                              CONFIGURATION
# =============================================================================
RESULTS_DIR = "results_ks_split_svr"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']

SPLIT_METHOD = 'KS'           # 'KS' (X only) or 'SPXY' (joint X-Y)
SPLIT_LEVEL = 'per_experiment'  # 'pooled' or 'per_experiment'
CAL_FRACTION = 0.75           # fraction of samples kept for calibration
SPXY_TARGET = 'gl'            # analyte used for the Y-distance term in SPXY
AUTOSCALE_FOR_DISTANCE = True  # standardize columns before distance (recommended)
RANDOM_SEED = 0               # only affects nothing in KS (deterministic); kept for clarity


# =============================================================================
#                              KS / SPXY core
# =============================================================================
def _select_ks(X, n_cal, y=None, method='KS'):
    """Return (cal_idx, test_idx) by Kennard-Stone (or SPXY) on rows of X."""
    n = X.shape[0]
    n_cal = int(np.clip(n_cal, 2, n - 1))

    Dx = cdist(X, X, metric='euclidean')
    if method == 'SPXY':
        if y is None:
            raise ValueError("SPXY requires y.")
        Dy = np.abs(y.reshape(-1, 1) - y.reshape(1, -1))
        D = Dx / (Dx.max() or 1.0) + Dy / (Dy.max() or 1.0)
    else:
        D = Dx

    # Start from the two most distant samples
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    min_d = np.minimum(D[i], D[j])      # each sample's distance to the selected set
    min_d[selected] = -np.inf

    while len(selected) < n_cal:
        pick = int(np.argmax(min_d))
        selected.append(pick)
        min_d = np.minimum(min_d, D[pick])
        min_d[selected] = -np.inf

    cal = np.array(sorted(selected))
    test = np.array(sorted(set(range(n)) - set(cal)))
    return cal, test


def make_split(A, Y, exp, method, level, frac):
    """Build global cal/test index arrays."""
    jt = R.ANALYTES.index(SPXY_TARGET)
    if AUTOSCALE_FOR_DISTANCE:
        Xd = (A - A.mean(0)) / np.where(A.std(0, ddof=0) == 0, 1.0, A.std(0, ddof=0))
    else:
        Xd = A

    if level == 'pooled':
        n_cal = round(frac * A.shape[0])
        cal, test = _select_ks(Xd, n_cal, Y[:, jt], method)
        return cal, test

    # per-experiment
    cal_all, test_all = [], []
    for e in np.unique(exp):
        idx = np.where(exp == e)[0]
        n_cal = max(2, round(frac * len(idx)))
        c, t = _select_ks(Xd[idx], n_cal, Y[idx, jt], method)
        cal_all.extend(idx[c]); test_all.extend(idx[t])
    return np.array(sorted(cal_all)), np.array(sorted(test_all))


# =============================================================================
#                              Data
# =============================================================================
def load_all():
    a_list, y_list, exp_list, wl = [], [], [], None
    for e in EXP_IDS:
        for split in ('cal', 'test'):
            refe = os.path.join(SPLITS, f"{e}_refe_{split}.txt")
            nonda = os.path.join(SPLITS, f"{e}_nonda_{split}.txt")
            if os.path.exists(refe) and os.path.exists(nonda):
                xi, ai, wl = M.load_pair(refe, nonda)
                a_list.append(ai); y_list.append(xi); exp_list.extend([e] * xi.shape[0])
    A_raw = np.vstack(a_list); Y = np.vstack(y_list); exp = np.array(exp_list)
    A_pre, _ = apply_pretreatment(R.PRETREATMENT, A_raw, wl, plot=False)
    return A_pre, Y, exp


def _metrics(yr, yp):
    rmse = float(np.sqrt(np.mean((yr - yp) ** 2)))
    sst = float(np.sum((yr - yr.mean()) ** 2)) or 1.0
    return rmse, 1 - float(np.sum((yr - yp) ** 2)) / sst


# =============================================================================
#                              Main
# =============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    A, Y, exp = load_all()
    print(f"Loaded {A.shape[0]} samples ({A.shape[1]} features) over {len(set(exp))} experiments.")

    cal, test = make_split(A, Y, exp, SPLIT_METHOD, SPLIT_LEVEL, CAL_FRACTION)
    print(f"\nSplit: {SPLIT_METHOD} / {SPLIT_LEVEL} / cal_frac={CAL_FRACTION}")
    print(f"  calibration: {len(cal)} samples | test: {len(test)} samples")
    from collections import Counter
    print(f"  test per experiment: {dict(Counter(exp[test]))}")

    A_cal, Y_cal = A[cal], Y[cal]
    A_test, Y_test = A[test], Y[test]

    # SVR (rbf) per analyte, tuned on calibration only
    combos = R.build_param_combos(R.SVR_GRID, R.SVR_KERNELS)
    bp_list, _, _ = R.grid_search_svr(A_cal, Y_cal, combos, R.K_FOLDS, R.CV_TYPE, R.ANALYTES, perm=None)

    print(f"\n{'='*62}")
    print(f"SVR on {SPLIT_METHOD} split ({SPLIT_LEVEL})".center(62))
    print(f"{'='*62}")
    head = f"{'Analyte':<8}|{'RMSEP':^9}|{'R2_pred':^9}| best SVR params"
    print("-" * 62); print(head); print("-" * 62)
    preds = {}
    for j, name in enumerate(R.ANALYTES):
        bp = bp_list[j]
        m = SVRModel()
        yp = m.fit_predict(A_cal, Y_cal[:, j].reshape(-1, 1), A_test,
                           C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
                           kernel=bp.get('kernel', 'rbf')).flatten()
        preds[name] = yp
        rmse, r2 = _metrics(Y_test[:, j], yp)
        print(f"{name:<8}|{rmse:^9.3f}|{r2:^+9.3f}| {R._params_label(bp, skip=())}")
    print("-" * 62)
    print("\n[!] KS/SPXY test is WITHIN-BATCH (optimistic). For new-batch performance")
    print("    use LOExO (run_loexo_model_comparison.py). Expect LOExO to be much worse.")

    # Save split indices
    import csv
    with open(os.path.join(RESULTS_DIR, "ks_split.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["global_index", "experiment", "set"])
        for i in range(A.shape[0]):
            w.writerow([i, exp[i], "cal" if i in set(cal) else "test"])

    # Predicted vs reference plot
    fig, axes = plt.subplots(1, len(R.ANALYTES), figsize=(6 * len(R.ANALYTES), 5), squeeze=False)
    for ax, (j, name) in zip(axes.flatten(), enumerate(R.ANALYTES)):
        yr, yp = Y_test[:, j], preds[name]
        rmse, r2 = _metrics(yr, yp)
        col = R.COLORS[j] if j < len(R.COLORS) else 'blue'
        ax.scatter(yr, yp, c=col, alpha=0.8)
        lo, hi = yr.min(), yr.max(); b = (hi - lo) * 0.08 or 1
        ax.plot([lo - b, hi + b], [lo - b, hi + b], 'k--', alpha=0.5)
        ax.text(0.05, 0.95, f"RMSEP={rmse:.2f}\n$R^2$={r2:.2f}", transform=ax.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(f"{name} - {SPLIT_METHOD} test", loc='left')
        ax.set_xlabel(f"Reference {R.UNITS}"); ax.set_ylabel(f"Predicted {R.UNITS}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"SVR_{SPLIT_METHOD}_{SPLIT_LEVEL}.png"), dpi=300)
    print(f"\nSaved split + plot in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
