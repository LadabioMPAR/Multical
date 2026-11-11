"""
Nested Leave-One-Experiment-Out (LOExO) validation of the SVR model for 'gl'.

Why nested: each experiment is a separate batch (time points of one run). Holding
out a whole experiment tests the realistic scenario "predict a brand-new batch".
To avoid ANY leakage, inside every fold the variable selection (VIP) and the SVR
hyper-parameter tuning are redone using ONLY the training experiments - the held-
out experiment touches nothing (not the fit, not the selection, not the tuning).

This also recovers exp9 (its 1-sample test file, dropped by the standard loader).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.models.svr import SVRModel
from src.multical.var_selection import calculate_vip, calculate_rmsecv_fast, select_k_ftest
import scripts.calibration.run_calibration_svr as R

# =============================================================================
#                              CONFIGURATION
# =============================================================================
TARGET = 'gl'
RESULTS_DIR = "results_loexo_svr"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']

MAX_LATENT_VARS = 15            # PLS LVs for VIP ranking
INNER_FOLDS = 5
INNER_CV = 'venetian'
INNER_VIP_THRESHOLDS = [0.0, 0.5, 0.7, 0.9, 1.1]   # 0.0 = keep all variables
INNER_KERNELS = ['poly']
INNER_GRID = {
    'C':       [3, 6, 10],
    'epsilon': [0.001, 0.01, 0.05],
    'gamma':   ['scale'],
    'degree':  [2],
    'coef0':   [1.0],
}


# =============================================================================
#                              HELPERS
# =============================================================================
def load_pair(refe, nonda):
    """Robust loader: handles single-sample files (atleast_2d)."""
    xi = np.atleast_2d(np.loadtxt(refe))
    if xi.shape[1] > 1:
        xi = xi[:, 1:]                       # drop time column
    with open(nonda) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    absi = np.atleast_2d(np.loadtxt(nonda, skiprows=1))
    return xi, absi[:, 1:], wl               # drop time column from spectra


def load_experiment(exp):
    """Load cal+test of one experiment, pretreat (row-wise -> leak-free)."""
    x_list, a_list, wl = [], [], None
    for split in ('cal', 'test'):
        refe = os.path.join(SPLITS, f"{exp}_refe_{split}.txt")
        nonda = os.path.join(SPLITS, f"{exp}_nonda_{split}.txt")
        if not (os.path.exists(refe) and os.path.exists(nonda)):
            continue
        xi, ai, wl = load_pair(refe, nonda)
        x_list.append(xi); a_list.append(ai)
    x = np.vstack(x_list)
    a_raw = np.vstack(a_list)
    a_pre, wl_pre = apply_pretreatment(R.PRETREATMENT, a_raw, wl, plot=False)
    return x, a_pre, wl_pre


def select_and_tune(A_tr, Y_tr_all, jt):
    """Nested VIP selection + SVR tuning on training data only. Returns (mask, params)."""
    y = Y_tr_all[:, jt]
    # optimal PLS k for VIP
    _, _, rmsecv_vec = calculate_rmsecv_fast(A_tr, Y_tr_all, MAX_LATENT_VARS, INNER_FOLDS, INNER_CV, target_idx=jt)
    k_opt = select_k_ftest(rmsecv_vec, n_cal=A_tr.shape[0])
    vip = calculate_vip(PLS(), A_tr, y.reshape(-1, 1), k_opt)   # no output_dir -> no plot

    combos = R.build_param_combos(INNER_GRID, INNER_KERNELS)
    best = (np.inf, None, None)              # (rmsecv, mask, params)
    for th in INNER_VIP_THRESHOLDS:
        mask = vip >= th
        if mask.sum() < 5:
            continue
        bp_list, rmse_list, _ = R.grid_search_svr(
            A_tr[:, mask], Y_tr_all[:, [jt]], combos, INNER_FOLDS, INNER_CV, [TARGET], perm=None
        )
        if rmse_list[0] < best[0]:
            best = (rmse_list[0], mask, bp_list[0])
    return best[1], best[2], best[0]


# =============================================================================
#                                 MAIN
# =============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jt = R.ANALYTES.index(TARGET)

    # Load every experiment (pretreated)
    exps = {}
    for e in EXP_IDS:
        x, a, wl = load_experiment(e)
        exps[e] = {'x': x, 'a': a}
        print(f"{e}: {x.shape[0]} samples")
    n_total = sum(exps[e]['x'].shape[0] for e in EXP_IDS)
    print(f"Total pooled: {n_total} samples across {len(EXP_IDS)} experiments\n")

    rows = []                                # per-experiment results
    all_ref, all_pred = [], []

    for held in EXP_IDS:
        train_e = [e for e in EXP_IDS if e != held]
        A_tr = np.vstack([exps[e]['a'] for e in train_e])
        Y_tr = np.vstack([exps[e]['x'] for e in train_e])
        A_te = exps[held]['a']
        y_te = exps[held]['x'][:, jt]

        # Nested selection + tuning on TRAIN ONLY
        mask, bp, inner_rmsecv = select_and_tune(A_tr, Y_tr, jt)
        n_sel = int(mask.sum())

        # Fit on train, predict the held-out experiment
        m = SVRModel()
        y_pred = m.fit_predict(
            A_tr[:, mask], Y_tr[:, jt].reshape(-1, 1), A_te[:, mask],
            C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
            kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0)
        ).flatten()

        rmsep = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
        rows.append((held, len(y_te), n_sel, inner_rmsecv, rmsep))
        all_ref.extend(y_te.tolist()); all_pred.extend(y_pred.tolist())
        print(f"[{held}] held out: {len(y_te)} samples | sel={n_sel} vars | "
              f"inner RMSECV={inner_rmsecv:.3f} | RMSEP={rmsep:.3f}  ({R._params_label(bp, skip=())})")

    # Pooled metrics across all held-out predictions
    all_ref = np.array(all_ref); all_pred = np.array(all_pred)
    rmsep_pool = float(np.sqrt(np.mean((all_ref - all_pred) ** 2)))
    sst = float(np.sum((all_ref - all_ref.mean()) ** 2)) or 1.0
    r2_pool = 1 - float(np.sum((all_ref - all_pred) ** 2)) / sst
    bias_pool = float(np.mean(all_pred - all_ref))

    print(f"\n{'='*68}")
    print(f"NESTED LOExO RESULT  -  target '{TARGET}'".center(68))
    print(f"{'='*68}")
    h = f"{'Held-out':<10}|{'n':^5}|{'#vars':^7}|{'inner RMSECV':^14}|{'RMSEP':^9}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for (e, n, nv, ir, rp) in rows:
        print(f"{e:<10}|{n:^5}|{nv:^7}|{ir:^14.3f}|{rp:^9.3f}")
    print("-" * len(h))
    print(f"\nPOOLED over {len(all_ref)} new-batch predictions:")
    print(f"  RMSEP = {rmsep_pool:.3f} {R.UNITS}   R2_pred = {r2_pool:.3f}   Bias = {bias_pool:+.3f}")
    print(f"\nReference points for '{TARGET}':")
    print(f"  Within-batch test (earlier): RMSEP~1.25, R2_pred~0.71")
    print(f"  5-fold CV (earlier):         RMSECV~1.72, R2_CV~0.73")

    # Save predictions + plot
    np.savetxt(os.path.join(RESULTS_DIR, "loexo_predictions.txt"),
               np.column_stack([all_ref, all_pred]),
               header="reference\tpredicted", fmt='%.5f', delimiter='\t')

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    start = 0
    for i, (e, n, nv, ir, rp) in enumerate(rows):
        yr = all_ref[start:start + n]; yp = all_pred[start:start + n]; start += n
        ax.scatter(yr, yp, color=cmap(i), label=f"{e} (RMSEP={rp:.2f})", s=55, alpha=0.85)
    lo, hi = all_ref.min(), all_ref.max()
    b = (hi - lo) * 0.08
    ax.plot([lo - b, hi + b], [lo - b, hi + b], 'k--', alpha=0.5)
    ax.text(0.05, 0.95, f"Pooled RMSEP={rmsep_pool:.3f}\n$R^2$={r2_pool:.3f}",
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_title(f"Nested LOExO - {TARGET} (predict a new batch)", loc='left')
    ax.set_xlabel(f"Reference {R.UNITS}"); ax.set_ylabel(f"Predicted {R.UNITS}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "LOExO_Predicted_vs_Reference.png"), dpi=300)
    print(f"\nSaved results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
