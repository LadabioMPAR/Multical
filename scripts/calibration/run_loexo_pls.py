"""
PLS under the SAME nested Leave-One-Experiment-Out (LOExO) protocol as SVR.

Purpose: confirm whether the failure to predict a NEW batch is a data limitation
(few batches) rather than something specific to SVR. For each held-out experiment
the PLS latent-variable count k is chosen by inner CV on the training experiments
only (leak-free); then PLS predicts the held-out batch.

SVR reference numbers (from run_loexo_preprocessing_svr.py) are printed alongside.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from src.multical.var_selection import calculate_rmsecv_fast, select_k_ftest
import scripts.calibration.run_calibration_svr as R

# =============================================================================
TARGET = 'gl'
RESULTS_DIR = "results_loexo_pls"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']

MAX_LV = 15
INNER_FOLDS = 5
INNER_CV = 'venetian'

CUT = ['Cut', 4400, 7500, 0]
SG = ['SG', 7, 2, 1, 0]
PIPELINES = {
    'Cut+SG (baseline)':  [CUT, SG],
    'Cut+SNV+SG':         [CUT, ['SNV', 0], SG],
    'Cut+SG+SNV':         [CUT, SG, ['SNV', 0]],
    'Cut+EMSC2+SG':       [CUT, ['EMSC', 2, 0], SG],
    'Cut+SNV (no deriv)': [CUT, ['SNV', 0]],
}

# SVR reference (pooled LOExO RMSEP, R2_pred) from the preprocessing screen
SVR_REF = {
    'Cut+SG (baseline)':  (5.887, -2.264),
    'Cut+SNV+SG':         (3.995, -0.503),
    'Cut+SG+SNV':         (3.565, -0.197),
    'Cut+EMSC2+SG':       (3.654, -0.258),
    'Cut+SNV (no deriv)': (5.231, -1.578),
}


def load_pair(refe, nonda):
    xi = np.atleast_2d(np.loadtxt(refe))
    if xi.shape[1] > 1:
        xi = xi[:, 1:]
    with open(nonda) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    absi = np.atleast_2d(np.loadtxt(nonda, skiprows=1))
    return xi, absi[:, 1:], wl


def load_raw():
    raw = {}
    for e in EXP_IDS:
        x_list, a_list, wl = [], [], None
        for split in ('cal', 'test'):
            refe = os.path.join(SPLITS, f"{e}_refe_{split}.txt")
            nonda = os.path.join(SPLITS, f"{e}_nonda_{split}.txt")
            if os.path.exists(refe) and os.path.exists(nonda):
                xi, ai, wl = load_pair(refe, nonda)
                x_list.append(xi); a_list.append(ai)
        raw[e] = (np.vstack(x_list), np.vstack(a_list), wl)
    return raw


def pls_predict_holdout(A_tr, y_tr, A_te, k):
    """Leak-free single-Y PLS: fit on train (k LVs), predict test in real units."""
    Xn, Xmed, Xsig = zscore_matlab_style(A_tr)
    yn, ymed, ysig = zscore_matlab_style(y_tr.reshape(-1, 1))
    _, _, P, _, Q, W, _, _ = PLS().nipals(Xn, yn, k)
    wk, pk, qk = W[:, :k], P[:, :k], Q[:, :k]
    beta = wk @ np.linalg.pinv(pk.T @ wk) @ qk.T
    Xsig_safe = Xsig.copy(); Xsig_safe[Xsig_safe == 0] = 1.0
    Xte_n = (A_te - Xmed) / Xsig_safe
    yte = (Xte_n @ beta) * ysig + ymed
    return yte.flatten()


def loexo_pls_for_pipeline(raw, pipeline, jt):
    exps = {}
    for e in EXP_IDS:
        x, a_raw, wl = raw[e]
        a_pre, _ = apply_pretreatment(pipeline, a_raw.copy(), wl.copy(), plot=False)
        exps[e] = (x, a_pre)

    all_ref, all_pred, per = [], [], []
    for held in EXP_IDS:
        train_e = [e for e in EXP_IDS if e != held]
        A_tr = np.vstack([exps[e][1] for e in train_e])
        Y_tr = np.vstack([exps[e][0] for e in train_e])
        A_te = exps[held][1]
        y_te = exps[held][0][:, jt]

        # inner k selection on TRAIN only
        _, _, rmsecv_vec = calculate_rmsecv_fast(A_tr, Y_tr, MAX_LV, INNER_FOLDS, INNER_CV, target_idx=jt)
        k_opt = select_k_ftest(rmsecv_vec, n_cal=A_tr.shape[0])

        y_pred = pls_predict_holdout(A_tr, Y_tr[:, jt], A_te, k_opt)
        per.append((held, k_opt, float(np.sqrt(np.mean((y_te - y_pred) ** 2)))))
        all_ref.extend(y_te.tolist()); all_pred.extend(y_pred.tolist())

    all_ref = np.array(all_ref); all_pred = np.array(all_pred)
    rmsep = float(np.sqrt(np.mean((all_ref - all_pred) ** 2)))
    sst = float(np.sum((all_ref - all_ref.mean()) ** 2)) or 1.0
    r2 = 1 - float(np.sum((all_ref - all_pred) ** 2)) / sst
    bias = float(np.mean(all_pred - all_ref))
    return rmsep, r2, bias, per


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jt = R.ANALYTES.index(TARGET)
    raw = load_raw()
    print("Samples per experiment:", {e: raw[e][0].shape[0] for e in EXP_IDS})

    results = {}
    for name, pipe in PIPELINES.items():
        rmsep, r2, bias, per = loexo_pls_for_pipeline(raw, pipe, jt)
        results[name] = (rmsep, r2, bias, per)
        print(f"\n### {name}: PLS pooled RMSEP={rmsep:.3f}  R2={r2:.3f}  bias={bias:+.3f}")
        print("   per-exp (k, RMSEP):", {e: (k, round(v, 2)) for e, k, v in per})

    print(f"\n{'='*72}")
    print(f"PLS vs SVR under nested LOExO  (target '{TARGET}')".center(72))
    print(f"{'='*72}")
    h = f"{'Pipeline':<22}|{'PLS RMSEP':^11}|{'PLS R2':^9}|{'SVR RMSEP':^11}|{'SVR R2':^9}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for name in PIPELINES:
        pr, pr2 = results[name][0], results[name][1]
        sr, sr2 = SVR_REF[name]
        print(f"{name:<22}|{pr:^11.3f}|{pr2:^9.3f}|{sr:^11.3f}|{sr2:^9.3f}")
    print("-" * len(h))
    print(f"(units {R.UNITS}; gl std ~3.26 -> R2~0 means 'no better than the mean')")

    # Bar chart PLS vs SVR pooled RMSEP
    names = list(PIPELINES.keys())
    pls_v = [results[n][0] for n in names]
    svr_v = [SVR_REF[n][0] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, pls_v, w, label='PLS', color='seagreen')
    ax.bar(x + w / 2, svr_v, w, label='SVR', color='steelblue')
    ax.axhline(3.26, color='red', ls='--', lw=1, label='gl std (R2=0)')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel(f"Pooled LOExO RMSEP ({R.UNITS})")
    ax.set_title(f"Cross-batch RMSEP: PLS vs SVR - {TARGET}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "LOExO_PLS_vs_SVR.png"), dpi=300)
    print(f"\nSaved chart in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
