"""
Compare batch-robust PREPROCESSING under nested Leave-One-Experiment-Out (LOExO).

Goal: does scatter correction (SNV / EMSC) rescue cross-batch generalisation for
'gl'? Each pipeline is evaluated with the SAME nested LOExO protocol:
  - hold out a whole experiment (batch),
  - tune the SVR by inner CV on the training experiments only,
  - predict the held-out batch.
No variable selection here (isolates the preprocessing effect; faster). Pre-
processing is applied per-experiment, so SNV (row-wise) and EMSC (self-reference)
are leak-free w.r.t. the held-out batch and the labels.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.svr import SVRModel
import scripts.calibration.run_calibration_svr as R

# =============================================================================
TARGET = 'gl'
RESULTS_DIR = "results_loexo_preproc_svr"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']

INNER_FOLDS = 5
INNER_CV = 'venetian'
INNER_KERNELS = ['poly', 'rbf']
INNER_GRID = {
    'C':       [3, 6, 10, 100],
    'epsilon': [0.001, 0.01, 0.05],
    'gamma':   ['scale'],
    'degree':  [2],
    'coef0':   [1.0],
}

CUT = ['Cut', 4400, 7500, 0]
SG = ['SG', 7, 2, 1, 0]
PIPELINES = {
    'Cut+SG (baseline)':  [CUT, SG],
    'Cut+SNV+SG':         [CUT, ['SNV', 0], SG],
    'Cut+SG+SNV':         [CUT, SG, ['SNV', 0]],
    'Cut+EMSC2+SG':       [CUT, ['EMSC', 2, 0], SG],
    'Cut+SNV (no deriv)': [CUT, ['SNV', 0]],
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


def _svr_kwargs(bp):
    return dict(C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
               kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0))


def loexo_for_pipeline(raw, pipeline, jt):
    # Pretreat each experiment independently (leak-free).
    exps = {}
    for e in EXP_IDS:
        x, a_raw, wl = raw[e]
        a_pre, _ = apply_pretreatment(pipeline, a_raw.copy(), wl.copy(), plot=False)
        exps[e] = (x, a_pre)

    combos = R.build_param_combos(INNER_GRID, INNER_KERNELS)
    all_ref, all_pred, per = [], [], []

    for held in EXP_IDS:
        train_e = [e for e in EXP_IDS if e != held]
        A_tr = np.vstack([exps[e][1] for e in train_e])
        Y_tr = np.vstack([exps[e][0] for e in train_e])
        A_te = exps[held][1]
        y_te = exps[held][0][:, jt]

        bp_list, _, _ = R.grid_search_svr(A_tr, Y_tr[:, [jt]], combos, INNER_FOLDS, INNER_CV, [TARGET], perm=None)
        m = SVRModel()
        y_pred = m.fit_predict(A_tr, Y_tr[:, jt].reshape(-1, 1), A_te, **_svr_kwargs(bp_list[0])).flatten()

        per.append((held, float(np.sqrt(np.mean((y_te - y_pred) ** 2)))))
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
        print(f"\n### Pipeline: {name}")
        rmsep, r2, bias, per = loexo_for_pipeline(raw, pipe, jt)
        results[name] = (rmsep, r2, bias, per)
        print(f"   pooled RMSEP={rmsep:.3f}  R2={r2:.3f}  bias={bias:+.3f}")
        print("   per-exp RMSEP:", {e: round(v, 2) for e, v in per})

    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPARISON (nested LOExO, target '{TARGET}')".center(70))
    print(f"{'='*70}")
    h = f"{'Pipeline':<22}|{'pooled RMSEP':^14}|{'R2_pred':^10}|{'bias':^9}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for name, (rmsep, r2, bias, _) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<22}|{rmsep:^14.3f}|{r2:^10.3f}|{bias:^+9.3f}")
    print("-" * len(h))
    print(f"(units {R.UNITS}; reference: prior within-batch test ~1.25 was leaky)")

    # Bar chart of pooled RMSEP
    names = list(results.keys())
    rmseps = [results[n][0] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), rmseps, color='steelblue')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel(f"Pooled LOExO RMSEP ({R.UNITS})")
    ax.set_title(f"Cross-batch RMSEP by preprocessing - {TARGET}")
    for b, v in zip(bars, rmseps):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "LOExO_preprocessing_comparison.png"), dpi=300)
    print(f"\nSaved chart in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
