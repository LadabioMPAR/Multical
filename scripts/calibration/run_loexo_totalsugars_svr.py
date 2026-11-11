"""
LOExO comparison: GLUCOSE vs TOTAL SUGARS (gl + xy) with the robust SVR pipeline.

Same nested Leave-One-Experiment-Out protocol as the other LOExO scripts, using the
default robust preprocessing from run_calibration_svr.py (Cut + SG + SNV) and an
rbf-only inner search. Goal: check whether 'total sugars' generalises better across
batches than glucose alone (it is the dominant, most spectrally-supported signal).
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

RESULTS_DIR = "results_loexo_totalsugars"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']

INNER_FOLDS = 5
INNER_CV = 'venetian'
INNER_KERNELS = ['rbf']
INNER_GRID = {
    'C':       [1, 10, 100, 1000],
    'epsilon': [0.001, 0.01, 0.05],
    'gamma':   ['scale', 'auto', 1e-4, 1e-3],
    'degree':  [2],
    'coef0':   [1.0],
}

# Targets: name -> function mapping an (n,3) concentration matrix to the target vector
GL, XY = 1, 2
TARGETS = {
    'glucose':       lambda x: x[:, GL],
    'total sugars':  lambda x: x[:, GL] + x[:, XY],
}


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
                xi, ai, wl = load_pair(refe, nonda)
                x_list.append(xi); a_list.append(ai)
        x = np.vstack(x_list)
        a_pre, _ = apply_pretreatment(R.PRETREATMENT, np.vstack(a_list), wl, plot=False)
        exps[e] = (x, a_pre)
    return exps


def _kwargs(bp):
    return dict(C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
               kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0))


def loexo_target(exps, y_of, name):
    combos = R.build_param_combos(INNER_GRID, INNER_KERNELS)
    all_ref, all_pred, per = [], [], []
    for held in EXP_IDS:
        train_e = [e for e in EXP_IDS if e != held]
        A_tr = np.vstack([exps[e][1] for e in train_e])
        y_tr = np.concatenate([y_of(exps[e][0]) for e in train_e])
        A_te = exps[held][1]
        y_te = y_of(exps[held][0])

        bp_list, _, _ = R.grid_search_svr(A_tr, y_tr.reshape(-1, 1), combos, INNER_FOLDS, INNER_CV, [name], perm=None)
        m = SVRModel()
        y_pred = m.fit_predict(A_tr, y_tr.reshape(-1, 1), A_te, **_kwargs(bp_list[0])).flatten()

        per.append((held, float(np.sqrt(np.mean((y_te - y_pred) ** 2)))))
        all_ref.extend(y_te.tolist()); all_pred.extend(y_pred.tolist())

    all_ref = np.array(all_ref); all_pred = np.array(all_pred)
    rmsep = float(np.sqrt(np.mean((all_ref - all_pred) ** 2)))
    sst = float(np.sum((all_ref - all_ref.mean()) ** 2)) or 1.0
    r2 = 1 - float(np.sum((all_ref - all_pred) ** 2)) / sst
    bias = float(np.mean(all_pred - all_ref))
    rel = rmsep / all_ref.mean() * 100.0
    return dict(rmsep=rmsep, r2=r2, bias=bias, rel=rel, mean=all_ref.mean(),
                ref=all_ref, pred=all_pred, per=per)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exps = load_and_pretreat()
    print("Samples per experiment:", {e: exps[e][0].shape[0] for e in EXP_IDS})

    res = {name: loexo_target(exps, fn, name) for name, fn in TARGETS.items()}

    print(f"\n{'='*72}")
    print("LOExO: GLUCOSE vs TOTAL SUGARS  (robust pipeline Cut+SG+SNV, rbf)".center(72))
    print(f"{'='*72}")
    h = f"{'Target':<14}|{'mean':^8}|{'RMSEP':^9}|{'RMSEP %':^9}|{'R2_pred':^9}|{'bias':^8}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for name in TARGETS:
        r = res[name]
        print(f"{name:<14}|{r['mean']:^8.2f}|{r['rmsep']:^9.3f}|{r['rel']:^9.1f}|{r['r2']:^9.3f}|{r['bias']:^+8.2f}")
    print("-" * len(h))
    print(f"(units {R.UNITS}; RMSEP % = RMSEP / mean -> comparable across targets)")

    # Plots: predicted vs reference for each target
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 5), squeeze=False)
    axes = axes.flatten()
    for ax, name in zip(axes, TARGETS):
        r = res[name]
        yr, yp = r['ref'], r['pred']
        ax.scatter(yr, yp, alpha=0.75, color='teal')
        lo, hi = yr.min(), yr.max(); b = (hi - lo) * 0.08
        ax.plot([lo - b, hi + b], [lo - b, hi + b], 'k--', alpha=0.5)
        ax.text(0.05, 0.95, f"RMSEP={r['rmsep']:.2f} ({r['rel']:.0f}%)\n$R^2$={r['r2']:.2f}",
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(f"{name} (LOExO)", loc='left')
        ax.set_xlabel(f"Reference {R.UNITS}"); ax.set_ylabel(f"Predicted {R.UNITS}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "LOExO_glucose_vs_totalsugars.png"), dpi=300)
    print(f"\nSaved chart in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
