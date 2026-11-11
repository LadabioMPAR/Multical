"""
Model comparison on the Kennard-Stone (KS) cal/test split, target = glucose.

Uses the SAME 9 models as run_loexo_model_comparison.py, but instead of LOExO it
trains on the pooled KS calibration set (all *_cal files) and predicts the pooled KS
test set (all *_test files). This is the WITHIN-BATCH scenario (cal and test of the
same experiment), i.e. the optimistic "predict inside a calibrated run" use case —
NOT cross-batch generalisation (for that, see run_loexo_model_comparison.py).

Each model still tunes/selects on the training (cal) data only -> leak-free wrt test.
Preprocessing: the robust default from run_calibration_svr.py (Cut + SG + SNV).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_calibration_svr as R
import scripts.calibration.run_loexo_model_comparison as M

RESULTS_DIR = "results_ks_models"
SPLITS = "data/splits"
EXP_IDS = ['exp4', 'exp5', 'exp6', 'exp7', 'exp8', 'exp9']
GL = 1

# LOExO reference (cross-batch) for the same models/pipeline, to contrast
LOEXO_R2 = {
    'mean (no-skill)': -0.11, 'Random Forest': -0.32, 'Ridge': -0.34, 'ElasticNet': -0.35,
    'kNN': -0.37, 'PLS': -0.43, 'SVR (rbf)': -0.60, 'PCR': -0.71, 'SPA (MLR)': -1.02,
}


def load_split(split):
    """Load + pretreat one KS split ('cal' or 'test') pooled over all experiments."""
    a_list, y_list, wl = [], [], None
    for e in EXP_IDS:
        refe = os.path.join(SPLITS, f"{e}_refe_{split}.txt")
        nonda = os.path.join(SPLITS, f"{e}_nonda_{split}.txt")
        if os.path.exists(refe) and os.path.exists(nonda):
            xi, ai, wl = M.load_pair(refe, nonda)
            a_list.append(ai); y_list.append(xi[:, GL])
    a_pre, _ = apply_pretreatment(R.PRETREATMENT, np.vstack(a_list), wl, plot=False)
    return a_pre, np.concatenate(y_list)


def metrics(yr, yp):
    rmse = float(np.sqrt(np.mean((yr - yp) ** 2)))
    sst = float(np.sum((yr - yr.mean()) ** 2)) or 1.0
    r2 = 1 - float(np.sum((yr - yp) ** 2)) / sst
    bias = float(np.mean(yp - yr))
    return rmse, r2, bias


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    A_cal, y_cal = load_split('cal')
    A_test, y_test = load_split('test')
    print(f"KS split: cal = {A_cal.shape[0]} samples, test = {A_test.shape[0]} samples, "
          f"{A_cal.shape[1]} features\n")

    res = {}
    for name, fn in M.MODELS.items():
        yp = fn(A_cal, y_cal, A_test)
        res[name] = metrics(y_test, yp)
        print(f"  {name:<16}: RMSEP={res[name][0]:.3f}  R2={res[name][1]:+.3f}  bias={res[name][2]:+.3f}")

    print(f"\n{'='*64}")
    print("KS (within-batch) MODEL COMPARISON - glucose".center(64))
    print(f"{'='*64}")
    h = f"{'Model':<16}|{'RMSEP':^8}|{'R2 (KS)':^9}|{'bias':^8}|{'R2 (LOExO)':^11}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for name, (rmse, r2, bias) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<16}|{rmse:^8.3f}|{r2:^+9.3f}|{bias:^+8.2f}|{LOEXO_R2.get(name, float('nan')):^+11.2f}")
    print("-" * len(h))
    print("(units g/L; KS test = within-batch/optimistic; LOExO = cross-batch/honest)")

    # Bar chart: KS R2 vs LOExO R2
    names = list(res.keys())
    ks = [res[n][1] for n in names]
    lo = [LOEXO_R2.get(n, np.nan) for n in names]
    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, ks, w, label='KS (within-batch)', color='seagreen')
    ax.bar(x + w / 2, lo, w, label='LOExO (cross-batch)', color='indianred')
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylabel("R²_pred (glucose)")
    ax.set_title("Glucose: KS within-batch vs LOExO cross-batch, by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "KS_vs_LOExO_by_model.png"), dpi=300)
    print(f"\nSaved chart in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
