"""
Validate the saved SVR calibration model on the independent TEST split.

Loads the *_test.txt files, applies the same pretreatment as calibration,
predicts with results_svr/model_calibration_svr.pkl and reports RMSEP / R2_pred
per analyte (compared against the stored RMSECV).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import load_and_predict_svr
import scripts.calibration.run_calibration_svr as R

# Test counterparts of R.DATA_FILES
TEST_FILES = [
    ('data/splits/exp4_refe_test.txt', 'data/splits/exp4_nonda_test.txt'),
    ('data/splits/exp5_refe_test.txt', 'data/splits/exp5_nonda_test.txt'),
    ('data/splits/exp6_refe_test.txt', 'data/splits/exp6_nonda_test.txt'),
    ('data/splits/exp7_refe_test.txt', 'data/splits/exp7_nonda_test.txt'),
    ('data/splits/exp8_refe_test.txt', 'data/splits/exp8_nonda_test.txt'),
    ('data/splits/exp9_refe_test.txt', 'data/splits/exp9_nonda_test.txt'),
]

MODEL_PATH = os.path.join(R.RESULTS_DIR, R.model_name)


def main():
    nc = len(R.ANALYTES)

    # --- Load + pretreat the TEST set (same pipeline as calibration) ---
    x_test, wl, absor_raw = R.load_data(TEST_FILES)
    absor_pre, wl_pre = apply_pretreatment(R.PRETREATMENT, absor_raw.copy(), wl.copy(), plot=False)
    print(f"Test samples: {x_test.shape[0]}, Features after pretreatment: {absor_pre.shape[1]}")

    # --- Predict with the saved SVR model ---
    y_pred, rmsecv_stored = load_and_predict_svr(absor_pre, wl_pre, MODEL_PATH)

    # --- Metrics ---
    rmsep = np.sqrt(np.mean((x_test - y_pred) ** 2, axis=0))
    sst = np.sum((x_test - x_test.mean(axis=0)) ** 2, axis=0)
    sst[sst == 0] = 1.0
    r2p = 1 - np.sum((x_test - y_pred) ** 2, axis=0) / sst
    bias = np.mean(y_pred - x_test, axis=0)

    print(f"\n{'='*72}")
    print("SVR VALIDATION ON INDEPENDENT TEST SET".center(72))
    print(f"{'='*72}")
    header = f"{'Analyte':<10}|{'RMSECV':^11}|{'RMSEP':^11}|{'R2_pred':^10}|{'Bias':^10}"
    print("-" * len(header)); print(header); print("-" * len(header))
    for j in range(nc):
        print(f"{R.ANALYTES[j]:<10}|{rmsecv_stored[j]:^11.4f}|{rmsep[j]:^11.4f}|{r2p[j]:^10.4f}|{bias[j]:^10.4f}")
    print("-" * len(header))
    print(f"(RMSECV/RMSEP in {R.UNITS}; Bias = mean(pred - ref))")

    # --- Plot predicted vs reference (test) ---
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5), squeeze=False)
    axes = axes.flatten()
    for j in range(nc):
        ax = axes[j]
        col = R.COLORS[j] if j < len(R.COLORS) else 'blue'
        yr, yp = x_test[:, j], y_pred[:, j]
        ax.scatter(yr, yp, c=col, alpha=0.75, label='Test')
        lo = min(yr.min(), yp.min()); hi = max(yr.max(), yp.max())
        buff = (hi - lo) * 0.05
        ax.plot([lo - buff, hi + buff], [lo - buff, hi + buff], 'k--', alpha=0.5, label='1:1')
        ax.text(0.05, 0.95, f"RMSEP={rmsep[j]:.3f}\n$R^2$={r2p[j]:.3f}",
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f"{R.ANALYTES[j]} - Test prediction", loc='left')
        ax.set_xlabel(f"Reference {R.UNITS}")
        ax.set_ylabel(f"Predicted {R.UNITS}")
        ax.legend()
    fig.tight_layout()
    out = os.path.join(R.RESULTS_DIR, "SVR_Test_Predicted_vs_Reference.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
