"""
Fine (coarse-to-fine) refinement of the SVR model for a single target analyte.

The coarse search in run_calibration_svr.py found, for 'gl':
    poly, C=10, epsilon=0.001, gamma=scale, degree=2, coef0=1.0  (RMSECV=1.830)

This script searches a fine grid centred on that optimum. The grid is a superset
of the coarse optimum, so the refined RMSECV can only match or improve it. If the
saved model exists, its target-analyte entry is retrained with the refined
hyper-parameters (the other analytes are kept untouched).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # headless: only saving figures
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.svr import SVRModel
import scripts.calibration.run_calibration_svr as R

# =============================================================================
#                              CONFIGURATION
# =============================================================================
TARGET = 'gl'                    # Analyte to refine
COARSE_RMSECV = 1.82996          # Reference (coarse-search) RMSECV for reporting

FINE_KERNELS = ['poly']          # Coarse winner for 'gl'; refine within it
# Fine grid centred on the coarse optimum (includes the coarse optimum itself).
FINE_GRID = {
    'C':       [4, 6, 8, 10, 12, 15, 20, 30, 50],
    'epsilon': [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01],
    'gamma':   ['scale', 'auto', 1e-4, 3e-4, 1e-3],
    'degree':  [2, 3],
    'coef0':   [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
}

MODEL_PATH = os.path.join(R.RESULTS_DIR, R.model_name)


# =============================================================================
#                                 MAIN
# =============================================================================
def main():
    if TARGET not in R.ANALYTES:
        raise ValueError(f"TARGET '{TARGET}' not in ANALYTES {R.ANALYTES}")
    jt = R.ANALYTES.index(TARGET)

    # --- Load + pretreat (identical pipeline to the main script) ---
    x0, wl, absor_raw = R.load_data(R.DATA_FILES)
    absor_pre, wl_pre = apply_pretreatment(R.PRETREATMENT, absor_raw.copy(), wl.copy(), plot=False)
    y = x0[:, jt]

    # venetian CV ignores 'perm'; arange keeps things deterministic/reproducible.
    perm = np.random.permutation(x0.shape[0]) if R.CV_TYPE == 'random' else np.arange(x0.shape[0])

    # --- Fine search (single analyte) ---
    combos = R.build_param_combos(FINE_GRID, FINE_KERNELS)
    print(f"Refining '{TARGET}': {len(combos)} combinations x {R.K_FOLDS} folds "
          f"(CV={R.CV_TYPE})")

    best_params, rmsecv_best, _ = R.grid_search_svr(
        absor_pre, x0[:, [jt]], combos, R.K_FOLDS, R.CV_TYPE, [TARGET], perm=perm
    )
    bp = best_params[0]
    rmsecv = float(rmsecv_best[0])

    # --- Metrics for the refined model ---
    y_cv = R.svr_cv_predict_single(absor_pre, y, bp, R.K_FOLDS, R.CV_TYPE, perm=perm)
    m = SVRModel()
    y_cal = m.fit_predict(
        absor_pre, y.reshape(-1, 1), absor_pre,
        C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
        kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0)
    ).flatten()

    sst = np.sum((y - y.mean()) ** 2)
    sst = sst if sst != 0 else 1.0
    r2cv = 1 - np.sum((y - y_cv) ** 2) / sst
    r2cal = 1 - np.sum((y - y_cal) ** 2) / sst
    rmsecal = np.sqrt(np.mean((y - y_cal) ** 2))

    improvement = (COARSE_RMSECV - rmsecv) / COARSE_RMSECV * 100.0

    print(f"\n{'='*60}")
    print(f"REFINED MODEL FOR '{TARGET}'".center(60))
    print(f"{'='*60}")
    print(f"  Params : {R._params_label(bp, skip=())}")
    print(f"  RMSECV : {rmsecv:.5f} {R.UNITS}   (coarse: {COARSE_RMSECV:.5f}  ->  {improvement:+.1f}%)")
    print(f"  RMSEC  : {rmsecal:.5f} {R.UNITS}")
    print(f"  R2_CV  : {r2cv:.4f}")
    print(f"  R2_Cal : {r2cal:.4f}")
    print(f"{'='*60}")

    # --- Refined Predicted vs Measured plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    col = R.COLORS[jt] if jt < len(R.COLORS) else 'blue'
    ax.scatter(y, y_cal, marker='o', facecolors='none', edgecolors=col, alpha=0.6, label='Calibration')
    ax.scatter(y, y_cv, c=col, marker='x', alpha=0.8, label='Cross-Validation')
    lo = min(y.min(), y_cal.min(), y_cv.min())
    hi = max(y.max(), y_cal.max(), y_cv.max())
    buff = (hi - lo) * 0.05
    ax.plot([lo - buff, hi + buff], [lo - buff, hi + buff], 'k--', alpha=0.5)
    stats = (f"Cal: $R^2$={r2cal:.3f}, RMSE={rmsecal:.3f}\n"
             f"CV: $R^2$={r2cv:.3f}, RMSE={rmsecv:.3f}")
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"{TARGET} (refined) - Predicted vs Measured", loc='left')
    ax.set_xlabel(f"Measured {R.UNITS}")
    ax.set_ylabel(f"Predicted {R.UNITS}")
    ax.legend()
    fig.tight_layout()
    out_plot = os.path.join(R.RESULTS_DIR, f"Refined_{TARGET}_Predicted_vs_Measured.png")
    fig.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {out_plot}")

    # --- Update the saved model in place (target analyte only) ---
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        svr_final = SVRModel()
        svr_final.fit_predict(
            absor_pre, y.reshape(-1, 1), None,
            C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
            kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0)
        )
        updated = False
        for a in model_data.get('analytes', []):
            if a['index'] == jt:
                a['params'] = bp
                a['model'] = svr_final
                a['rmsecv'] = rmsecv
                updated = True
        if updated:
            joblib.dump(model_data, MODEL_PATH)
            print(f"Updated '{TARGET}' entry in saved model: {MODEL_PATH}")
        else:
            print(f"Warning: analyte index {jt} not found in {MODEL_PATH}; model not updated.")
    else:
        print(f"Note: {MODEL_PATH} not found; run run_calibration_svr.py first to (re)create it.")


if __name__ == "__main__":
    main()
