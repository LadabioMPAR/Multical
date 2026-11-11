"""
Variable selection for an SVR model, targeting a single analyte ('gl').

Strategy (VIP -> SVR, validated on the independent test set):
  1. Rank all wavelengths by VIP (supervised, from a PLS fit on the target).
  2. For a range of VIP thresholds, build the variable subset and evaluate an
     SVR by cross-validation, re-tuning a small hyper-parameter grid each time
     (fewer variables change the optimal gamma/C).
  3. Keep the threshold with the lowest SVR RMSECV.
  4. Retrain the SVR on the selected bands and validate on *_test.txt (RMSEP),
     comparing against the full-spectrum baseline.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import joblib
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
RESULTS_DIR = "results_var_selection_svr"
MAX_LATENT_VARS = 15            # for the PLS used to rank variables (VIP)
VIP_THRESHOLDS = np.arange(0.2, 1.8, 0.1)
MIN_VARS = 5                    # ignore thresholds that keep fewer variables

# Small per-subset SVR grid (re-tuned for each candidate subset)
VS_KERNELS = ['poly', 'rbf']
VS_GRID = {
    'C':       [1, 3, 6, 10, 30, 100],
    'epsilon': [0.0001, 0.001, 0.01, 0.05],
    'gamma':   ['scale', 'auto'],
    'degree':  [2, 3],
    'coef0':   [0.5, 1.0],
}

TEST_FILES = [
    ('data/splits/exp4_refe_test.txt', 'data/splits/exp4_nonda_test.txt'),
    ('data/splits/exp5_refe_test.txt', 'data/splits/exp5_nonda_test.txt'),
    ('data/splits/exp6_refe_test.txt', 'data/splits/exp6_nonda_test.txt'),
    ('data/splits/exp7_refe_test.txt', 'data/splits/exp7_nonda_test.txt'),
    ('data/splits/exp8_refe_test.txt', 'data/splits/exp8_nonda_test.txt'),
    ('data/splits/exp9_refe_test.txt', 'data/splits/exp9_nonda_test.txt'),
]


def _metrics(y_ref, y_pred):
    rmse = float(np.sqrt(np.mean((y_ref - y_pred) ** 2)))
    sst = float(np.sum((y_ref - y_ref.mean()) ** 2)) or 1.0
    r2 = 1 - float(np.sum((y_ref - y_pred) ** 2)) / sst
    return rmse, r2


def _svr_fit_full(absor_sub, y, bp):
    """Train SVR on full subset and return the model and calibration preds."""
    m = SVRModel()
    y_cal = m.fit_predict(
        absor_sub, y.reshape(-1, 1), absor_sub,
        C=bp['C'], epsilon=bp['epsilon'], gamma=bp.get('gamma', 'scale'),
        kernel=bp['kernel'], degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0)
    ).flatten()
    return m, y_cal


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jt = R.ANALYTES.index(TARGET)
    perm = None  # venetian CV ignores perm

    # --- Load + pretreat calibration ---
    x0, wl, absor_raw = R.load_data(R.DATA_FILES)
    absor_pre, wl_pre = apply_pretreatment(R.PRETREATMENT, absor_raw.copy(), wl.copy(), plot=False)
    y = x0[:, jt]
    n_feat = absor_pre.shape[1]
    print(f"Calibration: {x0.shape[0]} samples, {n_feat} features. Target='{TARGET}'")

    # --- 1. Optimal k + VIP ranking (PLS, target analyte) ---
    _, _, rmsecv_vec = calculate_rmsecv_fast(absor_pre, x0, MAX_LATENT_VARS, R.K_FOLDS, R.CV_TYPE, target_idx=jt)
    k_opt = select_k_ftest(rmsecv_vec, n_cal=x0.shape[0])
    print(f"VIP PLS optimal k (target '{TARGET}'): {k_opt}")
    vip = calculate_vip(PLS(), absor_pre, y.reshape(-1, 1), k_opt, wavelengths=wl_pre, output_dir=RESULTS_DIR)

    combos = R.build_param_combos(VS_GRID, VS_KERNELS)

    # --- 2. Threshold sweep (SVR CV per subset) ---
    print(f"\nThreshold sweep ({len(VIP_THRESHOLDS)} thresholds, {len(combos)} SVR combos each)...")
    sweep = []  # (thresh, n_sel, rmsecv, bp)
    for th in VIP_THRESHOLDS:
        mask = vip >= th
        n_sel = int(mask.sum())
        if n_sel < MIN_VARS:
            continue
        bp_list, rmse_list, _ = R.grid_search_svr(
            absor_pre[:, mask], x0[:, [jt]], combos, R.K_FOLDS, R.CV_TYPE, [TARGET], perm=perm
        )
        sweep.append((float(th), n_sel, float(rmse_list[0]), bp_list[0]))
        print(f"  VIP>={th:.1f}: {n_sel:5d} vars -> RMSECV={rmse_list[0]:.4f} ({bp_list[0]['kernel']})")

    if not sweep:
        print("No threshold kept enough variables; aborting.")
        return

    # --- 3. Pick best subset ---
    best_th, best_nsel, best_rmsecv, best_bp = min(sweep, key=lambda t: t[2])
    best_mask = vip >= best_th
    print(f"\nBest subset: VIP>={best_th:.1f} | {best_nsel} vars | "
          f"RMSECV={best_rmsecv:.4f} | {R._params_label(best_bp, skip=())}")

    # --- Full-spectrum baseline with the SAME small grid (fair comparison) ---
    bp_full_list, rmse_full_list, _ = R.grid_search_svr(
        absor_pre, x0[:, [jt]], combos, R.K_FOLDS, R.CV_TYPE, [TARGET], perm=perm
    )
    full_rmsecv, full_bp = float(rmse_full_list[0]), bp_full_list[0]

    # --- 4. Validate selected vs full on the TEST set ---
    x_test, wl_t, absor_raw_t = R.load_data(TEST_FILES)
    absor_pre_t, _ = apply_pretreatment(R.PRETREATMENT, absor_raw_t.copy(), wl_t.copy(), plot=False)
    y_test = x_test[:, jt]

    # selected
    m_sel, ycal_sel = _svr_fit_full(absor_pre[:, best_mask], y, best_bp)
    ytest_sel = m_sel.predict(absor_pre_t[:, best_mask]).flatten()
    rmsep_sel, r2p_sel = _metrics(y_test, ytest_sel)
    ycv_sel = R.svr_cv_predict_single(absor_pre[:, best_mask], y, best_bp, R.K_FOLDS, R.CV_TYPE, perm=perm)
    _, r2cv_sel = _metrics(y, ycv_sel)

    # full
    m_full, _ = _svr_fit_full(absor_pre, y, full_bp)
    ytest_full = m_full.predict(absor_pre_t).flatten()
    rmsep_full, r2p_full = _metrics(y_test, ytest_full)
    ycv_full = R.svr_cv_predict_single(absor_pre, y, full_bp, R.K_FOLDS, R.CV_TYPE, perm=perm)
    _, r2cv_full = _metrics(y, ycv_full)

    print(f"\n{'='*70}")
    print(f"SVR VARIABLE SELECTION RESULT  -  target '{TARGET}'".center(70))
    print(f"{'='*70}")
    head = f"{'Model':<22}|{'#vars':^7}|{'RMSECV':^9}|{'R2_CV':^8}|{'RMSEP':^9}|{'R2_pred':^9}"
    print("-" * len(head)); print(head); print("-" * len(head))
    print(f"{'Full spectrum':<22}|{n_feat:^7}|{full_rmsecv:^9.4f}|{r2cv_full:^8.4f}|{rmsep_full:^9.4f}|{r2p_full:^9.4f}")
    print(f"{'Selected (VIP>=%.1f)' % best_th:<22}|{best_nsel:^7}|{best_rmsecv:^9.4f}|{r2cv_sel:^8.4f}|{rmsep_sel:^9.4f}|{r2p_sel:^9.4f}")
    print("-" * len(head))
    print(f"(units {R.UNITS}; test set = {x_test.shape[0]} samples)")

    # --- 5. Plots ---
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_spec = absor_pre.mean(axis=0)
    ax.plot(wl_pre, mean_spec, 'k-', alpha=0.3, label='Mean spectrum')
    ax.scatter(wl_pre[best_mask], mean_spec[best_mask], c='r', s=8, label=f'Selected ({best_nsel})')
    ax.set_title(f"Selected variables (VIP>={best_th:.1f}) - target {TARGET}")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)"); ax.set_ylabel("Absorbance"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "Selected_Variables.png"), dpi=300)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    col = R.COLORS[jt] if jt < len(R.COLORS) else 'blue'
    ax2.scatter(y_test, ytest_full, marker='s', facecolors='none', edgecolors='gray',
                alpha=0.7, label=f'Full ({n_feat} var)')
    ax2.scatter(y_test, ytest_sel, c=col, marker='o', alpha=0.8, label=f'Selected ({best_nsel} var)')
    lo = min(y_test.min(), ytest_full.min(), ytest_sel.min())
    hi = max(y_test.max(), ytest_full.max(), ytest_sel.max())
    b = (hi - lo) * 0.05
    ax2.plot([lo - b, hi + b], [lo - b, hi + b], 'k--', alpha=0.5)
    ax2.set_title(f"{TARGET} - Test prediction (full vs selected)", loc='left')
    ax2.set_xlabel(f"Reference {R.UNITS}"); ax2.set_ylabel(f"Predicted {R.UNITS}"); ax2.legend()
    fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS_DIR, "Test_Full_vs_Selected.png"), dpi=300)

    # --- 6. Save artifacts ---
    np.savetxt(os.path.join(RESULTS_DIR, "selected_wavelengths.txt"),
               wl_pre[best_mask], fmt='%.4f', header="Selected wavelengths (cm-1)")
    joblib.dump({
        'type': 'SVR_VARSEL', 'target': TARGET, 'analyte_index': jt,
        'mask': best_mask, 'wavelengths_full': wl_pre, 'wavelengths_selected': wl_pre[best_mask],
        'params': best_bp, 'model': m_sel,
        'rmsecv': best_rmsecv, 'rmsep': rmsep_sel, 'r2_pred': r2p_sel,
    }, os.path.join(RESULTS_DIR, "model_varsel_svr_gl.pkl"))
    print(f"\nSaved selected wavelengths, plots and model in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
