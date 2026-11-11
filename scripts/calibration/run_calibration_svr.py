import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import train_and_save_model_svr
from src.multical.models.svr import SVRModel

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
RESULTS_DIR = "results_svr"              # Directory where results and plots will be saved
model_name = "model_calibration_svr.pkl"  # Filename for the saved model (must end with .pkl)


# --- 2. Data Files ---
# List of (Concentration_File, Absorbance_File)
# Ensure these are in your 'data/' folder
DATA_FILES = [
    ('data/splits/exp4_refe_cal.txt', 'data/splits/exp4_nonda_cal.txt'),
    ('data/splits/exp5_refe_cal.txt', 'data/splits/exp5_nonda_cal.txt'),
    ('data/splits/exp6_refe_cal.txt', 'data/splits/exp6_nonda_cal.txt'),
    ('data/splits/exp7_refe_cal.txt', 'data/splits/exp7_nonda_cal.txt'),
    ('data/splits/exp8_refe_cal.txt', 'data/splits/exp8_nonda_cal.txt'),
    ('data/splits/exp9_refe_cal.txt', 'data/splits/exp9_nonda_cal.txt')
]


# --- 3. Model / Analyte Settings ---
ANALYTES = ['cb', 'gl', 'xy']        # Names of the analytes
COLORS = ['green', 'red', 'purple']  # Plot colors for each analyte
UNITS = 'g/L'                        # Unit of measurement

# --- 3b. SVR Hyper-parameter Grid ---
# DEFAULT = robust pipeline: kernel 'rbf' only.
#
# Rationale (from the LOExO cross-batch validation, see doe_plan/): the 'poly'
# kernel extrapolates explosively when predicting a new batch (catastrophic
# errors), while 'rbf' degrades gracefully. So 'rbf' is the default here.
# To run an exhaustive multi-kernel search instead, set
#   SVR_KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
# Per kernel, only the relevant parameters are swept:
#   linear  -> C, epsilon
#   rbf     -> C, epsilon, gamma            (default)
#   poly    -> C, epsilon, gamma, degree, coef0
#   sigmoid -> C, epsilon, gamma, coef0
TARGET_ANALYTE = 'gl'   # Analyte whose best model is highlighted in the report

SVR_KERNELS = ['rbf']
SVR_GRID = {
    'C':       [0.1, 1, 10, 100, 1000, 10000],
    'epsilon': [0.001, 0.01, 0.05, 0.1, 0.5],
    'gamma':   ['scale', 'auto', 1e-4, 1e-3, 1e-2, 1e-1],
    'degree':  [2, 3, 4],          # used only if 'poly' is re-enabled
    'coef0':   [0.0, 1.0, 10.0],   # used only if 'poly'/'sigmoid' is re-enabled
}

# --- 4. Cross-Validation Settings ---
K_FOLDS = 5               # Number of folds for CV
CV_TYPE = 'venetian'      # Type of CV: 'random', 'consecutive', 'venetian'

# --- 5. Pretreatment Pipeline ---
# Text-based pipeline definition.
# [Operation, Param1, Param2, ...]
PRETREATMENT = [
    ['Cut', 4400, 7500, 1],  # Cut spectral region (Min, Max, Plot?)
    ['SG', 7, 2, 1, 1],      # Savitzky-Golay: radius=7, Poly=2, Deriv=1
    ['SNV', 1],              # Standard Normal Variate (scatter correction; batch-robust)
]

# --- 6. Publication Plot Settings ---
PLOT_PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Calibri'],
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.loc': 'best',
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png',
}

# =============================================================================
#                              HELPERS
# =============================================================================

def load_data(files):
    """
    Loads concentration and absorbance files into single matrices.
    (Identical loader to run_calibration.py so PLS/SVR are directly comparable.)
    """
    x_list, absor_list, wavelengths = [], [], None

    print("Loading data...")
    for x_f, abs_f in files:
        if not (os.path.exists(x_f) and os.path.exists(abs_f)):
            print(f"  [Skipping] File not found: {x_f} or {abs_f}")
            continue

        print(f"  - {x_f} / {abs_f}")
        try:
            # Load Concentration
            xi = np.loadtxt(x_f)
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:]  # Drop time col if present
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            # Load Spectra (with header parsing)
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])

            absi = np.loadtxt(abs_f, skiprows=1)
            absi_data = absi[:, 1:]  # Drop time column (col 0)

            if wavelengths is None:
                wavelengths = wl_curr
            elif len(wavelengths) == len(wl_curr) and not np.allclose(wavelengths, wl_curr, atol=1e-1):
                print(f"Warning: Wavelength mismatch in {abs_f}")

            x_list.append(xi)
            absor_list.append(absi_data)
        except Exception as e:
            print(f"Error loading {x_f}: {e}")

    if not x_list:
        raise FileNotFoundError("No valid data loaded. Check file paths.")

    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    wavelengths = np.asarray(wavelengths)

    return x0, wavelengths, absor_data


def build_param_combos(grid, kernels):
    """
    Builds the list of SVR parameter dicts to evaluate.

    Each kernel only varies the parameters that affect it, so no redundant
    combinations are produced (e.g. 'linear' does not sweep gamma/degree/coef0).
    """
    combos = []
    for kernel in kernels:
        if kernel == 'linear':
            for C, eps in product(grid['C'], grid['epsilon']):
                combos.append({'kernel': 'linear', 'C': C, 'epsilon': eps})
        elif kernel == 'rbf':
            for C, eps, g in product(grid['C'], grid['epsilon'], grid['gamma']):
                combos.append({'kernel': 'rbf', 'C': C, 'epsilon': eps, 'gamma': g})
        elif kernel == 'poly':
            for C, eps, g, d, c0 in product(grid['C'], grid['epsilon'], grid['gamma'],
                                            grid['degree'], grid['coef0']):
                combos.append({'kernel': 'poly', 'C': C, 'epsilon': eps,
                               'gamma': g, 'degree': d, 'coef0': c0})
        elif kernel == 'sigmoid':
            for C, eps, g, c0 in product(grid['C'], grid['epsilon'], grid['gamma'], grid['coef0']):
                combos.append({'kernel': 'sigmoid', 'C': C, 'epsilon': eps, 'gamma': g, 'coef0': c0})
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    return combos


def _params_label(p, skip=('C',)):
    """Compact human-readable label for a parameter dict, skipping given keys."""
    order = ['kernel', 'C', 'epsilon', 'gamma', 'degree', 'coef0']
    parts = [f"{k}={p[k]}" for k in order if k in p and k not in skip]
    return ", ".join(parts)


def _fold_indices(n_samples, folds, cv_type, perm):
    """Yields (train_idx, val_idx) for each fold, matching run_calibration.py logic."""
    fold_size = int(np.ceil(n_samples / folds))
    for i in range(folds):
        if cv_type == 'venetian':
            val_idx = np.arange(i, n_samples, folds)
        else:
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            val_idx_raw = np.arange(start, end)
            val_idx_raw = val_idx_raw[val_idx_raw < n_samples]
            val_idx = perm[val_idx_raw]

        if len(val_idx) == 0:
            continue

        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        train_idx = np.arange(n_samples)[mask]
        yield train_idx, val_idx


def svr_cv_predict_single(absor, y, params, folds, cv_type='venetian', perm=None):
    """
    Cross-validated prediction for a single analyte with fixed hyper-parameters.

    Returns an (n_samples,) vector of out-of-fold predictions in real units.
    SVRModel internally StandardScales both X and y, so spectra are passed
    pretreated-but-unscaled.
    """
    n_samples = absor.shape[0]
    if perm is None:
        perm = np.arange(n_samples)

    y_pred_cv = np.zeros(n_samples)

    for train_idx, val_idx in _fold_indices(n_samples, folds, cv_type, perm):
        X_tr = absor[train_idx, :]
        X_val = absor[val_idx, :]
        y_tr = y[train_idx].reshape(-1, 1)

        model = SVRModel()
        y_hat = model.fit_predict(
            X_tr, y_tr, X_val,
            C=params['C'], epsilon=params['epsilon'],
            gamma=params.get('gamma', 'scale'), kernel=params.get('kernel', 'rbf'),
            degree=params.get('degree', 3), coef0=params.get('coef0', 0.0)
        )
        y_pred_cv[val_idx] = y_hat.flatten()

    return y_pred_cv


def grid_search_svr(absor, x0, combos, folds, cv_type, analytes, perm=None):
    """
    Per-analyte exhaustive search over the SVR hyper-parameter combinations
    (built by build_param_combos) using cross-validation.

    Returns:
        best_params : list (per analyte) of the winning param dict.
        rmsecv_best : (n_analytes,) RMSECV of the winning model.
        all_results : list (per analyte) of (params, rmsecv) tuples for every combo.
    """
    n_analytes = x0.shape[1]
    best_params = []
    rmsecv_best = np.zeros(n_analytes)
    all_results = []

    print(f"\n--- SVR Search ({len(combos)} combinations x {folds} folds per analyte) ---")
    for j in range(n_analytes):
        y_j = x0[:, j]
        best_rmse = np.inf
        best_p = None
        results_j = []

        for p in combos:
            y_cv = svr_cv_predict_single(absor, y_j, p, folds, cv_type, perm=perm)
            rmse = np.sqrt(np.mean((y_j - y_cv) ** 2))
            results_j.append((p, rmse))
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = p

        best_params.append(best_p)
        rmsecv_best[j] = best_rmse
        all_results.append(results_j)
        print(f"  {analytes[j]:>4}: best RMSECV={best_rmse:.5f} {UNITS}  ->  {_params_label(best_p, skip=())}")

    return best_params, rmsecv_best, all_results


# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def main():
    # --- Setup Styling ---
    plt.rcParams['figure.max_open_warning'] = 100
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load Data ---
    x0, wavelengths, absor_raw = load_data(DATA_FILES)
    nc = len(ANALYTES)
    print(f"Total samples: {x0.shape[0]}, Wavelengths: {len(wavelengths)}")
    if x0.shape[1] != nc:
        raise ValueError(f"ANALYTES ({nc}) does not match concentration columns ({x0.shape[1]}).")

    # --- Plot Raw vs Pretreated ---
    print("\nGenerating Spectra Plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, absor_raw.T)
    ax.set_title("a) Raw Spectra")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Absorbance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "Spectra_Raw.png"))
    plt.close(fig)

    absor_pre, wl_pre = apply_pretreatment(PRETREATMENT, absor_raw.copy(), wavelengths.copy(), plot=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(wl_pre, absor_pre.T)
    ax2.set_title("b) Pretreated Spectra")
    ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax2.set_ylabel("Absorbance")
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "Spectra_Pretreated.png"))
    plt.close(fig2)

    # --- Cross-Validated Hyper-parameter Search ---
    print("\nRunning SVR Calibration...")
    # Fixed permutation for 'random' CV so the grid search and final plots are consistent.
    perm = np.random.permutation(x0.shape[0]) if CV_TYPE == 'random' else np.arange(x0.shape[0])

    combos = build_param_combos(SVR_GRID, SVR_KERNELS)
    best_params, rmsecv_best, all_results = grid_search_svr(
        absor_pre, x0, combos, K_FOLDS, CV_TYPE, ANALYTES, perm=perm
    )

    # --- Final CV & Calibration predictions (with best params) ---
    y_pred_cv = np.zeros_like(x0, dtype=float)
    y_pred_cal = np.zeros_like(x0, dtype=float)
    for j in range(nc):
        y_pred_cv[:, j] = svr_cv_predict_single(absor_pre, x0[:, j], best_params[j], K_FOLDS, CV_TYPE, perm=perm)
        # Calibration fit: train and predict on the full set
        bp = best_params[j]
        m = SVRModel()
        y_pred_cal[:, j] = m.fit_predict(
            absor_pre, x0[:, j].reshape(-1, 1), absor_pre,
            C=bp['C'], epsilon=bp['epsilon'],
            gamma=bp.get('gamma', 'scale'), kernel=bp.get('kernel', 'rbf'),
            degree=bp.get('degree', 3), coef0=bp.get('coef0', 0.0)
        ).flatten()

    # --- Metrics ---
    rmsecal = np.sqrt(np.mean((x0 - y_pred_cal) ** 2, axis=0))
    sst = np.sum((x0 - x0.mean(axis=0)) ** 2, axis=0)
    sst[sst == 0] = 1.0
    r2cv = 1 - np.sum((x0 - y_pred_cv) ** 2, axis=0) / sst
    r2cal = 1 - np.sum((x0 - y_pred_cal) ** 2, axis=0) / sst

    print(f"\n{'='*64}")
    print("FINAL RESULTS REPORT (SVR)".center(64))
    print(f"{'='*64}")
    header = f"{'Analyte':<10}|{'RMSECV':^12}|{'RMSEC':^12}|{'R2_CV':^10}|{'R2_Cal':^10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for j in range(nc):
        print(f"{ANALYTES[j]:<10}|{rmsecv_best[j]:^12.5f}|{rmsecal[j]:^12.5f}|{r2cv[j]:^10.4f}|{r2cal[j]:^10.4f}")
    print("-" * len(header))

    # --- Highlight the target analyte ---
    if TARGET_ANALYTE in ANALYTES:
        jt = ANALYTES.index(TARGET_ANALYTE)
        print(f"\n>> Best model for target analyte '{TARGET_ANALYTE}':")
        print(f"   {_params_label(best_params[jt], skip=())}")
        print(f"   RMSECV={rmsecv_best[jt]:.5f} {UNITS} | R2_CV={r2cv[jt]:.4f}")

    # --- Save per-analyte RMSECV / metrics to text ---
    metrics = np.column_stack([rmsecv_best, rmsecal, r2cv, r2cal])
    np.savetxt(
        os.path.join(RESULTS_DIR, "SVR_metrics.txt"), metrics,
        header="RMSECV\tRMSEC\tR2_CV\tR2_Cal\n" + "\t".join(ANALYTES),
        fmt='%g', delimiter='\t'
    )

    # --- Plot: RMSECV vs C (analog of "RMSE vs Latent Variables") ---
    fig_c, axes_c = plt.subplots(1, nc, figsize=(6 * nc, 5), squeeze=False)
    axes_c = axes_c.flatten()
    for j in range(nc):
        bp = best_params[j]
        # Curve over the C grid, holding every other winning parameter fixed.
        c_vals, rmse_vals = [], []
        for (p, rmse) in all_results[j]:
            if all(p.get(k) == v for k, v in bp.items() if k != 'C'):
                c_vals.append(p['C'])
                rmse_vals.append(rmse)
        order = np.argsort(c_vals)
        c_vals = np.array(c_vals)[order]
        rmse_vals = np.array(rmse_vals)[order]

        ax = axes_c[j]
        col = COLORS[j] if j < len(COLORS) else 'blue'
        ax.semilogx(c_vals, rmse_vals, color=col, marker='o', linestyle='-', label='RMSECV')
        ax.semilogx([bp['C']], [rmsecv_best[j]], 'k*', markersize=15,
                    label=f"Best (C={bp['C']})")
        ax.set_xlabel('C (regularization)')
        ax.set_ylabel(f'RMSECV ({UNITS})')
        ax.set_title(f"{ANALYTES[j]} [{_params_label(bp, skip=('C',))}]")
        ax.legend()
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(RESULTS_DIR, 'RMSECV_vs_C.png'), dpi=300, bbox_inches='tight')

    # --- Plot: Predicted vs Measured (Cal + CV), publication style ---
    plt.rcParams.update(PLOT_PARAMS)
    fig_pred, axes_pred = plt.subplots(1, nc, figsize=(6 * nc, 5), squeeze=False)
    axes_pred = axes_pred.flatten()
    for j in range(nc):
        ax = axes_pred[j]
        col = COLORS[j] if j < len(COLORS) else 'blue'
        y_meas = x0[:, j]
        y_cal = y_pred_cal[:, j]
        y_cv = y_pred_cv[:, j]

        ax.scatter(y_meas, y_cal, marker='o', facecolors='none', alpha=0.6,
                   label='Calibration', edgecolors=col)
        ax.scatter(y_meas, y_cv, c=col, marker='x', alpha=0.8, label='Cross-Validation')

        min_val = min(y_meas.min(), y_cal.min(), y_cv.min())
        max_val = max(y_meas.max(), y_cal.max(), y_cv.max())
        buff = (max_val - min_val) * 0.05
        ax.plot([min_val - buff, max_val + buff], [min_val - buff, max_val + buff], 'k--', alpha=0.5)

        stats_text = (
            f"Cal: $R^2$={r2cal[j]:.3f}, RMSE={rmsecal[j]:.3f}\n"
            f"CV: $R^2$={r2cv[j]:.3f}, RMSE={rmsecv_best[j]:.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

        letter = chr(97 + j)
        ax.set_title(f'{letter}) {ANALYTES[j]} - Predicted vs Measured', loc='left')
        ax.set_ylabel(f'Predicted {UNITS}')
        ax.set_xlabel(f'Measured {UNITS}')
        ax.legend()

    fig_pred.tight_layout()
    fig_pred.savefig(os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Pub.png'))
    print(f"Saved Prediction Plot to: {os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Pub.png')}")

    # --- Save trained model ---
    print("\n--- Saving Fully Trained SVR Model ---")
    train_and_save_model_svr(
        absor_pre, x0, wl_pre, best_params,
        os.path.join(RESULTS_DIR, model_name), rmsecv_list=list(rmsecv_best)
    )

    print("\nProcessing complete. Close plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
