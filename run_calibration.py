import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import train_and_save_model_pls
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
RESULTS_DIR = "results"         # Directory where results and plots will be saved
model_name = "model_calibration.pkl" # Filename for the saved model (must end with .pkl)


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


# --- 3. Model Parameters ---
MODEL_TYPE = 1          # 1 = PLS (Partial Least Squares)
                        # 2 = SPA (Successive Projections Algorithm)
                        # 3 = PCR (Principal Component Regression)

MAX_LATENT_VARS = 15    # Maximum number of Latent Variables (Factors) to test
ANALYTES = ['cb', 'gl', 'xy']  # Names of the analytes (e.g. ['Glucose', 'xylose'])
COLORS = ['green', 'red', 'purple'] # Plot colors for each analyte
UNITS = 'g/L'           # Unit of measurement

# SPA/PCR Specifics (Advanced)
SPA_OPT_K_INI = 2       # 0=> lini = lambda(1); 1=> lini = below; 2=> optimize lini.
SPA_L_INI = 0           # [cm-1] Initial wavenumber (only if optkini=1)

# --- 4. Cross-Validation & Validation Settings ---
# Test Set Split
TEST_SPLIT = 0.0          # Fraction of data to keep as a pure Test Set (0.0 to 1.0)
MANUAL_TEST_SET = []      # Manual test set (X_test, Y_test) if available, else []

# Cross-Validation
VALIDATION_MODE = 'kfold' # 'kfold' or 'Val' (Holdout)

# If 'kfold':
K_FOLDS = 5               # Number of folds for CV
CV_TYPE = 'venetian'      # Type of CV: 'random', 'consecutive', 'venetian'

# If 'Val' (Holdout):
VAL_FRACTION = 0.20       # Fraction for validation holdout

# --- 5. Pretreatment Pipeline ---
# Text-based pipeline definition.
# [Operation, Param1, Param2, ...]
PRETREATMENT = [
    ['Cut', 4400, 7500, 1], # Cut spectral region (Min, Max, Plot?)
    ['SG', 7, 2, 1, 1],  # Savitzky-Golay: radius=7, Poly=2, Deriv=1

]

# --- 6. Analysis Settings ---
EVALUATE_OUTLIERS = True
OUTLIER_CONF_LEVEL = 0.95
REMOVE_OUTLIERS = False

OUTLIER_REMOVAL = 0     # (Deprecated legacy setting) 0 = Off, 1 = On (Student t-test on residuals)
USE_F_TEST = True       # Use Osten F-test for automatic model selection (Optimal k)
PRE_ANALYSIS = [['LB'], ['PCA']] # Analyses to run before calibration

# --- 7. Publication Plot Settings ---
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
#                              MAIN EXECUTION
# =============================================================================

def load_data(files):
    """
    Loads concentration and absorbance files into single matrices.
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
            # Handle dimensions
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:] # Drop time col if present
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            # Load Spectra (with header parsing)
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])
            
            # Load data skipping header
            absi = np.loadtxt(abs_f, skiprows=1)
            absi_data = absi[:, 1:] # Drop time column (col 0)

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
    
    # Engine expects wavelengths as the first row of absorbance matrix
    absor0 = np.vstack([wavelengths, absor_data]) 
    
    return x0, absor0, wavelengths, absor_data


def predict_pls2_cv(absor, x0, max_k, folds, cv_type='venetian'):
    n_samples = absor.shape[0]
    n_analytes = x0.shape[1]

    y_pred_cv = np.zeros((n_samples, n_analytes, max_k))
    pls_engine = PLS()

    indices = np.arange(n_samples)
    if cv_type == 'random':
        indices = np.random.permutation(n_samples)

    fold_size = int(np.ceil(n_samples / folds))

    for i in range(folds):
        if cv_type == 'venetian':
            val_idx = np.arange(i, n_samples, folds)
        else:
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            val_idx_raw = np.arange(start, end)
            val_idx_raw = val_idx_raw[val_idx_raw < n_samples]
            val_idx = indices[val_idx_raw]

        if len(val_idx) == 0:
            continue

        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        train_idx = np.arange(n_samples)[mask]

        x_train_raw = absor[train_idx, :]
        x_val_raw = absor[val_idx, :]
        y_train_raw = x0[train_idx, :]

        combined_x = np.vstack([x_train_raw, x_val_raw])
        combined_x_norm, _, _ = zscore_matlab_style(combined_x)
        n_tr = len(train_idx)
        x_train = combined_x_norm[:n_tr, :]
        x_val = combined_x_norm[n_tr:, :]

        y_train_norm, ymed_y, ysig_y = zscore_matlab_style(y_train_raw)

        _, _, p, _, q, w, _, _ = pls_engine.nipals(x_train, y_train_norm, max_k)

        for k in range(1, max_k + 1):
            wk = w[:, :k]
            pk = p[:, :k]
            qk = q[:, :k]
            pw = pk.T @ wk
            pw_inv = np.linalg.pinv(pw)
            beta_k = wk @ pw_inv @ qk.T

            ytp_norm = x_val @ beta_k
            ytp = ytp_norm * ysig_y + ymed_y

            y_pred_cv[val_idx, :, k - 1] = ytp

    return y_pred_cv


def predict_pls2_cal(absor, x0, max_k):
    n_samples = absor.shape[0]
    n_analytes = x0.shape[1]
    y_pred_cal = np.zeros((n_samples, n_analytes, max_k))
    pls_engine = PLS()

    x_norm, _, _ = zscore_matlab_style(absor)
    y_norm, ymed_y, ysig_y = zscore_matlab_style(x0)

    _, _, p, _, q, w, _, _ = pls_engine.nipals(x_norm, y_norm, max_k)

    for k in range(1, max_k + 1):
        wk = w[:, :k]
        pk = p[:, :k]
        qk = q[:, :k]
        pw = pk.T @ wk
        pw_inv = np.linalg.pinv(pw)
        beta_k = wk @ pw_inv @ qk.T

        yp_norm = x_norm @ beta_k
        yp = yp_norm * ysig_y + ymed_y

        y_pred_cal[:, :, k - 1] = yp

    return y_pred_cal

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
    x0, absor0, wavelengths, absor_raw = load_data(DATA_FILES)
    print(f"Total samples: {x0.shape[0]}, Wavelengths: {len(wavelengths)}")

    # --- Plot Raw vs Pretreated ---
    print("\nGenerating Plots...")
    
    # 1. Raw
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, absor_raw.T)
    ax.set_title("a) Raw Spectra")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Absorbance")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "Spectra_Raw.png"))
    plt.close(fig)
    
    # 2. Pretreated (Visualization only)
    absor_pre, wl_pre = apply_pretreatment(PRETREATMENT, absor_raw.copy(), wavelengths.copy(), plot=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(wl_pre, absor_pre.T)
    ax2.set_title("b) Pretreated Spectra")
    ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax2.set_ylabel("Absorbance")
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "Spectra_Pretreated.png"))
    # plt.close(fig2) # Keep open for user inspection

    # --- Run Multical Engine ---
    print("\nRunning Calibration Engine...")
    
    # Construct OptimModel
    if VALIDATION_MODE == 'kfold':
        OptimModel = ['kfold', K_FOLDS, CV_TYPE]
    elif VALIDATION_MODE == 'Val':
        OptimModel = ['Val', VAL_FRACTION]
    else:
        OptimModel = ['kfold', 5, 'venetian']

    nc = len(ANALYTES)
            
    engine = MulticalEngine()
    
    # Run
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc, R2CV, R2cal, best_k_dict = engine.run(
        MODEL_TYPE, SPA_OPT_K_INI, SPA_L_INI, MAX_LATENT_VARS, nc, ANALYTES, UNITS, 
        x0, absor0, TEST_SPLIT, MANUAL_TEST_SET, OptimModel, PRETREATMENT, 
        analysis_list=PRE_ANALYSIS, output_dir=RESULTS_DIR, outlier=OUTLIER_REMOVAL, 
        use_ftest=USE_F_TEST, colors=COLORS
    )

    # --- Outlier Evaluation Post-Calibration ---
    if EVALUATE_OUTLIERS:
        from src.multical.core.outliers import evaluate_and_plot_outliers
        absor_pre, wl_pre = apply_pretreatment(PRETREATMENT, absor0[1:], absor0[0], plot=False)
        absor_test = np.vstack([wl_pre, absor_pre])
        
        best_k_final = []
        for j in range(nc):
            if isinstance(best_k_dict, dict) and j in best_k_dict:
                k_sel = best_k_dict[j]
            else:
                k_sel = np.argmin(RMSECV_conc[:, j]) + 1
            best_k_final.append(k_sel)

        evaluate_and_plot_outliers(
            x0, absor_test, ANALYTES, best_k_final, RESULTS_DIR, COLORS, 
            conf_level=OUTLIER_CONF_LEVEL
        )

    # --- Helper for cleaner output ---
    def print_metric_table(title, matrix, names):
        if matrix is None: return
        print(f"\n> {title}")
        header_str = f"{'LV':<4} |" + "".join([f"{name:^12} |" for name in names])
        div_line = "-" * len(header_str)
        print(div_line)
        print(header_str)
        print(div_line)
        for k, row in enumerate(matrix):
            row_str = f"{k+1:<4} |" + "".join([f"{val:^12.5f} |" for val in row])
            print(row_str)
        print(div_line)


    if RMSECV is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS REPORT".center(60))
        print(f"{'='*60}")
        print_metric_table(f"RMSECV ({UNITS})", RMSECV_conc, ANALYTES)
        print_metric_table("R-Squared (CV)", R2CV, ANALYTES)

    # --- SAVE TRAINED MODEL ---
    print("\n--- Saving Fully Trained Model ---")
  
    
    # 1. Re-apply Pretreatment to get final spectral matrix
    # Note: engine.run does this internally but doesn't return the pretreated matrix.
    # We must do it here to ensure the saved model uses pretreated data.
    absor_pre_final, wl_final = apply_pretreatment(PRETREATMENT, absor_raw.copy(), wavelengths.copy(), plot=False)
    
    # 2. Get Optimal Latent Variables (k)
    # best_k_dict is returned by engine.run. It is a list [k_analyte1, k_analyte2...]
    # If the user wants to override, they can modify this list or the code below.
    best_k_list = best_k_dict 
    print(f"Optimal LVs detected: {best_k_list}")

    train_and_save_model_pls(absor_pre_final, x0, wl_final, best_k_list, os.path.join(RESULTS_DIR, model_name), rmsecv_list=RMSECV_conc)

    # --- Predicted_vs_Measured_Pub (same style as variable selection script) ---
    if MODEL_TYPE == 1 and VALIDATION_MODE == 'kfold':
        plt.rcParams.update(PLOT_PARAMS)

        y_pred_cv_all = predict_pls2_cv(absor_pre_final, x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
        y_pred_cal_all = predict_pls2_cal(absor_pre_final, x0, MAX_LATENT_VARS)

        best_k_final = []
        for j in range(nc):
            if isinstance(best_k_dict, dict) and j in best_k_dict:
                k_sel = best_k_dict[j]
            else:
                k_sel = np.argmin(RMSECV_conc[:, j]) + 1
            best_k_final.append(k_sel)

        # Match run_variable_selection.py behavior: prioritize Glucose and Xylose.
        plot_indices = []
        plot_names = []
        if len(ANALYTES) > 1:
            if 'gl' in ANALYTES:
                idx = ANALYTES.index('gl')
                plot_indices.append(idx)
                plot_names.append('Glucose')
            if 'xy' in ANALYTES:
                idx = ANALYTES.index('xy')
                plot_indices.append(idx)
                plot_names.append('Xylose')
        if not plot_indices:
            plot_indices = [1, 2] if len(ANALYTES) >= 3 else list(range(len(ANALYTES)))
            plot_names = [ANALYTES[i] for i in plot_indices]

        fig_pred, axes_pred = plt.subplots(len(plot_indices), 1, figsize=(6, 5 * len(plot_indices)))
        if len(plot_indices) == 1:
            axes_pred = [axes_pred]

        for i, (idx, name) in enumerate(zip(plot_indices, plot_names)):
            ax = axes_pred[i]
            analyte_color = COLORS[idx] if idx < len(COLORS) else 'blue'
            k_sel = best_k_final[idx]

            y_meas = x0[:, idx]
            y_cal = y_pred_cal_all[:, idx, k_sel - 1]
            y_cv = y_pred_cv_all[:, idx, k_sel - 1]

            ax.scatter(y_meas, y_cal, c=analyte_color, marker='o', facecolors='none', alpha=0.6, label='Calibration', edgecolors='black')
            ax.scatter(y_meas, y_cv, c=analyte_color, marker='x', alpha=0.8, label='Cross-Validation')

            min_val = min(y_meas.min(), y_cal.min(), y_cv.min())
            max_val = max(y_meas.max(), y_cal.max(), y_cv.max())
            buff = (max_val - min_val) * 0.05
            ax.plot([min_val - buff, max_val + buff], [min_val - buff, max_val + buff], 'k--', alpha=0.5)

            sse_cal = np.sum((y_meas - y_cal) ** 2)
            sst = np.sum((y_meas - np.mean(y_meas)) ** 2)
            r2_cal = 1 - sse_cal / sst
            rmse_cal = np.sqrt(np.mean((y_meas - y_cal) ** 2))

            sse_cv = np.sum((y_meas - y_cv) ** 2)
            r2_cv = 1 - sse_cv / sst
            rmse_cv = np.sqrt(np.mean((y_meas - y_cv) ** 2))

            stats_text = (
                f"Cal: $R^2$={r2_cal:.3f}, RMSE={rmse_cal:.3f}\n"
                f"CV: $R^2$={r2_cv:.3f}, RMSE={rmse_cv:.3f}"
            )

            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

            letter = chr(97 + i)
            ax.set_title(f'{letter}) {name} - Predicted vs Measured (LVs={k_sel})', loc='left')
            ax.set_ylabel(f'Predicted {UNITS}')
            ax.set_xlabel(f'Measured {UNITS}')
            ax.legend()

        fig_pred.tight_layout()
        fig_pred.savefig(os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Pub.png'))
        print(f"Saved Prediction Plot to: {os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Pub.png')}")
    elif MODEL_TYPE != 1:
        print("Skipping Predicted_vs_Measured_Pub: only implemented for PLS (MODEL_TYPE=1).")
    else:
        print("Skipping Predicted_vs_Measured_Pub: only implemented for kfold validation.")
    
    # Keep plots open at the end
    print("\nProcessing complete. Close plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
