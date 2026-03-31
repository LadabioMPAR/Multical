import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.var_selection import run_pso, run_simulated_annealing, calculate_vip, calculate_rmsecv_fast, select_k_ftest
from src.multical.utils import zscore_matlab_style

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. Selection Method ---
SELECTION_METHOD = 'VIP' # Options: 'VIP', 'SA' (Simulated Annealing), 'PSO' (Particle Swarm)

# --- 2. Output & Data ---
RESULTS_DIR = "results_var_selection"

MODEL_NAME = "model_variable_selection.pkl" # Filename for the saved model (must end with .pkl)
DATA_FILES = [
    #('data/exp4_refe.txt', 'data/exp4_nonda.txt'),
    ('data/exp5_refe.txt', 'data/exp5_nonda.txt'),
    #('data/exp6_refe.txt', 'data/exp6_nonda.txt'),
    #('data/exp7_refe.txt', 'data/exp7_nonda.txt'),  
]

# --- 3. Model Parameters ---
MODEL_TYPE = 1          # 1 = PLS (Required for variable selection)
MAX_LATENT_VARS = 15    # Maximum Latent Variables
ANALYTES = ['cb', 'gl', 'xy']
UNITS = 'g/L'
COLORS = ['green', 'red', 'purple']

# Cross-Validation Settings
K_FOLDS = 5
CV_TYPE = 'venetian'

# --- 4. Optimization Parameters ---

# A) VIP Settings
VIP_THRESHOLDS = np.arange(0.1, 1.5, 0.1) 

# B) Simulated Annealing (SA) Settings
SA_PARAMS = {
    'max_iter': 3000,     # Maximum iterations
    'initial_temp': 0.7, # Initial temperature
    'alpha': 0.92        # Cooling rate
}

# C) PSO Settings
PSO_PARAMS = {
    'n_particles': 100,
    'max_iter': 1000,
    'w': 0.9,   # Inertia weight
    'c1': 1.49, # Cognitive weight
    'c2': 1.49  # Social weight
}

# --- 5. Pretreatment Pipeline ---
# Applied BEFORE selection logic
PRETREATMENT = [
    ['Cut', 4400, 7500, 1],
    ['SG', 7, 2, 1, 1],  # Savitzky-Golay: Window=7, Poly=2, Deriv=1
]

# --- 6. Plot Settings (Publication Quality) ---
PLOT_PARAMS = {
    # Font Configuration
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Calibri'],
    'font.size': 16,            # Base Text Size
    
    # Axes
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.linewidth': 1.5,      # Edge width
    
    # Ticks
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,

    # Legend
    'legend.fontsize': 12,
    'legend.frameon': True,     # Box around legend?
    'legend.loc': 'best',
    
    # Lines & Markers
    'lines.linewidth': 2,
    'lines.markersize': 8,
    
    # Saving
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'
}

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def load_data(files):
    x_list, absor_list, wavelengths = [], [], None
    print("Loading data...")
    for x_f, abs_f in files:
        if not (os.path.exists(x_f) and os.path.exists(abs_f)):
            print(f"Skipping {x_f}: Not found.")
            continue
            
        try:
            xi = np.loadtxt(x_f)
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:]
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])
            absi = np.loadtxt(abs_f, skiprows=1)[:, 1:]

            if wavelengths is None: 
                wavelengths = wl_curr
            elif len(wavelengths) == len(wl_curr) and not np.allclose(wavelengths, wl_curr, atol=1e-1):
                print(f"Warning: Wavelength mismatch.")

            x_list.append(xi)
            absor_list.append(absi)
        except Exception as e:
            print(f"Error loading {x_f}: {e}")

    if not x_list: return None, None, None
    
    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    return x0, absor_data, wavelengths

def predict_pls2_cv(absor, x0, max_k, folds, cv_type='venetian'):
    n_samples = absor.shape[0]
    n_analytes = x0.shape[1]
    
    Y_pred_cv = np.zeros((n_samples, n_analytes, max_k))
    pls_engine = PLS()
    
    indices = np.arange(n_samples)
    if cv_type == 'random':
        indices = np.random.permutation(n_samples)
    
    fold_size = int(np.ceil(n_samples / folds))
    
    for i in range(folds):
        if cv_type == 'venetian':
             val_idx = np.arange(i, n_samples, folds)
        else:
             # Random/Consecutive logic
             start = i * fold_size
             end = min((i + 1) * fold_size, n_samples)
             val_idx_raw = np.arange(start, end)
             val_idx_raw = val_idx_raw[val_idx_raw < n_samples]
             val_idx = indices[val_idx_raw]

        if len(val_idx) == 0: continue
            
        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        train_idx = np.arange(n_samples)[mask]
        
        X_train_raw = absor[train_idx, :]
        X_val_raw = absor[val_idx, :]
        Y_train_raw = x0[train_idx, :]
        
        Combined_X = np.vstack([X_train_raw, X_val_raw])
        Combined_X_norm, Xmed, Xsig = zscore_matlab_style(Combined_X)
        n_tr = len(train_idx)
        X_train = Combined_X_norm[:n_tr, :]
        X_val = Combined_X_norm[n_tr:, :]
        
        Y_train_norm, Ymed_y, Ysig_y = zscore_matlab_style(Y_train_raw)

        _, _, P, _, Q, W, _, _ = pls_engine.nipals(X_train, Y_train_norm, max_k)
        
        for k in range(1, max_k + 1):
             wk = W[:, :k]
             pk = P[:, :k]
             qk = Q[:, :k]
             pw = pk.T @ wk
             pw_inv = np.linalg.pinv(pw)
             Beta_k = wk @ pw_inv @ qk.T
             
             Ytp_norm = X_val @ Beta_k
             Ytp = Ytp_norm * Ysig_y + Ymed_y # Back to raw units
             
             Y_pred_cv[val_idx, :, k-1] = Ytp
             
    return Y_pred_cv

def predict_pls2_cal(absor, x0, max_k):
    n_samples = absor.shape[0]
    n_analytes = x0.shape[1]
    Y_pred_cal = np.zeros((n_samples, n_analytes, max_k))
    pls_engine = PLS()
    
    X_norm, Xmed, Xsig = zscore_matlab_style(absor)
    Y_norm, Ymed_y, Ysig_y = zscore_matlab_style(x0)
    
    _, _, P, _, Q, W, _, _ = pls_engine.nipals(X_norm, Y_norm, max_k)
    
    for k in range(1, max_k + 1):
        wk = W[:, :k]
        pk = P[:, :k]
        qk = Q[:, :k]
        pw = pk.T @ wk
        pw_inv = np.linalg.pinv(pw)
        Beta_k = wk @ pw_inv @ qk.T

        Yp_norm = X_norm @ Beta_k
        Yp = Yp_norm * Ysig_y + Ymed_y
        
        Y_pred_cal[:, :, k-1] = Yp
        
    return Y_pred_cal


def main():
    # Use configurable plot settings
    plt.rcParams.update(PLOT_PARAMS)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    x0, absor_data, wavelengths = load_data(DATA_FILES)
    if x0 is None: return

    # 2. Pretreatment
    print("\n--- Applying Pretreatment ---")
    # We apply passing a copy as absor_raw
    absor_pre, wavelengths_pre = apply_pretreatment(PRETREATMENT, absor_data, wavelengths, output_dir=RESULTS_DIR, prefix="VS_")
    
    # 3. Variable Selection
    print(f"\n--- Running Variable Selection: {SELECTION_METHOD} ---")
    best_mask = np.ones(len(wavelengths_pre), dtype=bool) # Default is All

    if SELECTION_METHOD == 'VIP':
        # Step 1: Find optimal K with all variables
        print(" Determining optimal K (Full Spectrum)...")
        rmse_all, k_min, rmsecv_vec = calculate_rmsecv_fast(absor_pre, x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
        
        # Prefer F-test k_min if available, else plain min
        k_opt_ftest = select_k_ftest(rmsecv_vec, n_cal=x0.shape[0])
        print(f" Optimal K: {k_opt_ftest} (RMSE={rmse_all:.4f})")

        # Step 2: Calculate VIP Scores
        print(" Calculating VIP scores...")
        model_vip = PLS()
        vip_scores = calculate_vip(model_vip, absor_pre, x0, k_opt_ftest, wavelengths=wavelengths_pre, output_dir=RESULTS_DIR)
        
        # Step 3: Optimize Threshold
        print(" Optimizing VIP Threshold...")
        best_rmse = 1e9
        best_thresh = 0.0
        
        for th in VIP_THRESHOLDS:
            mask = vip_scores >= th
            n_sel = np.sum(mask)
            if n_sel < 2: continue # Too few variables
            
            rmse_sub, k_sub, _ = calculate_rmsecv_fast(absor_pre[:, mask], x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
            print(f"  Th={th:.1f}: {n_sel} variables | RMSECV: {rmse_sub:.4f} (k={k_sub})")
            
            if rmse_sub < best_rmse:
                best_rmse = rmse_sub
                best_thresh = th
                best_mask = mask
        
        print(f" Best Threshold: {best_thresh:.1f} (RMSE={best_rmse:.4f})")

    elif SELECTION_METHOD == 'SA':
        best_mask, _ = run_simulated_annealing(absor_pre, x0, MAX_LATENT_VARS, folds=K_FOLDS, cv_type=CV_TYPE, **SA_PARAMS)

    elif SELECTION_METHOD == 'PSO':
        best_mask, _ = run_pso(absor_pre, x0, MAX_LATENT_VARS, folds=K_FOLDS, cv_type=CV_TYPE, **PSO_PARAMS)

    else:
        print(" Unknown Method. Using all variables.")

    # 4. Visualization
    n_sel = np.sum(best_mask)
    print(f"\nSelected {n_sel} / {len(wavelengths_pre)} variables.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_spec = np.mean(absor_pre, axis=0)
    ax.plot(wavelengths_pre, mean_spec, 'k-', alpha=0.3, label='Mean Spectrum')
    ax.scatter(wavelengths_pre[best_mask], mean_spec[best_mask], c='r', s=5, label='Selected Variables')
    ax.set_title(f"Selected Variables ({SELECTION_METHOD})")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Absorbance")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Selected_Variables.png"))
    # plt.show() # Will show at the end

    # --- Save Selected Wavelengths ---
    wavelengths_selected = wavelengths_pre[best_mask]
    save_path = os.path.join(RESULTS_DIR, "selected_wavelengths.txt")
    np.savetxt(save_path, wavelengths_selected, fmt='%.4f', header="Selected Wavelengths (cm-1)")
    print(f"Saved selected wavelengths to: {save_path}")

    # 5. Final Calibration
    print("\n--- Running Final Calibration with Selected Variables ---")
    engine = MulticalEngine()
    
    absor_final = absor_pre[:, best_mask]
    wl_final = wavelengths_pre[best_mask]
    
    # Engine expects row 0 as wavelengths
    absor0_final = np.vstack([wl_final, absor_final])
    
    OptimModel = ['kfold', K_FOLDS, CV_TYPE]
    nc = len(ANALYTES)
    
    # Run engine with PRETREATED data (so pass empty list for pretreat pipeline)
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc, R2CV, R2cal, best_k_dict = engine.run(
        MODEL_TYPE, 2, 0, MAX_LATENT_VARS, nc, ANALYTES, UNITS, 
        x0, absor0_final, 0.0, [], OptimModel, pretreat_list=[], 
        analysis_list=[['LB'], ['PCA']], output_dir=RESULTS_DIR, colors=COLORS
    )

    if RMSECV is not None:
        print("\nFinal Model Generated.")

    # --- SAVE SELECTION MODEL ---
    from src.multical.core.saving import train_and_save_model_pls
    
    print("\n--- Saving Variable Selection Model ---")
    MODEL_FILENAME = MODEL_NAME
    
    # Ensure best_k_dict is converted to list and extract specific RMSECV values (Conc units)
    # best_k_dict is actually best_k_ftest which is a dict {idx: k}
    best_k_final = []
    best_rmsecv_final = []
    
    for j in range(nc):
        # Determine best K
        if j in best_k_dict:
             k_sel = best_k_dict[j]
        else:
             k_sel = np.argmin(RMSECV_conc[:, j]) + 1
        
        best_k_final.append(k_sel)
        
        # Get RMSECV (Concentration) at that K
        # RMSECV_conc is (kmax, nc)
        rmse_val = RMSECV_conc[k_sel-1, j]
        best_rmsecv_final.append(rmse_val)
        
        print(f"  Analyte {ANALYTES[j]}: K={k_sel}, RMSECV={rmse_val:.4f} {UNITS}")

    # Note: absor_final is already pretreated and reduced.
    # engine.run returns best_k_dict which is list of k
    train_and_save_model_pls(absor_final, x0, wl_final, best_k_final, os.path.join(RESULTS_DIR, MODEL_FILENAME), rmsecv_list=best_rmsecv_final)

    # --- START CUSTOM PLOTTING FOR PUBLICATION ---
    print("\n--- Generating Plots ---")
    
    # Generate Predictions for Plotting (PLS2 Model)
    Y_pred_cv_all = predict_pls2_cv(absor_final, x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
    Y_pred_cal_all = predict_pls2_cal(absor_final, x0, MAX_LATENT_VARS)
    
    # We only care about Glucose (idx 1) and Xylose (idx 2)
    # Check if indices exist (ANALYTES might change)
    plot_indices = []
    plot_names = []
    
    if len(ANALYTES) > 1:
        # Assuming cb, gl, xy order
        if 'gl' in ANALYTES:
             idx = ANALYTES.index('gl')
             plot_indices.append(idx)
             plot_names.append("Glucose")
        if 'xy' in ANALYTES:
             idx = ANALYTES.index('xy')
             plot_indices.append(idx)
             plot_names.append("Xylose")
             
    if not plot_indices:
        # Fallback if names differ
        plot_indices = [1, 2] if len(ANALYTES) >= 3 else list(range(len(ANALYTES)))
        plot_names = [ANALYTES[i] for i in plot_indices]

    # 1. RMSE Plot
    fig_rmse, axes_rmse = plt.subplots(len(plot_indices), 1, figsize=(6, 4*len(plot_indices)), sharex=True)
    if len(plot_indices) == 1: axes_rmse = [axes_rmse] # Make iterable
    
    for i, (idx, name) in enumerate(zip(plot_indices, plot_names)):
        ax = axes_rmse[i]
        analyte_color = COLORS[idx]
        
        # Calibration (Black), CV (Analyte Color)
        
        ax.plot(np.arange(1, MAX_LATENT_VARS + 1), RMSEcal_conc[:, idx], color=analyte_color, marker='o', mfc='white', label='Calibration')
        ax.plot(np.arange(1, MAX_LATENT_VARS + 1), RMSECV_conc[:, idx], color=analyte_color, marker='s', label='Cross-Validation')
        
        # Mark selected K
        k_sel = best_k_final[idx]
        rmse_sel = RMSECV_conc[k_sel-1, idx]
        ax.plot(k_sel, rmse_sel, color='black', marker='*', markersize=14, label=f'Selected ({k_sel})')
        
        ax.set_ylabel(f'RMSE ({UNITS})')
        letter = chr(97 + i)
        ax.set_title(f'{letter}) {name} - RMSE vs Latent Variables', loc='left')
        
        ax.legend()
            
    axes_rmse[-1].set_xlabel('Latent Variables')
    fig_rmse.tight_layout()
    fig_rmse.savefig(os.path.join(RESULTS_DIR, "RMSE_Calibration_CV_Pub.png"))
    print(f"Saved RMSE Plot to: {os.path.join(RESULTS_DIR, 'RMSE_Calibration_CV_Pub.png')}")
    
    # 2. Predicted vs Measured Plot
    fig_pred, axes_pred = plt.subplots(len(plot_indices), 1, figsize=(6, 5*len(plot_indices)))
    if len(plot_indices) == 1: axes_pred = [axes_pred]
    
    for i, (idx, name) in enumerate(zip(plot_indices, plot_names)):
        ax = axes_pred[i]
        analyte_color = COLORS[idx]
        k_sel = best_k_final[idx]
        
        # Get predictions for the specific selected K
        y_meas = x0[:, idx]
        y_cal = Y_pred_cal_all[:, idx, k_sel-1]
        y_cv = Y_pred_cv_all[:, idx, k_sel-1]
        
        # Plot
        # Calibration: Black circles, CV: Analyte Color x's
        ax.scatter(y_meas, y_cal, c=analyte_color, marker='o', facecolors='none', alpha=0.6, label='Calibration', edgecolors='black')
        ax.scatter(y_meas, y_cv, c=analyte_color, marker='x', alpha=0.8, label='Cross-Validation')
        
        # 1:1 Line
        min_val = min(y_meas.min(), y_cal.min(), y_cv.min())
        max_val = max(y_meas.max(), y_cal.max(), y_cv.max())
        buff = (max_val - min_val) * 0.05
        ax.plot([min_val-buff, max_val+buff], [min_val-buff, max_val+buff], 'k--', alpha=0.5)
        
        # Stats
        sse_cal = np.sum((y_meas - y_cal)**2)
        sst = np.sum((y_meas - np.mean(y_meas))**2)
        r2_cal = 1 - sse_cal/sst
        rmse_cal = np.sqrt(np.mean((y_meas - y_cal)**2))
        
        sse_cv = np.sum((y_meas - y_cv)**2)
        r2_cv = 1 - sse_cv/sst
        rmse_cv = np.sqrt(np.mean((y_meas - y_cv)**2))
        
        stats_text = (f"Cal: $R^2$={r2_cal:.3f}, RMSE={rmse_cal:.3f}\n"
                      f"CV: $R^2$={r2_cv:.3f}, RMSE={rmse_cv:.3f}")
        
        # Place text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)
        
        letter = chr(97 + i)
        ax.set_title(f'{letter}) {name} - Predicted vs Measured (LVs={k_sel})', loc='left')
        ax.set_ylabel(f'Predicted {UNITS}')
        ax.set_xlabel(f'Measured {UNITS}')
        
        ax.legend()

    fig_pred.tight_layout()
    fig_pred.savefig(os.path.join(RESULTS_DIR, "Predicted_vs_Measured_Pub.png"))
    print(f"Saved Prediction Plot to: {os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Pub.png')}")

    print("\nProcessing complete. Close plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()
