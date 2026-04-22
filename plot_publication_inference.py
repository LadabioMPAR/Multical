import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import pickle

# Import necessary modules from the project
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import load_and_predict_pls

# =============================================================================
#                                 CONFIGURATION
# =============================================================================
MODELS_TO_PLOT = [
    {
        "name": "Calibration_Baseline", 
        "path": "results/model_calibration.pkl", 
        "out_dir": "results_publication_plots/calibration",
        "pretreatment": [
            ['Cut', 4160, 10000, 0]
        ]
    },
    {
        "name": "Variable_Selection", 
        "path": "results_var_selection/model_variable_selection.pkl", 
        "out_dir": "results_publication_plots/var_selection",
        "pretreatment": [
            ['Cut', 4400, 7500, 0],
            ['SG', 7, 2, 1, 0]
        ]
    }
]

INFERENCE_FILES = [
    ('data/exp4_refe.txt', 'data/exp_04_inf_smoothed.txt'),
    ('data/exp5_refe.txt', 'data/exp_05_inf_smoothed.txt'),
    ('data/exp6_refe.txt', 'data/exp_06_inf_smoothed.txt'),
    ('data/exp7_refe.txt', 'data/exp_07_inf_smoothed.txt'),
]

ANALYTES = ['cb', 'gl', 'xy']       
UNITS = 'g/L'                       

# Strict Color Scheme for Glucose and Xylose
COLORS_MAP = {
    'gl': 'red',
    'xy': 'purple'
}

# Window size for the causal moving average
SMOOTHING_WINDOW = 3

# =============================================================================
#                                GLOBAL STYLE
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# =============================================================================
#                            HELPER FUNCTIONS
# =============================================================================

def load_inference_data(files, nc):
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    time_list = []

    print("Loading inference data...")
    
    for x_f, abs_f in files:
        if not os.path.exists(abs_f):
            continue
        try:
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            start_idx = 1
            wl = np.array([float(x) for x in header[start_idx:]])
            absi_full = np.loadtxt(abs_f, skiprows=1)
            
            # Load the raw file just to get the uncorrupted Time column
            raw_f = abs_f.replace('_smoothed', '')
            if os.path.exists(raw_f):
                ti_spec = np.loadtxt(raw_f, skiprows=1)[:, 0]
            else:
                ti_spec = absi_full[:, 0]
                
            absi = absi_full[:, 1:]

            if wavelengths is None: wavelengths = wl
            
            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            time_list.append(ti_spec)
            
            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
                    if ref_data.shape[1] >= nc + 1:
                        t_ref = ref_data[:, 0]
                        vals_ref = ref_data[:, 1:nc+1]
                        for i, t_val in enumerate(t_ref):
                            idx = (np.abs(ti_spec - t_val)).argmin()
                            if np.abs(ti_spec[idx] - t_val) < 5.0:
                                xi_aligned[idx, :] = vals_ref[i, :]
                except Exception as e:
                    pass
            x_list.append(xi_aligned)
        except Exception as e:
            pass

    if not x_list: return None, None, None, []
    
    xinf0 = np.vstack(x_list)
    absorinf_data = np.vstack(absor_list)
    absorinf0 = np.vstack([wavelengths, absorinf_data])
    tinf0 = np.hstack(time_list)
    
    return xinf0, absorinf0, tinf0, sizes

def filter_to_model_wavelengths(absor_pre, wl_pre, wl_model):
    # No filtering needed if lengths perfectly match (e.g. calibration baseline)
    if len(wl_pre) == len(wl_model) and np.allclose(wl_pre, wl_model, atol=1e-1):
        return absor_pre, wl_pre

    indices = []
    found_count = 0
    tolerance = 1e-1
    for wm in wl_model:
        diffs = np.abs(wl_pre - wm)
        idx = np.argmin(diffs)
        if diffs[idx] < tolerance:
            indices.append(idx)
            found_count += 1
    
    if found_count != len(wl_model):
        raise ValueError(f"Could not align inference wavelengths to model selection. Expected {len(wl_model)}, found {found_count}.")
        
    absor_filtered = absor_pre[:, indices]
    wl_filtered = wl_pre[indices]
    return absor_filtered, wl_filtered

def plot_publication_quality(t_exp, pred_raw, ref_vals, rmse_val, analyte_name, color, exp_name, output_dir, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(t_exp, pred_raw, color=color, linewidth=1, alpha=0.4, label='Raw Prediction')
    
    smoothed = pd.Series(pred_raw).rolling(window=SMOOTHING_WINDOW, min_periods=1, center=False).mean()
    ax.plot(t_exp, smoothed, color=color, linewidth=2.5, linestyle='-', label=f'Smoothed Prediction (W={SMOOTHING_WINDOW})')
    
    ax.fill_between(t_exp, smoothed - rmse_val, smoothed + rmse_val, 
                    color=color, alpha=0.2, label=f'RMSECV (\u00B1{rmse_val:.2f})')
    
    mask_valid = ~np.isnan(ref_vals)
    if np.any(mask_valid):
        ax.scatter(t_exp[mask_valid], ref_vals[mask_valid], 
                   color=color, edgecolors='black', s=80, marker='o', zorder=5, label='Offline HPLC')
        
    ax.set_title(f"{analyte_name.capitalize()} Monitoring ({model_name}) - {exp_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel(f"Concentration ({UNITS})", fontsize=12)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10, framealpha=0.9, edgecolor='gray')
    
    out_name = f"{exp_name}_{analyte_name}_publication.png"
    out_path = os.path.join(output_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close(fig)

# =============================================================================
#                               MAIN SCRIPT
# =============================================================================

def main():
    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None:
        print("No valid inference data found. Exiting.")
        return
        
    wl_inf = absorinf0[0, :]
    absor_inf_raw = absorinf0[1:, :]

    for model_cfg in MODELS_TO_PLOT:
        model_name = model_cfg['name']
        model_path = model_cfg['path']
        out_dir = model_cfg['out_dir']
        pretreatment = model_cfg['pretreatment']
        
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n[{model_name}] Loading Model from {model_path}...")
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Skipping...")
            continue
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"[{model_name}] Applying specific Pretreatment: {pretreatment}...")
        absor_inf_pre, wl_inf_pre = apply_pretreatment(pretreatment, absor_inf_raw, wl_inf, plot=False)

        wl_model = model_data.get('wavelengths', wl_inf_pre)
        
        try:
            absor_inf_sel, wl_inf_sel = filter_to_model_wavelengths(absor_inf_pre, wl_inf_pre, wl_model)
            X_pred, rmsecv_loaded = load_and_predict_pls(absor_inf_sel, wl_inf_sel, model_path)
            print(f"[{model_name}] Predictions shape: {X_pred.shape}")
        except Exception as e:
            print(f"[{model_name}] Prediction Error: {e}")
            continue
        
        print(f"[{model_name}] Creating Publication Plots...")

        current_idx = 0
        nc = X_pred.shape[1]
        
        for i, size in enumerate(inf_sizes):
            end_idx = current_idx + size
            
            t_exp = tinf0[current_idx:end_idx] / 60.0
            pred_exp = X_pred[current_idx:end_idx, :]
            ref_exp = xinf0[current_idx:end_idx, :] if xinf0 is not None else None
            
            if i < len(INFERENCE_FILES):
                base_name = os.path.basename(INFERENCE_FILES[i][1])
                exp_name = base_name.replace("_inf_smoothed.txt", "")
            else:
                exp_name = f"Exp_{i+1}"

            for j in range(nc):
                analyte_name = ANALYTES[j]
                
                if analyte_name not in COLORS_MAP:
                    continue
                    
                color = COLORS_MAP[analyte_name]
                
                rmse_entry = rmsecv_loaded[j] if j < len(rmsecv_loaded) else 0.0
                if np.ndim(rmse_entry) > 0:
                    rmse_val = rmse_entry[j] if len(rmse_entry) == nc else rmse_entry[0]
                else:
                    rmse_val = float(rmse_entry)

                pred_raw = pred_exp[:, j]
                ref_vals = ref_exp[:, j] if ref_exp is not None else np.full(size, np.nan)

                plot_publication_quality(
                    t_exp=t_exp,
                    pred_raw=pred_raw,
                    ref_vals=ref_vals,
                    rmse_val=rmse_val,
                    analyte_name=analyte_name,
                    color=color,
                    exp_name=exp_name,
                    output_dir=out_dir,
                    model_name=model_name
                )
            
            print(f"  Saved plots for {exp_name}")
            current_idx = end_idx

if __name__ == "__main__":
    main()
