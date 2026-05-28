"""
=============================================================================
                            RUN INFERENCE SCRIPT
=============================================================================
This script performs inference (prediction) using a pre-trained PLS model.
It loads a saved model (.pkl), applies pretreatment, predicts analyte 
concentrations, and generates time-series plots comparing predictions to 
reference values (if available).

Workflow:
1. Load Configuration: Define paths, analytes, and pretreatment settings.
2. Load Inference Data: Read absorbance spectra and optional reference data.
3. Pretreatment: Apply the SAME pretreatment used during calibration.
4. Model Loading: Load the trained PLS model (coefficients, normalization).
5. Prediction: Predict Y values for the new spectra.
6. Plotting: Generate time-series plots with RMSECV confidence intervals.

Author: GitHub Copilot
Date: 2026-02-27
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import load_and_predict_pls

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
# Directory where prediction results and plots will be saved
RESULTS_DIR = "results_test" 

# Path to the trained calibration model file (.pkl)
MODEL_PATH = "results/model_calibration.pkl" 

# --- 2. Data Files ---
INFERENCE_FILES = [
    ('data/splits/exp4_refe_test.txt', 'data/splits/exp4_nonda_test.txt'),
    ('data/splits/exp5_refe_test.txt', 'data/splits/exp5_nonda_test.txt'),
    ('data/splits/exp6_refe_test.txt', 'data/splits/exp6_nonda_test.txt'),
    ('data/splits/exp7_refe_test.txt', 'data/splits/exp7_nonda_test.txt'),
    ('data/splits/exp8_refe_test.txt', 'data/splits/exp8_nonda_test.txt'),
    ('data/splits/exp9_refe_test.txt', 'data/splits/exp9_nonda_test.txt')
]

# --- 3. Model Parameters ---
# Must match the calibration configuration
ANALYTES = ['cb', 'gl', 'xy']       # Names of analytes to predict
COLORS = ['green', 'red', 'purple'] # Colors for plotting each analyte
UNITS = 'g/L'                       # Concentration units

# --- 4. Pretreatment Pipeline ---
# CRITICAL: This must match the calibration model's pretreatment exactly.
# Format: [Method, Parameter1, Parameter2, ..., PlotFlag]
# Example: ['SG', Window=7, Poly=2, Ord=1, Plot=0]
PRETREATMENT = [
    ['Cut', 4400, 7500, 0],
    ['SG', 7, 2, 1, 0],
]

# --- 4. Plot Settings (Publication Quality) ---
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
#                            HELPER FUNCTIONS
# =============================================================================

def load_inference_data(files, nc):
    """
    Loads absorbance spectra and aligns optional reference data.

    Args:
        files: List of (ref_path, spec_path) tuples.
        nc: Number of analytes (columns to read from reference file).

    Returns:
        xinf0: Aligned reference data (or None).
        absorinf0: Spectral data with wavelengths row.
        tinf0: Time vector (concatenated).
        sizes: List of sample counts per file (for splitting plots later).
    """
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    time_list = []

    print("Loading inference data...")
    
    for x_f, abs_f in files:
        if not os.path.exists(abs_f): continue
        try:
            # Load Spectra
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            
            # Try to handle "Time" or "Wavenumber" headers
            start_idx = 1
            wl = np.array([float(x) for x in header[start_idx:]])
                
            absi_full = np.loadtxt(abs_f, skiprows=1)
            
            # Assuming col 0 is Time/Index and remaining are spectral points
            ti_spec = absi_full[:, 0]
            absi = absi_full[:, 1:]

            if wavelengths is None: wavelengths = wl
            
            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            time_list.append(ti_spec)
            
            # Load Reference (if available) and align
            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    
                    if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
                    
                    # Check basic shape: [Time, Ref1, Ref2, Ref3...]
                    if ref_data.shape[1] >= nc + 1:
                        # Match reference times to spectral times
                        # Assumption: Reference Time is in MINUTES, Spectral Time is in MINUTES
                        t_ref = ref_data[:, 0]
                        vals_ref = ref_data[:, 1:nc+1]
                        
                        for i, t_val in enumerate(t_ref):
                            t_val_min = t_val 
                            
                            # Find index in ti_spec closest to t_val_min
                            idx = (np.abs(ti_spec - t_val_min)).argmin()
                            
                            # Optional: Check if time difference is acceptable (e.g. < 5 mins)
                            if np.abs(ti_spec[idx] - t_val_min) < 5.0:
                                xi_aligned[idx, :] = vals_ref[i, :]
                                
                except Exception as e:
                    print(f"Warning loading ref {x_f}: {e}")
            
            x_list.append(xi_aligned)

        except Exception as e:
            print(f"Error loading {abs_f}: {e}")

    if not x_list: return None, None, None, []
    
    xinf0 = np.vstack(x_list)
    absorinf_data = np.vstack(absor_list)
    absorinf0 = np.vstack([wavelengths, absorinf_data])
    tinf0 = np.hstack(time_list)
    
    return xinf0, absorinf0, tinf0, sizes

def main():
    """
    Main Execution Function.
    1. Loads data.
    2. Applies pretreatment.
    3. Loads model and predicts.
    4. Saves results.
    5. Plots time-series comparison.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.max_open_warning'] = 100

    # 1. Load Data
    # ----------------------------------------------------
    # Reads all absorbance files, concatenates them, and attempts to align
    # reference data based on timestamps (assuming minutes).
    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None: 
        print("No valid inference data found. Exiting.")
        return
    
    # Separate Wavelengths (Row 0) and Spectra (Rows 1+)
    wl_inf = absorinf0[0, :]
    absor_inf_raw = absorinf0[1:, :]

    # 2. Pretreatment
    # ----------------------------------------------------
    # Applies Savitzky-Golay, Cutting, etc. to match calibration.
    print("Applying Pretreatment...")
    absor_inf_pre, wl_inf_pre = apply_pretreatment(PRETREATMENT, absor_inf_raw, wl_inf, plot=False)

    # 3. Prediction
    # ----------------------------------------------------
    # Loads the saved PLS model (.pkl) and predicts Y values.
    # The loading function handles feature matching and normalization.
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please run run_calibration.py first.")
        return

    try:
        # Returns: (Predictions, List of RMSECV errors per analyte)
        X_pred, rmsecv_loaded = load_and_predict_pls(absor_inf_pre, wl_inf_pre, MODEL_PATH)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return
    
    if X_pred is not None:
        print(f"\nInference Complete. Predictions shape: {X_pred.shape}")
        
        # Determine RMSECV (Error Bars)
        # If available in the model, use it. Otherwise default to 0.0.
        if rmsecv_loaded:
             print(f"Loaded RMSECV from model: {rmsecv_loaded}")
        else:
             print("No RMSECV found in model (using default 0.0)")
             rmsecv_loaded = [0.0]*X_pred.shape[1]
             
        # Save Predictions
        print(f"Results saved to: {RESULTS_DIR}")
        # Format: [Time, Val1, Val2...]
        output_matrix = np.column_stack([tinf0, X_pred])
        header = "Time\t" + "\t".join(ANALYTES)
        np.savetxt(os.path.join(RESULTS_DIR, "Predicted_Inference.txt"), output_matrix, delimiter='\t', header=header, comments='')

        # =========================================================================
        #                            CALCULATE RMSEP & SCATTER PLOTS
        # =========================================================================
        print("\nCreating Predicted vs Measured Scatter Plots...")

        if xinf0 is not None:
             print("\n--- RMSEP (Test Set Error) ---")
             
             fig_pred, axes_pred = plt.subplots(len(ANALYTES), 1, figsize=(7, 6 * len(ANALYTES)))
             if len(ANALYTES) == 1:
                 axes_pred = [axes_pred]
                 
             for j, analyte in enumerate(ANALYTES):
                 ax = axes_pred[j]
                 analyte_color = COLORS[j] if j < len(COLORS) else 'blue'
                 
                 # Filter out missing (NaN) reference values
                 mask_valid = ~np.isnan(xinf0[:, j])
                 ref_valid = xinf0[mask_valid, j]
                 pred_valid = X_pred[mask_valid, j]
                 
                 if len(ref_valid) > 0:
                     rmsep = np.sqrt(np.mean((ref_valid - pred_valid) ** 2))
                     sse = np.sum((ref_valid - pred_valid) ** 2)
                     sst = np.sum((ref_valid - np.mean(ref_valid)) ** 2)
                     r2p = 1 - sse / sst if sst != 0 else 0
                     print(f"{analyte} RMSEP: {rmsep:.3f} {UNITS} | R2: {r2p:.3f}")
                 else:
                     rmsep = 0.0
                     r2p = 0.0
                     print(f"{analyte}: No valid reference data for RMSEP.")

                 # Plot scatter
                 ax.scatter(ref_valid, pred_valid, c=analyte_color, marker='o', alpha=0.8, edgecolors='black', label='Test Data')

                 if len(ref_valid) > 0:
                     min_val = min(ref_valid.min(), pred_valid.min())
                     max_val = max(ref_valid.max(), pred_valid.max())
                     buff = (max_val - min_val) * 0.05
                     ax.plot([min_val - buff, max_val + buff], [min_val - buff, max_val + buff], 'k--', alpha=0.5)

                 stats_text = f"Test set: $R^2$={r2p:.3f}, RMSEP={rmsep:.3f}"
                 props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                 ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=props)

                 letter = chr(97 + j)
                 ax.set_title(f'{letter}) {analyte} - Predicted vs Measured (Test Set)', loc='left')
                 ax.set_ylabel(f'Predicted {UNITS}')
                 ax.set_xlabel(f'Measured {UNITS}')
                 ax.legend(loc='lower right')

             fig_pred.tight_layout(pad=3.0, h_pad=4.0)
             fig_pred.savefig(os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Test.png'), bbox_inches='tight')
             print(f"Saved Prediction Plot to: {os.path.join(RESULTS_DIR, 'Predicted_vs_Measured_Test.png')}")
        else:
             print("No reference data found for calculating RMSEP and scatter plots.")
            
    print("\nProcessing complete. plots saved to results folder. Close plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()
