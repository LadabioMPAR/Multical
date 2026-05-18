"""
Standalone publication-quality plotting script for real-time bioprocess
monitoring inference from the variable-selection model.

This script reads the same inference data structures used by the existing
inference scripts, then generates a publication-ready time series figure for
each analyte with:
1. Raw predictions as a background line.
2. Causal smoothed predictions using a backward-looking pandas rolling mean.
3. An uncertainty band based on +/- RMSECV around the smoothed line.
4. Discrete offline HPLC reference points on top.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
#                                 CONFIGURATION
# =============================================================================

RESULTS_DIR = "results_inference_var_sel_pub"
PREDICTIONS_PATH = "results_inference_var_sel/Predicted_Inference.txt"
MODEL_PATH = "results_var_selection/model_variable_selection.pkl"

INFERENCE_FILES = [
    ("data/exp4_refe.txt", "data/exp_04_inf.txt"),
    ("data/exp5_refe.txt", "data/exp_05_inf.txt"),
    ("data/exp6_refe.txt", "data/exp_06_inf.txt"),
    ("data/exp7_refe.txt", "data/exp_07_inf.txt"),
]

ANALYTES = ["cb", "gl", "xy"]
UNITS = "g/L"
SMOOTH_WINDOW = 4
PLOT_ANALYTES = ["gl", "xy"]
PLOT_START_HOUR = 0.0
MAX_PLOT_HOURS = 12.0

COLOR_MAP = {
    "gl": "red",
    "xy": "purple",
}

# --- Publication Quality Plot Parameters ---
PLOT_PARAMS = {
    # Font Configuration
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Calibri"],
    "font.size": 16,  # Base Text Size
    # Axes
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,  # Edge width
    # Ticks
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    # Legend
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.loc": "best",
    # Lines & Markers
    "lines.linewidth": 2,
    "lines.markersize": 8,
    # Saving
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.format": "png",
}


# =============================================================================
#                            DATA LOADING HELPERS
# =============================================================================


def load_inference_data(files, nc):
    """Load inference spectra and align optional HPLC reference data."""
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    time_list = []

    print("Loading inference data...")

    for x_f, abs_f in files:
        if not os.path.exists(abs_f):
            continue

        try:
            with open(abs_f, "r", encoding="utf-8") as f:
                header = f.readline().strip().split()

            wl = np.array([float(x) for x in header[1:]])
            absi_full = np.loadtxt(abs_f, skiprows=1)
            ti_spec = absi_full[:, 0]
            absi = absi_full[:, 1:]

            if wavelengths is None:
                wavelengths = wl

            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            time_list.append(ti_spec)

            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    if ref_data.ndim == 1:
                        ref_data = ref_data.reshape(1, -1)

                    if ref_data.shape[1] >= nc + 1:
                        t_ref = ref_data[:, 0]
                        vals_ref = ref_data[:, 1 : nc + 1]

                        for i, t_val in enumerate(t_ref):
                            idx = (np.abs(ti_spec - t_val)).argmin()
                            if np.abs(ti_spec[idx] - t_val) < 5.0:
                                xi_aligned[idx, :] = vals_ref[i, :]
                except Exception as exc:
                    print(f"Warning loading ref {x_f}: {exc}")

            x_list.append(xi_aligned)

        except Exception as exc:
            print(f"Error loading {abs_f}: {exc}")

    if not x_list:
        return None, None, None, []

    xinf0 = np.vstack(x_list)
    absorinf_data = np.vstack(absor_list)
    absorinf0 = np.vstack([wavelengths, absorinf_data])
    tinf0 = np.hstack(time_list)

    return xinf0, absorinf0, tinf0, sizes


def load_predictions(predictions_path):
    """Load time-series predictions saved by the existing inference scripts."""
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"Predictions file '{predictions_path}' was not found. "
            "Run run_inference.py first."
        )

    data = np.loadtxt(predictions_path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    times = data[:, 0]
    preds = data[:, 1:]
    return times, preds


def _read_rmsecv_table(results_dir):
    """Read the RMSECV table saved by calibration/selection scripts.

    Returns:
        tuple: (table, column_names) where column_names excludes the leading k.
    """
    candidates = [
        os.path.join(results_dir, "Erro_cv.txt"),
        os.path.join(results_dir, "Erro_cv_norm.txt"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    header = f.readline().strip().lstrip("#").split()

                table = np.loadtxt(candidate, comments="#", skiprows=1)
                if table.ndim == 1:
                    table = table.reshape(1, -1)
                return table, header[1:]
            except Exception as exc:
                print(f"Warning: could not read RMSECV table '{candidate}': {exc}")

    return None, None


def load_rmsecv_from_results(model_path, analytes_to_plot):
    """Load per-analyte RMSECV from the authoritative results table.

    The model pickle is only used to find the selected latent variables (k) for
    each analyte. The actual RMSECV values come from the calibration/selection
    results file in the same directory as the model.
    """
    if not os.path.exists(model_path):
        return {name: 0.0 for name in analytes_to_plot}

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        results_dir = os.path.dirname(model_path)
        rmsecv_table, table_names = _read_rmsecv_table(results_dir)
        analytes = model_data.get("analytes", [])

        rmsecv = {}
        for name in analytes_to_plot:
            if name in ANALYTES:
                model_idx = ANALYTES.index(name)
            else:
                model_idx = None

            if model_idx is not None and model_idx < len(analytes):
                k_sel = int(analytes[model_idx].get("k", model_idx + 1))
            else:
                k_sel = 1

            if (
                rmsecv_table is not None
                and table_names is not None
                and name in table_names
                and 1 <= k_sel <= rmsecv_table.shape[0]
            ):
                col_idx = table_names.index(name)
                rmsecv[name] = float(rmsecv_table[k_sel - 1, col_idx + 1 if rmsecv_table.shape[1] == len(table_names) + 1 else col_idx])
                continue

            if model_idx is not None and model_idx < len(analytes) and analytes[model_idx].get("rmsecv") is not None:
                rmsecv_entry = np.asarray(analytes[model_idx]["rmsecv"]).squeeze()
                if rmsecv_entry.ndim == 0:
                    rmsecv[name] = float(rmsecv_entry)
                elif rmsecv_entry.size >= model_idx + 1:
                    rmsecv[name] = float(rmsecv_entry[model_idx])
                else:
                    rmsecv[name] = float(rmsecv_entry.flat[0])
            else:
                rmsecv[name] = 0.0

        return rmsecv
    except Exception as exc:
        print(f"Warning: could not load RMSECV from results for '{model_path}': {exc}")
        return {name: 0.0 for name in analytes_to_plot}


# =============================================================================
#                             PLOTTING UTILITIES
# =============================================================================

def smooth_causal(values, window):
    """Backward-looking moving average with a right-closed window."""
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1, closed="right").mean().to_numpy()


def select_plot_window(time_hours, predictions, references=None, start_hour=0.0, max_hours=None):
    """Select a contiguous time window for plotting.

    This keeps only the samples whose time is between start_hour and
    start_hour + max_hours.
    """
    time_hours = np.asarray(time_hours)
    mask = time_hours >= start_hour
    if max_hours is not None:
        mask &= time_hours <= (start_hour + max_hours)

    if not np.any(mask):
        return time_hours, predictions, references

    time_sel = time_hours[mask]
    pred_sel = predictions[mask, :]
    ref_sel = references[mask, :] if references is not None else None
    return time_sel, pred_sel, ref_sel


def analyte_display_name(analyte):
    mapping = {
        "cb": "Cellobiose",
        "gl": "Glucose",
        "xy": "Xylose",
    }
    return mapping.get(analyte.lower(), analyte)


def plot_publication_inference(
    time_hours,
    predictions,
    references,
    ref_times_hours,
    analytes,
    rmsecv_values,
    output_dir,
    experiment_name,
    smooth_window=7,
):
    """Generate publication-quality inference plots for one experiment."""
    os.makedirs(output_dir, exist_ok=True)

    n_analytes = predictions.shape[1]
    time_hours = np.asarray(time_hours)
    ref_times_hours = np.asarray(ref_times_hours)

    for j in range(n_analytes):
        analyte = analytes[j]
        display_name = analyte_display_name(analyte)
        color = COLOR_MAP.get(analyte.lower(), "#4c4c4c")

        raw_pred = np.asarray(predictions[:, j])
        smoothed_pred = smooth_causal(raw_pred, smooth_window)
        rmsecv_entry = rmsecv_values.get(analyte, 0.0)
        if np.ndim(rmsecv_entry) > 0:
            rmsecv_array = np.asarray(rmsecv_entry).squeeze()
            if rmsecv_array.ndim == 0:
                rmsecv = float(rmsecv_array)
            elif rmsecv_array.size == n_analytes:
                rmsecv = float(rmsecv_array[j])
            else:
                rmsecv = float(rmsecv_array.flat[0])
        else:
            rmsecv = float(rmsecv_entry)

        fig, ax = plt.subplots(figsize=(11, 5.5))

        ax.plot(
            time_hours,
            raw_pred,
            color=color,
            linewidth=1.0,
            alpha=0.4,
            label="Raw prediction",
            zorder=1,
        )

        ax.fill_between(
            time_hours,
            smoothed_pred - rmsecv,
            smoothed_pred + rmsecv,
            color=color,
            alpha=0.2,
            linewidth=0,
            label=rf"$\pm$RMSECV ({rmsecv:.2f})",
            zorder=2,
        )

        ax.plot(
            time_hours,
            smoothed_pred,
            color=color,
            linewidth=2.8,
            linestyle="-",
            label="Smoothed prediction",
            zorder=3,
        )

        if references is not None:
            ref_vals = references[:, j]
            valid_mask = ~np.isnan(ref_vals)
            if np.any(valid_mask):
                ax.scatter(
                    ref_times_hours[valid_mask],
                    ref_vals[valid_mask],
                    s=42,
                    color=color,
                    edgecolors="none",
                    label="HPLC reference",
                    zorder=4,
                )

        ax.set_title(f"{display_name} Model Prediction")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (g/L)")
        ax.legend(frameon=True, fancybox=True, framealpha=0.95)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)

        fig.tight_layout()
        safe_experiment = experiment_name.replace(".txt", "").replace(".csv", "")
        output_path = os.path.join(output_dir, f"PubPlot_{safe_experiment}_{analyte}.png")
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
#                                    MAIN
# =============================================================================

def main():
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams["figure.max_open_warning"] = 100

    os.makedirs(RESULTS_DIR, exist_ok=True)

    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None:
        print("No valid inference data found.")
        return

    pred_times, predictions = load_predictions(PREDICTIONS_PATH)
    if len(pred_times) != len(tinf0):
        raise ValueError(
            f"Prediction length mismatch: predictions={len(pred_times)}, inference times={len(tinf0)}"
        )

    if not np.allclose(pred_times, tinf0):
        print("Warning: prediction times differ from the loaded inference times; using prediction times for plotting.")

    rmsecv_values = load_rmsecv_from_results(MODEL_PATH, PLOT_ANALYTES)

    current_idx = 0
    for i, size in enumerate(inf_sizes):
        end_idx = current_idx + size
        time_exp_hours = pred_times[current_idx:end_idx] / 60.0
        pred_exp = predictions[current_idx:end_idx, :]

        if xinf0 is not None:
            ref_exp = xinf0[current_idx:end_idx, :]
        else:
            ref_exp = None

        _, fname_spec = INFERENCE_FILES[i] if i < len(INFERENCE_FILES) else (None, f"Exp_{i+1}")
        exp_name = os.path.basename(fname_spec)

        time_exp_hours, pred_exp, ref_exp = select_plot_window(
            time_exp_hours,
            pred_exp,
            ref_exp,
            start_hour=PLOT_START_HOUR,
            max_hours=MAX_PLOT_HOURS,
        )

        plot_publication_inference(
            time_exp_hours,
            pred_exp[:, [ANALYTES.index(name) for name in PLOT_ANALYTES]],
            ref_exp[:, [ANALYTES.index(name) for name in PLOT_ANALYTES]] if ref_exp is not None else None,
            time_exp_hours,
            PLOT_ANALYTES,
            rmsecv_values,
            RESULTS_DIR,
            exp_name,
            smooth_window=SMOOTH_WINDOW,
        )

        current_idx = end_idx

    print(f"Publication plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()