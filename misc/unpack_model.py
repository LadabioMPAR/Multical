import argparse
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_model(filepath):
    """Loads the model dictionary from a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description="Unpack and Visualize Multical PLS Models")
    parser.add_argument("pkl_file", help="Path to the .pkl model file")
    parser.add_argument("--out", default=None, help="Output directory (default: <model_name>_unpacked)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of coefficients")
    
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from: {args.pkl_file}")
    model_data = load_model(args.pkl_file)
    
    # Basic Validation
    if not isinstance(model_data, dict) or 'type' not in model_data:
        print("Error: Invalid model file structure. Expected a dictionary with a 'type' key.")
        sys.exit(1)

    model_type = model_data.get('type', 'Unknown')
    wavelengths = model_data.get('wavelengths')
    analytes = model_data.get('analytes', [])
    
    print(f"\n--- Model Summary ---")
    print(f"Type: {model_type}")
    if wavelengths is not None:
        print(f"Wavelengths: {len(wavelengths)} points ({wavelengths[0]:.2f} to {wavelengths[-1]:.2f})")
    else:
        print("Wavelengths: None")
        
    print(f"Analytes: {len(analytes)}")
    
    # 2. Prepare Output Directory
    if args.out:
        out_dir = args.out
    else:
        base_name = os.path.splitext(os.path.basename(args.pkl_file))[0]
        out_dir = os.path.join(os.path.dirname(args.pkl_file), base_name + "_unpacked")
    
    ensure_dir(out_dir)
    print(f"Saving output to: {out_dir}")

    # 3. Export Wavelengths
    if wavelengths is not None:
        np.savetxt(os.path.join(out_dir, "wavelengths.txt"), wavelengths, fmt="%.6f", header="Wavelengths")

    # 4. Iterate Analytes
    print(f"\n{'Analyte':<10} | {'LVs (k)':<8} | {'RMSECV':<10} | {'Status'}")
    print("-" * 45)

    for i, analyte in enumerate(analytes):
        # Extract Data
        idx_ = analyte.get('index', i)
        k_ = analyte.get('k', 0)
        rmsecv_ = analyte.get('rmsecv', 0.0)
        B_pls = analyte.get('B_pls')         # Shape (n_features, 1) usually
        
        # Normalization Stats
        Xmed = analyte.get('Xmed')
        Xsig = analyte.get('Xsig')
        Ymed = analyte.get('Ymed')
        Ysig = analyte.get('Ysig')

        # Print Row
        rmsecv_val = rmsecv_ if rmsecv_ is not None else 0.0
        print(f"{idx_:<10} | {k_:<8} | {rmsecv_val:<10.5f} | Exporting...", end="")

        # Create Subfolder per analyte or just prefix files
        # Prefix is usually cleaner for small numbers of analytes
        prefix = f"analyte_{idx_}"
        
        # Save Coefficients (B_pls)
        if B_pls is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_coeffs.txt"), B_pls, header="B_pls (Regression Coefficients - Normalized)")
        
        # Save Statistics
        with open(os.path.join(out_dir, f"{prefix}_stats.txt"), 'w') as f_stat:
            f_stat.write(f"Index: {idx_}\n")
            f_stat.write(f"Latent Variables (k): {k_}\n")
            f_stat.write(f"RMSECV: {rmsecv_}\n")
            f_stat.write(f"Y Mean (Ymed): {Ymed}\n")
            f_stat.write(f"Y Std (Ysig): {Ysig}\n")
        
        # Save Normalization Vectors
        if Xmed is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_X_mean.txt"), Xmed.flatten(), header="X Mean (Calibration)")
        if Xsig is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_X_std.txt"), Xsig.flatten(), header="X Std (Calibration)")

        # Plotting
        if not args.no_plot and B_pls is not None and wavelengths is not None:
            plt.figure(figsize=(10, 6))
            # Handle potential shape mismatch if wavelengths and B_pls differ (though they shouldn't)
            if len(wavelengths) == B_pls.shape[0]:
                plt.plot(wavelengths, B_pls, label=f'Coefficients (k={k_})')
            else:
                plt.plot(B_pls, label=f'Coefficients (k={k_}) - Index')
                
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
            plt.title(f"Regression Coefficients - Analyte {idx_}")
            plt.xlabel("Wavelength / Wavenumber")
            plt.ylabel("Value (Normalized)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(out_dir, f"{prefix}_coeffs_plot.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
        
        print(" Done")

    print(f"\nUnpacking complete. See '{out_dir}' for files.")

if __name__ == "__main__":
    main()
