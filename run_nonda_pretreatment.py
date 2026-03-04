"""
=============================================================================
                        RUN NONDA PRETREATMENT
=============================================================================
This script reads *nonda.txt files, applies the defined pretreatment,
and saves the result as *_pretreated.txt.
Pretreatment matches run_calibration.py.
"""

import numpy as np
import os
import glob
from src.multical.preprocessing.pipeline import apply_pretreatment

# =============================================================================
#                               CONFIGURATION
# =============================================================================

INPUT_DIR = "data"
PATTERN = "*nonda.txt" 
PLOTS_DIR = "results_pretreatment_plots"

# Pretreatment identical to run_calibration.py
PRETREATMENT = [
    ['Cut', 4400, 7500, 1], # Cut spectral region (Min, Max, Plot?)
    ['EMSC', 2, 1],         # EMSC: Degree=2
]

OUTPUT_SUFFIX = "_pretreated.txt"

# =============================================================================
#                                  MAIN
# =============================================================================

def process_file(filepath):
    print(f"Processing {filepath}...")
    
    # 1. Read File
    try:
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
        
        header_parts = header_line.split()
        if header_parts[0] != "Time":
            print(f"Warning: {filepath} does not start with 'Time'. Assuming standard format anyway.")
            
        wavelengths = np.array([float(x) for x in header_parts[1:]])
        
        # Load data
        data = np.loadtxt(filepath, skiprows=1)
        
        if data.size == 0:
            print(f"Empty file: {filepath}")
            return

        time_col = data[:, 0]
        absor = data[:, 1:]
        
        # Determine base name for prefix
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        plot_prefix = name.replace("_nonda", "") + "_"

        # 2. Apply Pretreatment
        # We enable plotting and save to PLOTS_DIR
        new_absor, new_wavelengths = apply_pretreatment(
            PRETREATMENT, 
            absor, 
            wavelengths, 
            plot=True, 
            output_dir=PLOTS_DIR,
            prefix=plot_prefix
        )
        
        # 3. Save Output
        # Construct header
        new_header = "Time\t" + "\t".join([f"{w:.2f}" for w in new_wavelengths])
        
        # Concatenate Time + New Absorbance
        # time_col is (N,), new_absor is (N, M)
        output_data = np.hstack((time_col.reshape(-1, 1), new_absor))
        
        # Determine output filename
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        output_filename = name + OUTPUT_SUFFIX # e.g. exp4_nonda_pretreated.txt
        output_path = os.path.join(INPUT_DIR, output_filename)
        
        print(f"Saving to {output_path}")
        np.savetxt(output_path, output_data, delimiter='\t', header=new_header, comments='') # comments='' to avoid #
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    search_path = os.path.join(INPUT_DIR, PATTERN)
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No files found matching {search_path}")
        return
        
    print(f"Found {len(files)} files.")
    
    # Ensure plots directory exists
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created plot directory: {PLOTS_DIR}")
    
    for f in files:
        process_file(f)
        
    print("Done.")

if __name__ == "__main__":
    main()
