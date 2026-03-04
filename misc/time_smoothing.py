import pandas as pd
import numpy as np
import argparse
import os

# --- USER CONFIGURATION ---
# You can set the file paths here to run the script directly without command line arguments
FILES_TO_PROCESS = [
    "data/exp_04_inf.txt",
    "data/exp_05_inf.txt",
    "data/exp_06_inf.txt",
    "data/exp_07_inf.txt",
    # Add more files here as needed, e.g.:
    # "data/exp_07_inf.txt",
]  
DEFAULT_WINDOW = 9
# --------------------------

def moving_average_time_axis(input_file, output_file, window_size=3):
    """
    Applies a moving average on spectral data along the time axis (rows).
    Preserves the first row (header/wavenumbers) and ensures output has same shape.
    
    Args:
        input_file (str): Path to input file (tab-separated, first row is header/wavenumbers).
        output_file (str): Path to save the processed file.
        window_size (int): Size of the moving average window.
    """
    print(f"Processing {input_file} with window size {window_size}...")
    
    # Read the file
    # header=None because the first row is wavenumbers, which we treat as data row 0 to keep alignment but won't smooth it.
    try:
        df = pd.read_csv(input_file, sep='\t', header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract wavenumbers (row 0)
    wavenumbers = df.iloc[0:1].copy()
    
    # Extract spectral data (rows 1 to end)
    spectral_data = df.iloc[1:].copy()
    
    print(f"Data shape: {spectral_data.shape} (Time samples x Wavelengths)")
    
    # Apply rolling mean along time axis (axis=0 is default)
    # min_periods=1 ensures that we get a value even if we have fewer than window_size points (e.g. at start)
    # center=False (default) ensures we look at past data [t-window+1 : t]
    # For the first few points where available history < window_size, it uses whatever is available (growing window)
    smoothed_data = spectral_data.rolling(window=window_size, min_periods=1, center=False).mean()
    
    # Combine header and smoothed data
    final_df = pd.concat([wavenumbers, smoothed_data], axis=0)
    
    # Save the result
    print(f"Saving smoothed data to {output_file}...")
    final_df.to_csv(output_file, sep='\t', header=False, index=False, float_format='%.6f')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply causal moving average smoothing along time axis for spectral data.")
    # nargs='*' allows 0 or more arguments
    parser.add_argument("input_files", nargs="*", help="Path to input text files")
    parser.add_argument("--output", "-o", help="Path to the output file (only valid for single input)", default=None)
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Window size for moving average")
    
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use the list at the top
    # We prioritize command line arguments, but if none are given (args.input_files is empty), we use FILES_TO_PROCESS
    input_paths = args.input_files if args.input_files else FILES_TO_PROCESS
    
    if not input_paths:
        print("Error: No input files specified.") 
        print("Please provide file paths in the script (FILES_TO_PROCESS) or via command line.")
        exit(1)
        
    # Check if multiple inputs but single output specified
    if args.output and len(input_paths) > 1:
        print("Error: Cannot specify a single output file when processing multiple input files.")
        print("Please remove --output argument to use automatic naming.")
        exit(1)

    processed_count = 0
    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Error: File '{input_path}' not found. Skipping.")
            continue

        if args.output and len(input_paths) == 1:
            output_path = args.output
        else:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_smoothed{ext}"
            
        moving_average_time_axis(input_path, output_path, args.window)
        processed_count += 1
        
    if processed_count == 0:
        print("No files were processed.")
        exit(1)
