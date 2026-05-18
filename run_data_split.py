import numpy as np
import os
import argparse
from scipy.spatial.distance import cdist

# =============================================================================
#                                 CONFIGURATION
# =============================================================================
# Set the default parameters below if you are running this script directly
# without command line arguments.

# List of (Spectra_File, Target_File)
DATA_FILES = [
    ('data/exp4_nonda.txt', 'data/exp4_refe.txt'),
    ('data/exp5_nonda.txt', 'data/exp5_refe.txt'),
    ('data/exp6_nonda.txt', 'data/exp6_refe.txt'),
    ('data/exp7_nonda.txt', 'data/exp7_refe.txt'),
    #('data/exp8_nonda.txt', 'data/exp8_refe.txt'),
]

# Output directory for the split files
OUTPUT_DIR = "data/splits"

# Fraction of data to use for testing 
DEFAULT_TEST_FRACTION = 0.25

# =============================================================================
#                                  ALGORITHM
# =============================================================================

def kennard_stone(X, num_cal, ignore_first_col=True):
    """
    Kennard-Stone algorithm for splitting data into calibration and test sets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Data matrix (samples x features)
    num_cal : int
        Number of samples to select for the calibration set.
    ignore_first_col : bool
        Whether to ignore the first column (e.g. Time) for distance calculation.
        
    Returns
    -------
    cal_indices : numpy.ndarray
        Indices for the calibration set.
    test_indices : numpy.ndarray
        Indices for the test set.
    """
    num_samples = X.shape[0]
    
    # Optional filtering for distance metrics
    X_calc = X[:, 1:] if ignore_first_col and X.shape[1] > 1 else X
    
    if num_cal >= num_samples:
        return np.arange(num_samples), np.array([], dtype=int)
    if num_cal == 0:
        return np.array([], dtype=int), np.arange(num_samples)
    if num_cal == 1:
        # Just pick the sample closest to the mean
        mean_X = np.mean(X_calc, axis=0, keepdims=True)
        dists = cdist(X_calc, mean_X).flatten()
        idx = np.argmin(dists)
        cal = [idx]
        test = np.setdiff1d(np.arange(num_samples), cal)
        return np.array(cal), test

    # Calculate distance matrix between all samples
    dists = cdist(X_calc, X_calc, metric='euclidean')
    
    # Find the two points that are furthest apart
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    cal_indices = list(max_idx)
    
    # Initialize minimum distances from unselected samples to the calibration set
    min_dists = np.minimum(dists[:, cal_indices[0]], dists[:, cal_indices[1]])
    
    # Set the distance of already selected to -1 so they aren't picked again
    min_dists[cal_indices] = -1.0
    
    for _ in range(2, num_cal):
        # Sample with the maximum minimum-distance to the already selected
        next_idx = np.argmax(min_dists)
        cal_indices.append(next_idx)
        
        # Update minimum distances
        min_dists = np.minimum(min_dists, dists[:, next_idx])
        min_dists[cal_indices] = -1.0 
        
    all_indices = np.arange(num_samples)
    test_indices = np.setdiff1d(all_indices, cal_indices)
    
    return np.array(cal_indices), np.array(test_indices)

# =============================================================================
#                                 HELPER FUNCTIONS
# =============================================================================

def load_data_file(filepath, has_header=True):
    """Loads array data and optionally preserves the first line as a header string."""
    header = ""
    if has_header:
        with open(filepath, 'r') as f:
            header = f.readline().strip()
            if header.startswith('#'):
                header = header[1:].strip()
        data = np.loadtxt(filepath, skiprows=1)
    else:
        data = np.loadtxt(filepath)
        
    return data, header

def save_split_file(filepath, data, header):
    """Saves the data matrix with the specific header using tab delimiters."""
    if header:
        np.savetxt(filepath, data, delimiter='\t', header=header, comments='')
    else:
        np.savetxt(filepath, data, delimiter='\t')

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Split dataset using Kennard-Stone algorithm.")
    parser.add_argument('-x', '--spectra', type=str, nargs='*',
                        help="File path(s) for feature/spectra matrix X")
    parser.add_argument('-y', '--targets', type=str, nargs='*',
                        help="File path(s) for targets/concentration matrix Y")
    parser.add_argument('-t', '--test-fraction', type=float, default=DEFAULT_TEST_FRACTION,
                        help=f"Fraction of data for the test set (default: {DEFAULT_TEST_FRACTION})")
    parser.add_argument('-o', '--outdir', type=str, default=OUTPUT_DIR,
                        help=f"Directory to save outputs (default: {OUTPUT_DIR})")
                        
    parser.add_argument('--keep-first-col-dist', action='store_true',
                        help="Include the first column in the distance calculation. (By default it is ignored, assuming it's 'Time')")
                        
    args = parser.parse_args()

    # Determine file pairs to process
    file_pairs = []
    
    if args.spectra and args.targets:
        if len(args.spectra) != len(args.targets):
            print("Error: The number of spectra files must match the number of target files when passed via arguments.")
            return
        for mx, my in zip(args.spectra, args.targets):
            file_pairs.append((mx, my))
    else:
        file_pairs = DATA_FILES

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    for spectra_path, target_path in file_pairs:
        print(f"\n--- Processing: {spectra_path} & {target_path} ---")
        
        # Validate inputs
        if not os.path.exists(spectra_path):
            print(f"Error: Spectra file '{spectra_path}' not found. Skipping.")
            continue
        if not os.path.exists(target_path):
            print(f"Error: Target file '{target_path}' not found. Skipping.")
            continue

        print("Loading data...")
        X, header_X = load_data_file(spectra_path, has_header=True)
        Y, header_Y = load_data_file(target_path, has_header=False)

        num_samples = X.shape[0]
        if Y.shape[0] != num_samples:
            print(f"Error: Mismatch in number of samples. X has {num_samples}, Y has {Y.shape[0]}. Skipping.")
            continue

        num_test = int(num_samples * args.test_fraction)
        num_cal = num_samples - num_test
        
        print(f"Total samples: {num_samples} | Cal: {num_cal} | Test: {num_test}")
        print("Running Kennard-Stone algorithm...")

        cal_idx, test_idx = kennard_stone(X, num_cal, ignore_first_col=not args.keep_first_col_dist)

        print("Splitting data...")
        X_cal = X[cal_idx, :]
        Y_cal = Y[cal_idx, :] if Y.ndim == 2 else Y[cal_idx]
        
        X_test = X[test_idx, :]
        Y_test = Y[test_idx, :] if Y.ndim == 2 else Y[test_idx]
        
        # Format new filenames
        base_x = os.path.splitext(os.path.basename(spectra_path))[0]
        base_y = os.path.splitext(os.path.basename(target_path))[0]

        out_x_cal = os.path.join(args.outdir, f"{base_x}_cal.txt")
        out_x_test = os.path.join(args.outdir, f"{base_x}_test.txt")
        
        out_y_cal = os.path.join(args.outdir, f"{base_y}_cal.txt")
        out_y_test = os.path.join(args.outdir, f"{base_y}_test.txt")

        # Use the base_x name in console for simplicity
        print(f"Saving splits to '{args.outdir}'...")
        save_split_file(out_x_cal, X_cal, header_X)
        save_split_file(out_x_test, X_test, header_X)
        save_split_file(out_y_cal, Y_cal, header_Y)
        save_split_file(out_y_test, Y_test, header_Y)
        
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
