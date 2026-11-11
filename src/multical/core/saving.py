import pickle
import numpy as np
import os
from ..models.pls import PLS
from ..utils import zscore_matlab_style
# from ..models.spa import spa_apply # If needed
# from ..models.pcr import pcr_apply # If needed

def train_and_save_model_pls(x_cal_pre, y_cal, wavelengths, k_list, filename, rmsecv_list=None):
    """
    Trains a PLS model on the full calibration set for each analyte individually.
    Saves the coefficients and normalization parameters.
    
    Args:
        x_cal_pre: (n_samples, n_features) Pretreated calibration spectra
        y_cal: (n_samples, n_analytes) Calibration concentrations
        wavelengths: (n_features,) Wavelengths vector
        k_list: List of Length n_analytes. Optimal Latent Variables for each analyte.
        filename: Output pickle filename
        rmsecv_list: (Optional) List of RMSECV values for each analyte.
    """
    n_analytes = y_cal.shape[1]
    
    model_data = {
        'type': 'PLS',
        'wavelengths': wavelengths,
        'analytes': []
    }
    
    pls_engine = PLS()
    
    print("\n--- Training Final PLS Model for Saving ---")
    
    for i in range(n_analytes):
        k = k_list[i]
        y_curr = y_cal[:, i].reshape(-1, 1)
        
        # 1. Normalize Training Data
        # We use zscore_matlab_style as per original code, but we must store mean/std
        # Original code for teste_switch=0:
        # Xnorm, Xmed, Xsig = zscore_matlab_style(X)
        # Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
        
        Xnorm, Xmed, Xsig = zscore_matlab_style(x_cal_pre)
        Ynorm, Ymed, Ysig = zscore_matlab_style(y_curr)
        
        # 2. Run NIPALS
        # returns B, T, P, U_scores, Q_loadings, W, r2X, r2Y
        B_pls, _, _, _, _, _, _, _ = pls_engine.nipals(Xnorm, Ynorm, k)
        
        # B_pls relates Normalized X to Normalized Y: Ynorm = Xnorm * B_pls
        # We need to store everything to replicate:
        # Y_pred = (( (X_new - Xmed)/Xsig ) * B_pls ) * Ysig + Ymed
        
        analyte_dict = {
            'index': i,
            'k': k,
            'Xmed': Xmed,
            'Xsig': Xsig,
            'Ymed': Ymed,
            'Ysig': Ysig,
            'B_pls': B_pls,
            'rmsecv': rmsecv_list[i] if rmsecv_list is not None and i < len(rmsecv_list) else None
        }
        model_data['analytes'].append(analyte_dict)
        print(f"  Analyte {i+1}: Trained with k={k}")

    print(f"Saving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print("Save complete.")

def train_and_save_model_svr(x_cal_pre, y_cal, wavelengths, params_list, filename, rmsecv_list=None):
    """
    Trains one SVR model per analyte on the full calibration set and saves them.

    Each analyte gets its own (independently tuned) hyper-parameters, mirroring the
    per-analyte 'optimal k' philosophy of train_and_save_model_pls.

    Args:
        x_cal_pre:  (n_samples, n_features) Pretreated calibration spectra.
        y_cal:      (n_samples, n_analytes) Calibration concentrations (real units).
        wavelengths:(n_features,) Wavelengths vector.
        params_list:List of length n_analytes. Each item is a dict with the keys
                    'C', 'epsilon', 'gamma', 'kernel' for that analyte's SVR.
        filename:   Output file (joblib pickle, .pkl extension recommended).
        rmsecv_list:(Optional) List of RMSECV values for each analyte.
    """
    import joblib
    from ..models.svr import SVRModel

    n_analytes = y_cal.shape[1]

    model_data = {
        'type': 'SVR',
        'wavelengths': wavelengths,
        'analytes': []
    }

    print("\n--- Training Final SVR Model for Saving ---")

    for i in range(n_analytes):
        p = params_list[i]
        y_curr = y_cal[:, i].reshape(-1, 1)

        svr = SVRModel()
        # X_test=None -> fits the model (internal StandardScaler on X and Y) and returns None
        svr.fit_predict(
            x_cal_pre, y_curr, None,
            C=p['C'], epsilon=p['epsilon'],
            gamma=p.get('gamma', 'scale'), kernel=p.get('kernel', 'rbf'),
            degree=p.get('degree', 3), coef0=p.get('coef0', 0.0)
        )

        analyte_dict = {
            'index': i,
            'params': p,
            'model': svr,
            'rmsecv': rmsecv_list[i] if rmsecv_list is not None and i < len(rmsecv_list) else None
        }
        model_data['analytes'].append(analyte_dict)
        print(f"  Analyte {i+1}: Trained with {p}")

    if not filename.endswith('.pkl'):
        filename += '.pkl'

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    print(f"Saving model to {filename}...")
    joblib.dump(model_data, filename)
    print("Save complete.")


def load_and_predict_svr(x_new_pre, wavelengths_new, filename):
    """
    Loads a saved SVR model (per-analyte) and predicts on new pretreated data.

    Mirrors load_and_predict_pls: returns (y_pred, rmsecv_list).
    """
    import joblib

    if not filename.endswith('.pkl'):
        filename += '.pkl'

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")

    model_data = joblib.load(filename)

    if model_data.get('type') != 'SVR':
        raise ValueError(f"Model type mismatch. Expected SVR, got {model_data.get('type')}")

    wl_saved = model_data['wavelengths']
    if len(wl_saved) != len(wavelengths_new):
        print(f"Warning: Wavelength count mismatch ({len(wl_saved)} vs {len(wavelengths_new)}).")

    n_analytes = len(model_data['analytes'])
    n_samples = x_new_pre.shape[0]

    y_pred = np.zeros((n_samples, n_analytes))
    rmsecv_list = []

    for i in range(n_analytes):
        params = model_data['analytes'][i]
        rmsecv_list.append(params['rmsecv'] if params.get('rmsecv') is not None else 0.0)

        y_pred[:, i] = params['model'].predict(x_new_pre).flatten()

    return y_pred, rmsecv_list


def load_and_predict_pls(x_new_pre, wavelengths_new, filename):
    """
    Loads a saved PLS model and predicts on new data.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
        
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
        
    if model_data['type'] != 'PLS':
        raise ValueError(f"Model type mismatch. Expected PLS, got {model_data['type']}")
        
    # Check wavelengths compatibility
    # In robust systems, we'd interpolate. Here we assume identical grid from identical pretreatment.
    wl_saved = model_data['wavelengths']
    if len(wl_saved) != len(wavelengths_new):
         print(f"Warning: Wavelength count mismatch ({len(wl_saved)} vs {len(wavelengths_new)}).")
         # Proceeding anyway usually leads to dimension error in dot product if sizes differ
    
    n_analytes = len(model_data['analytes'])
    n_samples = x_new_pre.shape[0]
    
    y_pred = np.zeros((n_samples, n_analytes))
    rmsecv_list = []
    
    for i in range(n_analytes):
        params = model_data['analytes'][i]
        
        # Store RMSECV if available
        if 'rmsecv' in params and params['rmsecv'] is not None:
             rmsecv_list.append(params['rmsecv'])
        else:
             rmsecv_list.append(0.0)

        # Unpack parameters
        Xmed = params['Xmed']
        Xsig = params['Xsig']
        Ymed = params['Ymed']
        Ysig = params['Ysig']
        B_pls = params['B_pls']
        
        # --- Handle Dimensions (Fix for "tuple index out of range") ---
        # Ensure Xmed/Xsig are 2D row vectors (1, n_features)
        if Xmed.ndim == 1: Xmed = Xmed.reshape(1, -1)
        if Xsig.ndim == 1: Xsig = Xsig.reshape(1, -1)
        # Ensure Ymed/Ysig are 2D scalars/vectors (1, 1) or (1, -1)
        if Ymed.ndim == 1: Ymed = Ymed.reshape(1, -1)
        if Ysig.ndim == 1: Ysig = Ysig.reshape(1, -1)
        
        # Pre-check dimensions
        if x_new_pre.shape[1] != Xmed.shape[1]:
             # Raise explicit error because dot product implies it
             raise ValueError(f"Feature mismatch for Analyte {i}. Model expects {Xmed.shape[1]}, got {x_new_pre.shape[1]}")
        
        # Apply Model
        # 1. Normalize X using Saved Cal Params
        Xnorm_new = (x_new_pre - Xmed) / Xsig
        
        # 2. Predict Y Normalized
        Ynorm_pred = Xnorm_new @ B_pls
        
        # 3. Denormalize Y
        Y_pred_curr = Ynorm_pred * Ysig + Ymed
        
        y_pred[:, i] = Y_pred_curr.flatten()
        
    return y_pred, rmsecv_list
