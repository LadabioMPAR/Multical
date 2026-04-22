import os

cal_file = 'run_calibration.py'
with open(cal_file, 'r') as f:
    text = f.read()

# Fix rmsecv passed to train_and_save_model_pls
old_str = "train_and_save_model_pls(absor_pre_final, x0, wl_final, best_k_list, os.path.join(RESULTS_DIR, model_name), rmsecv_list=RMSECV_conc)"
new_str = "    best_rmsecv_correct = [RMSECV_conc[best_k_list[i]-1, i] if isinstance(best_k_list, dict) else RMSECV_conc[best_k_list[i]-1, i] for i in range(nc)]\n    train_and_save_model_pls(absor_pre_final, x0, wl_final, best_k_list, os.path.join(RESULTS_DIR, model_name), rmsecv_list=best_rmsecv_correct)"

text = text.replace(old_str, new_str)
with open(cal_file, 'w') as f:
    f.write(text)

plot_file = 'plot_publication_inference.py'
with open(plot_file, 'r') as f:
    text = f.read()

# Fix time loading in load_inference_data by replacing the smoothed file's ti_spec with the raw file's ti_spec
old_load = '''
            ti_spec = absi_full[:, 0]
            absi = absi_full[:, 1:]
'''
new_load = '''
            # Load the raw file just to get the uncorrupted Time column
            raw_f = abs_f.replace('_smoothed', '')
            if os.path.exists(raw_f):
                ti_spec = np.loadtxt(raw_f, skiprows=1)[:, 0]
            else:
                ti_spec = absi_full[:, 0]
                
            absi = absi_full[:, 1:]
'''
text = text.replace(old_load, new_load)
with open(plot_file, 'w') as f:
    f.write(text)
print('Patched successfully!')
