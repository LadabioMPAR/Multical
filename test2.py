with open('run_calibration.py') as f:
    text = f.read()

import re
print("Length of source:", len(text))
for line in text.split('\n'):
    if 'absor_pre_final' in line or 'np.linalg.lstsq' in line or 'import' in line:
        pass
