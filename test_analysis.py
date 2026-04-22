import sys
with open('src/multical/analysis.py', 'r') as f:
    text = f.read()

# insert print
text = text.replace("K, _, _, _ = np.linalg.lstsq(x, absor, rcond=None)", "print(f'shapes: x={x.shape}, absor={absor.shape}'); K, _, _, _ = np.linalg.lstsq(x, absor, rcond=None)")

with open('src/multical/analysis.py', 'w') as f:
    f.write(text)
