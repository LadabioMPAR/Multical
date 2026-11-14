# MultiCal: Multivariate Calibration Tool

**MultiCal** is a Python package for chemometrics and multivariate calibration: it builds predictive models for
biochemical processes from spectroscopic data (NIR, Raman, MIR, etc.), from the raw spectra to a model file used to
estimate new samples.

It can be used in three ways:

| Way | For whom | Where |
|---|---|---|
| **Desktop program** (interface in Portuguese) | people who need a model, not the theory | `python abrir_multical.py`, or the Windows program built by `packaging/build.py` |
| **Python API** | notebooks and your own scripts | `multical.api` |
| **Research scripts** | the original script-based workflow | `scripts/` |

## What it does

*   **Reads** the spectrometer's `.spp` time series and text files (axis in the first row, one spectrum per row), and
    links each sample of the reference table to its spectrum by time or by row order.
*   **Pretreatments**: spectral cut, Savitzky-Golay, moving average, LOESS, derivative, SNV, MSC, EMSC, baseline
    correction, autoscaling and mean centring, combined in a recipe that is learned on the calibration set and reapplied
    unchanged to new spectra.
*   **Models**, one per analyte: PLS, PCR, SPA and SVR (and, as an optional plugin that needs PyTorch, a neural network).
*   **Validation**: k-fold (venetian, consecutive or random), leave-one-out, hold-out, Kennard-Stone / SPXY and
    leave-one-experiment-out. The number of latent variables is chosen by an F-test or by the lowest validation error.
*   **Variable selection** per analyte: VIP, particle swarm (PSO) and simulated annealing.
*   **Model file** (`.multical`): everything needed to estimate new samples, stored as plain numbers. Each estimate
    comes with the model's typical error and a warning when the spectrum is unlike the calibration ones or the value is
    outside the calibrated range.
*   **Q × T² (influence) plots**: of the calibration samples, to find spectra unlike the others, and of the estimated
    spectra against the limits of the model.
*   **Export**: model, tables (CSV), figures, PDF report and the configuration used.
*   **Projects** (`.mcproj`) keep the choices made in the interface, so a calibration can be reopened and repeated.
*   **Plugins**: new pretreatments, models, validation schemes and variable selections are added as `.py` files, with no
    change to the program. See [`plugins/LEIA-ME.md`](plugins/LEIA-ME.md).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/LadabioMPAR/Multical.git
cd Multical
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

The desktop interface also needs Qt:

```bash
pip install PySide6-Essentials
```

## Desktop program

```bash
python abrir_multical.py
```

The first screen offers two paths: **create a model** (import, inspect, pretreat, train and validate, read the results,
export) and **use a model** (open a `.multical` file and estimate new spectra). The switch *Avançado* shows the choice
of model, validation, variable selection and the editing of the pretreatment recipe. The user guide, in Portuguese, is
[`docs/guia_do_usuario.html`](docs/guia_do_usuario.html).

To make the Windows program (a folder with `Multical.exe`, a zip of it and, when
[Inno Setup](https://jrsoftware.org/isinfo.php) is installed, an installer):

```bash
pip install pyinstaller PySide6-Essentials
python packaging/build.py
```

The result goes to `dist/`. `Multical.exe --autoteste report.txt` checks an installed copy and writes what it found.

Windows Smart App Control may refuse an unsigned program. When a code-signing certificate is available, set
`MULTICAL_CERT_THUMBPRINT` (certificate in the Windows store) or `MULTICAL_CERT_PFX` and `MULTICAL_CERT_PASSWORD`
before building, and the program files, the installer and the uninstaller are signed with `signtool` (Windows SDK).
The details are at the top of `packaging/build.py`.

## Python API

```python
import numpy as np
from multical import api

# 40 spectra of 300 points, two analytes (replace with your own data)
rng = np.random.RandomState(0)
axis = np.linspace(4000.0, 8000.0, 300)
conc = rng.uniform(0.5, 10.0, size=(40, 2))
bands = np.stack([np.exp(-0.5 * ((axis - c) / 300.0) ** 2) for c in (5000.0, 6800.0)])
spectra = conc @ bands + rng.normal(0.0, 0.02, size=(40, 300))
data = api.Dataset(spectra, axis, conc, analytes=["glucose", "xylose"], units="g/L")

config = api.RunConfig(
    pretreatment=[{"id": "sg", "params": {"radius": 7, "poly_order": 2, "deriv": 1}}],
    model={"id": "pls", "params": {"max_k": 10}},
    validation={"id": "kfold", "params": {"folds": 5}},
    variable_selection={"id": "vip", "params": {}},      # optional; leave it out to use every point
)
result = api.run_calibration(data, config)
for name, res in result.per_analyte.items():
    print(name, "RMSECV", round(res.best_rmsecv, 3), "latent variables", res.best_k, "points used", res.n_selected)

api.save_model(result, "model.multical")
model = api.load_model("model.multical")
print(api.predict(model, spectra[:3], axis))
```

`api.MODELS`, `api.PRETREATMENTS`, `api.SPLITTERS` and `api.SELECTORS` list what is available, each with the
description of its parameters. Run the snippet from the repository root with `src` on the Python path
(`set PYTHONPATH=src`, or `pip install -e .`).

Note that this layer fits **one model per analyte**, so its errors are not the ones of the joint PLS fitted by the
research scripts.

## Research scripts

The original workflow is unchanged. Edit the configuration at the top of a script and run it:

```bash
python scripts/calibration/run_calibration.py          # calibration; plots and model go to results/
python scripts/calibration/run_variable_selection.py   # VIP, PSO or simulated annealing
python scripts/inference/run_inference.py              # estimates for new spectra with a saved model
```

`python gui_main.py` opens the earlier Tk interface to the same engine.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests use synthetic data. They include comparisons of the API with the original calculation code (PLS, PCR, SPA,
SVR, VIP, PSO and simulated annealing).

## Documentation

*   User guide of the desktop program (Portuguese): [`docs/guia_do_usuario.html`](docs/guia_do_usuario.html)
*   Writing plugins (Portuguese): [`plugins/LEIA-ME.md`](plugins/LEIA-ME.md)
*   Script-based workflow and theoretical background: [readthedocs](https://multical.readthedocs.io/en/latest/)
