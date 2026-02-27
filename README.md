# MultiCal: Multivariate Calibration Tool
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/multical/badge/?version=latest)](https://multical.readthedocs.io/en/latest/?badge=latest)

**MultiCal** is a comprehensive Python package for Chemometrics and Multivariate Calibration, designed to generate robust predictive models for biochemical processes from spectroscopic data (NIR, Raman, MIR, etc.). It streamlines the workflow from raw spectra to deployed inference models.

## Key Features

*   **Algorithms**: Partial Least Squares (PLS), Principal Component Regression (PCR), Successive Projections Algorithm (SPA).
*   **Variable Selection**: VIP scores, Particle Swarm Optimization (PSO), Simulated Annealing (SA).
*   **Preprocessing**: Savitzky-Golay, MSC, SNV, Normalization.
*   **Interfaces**: 
    *   **Graphical User Interface (GUI)** for interactive analysis.
    *   **Script-based Engine** for automated/batch processing.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LadabioMPAR/Multical.git
    cd Multical
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    
    # Linux/Mac
    source venv/bin/activate
    
    # Windows
    venv\Scripts\activate
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Option 1: GUI (Recommended for beginners)
Launch the interactive interface:
```bash
python gui_main.py
```

### Option 2: Scripts (For automation)
1.  **Configure**: Edit settings in `run_calibration.py` (File paths, Model type).
2.  **Train**:
    ```bash
    python run_calibration.py
    ```
3.  **Predict**:
    ```bash
    python run_inference.py
    ```

## Documentation

Full documentation is available at [ReadTheDocs](https://multical.readthedocs.io/en/latest/).
