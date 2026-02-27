# MultiCal: Multivariate Calibration Tool

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

## Quick Start Guide

MultiCal can be used via its Graphical User Interface (GUI) or through configuring and running Python scripts.

### Option 1: Graphical User Interface (Recommended)
Launch the interactive interface:
```bash
python gui_main.py
```

### Option 2: Script-Based Workflow
For reproducible research and batch processing, use the provided `run_` scripts.

1.  **Configure Calibration**:
    Edit `run_calibration.py` to set your data files and parameters.
    ```python
    # Example Configuration in run_calibration.py
    DATA_FILES = [
        ('data/reference.txt', 'data/spectra.txt'),
    ]
    MODEL_TYPE = 1  # 1=PLS
    ```

2.  **Run Calibration**:
    ```bash
    python run_calibration.py
    ```
    This generates performance plots and saves the model to `results/`.

3.  **Variable Selection (Optional)**:
    Identify key wavelengths:
    ```bash
    python run_variable_selection.py
    ```

4.  **Run Inference**:
    Predict concentrations for new spectra using a trained model:
    ```bash
    python run_inference.py
    ```

## Documentation

Full documentation, including detailed API references and theoretical background, is available in the `docs/` folder or built via Sphinx.
