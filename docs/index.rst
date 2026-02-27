.. image:: logo.png
   :alt: MultiCal Logo
   :align: center
   :width: 300px

MultiCal Documentation
======================

**MultiCal** is a robust Python Library for Chemometrics and Multivariate Calibration. It provides a comprehensive toolkit for building predictive models from spectroscopic data (NIR, Raman, MIR, etc.), streamlining the workflow from raw spectra preprocessing to variable selection and model deployment.

.. note::
   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   modules

Key Features
------------

MultiCal offers a flexible environment for spectral analysis:

*   **Calibration Algorithms**:
    *   **PLS (Partial Least Squares)**: The industry standard for quantitative spectral analysis.
    *   **PCR (Principal Component Regression)**: Alternative latent variable method.
    *   **SPA (Successive Projections Algorithm)**: For minimizing collinearity and selecting discrete wavelengths.

*   **Variable Selection**:
    *   **VIP (Variable Importance in Projection)**: Identifies the most influential spectral regions.
    *   **Evolutionary Algorithms**: Particle Swarm Optimization (PSO) and Simulated Annealing (SA) for optimizing feature subsets.

*   **Preprocessing Pipeline**:
    *   Comprehensive suite including **Savitzky-Golay** (smoothing/derivatives), **MSC**, **SNV**, and **Normalization**.
    *   Customizable pipeline to chain multiple pretreatment steps.

*   **Workflow Flexibility**:
    *   **Script-Based**: Optimized for batch processing and reproducible research (`run_calibration.py`, etc.).
    *   **GUI-Based**: User-friendly interface for visual inspection and quick model building.

Quick Install
-------------

.. code-block:: bash

   git clone https://github.com/LadabioMPAR/Multical.git
   cd Multical
   pip install -r requirements.txt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Support & Contribute
--------------------

*   **Issues**: Report bugs or suggest features on `GitHub <https://github.com/LadabioMPAR/Multical>`_.
*   **License**: Licensed under the GNU General Public License v3.0.
