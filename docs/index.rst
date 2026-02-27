.. image:: logo.png
   :alt: MultiCal Logo
   :align: center
   :width: 300px

MPA Ribeiro's MultiCal
=====================

MultiCal is a Python package designed for multivariate calibration. Our main goal is to generate predictive models of biochemical processes from spectroscopic data.

Key Features
------------

*   **Calibration Algorithms**:
    *   **PLS (Partial Least Squares)**: The core workhorse for quantitative spectral analysis.
    *   **PCR (Principal Component Regression)**: Alternative latent variable method.
    *   **SPA (Successive Projections Algorithm)**: For selecting specific wavelengths to minimize collinearity.
    
*   **Advanced Variable Selection**:
    *   **VIP (Variable Importance in Projection)**: Selects the most influential spectral regions.
    *   **Evolutionary Algorithms**: Includes Particle Swarm Optimization (PSO) and Simulated Annealing (SA) for feature selection.

*   **Preprocessing Pipeline**:
    *   Full suite of spectral pretreatments including Savitzky-Golay (Derivatives/Smoothing), MSC (Multiplicative Scatter Correction), SNV (Standard Normal Variate), and Normalization.
    *   Flexible pipeline to chain multiple steps.

*   **Workflow Automation**:
    *   Dedicated scripts for **Calibration**, **Variable Selection**, and **Inference**.
    *   **GUI (Graphical User Interface)** for interactive model building and analysis.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Main Documentation

   user_guide
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contribute
----------

We welcome contributions! If you find a bug or have a suggestion, feel free to open an issue on our `GitHub repository <https://github.com/LadabioMPAR/Multical>`_.

License
-------

MultiCal is licensed under the GNU General Public License v3.0. See the `LICENSE <https://github.com/LadabioMPAR/Multical/blob/main/LICENSE>`_ file for details.
