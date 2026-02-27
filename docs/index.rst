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

Installation
------------

To install MultiCal, it is recommended to use a virtual environment.

1.  **Clone the repository:**

    .. code-block:: bash

        git clone https://github.com/LadabioMPAR/Multical.git
        cd Multical

2.  **Create and Activate a Virtual Environment (Optional but Recommended):**

    .. code-block:: bash

        # Linux/Mac
        python3 -m venv venv
        source venv/bin/activate

        # Windows
        python -m venv venv
        venv\Scripts\activate

3.  **Install Dependencies:**

    .. code-block:: bash

        pip install -r requirements.txt

Quick Start Guide
-----------------

MultiCal can be used via its Graphical User Interface (GUI) or through configuring and running Python scripts.

Option 1: Graphical User Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to get started is the GUI.

.. code-block:: bash

    python gui_main.py

The GUI allows you to:
1.  Import spectral (X) and reference (Y) data.
2.  Configure pretreatment steps visually.
3.  Select regression models (PLS, etc.).
4.  Run cross-validation and view plots (Predicted vs Measured, RMSECV).
5.  Save the trained model.

Option 2: Script-Based Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reproducible research and batch processing, use the provided `run_` scripts.

1.  **Configure Calibration**:
    Open ``run_calibration.py`` and edit the **CONFIGURATION** section:

    .. code-block:: python

        # Example Configuration in run_calibration.py
        DATA_FILES = [
            ('data/reference.txt', 'data/spectra.txt'),
        ]
        MODEL_TYPE = 1  # 1=PLS
        ANALYTES = ['AnalyteA', 'AnalyteB']

2.  **Run Calibration**:

    .. code-block:: bash

        python run_calibration.py

    This will generate performance plots (RMSECV, Prediction) and save the trained model to the ``results/`` folder.

3.  **Run Inference (Prediction)**:
    Once you have a saved model (e.g., `results/model_calibration.pkl`), use ``run_inference.py`` to predict new samples.

    .. code-block:: bash

        python run_inference.py

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide
   modules
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contribute
----------

We welcome contributions! If you find a bug or have a suggestion, feel free to open an issue on our `GitHub repository <https://github.com/your-repo>`_.

License
-------

MultiCal is licensed under the GNU General Public License v3.0. See the `LICENSE <https://github.com/your-repo/LICENSE>`_ file for details.

Contact
-------

If you have any questions, feel free to reach out to us at <your_email>.

----

*This project is powered by* **Sphinx** *and the* **ReadTheDocs** *theme.*
