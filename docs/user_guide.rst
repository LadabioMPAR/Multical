User Guide
==========

This guide provides a comprehensive overview of how to use **MultiCal** for multivariate calibration, variable selection, and inference.

.. contents:: Table of Contents
   :depth: 2
   :local:

Installation
------------

Ensure you have Python installed (3.8+ recommended).

1.  **Clone the repository**:

    .. code-block:: bash

        git clone https://github.com/LadabioMPAR/Multical.git
        cd Multical

2.  **Install dependencies**:

    .. code-block:: bash

        pip install -r requirements.txt

Data Preparation
----------------

MultiCal works with text-based data files. You typically need two types of files for calibration:

1.  **Spectra File (X-Block)**: Contains the absorbance or intensity data.
2.  **Concentration File (Y-Block)**: Contains the reference values for the analytes.

Spectra File Format
~~~~~~~~~~~~~~~~~~~
-   **File extension**: `.txt` (Tab or space separated).
-   **Header**: The first row must define the wavelengths/wavenumbers.
-   **Columns**:
    -   Column 0: Sample ID or Time (ignored by the loader).
    -   Columns 1+: Spectral data corresponding to the wavelengths in the header.

**Example** (`data/spectra.txt`):

::

    Time    400.0   402.0   404.0   ...
    10.0    0.123   0.125   0.128   ...
    20.0    0.140   0.142   0.145   ...

Concentration File Format
~~~~~~~~~~~~~~~~~~~~~~~~~
-   **File extension**: `.txt`.
-   **No Header** (usually).
-   **Columns**:
    -   Column 0: Sample ID or Time (ignored if more than 1 column exists).
    -   Columns 1+: Concentration values for each analyte.

**Example** (`data/reference.txt`):

::

    10.0    1.5     5.2     0.8
    20.0    1.6     5.1     0.9

.. note::
   The number of rows (samples) in the Spectra file must match the Concentration file.

Workflow 1: Calibration
-----------------------

The calibration workflow is controlled by ``run_calibration.py``.

1.  **Edit Configuration**: Open ``run_calibration.py`` and locate the **CONFIGURATION** section.

    .. code-block:: python

        DATA_FILES = [
            ('data/ref1.txt', 'data/spec1.txt'),
            ('data/ref2.txt', 'data/spec2.txt'),
        ]
        MODEL_TYPE = 1  # 1=PLS, 2=SPA, 3=PCR
        ANALYTES = ['Glucose', 'Ethanol']  # Match columns in reference file

2.  **Configure Pretreatment**: define the list of operations to apply to the spectra.

    .. code-block:: python

        PRETREATMENT = [
            ['Cut', 900, 1800, 1],  # Keep wavelengths between 900 and 1800
            ['SG', 7, 2, 1, 1],     # Savitzky-Golay (Window=7, Poly=2, Deriv=1)
            ['SNV', 1],             # Standard Normal Variate
        ]

3.  **Run the script**:

    .. code-block:: bash

        python run_calibration.py

    **Outputs**:
    -   Console: CV statistics (RMSECV, R², etc.).
    -   Plots: Saved in ``results/`` folder.
    -   Model: ``results/model_calibration.pkl``.

Workflow 2: Variable Selection
------------------------------

To identify the most important wavelengths, use ``run_variable_selection.py``.

1.  **Edit Configuration**:
    -   Set `SELECTION_METHOD` to `'VIP'`, `'SA'`, or `'PSO'`.
    -   Configure the method-specific parameters (e.g., `VIP_THRESHOLDS`, `SA_PARAMS`).
    -   Ensure `DATA_FILES` and `PRETREATMENT` match your calibration goals.

2.  **Run the script**:

    .. code-block:: bash

        python run_variable_selection.py

    **Outputs**:
    -   Plots showing selected variables vs. RMSECV.
    -   Best subset of variables.
    -   Saved model with selected variables: ``results_var_selection/model_variable_selection.pkl``.

Workflow 3: Inference
---------------------

Use ``run_inference.py`` to predict concentrations for new spectral data using a trained model.

1.  **Edit Configuration**:
    -   Set `MODEL_PATH` to your trained `.pkl` file.
    -   Set `INFERENCE_FILES`. The reference file can be `None` or a dummy path if you only have spectra and want predictions.

2.  **Run the script**:

    .. code-block:: bash

        python run_inference.py

    **Outputs**:
    -   Predicted concentrations saved to text files (optional, depends on script logic).
    -   Time-series plots comparing Prediction vs Reference (if available).

Preprocessing Reference
-----------------------

The `PRETREATMENT` list accepts specific codes for various spectral transformations.

+--------------+---------------------------------------------+-----------------------------------------------------------+
| Method       | Syntax                                      | Description                                               |
+==============+=============================================+===========================================================+
| **Cut**      | ``['Cut', min, max, ..., plot]``            | Selects wavelength ranges. Can specify multiple ranges.   |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **SG**       | ``['SG', window, poly, deriv, plot]``       | Savitzky-Golay smoothing and derivatives.                 |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **SNV**      | ``['SNV', plot]``                           | Standard Normal Variate normalization.                    |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **MSC**      | ``['MSC', plot]``                           | Multiplicative Scatter Correction.                        |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **EMSC**     | ``['EMSC', degree, plot]``                  | Extended MSC with polynomial baseline correction.         |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **Deriv**    | ``['Deriv', order, plot]``                  | Simple finite difference derivative (1st or 2nd).         |
+--------------+---------------------------------------------+-----------------------------------------------------------+
| **Loess**    | ``['Loess', alpha, order, plot]``           | Local regression smoothing.                               |
+--------------+---------------------------------------------+-----------------------------------------------------------+
|**MeanCenter**| ``['MeanCenter', plot]``                    | Subtracts the column mean (centering).                    |
+--------------+---------------------------------------------+-----------------------------------------------------------+
