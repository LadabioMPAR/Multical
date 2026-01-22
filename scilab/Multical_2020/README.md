# Multical_2020

-- Under development -- 

Scilab codes for multivariate calibration using NIR/UV-Vis spectra.
Outliers analysis: "leverage" (infer_ILS) and t-test (multi_ILS).

# Files:

main_multi_ILS.sce (new version) - spectra pretreatment and model optimization with leave-one-out Cross-Validation.

main_infer_ILS.sce (new version) - spectra pretreatment, model fitting and model inference.

pretrat_analysis.sce - spectra pretreatment -- new spectra saved in a separate file

=============

multi_ILS.sce (old version) - spectra pretreatment and model optimization with leave-one-out Cross-Validation.

infer_ILS.sce (old version) - spectra pretreatment, model fitting and model inference.

=============

# Folders:

/lib_multi: folder with functions needed
- diffmeu.sci: evaluates 1st and 2nd derivatives by differences - 2nd order approximation
- alisar.sci: smoothing with moving average
- loess.sci: smoothing with loess
- polyfit.sci: used by loess.sci
- pls_model.sci: fits pls model and evaluates model
- spa_model.sci: fits spa model and evaluates model
- pcr_model.sci: fits pcr model and evaluates model
- pls.sci: evaluates pls parameters
- spa_clean.sci: evaluates spa sequence
- func_analysis.sci: CLS (Lambert-Beer) and Principal component (PCA) analysis
- func_pretreatment.sci: pretreats data
- zscore.sci: normalize data -- subtracts mean and divides by the standard deviation
- func_multi_ILS.sci: function for spectra pretreatment, model fitting and model inference.
- func_infer_ILS.sci: function for spectra pretreatment, model fitting and model inference.
