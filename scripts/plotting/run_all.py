"""
run_all.py  -  one-command reproducibility for the paper figures.

Pipeline:
    1) build_paper_data.py  -> data_fig/*.csv  (provenance-correct, LOBO full-spectrum PLS)
    2) make_paper_figures.py -> figures/*.{pdf,svg,tif} + figures/*_data.csv

RAW INPUT FILES (read from data/):
    exp4_refe.txt, exp5_refe.txt, exp6_refe.txt, exp7_refe.txt   (HPLC: time_min, cb, glucose, xylose)
    exp4_nonda.txt, exp5_nonda.txt, exp6_nonda.txt, exp7_nonda.txt  (HPLC-paired NIR spectra)
    exp_07_inf.txt                                               (continuous online NIR of exp7)

Calibration = exp4/5/6 (20 g); independent test = exp7 (15 g).

Run from the project root (Multical/):
    python scripts/plotting/run_all.py
"""
import os
import runpy
import sys

HERE = os.path.dirname(__file__)


def _run(script):
    print(f"\n{'='*70}\n>> {script}\n{'='*70}")
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    _run("build_paper_data.py")
    _run("make_paper_figures.py")
    print("\nAll done. CSVs in data_fig/ and figures/; figures (+ *_data.csv) in figures/")
