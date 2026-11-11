"""
comparison_ftest_table.py — three-method table with F-test LV throughout (NON-DESTRUCTIVE)
==========================================================================================
Rebuilds the PLS full-spectrum rows of the three-method comparison using LV chosen by the
Osten F-test (alpha=0.05), so PLS is consistent with VIP-PLS and PSO-PLS (which already use
the F-test). Only the PLS rows are recomputed; VIP/PSO rows are carried over from the existing
generated table (figures/RUN_SUMMARY.md, from build_paper_data.py) since they are already
F-test-based. Reuses production functions verbatim (build_paper_data.cv_curve / pls_predict;
var_selection.select_k_ftest). Split identical to that table: cal = Assay 1-3 (exp4/5/6),
test = Assay 4 (exp7). Preprocessing Cut(4400-7500)+SG(7,2,1).

Output -> results_comparison_ftest/ only. Nothing in results/ or data_fig/ is touched.
Run:  MPLBACKEND=Agg py scripts/plotting/comparison_ftest_table.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import scripts.plotting.build_paper_data as B
from src.multical.var_selection import select_k_ftest

OUT = "results_comparison_ftest"
CAL = ["exp4", "exp5", "exp6"]      # Assay 1-3
TEST = "exp7"                        # Assay 4
ANALYTE_COL = {"glucose": 2, "xylose": 3}   # index in load_refe tuple (t, cb, gl, xy)

# VIP/PSO rows already F-test-based, carried from figures/RUN_SUMMARY.md (build_paper_data.py):
CARRIED = {
    ("VIP-PLS", "glucose"): dict(n_vars=354, lv=4, rmsecv=0.869, rmsep=1.228),
    ("PSO-PLS", "glucose"): dict(n_vars=1714, lv=4, rmsecv=1.048, rmsep=1.078),
    ("VIP-PLS", "xylose"):  dict(n_vars=384, lv=3, rmsecv=0.386, rmsep=2.054),
    ("PSO-PLS", "xylose"):  dict(n_vars=1812, lv=4, rmsecv=0.513, rmsep=0.872),
}
# Previous PLS rows (fixed LV) for the before/after diff:
OLD_PLS = {"glucose": dict(lv=6, rmsecv=1.174, rmsep=1.116),
           "xylose":  dict(lv=4, rmsecv=0.541, rmsep=0.998)}


def main():
    os.makedirs(OUT, exist_ok=True)
    A_cal = np.vstack([B.load_nonda(e)[2] for e in CAL])
    A_test = B.load_nonda(TEST)[2]
    n_vars = A_cal.shape[1]

    pls_rows = {}
    print(f"PLS full-spectrum ({n_vars} vars), cal=Assay 1-3, test=Assay 4 — LV by F-test:")
    for a, col in ANALYTE_COL.items():
        y_cal = np.concatenate([B.load_refe(e)[col] for e in CAL])
        y_test = B.load_refe(TEST)[col]
        rk, ycv, km = B.cv_curve(A_cal, y_cal, 15)
        lv = int(min(select_k_ftest(rk, n_cal=len(y_cal)), km))
        rmsecv = float(rk[lv - 1])
        tp = B.pls_predict(A_cal, y_cal, A_test, lv)
        rmsep = float(np.sqrt(np.mean((y_test - tp) ** 2)))
        pls_rows[a] = dict(n_vars=n_vars, lv=lv, rmsecv=rmsecv, rmsep=rmsep)
        o = OLD_PLS[a]
        print(f"  {a:8s}: LV {o['lv']}->{lv} | RMSECV {o['rmsecv']:.3f}->{rmsecv:.3f} | "
              f"RMSEP {o['rmsep']:.3f}->{rmsep:.3f}")

    # assemble full table
    order = [("PLS", "glucose"), ("VIP-PLS", "glucose"), ("PSO-PLS", "glucose"),
             ("PLS", "xylose"), ("VIP-PLS", "xylose"), ("PSO-PLS", "xylose")]
    rows = []
    for model, a in order:
        d = pls_rows[a] if model == "PLS" else CARRIED[(model, a)]
        rows.append({"model": model, "analyte": a, "n_vars": d["n_vars"], "lv": d["lv"],
                     "rmsecv": round(d["rmsecv"], 3), "rmsep": round(d["rmsep"], 3),
                     "lv_selection": "F-test (alpha=0.05)",
                     "source": "recomputed here" if model == "PLS" else "build_paper_data (RUN_SUMMARY.md)"})
    with open(os.path.join(OUT, "comparison_ftest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    md = ["# Three-method comparison — F-test LV throughout\n",
          "Split: cal = Assay 1-3 (exp4/5/6), test = Assay 4 (exp7); Cut(4400-7500)+SG(7,2,1). "
          "**All rows use LV chosen by the Osten F-test (alpha=0.05).** PLS rows recomputed here "
          "(reusing build_paper_data.cv_curve/pls_predict); VIP/PSO rows carried from "
          "figures/RUN_SUMMARY.md (already F-test).\n",
          "| Model | Analyte | # vars | LV | RMSECV | RMSEP (test) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['analyte']} | {r['n_vars']} | {r['lv']} | "
                  f"{r['rmsecv']:.3f} | {r['rmsep']:.3f} |")
    md.append("\n**Change vs the fixed-LV table:** only PLS-glucose "
              f"(LV {OLD_PLS['glucose']['lv']}->{pls_rows['glucose']['lv']}, "
              f"RMSECV {OLD_PLS['glucose']['rmsecv']:.3f}->{pls_rows['glucose']['rmsecv']:.3f}, "
              f"RMSEP {OLD_PLS['glucose']['rmsep']:.3f}->{pls_rows['glucose']['rmsep']:.3f}). "
              "PLS-xylose is unchanged (F-test already selects LV 4).\n")
    with open(os.path.join(OUT, "comparison_ftest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> {OUT}/comparison_ftest.csv, {OUT}/comparison_ftest.md")


if __name__ == "__main__":
    main()
