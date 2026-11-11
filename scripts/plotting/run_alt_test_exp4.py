"""
run_alt_test_exp4.py
=====================
Exploratory ALTERNATE split: calibration = Assay 2-4 (exp5, exp6, exp7), independent
test = Assay 1 (exp4). Mirrors the manuscript pipeline (build_paper_data.py /
make_paper_figures.py) but swaps which assay is held out, WITHOUT touching the
main paper deliverables (figures/, data_fig/).

Same full-spectrum PLS, same FIXED LV (6 glucose, 4 xylose) as the manuscript's
reference model, for direct comparability with Table tab:model_comparison / Fig 2-3.

Outputs: data_fig_test_exp4/*.csv, figures_test_exp4/*.{pdf,svg,tif}+*_data.csv

Run: python scripts/plotting/run_alt_test_exp4.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scripts.plotting.build_paper_data as B
import scripts.plotting.make_paper_figures as M

CAL = ["exp5", "exp6", "exp7"]
TEST = "exp4"
INF_FILE = os.path.join("data", "exp_04_inf.txt")
OUTDIR_DATA = "data_fig_test_exp4"
OUTDIR_FIG = "figures_test_exp4"
ANALYTES = B.ANALYTES            # {"glucose": 1, "xylose": 2}
LV_FIXED = B.LV_FIXED            # {"glucose": 6, "xylose": 4} - SAME as the manuscript model


def load_inf(path):
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))
    from src.multical.preprocessing.pipeline import apply_pretreatment
    t = raw[:, 0] / 60.0
    spec, _ = apply_pretreatment(B.PRETREAT, raw[:, 1:], wl, plot=False)
    return t, spec


def figure1_kinetics_dynamic_legend(hplc_csv, outdir, test_assay):
    """
    Local copy of make_paper_figures.figure1_kinetics with a DYNAMIC legend (lists the
    actual calibration assays and omits solids-loading numbers, which vary across
    assays in this alternate split). Avoids touching the shared module used for the
    manuscript's primary Figure 1 (where calibration is uniformly 20 g, Assays 1-3).
    """
    rows = M.read_table(hplc_csv)
    gkey = next((k for k in M.GROUP_KEYS if k in rows[0]), None)
    groups = [(None, rows)] if gkey is None else M._group(rows, gkey)
    fig, ax = plt.subplots(figsize=M.figsize_mm(90, 70))
    tmax = 0.0
    cal_assays = []
    for gname, g in groups:
        t = M.col(g, "time"); o = np.argsort(t); t = t[o]; tmax = max(tmax, t.max())
        is_test = (gname is not None and str(gname) == test_assay)
        if gname is not None and not is_test:
            cal_assays.append(str(gname))
        for a in M.ANALYTES:
            ax.plot(t, M.col(g, a)[o], marker=M.MARKERS[a], linestyle=M.LINESTYLES[a], color=M.COLORS[a],
                    markersize=5, linewidth=1.0, markerfacecolor=("none" if is_test else M.COLORS[a]),
                    markeredgecolor=M.COLORS[a], markeredgewidth=0.9)
    ax.set_xlabel("Time (h)"); ax.set_ylabel(M.CONC); ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    if tmax >= 24:
        ax.axvline(24, color="0.6", linestyle=":", linewidth=0.8)
    handles = [Line2D([0], [0], color=M.COLORS[a], marker=M.MARKERS[a], linestyle=M.LINESTYLES[a],
                      markersize=5, linewidth=1.0, label=M.LABELS[a]) for a in M.ANALYTES]
    if gkey is not None:
        cal_label = f"Calibration ({', '.join(sorted(cal_assays))})"
        handles += [Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="0.3",
                           label=cal_label),
                    Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="none",
                           label=f"Test ({test_assay})")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    M.despine(ax); fig.tight_layout()
    M.save_fig(fig, "fig1_kinetics", outdir); plt.close(fig)
    M.write_data_csv(outdir, "fig1_kinetics", list(rows[0].keys()), rows)


def figure3_monitoring_xmax(nir_csv, hplc_csv, outdir, rmsep, name, xmax, ad_limits_csv=None):
    """
    Local copy of make_paper_figures.figure3_monitoring with a PARAMETERISED x-axis
    truncation (XMAX). The shared function hardcodes XMAX=24 (matching exp7's HPLC
    duration); Assay 1 (exp4) HPLC stops at 12 h, so this script truncates there
    instead of touching the shared function used for the manuscript's primary Fig 3.
    """
    nir = M.read_table(nir_csv)
    t = M.col(nir, "time"); o = np.argsort(t); t = t[o]
    pred = {a: M.col(nir, f"{a}_pred")[o] for a in M.ANALYTES}
    indom = {a: M.col(nir, f"{a}_indomain")[o] > 0.5 for a in M.ANALYTES}
    T2 = {a: M.col(nir, f"{a}_T2")[o] for a in M.ANALYTES}
    t2lim = {}
    if ad_limits_csv and os.path.exists(ad_limits_csv):
        for r in M.read_table(ad_limits_csv):
            t2lim[r["analyte"]] = float(r["T2_lim"])
    t2_in = {a: (T2[a] <= t2lim[a]) if a in t2lim else np.ones(t.size, bool) for a in M.ANALYTES}

    hplc = M.read_table(hplc_csv)
    gkey = next((k for k in M.GROUP_KEYS if k in hplc[0]), None)
    if gkey is not None:
        ex = [r for r in hplc if str(r[gkey]) == M.TEST_ASSAY]
        if ex:
            hplc = ex
    inwin = t <= xmax
    fig, ax1 = plt.subplots(figsize=M.figsize_mm(90, 75))
    ax2 = ax1.twinx() if M.SECONDARY_Y_FIG3 else None
    axes_for = {"glucose": ax1, "xylose": (ax2 if M.SECONDARY_Y_FIG3 else ax1)}
    for a in M.ANALYTES:
        ax = axes_for[a]
        if M.AD_MODE == "none":
            m = inwin
        elif M.AD_MODE == "filter":
            m = indom[a] & inwin
        else:
            m = t2_in[a] & inwin
        ta = t[m]; ya = M.moving_average(pred[a][m], M.MA_WINDOW); band = rmsep.get(a, 0.0)
        ax.plot(ta, ya, color=M.COLORS[a], linestyle=M.LINESTYLES[a], linewidth=1.2, zorder=3)
        ax.fill_between(ta, ya - band, ya + band, color=M.COLORS[a], alpha=0.15, linewidth=0, zorder=2)
        hh = [r for r in hplc if float(r["time"]) <= xmax]
        th = M.col(hh, "time"); ho = np.argsort(th)
        ax.plot(th[ho], M.col(hh, a)[ho], marker=M.MARKERS[a], linestyle="none", markerfacecolor=M.COLORS[a],
                markeredgecolor="black", markeredgewidth=0.6, markersize=5, zorder=4)
        print(f"  Fig3({name}) {a}: plotted {int(m.sum())} pts within {xmax:.0f} h ({M.AD_MODE})")
    if M.AD_MODE != "none":
        both = t2_in["glucose"] & t2_in["xylose"] & inwin
        startup_end = float(t[both].min()) if both.any() else 0.0
        if startup_end > t[inwin].min():
            ax1.axvspan(0, startup_end, color="0.85", alpha=0.6, zorder=0)
    ax1.set_xlabel("Time (h)"); ax1.set_ylabel(M.CONC); ax1.set_ylim(bottom=0); ax1.set_xlim(0, xmax)
    if ax2 is not None:
        ax2.set_ylabel(M.CONC); ax2.set_ylim(bottom=0); ax2.spines["top"].set_visible(False)
    M.despine(ax1)
    ah = [Line2D([0], [0], color=M.COLORS[a], marker=M.MARKERS[a], linestyle=M.LINESTYLES[a], markersize=5,
                 markerfacecolor=M.COLORS[a], markeredgecolor="black", label=M.LABELS[a]) for a in M.ANALYTES]
    eh = [Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.2, label="NIR prediction"),
          Patch(facecolor="0.3", alpha=0.15, label="$\\pm$ RMSEP band"),
          Line2D([0], [0], color="0.3", marker="o", linestyle="none", markerfacecolor="0.3",
                 markeredgecolor="black", label=f"HPLC reference ({M.TEST_ASSAY})")]
    if M.AD_MODE != "none":
        eh.append(Patch(facecolor="0.85", alpha=0.6, label="Start-up (out-of-domain)"))
    leg1 = ax1.legend(handles=ah, frameon=False, loc="upper left", fontsize=7); ax1.add_artist(leg1)
    ax1.legend(handles=eh, frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    M.save_fig(fig, name, outdir); plt.close(fig)
    idx = np.where(inwin)[0]
    gma = M.moving_average(pred["glucose"][inwin], M.MA_WINDOW); xma = M.moving_average(pred["xylose"][inwin], M.MA_WINDOW)
    out = [{"time": round(float(t[j]), 4),
            "glucose_pred": round(float(pred["glucose"][j]), 4), "glucose_pred_ma": round(float(gma[k]), 4),
            "glucose_in_T2": int(t2_in["glucose"][j]), "glucose_in_T2Q": int(indom["glucose"][j]),
            "xylose_pred": round(float(pred["xylose"][j]), 4), "xylose_pred_ma": round(float(xma[k]), 4),
            "xylose_in_T2": int(t2_in["xylose"][j]), "xylose_in_T2Q": int(indom["xylose"][j])}
           for k, j in enumerate(idx)]
    M.write_data_csv(outdir, name,
                     ["time", "glucose_pred", "glucose_pred_ma", "glucose_in_T2", "glucose_in_T2Q",
                      "xylose_pred", "xylose_pred_ma", "xylose_in_T2", "xylose_in_T2Q"], out)


def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    print(f"ALT SPLIT: calibration = {[B.ASSAY_NAME[e] for e in CAL]} ({CAL}); "
          f"test = {B.ASSAY_NAME[TEST]} ({TEST})")
    print(f"Full-spectrum PLS, FIXED LV = {LV_FIXED}\n")

    refe = {e: B.load_refe(e) for e in CAL + [TEST]}
    nd = {e: B.load_nonda(e) for e in CAL + [TEST]}
    spec_pre = {e: nd[e][2] for e in nd}
    conc = {e: {"cb": refe[e][1], "glucose": refe[e][2], "xylose": refe[e][3]} for e in refe}

    # ---- hplc.csv ----
    rows = []
    for e in CAL + [TEST]:
        th, _, gl, xy = refe[e]
        rows += [{"time": round(float(t), 4), "glucose": round(float(g), 4),
                  "xylose": round(float(x), 4), "assay": B.ASSAY_NAME[e]} for t, g, x in zip(th, gl, xy)]
    B.write_csv(os.path.join(OUTDIR_DATA, "hplc.csv"), ["time", "glucose", "xylose", "assay"], rows)
    print(f"hplc.csv: {len(rows)} points")

    # ---- parity.csv (out-of-fold LOBO on calibration + test on held-out assay) ----
    A_cal = np.vstack([spec_pre[e] for e in CAL])
    prows, rmsep, r2 = [], {}, {}
    for held in CAL:
        tr = [e for e in CAL if e != held]
        A_tr = np.vstack([spec_pre[e] for e in tr])
        for a in ANALYTES:
            y_tr = np.concatenate([conc[e][a] for e in tr])
            yp = B.pls_predict(A_tr, y_tr, spec_pre[held], LV_FIXED[a])
            prows += [{"set": "calibration", "analyte": a, "measured": round(float(m), 4),
                       "predicted": round(float(p), 4)} for m, p in zip(conc[held][a], yp)]
    for a in ANALYTES:
        y_tr = np.concatenate([conc[e][a] for e in CAL])
        yp = B.pls_predict(A_cal, y_tr, spec_pre[TEST], LV_FIXED[a])
        y_te = conc[TEST][a]
        prows += [{"set": "test", "analyte": a, "measured": round(float(m), 4),
                   "predicted": round(float(p), 4)} for m, p in zip(y_te, yp)]
        rmsep[a] = float(np.sqrt(np.mean((y_te - yp) ** 2)))
        sst = float(np.sum((y_te - y_te.mean()) ** 2)) or 1.0
        r2[a] = 1 - float(np.sum((y_te - yp) ** 2)) / sst
    B.write_csv(os.path.join(OUTDIR_DATA, "parity.csv"), ["set", "analyte", "measured", "predicted"], prows)
    print(f"parity.csv: {sum(r['set']=='calibration' for r in prows)} cal + "
          f"{sum(r['set']=='test' for r in prows)} test")
    for a in ANALYTES:
        print(f"  {a:8s}: LV={LV_FIXED[a]}  RMSEP={rmsep[a]:.4f}  R2_test={r2[a]:.4f}")

    # ---- nir_exp4.csv (continuous trajectory; AD computed but figure ignores it) ----
    t_inf, A_inf = load_inf(INF_FILE)
    nir_rows, ad_rows = [], []
    pred, ad = {}, {}
    print(f"\n--- Applicability-domain screening of {TEST} (continuous, {len(t_inf)} spectra) ---")
    for a in ANALYTES:
        y_tr = np.concatenate([conc[e][a] for e in CAL])
        pred[a] = B.pls_predict(A_cal, y_tr, A_inf, LV_FIXED[a])
        ad[a] = B.applicability_domain(A_cal, y_tr, A_inf, LV_FIXED[a])
        n_ood = int((~ad[a]["in_domain"]).sum())
        print(f"  {a:8s} (A={LV_FIXED[a]}): T2_lim={ad[a]['T2_lim']:.2f}, Q_lim={ad[a]['Q_lim']:.1f} | "
              f"out-of-domain {n_ood}/{len(t_inf)}  (reported only; Fig 3 plots ALL points)")
        ad_rows.append({"analyte": a, "A": LV_FIXED[a], "T2_lim": round(ad[a]["T2_lim"], 4),
                        "Q_lim": round(ad[a]["Q_lim"], 4), "n_out_of_domain": n_ood, "n_total": len(t_inf)})
    for i, t in enumerate(t_inf):
        row = {"time": round(float(t), 4)}
        for a in ANALYTES:
            row[f"{a}_pred"] = round(float(pred[a][i]), 4)
            row[f"{a}_indomain"] = int(ad[a]["in_domain"][i])
            row[f"{a}_T2"] = round(float(ad[a]["T2"][i]), 4)
            row[f"{a}_Q"] = round(float(ad[a]["Q"][i]), 4)
        nir_rows.append(row)
    B.write_csv(os.path.join(OUTDIR_DATA, f"nir_{TEST}.csv"),
                ["time"] + [f"{a}_{s}" for a in ANALYTES for s in ("pred", "indomain", "T2", "Q")], nir_rows)
    B.write_csv(os.path.join(OUTDIR_DATA, "ad_limits.csv"),
                ["analyte", "A", "T2_lim", "Q_lim", "n_out_of_domain", "n_total"], ad_rows)
    print(f"nir_{TEST}.csv ({len(nir_rows)} spectra) written\n")

    # ---- figures (reuse make_paper_figures, monkey-patched for this split) ----
    M.set_style()
    M.TEST_ASSAY = B.ASSAY_NAME[TEST]      # "Assay 1" -> open markers / test legend
    M.AD_MODE = "none"                      # mirror current Fig-3 decision: plot all points
    M.RMSEP_OVERRIDE = None
    os.makedirs(OUTDIR_FIG, exist_ok=True)

    hplc_csv = os.path.join(OUTDIR_DATA, "hplc.csv")
    parity_csv = os.path.join(OUTDIR_DATA, "parity.csv")
    nir_csv = os.path.join(OUTDIR_DATA, "nir_{}.csv".format(TEST))
    adlim_csv = os.path.join(OUTDIR_DATA, "ad_limits.csv")

    rmsep_for_fig = M.test_rmsep_from_parity(parity_csv)
    figure1_kinetics_dynamic_legend(hplc_csv, OUTDIR_FIG, M.TEST_ASSAY)
    M.figure2_parity(parity_csv, OUTDIR_FIG, rmsep_for_fig)
    # Assay 1 (exp4) HPLC stops at 12 h (vs 24 h for exp7) -> truncate the trajectory there.
    figure3_monitoring_xmax(nir_csv, hplc_csv, OUTDIR_FIG, rmsep_for_fig,
                            "fig3_monitoring_exp4", xmax=12.0, ad_limits_csv=adlim_csv)
    print(f"Figures written to {OUTDIR_FIG}/")


if __name__ == "__main__":
    main()
