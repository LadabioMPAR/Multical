"""
make_paper_figures.py
=====================
Publication figures (Elsevier style) for NIR-monitored enzymatic hydrolysis.
numpy + matplotlib (+ Pillow for .tif). No pandas. Run build_paper_data.py first.

Every figure is saved .pdf + .svg + .tif@600 AND a companion *_data.csv with the data
plotted. Assays: exp4-6 = Assay 1-3 (calibration, 20 g); exp7 = Assay 4 (test, 15 g).

Figures:
  fig_spectra_raw            raw (after-cut) calibration spectra
  fig_spectra_pretreated     Cut + Savitzky-Golay (1st deriv) spectra
  fig1_kinetics              glucose & xylose vs time (HPLC), calibration vs test
  fig2_parity                full-spectrum PLS parity (calibration LOBO + test)
  fig3_monitoring_exp7       continuous NIR trajectory + RMSEP band + AD screening
  fig_parity_<model>         PLS / VIP-PLS / PSO-PLS : parity (cross-validation + test)
  fig_rmsecv_<model>         PLS / VIP-PLS / PSO-PLS : RMSECV vs latent variables (1-15)
  fig_selvars_<model>        VIP-PLS / PSO-PLS       : selected wavenumbers

Style: glucose = red (#C0392B), xylose = purple (#6C3483); circle/triangle, solid/dashed.
"""
import os
import csv
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------- config
DEFAULTS = dict(outdir="figures", datadir="data_fig")
RMSEP_OVERRIDE = None
MA_WINDOW = 5
SECONDARY_Y_FIG3 = False
AD_MODE = "none"                      # "none" (ignore AD, plot all pts) | "flag" (T2 screen+shade) | "filter" (strict T2&Q)
TEST_ASSAY = "Assay 4"               # the independent test assay (exp7)

# ----------------------------------------------------------------- style
COLORS = {"glucose": "#C0392B", "xylose": "#6C3483"}
MARKERS = {"glucose": "o", "xylose": "^"}
LINESTYLES = {"glucose": "-", "xylose": "--"}
LABELS = {"glucose": "Glucose", "xylose": "Xylose"}
ANALYTES = ["glucose", "xylose"]
MODELS = ["PLS", "VIP", "PSO"]
MODEL_LABEL = {"PLS": "PLS (full spectrum)", "VIP": "VIP-PLS", "PSO": "PSO-PLS"}
ASSAY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]   # NOT red/purple
GROUP_KEYS = ("assay", "batch", "exp", "experiment", "run")
MM = 1.0 / 25.4
CONC = "Concentration (g L$^{-1}$)"


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "lines.linewidth": 1.1, "lines.markersize": 5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def figsize_mm(w, h):
    return (w * MM, h * MM)


def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def save_fig(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    try:
        fig.savefig(os.path.join(outdir, f"{name}.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        tif = " + .tif@600"
    except Exception as e:
        tif = f" (.tif skipped: {e})"
    print(f"  saved {name}{tif}")


def write_data_csv(outdir, name, fieldnames, rows):
    with open(os.path.join(outdir, f"{name}_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def moving_average(values, window):
    v = np.asarray(values, float)
    if not window or window <= 1 or v.size == 0:
        return v
    half = window // 2
    return np.array([v[max(0, i - half):min(v.size, i + half + 1)].mean() for i in range(v.size)])


def read_table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, dtype=float):
    return np.array([dtype(r[key]) for r in rows])


def read_matrix(path):
    with open(path) as f:
        rd = csv.reader(f); header = next(rd)
        data = np.array([[float(v) for v in row] for row in rd])
    return data[:, 0], data[:, 1:].T, header[1:]


def _r2(y, yp):
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return 1 - float(np.sum((y - yp) ** 2)) / sst


def _rmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)))


# ===================================================== spectra (separate figs)
def figure_spectra(matrix_csv, title, ylabel, name, outdir):
    wl, S, names = read_matrix(matrix_csv)
    assays = sorted(set(n.rsplit("_", 1)[0] for n in names))
    cmap = {e: ASSAY_COLORS[i % len(ASSAY_COLORS)] for i, e in enumerate(assays)}
    fig, ax = plt.subplots(figsize=figsize_mm(90, 70))
    for i, n in enumerate(names):
        ax.plot(wl, S[i], color=cmap[n.rsplit("_", 1)[0]], lw=0.5, alpha=0.7)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    ax.legend(handles=[Line2D([0], [0], color=cmap[e], lw=1.4, label=e) for e in assays],
              frameon=False, loc="best", title="Assay")
    despine(ax); fig.tight_layout()
    save_fig(fig, name, outdir); plt.close(fig)
    print(f"    ({name} data: {os.path.basename(matrix_csv)})")


# ===================================================== Fig 1 kinetics
def _group(rows, key):
    gd = {}
    for r in rows:
        gd.setdefault(r[key], []).append(r)
    return sorted(gd.items())


def figure1_kinetics(hplc_csv, outdir):
    rows = read_table(hplc_csv)
    gkey = next((k for k in GROUP_KEYS if k in rows[0]), None)
    groups = [(None, rows)] if gkey is None else _group(rows, gkey)
    fig, ax = plt.subplots(figsize=figsize_mm(90, 70))
    tmax = 0.0
    for gname, g in groups:
        t = col(g, "time"); o = np.argsort(t); t = t[o]; tmax = max(tmax, t.max())
        is_test = (gname is not None and str(gname) == TEST_ASSAY)
        for a in ANALYTES:
            ax.plot(t, col(g, a)[o], marker=MARKERS[a], linestyle=LINESTYLES[a], color=COLORS[a],
                    markersize=5, linewidth=1.0, markerfacecolor=("none" if is_test else COLORS[a]),
                    markeredgecolor=COLORS[a], markeredgewidth=0.9)
    ax.set_xlabel("Time (h)"); ax.set_ylabel(CONC); ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    if tmax >= 24:
        ax.axvline(24, color="0.6", linestyle=":", linewidth=0.8)
    handles = [Line2D([0], [0], color=COLORS[a], marker=MARKERS[a], linestyle=LINESTYLES[a],
                      markersize=5, linewidth=1.0, label=LABELS[a]) for a in ANALYTES]
    if gkey is not None:
        handles += [Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="0.3",
                           label="Calibration (Assays 1-3, 20 g)"),
                    Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="none",
                           label=f"Test ({TEST_ASSAY}, 15 g)")]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    despine(ax); fig.tight_layout()
    save_fig(fig, "fig1_kinetics", outdir); plt.close(fig)
    write_data_csv(outdir, "fig1_kinetics", list(rows[0].keys()), rows)


# ===================================================== Fig 2 full-spectrum parity
def figure2_parity(parity_csv, outdir, rmsep):
    rows = read_table(parity_csv)
    fig, axes = plt.subplots(1, 2, figsize=figsize_mm(190, 92))
    for ax, a, letter in zip(axes, ANALYTES, ("a", "b")):
        sub = [r for r in rows if r["analyte"] == a]
        cal = [r for r in sub if r["set"].lower() == "calibration"]
        test = [r for r in sub if r["set"].lower() == "test"]
        _parity_panel(ax, a, letter, cal, "Calibration", test, rmsep[a])
    fig.tight_layout()
    save_fig(fig, "fig2_parity", outdir); plt.close(fig)
    write_data_csv(outdir, "fig2_parity", ["set", "analyte", "measured", "predicted"], rows)


def _parity_panel(ax, a, letter, base_rows, base_label, test_rows, rmsep_val):
    m_all = np.concatenate([col(base_rows, "measured"), col(test_rows, "measured")])
    p_all = np.concatenate([col(base_rows, "predicted"), col(test_rows, "predicted")])
    lo, hi = float(min(m_all.min(), p_all.min())), float(max(m_all.max(), p_all.max()))
    pad = (hi - lo) * 0.06 or 1.0; lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.scatter(col(base_rows, "measured"), col(base_rows, "predicted"), marker=MARKERS[a],
               facecolors=COLORS[a], edgecolors="none", s=26, alpha=0.85, label=base_label, zorder=2)
    ax.scatter(col(test_rows, "measured"), col(test_rows, "predicted"), marker=MARKERS[a],
               facecolors="none", edgecolors="black", linewidths=0.9, s=32,
               label=f"Independent test ({TEST_ASSAY})", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Measured {LABELS[a].lower()} (g L$^{{-1}}$)")
    ax.set_ylabel(f"Predicted {LABELS[a].lower()} (g L$^{{-1}}$)")
    ax.text(0.05, 0.95, f"Test RMSEP = {rmsep_val:.2f} g L$^{{-1}}$", transform=ax.transAxes,
            va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8))
    ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9, fontweight="bold")
    ax.legend(frameon=False, loc="lower right"); despine(ax)


# ===================================================== per-model figures
def figure_model_parity(model_parity_csv, model, outdir):
    rows = [r for r in read_table(model_parity_csv) if r["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=figsize_mm(190, 92))
    for ax, a, letter in zip(axes, ANALYTES, ("a", "b")):
        sub = [r for r in rows if r["analyte"] == a]
        cv = [r for r in sub if r["set"] == "cv"]
        test = [r for r in sub if r["set"] == "test"]
        rmsep_val = _rmse(col(test, "measured"), col(test, "predicted"))
        _parity_panel(ax, a, letter, cv, "Cross-validation", test, rmsep_val)
    fig.suptitle(MODEL_LABEL[model], fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"fig_parity_{model.lower()}", outdir); plt.close(fig)
    write_data_csv(outdir, f"fig_parity_{model.lower()}",
                   ["model", "analyte", "set", "measured", "predicted"], rows)


def figure_model_rmsecv(model_rmsecv_csv, model, outdir):
    rows = [r for r in read_table(model_rmsecv_csv) if r["model"] == model]
    fig, ax = plt.subplots(figsize=figsize_mm(90, 70))
    for a in ANALYTES:
        sub = sorted([r for r in rows if r["analyte"] == a], key=lambda r: int(r["lv"]))
        lv = col(sub, "lv"); rm = col(sub, "rmsecv")
        ax.plot(lv, rm, color=COLORS[a], linestyle=LINESTYLES[a], marker=MARKERS[a],
                markersize=4, linewidth=1.1, label=LABELS[a])
        sel = [r for r in sub if int(r["selected"]) == 1]
        if sel:
            ax.plot(int(sel[0]["lv"]), float(sel[0]["rmsecv"]), "*", color=COLORS[a],
                    markersize=14, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.set_xlabel("Latent variables (LVs)"); ax.set_ylabel("RMSECV (g L$^{-1}$)")
    ax.set_title(f"{MODEL_LABEL[model]}  ($\\bigstar$ = selected LVs)", loc="left", fontsize=8)
    ax.legend(frameon=False, loc="best"); despine(ax); fig.tight_layout()
    save_fig(fig, f"fig_rmsecv_{model.lower()}", outdir); plt.close(fig)
    write_data_csv(outdir, f"fig_rmsecv_{model.lower()}",
                   ["model", "analyte", "lv", "rmsecv", "selected"], rows)


def figure_model_selvars(model_selvars_csv, model, outdir):
    rows = [r for r in read_table(model_selvars_csv) if r["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=figsize_mm(190, 75))
    for ax, a, letter in zip(axes, ANALYTES, ("a", "b")):
        sub = [r for r in rows if r["analyte"] == a]
        wv = col(sub, "wavenumber"); ms = col(sub, "mean_abs"); sel = col(sub, "selected") > 0.5
        ax.plot(wv, ms, "k-", alpha=0.4, lw=0.9, label="Mean spectrum")
        ax.scatter(wv[sel], ms[sel], color=COLORS[a], s=8, alpha=0.8, label=f"Selected (n={int(sel.sum())})")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Absorbance (1st deriv.)")
        ax.set_title(f"({letter}) {LABELS[a]}", loc="left")
        ax.legend(frameon=False, loc="best"); despine(ax)
    fig.suptitle(f"{MODEL_LABEL[model]} - selected wavenumbers", fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"fig_selvars_{model.lower()}", outdir); plt.close(fig)
    write_data_csv(outdir, f"fig_selvars_{model.lower()}",
                   ["model", "analyte", "wavenumber", "selected", "mean_abs"], rows)


# ===================================================== Fig 3 monitoring
def figure3_monitoring(nir_csv, hplc_csv, outdir, rmsep, ad_limits_csv=None):
    XMAX = 24.0
    nir = read_table(nir_csv)
    t = col(nir, "time"); o = np.argsort(t); t = t[o]
    pred = {a: col(nir, f"{a}_pred")[o] for a in ANALYTES}
    indom = {a: col(nir, f"{a}_indomain")[o] > 0.5 for a in ANALYTES}
    T2 = {a: col(nir, f"{a}_T2")[o] for a in ANALYTES}
    t2lim = {}
    if ad_limits_csv and os.path.exists(ad_limits_csv):
        for r in read_table(ad_limits_csv):
            t2lim[r["analyte"]] = float(r["T2_lim"])
    t2_in = {a: (T2[a] <= t2lim[a]) if a in t2lim else np.ones(t.size, bool) for a in ANALYTES}

    hplc = read_table(hplc_csv)
    gkey = next((k for k in GROUP_KEYS if k in hplc[0]), None)
    if gkey is not None:
        ex = [r for r in hplc if str(r[gkey]) == TEST_ASSAY]
        if ex:
            hplc = ex
    in24 = t <= XMAX
    fig, ax1 = plt.subplots(figsize=figsize_mm(90, 75))
    ax2 = ax1.twinx() if SECONDARY_Y_FIG3 else None
    axes_for = {"glucose": ax1, "xylose": (ax2 if SECONDARY_Y_FIG3 else ax1)}
    for a in ANALYTES:
        ax = axes_for[a]
        if AD_MODE == "none":
            m = in24                                    # no AD screening: keep every point
        elif AD_MODE == "filter":
            m = indom[a] & in24
        else:                                            # "flag"
            m = t2_in[a] & in24
        ta = t[m]; ya = moving_average(pred[a][m], MA_WINDOW); band = rmsep.get(a, 0.0)
        ax.plot(ta, ya, color=COLORS[a], linestyle=LINESTYLES[a], linewidth=1.2, zorder=3)
        ax.fill_between(ta, ya - band, ya + band, color=COLORS[a], alpha=0.15, linewidth=0, zorder=2)
        hh = [r for r in hplc if float(r["time"]) <= XMAX]
        th = col(hh, "time"); ho = np.argsort(th)
        ax.plot(th[ho], col(hh, a)[ho], marker=MARKERS[a], linestyle="none", markerfacecolor=COLORS[a],
                markeredgecolor="black", markeredgewidth=0.6, markersize=5, zorder=4)
        print(f"  Fig3 {a}: plotted {int(m.sum())} pts ({AD_MODE}); strict T2&Q out-of-domain "
              f"{int((~indom[a] & in24).sum())}/{int(in24.sum())}")
    if AD_MODE != "none":
        both = t2_in["glucose"] & t2_in["xylose"] & in24
        startup_end = float(t[both].min()) if both.any() else 0.0
        if startup_end > t[in24].min():
            ax1.axvspan(0, startup_end, color="0.85", alpha=0.6, zorder=0)
    ax1.set_xlabel("Time (h)"); ax1.set_ylabel(CONC); ax1.set_ylim(bottom=0); ax1.set_xlim(0, XMAX)
    if ax2 is not None:
        ax2.set_ylabel(CONC); ax2.set_ylim(bottom=0); ax2.spines["top"].set_visible(False)
    despine(ax1)
    ah = [Line2D([0], [0], color=COLORS[a], marker=MARKERS[a], linestyle=LINESTYLES[a], markersize=5,
                 markerfacecolor=COLORS[a], markeredgecolor="black", label=LABELS[a]) for a in ANALYTES]
    eh = [Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.2, label="NIR prediction"),
          Patch(facecolor="0.3", alpha=0.15, label="$\\pm$ RMSEP band"),
          Line2D([0], [0], color="0.3", marker="o", linestyle="none", markerfacecolor="0.3",
                 markeredgecolor="black", label=f"HPLC reference ({TEST_ASSAY})")]
    if AD_MODE != "none":
        eh.append(Patch(facecolor="0.85", alpha=0.6, label="Start-up (out-of-domain)"))
    leg1 = ax1.legend(handles=ah, frameon=False, loc="upper left", fontsize=7); ax1.add_artist(leg1)
    ax1.legend(handles=eh, frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig3_monitoring_exp7", outdir); plt.close(fig)
    idx = np.where(in24)[0]
    gma = moving_average(pred["glucose"][in24], MA_WINDOW); xma = moving_average(pred["xylose"][in24], MA_WINDOW)
    out = [{"time": round(float(t[j]), 4),
            "glucose_pred": round(float(pred["glucose"][j]), 4), "glucose_pred_ma": round(float(gma[k]), 4),
            "glucose_in_T2": int(t2_in["glucose"][j]), "glucose_in_T2Q": int(indom["glucose"][j]),
            "xylose_pred": round(float(pred["xylose"][j]), 4), "xylose_pred_ma": round(float(xma[k]), 4),
            "xylose_in_T2": int(t2_in["xylose"][j]), "xylose_in_T2Q": int(indom["xylose"][j])}
           for k, j in enumerate(idx)]
    write_data_csv(outdir, "fig3_monitoring_exp7",
                   ["time", "glucose_pred", "glucose_pred_ma", "glucose_in_T2", "glucose_in_T2Q",
                    "xylose_pred", "xylose_pred_ma", "xylose_in_T2", "xylose_in_T2Q"], out)


def test_rmsep_from_parity(parity_csv):
    rows = read_table(parity_csv); out = {}
    for a in ANALYTES:
        tt = [r for r in rows if r["analyte"] == a and r["set"].lower() == "test"]
        if tt:
            out[a] = _rmse(col(tt, "measured"), col(tt, "predicted"))
    return out


# ===================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=DEFAULTS["outdir"]); ap.add_argument("--datadir", default=DEFAULTS["datadir"])
    args = ap.parse_args(); set_style(); d = args.datadir; o = args.outdir

    P = {k: os.path.join(d, v) for k, v in dict(
        hplc="hplc.csv", parity="parity.csv", nir="nir_exp7.csv", adlim="ad_limits.csv",
        raw="spectra_raw_calibration.csv", pre="spectra_pretreated_calibration.csv",
        mpar="model_parity.csv", mrm="model_rmsecv.csv", msel="model_selected_vars.csv").items()}

    rmsep = dict(RMSEP_OVERRIDE) if RMSEP_OVERRIDE else (
        test_rmsep_from_parity(P["parity"]) if os.path.exists(P["parity"]) else {})
    print("Building figures...")
    if os.path.exists(P["raw"]):
        figure_spectra(P["raw"], "Raw spectra (after cut)", "Absorbance", "fig_spectra_raw", o)
    if os.path.exists(P["pre"]):
        figure_spectra(P["pre"], "Pretreated spectra (Cut + SG, 1st deriv.)", "Absorbance (1st deriv.)",
                       "fig_spectra_pretreated", o)
    if os.path.exists(P["hplc"]):
        figure1_kinetics(P["hplc"], o)
    if os.path.exists(P["parity"]):
        figure2_parity(P["parity"], o, rmsep)
    if os.path.exists(P["nir"]) and os.path.exists(P["hplc"]):
        figure3_monitoring(P["nir"], P["hplc"], o, rmsep, P["adlim"])
    # per-model sets
    for M in MODELS:
        if os.path.exists(P["mpar"]):
            figure_model_parity(P["mpar"], M, o)
        if os.path.exists(P["mrm"]):
            figure_model_rmsecv(P["mrm"], M, o)
        if M in ("VIP", "PSO") and os.path.exists(P["msel"]):
            figure_model_selvars(P["msel"], M, o)
    print(f"Done. Figures + *_data.csv in: {o}/")


if __name__ == "__main__":
    main()
