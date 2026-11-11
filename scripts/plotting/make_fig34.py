"""
make_fig34.py — Section 3.4 paper figures (NON-DESTRUCTIVE)
===========================================================
  FIG A  figures_paper/fig_monitor_exp4.{pdf,tif}    exp4 monitoring (M_menos1), t <= 12 h
  FIG B  figures_paper/fig_influence_exp7.{pdf,tif}  exp7 influence plot T2 vs Q (M_primario)

Data provenance
---------------
FIG A: regenerated with the SAME pipeline that produced results_refe_vs_inf/exp4_trajectory.csv
       (M_menos1 = train Assay 2-4 nonda; exp4 `_inf_smoothed` stream; Cut 4400-7500 -> SG(15,2,1)
       -> autoscale cal mean/scale; glucose/xylose LV by Osten F-test; AD = Box chi^2 Q-limit +
       F-dist T2-limit, 95%, limits from training only).
       Regenerated (not read from the CSV) for two reasons, both reported in the summary:
         1. the CSV stores the backward-MA only for RETAINED points; the 39 flagged points have
            blank predictions, so their raw predictions (needed for the grey X markers) are absent;
         2. the CSV's time column comes from the `_inf_smoothed` file, whose timestamps carry a
            12-min lag (misc/time_smoothing.py rolls the TIME column together with the spectra).
            Here the timestamps are taken from the RAW `exp_04_inf.txt` (same row order), which is
            the workaround the repo already uses in scripts/calibration/fix_cal_and_plot.py.
FIG B: exp7 per-spectrum T2/Q from results_refe_vs_inf/exp7_refe_ad.csv (the 16 HPLC-paired
       reference/nonda spectra). The `_inf` overlay is recomputed here because no per-spectrum CSV
       exists for it (results_refe_vs_inf/exp7_three_streams.csv is summary-level only).

Nothing in results/ or the production pipeline is touched. Run:
  MPLBACKEND=Agg py scripts/plotting/make_fig34.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

FIGDIR, DATA, RES = "figures_paper", "data", "results_refe_vs_inf"
GLU_C, XYL_C, FLAG_C = "#C0392B", "#6C3483", "#7f7f7f"
RMSEP = {"glucose": 1.06, "xylose": 1.00}      # as specified for the band
MM = 1 / 25.4
W90 = 90 * MM                                   # Elsevier single column

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.8, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, f"{name}.tif"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


def load_nonda(e):
    p = os.path.join(DATA, f"{e}_nonda.txt")
    with open(p) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    a = np.atleast_2d(np.loadtxt(p, skiprows=1))
    return apply_pretreatment(B.PRETREAT, a[:, 1:], wl, plot=False)[0]


def load_online(fname, tmax_h, time_from=None):
    """Spectra from `fname`; timestamps from `time_from` (raw file) when given — fixes the
    12-min lag of the *_inf_smoothed time column (rows are row-aligned with the raw file)."""
    p = os.path.join(DATA, fname)
    with open(p) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    a = np.atleast_2d(np.loadtxt(p, skiprows=1))
    if time_from:
        r = np.atleast_2d(np.loadtxt(os.path.join(DATA, time_from), skiprows=1))
        assert r.shape[0] == a.shape[0], "raw/smoothed row mismatch — cannot transfer timestamps"
        t = r[:, 0] / 60.0
    else:
        t = a[:, 0] / 60.0
    keep = t <= tmax_h
    return apply_pretreatment(B.PRETREAT, a[keep, 1:], wl, plot=False)[0], t[keep]


def ftest_lv(X, y):
    rk, _, km = B.cv_curve(X, y, 15)
    return int(min(V.select_k_ftest(rk, n_cal=len(y)), km))


def box_qlim(Qcal, alpha=0.95):
    m = float(np.mean(Qcal)); v = float(np.var(Qcal, ddof=1)) or 1e-10
    return (v / (2 * m)) * chi2.ppf(alpha, 2 * m * m / v)


def context(train):
    X = np.vstack([load_nonda(e) for e in train])
    y = np.concatenate([B.load_refe(e)[2] for e in train])
    A = ftest_lv(X, y)
    adc = B.applicability_domain(X, y, X, A)
    return dict(X=X, y=y, A=A, T2_lim=adc["T2_lim"], Q_lim_Box=box_qlim(adc["Q"]),
                Q_lim_JM=adc["Q_lim"], T2_cal_med=float(np.median(adc["T2"])),
                Q_cal_med=float(np.median(adc["Q"])), n_cal=adc["n"])


def bma(vals, mask, w=4):
    out = np.full(len(vals), np.nan); idx = np.where(mask)[0]
    for j, i in enumerate(idx):
        out[i] = np.mean(vals[idx[max(0, j - w + 1):j + 1]])
    return out


# ============================================================ FIG A
def fig_a():
    ctx = context(["exp5", "exp6", "exp7"])                  # M_menos1
    X, t = load_online("exp_04_inf_smoothed.txt", 12.0, time_from="exp_04_inf.txt")
    ad = B.applicability_domain(ctx["X"], ctx["y"], X, ctx["A"])
    eT2, eQ = ad["T2"] > ctx["T2_lim"], ad["Q"] > ctx["Q_lim_Box"]
    ret = ~(eT2 | eQ)

    y_glu = np.concatenate([B.load_refe(e)[2] for e in ["exp5", "exp6", "exp7"]])
    y_xyl = np.concatenate([B.load_refe(e)[3] for e in ["exp5", "exp6", "exp7"]])
    lv_g, lv_x = ftest_lv(ctx["X"], y_glu), ftest_lv(ctx["X"], y_xyl)
    pred = {"glucose": B.pls_predict(ctx["X"], y_glu, X, lv_g),
            "xylose":  B.pls_predict(ctx["X"], y_xyl, X, lv_x)}
    ma = {a: bma(pred[a], ret) for a in pred}

    r4 = np.atleast_2d(np.loadtxt(os.path.join(DATA, "exp4_refe.txt")))
    hplc = {"t": r4[:, 0] / 60.0, "glucose": r4[:, 2], "xylose": r4[:, 3]}

    band = (t >= 1.0) & (t < 12.0)
    frac = 100 * (band & ret).sum() / max(int(band.sum()), 1)
    print(f"[FIG A] M_menos1 LV glu={lv_g}, xyl={lv_x}; T2_lim={ctx['T2_lim']:.2f}, "
          f"Q_lim(Box)={ctx['Q_lim_Box']:.0f}")
    print(f"[FIG A] online <=12h n={len(t)}; retained={int(ret.sum())}; flagged={int((~ret).sum())}")
    print(f"[FIG A] retained in 1-12 h = {int((band & ret).sum())}/{int(band.sum())} = {frac:.1f}%")
    print(f"[FIG A] flagged by time band (of {int((~ret).sum())} total):")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 12)]:
        m = (~ret) & (t >= lo) & (t < hi)
        print(f"          {lo:2d}-{hi:2d} h: {int(m.sum()):3d}")

    fig, axs = plt.subplots(2, 1, figsize=(W90, 4.6), sharex=True)
    for k, (a, col, mk, ls) in enumerate([("glucose", GLU_C, "o", "-"), ("xylose", XYL_C, "^", "--")]):
        ax = axs[k]
        ax.fill_between(t, ma[a] - RMSEP[a], ma[a] + RMSEP[a], where=np.isfinite(ma[a]),
                        color=col, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(t, ma[a], ls=ls, color=col, lw=1.6, zorder=3)
        ax.scatter(t[~ret], pred[a][~ret], marker="x", c=FLAG_C, s=14, linewidths=0.8, zorder=2)
        ax.scatter(hplc["t"], hplc[a], marker=mk, s=30, facecolor=col, edgecolor="black",
                   linewidths=0.6, zorder=5)
        ax.set_ylabel("Concentration (g L$^{-1}$)")
        ax.set_title(f"({chr(97+k)}) {a.capitalize()}", loc="left", fontsize=9, fontweight="bold", pad=3)
        handles = [Line2D([], [], color=col, ls=ls, lw=1.6, label="NIR (retained)"),
                   Line2D([], [], marker=mk, ls="none", markerfacecolor=col, markeredgecolor="black",
                          markersize=4.5, label="HPLC"),
                   Line2D([], [], marker="x", ls="none", color=FLAG_C, markersize=4.5,
                          label="Flagged (out of AD)"),
                   plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.15, label=f"$\\pm$RMSEP ({RMSEP[a]:.2f})")]
        loc = "lower right" if k == 0 else "upper right"
        ax.legend(handles=handles, loc=loc, frameon=True, framealpha=0.92, fontsize=6.5,
                  handlelength=1.3, borderpad=0.35, labelspacing=0.3)
        ax.set_xlim(-0.3, 12.3)
    axs[1].set_xlabel("Time (h)")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_monitor_exp4"); plt.close(fig)
    return frac, len(t), int(ret.sum())


# ============================================================ FIG B
def fig_b():
    ctx = context(["exp4", "exp5", "exp6"])                  # M_primario
    # reference spectra (n=16) straight from the existing CSV
    refe = {"t": [], "T2": [], "Q": []}
    with open(os.path.join(RES, "exp7_refe_ad.csv")) as f:
        for r in csv.DictReader(f):
            refe["t"].append(float(r["t_h"])); refe["T2"].append(float(r["T2"])); refe["Q"].append(float(r["Q"]))
    refe = {k: np.array(v) for k, v in refe.items()}
    # _inf overlay recomputed (no per-spectrum CSV exists)
    Xi, ti = load_online("exp_07_inf.txt", 24.0)
    adi = B.applicability_domain(ctx["X"], ctx["y"], Xi, ctx["A"])
    T2i, Qi = adi["T2"], adi["Q"]

    out_refe = (refe["T2"] > ctx["T2_lim"]) | (refe["Q"] > ctx["Q_lim_Box"])
    out_inf = (T2i > ctx["T2_lim"]) | (Qi > ctx["Q_lim_Box"])
    print(f"\n[FIG B] M_primario LV={ctx['A']}, n_cal={ctx['n_cal']}; T2_lim={ctx['T2_lim']:.2f}, "
          f"Q_lim(Box)={ctx['Q_lim_Box']:.0f}")
    print(f"[FIG B] _refe n={len(refe['t'])}, out={int(out_refe.sum())} ({100*out_refe.mean():.0f}%); "
          f"T2 [{refe['T2'].min():.1f},{refe['T2'].max():.1f}] Q [{refe['Q'].min():.0f},{refe['Q'].max():.0f}]")
    print(f"[FIG B] _inf  n={len(ti)}, out={int(out_inf.sum())} ({100*out_inf.mean():.0f}%); "
          f"T2 [{T2i.min():.1f},{T2i.max():.1f}] Q [{Qi.min():.0f},{Qi.max():.0f}]")
    print(f"[FIG B] calibration centroid: T2_med={ctx['T2_cal_med']:.2f}, Q_med={ctx['Q_cal_med']:.0f}")

    fig, ax = plt.subplots(figsize=(W90, 3.5))
    vmax = max(refe["t"].max(), ti.max())
    ax.scatter(T2i, Qi, c=ti, cmap="viridis", vmin=0, vmax=vmax, s=5, alpha=0.30,
               linewidths=0, zorder=2)
    sc = ax.scatter(refe["T2"], refe["Q"], c=refe["t"], cmap="viridis", vmin=0, vmax=vmax,
                    s=42, edgecolors="black", linewidths=0.5, zorder=4)
    ax.scatter([ctx["T2_cal_med"]], [ctx["Q_cal_med"]], marker="*", s=190, facecolor="white",
               edgecolor="black", linewidths=0.8, zorder=6)
    ax.axvline(ctx["T2_lim"], color="0.25", ls="--", lw=0.9, zorder=3)
    ax.axhline(ctx["Q_lim_Box"], color="0.25", ls="--", lw=0.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.6, 4000); ax.set_ylim(600, 8e6)      # room for the centroid (bottom-left) and legend
    ax.set_xlabel("Hotelling's $T^2$ (log scale)")
    ax.set_ylabel("$Q$-residual / SPE (log scale)")
    ax.text(0.035, 0.96, "100% out-of-domain", transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("t (h)", fontsize=8); cb.ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="0.55", markeredgecolor="black",
                      markersize=4.5, label="Reference (n=16)"),
               Line2D([], [], marker="o", ls="none", color="0.55", alpha=0.5, markersize=3,
                      label=f"Online (n={len(ti)})"),
               Line2D([], [], marker="*", ls="none", markerfacecolor="white", markeredgecolor="black",
                      markersize=9, label="Cal. centroid"),
               Line2D([], [], color="0.25", ls="--", lw=0.9, label="95% limits")]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=6.5,
              handlelength=1.3, borderpad=0.35, labelspacing=0.3)
    ax.grid(True, which="both", lw=0.25, alpha=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_influence_exp7"); plt.close(fig)
    return 100 * out_refe.mean(), 100 * out_inf.mean(), len(ti)


# ============================================================ FIG C
def fig_c():
    """Influence plot: calibration Assay 2-4 (red) vs exp4/Assay 1 test (blue diamonds).
    run_calibration style (linear axes, Box chi^2 Q-limit), English, NO per-point labels."""
    ctx = context(["exp5", "exp6", "exp7"])                  # M_menos1, cal = Assay 2-4
    adc = B.applicability_domain(ctx["X"], ctx["y"], ctx["X"], ctx["A"])
    T2c, Qc = adc["T2"], adc["Q"]
    Xt = load_nonda("exp4"); adt = B.applicability_domain(ctx["X"], ctx["y"], Xt, ctx["A"])
    T2t, Qt = adt["T2"], adt["Q"]
    t2lim, qlim = ctx["T2_lim"], ctx["Q_lim_Box"]
    cal_out = (T2c > t2lim) | (Qc > qlim)
    test_out = (T2t > t2lim) | (Qt > qlim)
    print(f"\n[FIG C] M_menos1 LV={ctx['A']}, n_cal={len(T2c)}; T2_lim={t2lim:.2f}, Q_lim(Box)={qlim:.0f}")
    print(f"[FIG C] calibration outliers={int(cal_out.sum())}/{len(T2c)}; "
          f"exp4 test out={int(test_out.sum())}/{len(T2t)} (in={int((~test_out).sum())})")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(T2c[~cal_out], Qc[~cal_out], c=GLU_C, s=42, alpha=0.7, edgecolors="black",
               linewidths=0.4, zorder=3, label=f"Calibration (Assay 2–4, n={len(T2c)})")
    if cal_out.any():
        ax.scatter(T2c[cal_out], Qc[cal_out], c=GLU_C, marker="x", s=48, linewidths=1.4,
                   zorder=4, label=f"Calibration outlier ({int(cal_out.sum())})")
    ax.scatter(T2t[~test_out], Qt[~test_out], c="#2471A3", marker="D", s=48, edgecolors="black",
               linewidths=0.5, zorder=5, label=f"Test Assay 1 (exp4) — in ({int((~test_out).sum())})")
    if test_out.any():
        ax.scatter(T2t[test_out], Qt[test_out], facecolors="none", edgecolors="#2471A3", marker="D",
                   s=70, linewidths=1.5, zorder=5, label=f"Test Assay 1 (exp4) — out ({int(test_out.sum())})")
    ax.axvline(t2lim, color="0.25", ls="--", lw=1.0, label=f"$T^2$ limit (95%)")
    ax.axhline(qlim, color="0.25", ls=":", lw=1.0, label="$Q$ limit (95%)")
    ax.set_xlabel("Hotelling's $T^2$"); ax.set_ylabel("$Q$-residual / SPE")
    ax.set_title(f"Glucose — calibration Assay 2–4, test Assay 1 (LVs={ctx['A']})",
                 loc="left", fontsize=9)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7, handlelength=1.4,
              borderpad=0.35, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_influence_exp4_glucose"); plt.close(fig)


# ============================================================ FIG D  (mirror of FIG C for exp7)
def fig_influence_glucose(train, test_exp, test_assay, cal_assay, out_name, legend_loc="upper right"):
    """Generic run_calibration-style influence plot: cal (red) vs held-out test (blue diamonds),
    linear axes, Box chi^2 Q-limit, English, no annotations, no grid."""
    ctx = context(train)
    adc = B.applicability_domain(ctx["X"], ctx["y"], ctx["X"], ctx["A"])
    T2c, Qc = adc["T2"], adc["Q"]
    Xt = load_nonda(test_exp); adt = B.applicability_domain(ctx["X"], ctx["y"], Xt, ctx["A"])
    T2t, Qt = adt["T2"], adt["Q"]
    t2lim, qlim = ctx["T2_lim"], ctx["Q_lim_Box"]
    cal_out = (T2c > t2lim) | (Qc > qlim)
    test_out = (T2t > t2lim) | (Qt > qlim)
    print(f"\n[{out_name}] cal={cal_assay} LV={ctx['A']}, n_cal={len(T2c)}; T2_lim={t2lim:.2f}, "
          f"Q_lim(Box)={qlim:.0f}; cal outliers={int(cal_out.sum())}/{len(T2c)}; "
          f"{test_assay} out={int(test_out.sum())}/{len(T2t)} (in={int((~test_out).sum())})")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(T2c[~cal_out], Qc[~cal_out], c=GLU_C, s=42, alpha=0.7, edgecolors="black",
               linewidths=0.4, zorder=3, label=f"Calibration ({cal_assay}, n={len(T2c)})")
    if cal_out.any():
        ax.scatter(T2c[cal_out], Qc[cal_out], c=GLU_C, marker="x", s=48, linewidths=1.4,
                   zorder=4, label=f"Calibration outlier ({int(cal_out.sum())})")
    ax.scatter(T2t[~test_out], Qt[~test_out], c="#2471A3", marker="D", s=48, edgecolors="black",
               linewidths=0.5, zorder=5, label=f"Test {test_assay} ({test_exp}) — in ({int((~test_out).sum())})")
    if test_out.any():
        ax.scatter(T2t[test_out], Qt[test_out], facecolors="none", edgecolors="#2471A3", marker="D",
                   s=70, linewidths=1.5, zorder=5, label=f"Test {test_assay} ({test_exp}) — out ({int(test_out.sum())})")
    ax.axvline(t2lim, color="0.25", ls="--", lw=1.0, label="$T^2$ limit (95%)")
    ax.axhline(qlim, color="0.25", ls=":", lw=1.0, label="$Q$ limit (95%)")
    ax.set_xlabel("Hotelling's $T^2$"); ax.set_ylabel("$Q$-residual / SPE")
    ax.set_title(f"Glucose — calibration {cal_assay}, test {test_assay} (LVs={ctx['A']})",
                 loc="left", fontsize=9)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.92, fontsize=7, handlelength=1.4,
              borderpad=0.35, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, out_name); plt.close(fig)


if __name__ == "__main__":
    frac, n_on, n_ret = fig_a()
    p_refe, p_inf, n_inf = fig_b()
    fig_c()
    fig_influence_glucose(["exp4", "exp5", "exp6"], "exp7", "Assay 4", "Assay 1–3",
                          "fig_influence_exp7_glucose", legend_loc="lower right")
    print(f"\nSaved -> {FIGDIR}/fig_monitor_exp4.{{pdf,tif}}, {FIGDIR}/fig_influence_exp7.{{pdf,tif}}, "
          f"{FIGDIR}/fig_influence_exp4_glucose.{{pdf,tif}}, {FIGDIR}/fig_influence_exp7_glucose.{{pdf,tif}}")
