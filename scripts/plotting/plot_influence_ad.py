"""
plot_influence_ad.py — Q vs T2 influence plots for each calibration (NON-DESTRUCTIVE)
====================================================================================
Generates one Hotelling-T2 vs Q-residual influence plot per model, using the SAME
production pipeline (Cut 4400-7500 + Savitzky-Golay 1st derivative; autoscale; glucose
F-test LV) reused from build_paper_data.applicability_domain. Each plot shows the
calibration cloud (self-projected), the online monitoring stream, and the 95% limits.

  M_primario : cal = Assay 1-3 (exp4/5/6); monitor = Assay 4 (exp7); limits from Assay 1-3
  M_menos1   : cal = Assay 2-4 (exp5/6/7); monitor = Assay 1 (exp4); limits from Assay 2-4

Writes NEW files only (does not overwrite fig_task_influence_exp7 or anything else):
  figures/fig_influence_M_primario_exp7.{png,pdf,svg}
  figures/fig_influence_M_menos1_exp4.{png,pdf,svg}
  figures/fig_influence_both.png   (2-panel side-by-side)

Run:  MPLBACKEND=Agg py scripts/plotting/plot_influence_ad.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

FIGDIR = "figures"
DATA = "data"
PRE = [["Cut", 4400, 7500, 0], ["SG", 7, 2, 1, 0]]     # Cut + SG only (production baseline)
INF = {"exp4": os.path.join(DATA, "exp_04_inf.txt"), "exp7": os.path.join(DATA, "exp_07_inf.txt")}
XMAX_H = 24.0
MODELS = {
    "M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7", "cal_lbl": "Assay 1-3", "test_lbl": "Assay 4 (exp7)"},
    "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4", "cal_lbl": "Assay 2-4", "test_lbl": "Assay 1 (exp4)"},
}
C_CAL, C_IN, C_OUT = "#2C3E50", "#27AE60", "#C0392B"


def load_spec(e):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
    return apply_pretreatment(PRE, raw, wl, plot=False)[0]

def load_online(e):
    with open(INF[e]) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(INF[e], skiprows=1))
    t = raw[:, 0] / 60.0
    keep = t <= XMAX_H
    return t[keep], apply_pretreatment(PRE, raw[keep, 1:], wl, plot=False)[0]

def ftest_glu_lv(A_tr, y_tr):
    rk, _, km = B.cv_curve(A_tr, y_tr, 15)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))


def compute(mname):
    m = MODELS[mname]
    A_tr = np.vstack([load_spec(e) for e in m["train"]])
    y_glu = np.concatenate([B.load_refe(e)[2] for e in m["train"]])
    A = ftest_glu_lv(A_tr, y_glu)
    t_on, X_on = load_online(m["test"])
    ad_on = B.applicability_domain(A_tr, y_glu, X_on, A)
    ad_cal = B.applicability_domain(A_tr, y_glu, A_tr, A)
    t2L, qL = ad_on["T2_lim"], ad_on["Q_lim"]
    ind = (ad_on["T2"] <= t2L) & (ad_on["Q"] <= qL)
    return dict(mname=mname, A=A, n_cal=ad_cal["n"], t2L=t2L, qL=qL,
                T2c=ad_cal["T2"], Qc=ad_cal["Q"], T2o=ad_on["T2"], Qo=ad_on["Q"],
                ind=ind, cal_lbl=m["cal_lbl"], test_lbl=m["test_lbl"], n_on=len(t_on))


def draw(ax, d):
    T2c, Qc, T2o, Qo, ind = d["T2c"], d["Qc"], d["T2o"], d["Qo"], d["ind"]
    t2L, qL = d["t2L"], d["qL"]
    # in-domain quadrant shading (bottom-left of the two limit lines)
    xlo = min(T2c.min(), T2o.min()) * 0.5
    ylo = min(Qc.min(), Qo.min()) * 0.5
    ax.axhspan(ylo, qL, xmin=0, xmax=1, color="0.9", zorder=0)  # under Q lim
    ax.add_patch(plt.Rectangle((xlo, ylo), t2L - xlo, qL - ylo, color="#EAF5EA", zorder=0))
    # points
    ax.scatter(T2c, Qc, s=34, color=C_CAL, edgecolors="white", linewidths=0.4, alpha=0.9,
               zorder=4, label=f"Calibração ({d['cal_lbl']}, n={d['n_cal']})")
    ax.scatter(T2o[ind], Qo[ind], s=14, color=C_IN, alpha=0.75, zorder=3,
               label=f"Online dentro ({int(ind.sum())})")
    ax.scatter(T2o[~ind], Qo[~ind], s=14, color=C_OUT, alpha=0.55, zorder=2,
               label=f"Online fora ({int((~ind).sum())})")
    # 95% limit lines
    ax.axvline(t2L, color="0.25", linestyle="--", linewidth=1.0, zorder=5)
    ax.axhline(qL, color="0.25", linestyle="--", linewidth=1.0, zorder=5)
    ax.text(t2L, ax.get_ylim()[1], f"  $T^2_{{lim}}$={t2L:.1f}", color="0.25", va="top", ha="left", fontsize=8)
    ax.text(ax.get_xlim()[1], qL, f"$Q_{{lim}}$={qL:.0f}  ", color="0.25", va="bottom", ha="right", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Hotelling $T^2$ (dentro do modelo)")
    ax.set_ylabel("$Q$ residual / SPE (fora do modelo)")
    pct = 100 * (~ind).mean()
    ax.set_title(f"{d['mname']} — monitor {d['test_lbl']}\nCut+SG, glicose LV={d['A']}; "
                 f"reprovados união = {int((~ind).sum())}/{d['n_on']} ({pct:.0f}%)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ds = {mn: compute(mn) for mn in MODELS}
    for mn, d in ds.items():
        print(f"{mn}: n_cal={d['n_cal']}, LV={d['A']}, T2_lim={d['t2L']:.2f}, Q_lim={d['qL']:.1f}, "
              f"online flagged={int((~d['ind']).sum())}/{d['n_on']}")
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        draw(ax, d)
        fig.tight_layout()
        save(fig, f"fig_influence_{mn}_{MODELS[mn]['test']}")
        plt.close(fig)
    # combined 2-panel
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.3))
    for ax, mn in zip(axs, MODELS):
        draw(ax, ds[mn])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_influence_both.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved -> figures/fig_influence_M_primario_exp7.{png,pdf,svg}, "
          "fig_influence_M_menos1_exp4.{png,pdf,svg}, fig_influence_both.png")


if __name__ == "__main__":
    main()
