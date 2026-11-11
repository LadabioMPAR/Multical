"""
plot_influence_annotated.py — run_calibration-STYLE influence plots, annotated (NON-DESTRUCTIVE)
==============================================================================================
Glucose influence plot in the SAME style as run_calibration.py / evaluate_and_plot_outliers
(production PLS.compute_outlier_metrics: T2 via F-dist, Q-limit via Box chi^2; LINEAR axes),
calibrating on Assay 2-4 (exp5/exp6/exp7 nonda) — with:
  * every point identified by exp:t=<min> annotations (like the run_calibration outliers),
  * a clearer legend + a stats box (LV, limits, counts).

Two test/monitoring streams, run in parallel (only the test data differs):
  A) exp4 nonda        (HPLC-paired reference spectra of the held-out Assay 1)
  B) exp_04_inf_smoothed.txt  (the SMOOTHED online inference stream of exp4)

Reuses production classes so numbers match run_calibration. Cut 4400-7500 + SG(7,2,1).
Writes NEW files only:
  figures/fig_influence_glucose_exp4nonda_annot.{png,pdf,svg}
  figures/fig_influence_glucose_exp4smoothed_annot.{png,pdf,svg}
  results_influence/points_calibration.csv, points_exp4nonda.csv, points_exp4smoothed.csv

Run:  MPLBACKEND=Agg py scripts/plotting/plot_influence_annotated.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from src.multical.var_selection import select_k_ftest

FIGDIR, RESDIR, DATA = "figures", "results_influence", "data"
PRE = [["Cut", 4400, 7500, 1], ["SG", 7, 2, 1, 1]]
CAL = ["exp5", "exp6", "exp7"]     # Assay 2-4
GLU = 2                            # (t, cb, gl, xy) -> glucose index
MAXK, CONF = 15, 0.95
GREEN, BLUE = "#27AE60", "#2471A3"


def load_nonda_labeled(e):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    arr = np.atleast_2d(np.loadtxt(path, skiprows=1))
    t = arr[:, 0]
    spec = apply_pretreatment(PRE, arr[:, 1:], wl, plot=False)[0]
    return spec, t

def load_inf_labeled(fname):
    path = os.path.join(DATA, fname)
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    arr = np.atleast_2d(np.loadtxt(path, skiprows=1))
    t = arr[:, 0]
    spec = apply_pretreatment(PRE, arr[:, 1:], wl, plot=False)[0]
    return spec, t

def load_glucose(e):
    r = np.atleast_2d(np.loadtxt(os.path.join(DATA, f"{e}_refe.txt")))
    return r[:, GLU]


def wcsv(name, rows):
    with open(os.path.join(RESDIR, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def make_figure(cal, test, k, t2lim, qlim, title, out, test_lbl, test_s=70, annot_mode="out", logscale=False):
    """cal/test are dicts with T2, Q, exp[], t[], out(mask). Renders run_calibration-style.
    annot_mode: 'out' -> label every out-of-domain test point; 'topQ' -> label 5 highest-Q."""
    plt.rcParams.update({"font.size": 12, "axes.labelweight": "bold", "axes.titlesize": 14,
                         "font.weight": "bold"})
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cn, co = ~cal["out"], cal["out"]
    ax.scatter(cal["T2"][cn], cal["Q"][cn], c=GREEN, alpha=0.6, s=70, edgecolors="black",
               linewidths=0.4, label=f"Calibração Assay 2-4 — dentro ({int(cn.sum())})", zorder=3)
    ax.scatter(cal["T2"][co], cal["Q"][co], c="black", marker="x", s=80, linewidths=1.8,
               label=f"Calibração — outlier ({int(co.sum())})", zorder=4)
    tn, to = ~test["out"], test["out"]
    ax.scatter(test["T2"][tn], test["Q"][tn], c=BLUE, marker="D", s=test_s, edgecolors="black",
               linewidths=0.4, alpha=0.75, label=f"{test_lbl} — dentro ({int(tn.sum())})", zorder=5)
    ax.scatter(test["T2"][to], test["Q"][to], facecolors="none", edgecolors="#C0392B", marker="D",
               s=test_s + 25, linewidths=1.2, alpha=0.9, label=f"{test_lbl} — fora ({int(to.sum())})", zorder=6)
    ax.axvline(t2lim, color="k", linestyle="--", linewidth=1.3, label=f"$T^2$ limite ({CONF*100:.0f}%)")
    ax.axhline(qlim, color="k", linestyle=":", linewidth=1.3, label=f"$Q$ limite ({CONF*100:.0f}%)")

    def annot(T2, Q, exp, t, mask):
        for i in np.where(mask)[0]:
            ax.annotate(f"{exp[i]}:t={int(t[i])}", (T2[i], Q[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7.5, fontweight="normal", zorder=8)
    annot(cal["T2"], cal["Q"], cal["exp"], cal["t"], co)                       # cal outliers only
    if annot_mode == "out":
        annot(test["T2"], test["Q"], test["exp"], test["t"], test["out"])      # all out-of-domain test pts
    else:                                                                       # topQ: 5 highest-Q test pts
        m = np.zeros(len(test["Q"]), bool); m[np.argsort(test["Q"])[-5:]] = True
        annot(test["T2"], test["Q"], test["exp"], test["t"], m)
    if logscale:
        ax.set_xscale("log"); ax.set_yscale("log")

    txt = (f"Glucose  LV={k}\n$T^2_{{lim}}$={t2lim:.1f}\n$Q_{{lim}}$={qlim:.0f}\n"
           f"cal n={len(cal['T2'])}\n{test_lbl} n={len(test['T2'])}\nfora={int(to.sum())}/{len(test['T2'])}")
    ax.text(0.985, 0.62, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            fontweight="normal", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"))
    ax.set_xlabel("Hotelling's $T^2$"); ax.set_ylabel("$Q$-residuals (SPE)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.93, ncol=1)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{out}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(RESDIR, exist_ok=True)

    # ---- calibration Assay 2-4 (with per-sample exp:t labels) ----
    specs, cal_exp, cal_t = [], [], []
    for e in CAL:
        s, t = load_nonda_labeled(e)
        specs.append(s); cal_exp += [e] * len(t); cal_t += list(t)
    Xc = np.vstack(specs); yc = np.concatenate([load_glucose(e) for e in CAL])
    cal_exp = np.array(cal_exp); cal_t = np.array(cal_t)

    # ---- LV by F-test (venetian-5), production-style ----
    pls = PLS()
    n = Xc.shape[0]; ypred = np.zeros((n, MAXK))
    for i in range(5):
        val = np.arange(i, n, 5); tr = np.setdiff1d(np.arange(n), val)
        comb = np.vstack([Xc[tr], Xc[val]]); combn, _, _ = zscore_matlab_style(comb)
        ntr = len(tr); Xtr, Xval = combn[:ntr], combn[ntr:]
        ytr_n, ym, ys = zscore_matlab_style(yc[tr].reshape(-1, 1))
        _, _, p, _, q, w, _, _ = pls.nipals(Xtr, ytr_n, MAXK)
        for kk in range(1, MAXK + 1):
            beta = w[:, :kk] @ np.linalg.pinv(p[:, :kk].T @ w[:, :kk]) @ q[:, :kk].T
            ypred[val, kk - 1] = ((Xval @ beta) * ys + ym).ravel()
    rk = np.sqrt(np.mean((yc[:, None] - ypred) ** 2, axis=0))
    k = int(select_k_ftest(rk, n_cal=n))

    # ---- fit + production outlier metrics (Box chi^2) ----
    Xn, Xmed, Xsig = zscore_matlab_style(Xc)
    Yn, _, _ = zscore_matlab_style(yc.reshape(-1, 1))
    Bm, T, P, U, Qld, W, r2X, r2Y = pls.nipals(Xn, Yn, k)
    T2c, t2lim, Qc, qlim = pls.compute_outlier_metrics(Xn, T, P, conf_level=CONF)
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    R = W @ np.linalg.pinv(P.T @ W)
    score_var = np.var(T, axis=0, ddof=1); score_var[score_var == 0] = np.finfo(float).eps
    cal_out = (T2c > t2lim) | (Qc > qlim)
    print(f"glucose LV={k}; T2_lim={t2lim:.2f}, Q_lim={qlim:.2f}; cal outliers {int(cal_out.sum())}/{n}")
    cal = dict(T2=T2c, Q=Qc, exp=cal_exp, t=cal_t, out=cal_out)
    wcsv("points_calibration.csv", [{"set": "calibration", "exp": cal_exp[i], "t_min": int(cal_t[i]),
          "T2": round(float(T2c[i]), 3), "Q": round(float(Qc[i]), 2),
          "exceeds_T2": int(T2c[i] > t2lim), "exceeds_Q": int(Qc[i] > qlim),
          "outlier": int(cal_out[i])} for i in range(n)])

    def project(Xt_pre, t_arr, exp_tag):
        Xtn = (Xt_pre - Xmed) / Xs
        Tt = Xtn @ R
        T2t = np.sum((Tt ** 2) / score_var, axis=1)
        Qt = np.sum((Xtn - Tt @ P.T) ** 2, axis=1)
        out = (T2t > t2lim) | (Qt > qlim)
        return dict(T2=T2t, Q=Qt, exp=np.array([exp_tag] * len(t_arr)), t=t_arr, out=out)

    # ---- A) exp4 nonda test ----
    Xa, ta = load_nonda_labeled("exp4")
    A = project(Xa, ta, "exp4")
    print(f"[A nonda]  exp4 test outside: {int(A['out'].sum())}/{len(A['T2'])}; "
          f"T2 [{A['T2'].min():.1f},{A['T2'].max():.1f}]  Q [{A['Q'].min():.0f},{A['Q'].max():.0f}]")
    make_figure(cal, A, k, t2lim, qlim, "Influence Plot: Glucose — teste exp4 (nonda/HPLC)",
                "fig_influence_glucose_exp4nonda_annot", test_lbl="exp4 nonda", test_s=80, annot_mode="out")
    wcsv("points_exp4nonda.csv", [{"set": "exp4_nonda", "exp": "exp4", "t_min": int(A["t"][i]),
          "T2": round(float(A["T2"][i]), 3), "Q": round(float(A["Q"][i]), 2),
          "exceeds_T2": int(A["T2"][i] > t2lim), "exceeds_Q": int(A["Q"][i] > qlim),
          "outside_domain": int(A["out"][i])} for i in range(len(A["T2"]))])

    # ---- B) exp4 smoothed online stream ----
    Xb, tb = load_inf_labeled("exp_04_inf_smoothed.txt")
    Bd = project(Xb, tb, "exp4")
    print(f"[B smooth] exp4 online(smoothed) outside: {int(Bd['out'].sum())}/{len(Bd['T2'])}; "
          f"T2 [{Bd['T2'].min():.1f},{Bd['T2'].max():.1f}]  Q [{Bd['Q'].min():.0f},{Bd['Q'].max():.0f}]  "
          f"Q median={np.median(Bd['Q']):.0f}")
    make_figure(cal, Bd, k, t2lim, qlim, "Influence Plot: Glucose — teste exp4 (_inf_smoothed online)",
                "fig_influence_glucose_exp4smoothed_annot", test_lbl="exp4 smoothed", test_s=24, annot_mode="topQ")
    make_figure(cal, Bd, k, t2lim, qlim, "Influence Plot: Glucose — exp4 _inf_smoothed (eixos log)",
                "fig_influence_glucose_exp4smoothed_log", test_lbl="exp4 smoothed", test_s=24,
                annot_mode="topQ", logscale=True)
    wcsv("points_exp4smoothed.csv", [{"set": "exp4_inf_smoothed", "exp": "exp4", "t_min": int(Bd["t"][i]),
          "t_h": round(float(Bd["t"][i]) / 60, 3), "T2": round(float(Bd["T2"][i]), 3),
          "Q": round(float(Bd["Q"][i]), 2), "exceeds_T2": int(Bd["T2"][i] > t2lim),
          "exceeds_Q": int(Bd["Q"][i] > qlim), "outside_domain": int(Bd["out"][i])}
          for i in range(len(Bd["T2"]))])

    print("\nSaved -> figures/fig_influence_glucose_exp4nonda_annot.{png,pdf,svg}, "
          "fig_influence_glucose_exp4smoothed_annot.{png,pdf,svg}")
    print("        results_influence/points_{calibration,exp4nonda,exp4smoothed}.csv")


if __name__ == "__main__":
    main()
