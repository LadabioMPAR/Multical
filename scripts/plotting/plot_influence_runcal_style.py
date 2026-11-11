"""
plot_influence_runcal_style.py — run_calibration-STYLE influence plot (NON-DESTRUCTIVE)
=====================================================================================
Reproduces EXACTLY the influence plot that run_calibration.py / evaluate_and_plot_outliers
produce (production PLS.compute_outlier_metrics: T2 via F-dist, Q-limit via Box chi^2;
LINEAR axes; in-sample calibration cloud), for GLUCOSE, calibrating on Assay 2-4
(exp5/exp6/exp7) — and overlays the exp4 (Assay 1) test spectra projected onto that model.

Difference from my earlier figures (which looked "strange"):
  - earlier: continuous ONLINE stream (exp_04_inf.txt, 438 raw probe spectra), Q-limit by
    Jackson-Mudholkar, log-log axes -> Q up to ~1e6.
  - here:    HPLC-paired nonda spectra (the same data run_calibration uses), Q-limit by
    Box chi^2, LINEAR axes -> the value scale you are used to.

Uses the PRODUCTION classes/functions (src.multical.models.pls.PLS,
src.multical.utils.zscore_matlab_style, var_selection.select_k_ftest) so the numbers match
run_calibration. Cut 4400-7500 + Savitzky-Golay(radius 7, poly 2, 1st deriv) — identical.

Writes NEW files only:
  figures/fig_influence_runcal_glucose_exp4test.{png,pdf,svg}

Run:  MPLBACKEND=Agg py scripts/plotting/plot_influence_runcal_style.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from src.multical.var_selection import select_k_ftest

FIGDIR = "figures"
DATA = "data"
PRE = [["Cut", 4400, 7500, 1], ["SG", 7, 2, 1, 1]]     # identical to run_calibration PRETREATMENT
CAL = ["exp5", "exp6", "exp7"]                           # Assay 2-4
TEST = "exp4"                                            # Assay 1
ASSAY = {"exp4": "Assay 1", "exp5": "Assay 2", "exp6": "Assay 3", "exp7": "Assay 4"}
GLU = 2         # index in load_refe tuple (t, cb, gl, xy) -> glucose = 2
MAXK = 15
CONF = 0.95


def load_refe(e):
    r = np.atleast_2d(np.loadtxt(os.path.join(DATA, f"{e}_refe.txt")))
    return r[:, GLU]                                     # glucose column (g/L)

def load_spec(e):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
    return apply_pretreatment(PRE, raw, wl, plot=False)[0]


def venetian_rmsecv(X, y, maxk=MAXK, folds=5):
    """Venetian-blinds 5-fold PLS1 RMSECV — mirrors run_calibration.predict_pls2_cv."""
    n = X.shape[0]; ypred = np.zeros((n, maxk)); pls = PLS()
    for i in range(folds):
        val = np.arange(i, n, folds); tr = np.setdiff1d(np.arange(n), val)
        comb = np.vstack([X[tr], X[val]])
        combn, _, _ = zscore_matlab_style(comb)
        ntr = len(tr); Xtr, Xval = combn[:ntr], combn[ntr:]
        ytr_n, ym, ys = zscore_matlab_style(y[tr].reshape(-1, 1))
        _, _, p, _, q, w, _, _ = pls.nipals(Xtr, ytr_n, maxk)
        for k in range(1, maxk + 1):
            wk, pk, qk = w[:, :k], p[:, :k], q[:, :k]
            beta = wk @ np.linalg.pinv(pk.T @ wk) @ qk.T
            ypred[val, k - 1] = ((Xval @ beta) * ys + ym).ravel()
    return np.sqrt(np.mean((y[:, None] - ypred) ** 2, axis=0))


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    # ---- calibration Assay 2-4 (nonda, HPLC-paired) ----
    Xc = np.vstack([load_spec(e) for e in CAL])
    yc = np.concatenate([load_refe(e) for e in CAL])
    Xt = load_spec(TEST)                                 # exp4 test spectra (nonda)
    yt = load_refe(TEST)

    # ---- LV by F-test (venetian-5), exactly like run_calibration ----
    rk = venetian_rmsecv(Xc, yc)
    k = int(select_k_ftest(rk, n_cal=len(yc)))
    print(f"Calibration Assay 2-4: n_cal={Xc.shape[0]}, vars={Xc.shape[1]}; glucose F-test LV={k}")
    print(f"Test exp4 (Assay 1): n_test={Xt.shape[0]}")

    # ---- fit PLS + production outlier metrics (Box chi^2 Q-limit) ----
    pls = PLS()
    Xn, Xmed, Xsig = zscore_matlab_style(Xc)
    Yn, _, _ = zscore_matlab_style(yc.reshape(-1, 1))
    Bm, T, P, U, Qld, W, r2X, r2Y = pls.nipals(Xn, Yn, k)
    T2c, t2lim, Qc, qlim = pls.compute_outlier_metrics(Xn, T, P, conf_level=CONF)

    # ---- project exp4 test onto the SAME model / SAME limits ----
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    Xtn = (Xt - Xmed) / Xs
    R = W @ np.linalg.pinv(P.T @ W)
    Tt = Xtn @ R
    score_var = np.var(T, axis=0, ddof=1); score_var[score_var == 0] = np.finfo(float).eps
    T2t = np.sum((Tt ** 2) / score_var, axis=1)
    Qt = np.sum((Xtn - Tt @ P.T) ** 2, axis=1)

    cal_out = (T2c > t2lim) | (Qc > qlim)
    test_out = (T2t > t2lim) | (Qt > qlim)
    print(f"T2_lim={t2lim:.2f}, Q_lim={qlim:.2f}")
    print(f"calibration outliers: {int(cal_out.sum())}/{len(T2c)}")
    print(f"exp4 test outside domain: {int(test_out.sum())}/{len(T2t)}  "
          f"(T2>lim {int((T2t>t2lim).sum())}, Q>lim {int((Qt>qlim).sum())})")
    print(f"exp4 test  T2 range [{T2t.min():.2f},{T2t.max():.2f}]  Q range [{Qt.min():.1f},{Qt.max():.1f}]")
    print(f"calib      T2 range [{T2c.min():.2f},{T2c.max():.2f}]  Q range [{Qc.min():.1f},{Qc.max():.1f}]")

    # ---- plot (run_calibration style: linear axes) ----
    plt.rcParams.update({"font.size": 12, "axes.labelweight": "bold"})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # calibration
    ax.scatter(T2c[~cal_out], Qc[~cal_out], c="red", alpha=0.55, s=60,
               edgecolors="black", linewidths=0.5, label=f"Calibração Assay 2-4 (n={len(T2c)})")
    if cal_out.any():
        ax.scatter(T2c[cal_out], Qc[cal_out], c="black", marker="x", s=70, linewidths=1.6,
                   label=f"Outlier calibração ({int(cal_out.sum())})")
    # exp4 test
    ax.scatter(T2t[~test_out], Qt[~test_out], c="#2471A3", marker="D", s=70,
               edgecolors="black", linewidths=0.6, label=f"exp4 (Assay 1) dentro ({int((~test_out).sum())})")
    if test_out.any():
        ax.scatter(T2t[test_out], Qt[test_out], facecolors="none", edgecolors="#2471A3", marker="D",
                   s=110, linewidths=1.8, label=f"exp4 fora ({int(test_out.sum())})")
    ax.axvline(t2lim, color="k", linestyle="--", linewidth=1.2, label=f"$T^2$ limite ({CONF*100:.0f}%)")
    ax.axhline(qlim, color="k", linestyle=":", linewidth=1.2, label=f"$Q$ limite ({CONF*100:.0f}%)")
    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("$Q$-residuals (SPE)")
    ax.set_title(f"Influence Plot: Glucose (LVs={k})\ncalibração Assay 2-4, teste = exp4", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"fig_influence_runcal_glucose_exp4test.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved -> figures/fig_influence_runcal_glucose_exp4test.{png,pdf,svg}")


if __name__ == "__main__":
    main()
