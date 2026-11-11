"""
Generate the publication figures for the LOExO PLS/VIP/PSO study (exp4..exp7),
for GLUCOSE and XYLOSE, into results_pso_tuning_glucose/.

Figures (English labels):
  1. spectra.png                      - raw and pretreated spectra (Cut + SG 1st deriv.)
  2. selected_wavenumbers_<a>.png     - mean spectrum + VIP/PSO selected wavenumbers
  3. parity_<a>.png                   - measured vs predicted (Cal / CV / Test), PLS|VIP|PSO
  4. rmsecv_vs_lv_<a>.png             - RMSECV vs # latent variables, PLS|VIP|PSO

For each method/analyte the BEST held-out batch by RMSEP (from final_report.txt) is used.
PSO always targets glucose (config from the tuning study: np=30, default accel).

Usage:
  py scripts/calibration/plot_loexo_results.py          # full (re-runs PSO ~8 min)
  py scripts/calibration/plot_loexo_results.py smoke    # quick (tiny PSO)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.var_selection import run_pso, select_k_ftest
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V

SMOKE = (len(sys.argv) > 1 and sys.argv[1].lower() == 'smoke')

RESULTS_DIR = "results_pso_tuning_glucose"
EXPERIMENTS = ['exp4', 'exp5', 'exp6', 'exp7']
ANALYTES = {'gl': ('Glucose', 1), 'xy': ('Xylose', 2)}
GL = 1
COLORS = {'PLS': 'tab:blue', 'VIP': 'tab:green', 'PSO': 'tab:red'}
METHODS = ['PLS', 'VIP', 'PSO']
UNITS = 'g/L'

# Best PSO config from the tuning study (defaults won)
PSO_NP, PSO_W, PSO_C1, PSO_C2 = 30, 0.9, 1.49, 1.49
PSO_MAX_ITER, PSO_REPEATS = 200, 3

# Best held-out batch by RMSEP (from final_report.txt)
BEST_BATCH = {
    'gl': {'PLS': 'exp4', 'VIP': 'exp7', 'PSO': 'exp7'},
    'xy': {'PLS': 'exp4', 'VIP': 'exp4', 'PSO': 'exp4'},
}

if SMOKE:
    PSO_MAX_ITER, PSO_REPEATS = 6, 1


def _r2(y, yp):
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return 1 - float(np.sum((y - yp) ** 2)) / sst


def _rmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def load_raw(e):
    with open(os.path.join("data", f"{e}_nonda.txt")) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    ai = np.atleast_2d(np.loadtxt(os.path.join("data", f"{e}_nonda.txt"), skiprows=1))[:, 1:]
    return ai, wl


def split_for(data, held):
    train = [e for e in EXPERIMENTS if e != held]
    A_tr = np.vstack([data[e][1] for e in train])
    Y_tr = np.vstack([data[e][0] for e in train])
    return train, A_tr, Y_tr, data[held][1], data[held][0]


def pso_glucose_mask(data, held, cache):
    if held in cache:
        return cache[held]
    _, A_tr, Y_tr, _, _ = split_for(data, held)
    best = (np.inf, None)
    for r in range(PSO_REPEATS):
        m, s = run_pso(A_tr, Y_tr, V.MAX_LV, V.K_FOLDS, V.CV_TYPE, target_idx=GL,
                       n_particles=PSO_NP, max_iter=PSO_MAX_ITER, w=PSO_W, c1=PSO_C1, c2=PSO_C2,
                       seed=900 + r, verbose=False)
        if s < best[0]:
            best = (s, m)
    cache[held] = best[1]
    return best[1]


def fit_for_plot(A_tr, y_tr, A_te, y_te):
    ycv, kmax = V._cv_pred(A_tr, y_tr, V.MAX_LV)
    rmsecv_k = np.sqrt(np.mean((y_tr[:, None] - ycv) ** 2, axis=0))
    k = int(min(select_k_ftest(rmsecv_k, n_cal=len(y_tr)), kmax))
    y_cv = ycv[:, k - 1]
    Xn, Xmu, Xsig = V.zscore_matlab_style(A_tr)
    yn, ymu, ysig = V.zscore_matlab_style(y_tr.reshape(-1, 1))
    _, _, P, _, Q, W, _, _ = V.PLS().nipals(Xn, yn, k)
    beta = V._beta(P, Q, W, k)
    y_cal = ((Xn @ beta) * ysig + ymu).flatten()
    Xsig_safe = Xsig.copy(); Xsig_safe[Xsig_safe == 0] = 1.0
    y_test = ((((A_te - Xmu) / Xsig_safe) @ beta) * ysig + ymu).flatten()
    return dict(rmsecv_k=rmsecv_k[:kmax], k=k, y_cal=y_cal, y_cv=y_cv, y_test=y_test)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"MODE: {'SMOKE' if SMOKE else 'FULL'}")
    data = {e: V.load_full(e) for e in EXPERIMENTS}          # (conc, pretreated spectra, wl)
    wl = data['exp4'][2]
    A_all = np.vstack([data[e][1] for e in EXPERIMENTS])
    mean_spec = A_all.mean(0)

    # ---- compute models/masks per analyte/method ----
    pso_cache = {}
    res = {a: {} for a in ANALYTES}
    masks = {a: {} for a in ANALYTES}
    for a, (aname, jt) in ANALYTES.items():
        for m in METHODS:
            held = BEST_BATCH[a][m]
            train, A_tr, Y_tr, A_te, Y_te = split_for(data, held)
            if m == 'PLS':
                mask = np.ones(A_tr.shape[1], dtype=bool)
            elif m == 'VIP':
                mask = V.vip_mask(A_tr, Y_tr, jt)
            else:
                mask = pso_glucose_mask(data, held, pso_cache)
            r = fit_for_plot(A_tr[:, mask], Y_tr[:, jt], A_te[:, mask], Y_te[:, jt])
            r.update(held=held, y_tr=Y_tr[:, jt], y_te=Y_te[:, jt], n_vars=int(mask.sum()))
            res[a][m] = r
            masks[a][m] = mask
            print(f"  {aname} / {m}: held-out {held}, {int(mask.sum())} vars, LVs={r['k']}, "
                  f"RMSEP={_rmse(Y_te[:, jt], r['y_test']):.3f}")

    # ================= 1. SPECTRA =================
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for e in EXPERIMENTS:
        raw, wl_raw = load_raw(e)
        ax[0].plot(wl_raw, raw.T, lw=0.5)
    ax[0].set_title("a) Raw spectra"); ax[0].set_xlabel("Wavenumber (cm$^{-1}$)"); ax[0].set_ylabel("Absorbance")
    ax[1].plot(wl, A_all.T, lw=0.5)
    ax[1].set_title("b) Pretreated spectra (Cut + Savitzky-Golay, 1st derivative)")
    ax[1].set_xlabel("Wavenumber (cm$^{-1}$)"); ax[1].set_ylabel("Absorbance (1st deriv.)")
    fig.suptitle("Spectra (experiments 4-7)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectra.png"), dpi=300); plt.close(fig)

    # ================= per-analyte figures =================
    for a, (aname, jt) in ANALYTES.items():
        # ---- 2. SELECTED WAVENUMBERS ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wl, mean_spec, 'k-', alpha=0.4, lw=1, label='Mean spectrum')
        vmask, pmask = masks[a]['VIP'], masks[a]['PSO']
        ax.scatter(wl[vmask], mean_spec[vmask], c=COLORS['VIP'], s=10, alpha=0.7,
                   label=f"VIP-selected (n={int(vmask.sum())})")
        ax.scatter(wl[pmask], mean_spec[pmask], c=COLORS['PSO'], s=10, marker='x', alpha=0.6,
                   label=f"PSO-selected (n={int(pmask.sum())})")
        ax.set_title(f"Selected wavenumbers - {aname} (PLS uses all {A_all.shape[1]})")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Absorbance (1st deriv.)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"selected_wavenumbers_{a}.png"), dpi=300); plt.close(fig)

        # ---- 3. PARITY ----
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, m in zip(axes, METHODS):
            r = res[a][m]
            ytr, yte = r['y_tr'], r['y_te']
            ax.scatter(ytr, r['y_cal'], facecolors='none', edgecolors=COLORS[m], alpha=0.6, label='Calibration')
            ax.scatter(ytr, r['y_cv'], c=COLORS[m], marker='x', alpha=0.7, label='Cross-validation')
            ax.scatter(yte, r['y_test'], c='black', marker='D', s=45, alpha=0.85, label='Test')
            lo = min(ytr.min(), yte.min(), r['y_cal'].min(), r['y_test'].min())
            hi = max(ytr.max(), yte.max(), r['y_cal'].max(), r['y_test'].max())
            b = (hi - lo) * 0.05 or 1
            ax.plot([lo - b, hi + b], [lo - b, hi + b], 'k--', alpha=0.5, label='1:1 line')
            txt = (f"Cal: $R^2$={_r2(ytr, r['y_cal']):.3f}, RMSE={_rmse(ytr, r['y_cal']):.3f}\n"
                   f"CV: $R^2$={_r2(ytr, r['y_cv']):.3f}, RMSECV={_rmse(ytr, r['y_cv']):.3f}\n"
                   f"Test: $R^2$={_r2(yte, r['y_test']):.3f}, RMSEP={_rmse(yte, r['y_test']):.3f}")
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            ax.set_title(f"{m} (held-out {r['held']}, {r['n_vars']} vars, LVs={r['k']})")
            ax.set_xlabel(f"Measured {aname} ({UNITS})"); ax.set_ylabel(f"Predicted {aname} ({UNITS})")
            ax.legend(loc='lower right', fontsize=8)
        fig.suptitle(f"Parity plots - {aname}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"parity_{a}.png"), dpi=300); plt.close(fig)

        # ---- 4. RMSECV vs LVs ----
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in METHODS:
            r = res[a][m]
            ks = np.arange(1, len(r['rmsecv_k']) + 1)
            ax.plot(ks, r['rmsecv_k'], '-o', color=COLORS[m], ms=4,
                    label=f"{m} ({r['n_vars']} vars)")
            ax.plot(r['k'], r['rmsecv_k'][r['k'] - 1], '*', color=COLORS[m], ms=16,
                    markeredgecolor='black')
        ax.set_title(f"RMSECV vs latent variables - {aname} (★ = F-test selected LVs)")
        ax.set_xlabel("Latent variables (LVs)"); ax.set_ylabel(f"RMSECV ({UNITS})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"rmsecv_vs_lv_{a}.png"), dpi=300); plt.close(fig)

    print(f"\nSaved figures in: {RESULTS_DIR}/")
    print("  spectra.png, selected_wavenumbers_{gl,xy}.png, parity_{gl,xy}.png, rmsecv_vs_lv_{gl,xy}.png")


if __name__ == "__main__":
    main()
