"""
build_lobo_all_models.py
=========================
Full leave-one-batch-out (LOBO) comparison of PLS / VIP-PLS / PSO-PLS, for BOTH
glucose and xylose, across ALL 4 assays held out in turn. For every
(model, analyte, held-out assay) combination, generates:

  1. RMSECV vs #latent-variables curve (glucose + xylose overlaid, star = F-test LV)
  2. Parity plot (cross-validation + held-out test, 2 panels: glucose, xylose)
  3. Continuous-monitoring trajectory of the held-out batch (+/- RMSEP band,
     HPLC overlay), truncated at that assay's real duration (Table table_Exp)

Differs from the earlier LOBO run (results_pso_tuning_glucose/final_report.txt):
PSO is now run INDEPENDENTLY for xylose (target_idx=xylose), not by reusing the
glucose-selected mask. PLS uses all variables; LV always chosen per-fold by the
Osten F-test (no fixed LV here - same convention as Table tab:lobo).

Reuses verified loaders/models from build_paper_data.py (B) and
run_loexo_plsvarsel_exp4to7.py (V); figure styling from make_paper_figures.py (M).
Outputs: data_fig_lobo_all/*.csv (raw results) + figures_lobo_all/*.{pdf,svg,tif}+_data.csv

Run: python scripts/plotting/build_lobo_all_models.py [smoke]
"""
import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B
import scripts.plotting.make_paper_figures as M

SMOKE = (len(sys.argv) > 1 and sys.argv[1].lower() == "smoke")

OUTDIR_DATA = "data_fig_lobo_all"
OUTDIR_FIG = "figures_lobo_all"
EXPERIMENTS = ["exp4", "exp5", "exp6", "exp7"]
ANALYTES = B.ANALYTES                       # {"glucose": 1, "xylose": 2}
ASSAY_NAME = B.ASSAY_NAME                   # exp4->"Assay 1" ...
ASSAY_DURATION_H = {"exp4": 12.0, "exp5": 24.0, "exp6": 24.0, "exp7": 24.0}   # Table table_Exp
INF_FILES = {e: os.path.join("data", f"exp_{e[3:].zfill(2)}_inf.txt") for e in EXPERIMENTS}
MODELS = ["PLS", "VIP", "PSO"]
MAX_LV = 15
PSO_PARAMS = dict(n_particles=30, max_iter=120, w=0.9, c1=1.49, c2=1.49)   # tuned best config

if SMOKE:
    EXPERIMENTS_RUN = ["exp4"]
    PSO_PARAMS = dict(n_particles=6, max_iter=4, w=0.9, c1=1.49, c2=1.49)
else:
    EXPERIMENTS_RUN = EXPERIMENTS


def load_inf(path):
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))
    from src.multical.preprocessing.pipeline import apply_pretreatment
    t = raw[:, 0] / 60.0
    spec, _ = apply_pretreatment(B.PRETREAT, raw[:, 1:], wl, plot=False)
    return t, spec


def fit_eval(A_tr, y_tr, A_te, y_te, A_inf, mask, max_lv=MAX_LV):
    """Fit a model on the masked variables, select LV by F-test, evaluate everywhere."""
    A_sub, A_te_sub, A_inf_sub = A_tr[:, mask], A_te[:, mask], A_inf[:, mask]
    rk, ycv, km = B.cv_curve(A_sub, y_tr, max_lv)
    lv = int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))
    cvp = ycv[:, lv - 1]
    tp = B.pls_predict(A_sub, y_tr, A_te_sub, lv)
    infp = B.pls_predict(A_sub, y_tr, A_inf_sub, lv)
    rmsep = float(np.sqrt(np.mean((y_te - tp) ** 2)))
    return dict(n_vars=int(mask.sum()), lv=lv, km=km, rk=rk, cvp=cvp, tp=tp, infp=infp,
               rmsecv=float(rk[lv - 1]), rmsep=rmsep)


def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    os.makedirs(OUTDIR_FIG, exist_ok=True)
    M.set_style()
    print(f"MODE: {'SMOKE' if SMOKE else 'FULL'} | folds: {EXPERIMENTS_RUN}\n")

    refe = {e: B.load_refe(e) for e in EXPERIMENTS}
    nd = {e: B.load_nonda(e) for e in EXPERIMENTS}
    spec_pre = {e: nd[e][2] for e in nd}
    conc = {e: {"cb": refe[e][1], "glucose": refe[e][2], "xylose": refe[e][3]} for e in refe}
    Ymat = {e: np.column_stack([conc[e]["cb"], conc[e]["glucose"], conc[e]["xylose"]]) for e in refe}
    time_h = {e: refe[e][0] for e in refe}

    summary_rows = []
    t0_all = time.perf_counter()

    for held in EXPERIMENTS_RUN:
        train = [e for e in EXPERIMENTS if e != held]
        A_tr = np.vstack([spec_pre[e] for e in train])
        Y_tr = np.vstack([Ymat[e] for e in train])
        A_te, Y_te = spec_pre[held], Ymat[held]
        t_inf, A_inf = load_inf(INF_FILES[held])
        assay = ASSAY_NAME[held]
        slug = assay.lower().replace(" ", "")
        M.TEST_ASSAY = assay   # _parity_panel() reads this module global for its legend label
        print(f"### Held-out = {assay} ({held}); train on {[ASSAY_NAME[e] for e in train]}")

        # --- masks + fits, per analyte ---
        res = {a: {} for a in ANALYTES}
        for a, j in ANALYTES.items():
            y_tr, y_te = Y_tr[:, j], Y_te[:, j]
            t0 = time.perf_counter()
            mask_vip = V.vip_mask(A_tr, Y_tr, j)
            mask_pso, pso_score = V.run_pso(A_tr, Y_tr, MAX_LV, V.K_FOLDS, V.CV_TYPE, target_idx=j,
                                            seed=2000 + j, verbose=False, **PSO_PARAMS)
            dt = time.perf_counter() - t0
            masks = {"PLS": np.ones(A_tr.shape[1], dtype=bool), "VIP": mask_vip, "PSO": mask_pso}
            for M_ in MODELS:
                r = fit_eval(A_tr, y_tr, A_te, y_te, A_inf, masks[M_])
                res[a][M_] = r
                print(f"    [{a:8s}/{M_}] vars={r['n_vars']:5d} LV={r['lv']:2d} "
                      f"RMSECV={r['rmsecv']:.3f} RMSEP={r['rmsep']:.3f}  ({dt:.0f}s VIP+PSO selection)")
                summary_rows.append({"assay": assay, "held_out": held, "analyte": a, "model": M_,
                                     "n_vars": r["n_vars"], "lv": r["lv"], "rmsecv": round(r["rmsecv"], 4),
                                     "rmsep": round(r["rmsep"], 4)})

        # --- figures: RMSECV-vs-LV, parity, monitoring (per model) ---
        xmax = ASSAY_DURATION_H[held]
        for M_ in MODELS:
            mlabel = M.MODEL_LABEL[M_]

            # 1) RMSECV vs LV
            fig, ax = plt.subplots(figsize=M.figsize_mm(90, 70))
            for a in ANALYTES:
                r = res[a][M_]
                lvs = np.arange(1, r["km"] + 1)
                ax.plot(lvs, r["rk"], color=M.COLORS[a], linestyle=M.LINESTYLES[a], marker=M.MARKERS[a],
                        markersize=4, linewidth=1.1, label=M.LABELS[a])
                ax.plot(r["lv"], r["rk"][r["lv"] - 1], "*", color=M.COLORS[a], markersize=14,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            ax.set_xlabel("Latent variables (LVs)"); ax.set_ylabel("RMSECV (g L$^{-1}$)")
            ax.set_title(f"{mlabel} - held out {assay}  ($\\bigstar$ = selected LVs)", loc="left", fontsize=8)
            ax.legend(frameon=False, loc="best"); M.despine(ax); fig.tight_layout()
            name = f"fig_rmsecv_{M_.lower()}_{slug}"
            M.save_fig(fig, name, OUTDIR_FIG); plt.close(fig)
            rows = [{"analyte": a, "lv": kk, "rmsecv": round(float(res[a][M_]["rk"][kk - 1]), 4),
                    "selected": int(kk == res[a][M_]["lv"])}
                    for a in ANALYTES for kk in range(1, res[a][M_]["km"] + 1)]
            M.write_data_csv(OUTDIR_FIG, name, ["analyte", "lv", "rmsecv", "selected"], rows)

            # 2) Parity (cv + held-out test)
            fig, axes = plt.subplots(1, 2, figsize=M.figsize_mm(190, 92))
            par_rows = []
            for ax, a, letter in zip(axes, ANALYTES, ("a", "b")):
                r = res[a][M_]; y_tr_a, y_te_a = Y_tr[:, ANALYTES[a]], Y_te[:, ANALYTES[a]]
                cv_rows = [{"set": "cv", "analyte": a, "measured": round(float(m), 4),
                           "predicted": round(float(p), 4)} for m, p in zip(y_tr_a, r["cvp"])]
                te_rows = [{"set": "test", "analyte": a, "measured": round(float(m), 4),
                           "predicted": round(float(p), 4)} for m, p in zip(y_te_a, r["tp"])]
                par_rows += cv_rows + te_rows
                M._parity_panel(ax, a, letter, cv_rows, "Cross-validation", te_rows, r["rmsep"])
            fig.suptitle(f"{mlabel} - held out {assay}", fontsize=9)
            fig.tight_layout()
            name = f"fig_parity_{M_.lower()}_{slug}"
            M.save_fig(fig, name, OUTDIR_FIG); plt.close(fig)
            M.write_data_csv(OUTDIR_FIG, name, ["set", "analyte", "measured", "predicted"], par_rows)

            # 3) Continuous monitoring of the held-out batch
            fig, ax1 = plt.subplots(figsize=M.figsize_mm(90, 75))
            mon_rows = []
            for a in ANALYTES:
                r = res[a][M_]
                y = M.moving_average(r["infp"], M.MA_WINDOW)
                ax1.plot(t_inf, y, color=M.COLORS[a], linestyle=M.LINESTYLES[a], linewidth=1.2, zorder=3)
                ax1.fill_between(t_inf, y - r["rmsep"], y + r["rmsep"], color=M.COLORS[a], alpha=0.15,
                                 linewidth=0, zorder=2)
                ax1.plot(time_h[held], conc[held][a], marker=M.MARKERS[a], linestyle="none",
                         markerfacecolor=M.COLORS[a], markeredgecolor="black", markeredgewidth=0.6,
                         markersize=5, zorder=4)
                mon_rows += [{"analyte": a, "time": round(float(tt), 4), "pred": round(float(yy), 4),
                             "pred_raw": round(float(pp), 4)}
                             for tt, yy, pp in zip(t_inf, y, r["infp"])]
            ax1.set_xlabel("Time (h)"); ax1.set_ylabel(M.CONC); ax1.set_ylim(bottom=0); ax1.set_xlim(0, xmax)
            M.despine(ax1)
            ah = [Line2D([0], [0], color=M.COLORS[a], marker=M.MARKERS[a], linestyle=M.LINESTYLES[a],
                         markersize=5, markerfacecolor=M.COLORS[a], markeredgecolor="black",
                         label=M.LABELS[a]) for a in ANALYTES]
            eh = [Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.2, label="NIR prediction"),
                  Patch(facecolor="0.3", alpha=0.15, label="$\\pm$ RMSEP band"),
                  Line2D([0], [0], color="0.3", marker="o", linestyle="none", markerfacecolor="0.3",
                         markeredgecolor="black", label=f"HPLC reference ({assay})")]
            leg1 = ax1.legend(handles=ah, frameon=False, loc="upper left", fontsize=7); ax1.add_artist(leg1)
            ax1.legend(handles=eh, frameon=False, loc="lower right", fontsize=7)
            ax1.set_title(f"{mlabel} - held out {assay}", loc="left", fontsize=8)
            fig.tight_layout()
            name = f"fig_monitoring_{M_.lower()}_{slug}"
            M.save_fig(fig, name, OUTDIR_FIG); plt.close(fig)
            M.write_data_csv(OUTDIR_FIG, name, ["analyte", "time", "pred", "pred_raw"], mon_rows)

        print(f"  -> figures for {assay} done ({time.perf_counter() - t0_all:.0f}s elapsed total)\n")

    B.write_csv(os.path.join(OUTDIR_DATA, "lobo_all_summary.csv"),
                ["assay", "held_out", "analyte", "model", "n_vars", "lv", "rmsecv", "rmsep"], summary_rows)
    print(f"Summary written: {OUTDIR_DATA}/lobo_all_summary.csv")
    print(f"Figures written: {OUTDIR_FIG}/  (total time {time.perf_counter() - t0_all:.0f}s)")


if __name__ == "__main__":
    main()
