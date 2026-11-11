"""
manuscript_analysis.py
======================
Consolidated NIR/PLS analysis for the manuscript (Tasks 1-6, +7 figures). REUSES the
existing pipeline identically (build_paper_data.py = B, run_loexo_plsvarsel_exp4to7.py = V):
  - preprocessing:  Cut(4400-7500) + Savitzky-Golay(window 15 / radius 7, order 2, 1st deriv) = 3101 vars
  - LV selection:   Osten F-test (alpha=0.05) via V.select_k_ftest
  - reference model: full-spectrum PLS (deterministic; no seeds)

Two models:
  M_primario : train exp4+exp5+exp6 (Assay 1-3) -> held-out exp7 (Assay 4)
  M_menos1   : train exp5+exp6+exp7 (Assay 2-4) -> held-out exp4 (Assay 1)

Anti-circularity: each monitoring stream's applicability-domain (T2, Q) limits come ONLY
from that model's own training set. Monitor exp7 -> limits from exp4/5/6; monitor exp4 ->
limits from exp5/6/7.

Applicability domain:
  Hotelling T2 limit:  T2_lim = A*(n-1)/(n-A) * F(alpha, A, n-A)          [F-distribution]
  Q / SPE limit:       Jackson-Mudholkar approximation on the eigenvalues of the calibration
                       X-residual after A PLS components.
  A = F-test-selected LVs for that (model, analyte). Primary T2/Q/flag columns use the GLUCOSE
  model (primary analyte); xylose AD given as extra columns. The 95% AD here is byte-identical
  to B.applicability_domain (asserted at runtime).

Outputs: ./results/*.csv + ./results/summary.md ; ./figures/fig_task_*.{pdf,tif}
Run:  python scripts/plotting/manuscript_analysis.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import f as f_dist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

RESULTS, FIGDIR = "results", "figures"
EXPERIMENTS = ["exp4", "exp5", "exp6", "exp7"]
ASSAY = {"exp4": "Assay 1", "exp5": "Assay 2", "exp6": "Assay 3", "exp7": "Assay 4"}
ANALYTES = ["glucose", "xylose"]
INF = {e: os.path.join("data", f"exp_{e[3:].zfill(2)}_inf.txt") for e in EXPERIMENTS}
MAX_LV = 15
XMAX_H = 24.0
TABLE2 = {"glucose": {"exp4": 0.99, "exp5": 4.53, "exp6": 3.02, "exp7": 1.06},
          "xylose":  {"exp4": 0.91, "exp5": 1.68, "exp6": 1.15, "exp7": 1.00}}
COLORS = {"glucose": "#C0392B", "xylose": "#6C3483"}
MARK = {"glucose": "o", "xylose": "^"}
LS = {"glucose": "-", "xylose": "--"}
LAB = {"glucose": "Glucose", "xylose": "Xylose"}
_md = []


def log(s=""):
    print(s); _md.append(s)


def wcsv(name, fieldnames, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def load_inf(path):
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))
    return raw[:, 0] / 60.0, apply_pretreatment(B.PRETREAT, raw[:, 1:], wl, plot=False)[0]


def ftest_lv(A_tr, y_tr):
    rk, ycv, km = B.cv_curve(A_tr, y_tr, MAX_LV)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km)), rk, ycv, km


# ---- applicability domain: ingredients once, limits at any alpha (mirrors B exactly) ----
def ad_ingredients(X_cal, y_cal, X_new, A):
    Xn, Xmu, Xsig = V.zscore_matlab_style(X_cal)
    yn, _, _ = V.zscore_matlab_style(y_cal.reshape(-1, 1))
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    _, _, P, _, _, W, _, _ = V.PLS().nipals(Xn, yn, A)
    P, W = P[:, :A], W[:, :A]
    R = W @ np.linalg.pinv(P.T @ W)
    n = Xn.shape[0]; T_cal = Xn @ R
    s2 = T_cal.var(axis=0, ddof=1); s2[s2 <= 0] = 1e-12
    E = Xn - T_cal @ P.T
    sv = np.linalg.svd(E, compute_uv=False); lam = (sv ** 2) / (n - 1); lam = lam[lam > 1e-12]
    th = (float(lam.sum()), float((lam ** 2).sum()), float((lam ** 3).sum()))
    Xn_new = (X_new - Xmu) / Xs; T_new = Xn_new @ R
    T2 = np.sum((T_new ** 2) / s2, axis=1)
    Q = np.sum((Xn_new - T_new @ P.T) ** 2, axis=1)
    return dict(T2=T2, Q=Q, th=th, n=n, A=A)


def t2_lim(A, n, alpha):
    return A * (n - 1) / (n - A) * f_dist.ppf(alpha, A, n - A)


def q_lim(th, alpha):
    th1, th2, th3 = th
    if th1 <= 0 or th2 <= 0:
        return np.inf
    h0 = max(1e-3, 1.0 - (2.0 * th1 * th3) / (3.0 * th2 ** 2)); z = norm.ppf(alpha)
    return th1 * ((z * np.sqrt(2.0 * th2 * h0 ** 2)) / th1 + 1.0 + (th2 * h0 * (h0 - 1.0)) / th1 ** 2) ** (1.0 / h0)


def backward_ma(vals, w=4):
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        out[i] = np.mean(vals[max(0, i - w + 1):i + 1])
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    refe = {e: B.load_refe(e) for e in EXPERIMENTS}
    nd = {e: B.load_nonda(e) for e in EXPERIMENTS}
    spec = {e: nd[e][2] for e in EXPERIMENTS}
    conc = {e: {"glucose": refe[e][2], "xylose": refe[e][3]} for e in EXPERIMENTS}
    thplc = {e: refe[e][0] for e in EXPERIMENTS}
    n_samp = {e: spec[e].shape[0] for e in EXPERIMENTS}
    MODELS = {"M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7"},
              "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4"}}

    # sanity: our 95% AD ingredients reproduce B.applicability_domain
    A_tr0 = np.vstack([spec[e] for e in MODELS["M_primario"]["train"]])
    y0 = np.concatenate([conc[e]["glucose"] for e in MODELS["M_primario"]["train"]])
    ing0 = ad_ingredients(A_tr0, y0, spec["exp7"], 6)
    b0 = B.applicability_domain(A_tr0, y0, spec["exp7"], 6)
    assert abs(q_lim(ing0["th"], 0.95) - b0["Q_lim"]) < 1e-6 and abs(t2_lim(6, ing0["n"], 0.95) - b0["T2_lim"]) < 1e-6, \
        "AD helper diverges from B.applicability_domain"

    log("# Consolidated NIR/PLS analysis — manuscript scalars\n")
    log(f"Pipeline (reused identically): Cut 4400-7500 cm^-1 + Savitzky-Golay (window 15, order 2, "
        f"1st deriv) = {spec['exp4'].shape[1]} variables; full-spectrum PLS; LVs by Osten F-test (alpha=0.05).")
    log("NOTE: the code pipeline applies Cut+SG only (the SG 1st-derivative removes baseline); there is no "
        "separate SNV/normalisation step, despite the manuscript wording 'normalized'. Flagged for your review.")
    log("Samples per assay (HPLC-paired): " + ", ".join(f"{ASSAY[e]}={n_samp[e]}" for e in EXPERIMENTS) + ".")
    log(f"M_primario train=Assay 1-3 (n={sum(n_samp[e] for e in MODELS['M_primario']['train'])}), test=Assay 4 (n={n_samp['exp7']}).")
    log(f"M_menos1   train=Assay 2-4 (n={sum(n_samp[e] for e in MODELS['M_menos1']['train'])}), test=Assay 1 (n={n_samp['exp4']}).\n")

    # ===================== TASK 1 =====================
    log("## Task 1 — Concentration ranges (HPLC) + interpolation/extrapolation")
    rng = {e: {a: (float(conc[e][a].min()), float(conc[e][a].max())) for a in ANALYTES} for e in EXPERIMENTS}
    range_rows = [{"level": "batch", "id": ASSAY[e], "analyte": a, "min": round(rng[e][a][0], 3),
                   "max": round(rng[e][a][1], 3), "cal_min": "", "cal_max": "", "class": ""}
                  for e in EXPERIMENTS for a in ANALYTES]
    for e in EXPERIMENTS:
        log(f"  {ASSAY[e]}: glucose[{rng[e]['glucose'][0]:.2f}, {rng[e]['glucose'][1]:.2f}], "
            f"xylose[{rng[e]['xylose'][0]:.2f}, {rng[e]['xylose'][1]:.2f}] g/L")
    for mname, m in MODELS.items():
        te = m["test"]
        for a in ANALYTES:
            cmin = min(rng[e][a][0] for e in m["train"]); cmax = max(rng[e][a][1] for e in m["train"])
            tmin, tmax = rng[te][a]
            cls = "interpolation" if (tmin >= cmin and tmax <= cmax) else "extrapolation"
            range_rows.append({"level": mname, "id": ASSAY[te], "analyte": a, "min": round(tmin, 3),
                               "max": round(tmax, 3), "cal_min": round(cmin, 3), "cal_max": round(cmax, 3), "class": cls})
            log(f"  {mname} {a}: cal[{cmin:.2f}, {cmax:.2f}] vs {ASSAY[te]}[{tmin:.2f}, {tmax:.2f}] -> {cls.upper()}")
    wcsv("ranges.csv", ["level", "id", "analyte", "min", "max", "cal_min", "cal_max", "class"], range_rows)
    log("")

    # ===================== TASK 2 =====================
    log("## Task 2 — Full LOBO (single pipeline, F-test LVs) — canonical RMSEP")
    lobo_vals = {a: [] for a in ANALYTES}; lobo_lv = {a: [] for a in ANALYTES}; lobo_out = []
    for held in EXPERIMENTS:
        train = [e for e in EXPERIMENTS if e != held]
        A_tr = np.vstack([spec[e] for e in train]); A_te = spec[held]
        row = {"held_out": ASSAY[held]}
        for a in ANALYTES:
            y_tr = np.concatenate([conc[e][a] for e in train]); y_te = conc[held][a]
            lv, *_ = ftest_lv(A_tr, y_tr)
            yp = B.pls_predict(A_tr, y_tr, A_te, lv)
            rmsep = float(np.sqrt(np.mean((y_te - yp) ** 2)))
            row[f"rmsep_{a}"] = round(rmsep, 3); row[f"lv_{a}"] = lv
            lobo_vals[a].append(rmsep); lobo_lv[a].append(lv)
        lobo_out.append(row)
    log("| Held-out | RMSEP glu | LV glu | RMSEP xyl | LV xyl |")
    log("|---|---|---|---|---|")
    for r in lobo_out:
        log(f"| {r['held_out']} | {r['rmsep_glucose']:.3f} | {r['lv_glucose']} | {r['rmsep_xylose']:.3f} | {r['lv_xylose']} |")
    ms = {}
    for a in ANALYTES:
        mean, sd = float(np.mean(lobo_vals[a])), float(np.std(lobo_vals[a], ddof=1)); ms[a] = (mean, sd)
        log(f"  {a}: mean+/-SD = {mean:.3f} +/- {sd:.3f} g/L  (2 dp: {mean:.2f} +/- {sd:.2f})")
    log("Cross-check vs current Table 2 (glu 0.99/4.53/3.02/1.06; xyl 0.91/1.68/1.15/1.00):")
    diverge = False
    for a in ANALYTES:
        for i, held in enumerate(EXPERIMENTS):
            d = abs(lobo_vals[a][i] - TABLE2[a][held])
            fl = "" if d <= 0.01 else f"  <-- DIVERGENCE (delta={d:.3f})"
            if d > 0.01:
                diverge = True
            log(f"  {a} {ASSAY[held]}: computed={lobo_vals[a][i]:.3f} vs Table2={TABLE2[a][held]:.2f}{fl}")
    log(f"  -> {'DIVERGENCES found (see above)' if diverge else 'all match Table 2 within 0.01'}")
    canon = lobo_vals["glucose"][EXPERIMENTS.index("exp7")]
    log(f"CANONICAL glucose RMSEP, held-out Assay 4 (exp7), F-test LV={lobo_lv['glucose'][EXPERIMENTS.index('exp7')]}: "
        f"{canon:.3f} g/L (2 dp: {canon:.2f}). The 1.116/1.12 elsewhere come from a FIXED LV=6 model, "
        f"not the F-test pipeline.")
    lobo_csv = [{"held_out": r["held_out"], "rmsep_glucose": r["rmsep_glucose"], "lv_glucose": r["lv_glucose"],
                 "rmsep_xylose": r["rmsep_xylose"], "lv_xylose": r["lv_xylose"]} for r in lobo_out]
    lobo_csv += [{"held_out": "MEAN", "rmsep_glucose": round(ms["glucose"][0], 3), "lv_glucose": "",
                  "rmsep_xylose": round(ms["xylose"][0], 3), "lv_xylose": ""},
                 {"held_out": "SD", "rmsep_glucose": round(ms["glucose"][1], 3), "lv_glucose": "",
                  "rmsep_xylose": round(ms["xylose"][1], 3), "lv_xylose": ""}]
    wcsv("lobo.csv", ["held_out", "rmsep_glucose", "lv_glucose", "rmsep_xylose", "lv_xylose"], lobo_csv)
    log("")

    # ===================== TASK 3 =====================
    log("## Task 3 — M_menos1 parity on held-out Assay 1 (exp4)")
    m = MODELS["M_menos1"]; A_tr = np.vstack([spec[e] for e in m["train"]]); A_te = spec["exp4"]
    par_rows = []; m1 = {}
    for a in ANALYTES:
        y_tr = np.concatenate([conc[e][a] for e in m["train"]]); y_te = conc["exp4"][a]
        lv, *_ = ftest_lv(A_tr, y_tr); yp = B.pls_predict(A_tr, y_tr, A_te, lv)
        rmsep = float(np.sqrt(np.mean((y_te - yp) ** 2))); m1[a] = dict(lv=lv, rmsep=rmsep, meas=y_te, pred=yp)
        par_rows += [{"analyte": a, "time_h": round(float(t), 4), "measured": round(float(me), 4),
                      "predicted": round(float(pr), 4)} for t, me, pr in zip(thplc["exp4"], y_te, yp)]
        log(f"  {a}: LV={lv}, RMSEP={rmsep:.3f} g/L (n={len(y_te)} HPLC points)")
    wcsv("parity_exp4.csv", ["analyte", "time_h", "measured", "predicted"], par_rows)
    log("")

    # ===================== TASKS 4 & 5 =====================
    def monitor(mname, stream, out_csv):
        m = MODELS[mname]; A_tr = np.vstack([spec[e] for e in m["train"]])
        t_on, S_on = load_inf(INF[stream]); keep = t_on <= XMAX_H; t_on, S_on = t_on[keep], S_on[keep]
        r = {}
        for a in ANALYTES:
            y_tr = np.concatenate([conc[e][a] for e in m["train"]]); lv, *_ = ftest_lv(A_tr, y_tr)
            ing = ad_ingredients(A_tr, y_tr, S_on, lv)
            r[a] = dict(lv=lv, pred=B.pls_predict(A_tr, y_tr, S_on, lv), T2=ing["T2"], Q=ing["Q"],
                        th=ing["th"], n=ing["n"])
        g = r["glucose"]
        t2L, qL = t2_lim(g["lv"], g["n"], 0.95), q_lim(g["th"], 0.95)
        eT2, eQ = g["T2"] > t2L, g["Q"] > qL
        union, inter = eT2 | eQ, eT2 & eQ; retained = ~union
        gret = np.full(len(t_on), np.nan); xret = np.full(len(t_on), np.nan)
        if retained.any():
            gret[retained] = backward_ma(r["glucose"]["pred"][retained], 4)
            xret[retained] = backward_ma(r["xylose"]["pred"][retained], 4)
        rows = [{"time_h": round(float(t_on[i]), 4),
                 "pred_glucose_raw": round(float(r["glucose"]["pred"][i]), 4),
                 "pred_xylose_raw": round(float(r["xylose"]["pred"][i]), 4),
                 "T2": round(float(g["T2"][i]), 4), "Q": round(float(g["Q"][i]), 4),
                 "exceeds_T2": int(eT2[i]), "exceeds_Q": int(eQ[i]),
                 "exceeds_union": int(union[i]), "exceeds_intersection": int(inter[i]),
                 "pred_glucose_retained": "" if np.isnan(gret[i]) else round(float(gret[i]), 4),
                 "pred_xylose_retained": "" if np.isnan(xret[i]) else round(float(xret[i]), 4),
                 "xylose_T2": round(float(r["xylose"]["T2"][i]), 4),
                 "xylose_Q": round(float(r["xylose"]["Q"][i]), 4)} for i in range(len(t_on))]
        wcsv(out_csv, list(rows[0].keys()), rows)
        return dict(t=t_on, r=r, t2L=t2L, qL=qL, eT2=eT2, eQ=eQ, union=union, inter=inter,
                    retained=retained, gret=gret, xret=xret, stream=stream, model=mname)

    def report(mon, task):
        t, g = mon["t"], mon["r"]["glucose"]; x = mon["r"]["xylose"]
        log(f"### {task} — stream {ASSAY[mon['stream']]} ({mon['stream']}); AD limits from {mon['model']} training")
        log(f"  online spectra (<=24 h)={len(t)}; calibration n={g['n']}; A(glu LV)={g['lv']}, A(xyl LV)={x['lv']}")
        log(f"  Q limit: Jackson-Mudholkar; T2 limit: F-distribution.")
        log(f"  GLUCOSE model (primary): T2_lim(95%)={mon['t2L']:.3f}, Q_lim(95%)={mon['qL']:.1f}")
        log(f"  XYLOSE  model: T2_lim(95%)={t2_lim(x['lv'], x['n'], 0.95):.3f}, Q_lim(95%)={q_lim(x['th'], 0.95):.1f}")
        for nm, mask in [("T2-only", mon["eT2"] & ~mon["eQ"]), ("Q-only", mon["eQ"] & ~mon["eT2"]),
                         ("union", mon["union"]), ("intersection", mon["inter"])]:
            n = int(mask.sum()); tw = f" @ t=[{t[mask].min():.2f},{t[mask].max():.2f}] h" if n else ""
            log(f"  flagged {nm:13s}: {n}/{len(t)}{tw}")
        su = t < 1.0
        log(f"  start-up (t<1 h): {int(su.sum())} spectra; union-flagged={int((su & mon['union']).sum())} "
            f"(by T2={int((su & mon['eT2']).sum())}, by Q={int((su & mon['eQ']).sum())}); "
            f"retained overall={int(mon['retained'].sum())}/{len(t)}")

    log("## Task 4 — Monitoring exp7 (Assay 4) by M_primario (limits from Assay 1-3)")
    mon7 = monitor("M_primario", "exp7", "monitor_exp7.csv"); report(mon7, "Task 4"); log("")
    log("## Task 5 — Monitoring exp4 (Assay 1) by M_menos1 — specificity (limits from Assay 2-4)")
    mon4 = monitor("M_menos1", "exp4", "monitor_exp4.csv"); report(mon4, "Task 5"); log("")

    # ===================== TASK 6 =====================
    log("## Task 6 — Exclusion-rule summary (T2 vs Q; union vs intersection; 95% & 99%)")
    fs_rows = []
    for mon, nm in [(mon7, "exp7 (M_primario)"), (mon4, "exp4 (M_menos1)")]:
        g = mon["r"]["glucose"]; t = mon["t"]
        for level, al in [("95%", 0.95), ("99%", 0.99)]:
            t2L = t2_lim(g["lv"], g["n"], al); qL = q_lim(g["th"], al)
            eT2, eQ = g["T2"] > t2L, g["Q"] > qL
            fs_rows.append({"stream": nm, "level": level, "T2_lim": round(float(t2L), 3),
                            "Q_lim": round(float(qL), 1), "n_T2_only": int((eT2 & ~eQ).sum()),
                            "n_Q_only": int((eQ & ~eT2).sum()), "n_union": int((eT2 | eQ).sum()),
                            "n_intersection": int((eT2 & eQ).sum()), "n_total": len(t)})
            log(f"  {nm} @ {level}: T2_lim={t2L:.2f}, Q_lim={qL:.0f} | T2-only={int((eT2 & ~eQ).sum())}, "
                f"Q-only={int((eQ & ~eT2).sum())}, union={int((eT2 | eQ).sum())}, inter={int((eT2 & eQ).sum())} (of {len(t)})")
    wcsv("flag_summary.csv", ["stream", "level", "T2_lim", "Q_lim", "n_T2_only", "n_Q_only",
                              "n_union", "n_intersection", "n_total"], fs_rows)
    log("")

    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(_md))

    # ===================== TASK 7 (figures) =====================
    figures(spec, conc, thplc, MODELS, mon7, m1, lobo_vals, lobo_lv)
    print(f"\n[Saved results/*.csv, results/summary.md, figures/fig_task_*]")


def _save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    try:
        fig.savefig(os.path.join(FIGDIR, f"{name}.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:
        print(f"  tif skipped: {e}")
    plt.close(fig)


def figures(spec, conc, thplc, MODELS, mon7, m1, lobo_vals, lobo_lv):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})
    mm = 1 / 25.4

    def despine(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (a) kinetics
    fig, ax = plt.subplots(figsize=(90 * mm, 70 * mm))
    for e in EXPERIMENTS:
        test = (e == "exp7")
        for a in ANALYTES:
            t = thplc[e]; o = np.argsort(t)
            ax.plot(t[o], conc[e][a][o], marker=MARK[a], linestyle=LS[a], color=COLORS[a], markersize=5,
                    linewidth=1.0, markerfacecolor=("none" if test else COLORS[a]), markeredgecolor=COLORS[a])
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (g L$^{-1}$)"); ax.set_ylim(bottom=0); ax.set_xlim(left=0)
    h = [Line2D([0], [0], color=COLORS[a], marker=MARK[a], linestyle=LS[a], label=LAB[a]) for a in ANALYTES]
    h += [Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="0.3", label="Calibration (Assay 1-3)"),
          Line2D([0], [0], color="0.3", marker="s", linestyle="none", markerfacecolor="none", label="Test (Assay 4)")]
    ax.legend(handles=h, frameon=False, loc="lower right"); despine(ax); _save(fig, "fig_task_kinetics")

    # (b) parity: out-of-fold LOBO within cal(exp4/5/6) + M_primario->exp7
    fig, axes = plt.subplots(1, 2, figsize=(190 * mm, 92 * mm))
    cal = MODELS["M_primario"]["train"]
    for ax, a, letter in zip(axes, ANALYTES, ("a", "b")):
        mcal, pcal = [], []
        for held in cal:
            tr = [e for e in cal if e != held]; A_tr = np.vstack([spec[e] for e in tr])
            y_tr = np.concatenate([conc[e][a] for e in tr]); lv, *_ = ftest_lv(A_tr, y_tr)
            yp = B.pls_predict(A_tr, y_tr, spec[held], lv)
            mcal += list(conc[held][a]); pcal += list(yp)
        A_tr = np.vstack([spec[e] for e in cal]); y_tr = np.concatenate([conc[e][a] for e in cal])
        lv, *_ = ftest_lv(A_tr, y_tr); ptest = B.pls_predict(A_tr, y_tr, spec["exp7"], lv)
        mtest = conc["exp7"][a]
        rmsep_t = float(np.sqrt(np.mean((mtest - ptest) ** 2)))
        allv = np.array(mcal + list(mtest) + pcal + list(ptest)); lo, hi = allv.min(), allv.max()
        pad = (hi - lo) * 0.06; lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=0.8)
        ax.scatter(mcal, pcal, marker=MARK[a], facecolors=COLORS[a], edgecolors="none", s=26, alpha=0.85,
                   label="Calibration (out-of-fold)")
        ax.scatter(mtest, ptest, marker=MARK[a], facecolors="none", edgecolors="black", linewidths=0.9, s=34,
                   label="Independent test (Assay 4)")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", "box")
        ax.set_xlabel(f"Measured {a} (g L$^{{-1}}$)"); ax.set_ylabel(f"Predicted {a} (g L$^{{-1}}$)")
        ax.text(0.05, 0.95, f"Test RMSEP = {rmsep_t:.2f} g L$^{{-1}}$ (LV={lv})", transform=ax.transAxes,
                va="top", fontsize=7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8))
        ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.legend(frameon=False, loc="lower right"); despine(ax)
    _save(fig, "fig_task_parity")

    # (c) monitoring exp7: raw (light) + retained (solid) + band + HPLC + startup union-flags as x
    fig, ax = plt.subplots(figsize=(90 * mm, 75 * mm))
    t = mon7["t"]; rmsep = {"glucose": lobo_vals["glucose"][EXPERIMENTS.index("exp7")],
                            "xylose": lobo_vals["xylose"][EXPERIMENTS.index("exp7")]}
    for a in ANALYTES:
        raw = mon7["r"][a]["pred"]
        ax.plot(t, raw, color=COLORS[a], alpha=0.25, linewidth=0.6, zorder=1)
        ret = mon7["gret"] if a == "glucose" else mon7["xret"]
        ok = ~np.isnan(ret)
        ax.plot(t[ok], ret[ok], color=COLORS[a], linestyle=LS[a], linewidth=1.3, zorder=3)
        ax.fill_between(t[ok], ret[ok] - rmsep[a], ret[ok] + rmsep[a], color=COLORS[a], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(thplc["exp7"], conc["exp7"][a], marker=MARK[a], linestyle="none", markerfacecolor=COLORS[a],
                markeredgecolor="black", markeredgewidth=0.6, markersize=5, zorder=4)
    su = (t < 1.0) & mon7["union"]
    if su.any():
        for a in ANALYTES:
            ax.plot(t[su], mon7["r"][a]["pred"][su], marker="x", linestyle="none", color="0.5", markersize=4, zorder=5)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (g L$^{-1}$)"); ax.set_ylim(bottom=0); ax.set_xlim(0, XMAX_H)
    ah = [Line2D([0], [0], color=COLORS[a], marker=MARK[a], linestyle=LS[a], markerfacecolor=COLORS[a],
                 markeredgecolor="black", label=LAB[a]) for a in ANALYTES]
    eh = [Line2D([0], [0], color="0.3", linewidth=1.3, label="Retained (moving avg.)"),
          Line2D([0], [0], color="0.3", alpha=0.25, linewidth=0.6, label="Raw per-spectrum"),
          Patch(facecolor="0.3", alpha=0.15, label="$\\pm$ RMSEP"),
          Line2D([0], [0], color="0.3", marker="o", linestyle="none", markerfacecolor="0.3",
                 markeredgecolor="black", label="HPLC (Assay 4)"),
          Line2D([0], [0], color="0.5", marker="x", linestyle="none", label="Start-up flagged (union)")]
    l1 = ax.legend(handles=ah, frameon=False, loc="upper left", fontsize=7); ax.add_artist(l1)
    ax.legend(handles=eh, frameon=False, loc="lower right", fontsize=6); despine(ax)
    _save(fig, "fig_task_monitor_exp7")

    # (d) influence plot: T2 vs Q (log-log so the 95% limits and structure are readable)
    fig, ax = plt.subplots(figsize=(90 * mm, 75 * mm))
    g = mon7["r"]["glucose"]; t = mon7["t"]
    su = t < 1.0
    ax.scatter(g["T2"][~su], g["Q"][~su], s=10, color="0.5", alpha=0.6, label="Online spectra (t$\\geq$1 h)")
    ax.scatter(g["T2"][su], g["Q"][su], s=18, color=COLORS["glucose"], edgecolors="black", linewidths=0.4,
               label="Start-up (t<1 h)", zorder=4)
    ax.axvline(mon7["t2L"], color="0.3", linestyle="--", linewidth=0.9)
    ax.axhline(mon7["qL"], color="0.3", linestyle="--", linewidth=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Hotelling $T^2$"); ax.set_ylabel("Q residual (SPE)")
    ax.text(0.97, 0.05, f"95% limits (dashed):\n$T^2$={mon7['t2L']:.1f}, Q={mon7['qL']:.0f}\n"
            f"All {len(t)} spectra exceed Q", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5); despine(ax)
    _save(fig, "fig_task_influence_exp7")
    print("  figures: fig_task_{kinetics,parity,monitor_exp7,influence_exp7}")


if __name__ == "__main__":
    main()
