"""
diagnose_ad.py  — READ-ONLY DIAGNOSTIC (does NOT modify any analysis script)
============================================================================
Audits the applicability-domain (T2 / Q) computation used for exp7 (Assay 4)
monitoring under M_primario (train Assay 1-3). Reproduces the EXACT path of
manuscript_analysis.py / build_paper_data.applicability_domain by importing the
same functions; instruments each step. No refit, no fix, no RMSEP recompute.

Blocks: AUDIT A (preprocessing trace), B (normalization traceability),
C (calibration self-check — decisive), D (exp7 Q/T2 distribution), E (figure vs count).
Saves: results/audit_preprocessing.csv, ad_calibration_check.csv, ad_exp7_distribution.csv
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import f as f_dist, norm
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

RESULTS = "results"
CAL = ["exp4", "exp5", "exp6"]           # M_primario training = Assay 1-3
INF7 = os.path.join("data", "exp_07_inf.txt")
XMAX_H = 24.0


def wcsv(name, fieldnames, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def stats(M):
    return dict(shape=M.shape, gmin=float(M.min()), gmax=float(M.max()),
                gmean=float(M.mean()), gstd=float(M.std()))


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # ---- rebuild the exact inputs (same loaders as the analysis) ----
    # calibration raw -> cut -> SG   (per exp, then stacked)
    cal_raw, cal_cut, cal_pre = [], [], []
    for e in CAL:
        path = os.path.join("data", f"{e}_nonda.txt")
        with open(path) as f:
            wl = np.array([float(v) for v in f.readline().split()[1:]])
        raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
        cut = apply_pretreatment(B.PRETREAT_CUT, raw, wl, plot=False)[0]
        pre = apply_pretreatment(B.PRETREAT, raw, wl, plot=False)[0]
        cal_raw.append(raw); cal_cut.append(cut); cal_pre.append(pre)
    Xcal_raw = np.vstack(cal_raw); Xcal_cut = np.vstack(cal_cut); Xcal_pre = np.vstack(cal_pre)

    # online exp7 raw -> cut -> SG  (truncate <=24h to mirror manuscript_analysis)
    with open(INF7) as f:
        wl7 = np.array([float(v) for v in f.readline().split()[1:]])
    raw7_full = np.atleast_2d(np.loadtxt(INF7, skiprows=1))
    t7 = raw7_full[:, 0] / 60.0
    keep = t7 <= XMAX_H
    On_raw = raw7_full[keep, 1:]
    On_cut = apply_pretreatment(B.PRETREAT_CUT, On_raw, wl7, plot=False)[0]
    On_pre = apply_pretreatment(B.PRETREAT, On_raw, wl7, plot=False)[0]

    # F-test LV for glucose on the M_primario training (same as analysis' primary AD)
    y_cal_glu = np.concatenate([B.load_refe(e)[2] for e in CAL])
    rk, _, km = B.cv_curve(Xcal_pre, y_cal_glu, 15)
    A = int(min(V.select_k_ftest(rk, n_cal=len(y_cal_glu)), km))

    # ---- replicate applicability_domain internals to expose the scaling objects ----
    Xn, Xmu, Xsig = V.zscore_matlab_style(Xcal_pre)          # fit mean/scale on CALIBRATION
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    yn, _, _ = V.zscore_matlab_style(y_cal_glu.reshape(-1, 1))
    _, _, P, _, _, W, _, _ = V.PLS().nipals(Xn, yn, A)
    P, W = P[:, :A], W[:, :A]
    R = W @ np.linalg.pinv(P.T @ W)
    n = Xn.shape[0]; T_cal = Xn @ R
    s2 = T_cal.var(axis=0, ddof=1); s2[s2 <= 0] = 1e-12
    T2_lim = A * (n - 1) / (n - A) * f_dist.ppf(0.95, A, n - A)
    E = Xn - T_cal @ P.T
    sv = np.linalg.svd(E, compute_uv=False); lam = (sv ** 2) / (n - 1); lam = lam[lam > 1e-12]
    th1, th2, th3 = lam.sum(), (lam ** 2).sum(), (lam ** 3).sum()
    h0 = max(1e-3, 1.0 - (2.0 * th1 * th3) / (3.0 * th2 ** 2)); z = norm.ppf(0.95)
    Q_lim = float(th1 * ((z * np.sqrt(2.0 * th2 * h0 ** 2)) / th1 + 1.0 + (th2 * h0 * (h0 - 1.0)) / th1 ** 2) ** (1.0 / h0))
    T2_lim = float(T2_lim)

    # online autoscaled with CALIBRATION mean/scale (the exact code line)
    Xn_on = (On_pre - Xmu) / Xs
    T_on = Xn_on @ R
    Q_on = np.sum((Xn_on - T_on @ P.T) ** 2, axis=1)
    T2_on = np.sum((T_on ** 2) / s2, axis=1)
    # calibration's own T2/Q (self-projection)
    Q_cal = np.sum((Xn - T_cal @ P.T) ** 2, axis=1)
    T2_cal = np.sum((T_cal ** 2) / s2, axis=1)

    # sanity: identical to library function?
    ad_lib = B.applicability_domain(Xcal_pre, y_cal_glu, On_pre, A)
    same = np.allclose(ad_lib["Q"], Q_on) and np.allclose(ad_lib["T2"], T2_on) and abs(ad_lib["Q_lim"] - Q_lim) < 1e-6

    # ================= AUDIT A =================
    print("=" * 78 + "\nAUDIT A — preprocessing trace (calibration vs online)\n" + "=" * 78)
    print(f"F-test glucose LV (A) used for AD = {A}; calibration n = {n}")
    print(f"Reproduces build_paper_data.applicability_domain exactly: {same}")
    print("\n[CALIBRATION Assay 1-3]")
    for nm, M in [("raw (after drop time col)", Xcal_raw), ("after Cut", Xcal_cut),
                  ("after Cut+SG (1st deriv)", Xcal_pre), ("after autoscale (z-score)", Xn)]:
        s = stats(M); print(f"  {nm:32s} shape={s['shape']}  min={s['gmin']:+.4g} max={s['gmax']:+.4g} "
                            f"mean={s['gmean']:+.4g} std={s['gstd']:.4g}")
    print("\n[ONLINE exp7 (<=24 h)]")
    for nm, M in [("raw (after drop time col)", On_raw), ("after Cut", On_cut),
                  ("after Cut+SG (1st deriv)", On_pre), ("after autoscale (cal mean/scale)", Xn_on)]:
        s = stats(M); print(f"  {nm:32s} shape={s['shape']}  min={s['gmin']:+.4g} max={s['gmax']:+.4g} "
                            f"mean={s['gmean']:+.4g} std={s['gstd']:.4g}")
    print("\nNormalization/scatter/scaling steps applied (in order): Cut -> Savitzky-Golay(1st deriv) "
          "-> column autoscale (z-score). NO SNV / MSC / vector-norm / min-max.")
    on_own_mu = On_pre.mean(axis=0)
    print(f"\nMean-centering & scale: FIT on calibration, REUSED on online (same Xmu/Xsig object).")
    print(f"  Xmu[:5] (calibration, applied to BOTH) = {np.round(Xmu[:5], 5)}")
    print(f"  online's OWN mean[:5] (NOT used by code) = {np.round(on_own_mu[:5], 5)}")
    print(f"  -> the code applies the calibration mean to the online set (correct reuse).")
    scal, son = stats(Xn), stats(Xn_on)
    print(f"\nFinal matrix into PLS/Q/T2 — GLOBAL stats:")
    print(f"  calibration autoscaled: min={scal['gmin']:+.3g} max={scal['gmax']:+.3g} mean={scal['gmean']:+.3g} std={scal['gstd']:.3g}")
    print(f"  online     autoscaled: min={son['gmin']:+.3g} max={son['gmax']:+.3g} mean={son['gmean']:+.3g} std={son['gstd']:.3g}")
    wcsv("audit_preprocessing.csv",
         ["set", "stage", "rows", "cols", "min", "max", "mean", "std"],
         [{"set": st, "stage": nm, "rows": M.shape[0], "cols": M.shape[1],
           "min": round(float(M.min()), 6), "max": round(float(M.max()), 6),
           "mean": round(float(M.mean()), 6), "std": round(float(M.std()), 6)}
          for st, seq in [("calibration", [("raw", Xcal_raw), ("cut", Xcal_cut), ("cut_sg", Xcal_pre), ("autoscaled", Xn)]),
                          ("online_exp7", [("raw", On_raw), ("cut", On_cut), ("cut_sg", On_pre), ("autoscaled", Xn_on)])]
          for nm, M in seq])

    # ================= AUDIT C =================  (decisive; printed before D)
    exc_T2c = T2_cal > T2_lim; exc_Qc = Q_cal > Q_lim; exc_uc = exc_T2c | exc_Qc
    print("\n" + "=" * 78 + "\nAUDIT C — calibration self-check (decisive bug-vs-real test)\n" + "=" * 78)
    print(f"  T2_lim(95%)={T2_lim:.3f}   Q_lim(95%)={Q_lim:.3f}   (n={n}, A={A})")
    print(f"  exceeding T2 : {int(exc_T2c.sum())}/{n} ({100*exc_T2c.mean():.1f}%)  [expected ~5%]")
    print(f"  exceeding Q  : {int(exc_Qc.sum())}/{n} ({100*exc_Qc.mean():.1f}%)  [expected ~5%]")
    print(f"  exceeding union: {int(exc_uc.sum())}/{n} ({100*exc_uc.mean():.1f}%)")
    print(f"  Q  calibration : min={Q_cal.min():.1f} median={np.median(Q_cal):.1f} mean={Q_cal.mean():.1f} max={Q_cal.max():.1f}")
    print(f"  T2 calibration : min={T2_cal.min():.2f} median={np.median(T2_cal):.2f} mean={T2_cal.mean():.2f} max={T2_cal.max():.2f}")
    calib_ok = (exc_uc.mean() < 0.20) and (np.median(Q_cal) < 5 * Q_lim)
    print(f"  INTERPRETATION: calibration exceedance {'~= expected (5-15%)' if calib_ok else 'FAR above 5% / Q huge'} "
          f"-> {'calculation OK; exp7 shift can be real' if calib_ok else 'calculation likely BROKEN'}")
    wcsv("ad_calibration_check.csv", ["metric", "value"],
         [{"metric": "n", "value": n}, {"metric": "A", "value": A},
          {"metric": "T2_lim", "value": round(T2_lim, 4)}, {"metric": "Q_lim", "value": round(Q_lim, 4)},
          {"metric": "pct_exceed_T2", "value": round(100 * float(exc_T2c.mean()), 2)},
          {"metric": "pct_exceed_Q", "value": round(100 * float(exc_Qc.mean()), 2)},
          {"metric": "pct_exceed_union", "value": round(100 * float(exc_uc.mean()), 2)},
          {"metric": "Q_cal_min", "value": round(float(Q_cal.min()), 3)},
          {"metric": "Q_cal_median", "value": round(float(np.median(Q_cal)), 3)},
          {"metric": "Q_cal_mean", "value": round(float(Q_cal.mean()), 3)},
          {"metric": "Q_cal_max", "value": round(float(Q_cal.max()), 3)},
          {"metric": "T2_cal_median", "value": round(float(np.median(T2_cal)), 3)},
          {"metric": "T2_cal_max", "value": round(float(T2_cal.max()), 3)}])

    # ================= AUDIT D =================
    pcts = [5, 25, 50, 75, 95, 99]
    print("\n" + "=" * 78 + "\nAUDIT D — exp7 online Q/T2 distribution (n={})\n".format(len(Q_on)) + "=" * 78)
    print(f"  percentile:   " + "  ".join(f"p{p:<2d}" for p in pcts))
    print(f"  Q          :  " + "  ".join(f"{np.percentile(Q_on, p):.0f}" for p in pcts))
    print(f"  T2         :  " + "  ".join(f"{np.percentile(T2_on, p):.2f}" for p in pcts))
    print(f"  frac Q > Q_lim  = {100*np.mean(Q_on > Q_lim):.1f}%   frac T2 > T2_lim = {100*np.mean(T2_on > T2_lim):.1f}%")
    ratio = np.median(Q_on) / np.median(Q_cal)
    print(f"  median Q exp7 = {np.median(Q_on):.0f} ; median Q calibration = {np.median(Q_cal):.0f} ; "
          f"ratio = {ratio:.2f}x")
    print(f"  (ratio near 1-2x => mass barely above limit; ratio in the 100s-1000s => deeply out)")
    wcsv("ad_exp7_distribution.csv", ["percentile", "Q", "T2"],
         [{"percentile": f"p{p}", "Q": round(float(np.percentile(Q_on, p)), 3),
           "T2": round(float(np.percentile(T2_on, p)), 3)} for p in pcts] +
         [{"percentile": "frac_over_lim", "Q": round(float(np.mean(Q_on > Q_lim)), 4),
           "T2": round(float(np.mean(T2_on > T2_lim)), 4)},
          {"percentile": "median_ratio_exp7_over_cal", "Q": round(float(ratio), 3), "T2": ""}])

    # ================= AUDIT E =================
    n_above = int(np.sum(Q_on > Q_lim))
    print("\n" + "=" * 78 + "\nAUDIT E — figure vs count reconciliation\n" + "=" * 78)
    print(f"  fig_task_influence_exp7 code: x/y both LOG scale; axhline drawn at Q_lim={Q_lim:.1f}; "
          f"y-values = glucose-model Q (same as count).")
    print(f"  points with Q > Q_lim (should be above the dashed line): {n_above}/{len(Q_on)}")
    print(f"  reported union-flag count was 481/481; Q-exceed here = {n_above}/{len(Q_on)}.")
    print(f"  -> if {n_above}=={len(Q_on)}: figure & count are CONSISTENT (all points above the line); "
          f"the visual impression of 'most below' is the log-scale line sitting just under a dense cloud "
          f"whose smallest Q ({Q_on.min():.0f}) is still > Q_lim ({Q_lim:.0f}).")

    # ================= VERDICT =================
    verdict = ("CALCULO OK, SHIFT APARENTEMENTE REAL" if calib_ok
               else "PROVAVEL BUG DE PRE-PROCESSAMENTO")
    print("\n" + "#" * 78 + f"\nDIAGNOSTICO AD — VEREDITO: {verdict}\n" + "#" * 78)
    print(f"min Q exp7 = {Q_on.min():.0f} vs Q_lim = {Q_lim:.0f}; calibration union-exceed = "
          f"{100*exc_uc.mean():.1f}% (expected ~5-10%).")


if __name__ == "__main__":
    main()
