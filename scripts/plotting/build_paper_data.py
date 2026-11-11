"""
build_paper_data.py
===================
Generate every CSV consumed by make_paper_figures.py, with GUARANTEED model
provenance (leave-one-batch-out, full-spectrum PLS) plus a per-model comparison
(PLS / VIP-PLS / PSO-PLS). Dependency-light: numpy + scipy + csv (no pandas).

RAW INPUT FILES (data/):
  exp{4,5,6,7}_refe.txt  - HPLC reference: [time_min, cellobiose, glucose, xylose]
  exp{4,5,6,7}_nonda.txt - HPLC-paired NIR: row1 = "Time"+wavenumbers; rows = time_min+absorbance
  exp_07_inf.txt         - continuous ONLINE NIR of exp7 (600 spectra)

Assays:  exp4=Assay 1, exp5=Assay 2, exp6=Assay 3 (calibration, 20 g); exp7=Assay 4 (test, 15 g).
Preprocessing: Cut(4400-7500) [="raw" for spectra figure] + Savitzky-Golay(7,2,1).
Full-spectrum PLS LV = 6 (glucose), 4 (xylose); VIP/PSO LV chosen by Osten F-test.

OUTPUTS (data_fig/): hplc.csv, parity.csv, nir_exp7.csv, ad_limits.csv,
  spectra_raw_calibration.csv (after cut), spectra_pretreated_calibration.csv,
  model_rmsecv.csv, model_parity.csv, model_selected_vars.csv

Run:  python scripts/plotting/build_paper_data.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import f as f_dist, norm
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V   # PLS, zscore, _beta, _cv_pred, vip_mask, run_pso, select_k_ftest

DATA, OUTDIR = "data", "data_fig"
CAL, TEST = ["exp4", "exp5", "exp6"], "exp7"
INF_FILE = os.path.join(DATA, "exp_07_inf.txt")
ASSAY_NAME = {"exp4": "Assay 1", "exp5": "Assay 2", "exp6": "Assay 3", "exp7": "Assay 4"}

PRETREAT = [["Cut", 4400, 7500, 0], ["SG", 7, 2, 1, 0]]   # full pipeline
PRETREAT_CUT = [["Cut", 4400, 7500, 0]]                    # "raw" = after cut only
ANALYTES = {"glucose": 1, "xylose": 2}                     # index in (cb, gl, xy)
LV_FIXED = {"glucose": 6, "xylose": 4}                     # full-spectrum PLS
MAX_LV, K_FOLDS, CV_TYPE = 15, V.K_FOLDS, V.CV_TYPE
PSO_PARAMS = dict(n_particles=30, max_iter=120, w=0.9, c1=1.49, c2=1.49)


# ----------------------------------------------------------------- io
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def write_matrix_csv(path, columns):
    names = list(columns); n = len(columns[names[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(names)
        for i in range(n):
            w.writerow([round(float(columns[c][i]), 6) for c in names])


def load_refe(e):
    r = np.atleast_2d(np.loadtxt(os.path.join(DATA, f"{e}_refe.txt")))   # time, cb, gl, xy
    return r[:, 0] / 60.0, r[:, 1], r[:, 2], r[:, 3]


def load_nonda(e):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
    cut = apply_pretreatment(PRETREAT_CUT, raw, wl, plot=False)
    pre = apply_pretreatment(PRETREAT, raw, wl, plot=False)
    return cut[0], cut[1], pre[0], pre[1]      # spec_cut, wl_cut, spec_pre, wl_pre


def load_inf():
    with open(INF_FILE) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(INF_FILE, skiprows=1))
    return raw[:, 0] / 60.0, apply_pretreatment(PRETREAT, raw[:, 1:], wl, plot=False)[0]


# ----------------------------------------------------------------- model helpers
def pls_predict(A_tr, y_tr, A_te, k):
    Xn, Xmu, Xsig = V.zscore_matlab_style(A_tr)
    yn, ymu, ysig = V.zscore_matlab_style(y_tr.reshape(-1, 1))
    _, _, P, _, Q, W, _, _ = V.PLS().nipals(Xn, yn, k)
    beta = V._beta(P, Q, W, k)
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    return ((((A_te - Xmu) / Xs) @ beta) * ysig + ymu).flatten()


def cv_curve(A, y, kmax=MAX_LV):
    ycv, km = V._cv_pred(A, y, kmax)
    rk = np.sqrt(np.mean((y[:, None] - ycv) ** 2, axis=0))
    return rk, ycv, km


def applicability_domain(X_cal, y_cal, X_new, A):
    Xn, Xmu, Xsig = V.zscore_matlab_style(X_cal)
    yn, _, _ = V.zscore_matlab_style(y_cal.reshape(-1, 1))
    Xs = Xsig.copy(); Xs[Xs == 0] = 1.0
    _, _, P, _, _, W, _, _ = V.PLS().nipals(Xn, yn, A)
    P, W = P[:, :A], W[:, :A]
    R = W @ np.linalg.pinv(P.T @ W)
    n = Xn.shape[0]; T_cal = Xn @ R
    s2 = T_cal.var(axis=0, ddof=1); s2[s2 <= 0] = 1e-12
    T2_lim = A * (n - 1) / (n - A) * f_dist.ppf(0.95, A, n - A)
    E = Xn - T_cal @ P.T
    sv = np.linalg.svd(E, compute_uv=False); lam = (sv ** 2) / (n - 1); lam = lam[lam > 1e-12]
    th1, th2, th3 = lam.sum(), (lam ** 2).sum(), (lam ** 3).sum()
    if th1 > 0 and th2 > 0:
        h0 = max(1e-3, 1.0 - (2.0 * th1 * th3) / (3.0 * th2 ** 2)); z = norm.ppf(0.95)
        Q_lim = th1 * ((z * np.sqrt(2.0 * th2 * h0 ** 2)) / th1 + 1.0 + (th2 * h0 * (h0 - 1.0)) / th1 ** 2) ** (1.0 / h0)
    else:
        Q_lim = np.inf
    Xn_new = (X_new - Xmu) / Xs; T_new = Xn_new @ R
    T2 = np.sum((T_new ** 2) / s2, axis=1); Q = np.sum((Xn_new - T_new @ P.T) ** 2, axis=1)
    return dict(T2=T2, Q=Q, T2_lim=float(T2_lim), Q_lim=float(Q_lim), n=n, in_domain=(T2 <= T2_lim) & (Q <= Q_lim))


# ----------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Preprocessing: {[s[0] for s in PRETREAT]}  (raw spectra = after Cut only)")

    refe = {e: load_refe(e) for e in CAL + [TEST]}
    nd = {e: load_nonda(e) for e in CAL + [TEST]}
    spec_cut = {e: nd[e][0] for e in nd}
    spec_pre = {e: nd[e][2] for e in nd}
    wl_cut, wl_pre = nd[CAL[0]][1], nd[CAL[0]][3]
    conc = {e: {"cb": refe[e][1], "glucose": refe[e][2], "xylose": refe[e][3]} for e in refe}
    Ymat = {e: np.column_stack([conc[e]["cb"], conc[e]["glucose"], conc[e]["xylose"]]) for e in refe}

    # ---- hplc.csv ----
    rows = []
    for e in CAL + [TEST]:
        th, _, gl, xy = refe[e]
        rows += [{"time": round(float(t), 4), "glucose": round(float(g), 4),
                  "xylose": round(float(x), 4), "assay": ASSAY_NAME[e]} for t, g, x in zip(th, gl, xy)]
    write_csv(os.path.join(OUTDIR, "hplc.csv"), ["time", "glucose", "xylose", "assay"], rows)
    print(f"hplc.csv: {len(rows)} points")

    # ---- spectra CSVs (raw = after cut; pretreated = cut+SG) ----
    raw_cols, pre_cols = {"wavenumber": wl_cut}, {"wavenumber": wl_pre}
    for e in CAL:
        for i in range(spec_cut[e].shape[0]):
            raw_cols[f"{ASSAY_NAME[e]}_{i+1}"] = spec_cut[e][i]
            pre_cols[f"{ASSAY_NAME[e]}_{i+1}"] = spec_pre[e][i]
    write_matrix_csv(os.path.join(OUTDIR, "spectra_raw_calibration.csv"), raw_cols)
    write_matrix_csv(os.path.join(OUTDIR, "spectra_pretreated_calibration.csv"), pre_cols)
    print(f"spectra CSVs: {len(raw_cols)-1} calibration spectra")

    # ---- parity.csv (full-spectrum PLS LOBO; kept for Fig 2) ----
    A_cal = np.vstack([spec_pre[e] for e in CAL])
    prows = []
    for held in CAL:
        tr = [e for e in CAL if e != held]; A_tr = np.vstack([spec_pre[e] for e in tr])
        for a, j in ANALYTES.items():
            y_tr = np.concatenate([conc[e][a] for e in tr])
            yp = pls_predict(A_tr, y_tr, spec_pre[held], LV_FIXED[a])
            prows += [{"set": "calibration", "analyte": a, "measured": round(float(m), 4),
                       "predicted": round(float(p), 4)} for m, p in zip(conc[held][a], yp)]
    for a, j in ANALYTES.items():
        y_tr = np.concatenate([conc[e][a] for e in CAL])
        yp = pls_predict(A_cal, y_tr, spec_pre[TEST], LV_FIXED[a])
        prows += [{"set": "test", "analyte": a, "measured": round(float(m), 4),
                   "predicted": round(float(p), 4)} for m, p in zip(conc[TEST][a], yp)]
    write_csv(os.path.join(OUTDIR, "parity.csv"), ["set", "analyte", "measured", "predicted"], prows)
    print(f"parity.csv: {sum(r['set']=='calibration' for r in prows)} cal + {sum(r['set']=='test' for r in prows)} test")

    # ---- per-model comparison: PLS / VIP-PLS / PSO-PLS ----
    Y_cal = np.vstack([Ymat[e] for e in CAL])
    A_test = spec_pre[TEST]; Y_test = Ymat[TEST]
    rmsecv_rows, par_rows, sel_rows = [], [], []
    mean_spec = A_cal.mean(0)
    print("\n--- Per-model comparison (cal=Assay 1-3, test=Assay 4) ---")
    for a, j in ANALYTES.items():
        y_cal, y_test = Y_cal[:, j], Y_test[:, j]
        for M in ["PLS", "VIP", "PSO"]:
            if M == "PLS":
                mask = np.ones(A_cal.shape[1], dtype=bool)
            elif M == "VIP":
                mask = V.vip_mask(A_cal, Y_cal, j)
            else:
                mask, _ = V.run_pso(A_cal, Y_cal, MAX_LV, K_FOLDS, CV_TYPE, target_idx=j,
                                    seed=42, verbose=False, **PSO_PARAMS)
            A_sub, A_te_sub = A_cal[:, mask], A_test[:, mask]
            rk, ycv, km = cv_curve(A_sub, y_cal, MAX_LV)
            lv = LV_FIXED[a] if M == "PLS" else int(V.select_k_ftest(rk, n_cal=len(y_cal)))
            lv = int(min(lv, km))
            cvp = ycv[:, lv - 1]
            tp = pls_predict(A_sub, y_cal, A_te_sub, lv)
            rmsecv_rows += [{"model": M, "analyte": a, "lv": kk, "rmsecv": round(float(rk[kk - 1]), 4),
                             "selected": int(kk == lv)} for kk in range(1, km + 1)]
            par_rows += [{"model": M, "analyte": a, "set": "cv", "measured": round(float(m), 4),
                          "predicted": round(float(p), 4)} for m, p in zip(y_cal, cvp)]
            par_rows += [{"model": M, "analyte": a, "set": "test", "measured": round(float(m), 4),
                          "predicted": round(float(p), 4)} for m, p in zip(y_test, tp)]
            if M in ("VIP", "PSO"):
                sel_rows += [{"model": M, "analyte": a, "wavenumber": round(float(w), 4),
                              "selected": int(s), "mean_abs": round(float(ms), 6)}
                             for w, s, ms in zip(wl_pre, mask, mean_spec)]
            rmsep = float(np.sqrt(np.mean((y_test - tp) ** 2)))
            print(f"  {M:3s} {a:8s}: vars={int(mask.sum()):5d}, LV={lv}, RMSECV={rk[lv-1]:.3f}, RMSEP={rmsep:.3f}")
    write_csv(os.path.join(OUTDIR, "model_rmsecv.csv"), ["model", "analyte", "lv", "rmsecv", "selected"], rmsecv_rows)
    write_csv(os.path.join(OUTDIR, "model_parity.csv"), ["model", "analyte", "set", "measured", "predicted"], par_rows)
    write_csv(os.path.join(OUTDIR, "model_selected_vars.csv"),
              ["model", "analyte", "wavenumber", "selected", "mean_abs"], sel_rows)
    print("model_rmsecv.csv, model_parity.csv, model_selected_vars.csv written")

    # ---- nir_exp7.csv (continuous trajectory + AD) ----
    t_inf, A_inf = load_inf()
    pred, ad, ad_rows = {}, {}, []
    print("\n--- Applicability-domain screening of exp7 (continuous) ---")
    for a, j in ANALYTES.items():
        y_tr = np.concatenate([conc[e][a] for e in CAL])
        pred[a] = pls_predict(A_cal, y_tr, A_inf, LV_FIXED[a])
        ad[a] = applicability_domain(A_cal, y_tr, A_inf, LV_FIXED[a])
        ood = ~ad[a]["in_domain"]; n_ood = int(ood.sum()); t_ood = t_inf[ood]
        tmin = float(t_ood.min()) if n_ood else float("nan"); tmax = float(t_ood.max()) if n_ood else float("nan")
        print(f"  {a:8s} (A={LV_FIXED[a]}): T2_lim={ad[a]['T2_lim']:.2f}, Q_lim={ad[a]['Q_lim']:.1f} | "
              f"out-of-domain {n_ood}/{len(t_inf)}")
        ad_rows.append({"analyte": a, "A": LV_FIXED[a], "n_cal": ad[a]["n"],
                        "T2_lim": round(ad[a]["T2_lim"], 4), "Q_lim": round(ad[a]["Q_lim"], 4),
                        "n_out_of_domain": n_ood, "n_total": len(t_inf),
                        "t_ood_min_h": round(tmin, 3), "t_ood_max_h": round(tmax, 3)})
    nir_rows = []
    for i, t in enumerate(t_inf):
        row = {"time": round(float(t), 4)}
        for a in ANALYTES:
            row[f"{a}_pred"] = round(float(pred[a][i]), 4)
            row[f"{a}_indomain"] = int(ad[a]["in_domain"][i])
            row[f"{a}_T2"] = round(float(ad[a]["T2"][i]), 4)
            row[f"{a}_Q"] = round(float(ad[a]["Q"][i]), 4)
        nir_rows.append(row)
    write_csv(os.path.join(OUTDIR, "nir_exp7.csv"),
              ["time"] + [f"{a}_{s}" for a in ANALYTES for s in ("pred", "indomain", "T2", "Q")], nir_rows)
    write_csv(os.path.join(OUTDIR, "ad_limits.csv"),
              ["analyte", "A", "n_cal", "T2_lim", "Q_lim", "n_out_of_domain", "n_total", "t_ood_min_h", "t_ood_max_h"], ad_rows)
    print(f"nir_exp7.csv ({len(nir_rows)} spectra) + ad_limits.csv written")
    print(f"\nDone -> {OUTDIR}/")


if __name__ == "__main__":
    main()
