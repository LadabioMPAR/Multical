"""
snv_ab_test.py — PARALLEL, NON-DESTRUCTIVE A/B/C TEST (N arms)
=============================================================
Tests preprocessing variants for the exp7 out-of-domain blow-up and RMSEP cost.

  baseline : Cut(4400-7500) -> SG(15,2,1)        -> autoscale   (current pipeline)
  snv      : Cut -> SNV -> SG(15,2,1)            -> autoscale   (SNV in addition to SG)
  snv_only : Cut -> SNV                          -> autoscale   (SNV instead of SG; NO derivative)

The ONLY difference between arms is the pretreatment list. Everything else (data, batches,
F-test LV selection alpha=0.05, T2/Q computation, 95% limits) is reused IDENTICALLY from the
production functions (build_paper_data.pls_predict / applicability_domain / cv_curve;
run_loexo_plsvarsel_exp4to7.select_k_ftest). SNV = repo's preprocessing.snv, i.e.
x_snv = (x - mean(x)) / std(x) per spectrum (ddof=1), applied per row to cal AND online.

READ-ONLY toward production: does NOT modify/overwrite any existing script/model/CSV/figure.
ALL outputs go to a NEW isolated folder: results_snv_test/. results/ is never touched.

  1  RMSEP cost   (full LOBO, F-test LV/fold, all arms)   -> results_snv_test/rmsep_ab.csv
  2  exp7 out-of-domain fraction (M_primario)             -> results_snv_test/ad_exp7_ab.csv
  3  exp4 out-of-domain fraction (M_menos1, specificity)  -> results_snv_test/ad_exp4_ab.csv
  4  calibration self-check sanity (all arms)             -> results_snv_test/ad_calibration_ab.csv
  final consolidated markdown                             -> results_snv_test/SNV_TEST_RESUMO.md

Run:  MPLBACKEND=Agg py scripts/plotting/snv_ab_test.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

OUT = "results_snv_test"
DATA = "data"
EXPERIMENTS = ["exp4", "exp5", "exp6", "exp7"]
ASSAY = {"exp4": "Assay 1", "exp5": "Assay 2", "exp6": "Assay 3", "exp7": "Assay 4"}
ANALYTES = ["glucose", "xylose"]
INF = {e: os.path.join(DATA, f"exp_{e[3:].zfill(2)}_inf.txt") for e in EXPERIMENTS}
MAX_LV = 15
XMAX_H = 24.0

# The ONLY thing that differs between arms:
ARMS = {
    "baseline": [["Cut", 4400, 7500, 0], ["SG", 7, 2, 1, 0]],
    "snv":      [["Cut", 4400, 7500, 0], ["SNV", 0], ["SG", 7, 2, 1, 0]],
    "snv_only": [["Cut", 4400, 7500, 0], ["SNV", 0]],
}
ARM_DESC = {
    "baseline": "Cut -> SG -> autoscale",
    "snv":      "Cut -> SNV -> SG -> autoscale",
    "snv_only": "Cut -> SNV -> autoscale (no derivative)",
}
ARMLIST = list(ARMS)
MODELS = {
    "M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7"},   # limits from Assay 1-3
    "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4"},   # limits from Assay 2-4
}

_md = []
def log(s=""):
    print(s); _md.append(s)

def wcsv(name, fieldnames, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- loaders (arm-parametrised)
def load_spec(e, pretreat):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
    return apply_pretreatment(pretreat, raw, wl, plot=False)[0]

def load_online(e, pretreat):
    with open(INF[e]) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(INF[e], skiprows=1))
    t = raw[:, 0] / 60.0
    keep = t <= XMAX_H
    return t[keep], apply_pretreatment(pretreat, raw[keep, 1:], wl, plot=False)[0]


def ftest_glu_lv(A_tr, y_tr):
    """F-test (Osten, alpha=0.05) LV on the training set — identical to production."""
    rk, _, km = B.cv_curve(A_tr, y_tr, MAX_LV)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))


def ad_full(A_tr, y_glu, X_online, t_online, A):
    """Reuse B.applicability_domain exactly; also project cal on itself for the self-check."""
    ad_on = B.applicability_domain(A_tr, y_glu, X_online, A)
    ad_cal = B.applicability_domain(A_tr, y_glu, A_tr, A)
    ql, t2l = ad_on["Q_lim"], ad_on["T2_lim"]
    Q, T2 = ad_on["Q"], ad_on["T2"]
    exc_T2, exc_Q = T2 > t2l, Q > ql
    union, inter = exc_T2 | exc_Q, exc_T2 & exc_Q
    ret_mask = ~union
    Qc, T2c = ad_cal["Q"], ad_cal["T2"]
    exc_T2c, exc_Qc = T2c > t2l, Qc > ql
    return dict(
        A=A, n_online=len(Q), Q_lim=ql, T2_lim=t2l,
        frac_T2=float(exc_T2.mean()), frac_Q=float(exc_Q.mean()),
        frac_union=float(union.mean()), frac_inter=float(inter.mean()),
        retained=int(ret_mask.sum()), surv_t=np.sort(t_online[ret_mask]),
        Q_pcts={p: float(np.percentile(Q, p)) for p in (5, 25, 50, 75, 95, 99)},
        T2_pcts={p: float(np.percentile(T2, p)) for p in (5, 25, 50, 75, 95, 99)},
        Q_med=float(np.median(Q)), Qcal_med=float(np.median(Qc)),
        n_cal=len(Qc), cal_exc_T2=float(exc_T2c.mean()), cal_exc_Q=float(exc_Qc.mean()),
        cal_exc_union=float((exc_T2c | exc_Qc).mean()),
        Qcal_min=float(Qc.min()), Qcal_max=float(Qc.max()),
        t_online=t_online, union_mask=union,
    )


# ==================================================================== main
def main():
    os.makedirs(OUT, exist_ok=True)

    refe = {e: B.load_refe(e) for e in EXPERIMENTS}
    conc = {e: {"glucose": refe[e][2], "xylose": refe[e][3]} for e in EXPERIMENTS}

    spec = {arm: {e: load_spec(e, ARMS[arm]) for e in EXPERIMENTS} for arm in ARMS}
    online = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for m in MODELS.values():
            te = m["test"]
            if te not in online[arm]:
                online[arm][te] = load_online(te, ARMS[arm])

    print("Arms:")
    for arm in ARMLIST:
        print(f"  {arm:9s}: {ARM_DESC[arm]}  ({spec[arm]['exp4'].shape[1]} vars)")
    print(f"Online (<= {XMAX_H:.0f} h): exp7={online['baseline']['exp7'][1].shape[0]}, "
          f"exp4={online['baseline']['exp4'][1].shape[0]}")

    # ================= TEST 1 — RMSEP (LOBO, F-test LV/fold) =================
    print("\n" + "=" * 78 + "\nTEST 1 — RMSEP cost (full LOBO, F-test LV per fold)\n" + "=" * 78)
    rmsep = {arm: {a: {} for a in ANALYTES} for arm in ARMS}
    lv_used = {arm: {a: {} for a in ANALYTES} for arm in ARMS}
    for arm in ARMS:
        for held in EXPERIMENTS:
            train = [e for e in EXPERIMENTS if e != held]
            A_tr = np.vstack([spec[arm][e] for e in train]); A_te = spec[arm][held]
            for a in ANALYTES:
                y_tr = np.concatenate([conc[e][a] for e in train]); y_te = conc[held][a]
                lv = ftest_glu_lv(A_tr, y_tr)
                yp = B.pls_predict(A_tr, y_tr, A_te, lv)
                rmsep[arm][a][held] = float(np.sqrt(np.mean((y_te - yp) ** 2)))
                lv_used[arm][a][held] = lv
    for a in ANALYTES:
        hdr = "  held-out |" + "".join(f" {arm:>16s} |" for arm in ARMLIST)
        print(f"\n[{a}]{hdr}")
        for held in EXPERIMENTS:
            cells = "".join(f" {rmsep[arm][a][held]:6.3f} (LV{lv_used[arm][a][held]}) |" for arm in ARMLIST)
            print(f"  {ASSAY[held]:8s} |{cells}")
        cells = "".join(f" {np.mean([rmsep[arm][a][h] for h in EXPERIMENTS]):.2f}±"
                        f"{np.std([rmsep[arm][a][h] for h in EXPERIMENTS], ddof=1):.2f}      |" for arm in ARMLIST)
        print(f"  mean±SD  |{cells}")
    # csv (arm-generic, delta vs baseline)
    rfields = ["analyte", "held_out"]
    for arm in ARMLIST:
        rfields += [f"{arm}_rmsep", f"{arm}_lv"]
        if arm != "baseline":
            rfields += [f"{arm}_delta"]
    rmsep_rows = []
    for a in ANALYTES:
        for held in list(EXPERIMENTS) + ["MEAN", "SD"]:
            row = {"analyte": a, "held_out": ASSAY.get(held, held)}
            for arm in ARMLIST:
                if held == "MEAN":
                    v = float(np.mean([rmsep[arm][a][h] for h in EXPERIMENTS])); lv = ""
                elif held == "SD":
                    v = float(np.std([rmsep[arm][a][h] for h in EXPERIMENTS], ddof=1)); lv = ""
                else:
                    v = rmsep[arm][a][held]; lv = lv_used[arm][a][held]
                row[f"{arm}_rmsep"] = round(v, 3); row[f"{arm}_lv"] = lv
                if arm != "baseline":
                    if held in ("MEAN", "SD"):
                        bv = float(np.mean([rmsep["baseline"][a][h] for h in EXPERIMENTS])) if held == "MEAN" \
                            else float(np.std([rmsep["baseline"][a][h] for h in EXPERIMENTS], ddof=1))
                    else:
                        bv = rmsep["baseline"][a][held]
                    row[f"{arm}_delta"] = round(v - bv, 3)
            rmsep_rows.append(row)
    wcsv("rmsep_ab.csv", rfields, rmsep_rows)

    # ================= AD per arm per model =================
    ad = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for mname, m in MODELS.items():
            train, te = m["train"], m["test"]
            A_tr = np.vstack([spec[arm][e] for e in train])
            y_glu = np.concatenate([conc[e]["glucose"] for e in train])
            A = ftest_glu_lv(A_tr, y_glu)
            t_on, X_on = online[arm][te]
            ad[arm][mname] = ad_full(A_tr, y_glu, X_on, t_on, A)

    def ad_block(mname, csvname, header):
        print("\n" + "=" * 78 + f"\n{header}\n" + "=" * 78)
        d = {arm: ad[arm][mname] for arm in ARMS}
        print("  " + " " * 24 + "".join(f"{arm:>12s}" for arm in ARMLIST))
        def row(metric, getter, fmt):
            print(f"  {metric:24s}" + "".join(f"{fmt.format(getter(d[arm])):>12s}" for arm in ARMLIST))
        row("AD LV (F-test glu)", lambda x: x["A"], "{:.0f}")
        row("Q_lim", lambda x: x["Q_lim"], "{:.1f}")
        row("T2_lim", lambda x: x["T2_lim"], "{:.2f}")
        row("flag_T2_%", lambda x: 100*x["frac_T2"], "{:.1f}")
        row("flag_Q_%", lambda x: 100*x["frac_Q"], "{:.1f}")
        row("flag_union_%", lambda x: 100*x["frac_union"], "{:.1f}")
        row("flag_inter_%", lambda x: 100*x["frac_inter"], "{:.1f}")
        row("retained_n", lambda x: x["retained"], "{:.0f}")
        for p in (5, 25, 50, 75, 95, 99):
            row(f"Q_p{p}", lambda x, p=p: x["Q_pcts"][p], "{:.0f}")
        row("Q_median_online", lambda x: x["Q_med"], "{:.0f}")
        row("Q_median_cal", lambda x: x["Qcal_med"], "{:.0f}")
        row("ratio_medQ_on/cal", lambda x: x["Q_med"]/x["Qcal_med"], "{:.2f}")
        for arm in ARMLIST:
            dd = d[arm]
            if dd["retained"] > 0:
                st = dd["surv_t"]
                su = bool(np.all(dd["union_mask"][dd["t_online"] < 1.0])) if np.any(dd["t_online"] < 1.0) else None
                print(f"  [{arm}] retained {dd['retained']}/{dd['n_online']}: "
                      f"t in [{st.min():.2f}, {st.max():.2f}] h; startup(t<1h) all flagged: {su}")
            else:
                print(f"  [{arm}] retained 0/{dd['n_online']}")
        # csv
        fields = ["metric"] + ARMLIST
        metrics = [("AD_LV", lambda x: x["A"]), ("Q_lim", lambda x: x["Q_lim"]),
                   ("T2_lim", lambda x: x["T2_lim"]), ("flag_T2_%", lambda x: 100*x["frac_T2"]),
                   ("flag_Q_%", lambda x: 100*x["frac_Q"]), ("flag_union_%", lambda x: 100*x["frac_union"]),
                   ("flag_inter_%", lambda x: 100*x["frac_inter"]), ("retained_n", lambda x: x["retained"])]
        metrics += [(f"Q_p{p}", (lambda x, p=p: x["Q_pcts"][p])) for p in (5, 25, 50, 75, 95, 99)]
        metrics += [("Q_median_online", lambda x: x["Q_med"]), ("Q_median_cal", lambda x: x["Qcal_med"]),
                    ("ratio_medQ_on_cal", lambda x: x["Q_med"]/x["Qcal_med"])]
        rows = [dict({"metric": mkey}, **{arm: round(float(g(d[arm])), 4) for arm in ARMLIST}) for mkey, g in metrics]
        wcsv(csvname, fields, rows)
        return d

    d7 = ad_block("M_primario", "ad_exp7_ab.csv",
                  "TEST 2 — exp7 (Assay 4) out-of-domain, M_primario")
    d4 = ad_block("M_menos1", "ad_exp4_ab.csv",
                  "TEST 3 — exp4 (Assay 1) out-of-domain, M_menos1 (specificity control)")

    # ================= TEST 4 — calibration self-check =================
    print("\n" + "=" * 78 + "\nTEST 4 — calibration self-check (limit construction sanity)\n" + "=" * 78)
    cal_rows = []
    for mname in MODELS:
        d = {arm: ad[arm][mname] for arm in ARMS}
        print(f"\n[{mname}]  (n_cal={d['baseline']['n_cal']})" + " " * 6 + "".join(f"{arm:>12s}" for arm in ARMLIST))
        for metric, g, fmt in [
            ("cal_exceed_T2_%", lambda x: 100*x["cal_exc_T2"], "{:.1f}"),
            ("cal_exceed_Q_%", lambda x: 100*x["cal_exc_Q"], "{:.1f}"),
            ("cal_exceed_union_%", lambda x: 100*x["cal_exc_union"], "{:.1f}"),
            ("Qcal_min", lambda x: x["Qcal_min"], "{:.0f}"),
            ("Qcal_median", lambda x: x["Qcal_med"], "{:.0f}"),
            ("Qcal_max", lambda x: x["Qcal_max"], "{:.0f}"),
        ]:
            print(f"  {metric:22s}" + "".join(f"{fmt.format(g(d[arm])):>12s}" for arm in ARMLIST))
            cal_rows.append(dict({"model": mname, "metric": metric},
                                 **{arm: round(float(g(d[arm])), 3) for arm in ARMLIST}))
    wcsv("ad_calibration_ab.csv", ["model", "metric"] + ARMLIST, cal_rows)

    # ================= VERDICT =================
    print("\n" + "#" * 78 + "\nVEREDITO (por braço, vs baseline)\n" + "#" * 78)
    for arm in ARMLIST:
        r = rmsep[arm]["glucose"]["exp7"]; dr = r - rmsep["baseline"]["glucose"]["exp7"]
        u = 100*d7[arm]["frac_union"]; ret = d7[arm]["retained"]
        print(f"  {arm:9s}: RMSEP exp7 glu={r:.2f} (Δ{dr:+.2f}) | exp7 união={u:.0f}% | retidos={ret}/{d7[arm]['n_online']}")

    # ---------------- markdown ----------------
    _md.clear()
    _md.append("# TESTE SNV — RESUMO (3 braços)\n")
    _md.append("Braços — única diferença é o pré-processamento; tudo o mais (dados, bateladas, LV por teste-F "
               "α=0.05, T²/Q, limites 95%) reutilizado dos scripts de produção. SNV = `(x-média_linha)/desvio_linha` "
               "(ddof=1) por espectro, cal e online. Nada em `results/` ou no pipeline foi alterado.\n")
    for arm in ARMLIST:
        _md.append(f"- **{arm}**: {ARM_DESC[arm]}")
    _md.append("")
    _md.append("## Veredito por braço (exp7, M_primario)\n")
    _md.append("| Braço | RMSEP exp7 glu (Δ vs base) | exp7 união 95% | retidos/481 |")
    _md.append("|---|---|---|---|")
    for arm in ARMLIST:
        r = rmsep[arm]["glucose"]["exp7"]; dr = r - rmsep["baseline"]["glucose"]["exp7"]
        _md.append(f"| {arm} | {r:.2f} ({dr:+.2f}) | {100*d7[arm]['frac_union']:.0f}% | {d7[arm]['retained']} |")
    _md.append("")

    _md.append("## Teste 1 — RMSEP (LOBO, LV por teste-F)\n")
    _md.append("| Analito | Held-out | " + " | ".join(ARMLIST) + " |")
    _md.append("|" + "---|" * (2 + len(ARMLIST)))
    for a in ANALYTES:
        for held in EXPERIMENTS:
            cells = " | ".join(f"{rmsep[arm][a][held]:.3f} (LV{lv_used[arm][a][held]})" for arm in ARMLIST)
            _md.append(f"| {a} | {ASSAY[held]} | {cells} |")
        cells = " | ".join(f"{np.mean([rmsep[arm][a][h] for h in EXPERIMENTS]):.3f} ± "
                           f"{np.std([rmsep[arm][a][h] for h in EXPERIMENTS], ddof=1):.3f}" for arm in ARMLIST)
        _md.append(f"| {a} | **mean±SD** | {cells} |")
    _md.append("")

    def md_ad(title, d):
        _md.append(f"## {title}\n")
        _md.append("| Métrica | " + " | ".join(ARMLIST) + " |")
        _md.append("|" + "---|" * (1 + len(ARMLIST)))
        def mrow(metric, g, fmt):
            _md.append(f"| {metric} | " + " | ".join(fmt.format(g(d[arm])) for arm in ARMLIST) + " |")
        mrow("AD LV (teste-F)", lambda x: x["A"], "{:.0f}")
        mrow("Q_lim", lambda x: x["Q_lim"], "{:.0f}")
        mrow("Reprov. T² (%)", lambda x: 100*x["frac_T2"], "{:.1f}")
        mrow("Reprov. Q (%)", lambda x: 100*x["frac_Q"], "{:.1f}")
        mrow("**Reprov. união (%)**", lambda x: 100*x["frac_union"], "{:.1f}")
        mrow("**Retidos (n)**", lambda x: x["retained"], "{:.0f}")
        for p in (25, 50, 75, 95, 99):
            mrow(f"Q p{p}", lambda x, p=p: x["Q_pcts"][p], "{:.0f}")
        mrow("razão med(Q_on)/med(Q_cal)", lambda x: x["Q_med"]/x["Qcal_med"], "{:.2f}")
        for arm in ARMLIST:
            dd = d[arm]
            if dd["retained"] > 0:
                st = dd["surv_t"]
                su = bool(np.all(dd["union_mask"][dd["t_online"] < 1.0])) if np.any(dd["t_online"] < 1.0) else None
                _md.append(f"\n{arm}: retidos em t∈[{st.min():.2f}, {st.max():.2f}] h; arranque(t<1h) todo reprovado: {su}.")
            else:
                _md.append(f"\n{arm}: 0 retidos.")
        _md.append("")

    md_ad("Teste 2 — exp7 (Assay 4), M_primario — o benefício", d7)
    md_ad("Teste 3 — exp4 (Assay 1), M_menos1 — especificidade (controle)", d4)

    _md.append("## Teste 4 — sanidade na calibração (limite não quebrou)\n")
    _md.append("| Modelo | Métrica | " + " | ".join(ARMLIST) + " |")
    _md.append("|" + "---|" * (2 + len(ARMLIST)))
    for mname in MODELS:
        d = {arm: ad[arm][mname] for arm in ARMS}
        for metric, g, fmt in [("excede união (%)", lambda x: 100*x["cal_exc_union"], "{:.1f}"),
                               ("Q_cal med", lambda x: x["Qcal_med"], "{:.0f}"),
                               ("Q_cal max", lambda x: x["Qcal_max"], "{:.0f}")]:
            _md.append(f"| {mname} | {metric} | " + " | ".join(fmt.format(g(d[arm])) for arm in ARMLIST) + " |")
    _md.append("")

    with open(os.path.join(OUT, "SNV_TEST_RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(_md))
    print(f"\nSaved -> {OUT}/rmsep_ab.csv, ad_exp7_ab.csv, ad_exp4_ab.csv, ad_calibration_ab.csv, SNV_TEST_RESUMO.md")


if __name__ == "__main__":
    main()
