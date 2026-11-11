"""
cut_band_test.py — PARALLEL, NON-DESTRUCTIVE test of excluding the water band
============================================================================
LOBO PLS with the current window vs the same window minus the O-H water combination
band (4810-5320 cm^-1). Savitzky-Golay 1st derivative (window 15, poly order 2) kept
IDENTICAL. Two ways to exclude the band, to separate the real effect from the SG seam
artifact that appears when the derivative is run on a discontinuous wavenumber axis:

  full          : Cut(4400-7500)                         -> SG(15,2,1)       -> autoscale
  excl_water     : Cut(4400-4810, 5320-7500) -> SG(15,2,1)                   -> autoscale
                   (SG on a DISCONTINUOUS axis -> ~14 seam variables distorted; prints
                    "x must be linearly spaced")
  excl_sgfirst  : Cut(4400-7500) -> SG(15,2,1) -> drop 4810-5320            -> autoscale
                   (SG on the CONTIGUOUS axis first, THEN drop columns -> no seam artifact)

Everything else (data, batches, F-test LV alpha=0.05, T2/Q, 95% limits) reused IDENTICALLY
from production (build_paper_data / run_loexo_plsvarsel_exp4to7). READ-ONLY toward production.
ALL outputs -> results_cutband_test/. results/ and results_snv_test/ untouched.

  1  RMSEP (full LOBO, F-test LV)    -> results_cutband_test/rmsep_ab.csv   [the ask]
  2  exp7 out-of-domain (M_primario) -> results_cutband_test/ad_exp7_ab.csv
  3  exp4 out-of-domain (M_menos1)   -> results_cutband_test/ad_exp4_ab.csv
  4  calibration self-check          -> results_cutband_test/ad_calibration_ab.csv
  markdown                           -> results_cutband_test/CUTBAND_TEST_RESUMO.md

Run:  MPLBACKEND=Agg py scripts/plotting/cut_band_test.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

OUT = "results_cutband_test"
DATA = "data"
EXPERIMENTS = ["exp4", "exp5", "exp6", "exp7"]
ASSAY = {"exp4": "Assay 1", "exp5": "Assay 2", "exp6": "Assay 3", "exp7": "Assay 4"}
ANALYTES = ["glucose", "xylose"]
INF = {e: os.path.join(DATA, f"exp_{e[3:].zfill(2)}_inf.txt") for e in EXPERIMENTS}
MAX_LV = 15
XMAX_H = 24.0
BASE = "full"

ARMS = {
    "full":         [["Cut", 4400, 7500, 0], ["SG", 7, 2, 1, 0]],
    "excl_water":   [["Cut", 4400, 4810, 5320, 7500, 0], ["SG", 7, 2, 1, 0]],
    "excl_sgfirst": [["Cut", 4400, 7500, 0], ["SG", 7, 2, 1, 0], ["Cut", 4400, 4810, 5320, 7500, 0]],
}
ARM_DESC = {
    "full":         "Cut 4400-7500 -> SG",
    "excl_water":   "Cut 4400-4810+5320-7500 -> SG (SG on discontinuous axis; seam artifact)",
    "excl_sgfirst": "Cut 4400-7500 -> SG -> drop 4810-5320 (SG contiguous first; clean)",
}
ARMLIST = list(ARMS)
MODELS = {
    "M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7"},
    "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4"},
}


def wcsv(name, fieldnames, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

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
    rk, _, km = B.cv_curve(A_tr, y_tr, MAX_LV)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))

def ad_full(A_tr, y_glu, X_online, t_online, A):
    ad_on = B.applicability_domain(A_tr, y_glu, X_online, A)
    ad_cal = B.applicability_domain(A_tr, y_glu, A_tr, A)
    ql, t2l = ad_on["Q_lim"], ad_on["T2_lim"]
    Q, T2 = ad_on["Q"], ad_on["T2"]
    exc_T2, exc_Q = T2 > t2l, Q > ql
    union = exc_T2 | exc_Q
    Qc, T2c = ad_cal["Q"], ad_cal["T2"]
    exc_T2c, exc_Qc = T2c > t2l, Qc > ql
    return dict(
        A=A, n_online=len(Q), Q_lim=ql, T2_lim=t2l,
        frac_T2=float(exc_T2.mean()), frac_Q=float(exc_Q.mean()),
        frac_union=float(union.mean()), frac_inter=float((exc_T2 & exc_Q).mean()),
        retained=int((~union).sum()), surv_t=np.sort(t_online[~union]),
        Q_pcts={p: float(np.percentile(Q, p)) for p in (5, 25, 50, 75, 95, 99)},
        Q_med=float(np.median(Q)), Qcal_med=float(np.median(Qc)),
        n_cal=len(Qc), cal_exc_union=float((exc_T2c | exc_Qc).mean()),
        Qcal_min=float(Qc.min()), Qcal_max=float(Qc.max()),
        t_online=t_online, union_mask=union,
    )


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

    print("\nArms:")
    for arm in ARMLIST:
        print(f"  {arm:13s}: {ARM_DESC[arm]}  ({spec[arm]['exp4'].shape[1]} vars)")

    # ===================== TEST 1 — RMSEP (LOBO, F-test LV) =====================
    print("\n" + "=" * 90 + "\nTEST 1 — LOBO PLS RMSEP (F-test LV per fold)  [the ask]\n" + "=" * 90)
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
        print(f"\n[{a}]  held-out |" + "".join(f" {arm:>20s} |" for arm in ARMLIST))
        for held in EXPERIMENTS:
            print(f"  {ASSAY[held]:8s} |" +
                  "".join(f" {rmsep[arm][a][held]:6.3f} (LV{lv_used[arm][a][held]:>2d}) |" for arm in ARMLIST))
        print("  mean±SD  |" + "".join(
            f" {np.mean([rmsep[arm][a][h] for h in EXPERIMENTS]):5.3f}±"
            f"{np.std([rmsep[arm][a][h] for h in EXPERIMENTS], ddof=1):.3f}      |" for arm in ARMLIST))
    rfields = ["analyte", "held_out"]
    for arm in ARMLIST:
        rfields += [f"{arm}_rmsep", f"{arm}_lv"] + ([f"{arm}_delta"] if arm != BASE else [])
    rrows = []
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
                if arm != BASE:
                    if held == "MEAN":
                        bv = float(np.mean([rmsep[BASE][a][h] for h in EXPERIMENTS]))
                    elif held == "SD":
                        bv = float(np.std([rmsep[BASE][a][h] for h in EXPERIMENTS], ddof=1))
                    else:
                        bv = rmsep[BASE][a][held]
                    row[f"{arm}_delta"] = round(v - bv, 3)
            rrows.append(row)
    wcsv("rmsep_ab.csv", rfields, rrows)

    # ===================== AD =====================
    ad = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for mname, m in MODELS.items():
            A_tr = np.vstack([spec[arm][e] for e in m["train"]])
            y_glu = np.concatenate([conc[e]["glucose"] for e in m["train"]])
            A = ftest_glu_lv(A_tr, y_glu)
            t_on, X_on = online[arm][m["test"]]
            ad[arm][mname] = ad_full(A_tr, y_glu, X_on, t_on, A)

    def ad_block(mname, csvname, header):
        print("\n" + "=" * 90 + f"\n{header}\n" + "=" * 90)
        d = {arm: ad[arm][mname] for arm in ARMS}
        print("  " + " " * 20 + "".join(f"{arm:>16s}" for arm in ARMLIST))
        def row(metric, g, fmt):
            print(f"  {metric:20s}" + "".join(f"{fmt.format(g(d[arm])):>16s}" for arm in ARMLIST))
        row("AD LV (F-test glu)", lambda x: x["A"], "{:.0f}")
        row("Q_lim", lambda x: x["Q_lim"], "{:.1f}")
        row("flag_T2_%", lambda x: 100*x["frac_T2"], "{:.1f}")
        row("flag_Q_%", lambda x: 100*x["frac_Q"], "{:.1f}")
        row("flag_union_%", lambda x: 100*x["frac_union"], "{:.1f}")
        row("retained_n", lambda x: x["retained"], "{:.0f}")
        for p in (50, 95, 99):
            row(f"Q_p{p}", lambda x, p=p: x["Q_pcts"][p], "{:.0f}")
        row("ratio_medQ_on/cal", lambda x: x["Q_med"]/x["Qcal_med"], "{:.2f}")
        for arm in ARMLIST:
            dd = d[arm]
            if dd["retained"] > 0:
                st = dd["surv_t"]
                su = bool(np.all(dd["union_mask"][dd["t_online"] < 1.0])) if np.any(dd["t_online"] < 1.0) else None
                print(f"  [{arm}] retained {dd['retained']}/{dd['n_online']}: t in [{st.min():.2f},{st.max():.2f}]h; "
                      f"startup(t<1h) all flagged: {su}")
            else:
                print(f"  [{arm}] retained 0/{dd['n_online']}")
        metrics = [("AD_LV", lambda x: x["A"]), ("Q_lim", lambda x: x["Q_lim"]),
                   ("flag_T2_%", lambda x: 100*x["frac_T2"]), ("flag_Q_%", lambda x: 100*x["frac_Q"]),
                   ("flag_union_%", lambda x: 100*x["frac_union"]), ("retained_n", lambda x: x["retained"])]
        metrics += [(f"Q_p{p}", (lambda x, p=p: x["Q_pcts"][p])) for p in (50, 95, 99)]
        metrics += [("ratio_medQ_on_cal", lambda x: x["Q_med"]/x["Qcal_med"])]
        wcsv(csvname, ["metric"] + ARMLIST,
             [dict({"metric": mk}, **{arm: round(float(g(d[arm])), 4) for arm in ARMLIST}) for mk, g in metrics])
        return d

    d7 = ad_block("M_primario", "ad_exp7_ab.csv", "TEST 2 — exp7 (Assay 4), M_primario")
    d4 = ad_block("M_menos1", "ad_exp4_ab.csv", "TEST 3 — exp4 (Assay 1), M_menos1 specificity")

    # ===================== TEST 4 =====================
    print("\n" + "=" * 90 + "\nTEST 4 — calibration self-check\n" + "=" * 90)
    cal_rows = []
    for mname in MODELS:
        d = {arm: ad[arm][mname] for arm in ARMS}
        print(f"\n[{mname}]  (n_cal={d[BASE]['n_cal']})" + " " * 2 + "".join(f"{arm:>16s}" for arm in ARMLIST))
        for metric, g, fmt in [("cal_exceed_union_%", lambda x: 100*x["cal_exc_union"], "{:.1f}"),
                               ("Qcal_median", lambda x: x["Qcal_med"], "{:.0f}"),
                               ("Qcal_max", lambda x: x["Qcal_max"], "{:.0f}")]:
            print(f"  {metric:18s}" + "".join(f"{fmt.format(g(d[arm])):>16s}" for arm in ARMLIST))
            cal_rows.append(dict({"model": mname, "metric": metric},
                                 **{arm: round(float(g(d[arm])), 3) for arm in ARMLIST}))
    wcsv("ad_calibration_ab.csv", ["model", "metric"] + ARMLIST, cal_rows)

    # ===================== SUMMARY =====================
    print("\n" + "#" * 90 + "\nRESUMO\n" + "#" * 90)
    for a in ANALYTES:
        cells = "  ".join(f"{arm}={np.mean([rmsep[arm][a][h] for h in EXPERIMENTS]):.2f}" for arm in ARMLIST)
        print(f"  RMSEP mean {a:8s}: {cells}")
    print(f"  exp7 glu RMSEP: " + "  ".join(f"{arm}={rmsep[arm]['glucose']['exp7']:.2f}" for arm in ARMLIST))
    print(f"  exp7 uniao 95%: " + "  ".join(f"{arm}={100*d7[arm]['frac_union']:.0f}%(ret {d7[arm]['retained']})" for arm in ARMLIST))

    # ---------- markdown ----------
    md = ["# TESTE CORTE (excluir banda da água 4810-5320) — RESUMO\n"]
    md.append("Único ajuste: a faixa espectral. SG (janela 15, ordem 2, 1ª derivada) mantido idêntico. "
              "Dados/bateladas/LV-por-teste-F(α=0.05)/T²/Q/limites reutilizados dos scripts de produção. "
              "Nada em `results/` alterado; saídas em `results_cutband_test/`.\n")
    md.append("**Aviso técnico:** no braço `excl_water` o SG roda sobre um eixo de números de onda "
              "**descontínuo** (buraco em 4810–5320), então ~14 variáveis na emenda ficam com derivada "
              "distorcida (imprime `x must be linearly spaced`). O braço `excl_sgfirst` aplica o SG no "
              "espectro contíguo 4400–7500 e **depois** descarta a banda — sem artefato de costura. "
              "Compare os dois para separar o efeito real da banda do artefato numérico.\n")
    for arm in ARMLIST:
        md.append(f"- **{arm}**: {ARM_DESC[arm]} ({spec[arm]['exp4'].shape[1]} vars)")
    md.append("")
    md.append("## Teste 1 — RMSEP LOBO (LV por teste-F) — o pedido\n")
    md.append("| Analito | Held-out | " + " | ".join(ARMLIST) + " |")
    md.append("|" + "---|" * (2 + len(ARMLIST)))
    for a in ANALYTES:
        for held in EXPERIMENTS:
            md.append(f"| {a} | {ASSAY[held]} | " +
                      " | ".join(f"{rmsep[arm][a][held]:.3f} (LV{lv_used[arm][a][held]})" for arm in ARMLIST) + " |")
        md.append(f"| {a} | **mean±SD** | " +
                  " | ".join(f"{np.mean([rmsep[arm][a][h] for h in EXPERIMENTS]):.3f} ± "
                            f"{np.std([rmsep[arm][a][h] for h in EXPERIMENTS], ddof=1):.3f}" for arm in ARMLIST) + " |")
    md.append("")
    def md_ad(title, d):
        md.append(f"## {title}\n")
        md.append("| Métrica | " + " | ".join(ARMLIST) + " |")
        md.append("|" + "---|" * (1 + len(ARMLIST)))
        def mrow(metric, g, fmt):
            md.append(f"| {metric} | " + " | ".join(fmt.format(g(d[arm])) for arm in ARMLIST) + " |")
        mrow("AD LV (teste-F)", lambda x: x["A"], "{:.0f}")
        mrow("Q_lim", lambda x: x["Q_lim"], "{:.0f}")
        mrow("Reprov. T² (%)", lambda x: 100*x["frac_T2"], "{:.1f}")
        mrow("Reprov. Q (%)", lambda x: 100*x["frac_Q"], "{:.1f}")
        mrow("**Reprov. união (%)**", lambda x: 100*x["frac_union"], "{:.1f}")
        mrow("**Retidos (n)**", lambda x: x["retained"], "{:.0f}")
        for p in (50, 95, 99):
            mrow(f"Q p{p}", lambda x, p=p: x["Q_pcts"][p], "{:.0f}")
        md.append("")
    md_ad("Teste 2 — exp7 (Assay 4), M_primario", d7)
    md_ad("Teste 3 — exp4 (Assay 1), M_menos1 — especificidade", d4)
    md.append("## Teste 4 — sanidade na calibração\n")
    md.append("| Modelo | Métrica | " + " | ".join(ARMLIST) + " |")
    md.append("|" + "---|" * (2 + len(ARMLIST)))
    for mname in MODELS:
        d = {arm: ad[arm][mname] for arm in ARMS}
        md.append(f"| {mname} | excede união (%) | " +
                  " | ".join(f"{100*d[arm]['cal_exc_union']:.1f}" for arm in ARMLIST) + " |")
        md.append(f"| {mname} | Q_cal med/max | " +
                  " | ".join(f"{d[arm]['Qcal_med']:.0f}/{d[arm]['Qcal_max']:.0f}" for arm in ARMLIST) + " |")
    md.append("")
    with open(os.path.join(OUT, "CUTBAND_TEST_RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> {OUT}/rmsep_ab.csv, ad_exp7_ab.csv, ad_exp4_ab.csv, ad_calibration_ab.csv, CUTBAND_TEST_RESUMO.md")


if __name__ == "__main__":
    main()
