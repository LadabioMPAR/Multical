"""
refe_vs_inf_test.py — REFERENCE spectra vs ONLINE stream under the AD filter (NON-DESTRUCTIVE)
=============================================================================================
Central question: are the exp7 REFERENCE spectra (HPLC-paired) inside or outside the
M_primario domain?  -> FORA = regime/load extrapolation (15 g vs 20 g).  DENTRO (only _inf
outside) = acquisition difference (reference vs continuous online), not load.

TERMINOLOGY: in this repo `*_refe.txt` are HPLC CONCENTRATIONS (time,cb,gl,xy), NOT spectra.
The reference SPECTRA paired with HPLC are `*_nonda.txt`. So "reference spectra" = nonda
(exp7 -> 16, exp4 -> 13). This is exactly the set that gave "8/13 inside" for exp4 earlier.

Same production pipeline (Cut 4400-7500 -> SG(15,2,1) -> autoscale cal mean/scale) and the same
T2/Q (build_paper_data.applicability_domain), LV by Osten F-test (alpha=0.05), glucose.
ANTI-CIRCULAR: M_primario (train Assay 1-3) tests ONLY exp7; M_menos1 (train Assay 2-4) tests
ONLY exp4. Limits 95% from training only. Q-limit: Box chi^2 (principal) + Jackson-Mudholkar
(parallel). T2: F-distribution.

ALL outputs -> results_refe_vs_inf/ only. Nothing in results/ or the pipeline is touched.
Run:  MPLBACKEND=Agg py scripts/plotting/refe_vs_inf_test.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import chi2
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

OUT, DATA = "results_refe_vs_inf", "data"
MODELS = {"M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7", "tmax_h": 24.0},
          "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4", "tmax_h": 12.0}}
STREAMS = {"exp7": {"_inf": "exp_07_inf.txt", "_inf_smoothed": "exp_07_inf_smoothed.txt"},
           "exp4": {"_inf": "exp_04_inf.txt", "_inf_smoothed": "exp_04_inf_smoothed.txt"}}
PCTS = [5, 25, 50, 75, 95, 99]


def wcsv(name, fieldnames, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def load_nonda_labeled(e):
    """Reference spectra (HPLC-paired) with per-sample time (hours)."""
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    arr = np.atleast_2d(np.loadtxt(path, skiprows=1))
    return apply_pretreatment(B.PRETREAT, arr[:, 1:], wl, plot=False)[0], arr[:, 0] / 60.0

def load_online_labeled(fname, tmax_h):
    path = os.path.join(DATA, fname)
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    arr = np.atleast_2d(np.loadtxt(path, skiprows=1))
    t = arr[:, 0] / 60.0; keep = t <= tmax_h
    return apply_pretreatment(B.PRETREAT, arr[keep, 1:], wl, plot=False)[0], t[keep]

def ftest_lv(A_tr, y_tr):
    rk, _, km = B.cv_curve(A_tr, y_tr, 15)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))

def box_qlim(Qcal, alpha=0.95):
    m = float(np.mean(Qcal)); v = float(np.var(Qcal, ddof=1)) or 1e-10
    return (v / (2 * m)) * chi2.ppf(alpha, 2 * m * m / v)

def model_context(mname):
    m = MODELS[mname]
    A_tr = np.vstack([load_nonda_labeled(e)[0] for e in m["train"]])
    y_glu = np.concatenate([B.load_refe(e)[2] for e in m["train"]])
    A = ftest_lv(A_tr, y_glu)
    adc = B.applicability_domain(A_tr, y_glu, A_tr, A)
    return dict(A_tr=A_tr, y_glu=y_glu, A=A, n_cal=adc["n"], T2_lim=adc["T2_lim"],
                Q_lim_JM=adc["Q_lim"], Q_lim_Box=box_qlim(adc["Q"]),
                T2_cal=adc["T2"], Q_cal=adc["Q"], Q_cal_med=float(np.median(adc["Q"])))

def project(ctx, X):
    ad = B.applicability_domain(ctx["A_tr"], ctx["y_glu"], X, ctx["A"])
    return ad["T2"], ad["Q"]

def flags(T2, Q, t2lim, qlim):
    eT2, eQ = T2 > t2lim, Q > qlim
    return eT2, eQ, (eT2 | eQ)

def dominance(eT2, eQ, un):
    nT2, nQ = int((un & eT2).sum()), int((un & eQ).sum())
    return "T2" if nT2 > nQ else ("Q" if nQ > nT2 else "tie"), nT2, nQ


def main():
    os.makedirs(OUT, exist_ok=True)
    ctxP, ctxM = model_context("M_primario"), model_context("M_menos1")
    print(f"M_primario: n_cal={ctxP['n_cal']}, LV={ctxP['A']}, T2_lim={ctxP['T2_lim']:.2f}, "
          f"Q_lim Box={ctxP['Q_lim_Box']:.0f} / JM={ctxP['Q_lim_JM']:.0f}")
    print(f"M_menos1:   n_cal={ctxM['n_cal']}, LV={ctxM['A']}, T2_lim={ctxM['T2_lim']:.2f}, "
          f"Q_lim Box={ctxM['Q_lim_Box']:.0f} / JM={ctxM['Q_lim_JM']:.0f}")

    # load all streams
    S7 = {"_refe": load_nonda_labeled("exp7")}
    S7["_inf"] = load_online_labeled(STREAMS["exp7"]["_inf"], 24.0)
    S7["_inf_smoothed"] = load_online_labeled(STREAMS["exp7"]["_inf_smoothed"], 24.0)
    S4 = {"_refe": load_nonda_labeled("exp4")}
    S4["_inf"] = load_online_labeled(STREAMS["exp4"]["_inf"], 12.0)
    S4["_inf_smoothed"] = load_online_labeled(STREAMS["exp4"]["_inf_smoothed"], 12.0)

    # ================= TASK 1 — exp7 _refe (decisive) =================
    X7r, t7r = S7["_refe"]; T2, Q = project(ctxP, X7r)
    eT2_B, eQ_B, un_B = flags(T2, Q, ctxP["T2_lim"], ctxP["Q_lim_Box"])
    eT2_J, eQ_J, un_J = flags(T2, Q, ctxP["T2_lim"], ctxP["Q_lim_JM"])
    inB = int((~un_B).sum())
    dom_B, nT2, nQ = dominance(eT2_B, eQ_B, un_B)
    print("\n" + "=" * 90 + "\nTASK 1 — exp7 REFERENCE spectra (nonda) under M_primario\n" + "=" * 90)
    print(f"  n={len(T2)}; DENTRO(uniao Box)={inB}/{len(T2)} ({100*inB/len(T2):.0f}%); "
          f"DENTRO(uniao JM)={int((~un_J).sum())}/{len(T2)}")
    print(f"  Q median={np.median(Q):.0f}; ratio med(Q_refe)/med(Q_cal)={np.median(Q)/ctxP['Q_cal_med']:.2f}; "
          f"dominance(Box)={dom_B} (T2={nT2}, Q={nQ})")
    print(f"  fora (Box): " + ", ".join(f"t={t7r[i]:.1f}h" for i in np.where(un_B)[0]) or "  (nenhum)")
    t1 = [{"idx": i + 1, "t_h": round(float(t7r[i]), 3), "T2": round(float(T2[i]), 3), "Q": round(float(Q[i]), 1),
           "exceeds_T2_F": int(eT2_B[i]), "exceeds_Q_Box": int(eQ_B[i]), "exceeds_Q_JM": int(eQ_J[i]),
           "in_domain_Box": int(~un_B[i]), "in_domain_JM": int(~un_J[i])} for i in range(len(T2))]
    wcsv("exp7_refe_ad.csv", list(t1[0].keys()), t1)

    # ================= helper: stream summary rows =================
    def summ(ctx, streams, label):
        rows = []
        for sname, (X, t) in streams.items():
            T2s, Qs = project(ctx, X)
            for lname, qlim in [("Box", ctx["Q_lim_Box"]), ("JM", ctx["Q_lim_JM"])]:
                eT2, eQ, un = flags(T2s, Qs, ctx["T2_lim"], qlim)
                rows.append({"case": label, "stream": sname, "limit": lname, "n": len(T2s),
                             "pct_out_union": round(100 * un.mean(), 1), "pct_out_T2": round(100 * eT2.mean(), 1),
                             "pct_out_Q": round(100 * eQ.mean(), 1), "retained_n": int((~un).sum()),
                             "Q_median": round(float(np.median(Qs)), 1)})
        return rows

    # ================= TASK 2 — exp7 three streams =================
    t2rows = summ(ctxP, S7, "exp7")
    print("\n" + "=" * 90 + "\nTASK 2 — exp7 three streams (M_primario)\n" + "=" * 90)
    print(f"  {'stream':16s}{'lim':5s}{'n':>5s}{'%out_un':>9s}{'%out_T2':>9s}{'%out_Q':>9s}{'retid':>7s}{'Q_med':>9s}")
    for r in t2rows:
        print(f"  {r['stream']:16s}{r['limit']:5s}{r['n']:>5d}{r['pct_out_union']:>9.1f}"
              f"{r['pct_out_T2']:>9.1f}{r['pct_out_Q']:>9.1f}{r['retained_n']:>7d}{r['Q_median']:>9.0f}")
    wcsv("exp7_three_streams.csv", list(t2rows[0].keys()), t2rows)

    # ================= TASK 3 — exp4 three streams (<=12h) + timeband =================
    t3rows = summ(ctxM, S4, "exp4")
    print("\n" + "=" * 90 + "\nTASK 3 — exp4 three streams (M_menos1, <=12h)\n" + "=" * 90)
    for r in t3rows:
        print(f"  {r['stream']:16s}{r['limit']:5s}{r['n']:>5d}{r['pct_out_union']:>9.1f}"
              f"{r['pct_out_T2']:>9.1f}{r['pct_out_Q']:>9.1f}{r['retained_n']:>7d}{r['Q_median']:>9.0f}")
    wcsv("exp4_three_streams.csv", list(t3rows[0].keys()), t3rows)

    BANDS = [("t<1h", 0.0, 1.0), ("1-12h", 1.0, 12.0)]
    tb_rows = []
    print("\n  -- timeband (Box) --")
    for sname, (X, t) in S4.items():
        T2s, Qs = project(ctxM, X)
        eT2, eQ, un = flags(T2s, Qs, ctxM["T2_lim"], ctxM["Q_lim_Box"])
        for bname, lo, hi in BANDS:
            sel = (t >= lo) & (t < hi); n = int(sel.sum())
            if n == 0:
                continue
            nin = int((sel & ~un).sum()); nout = int((sel & un).sum())
            dom, nT2b, nQb = dominance(eT2, eQ, un & sel)
            startups = ", ".join(f"{t[i]:.2f}h" for i in np.where(sel & un)[0]) if bname == "t<1h" else ""
            print(f"    {sname:16s} {bname:6s}: dentro={nin:3d} fora={nout:3d} ({100*nin/n:4.0f}%) dom={dom}"
                  + (f"  arranque_marcado=[{startups}]" if startups else ""))
            tb_rows.append({"stream": sname, "band": bname, "n": n, "n_dentro": nin, "n_fora": nout,
                            "pct_dentro": round(100 * nin / n, 1), "flag_T2": nT2b, "flag_Q": nQb,
                            "dominance": dom, "startup_flagged_times_h": startups})
    wcsv("exp4_timeband.csv", list(tb_rows[0].keys()), tb_rows)

    # ================= TASK 4 — cross comparison =================
    def pct_out_union_box(ctx, X):
        T2s, Qs = project(ctx, X)
        _, _, un = flags(T2s, Qs, ctx["T2_lim"], ctx["Q_lim_Box"])
        return 100 * un.mean()
    cross = {("exp7", s): pct_out_union_box(ctxP, S7[s][0]) for s in ["_refe", "_inf", "_inf_smoothed"]}
    cross.update({("exp4", s): pct_out_union_box(ctxM, S4[s][0]) for s in ["_refe", "_inf", "_inf_smoothed"]})
    print("\n" + "=" * 90 + "\nTASK 4 — cross comparison: % FORA do dominio (uniao 95%, Box)\n" + "=" * 90)
    print(f"  {'':22s}{'_refe':>10s}{'_inf':>10s}{'_inf_smoothed':>16s}")
    print(f"  {'exp4 (M_menos1,<=12h)':22s}{cross[('exp4','_refe')]:>10.1f}{cross[('exp4','_inf')]:>10.1f}{cross[('exp4','_inf_smoothed')]:>16.1f}")
    print(f"  {'exp7 (M_primario)':22s}{cross[('exp7','_refe')]:>10.1f}{cross[('exp7','_inf')]:>10.1f}{cross[('exp7','_inf_smoothed')]:>16.1f}")
    exp7_refe_out = cross[("exp7", "_refe")]; exp4_refe_out = cross[("exp4", "_refe")]
    if exp7_refe_out > 50 and exp4_refe_out < 50:
        verdict = "diferenca de REGIME/CARGA (15 g e um regime espectral novo; exp7 _refe majoritariamente FORA)"
    elif exp7_refe_out < 50 and exp4_refe_out < 50:
        verdict = "diferenca de AQUISICAO (ambos _refe majoritariamente DENTRO; so os _inf ficam fora)"
    else:
        verdict = f"MISTO/inconclusivo (exp7 _refe fora={exp7_refe_out:.0f}%, exp4 _refe fora={exp4_refe_out:.0f}%)"
    print(f"  INTERPRETACAO: {verdict}")
    wcsv("cross_comparison.csv", ["case", "refe_pct_out", "inf_pct_out", "inf_smoothed_pct_out"],
         [{"case": "exp4 (M_menos1,<=12h)", "refe_pct_out": round(cross[("exp4", "_refe")], 1),
           "inf_pct_out": round(cross[("exp4", "_inf")], 1), "inf_smoothed_pct_out": round(cross[("exp4", "_inf_smoothed")], 1)},
          {"case": "exp7 (M_primario)", "refe_pct_out": round(cross[("exp7", "_refe")], 1),
           "inf_pct_out": round(cross[("exp7", "_inf")], 1), "inf_smoothed_pct_out": round(cross[("exp7", "_inf_smoothed")], 1)}])

    # ================= TASK 5 — exp4 retained trajectory =================
    Xsm, tsm = S4["_inf_smoothed"]
    T2s, Qs = project(ctxM, Xsm)
    eT2, eQ, un = flags(T2s, Qs, ctxM["T2_lim"], ctxM["Q_lim_Box"])
    ret = ~un
    # predictions (M_menos1, F-test LV per analyte)
    y_glu_tr = np.concatenate([B.load_refe(e)[2] for e in MODELS["M_menos1"]["train"]])
    y_xyl_tr = np.concatenate([B.load_refe(e)[3] for e in MODELS["M_menos1"]["train"]])
    lv_glu = ftest_lv(ctxM["A_tr"], y_glu_tr); lv_xyl = ftest_lv(ctxM["A_tr"], y_xyl_tr)
    pg = B.pls_predict(ctxM["A_tr"], y_glu_tr, Xsm, lv_glu)
    px = B.pls_predict(ctxM["A_tr"], y_xyl_tr, Xsm, lv_xyl)

    def bma(vals, mask, w=4):
        out = np.full(len(vals), np.nan); idx = np.where(mask)[0]
        for j, i in enumerate(idx):
            lo = max(0, j - w + 1); out[i] = np.mean(vals[idx[lo:j + 1]])
        return out
    pg_ma, px_ma = bma(pg, ret), bma(px, ret)
    fb = np.where(eT2 & eQ, "both", np.where(eT2, "T2", np.where(eQ, "Q", "none")))
    traj = [{"source": "online", "time_h": round(float(tsm[i]), 3),
             "pred_glucose": "" if np.isnan(pg_ma[i]) else round(float(pg_ma[i]), 3),
             "pred_xylose": "" if np.isnan(px_ma[i]) else round(float(px_ma[i]), 3),
             "retained": int(ret[i]), "flagged_by": fb[i]} for i in range(len(tsm))]
    # HPLC 13 points (exp4 refe: time,cb,gl,xy)
    r4 = np.atleast_2d(np.loadtxt(os.path.join(DATA, "exp4_refe.txt")))
    traj += [{"source": "hplc", "time_h": round(float(r4[i, 0] / 60.0), 3),
              "pred_glucose": round(float(r4[i, 2]), 3), "pred_xylose": round(float(r4[i, 3]), 3),
              "retained": "", "flagged_by": ""} for i in range(r4.shape[0])]
    wcsv("exp4_trajectory.csv", ["source", "time_h", "pred_glucose", "pred_xylose", "retained", "flagged_by"], traj)
    print(f"\nTASK 5: exp4 smoothed <=12h retained={int(ret.sum())}/{len(ret)} (glu LV={lv_glu}, xyl LV={lv_xyl}); "
          f"+{r4.shape[0]} HPLC pts -> exp4_trajectory.csv")

    # ================= TASK 6 — sanity =================
    san = []
    print("\n" + "=" * 90 + "\nTASK 6 — calibration sanity\n" + "=" * 90)
    for mname, ctx in [("M_primario", ctxP), ("M_menos1", ctxM)]:
        for lname, qlim in [("Box", ctx["Q_lim_Box"]), ("JM", ctx["Q_lim_JM"])]:
            eT2, eQ, un = flags(ctx["T2_cal"], ctx["Q_cal"], ctx["T2_lim"], qlim)
            print(f"  {mname:11s} {lname:4s}: T2={100*eT2.mean():4.1f}% Q={100*eQ.mean():4.1f}% "
                  f"uniao={100*un.mean():4.1f}% | LV={ctx['A']} T2_lim={ctx['T2_lim']:.2f} Q_lim={qlim:.0f} "
                  f"Q_cal med={np.median(ctx['Q_cal']):.0f}")
            san.append({"model": mname, "limit": lname, "n_cal": ctx["n_cal"], "LV": ctx["A"],
                        "T2_lim": round(ctx["T2_lim"], 3), "Q_lim": round(qlim, 1),
                        "pct_exceed_T2": round(100 * float(eT2.mean()), 1),
                        "pct_exceed_Q": round(100 * float(eQ.mean()), 1),
                        "pct_exceed_union": round(100 * float(un.mean()), 1),
                        "Q_cal_median": round(float(np.median(ctx["Q_cal"])), 1)})
    wcsv("calibration_sanity.csv", list(san[0].keys()), san)

    # ================= VERDICT + MARKDOWN =================
    v1 = f"exp7 _refe (M_primario, Box, 95%): {inB}/{len(T2)} DENTRO do dominio ({100*inB/len(T2):.0f}%)"
    inB4 = int((~flags(*project(ctxM, S4['_refe'][0]), ctxM['T2_lim'], ctxM['Q_lim_Box'])[2])
               ) if False else None
    T2_4r, Q_4r = project(ctxM, S4["_refe"][0])
    _, _, un4r = flags(T2_4r, Q_4r, ctxM["T2_lim"], ctxM["Q_lim_Box"]); inB4 = int((~un4r).sum())
    v2 = f"exp4 _refe (M_menos1, Box, 95%): {inB4}/{len(T2_4r)} DENTRO do dominio ({100*inB4/len(T2_4r):.0f}%)"
    v3 = f"VEREDITO: {verdict}"
    print("\n" + "#" * 90 + "\n  " + v1 + "\n  " + v2 + "\n  " + v3 + "\n" + "#" * 90)

    md = ["# REFE vs INF — RESUMO\n",
          f"**1) {v1}**  ", f"**2) {v2}**  ", f"**3) {v3}**\n",
          "Terminologia: “espectros de referência” = `*_nonda` (NIR pareados com HPLC); `*_refe` são as "
          "concentrações HPLC. Pré-processamento e T²/Q de produção; LV por teste-F (glicose); limites 95% "
          "só do treino (anti-circular); exp4 truncada em ≤12 h nos streams online. Q-limite: Box χ² (principal) "
          "e Jackson-Mudholkar (paralelo). Nada de produção alterado.\n",
          f"Limites — M_primario: LV={ctxP['A']}, T²_lim={ctxP['T2_lim']:.2f}, Q_lim Box={ctxP['Q_lim_Box']:.0f} / "
          f"JM={ctxP['Q_lim_JM']:.0f}. M_menos1: LV={ctxM['A']}, T²_lim={ctxM['T2_lim']:.2f}, "
          f"Q_lim Box={ctxM['Q_lim_Box']:.0f} / JM={ctxM['Q_lim_JM']:.0f}.\n"]

    md.append("## Tarefa 1 — exp7 _refe (nonda, 16) sob M_primario\n")
    md.append(f"DENTRO (Box): **{inB}/{len(T2)} ({100*inB/len(T2):.0f}%)**; Q mediana={np.median(Q):.0f} "
              f"(razão vs cal={np.median(Q)/ctxP['Q_cal_med']:.2f}); dominância={dom_B}.\n")
    md.append("| t (h) | T² | Q | exc.T² | exc.Q(Box) | dentro(Box) |")
    md.append("|---|---|---|---|---|---|")
    for r in t1:
        md.append(f"| {r['t_h']:.2f} | {r['T2']:.2f} | {r['Q']:.0f} | {r['exceeds_T2_F']} | {r['exceeds_Q_Box']} | {r['in_domain_Box']} |")
    md.append("")

    def stream_table(title, rows):
        md.append(f"## {title}\n")
        md.append("| Stream | Limite | n | %fora união | %fora T² | %fora Q | retidos | Q mediana |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['stream']} | {r['limit']} | {r['n']} | {r['pct_out_union']} | {r['pct_out_T2']} "
                      f"| {r['pct_out_Q']} | {r['retained_n']} | {r['Q_median']:.0f} |")
        md.append("")
    stream_table("Tarefa 2 — exp7 três streams (M_primario)", t2rows)
    stream_table("Tarefa 3 — exp4 três streams (M_menos1, ≤12 h)", t3rows)

    md.append("### Tarefa 3b — exp4 por faixa de tempo (Box)\n")
    md.append("| Stream | Faixa | dentro | fora | % dentro | domina | arranque marcado |")
    md.append("|---|---|---|---|---|---|---|")
    for r in tb_rows:
        md.append(f"| {r['stream']} | {r['band']} | {r['n_dentro']} | {r['n_fora']} | {r['pct_dentro']}% "
                  f"| {r['dominance']} | {r['startup_flagged_times_h']} |")
    md.append("")

    md.append("## Tarefa 4 — comparação cruzada: % FORA (união 95%, Box)\n")
    md.append("| Caso | _refe | _inf | _inf_smoothed |")
    md.append("|---|---|---|---|")
    md.append(f"| exp4 (M_menos1, ≤12h) | {cross[('exp4','_refe')]:.1f} | {cross[('exp4','_inf')]:.1f} | {cross[('exp4','_inf_smoothed')]:.1f} |")
    md.append(f"| exp7 (M_primario) | {cross[('exp7','_refe')]:.1f} | {cross[('exp7','_inf')]:.1f} | {cross[('exp7','_inf_smoothed')]:.1f} |")
    md.append(f"\n**Interpretação:** {verdict}\n")

    md.append("## Tarefa 6 — sanidade da calibração\n")
    md.append("| Modelo | Limite | LV | T²_lim | Q_lim | exc.T² % | exc.Q % | união % | Q_cal med |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in san:
        md.append(f"| {r['model']} | {r['limit']} | {r['LV']} | {r['T2_lim']:.2f} | {r['Q_lim']:.0f} "
                  f"| {r['pct_exceed_T2']} | {r['pct_exceed_Q']} | {r['pct_exceed_union']} | {r['Q_cal_median']:.0f} |")
    md.append("\n(Tarefa 5: trajetória retida da exp4 em `exp4_trajectory.csv`.)\n")

    with open(os.path.join(OUT, "REFE_vs_INF_RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> {OUT}/ (exp7_refe_ad, exp7_three_streams, exp4_three_streams, exp4_timeband, "
          f"cross_comparison, exp4_trajectory, calibration_sanity .csv + REFE_vs_INF_RESUMO.md)")


if __name__ == "__main__":
    main()
