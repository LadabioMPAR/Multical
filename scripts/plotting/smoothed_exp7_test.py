"""
smoothed_exp7_test.py — exp7 raw vs smoothed online stream under the AD filter (NON-DESTRUCTIVE)
==============================================================================================
Single question: does using the SMOOTHED online stream (`_inf_smoothed`) for exp7 (Assay 4, 15 g)
make the AD filter RETAIN the steady-state regime — as it did for exp4 (Assay 1, 20 g) — or does
exp7 stay out of domain?

The ONLY variable is raw vs smoothed online stream. Everything else identical to production:
  - preprocessing Cut(4400-7500) -> Savitzky-Golay(15,2,1) -> autoscale (cal mean/scale reused)
  - T2/Q via build_paper_data.applicability_domain (reused verbatim)
  - LV by Osten F-test (alpha=0.05), glucose, on the M_primario training (Assay 1-3)
  - AD limits from the M_primario TRAINING ONLY (anti-circular)
Both Q-limit definitions reported side by side:
  - Jackson-Mudholkar (build_paper_data)   - from residual eigenvalues of calibration
  - Box chi^2 (run_calibration)            - from the moments of the calibration in-sample Q
(T2 limit is the SAME F-distribution formula in both -> reported once.)

ALL outputs -> results_smoothed_exp7/ only. results/ and the pipeline are untouched.

Run:  MPLBACKEND=Agg py scripts/plotting/smoothed_exp7_test.py
"""
import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.stats import chi2
from src.multical.preprocessing.pipeline import apply_pretreatment
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V
import scripts.plotting.build_paper_data as B

OUT = "results_smoothed_exp7"
DATA = "data"
XMAX_H = 24.0
MODELS = {
    "M_primario": {"train": ["exp4", "exp5", "exp6"], "test": "exp7"},
    "M_menos1":   {"train": ["exp5", "exp6", "exp7"], "test": "exp4"},
}
STREAM = {  # (raw, smoothed) online files per experiment
    "exp7": ("exp_07_inf.txt", "exp_07_inf_smoothed.txt"),
    "exp4": ("exp_04_inf.txt", "exp_04_inf_smoothed.txt"),
}
PCTS = [5, 25, 50, 75, 95, 99]


def wcsv(name, fieldnames, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def load_spec(e):
    path = os.path.join(DATA, f"{e}_nonda.txt")
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    raw = np.atleast_2d(np.loadtxt(path, skiprows=1))[:, 1:]
    return apply_pretreatment(B.PRETREAT, raw, wl, plot=False)[0]

def load_online(fname):
    path = os.path.join(DATA, fname)
    with open(path) as f:
        wl = np.array([float(v) for v in f.readline().split()[1:]])
    arr = np.atleast_2d(np.loadtxt(path, skiprows=1))
    t = arr[:, 0] / 60.0
    keep = t <= XMAX_H
    return t[keep], apply_pretreatment(B.PRETREAT, arr[keep, 1:], wl, plot=False)[0]

def ftest_glu_lv(A_tr, y_tr):
    rk, _, km = B.cv_curve(A_tr, y_tr, 15)
    return int(min(V.select_k_ftest(rk, n_cal=len(y_tr)), km))

def box_qlim(Qcal, alpha=0.95):
    m = float(np.mean(Qcal)); v = float(np.var(Qcal, ddof=1)) or 1e-10
    return (v / (2 * m)) * chi2.ppf(alpha, 2 * m * m / v)


def model_context(mname):
    """Fit M_* on its training set; return T2/Q ingredients + both Q-limits (anti-circular)."""
    m = MODELS[mname]
    A_tr = np.vstack([load_spec(e) for e in m["train"]])
    y_glu = np.concatenate([B.load_refe(e)[2] for e in m["train"]])
    A = ftest_glu_lv(A_tr, y_glu)
    adc = B.applicability_domain(A_tr, y_glu, A_tr, A)      # cal self-projection
    T2_lim = adc["T2_lim"]; Q_lim_JM = adc["Q_lim"]
    Q_lim_Box = box_qlim(adc["Q"])
    return dict(A_tr=A_tr, y_glu=y_glu, A=A, n_cal=adc["n"],
                T2_lim=T2_lim, Q_lim_JM=Q_lim_JM, Q_lim_Box=Q_lim_Box,
                T2_cal=adc["T2"], Q_cal=adc["Q"], Q_cal_med=float(np.median(adc["Q"])))


def stream_metrics(ctx, fname):
    t, X = load_online(fname)
    ad = B.applicability_domain(ctx["A_tr"], ctx["y_glu"], X, ctx["A"])
    return dict(t=t, T2=ad["T2"], Q=ad["Q"], n=len(t))


def flags(T2, Q, t2lim, qlim):
    eT2, eQ = T2 > t2lim, Q > qlim
    return eT2, eQ, (eT2 | eQ), (eT2 & eQ)


def main():
    os.makedirs(OUT, exist_ok=True)

    ctxP = model_context("M_primario")
    print(f"M_primario: n_cal={ctxP['n_cal']}, glucose F-test LV={ctxP['A']}")
    print(f"  T2_lim(F)={ctxP['T2_lim']:.3f} | Q_lim JM={ctxP['Q_lim_JM']:.1f} | Q_lim Box={ctxP['Q_lim_Box']:.1f}")

    raw = stream_metrics(ctxP, STREAM["exp7"][0])
    smo = stream_metrics(ctxP, STREAM["exp7"][1])
    print(f"  exp7 online n: raw(<=24h)={raw['n']}  smoothed(<=24h)={smo['n']}")

    # =============== TASK 1 — raw vs smoothed, both limits ===============
    def col(stream, qlim):
        T2, Q = stream["T2"], stream["Q"]
        eT2, eQ, un, inter = flags(T2, Q, ctxP["T2_lim"], qlim)
        d = {"n_online": stream["n"],
             "pct_T2": 100 * eT2.mean(), "pct_Q": 100 * eQ.mean(),
             "pct_union": 100 * un.mean(), "pct_inter": 100 * inter.mean(),
             "retained_n": int((~un).sum()),
             "Q_median": float(np.median(Q)),
             "ratio_medQ_on_cal": float(np.median(Q) / ctxP["Q_cal_med"])}
        for p in PCTS:
            d[f"Q_p{p}"] = float(np.percentile(Q, p))
        return d
    cols = {"raw_JM": col(raw, ctxP["Q_lim_JM"]), "raw_Box": col(raw, ctxP["Q_lim_Box"]),
            "smoothed_JM": col(smo, ctxP["Q_lim_JM"]), "smoothed_Box": col(smo, ctxP["Q_lim_Box"])}
    order = ["raw_JM", "raw_Box", "smoothed_JM", "smoothed_Box"]
    metrics = (["n_online", "pct_T2", "pct_Q", "pct_union", "pct_inter", "retained_n",
                "Q_median", "ratio_medQ_on_cal"] + [f"Q_p{p}" for p in PCTS])
    print("\n" + "=" * 92 + "\nTASK 1 — exp7 raw vs smoothed (T2_lim shared; Q_lim JM vs Box)\n" + "=" * 92)
    print(f"  {'metric':20s}" + "".join(f"{c:>16s}" for c in order))
    for mk in metrics:
        fmt = "{:.0f}" if mk in ("n_online", "retained_n") else ("{:.2f}" if "ratio" in mk else "{:.1f}")
        print(f"  {mk:20s}" + "".join(f"{fmt.format(cols[c][mk]):>16s}" for c in order))
    wcsv("exp7_raw_vs_smoothed.csv", ["metric"] + order,
         [dict({"metric": mk}, **{c: round(cols[c][mk], 3) for c in order}) for mk in metrics])

    # =============== TASK 2 — time-band breakdown ===============
    BANDS = [("t<1h (arranque)", 0.0, 1.0), ("1-12h (regime)", 1.0, 12.0), ("12-24h (deriva)", 12.0, 24.0)]
    tb_rows = []
    print("\n" + "=" * 92 + "\nTASK 2 — exp7 time-band (dentro/fora) — both limits\n" + "=" * 92)
    for sname, stream in [("raw", raw), ("smoothed", smo)]:
        for lname, qlim in [("JM", ctxP["Q_lim_JM"]), ("Box", ctxP["Q_lim_Box"])]:
            eT2, eQ, un, _ = flags(stream["T2"], stream["Q"], ctxP["T2_lim"], qlim)
            for bname, lo, hi in BANDS:
                sel = (stream["t"] >= lo) & (stream["t"] < hi)
                n = int(sel.sum())
                if n == 0:
                    continue
                nin = int((sel & ~un).sum()); nout = int((sel & un).sum())
                nT2 = int((sel & eT2).sum()); nQ = int((sel & eQ).sum())
                dom = "T2" if nT2 > nQ else ("Q" if nQ > nT2 else "tie")
                print(f"  {sname:9s} {lname:4s} {bname:16s}: dentro={nin:3d} fora={nout:3d} "
                      f"({100*nin/n:4.0f}% dentro)  dom={dom} (T2={nT2},Q={nQ})")
                tb_rows.append({"stream": sname, "limit": lname, "band": bname, "n": n,
                                "n_dentro": nin, "n_fora": nout, "pct_dentro": round(100*nin/n, 1),
                                "flag_by_T2": nT2, "flag_by_Q": nQ, "dominance": dom})
    wcsv("exp7_by_timeband.csv",
         ["stream", "limit", "band", "n", "n_dentro", "n_fora", "pct_dentro", "flag_by_T2", "flag_by_Q", "dominance"], tb_rows)

    # =============== TASK 3 — exp4 vs exp7 (smoothed) ===============
    ctxM = model_context("M_menos1")
    smo4 = stream_metrics(ctxM, STREAM["exp4"][1])
    print("\n" + "=" * 92 + "\nTASK 3 — exp4 (M_menos1) vs exp7 (M_primario), SMOOTHED streams\n" + "=" * 92)
    cmp_rows = []
    def regime_pct_in(stream, t2lim, qlim):
        sel = (stream["t"] >= 1.0) & (stream["t"] < 12.0)
        _, _, un, _ = flags(stream["T2"], stream["Q"], t2lim, qlim)
        return 100 * (sel & ~un).sum() / max(int(sel.sum()), 1)
    def dom_overall(stream, t2lim, qlim):
        eT2, eQ, un, _ = flags(stream["T2"], stream["Q"], t2lim, qlim)
        nT2 = int((un & eT2).sum()); nQ = int((un & eQ).sum())
        return "T2" if nT2 > nQ else ("Q" if nQ > nT2 else "tie")
    for label, stream, ctx in [("exp4 / M_menos1", smo4, ctxM), ("exp7 / M_primario", smo, ctxP)]:
        for lname, qlim in [("JM", ctx["Q_lim_JM"]), ("Box", ctx["Q_lim_Box"])]:
            _, _, un, _ = flags(stream["T2"], stream["Q"], ctx["T2_lim"], qlim)
            row = {"case": label, "limit": lname, "n_online": stream["n"],
                   "pct_out_total": round(100 * un.mean(), 1),
                   "pct_in_regime_1_12h": round(regime_pct_in(stream, ctx["T2_lim"], qlim), 1),
                   "Q_median": round(float(np.median(stream["Q"])), 1),
                   "dominance": dom_overall(stream, ctx["T2_lim"], qlim)}
            cmp_rows.append(row)
            print(f"  {label:20s} {lname:4s}: fora_total={row['pct_out_total']:5.1f}%  "
                  f"dentro_regime(1-12h)={row['pct_in_regime_1_12h']:5.1f}%  "
                  f"Q_med={row['Q_median']:.0f}  dom={row['dominance']}")
    wcsv("exp4_vs_exp7_smoothed.csv",
         ["case", "limit", "n_online", "pct_out_total", "pct_in_regime_1_12h", "Q_median", "dominance"], cmp_rows)

    # =============== TASK 4 — calibration sanity (smoothed arm uses same cal) ===============
    print("\n" + "=" * 92 + "\nTASK 4 — calibration self-check (limits are training-derived; smoothing doesn't touch cal)\n" + "=" * 92)
    san_rows = []
    for mname, ctx in [("M_primario", ctxP), ("M_menos1", ctxM)]:
        for lname, qlim in [("JM", ctx["Q_lim_JM"]), ("Box", ctx["Q_lim_Box"])]:
            eT2, eQ, un, _ = flags(ctx["T2_cal"], ctx["Q_cal"], ctx["T2_lim"], qlim)
            print(f"  {mname:11s} {lname:4s}: exceeds T2={100*eT2.mean():4.1f}%  Q={100*eQ.mean():4.1f}%  "
                  f"union={100*un.mean():4.1f}%  | Q_cal min/med/max={ctx['Q_cal'].min():.0f}/"
                  f"{np.median(ctx['Q_cal']):.0f}/{ctx['Q_cal'].max():.0f}")
            san_rows.append({"model": mname, "limit": lname, "n_cal": ctx["n_cal"],
                             "pct_exceed_T2": round(100*float(eT2.mean()), 1),
                             "pct_exceed_Q": round(100*float(eQ.mean()), 1),
                             "pct_exceed_union": round(100*float(un.mean()), 1),
                             "Qcal_min": round(float(ctx["Q_cal"].min()), 1),
                             "Qcal_median": round(float(np.median(ctx["Q_cal"])), 1),
                             "Qcal_max": round(float(ctx["Q_cal"].max()), 1)})
    wcsv("calibration_sanity.csv",
         ["model", "limit", "n_cal", "pct_exceed_T2", "pct_exceed_Q", "pct_exceed_union",
          "Qcal_min", "Qcal_median", "Qcal_max"], san_rows)

    # =============== VERDICT + MARKDOWN ===============
    _, _, un_raw_JM, _ = flags(raw["T2"], raw["Q"], ctxP["T2_lim"], ctxP["Q_lim_JM"])
    _, _, un_smo_JM, _ = flags(smo["T2"], smo["Q"], ctxP["T2_lim"], ctxP["Q_lim_JM"])
    ret_raw = int((~un_raw_JM).sum()); ret_smo = int((~un_smo_JM).sum())
    reg_in_smo = regime_pct_in(smo, ctxP["T2_lim"], ctxP["Q_lim_JM"])
    v1 = f"exp7 retidos (união 95%, Jackson-Mudholkar): cru = {ret_raw}/{raw['n']}  ->  suavizado = {ret_smo}/{smo['n']}"
    v2 = f"exp7 regime estacionario (1-12 h): % dentro do dominio no suavizado (JM) = {reg_in_smo:.0f}%"
    print("\n" + "#" * 92 + "\nVEREDITO\n" + "#" * 92)
    print("  1) " + v1); print("  2) " + v2)

    md = ["# TESTE SMOOTHED exp7 — RESUMO\n"]
    md.append(f"**1) {v1}**  ")
    md.append(f"**2) {v2}**\n")
    md.append("Única variável: stream online CRU vs SUAVIZADO (`_inf_smoothed`). Mesmo pré-processamento "
              "(Cut+SG+autoescala), mesmo cálculo de T²/Q, mesma LV por teste-F (glicose, Assay 1-3 → "
              f"LV={ctxP['A']}), limites 95% só do treino M_primário (anti-circular). Ambos os limites de Q "
              "reportados: Jackson-Mudholkar (build_paper_data) e Box χ² (run_calibration); T² usa a mesma "
              "distribuição F nos dois. Truncado em ≤24 h nos dois streams. Nada de produção foi alterado.\n")
    md.append(f"Limites (M_primário, n_cal={ctxP['n_cal']}, LV={ctxP['A']}): "
              f"T²_lim={ctxP['T2_lim']:.2f}; Q_lim JM={ctxP['Q_lim_JM']:.0f}; Q_lim Box={ctxP['Q_lim_Box']:.0f}.\n")

    md.append("## Tarefa 1 — exp7 cru vs suavizado (JM | Box)\n")
    md.append("| Métrica | cru·JM | cru·Box | suav·JM | suav·Box |")
    md.append("|---|---|---|---|---|")
    lbl = {"n_online": "n online", "pct_T2": "reprov T² %", "pct_Q": "reprov Q %",
           "pct_union": "reprov união %", "pct_inter": "reprov interseção %", "retained_n": "**RETIDOS n**",
           "Q_median": "Q mediana", "ratio_medQ_on_cal": "razão Q_med on/cal"}
    for mk in ["n_online", "pct_T2", "pct_Q", "pct_union", "pct_inter", "retained_n", "Q_median", "ratio_medQ_on_cal"] + [f"Q_p{p}" for p in PCTS]:
        name = lbl.get(mk, mk)
        fmt = (lambda x: f"{x:.0f}") if mk in ("n_online", "retained_n") else ((lambda x: f"{x:.2f}") if "ratio" in mk else (lambda x: f"{x:.1f}"))
        md.append(f"| {name} | " + " | ".join(fmt(cols[c][mk]) for c in order) + " |")
    md.append("")

    md.append("## Tarefa 2 — quebra por faixa de tempo (limite JM)\n")
    md.append("| Stream | Faixa | dentro | fora | % dentro | domina |")
    md.append("|---|---|---|---|---|---|")
    for r in tb_rows:
        if r["limit"] != "JM":
            continue
        md.append(f"| {r['stream']} | {r['band']} | {r['n_dentro']} | {r['n_fora']} | {r['pct_dentro']}% | {r['dominance']} |")
    md.append("\n(tabela completa com Box em `exp7_by_timeband.csv`)\n")

    md.append("## Tarefa 3 — exp4 vs exp7 (streams suavizados)\n")
    md.append("| Caso | Limite | n | fora total % | dentro regime 1-12h % | Q mediana | domina |")
    md.append("|---|---|---|---|---|---|---|")
    for r in cmp_rows:
        md.append(f"| {r['case']} | {r['limit']} | {r['n_online']} | {r['pct_out_total']} | "
                  f"{r['pct_in_regime_1_12h']} | {r['Q_median']} | {r['dominance']} |")
    md.append("")

    md.append("## Tarefa 4 — sanidade na calibração\n")
    md.append("| Modelo | Limite | excede T² % | excede Q % | excede união % | Q_cal min/med/max |")
    md.append("|---|---|---|---|---|---|")
    for r in san_rows:
        md.append(f"| {r['model']} | {r['limit']} | {r['pct_exceed_T2']} | {r['pct_exceed_Q']} | "
                  f"{r['pct_exceed_union']} | {r['Qcal_min']:.0f}/{r['Qcal_median']:.0f}/{r['Qcal_max']:.0f} |")
    md.append("")

    with open(os.path.join(OUT, "TESTE_SMOOTHED_exp7_RESUMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> {OUT}/exp7_raw_vs_smoothed.csv, exp7_by_timeband.csv, exp4_vs_exp7_smoothed.csv, "
          f"calibration_sanity.csv, TESTE_SMOOTHED_exp7_RESUMO.md")


if __name__ == "__main__":
    main()
