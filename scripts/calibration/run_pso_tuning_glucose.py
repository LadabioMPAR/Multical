"""
Overnight PSO tuning for GLUCOSE variable selection + final LOExO report (exp4..exp7).

Phases:
  A) PSO hyper-parameter study (target = glucose) on pooled exp4-7: grid over
     n_particles and (w, c1, c2), each repeated several times (PSO is stochastic).
     Records final RMSECV, #variables, convergence iteration and full convergence
     history. Incremental CSV so partial results survive interruption.
  B) Convergence-curve plot (best RMSECV vs iteration) per configuration.
  C) Final LOExO (PLS / VIP / PSO, glucose & xylose) using the BEST PSO config.
     PSO ALWAYS optimises glucose; the glucose-selected mask is reused for xylose.
     Best held-out batch is chosen by **RMSEP** (lowest).

LV selection: Osten F-test (select_k_ftest) — same as run_variable_selection.py.
Preprocessing: Cut(4400-7500) + Savitzky-Golay(7,2,1).

Usage:
  py scripts/calibration/run_pso_tuning_glucose.py          # full overnight run
  py scripts/calibration/run_pso_tuning_glucose.py smoke    # quick validation
"""
import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.var_selection import run_pso
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V   # reuse load_full, pls_fold_metrics, vip_mask, ...

SMOKE = (len(sys.argv) > 1 and sys.argv[1].lower() == 'smoke')

# =============================================================================
#                              CONFIGURATION
# =============================================================================
RESULTS_DIR = "results_pso_tuning_glucose"
EXPERIMENTS = ['exp4', 'exp5', 'exp6', 'exp7']
GL, XY = 1, 2
MAX_LV = V.MAX_LV
K_FOLDS = V.K_FOLDS
CV_TYPE = V.CV_TYPE

# --- PSO hyper-parameter grid (Phase A) ---
GRID_N_PARTICLES = [30, 60, 100]
GRID_ACCEL = [                      # (w_initial, c1, c2)   w decays to 0.4 internally
    (0.9, 1.49, 1.49),              # default
    (0.5, 2.0, 2.0),                # low inertia, high acceleration
    (0.7, 2.0, 1.5),                # moderate, cognitive-leaning
]
MAX_ITER_STUDY = 120
REPEATS = 3

# --- Final report (Phase C) ---
PSO_FINAL_REPEATS = 3
MAX_ITER_FINAL = 200

if SMOKE:
    GRID_N_PARTICLES = [10]
    GRID_ACCEL = [(0.9, 1.49, 1.49)]
    MAX_ITER_STUDY = 4
    REPEATS = 1
    PSO_FINAL_REPEATS = 1
    MAX_ITER_FINAL = 4

METHODS = ['PLS', 'VIP', 'PSO']
ANALYTES = {'gl': GL, 'xy': XY}


def _last_improve(history):
    scores = [h[1] for h in history]
    last = 1
    for i in range(1, len(scores)):
        if scores[i] < scores[i - 1] - 1e-9:
            last = i + 1
    return last


def _load_existing(summ_path):
    """Read an existing summary CSV -> (summary dict, set of done (config_id, rep))."""
    summary, done = {}, set()
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            for row in csv.DictReader(f):
                cid, rep = int(row['config_id']), int(row['rep'])
                s = summary.setdefault(cid, dict(n_particles=int(row['n_particles']),
                        w=float(row['w']), c1=float(row['c1']), c2=float(row['c2']),
                        scores=[], nvars=[]))
                s['scores'].append(float(row['final_rmsecv']))
                s['nvars'].append(int(row['n_vars']))
                done.add((cid, rep))
    return summary, done


def phase_a(A_pool, Y_pool):
    """RESUMABLE: skips (config, rep) pairs already present in the summary CSV."""
    print(f"\n{'='*70}\nPHASE A: PSO hyper-parameter study (glucose)\n{'='*70}")
    summ_path = os.path.join(RESULTS_DIR, "pso_study_summary.csv")
    hist_path = os.path.join(RESULTS_DIR, "pso_study_history.csv")
    summary, done = _load_existing(summ_path)
    if not os.path.exists(summ_path):
        with open(summ_path, "w", newline="") as f:
            csv.writer(f).writerow(["config_id", "n_particles", "w", "c1", "c2", "rep", "seed",
                                    "final_rmsecv", "n_vars", "last_improve_iter", "seconds"])
    if not os.path.exists(hist_path):
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(["config_id", "rep", "iteration", "best_rmsecv", "n_vars"])
    if done:
        print(f"Resuming: {len(done)} run(s) already in CSV will be skipped.")

    cid = 0
    for npart in GRID_N_PARTICLES:
        for (w, c1, c2) in GRID_ACCEL:
            cid += 1
            summary.setdefault(cid, dict(n_particles=npart, w=w, c1=c1, c2=c2, scores=[], nvars=[]))
            for rep in range(REPEATS):
                if (cid, rep) in done:
                    continue
                seed = 1000 * cid + rep
                t0 = time.perf_counter()
                mask, score, hist = run_pso(A_pool, Y_pool, MAX_LV, K_FOLDS, CV_TYPE, target_idx=GL,
                                            n_particles=npart, max_iter=MAX_ITER_STUDY,
                                            w=w, c1=c1, c2=c2, seed=seed, return_history=True, verbose=False)
                dt = time.perf_counter() - t0
                nvars = int(mask.sum()); li = _last_improve(hist)
                summary[cid]['scores'].append(score); summary[cid]['nvars'].append(nvars)
                with open(summ_path, "a", newline="") as f:
                    csv.writer(f).writerow([cid, npart, w, c1, c2, rep, seed,
                                            f"{score:.5f}", nvars, li, f"{dt:.1f}"])
                with open(hist_path, "a", newline="") as f:
                    wri = csv.writer(f)
                    for (it, sc, nv) in hist:
                        wri.writerow([cid, rep, it, f"{sc:.5f}", nv])
                print(f"  cfg{cid} (np={npart}, w={w}, c1={c1}, c2={c2}) rep{rep}: "
                      f"RMSECV={score:.4f}, vars={nvars}, conv@{li}/{MAX_ITER_STUDY}, {dt:.0f}s")

    print(f"\n--- Phase A ranking (by mean final RMSECV) ---")
    ranked = sorted(summary.items(), key=lambda kv: np.mean(kv[1]['scores']) if kv[1]['scores'] else np.inf)
    for cid_, s in ranked:
        if not s['scores']:
            continue
        print(f"  cfg{cid_}: np={s['n_particles']}, w={s['w']}, c1={s['c1']}, c2={s['c2']} | "
              f"RMSECV {np.mean(s['scores']):.4f} ± {np.std(s['scores']):.4f} | vars {int(np.mean(s['nvars']))}")
    best_cid = ranked[0][0]
    return summary, best_cid


def phase_b():
    """Reads the history CSV (works across restarts) and plots convergence curves."""
    import collections
    summ_path = os.path.join(RESULTS_DIR, "pso_study_summary.csv")
    hist_path = os.path.join(RESULTS_DIR, "pso_study_history.csv")
    summary, _ = _load_existing(summ_path)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            for row in csv.DictReader(f):
                by[int(row['config_id'])][int(row['rep'])].append(
                    (int(row['iteration']), float(row['best_rmsecv'])))
    if not by:
        print("No history to plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cid in sorted(by):
        runs = []
        for rep, pts in by[cid].items():
            pts.sort()
            runs.append([p[1] for p in pts])
        L = min(len(r) for r in runs)
        mean_curve = np.array([r[:L] for r in runs]).mean(0)
        s = summary.get(cid, {})
        ax.plot(range(1, L + 1), mean_curve,
                label=f"cfg{cid}: np={s.get('n_particles')}, w={s.get('w')}, c1={s.get('c1')}, c2={s.get('c2')}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best RMSECV (glucose)")
    ax.set_title("PSO convergence by configuration (mean over repeats)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pso_convergence.png"), dpi=300)
    print(f"\nSaved convergence plot: {RESULTS_DIR}/pso_convergence.png")


def phase_c(data, best_params):
    npart, w, c1, c2 = best_params
    print(f"\n{'='*70}\nPHASE C: final LOExO with best PSO config "
          f"(np={npart}, w={w}, c1={c1}, c2={c2})\n{'='*70}")
    results = {m: {a: {} for a in ANALYTES} for m in METHODS}

    for held in EXPERIMENTS:
        train_e = [e for e in EXPERIMENTS if e != held]
        A_tr = np.vstack([data[e][1] for e in train_e])
        Y_tr = np.vstack([data[e][0] for e in train_e])
        A_te = data[held][1]; Y_te = data[held][0]
        print(f"\n### Held-out = {held} (train {train_e})")

        # PSO mask: ALWAYS glucose; keep the best of several repeats (lowest RMSECV)
        best_mask, best_s = None, np.inf
        for r in range(PSO_FINAL_REPEATS):
            m, s = run_pso(A_tr, Y_tr, MAX_LV, K_FOLDS, CV_TYPE, target_idx=GL,
                           n_particles=npart, max_iter=MAX_ITER_FINAL, w=w, c1=c1, c2=c2,
                           seed=777 + r, return_history=False, verbose=False)
            if s < best_s:
                best_s, best_mask = s, m
        print(f"    PSO(glucose) mask: {int(best_mask.sum())} vars, RMSECV={best_s:.4f}")

        masks_per_analyte = {a: {'PLS': np.ones(A_tr.shape[1], dtype=bool),
                                 'VIP': V.vip_mask(A_tr, Y_tr, jt),
                                 'PSO': best_mask} for a, jt in ANALYTES.items()}

        for a, jt in ANALYTES.items():
            for m in METHODS:
                mask = masks_per_analyte[a][m]
                met = V.pls_fold_metrics(A_tr[:, mask], Y_tr[:, jt], A_te[:, mask], Y_te[:, jt])
                results[m][a][held] = met
                print(f"      {m:<4} {a}: vars={met['n_vars']:5d} LV={met['k']:2d} "
                      f"R2cal={met['r2cal']:+.3f} R2cv={met['r2cv']:+.3f} R2test={met['r2test']:+.3f} "
                      f"RMSECV={met['rmsecv']:.3f} RMSEP={met['rmsep']:.3f}")

    # Report (best batch by MIN RMSEP)
    lines = []
    lines.append("#" * 78)
    lines.append(f"FINAL LOExO REPORT (best PSO cfg: np={npart}, w={w}, c1={c1}, c2={c2}) "
                 f"- best batch by RMSEP".center(78))
    lines.append("#" * 78)
    for a in ANALYTES:
        lines.append(f"\n================  ANALYTE: {a.upper()}  ================")
        for m in METHODS:
            lines.append(f"\n--- Method: {m} ---")
            hdr = f"{'held-out':<9}|{'#wave':^7}|{'LVs':^5}|{'R2cal':^8}|{'R2cv':^8}|{'R2test':^8}|{'RMSECV':^8}|{'RMSEP':^8}"
            lines.append("-" * len(hdr)); lines.append(hdr); lines.append("-" * len(hdr))
            best_e, best_rmsep = None, np.inf
            for held in EXPERIMENTS:
                r = results[m][a][held]
                lines.append(f"{held:<9}|{r['n_vars']:^7}|{r['k']:^5}|{r['r2cal']:^+8.3f}|{r['r2cv']:^+8.3f}|"
                             f"{r['r2test']:^+8.3f}|{r['rmsecv']:^8.3f}|{r['rmsep']:^8.3f}")
                if r['rmsep'] < best_rmsep:
                    best_rmsep, best_e = r['rmsep'], held
            lines.append("-" * len(hdr))
            r = results[m][a][best_e]
            lines.append(f">> BEST batch (by RMSEP): {best_e} | #wavenumbers={r['n_vars']} | LVs={r['k']}")
            lines.append(f"   R2_cal={r['r2cal']:.3f} R2_cv={r['r2cv']:.3f} R2_test={r['r2test']:.3f} "
                         f"RMSECV={r['rmsecv']:.3f} RMSEP={r['rmsep']:.3f} g/L")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "final_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nSaved final report: {RESULTS_DIR}/final_report.txt")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"MODE: {'SMOKE' if SMOKE else 'FULL (overnight)'}")
    data = {e: V.load_full(e) for e in EXPERIMENTS}
    A_pool = np.vstack([data[e][1] for e in EXPERIMENTS])
    Y_pool = np.vstack([data[e][0] for e in EXPERIMENTS])
    print(f"Pooled training for Phase A: {A_pool.shape[0]} samples, {A_pool.shape[1]} wavenumbers")

    summary, best_cid = phase_a(A_pool, Y_pool)
    phase_b()
    s = summary[best_cid]
    best_params = (s['n_particles'], s['w'], s['c1'], s['c2'])
    print(f"\n>>> Best PSO config from Phase A: cfg{best_cid} -> "
          f"np={s['n_particles']}, w={s['w']}, c1={s['c1']}, c2={s['c2']}")
    phase_c(data, best_params)
    print("\nALL DONE.")


if __name__ == "__main__":
    main()
