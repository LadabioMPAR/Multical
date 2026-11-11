"""
Demonstrate the sparsity-penalised PSO for GLUCOSE variable selection.

Sweeps the penalty lambda and shows the trade-off:
    fitness = RMSECV * (1 + lambda * n_selected/n_total)
For each lambda: run PSO (np=30, default accel) a few times, keep the run with the
best (penalised) PSO score, then report that mask's TRUE RMSECV (F-test LVs) and the
number of wavenumbers retained. Produces a Pareto-style plot (#vars & RMSECV vs lambda).

Pooled training = exp4..exp7. Preprocessing Cut+SG (run_variable_selection default).

Usage:
  py scripts/calibration/run_pso_sparsity_glucose.py          # full (~30 min)
  py scripts/calibration/run_pso_sparsity_glucose.py smoke    # quick check
"""
import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.multical.var_selection import run_pso, select_k_ftest
import scripts.calibration.run_loexo_plsvarsel_exp4to7 as V

SMOKE = (len(sys.argv) > 1 and sys.argv[1].lower() == 'smoke')

RESULTS_DIR = "results_pso_sparsity_glucose"
EXPERIMENTS = ['exp4', 'exp5', 'exp6', 'exp7']
GL = 1
MAX_LV = V.MAX_LV
K_FOLDS = V.K_FOLDS
CV_TYPE = V.CV_TYPE

# Best PSO config from the tuning study (defaults won)
PSO_NP, PSO_W, PSO_C1, PSO_C2 = 30, 0.9, 1.49, 1.49
MAX_ITER = 120
REPEATS = 2
# Sweep the initial inclusion probability p_init (the effective sparsity knob).
# A soft penalty alone does NOT reduce variables on this PSO (masks start ~50% dense);
# starting sparse (low p_init) lets the swarm reach sparse solutions. lambda adds pressure.
LAMBDA_FIXED = 1.0
PINIT_GRID = [None, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01]   # None = original dense init

if SMOKE:
    MAX_ITER = 6
    REPEATS = 1
    PINIT_GRID = [None, 0.1, 0.02]


def true_rmsecv(A_sub, y):
    """Unpenalised RMSECV (and F-test LVs) of a given variable subset."""
    ycv, kmax = V._cv_pred(A_sub, y, MAX_LV)
    rk = np.sqrt(np.mean((y[:, None] - ycv) ** 2, axis=0))
    k = int(min(select_k_ftest(rk, n_cal=len(y)), kmax))
    return float(rk[k - 1]), k


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"MODE: {'SMOKE' if SMOKE else 'FULL'}")
    data = {e: V.load_full(e) for e in EXPERIMENTS}
    A = np.vstack([data[e][1] for e in EXPERIMENTS])
    Y = np.vstack([data[e][0] for e in EXPERIMENTS])
    y = Y[:, GL]
    n_total = A.shape[1]
    print(f"Pooled: {A.shape[0]} samples, {n_total} wavenumbers. Target=glucose.\n")

    print(f"Sweeping p_init (initial inclusion prob), lambda fixed = {LAMBDA_FIXED}\n")
    csv_path = os.path.join(RESULTS_DIR, "sparsity_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["p_init", "lambda", "n_vars", "frac_vars", "true_rmsecv", "LVs", "pso_score", "seconds"])

    rows = []
    for li, pinit in enumerate(PINIT_GRID):
        best = (np.inf, None)               # (penalised pso score, mask)
        t0 = time.perf_counter()
        for rep in range(REPEATS):
            mask, pscore = run_pso(A, Y, MAX_LV, K_FOLDS, CV_TYPE, target_idx=GL,
                                   n_particles=PSO_NP, max_iter=MAX_ITER,
                                   w=PSO_W, c1=PSO_C1, c2=PSO_C2, seed=100 * li + rep,
                                   verbose=False, lambda_sparsity=LAMBDA_FIXED, p_init=pinit)
            if pscore < best[0]:
                best = (pscore, mask)
        dt = time.perf_counter() - t0
        mask = best[1]
        nvars = int(mask.sum())
        rmsecv, k = true_rmsecv(A[:, mask], y)
        rows.append((pinit, nvars, nvars / n_total, rmsecv, k, best[0]))
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([pinit, LAMBDA_FIXED, nvars, f"{nvars/n_total:.3f}",
                                    f"{rmsecv:.4f}", k, f"{best[0]:.4f}", f"{dt:.1f}"])
        print(f"  p_init={str(pinit):<5}: vars={nvars:5d} ({100*nvars/n_total:4.1f}%)  "
              f"true_RMSECV={rmsecv:.4f}  LVs={k}  ({dt:.0f}s)")

    # ---- Report ----
    print(f"\n{'='*62}\nSPARSITY SWEEP (glucose) - PSO np={PSO_NP}, lambda={LAMBDA_FIXED}\n{'='*62}")
    h = f"{'p_init':^8}|{'#vars':^8}|{'% vars':^8}|{'RMSECV':^9}|{'LVs':^5}"
    print("-" * len(h)); print(h); print("-" * len(h))
    for (pinit, nv, fr, rm, k, _) in rows:
        print(f"{str(pinit):^8}|{nv:^8}|{100*fr:^8.1f}|{rm:^9.4f}|{k:^5}")
    print("-" * len(h))

    # ---- Plot: #vars and RMSECV vs p_init ----
    labels = [str(r[0]) for r in rows]
    nvs = [r[1] for r in rows]
    rms = [r[3] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    ax1.bar(x, nvs, color='steelblue', alpha=0.7)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_xlabel("p_init (initial inclusion probability)")
    ax1.set_ylabel("# wavenumbers retained", color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(x, rms, 'o-', color='darkred')
    ax2.set_ylabel("true RMSECV glucose (g/L)", color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax1.set_title(f"Sparse PSO (lambda={LAMBDA_FIXED}): variables vs RMSECV (glucose)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sparsity_tradeoff.png"), dpi=300)
    print(f"\nSaved CSV + plot in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
