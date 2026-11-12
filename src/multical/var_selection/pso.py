import numpy as np
from multiprocessing import Pool, cpu_count
from .utils import pso_worker_wrapper

def run_pso(absor, x, kmax, folds, cv_type, n_particles=30, max_iter=50, w=0.9, c1=1.49, c2=1.49,
            target_idx=None, seed=None, return_history=False, verbose=True, lambda_sparsity=0.0,
            p_init=None):
    """
    Performs Particle Swarm Optimization (PSO) for variable selection with Parallel Batch Processing.
    Uses a binary PSO approach where position [0,1] is probability of inclusion.

    Extra (optional, non-breaking) args:
        seed           : if given, np.random.seed(seed) for reproducible runs.
        return_history : if True, also return a list of (iteration, best_score, n_vars).
        verbose        : if False, suppress the progress prints.
        lambda_sparsity: sparsity penalty. Fitness = RMSECV * (1 + lambda * n_sel/n_total).
                         0.0 (default) = original behaviour (pure RMSECV); larger -> fewer
                         variables. Note: with lambda>0 the returned/best score is the
                         PENALISED fitness, but the returned MASK is what matters downstream.
        p_init         : initial inclusion probability per variable. None (default) keeps
                         the original init (X_pos ~ U[0,1] -> ~50-73% inclusion, which
                         cannot reach sparse regions). Set e.g. 0.1 so the swarm STARTS
                         sparse (~10% vars) and can actually explore sparse solutions;
                         combine with lambda_sparsity to control the trade-off.
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = absor.shape[1]
    history = []

    # Initialize Particles
    # X: Position (logit space; sigmoid(X) = inclusion probability)
    # V: Velocity
    if p_init is None:
        X_pos = np.random.rand(n_particles, n_vars)              # original: ~50-73% inclusion
    else:
        base = np.log(p_init / (1.0 - p_init))                  # logit(p_init)
        X_pos = base + (np.random.rand(n_particles, n_vars) - 0.5) * 2.0   # spread around p_init
    V_vel = np.random.uniform(-1, 1, (n_particles, n_vars))

    P_best_pos = X_pos.copy()
    P_best_score = np.full(n_particles, np.inf)

    G_best_pos = np.zeros(n_vars)
    G_best_score = np.inf
    G_best_mask = np.ones(n_vars, dtype=bool)

    if verbose:
        print(f"PSO Initialization: {n_particles} particles, {max_iter} iterations")

    # Determine parallel power
    try:
        n_cpu = max(1, cpu_count() - 1)
    except:
        n_cpu = 1

    if verbose:
        print(f"Parallel Execution on {n_cpu} cores.")
    
    # Initialize Pool ONCE
    with Pool(processes=n_cpu) as pool:
        for iteration in range(max_iter):
            
            # Linear Decay of Inertia Weight (w)
            w_curr = w - ((w - 0.4) * (iteration / max_iter))
            
            # 1. Generate all masks for this iteration
            masks_to_eval = []
            for i in range(n_particles):
                # Sigmoid transfer
                sigmoid_x = 1 / (1 + np.exp(-X_pos[i]))
                current_mask = np.random.rand(n_vars) < sigmoid_x
                masks_to_eval.append(current_mask)
            
            # 2. Evaluate in Parallel
            # Create args list. Note: Absor/X are reused, Copy-on-Write handles memory on Linux
            args_list = [(m, absor, x, kmax, folds, cv_type, target_idx) for m in masks_to_eval]
            
            # This blocks until all particles are evaluated
            raw_scores = pool.map(pso_worker_wrapper, args_list)

            # Sparsity penalty: penalise masks that keep many variables
            if lambda_sparsity > 0:
                scores = [rs * (1.0 + lambda_sparsity * (m.sum() / n_vars))
                          for rs, m in zip(raw_scores, masks_to_eval)]
            else:
                scores = raw_scores

            # 3. Update Swarm Logic
            for i in range(n_particles):
                score = scores[i]
                current_mask = masks_to_eval[i]
                
                # Update Personal Best
                if score < P_best_score[i]:
                    P_best_score[i] = score
                    P_best_pos[i] = X_pos[i].copy()
                    
                    # Update Global Best
                    if score < G_best_score:
                        G_best_score = score
                        G_best_pos = X_pos[i].copy()
                        G_best_mask = current_mask.copy()
                        if verbose:
                            print(f"  [Iter {iteration+1}] New Global Best: RMSE={G_best_score:.4f} (Vars: {np.sum(G_best_mask)}) w={w_curr:.2f}")

            # Record convergence history (best-so-far at the end of this iteration)
            history.append((iteration + 1, float(G_best_score), int(np.sum(G_best_mask))))

            # 4. Update Velocity and Position
            r1 = np.random.rand(n_particles, n_vars)
            r2 = np.random.rand(n_particles, n_vars)
            
            V_vel = (w_curr * V_vel) + (c1 * r1 * (P_best_pos - X_pos)) + (c2 * r2 * (G_best_pos - X_pos))
            V_vel = np.clip(V_vel, -4, 4)
            X_pos = X_pos + V_vel

    if return_history:
        return G_best_mask, G_best_score, history
    return G_best_mask, G_best_score
