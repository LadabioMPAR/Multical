"""
Built-in variable selections: for each analyte, which points of the pretreated
spectrum the model uses.

The three methods follow `multical.var_selection` (VIP with the search for the
limit made by scripts/calibration/run_variable_selection.py, particle swarm and
simulated annealing), with one difference fixed for this layer: the error they
minimise is that of the model of a single analyte, not of a joint model. Random
numbers are drawn in the same order as in that code, so the same seed gives the
same choice.
"""
import numpy as np

from ..models.pls import PLS
from ..utils import zscore_matlab_style
from ..var_selection.utils import select_k_ftest
from .params import Param
from .registry import SELECTORS, Selector, register

MIN_POINTS = 2      # as in the original code: a choice with fewer points than this is not evaluated


def _autoscale(X):
    _, mean, std = zscore_matlab_style(X)
    return (X - mean) / np.where(std == 0, 1.0, std)


def _scale_y(y):
    yn, _, _ = zscore_matlab_style(np.asarray(y, dtype=float).reshape(-1, 1))
    return yn


def vip_scores(X, y, k):
    """
    Importance of each point of the spectrum (VIP) in a PLS with `k` latent
    variables fitted to autoscaled X and y, as `multical.var_selection.vip.calculate_vip`.
    Points above 1 matter more than the average.
    """
    X = np.asarray(X, dtype=float)
    _, _, _, _, _, W, _, r2y = PLS().nipals(_autoscale(X), _scale_y(y), k)
    return np.sqrt(X.shape[1] * ((W[:, :k] ** 2) @ r2y[:k]) / np.sum(r2y[:k]))


def pls_complexity(X, y, folds, max_k):
    """Number of latent variables the F-test picks for a PLS on every point, validated on `folds`."""
    Xn = _autoscale(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n = Xn.shape[0]
    max_k = int(min(max_k, min(len(train) for train, _ in folds) - 1, Xn.shape[1]))
    if max_k < 1:
        raise ValueError("Há poucas amostras para calcular a importância dos pontos do espectro (VIP).")
    predicted = np.full((max_k, n), np.nan)
    for train, val in folds:
        yn, y_mean, y_std = zscore_matlab_style(y[train].reshape(-1, 1))
        _, _, P, _, Q, W, _, _ = PLS().nipals(Xn[train], yn, max_k)
        for k in range(1, max_k + 1):
            beta = W[:, :k] @ np.linalg.pinv(P[:, :k].T @ W[:, :k]) @ Q[:, :k].T
            predicted[k - 1, val] = (Xn[val] @ beta).ravel() * float(y_std[0]) + float(y_mean[0])
    seen = ~np.isnan(predicted[0])
    rmsecv = np.sqrt(np.mean((predicted[:, seen] - y[seen]) ** 2, axis=1))
    return int(select_k_ftest(rmsecv, n))


def _number(value):
    return f"{value:.4g}".replace(".", ",")


@register(SELECTORS)
class VIPSelector(Selector):
    id = "vip"
    label = "VIP (importância de cada ponto em um PLS)"
    params = (
        Param("auto", "Escolher o limite automaticamente", "bool", default=True,
              help="Testa limites de 0,1 a 1,4 e fica com o de menor erro de validação."),
        Param("threshold", "Limite de importância", "float", default=1.0, min=0.0,
              help="Ficam os pontos com importância (VIP) a partir deste valor. Usado quando a escolha automática está desligada."),
        Param("max_k", "Máximo de variáveis latentes do PLS usado para medir a importância", "int", default=15, min=1,
              advanced=True),
    )
    thresholds = np.arange(0.1, 1.5, 0.1)

    def select(self, X, y, error, rng, progress, auto, threshold, max_k):
        k = pls_complexity(X, y, error.folds, max_k)
        scores = vip_scores(X, y, k)
        if not auto:
            mask = scores >= threshold
            if mask.sum() < MIN_POINTS:
                raise ValueError(
                    f"Com o limite {_number(threshold)}, sobram {int(mask.sum())} ponto(s) do espectro; "
                    f"o maior valor de importância é {_number(scores.max())}. Reduza o limite.")
            return mask
        best_mask, best_error = np.ones(X.shape[1], dtype=bool), np.inf
        for number, limit in enumerate(self.thresholds, start=1):
            mask = scores >= limit
            if mask.sum() >= MIN_POINTS:
                value = error(mask)
                if value < best_error:
                    best_error, best_mask = value, mask
            progress(number / len(self.thresholds), f"limite {_number(limit)} testado")
        return best_mask


@register(SELECTORS)
class PSOSelector(Selector):
    """Binary particle swarm, as `multical.var_selection.pso.run_pso`."""
    id = "pso"
    label = "PSO (enxame de partículas)"
    params = (
        Param("n_particles", "Número de partículas", "int", default=30, min=2,
              help="Quantas escolhas de pontos são testadas em cada rodada."),
        Param("max_iter", "Número de rodadas", "int", default=50, min=1,
              help="O tempo cresce com partículas × rodadas."),
        Param("lambda_sparsity", "Penalidade por usar muitos pontos", "float", default=0.0, min=0.0, advanced=True,
              help="0: só o erro conta. Valores maiores preferem escolhas com menos pontos "
                   "(o erro é multiplicado por 1 + penalidade × fração de pontos usados)."),
        Param("p_init", "Fração de pontos nas escolhas iniciais", "float", default=0.0, min=0.0, max=0.99, advanced=True,
              help="0: partida original do método (cerca de 60% dos pontos). Por exemplo 0,1 começa com 10% dos pontos."),
        Param("w", "Inércia inicial", "float", default=0.9, min=0.0, advanced=True,
              help="Diminui até 0,4 ao longo das rodadas."),
        Param("c1", "Peso do melhor resultado da própria partícula", "float", default=1.49, min=0.0, advanced=True),
        Param("c2", "Peso do melhor resultado do enxame", "float", default=1.49, min=0.0, advanced=True),
    )

    def select(self, X, y, error, rng, progress, n_particles, max_iter, lambda_sparsity, p_init, w, c1, c2):
        n_points = X.shape[1]
        if p_init <= 0:
            position = rng.rand(n_particles, n_points)
        else:
            position = np.log(p_init / (1.0 - p_init)) + (rng.rand(n_particles, n_points) - 0.5) * 2.0
        velocity = rng.uniform(-1, 1, (n_particles, n_points))
        own_best, own_score = position.copy(), np.full(n_particles, np.inf)
        best, best_score, best_mask = np.zeros(n_points), np.inf, np.ones(n_points, dtype=bool)

        for iteration in range(max_iter):
            inertia = w - ((w - 0.4) * (iteration / max_iter))
            masks = [rng.rand(n_points) < 1 / (1 + np.exp(-position[i])) for i in range(n_particles)]
            for i, mask in enumerate(masks):
                score = error(mask) if mask.sum() >= MIN_POINTS else 1e9
                if lambda_sparsity > 0:
                    score = score * (1.0 + lambda_sparsity * (mask.sum() / n_points))
                if score < own_score[i]:
                    own_score[i], own_best[i] = score, position[i].copy()
                    if score < best_score:
                        best_score, best, best_mask = score, position[i].copy(), mask.copy()
                progress((iteration + (i + 1) / n_particles) / max_iter,
                         f"rodada {iteration + 1} de {max_iter}; melhor até agora: {int(best_mask.sum())} pontos")
            r1, r2 = rng.rand(n_particles, n_points), rng.rand(n_particles, n_points)
            velocity = np.clip(inertia * velocity + c1 * r1 * (own_best - position) + c2 * r2 * (best - position), -4, 4)
            position = position + velocity
        return best_mask


@register(SELECTORS)
class AnnealingSelector(Selector):
    """Simulated annealing, as `multical.var_selection.sa.run_simulated_annealing`."""
    id = "sa"
    label = "Recozimento simulado"
    params = (
        Param("max_iter", "Número de tentativas", "int", default=100, min=1,
              help="Em cada tentativa, um ou alguns pontos entram ou saem da escolha."),
        Param("initial_temp", "Temperatura inicial", "float", default=0.1, min=1e-9, advanced=True,
              help="Quanto maior, mais o método aceita pioras no começo."),
        Param("alpha", "Resfriamento por tentativa", "float", default=0.90, min=0.01, max=1.0, advanced=True,
              help="A temperatura é multiplicada por este valor a cada tentativa."),
    )

    def select(self, X, y, error, rng, progress, max_iter, initial_temp, alpha):
        n_points = X.shape[1]
        current = rng.rand(n_points) > 0.5
        if current.sum() < MIN_POINTS:
            current[:MIN_POINTS] = True
        current_error = error(current)
        best, best_error = current.copy(), current_error
        temperature = initial_temp
        for i in range(max_iter):
            progress(i / max_iter, f"tentativa {i + 1} de {max_iter}; melhor até agora: {int(best.sum())} pontos")
            n_change = 1 if rng.rand() < 0.7 else max(2, int(0.01 * n_points))
            neighbour = current.copy()
            flipped = rng.choice(n_points, min(n_change, n_points), replace=False)
            neighbour[flipped] = ~neighbour[flipped]
            if neighbour.sum() < MIN_POINTS:
                continue
            neighbour_error = error(neighbour)
            delta = neighbour_error - current_error
            with np.errstate(over="ignore", invalid="ignore"):
                accept = delta < 0 or rng.rand() < np.exp(-delta / temperature)
            if accept:
                current, current_error = neighbour, neighbour_error
                if current_error < best_error:
                    best_error, best = current_error, current.copy()
            temperature *= alpha
        return best
