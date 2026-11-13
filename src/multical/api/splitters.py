"""Built-in validation schemes. Fold construction mirrors `MulticalEngine.run_cv`."""
import numpy as np

from .params import Param
from .registry import SPLITTERS, Splitter, register


def _complement(n, val_idx):
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    return np.flatnonzero(mask)


@register(SPLITTERS)
class KFold(Splitter):
    id = "kfold"
    label = "Validação cruzada em partes (k-fold)"
    kind = "cv"
    params = (
        Param("folds", "Número de partes", "int", default=5, min=2,
              help="Os dados são divididos neste número de partes; cada parte é prevista pelo modelo treinado com as outras."),
        Param("order", "Como dividir", "choice", default="venetian", choices=("venetian", "consecutive", "random"),
              help="venetian: alternado (1ª, 6ª, 11ª...); consecutive: blocos seguidos; random: sorteio.", advanced=True),
    )

    def split(self, n, groups=None, X=None, Y=None, rng=None, folds=5, order="venetian"):
        if folds > n:
            raise ValueError(f"Há {n} amostras para {folds} partes; reduza o número de partes.")
        if order == "random":
            rng = rng if rng is not None else np.random.RandomState()
            indices = rng.permutation(n)
        else:
            indices = np.arange(n)
        fold_size = int(np.ceil(n / folds))
        for i in range(folds):
            if order == "venetian":
                val_idx = np.arange(i, n, folds)
            else:
                val_idx = indices[i * fold_size:min((i + 1) * fold_size, n)]
            if len(val_idx) == 0:
                continue
            yield _complement(n, val_idx), val_idx


@register(SPLITTERS)
class LeaveOneOut(Splitter):
    id = "loo"
    label = "Deixar uma amostra de fora"
    kind = "cv"

    def split(self, n, groups=None, X=None, Y=None, rng=None):
        for i in range(n):
            val_idx = np.array([i])
            yield _complement(n, val_idx), val_idx


@register(SPLITTERS)
class Holdout(Splitter):
    id = "holdout"
    label = "Separar uma parte para validação"
    kind = "holdout"
    params = (
        Param("fraction", "Fração separada para validação", "float", default=0.2, min=0.01, max=0.9,
              help="Parte das amostras, escolhida por sorteio, que não entra no treino."),
    )

    def split(self, n, groups=None, X=None, Y=None, rng=None, fraction=0.2):
        rng = rng if rng is not None else np.random.RandomState()
        order = rng.permutation(n)
        n_val = int(np.floor(n * fraction))
        if n_val == 0:
            raise ValueError(f"Com {n} amostras, a fração {fraction} não separa nenhuma amostra para validação.")
        yield order[n_val:], order[:n_val]


@register(SPLITTERS)
class LeaveOneExperimentOut(Splitter):
    id = "loexo"
    label = "Deixar um experimento de fora"
    kind = "cv"
    needs_groups = True
    params = ()

    def split(self, n, groups=None, X=None, Y=None, rng=None):
        if groups is None:
            raise ValueError("Esta validação precisa saber a qual experimento cada amostra pertence.")
        groups = np.asarray(groups)
        names = list(dict.fromkeys(groups.tolist()))
        if len(names) < 2:
            raise ValueError("Para deixar um experimento de fora são necessários pelo menos 2 experimentos.")
        everything = np.arange(n)
        for name in names:
            held = groups == name
            yield everything[~held], everything[held]


def kennard_stone(X, n_keep, Y=None):
    """
    Positions of `n_keep` rows of X chosen to cover the data evenly: start with
    the two rows farthest apart, then repeatedly add the row farthest from the
    ones already chosen. With `Y`, distances in X and in Y are added after each
    is divided by its largest value (SPXY).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    n_keep = int(np.clip(n_keep, 2, n - 1))

    def distances(A):
        squares = (A ** 2).sum(axis=1)
        return np.sqrt(np.maximum(squares[:, None] + squares[None, :] - 2.0 * A @ A.T, 0.0))

    D = distances(X)
    if Y is not None:
        Dy = distances(np.asarray(Y, dtype=float).reshape(n, -1))
        D = D / (D.max() or 1.0) + Dy / (Dy.max() or 1.0)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    chosen = [int(i), int(j)]
    nearest = np.minimum(D[i], D[j])
    nearest[chosen] = -np.inf
    while len(chosen) < n_keep:
        pick = int(np.argmax(nearest))
        chosen.append(pick)
        nearest = np.minimum(nearest, D[pick])
        nearest[chosen] = -np.inf
    return np.array(sorted(chosen))


@register(SPLITTERS)
class KennardStone(Splitter):
    id = "ks"
    label = "Separar uma parte para validação (Kennard-Stone)"
    kind = "holdout"
    params = (
        Param("fraction", "Fração separada para validação", "float", default=0.2, min=0.01, max=0.9,
              help="As amostras de treino são escolhidas para cobrir bem a variedade dos espectros; o resto valida."),
        Param("method", "Critério de escolha", "choice", default="ks", choices=("ks", "spxy"), advanced=True,
              help="ks: só pelos espectros; spxy: pelos espectros e pelos valores medidos."),
        Param("autoscale", "Padronizar antes de medir distâncias", "bool", default=True, advanced=True),
    )

    def split(self, n, groups=None, X=None, Y=None, rng=None, fraction=0.2, method="ks", autoscale=True):
        if X is None:
            raise ValueError("Esta validação precisa dos espectros para escolher as amostras.")
        X = np.asarray(X, dtype=float)

        def scaled(A):
            A = np.asarray(A, dtype=float).reshape(n, -1)
            if not autoscale:
                return A
            std = A.std(axis=0)
            return (A - A.mean(axis=0)) / np.where(std == 0, 1.0, std)

        if method == "spxy" and Y is None:
            raise ValueError("O critério spxy precisa dos valores medidos.")
        train = kennard_stone(scaled(X), round((1.0 - fraction) * n), scaled(Y) if method == "spxy" else None)
        yield train, np.setdiff1d(np.arange(n), train)
