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
