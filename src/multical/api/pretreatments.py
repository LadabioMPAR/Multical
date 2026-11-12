"""
Built-in pretreatments.

Each one wraps the existing numerical function from `multical.preprocessing`
(the same code the scripts use through `apply_pretreatment`), adds a parameter
description and, where the correction depends on the calibration set, keeps the
learned state so it can be reapplied to new samples.
"""
import numpy as np

from ..preprocessing.derivatives import diffmeu
from ..preprocessing.loess import loess
from ..preprocessing.savitzky_golay import sgolay_filt
from ..preprocessing.scatter_correction import emsc, snv
from ..preprocessing.smoothing import alisar
from ..utils import zscore_matlab_style
from .params import Param
from .registry import PRETREATMENTS, Pretreatment, pretreatment, register

_REGION, _SMOOTH, _DERIV, _SCATTER, _SCALE = "Região", "Suavização", "Derivada", "Espalhamento", "Escala"


@pretreatment(
    id="cut", label="Cortar região", group=_REGION,
    params=[Param("ranges", "Faixas mantidas", "ranges", default=[],
                  help="Só os pontos dentro destas faixas são usados. Sem faixas, nada é cortado.")],
)
def _cut(X, wl, ranges):
    if not ranges:
        return X, wl
    masks = [(wl >= lo) & (wl <= hi) for lo, hi in ranges]
    wl_new = np.concatenate([wl[m] for m in masks])
    if wl_new.size == 0:
        raise ValueError("O corte removeu todos os pontos do espectro; confira as faixas.")
    return np.hstack([X[:, m] for m in masks]), wl_new


@pretreatment(
    id="sg", label="Savitzky-Golay", group=_DERIV,
    params=[
        Param("radius", "Meia-janela (pontos)", "int", default=7, min=1,
              help="A janela do filtro tem 2 x meia-janela + 1 pontos."),
        Param("poly_order", "Grau do polinômio", "int", default=2, min=0, advanced=True),
        Param("deriv", "Ordem da derivada", "int", default=1, min=0, max=4,
              help="0 apenas suaviza; 1 e 2 calculam a primeira e a segunda derivada."),
    ],
)
def _sg(X, wl, radius, poly_order, deriv):
    window = 2 * radius + 1
    if window > X.shape[1]:
        raise ValueError(f"A janela do filtro ({window} pontos) é maior que o espectro ({X.shape[1]} pontos).")
    if poly_order >= window:
        raise ValueError(f"O grau do polinômio ({poly_order}) precisa ser menor que a janela ({window} pontos).")
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = sgolay_filt(wl, X[i], poly_order, window, deriv)
    return out, wl


@pretreatment(
    id="ma", label="Média móvel", group=_SMOOTH,
    params=[
        Param("radius", "Meia-janela (pontos)", "int", default=3, min=1),
        Param("mode", "Pontos do resultado", "choice", default="same_grid", choices=("same_grid", "downsample"),
              help="same_grid mantém todos os pontos; downsample reduz o número de pontos.", advanced=True),
    ],
)
def _ma(X, wl, radius, mode):
    return alisar(1 if mode == "same_grid" else 2, X, wl, radius)


@pretreatment(
    id="loess", label="Suavização LOESS", group=_SMOOTH,
    params=[
        Param("alpha", "Fração da vizinhança", "float", default=0.2, min=0.01, max=1.0),
        Param("order", "Grau do polinômio local", "choice", default=2, choices=(1, 2), advanced=True),
    ],
)
def _loess(X, wl, alpha, order):
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = loess(wl, X[i], alpha, order)
    return out, wl


@pretreatment(
    id="deriv", label="Derivada simples", group=_DERIV,
    params=[Param("order", "Ordem da derivada", "choice", default=1, choices=(1, 2))],
)
def _deriv(X, wl, order):
    first, second = diffmeu(X, wl)
    return (first if order == 1 else second), wl


@pretreatment(id="snv", label="SNV (variação normal padrão)", group=_SCATTER)
def _snv(X, wl):
    return snv(X), wl


@pretreatment(
    id="blctr", label="Correção de linha de base por região", group=_SCATTER,
    params=[
        Param("start", "Início da região", "float", default=0.0),
        Param("end", "Fim da região", "float", default=0.0),
        Param("target", "Valor desejado na região", "float", default=0.0, advanced=True),
    ],
)
def _blctr(X, wl, start, end, target):
    mask = (wl >= start) & (wl <= end)
    if not np.any(mask):
        raise ValueError(f"A região {start}–{end} não contém nenhum ponto do espectro.")
    shift = np.mean(X[:, mask], axis=1, keepdims=True) - target
    return X - shift, wl


class _ReferenceSpectrum(Pretreatment):
    """Scatter corrections that use the mean calibration spectrum as reference."""
    group = _SCATTER

    def fit(self, X, wavelengths, **params):
        return {"reference": np.mean(X, axis=0)}


@register(PRETREATMENTS)
class Emsc(_ReferenceSpectrum):
    id = "emsc"
    label = "EMSC (correção de espalhamento estendida)"
    params = (Param("degree", "Grau da linha de base", "int", default=2, min=0, max=6, advanced=True),)

    def apply(self, X, wavelengths, state, degree):
        return emsc(X, wavelengths, degree=degree, reference=state["reference"]), wavelengths


@register(PRETREATMENTS)
class Msc(_ReferenceSpectrum):
    id = "msc"
    label = "MSC (correção de espalhamento multiplicativo)"

    def apply(self, X, wavelengths, state):
        return emsc(X, wavelengths, degree=0, reference=state["reference"]), wavelengths


@register(PRETREATMENTS)
class AutoScale(Pretreatment):
    id = "autoscale"
    label = "Autoescalar (média zero, desvio um)"
    group = _SCALE

    def fit(self, X, wavelengths):
        _, mean, std = zscore_matlab_style(X)
        std = np.where(std == 0, 1.0, std)
        return {"mean": mean, "std": std}

    def apply(self, X, wavelengths, state):
        return (X - state["mean"]) / state["std"], wavelengths


@register(PRETREATMENTS)
class MeanCenter(Pretreatment):
    id = "meancenter"
    label = "Centrar na média"
    group = _SCALE

    def fit(self, X, wavelengths):
        return {"mean": np.mean(X, axis=0)}

    def apply(self, X, wavelengths, state):
        return X - state["mean"], wavelengths


def recipe_from_legacy(pretreat_list):
    """
    Convert the list format used by the scripts, e.g.
    [['Cut', 4400, 7500, 1], ['SG', 7, 2, 1, 1]], into a recipe.
    The last element of each legacy step is the "plot" flag and is dropped.
    """
    recipe = []
    for step in pretreat_list or []:
        name, args = step[0], list(step[1:-1]) if len(step) > 1 else []
        if name == "Cut":
            params = {"ranges": [[args[i], args[i + 1]] for i in range(0, len(args) - 1, 2)]}
            recipe.append({"id": "cut", "params": params})
        elif name == "SG":
            recipe.append({"id": "sg", "params": {"radius": args[0], "poly_order": args[1], "deriv": args[2]}})
        elif name == "MA":
            mode = "same_grid" if args[1] == 1 else "downsample"
            recipe.append({"id": "ma", "params": {"radius": args[0], "mode": mode}})
        elif name == "Loess":
            recipe.append({"id": "loess", "params": {"alpha": args[0], "order": args[1]}})
        elif name == "Deriv":
            recipe.append({"id": "deriv", "params": {"order": args[0]}})
        elif name == "SNV":
            recipe.append({"id": "snv", "params": {}})
        elif name == "EMSC":
            recipe.append({"id": "emsc", "params": {"degree": int(args[0])}})
        elif name == "MSC":
            recipe.append({"id": "msc", "params": {}})
        elif name == "BLCtr":
            recipe.append({"id": "blctr", "params": {"start": args[0], "end": args[1], "target": args[2]}})
        elif name == "AutoScale":
            recipe.append({"id": "autoscale", "params": {}})
        elif name == "MeanCenter":
            recipe.append({"id": "meancenter", "params": {}})
        else:
            raise ValueError(f"Pré-tratamento antigo desconhecido: {name!r}.")
    return recipe
