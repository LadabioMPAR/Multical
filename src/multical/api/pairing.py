"""
Linking measured values to the spectra of a time series.

A spectrometer file holds one spectrum every couple of minutes for the whole
experiment, with the time in seconds since the first spectrum. The measured
(reference) values come in a table with the sampling time in minutes. Two ways
of choosing the spectrum for each sample:

- "exact": the spectrum closest to the sampling time;
- "window": the mean of every spectrum within `radius_min` minutes of it.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .dataset import Dataset

MATCH_MODES = {
    "exact": "Tempo exato da amostra (espectro mais próximo)",
    "window": "Janela em torno do tempo da amostra (média dos espectros)",
}


@dataclass
class TimeMatch:
    """Which spectra stand for each row of the reference table."""
    mode: str
    radius_min: Optional[float]         # window mode: half-width of the window, minutes
    limit_min: float                    # largest accepted distance to a spectrum, minutes
    ref_times_min: np.ndarray
    indices: List[np.ndarray]           # per reference row: positions of the spectra used (empty: none)
    distance_min: np.ndarray            # per reference row: distance to the closest spectrum, minutes

    @property
    def matched(self):
        return np.array([idx.size > 0 for idx in self.indices], dtype=bool)

    def problems(self):
        """One sentence per reference row left without a spectrum."""
        out = []
        for time, idx, distance in zip(self.ref_times_min, self.indices, self.distance_min):
            if idx.size:
                continue
            if self.mode == "window":
                out.append(f"Tempo {time:g} min: nenhum espectro a até {self.radius_min:g} min "
                           f"(o mais próximo está a {distance:.1f} min).")
            else:
                out.append(f"Tempo {time:g} min: o espectro mais próximo está a {distance:.1f} min "
                           f"(limite de {self.limit_min:.1f} min).")
        return out


def match_times(spectrum_times_s, ref_times_min, mode="exact", radius_min=None, max_distance_min=None):
    """
    Choose the spectra for each reference time.

    `spectrum_times_s` are seconds since the first spectrum; `ref_times_min` are
    minutes on the same clock. In "exact" mode a sample is left unmatched when
    the closest spectrum is farther than `max_distance_min` (default: the usual
    interval between spectra, so only samples outside the recorded period fail).
    In "window" mode `radius_min` is required.
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"Modo de ligação desconhecido: {mode!r}. Use 'exact' ou 'window'.")
    spec = np.asarray(spectrum_times_s, dtype=float).ravel() / 60.0
    ref = np.asarray(ref_times_min, dtype=float).ravel()
    if spec.size == 0:
        raise ValueError("Não há espectros para ligar aos valores medidos.")
    if ref.size == 0:
        raise ValueError("A tabela de valores medidos está vazia.")
    if not (np.isfinite(spec).all() and np.isfinite(ref).all()):
        raise ValueError("Há tempos ausentes ou inválidos.")

    gaps = np.abs(spec[None, :] - ref[:, None])        # (n_ref, n_spectra)
    nearest = gaps.argmin(axis=1)
    distance = gaps[np.arange(ref.size), nearest]

    if mode == "window":
        if radius_min is None or not float(radius_min) > 0:
            raise ValueError("Informe o raio da janela, em minutos (maior que zero).")
        radius_min = limit = float(radius_min)
        indices = [np.flatnonzero(row <= limit + 1e-9) for row in gaps]
    else:
        radius_min = None
        if max_distance_min is not None:
            limit = float(max_distance_min)
        elif spec.size > 1:
            limit = float(np.median(np.diff(np.sort(spec))))
        else:
            limit = float("inf")
        indices = [np.array([j]) if d <= limit + 1e-9 else np.array([], dtype=int) for j, d in zip(nearest, distance)]
    return TimeMatch(mode, radius_min, limit, ref, indices, distance)


def dataset_from_match(match, rows, wavelengths, Y, analytes=None, units="", label="amostra"):
    """
    Build the calibration Dataset for a `TimeMatch`.

    `rows(indices)` returns the spectra at those positions. Reference rows left
    without a spectrum are dropped; `match.problems()` lists them.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.shape[0] != match.ref_times_min.size:
        raise ValueError(f"A tabela tem {match.ref_times_min.size} tempos e {Y.shape[0]} linhas de valores medidos.")
    keep = np.flatnonzero(match.matched)
    if keep.size == 0:
        raise ValueError("Nenhuma amostra pôde ser ligada a um espectro. " + " ".join(match.problems()[:3]))
    X = np.vstack([np.asarray(rows(match.indices[i]), dtype=float).mean(axis=0) for i in keep])
    return Dataset(
        X=X, wavelengths=wavelengths, Y=Y[keep], analytes=list(analytes or []), units=units,
        sample_ids=[f"{label}_{match.ref_times_min[i]:g}min" for i in keep],
        groups=np.array([label] * keep.size),
        times=match.ref_times_min[keep],
    )


def pair_with_reference(spectra, ref_times_min, Y, analytes=None, units="",
                        mode="exact", radius_min=None, max_distance_min=None):
    """
    Link a reference table to the spectra of one experiment already in memory.

    Returns `(dataset, match)`: the Dataset has one row per linked sample (time
    in minutes); `match` tells which spectra were used and which samples failed.
    """
    if spectra.times is None:
        raise ValueError("Os espectros não têm tempo associado; a ligação pelo tempo não é possível.")
    label = "amostra"
    if spectra.groups is not None:
        names = list(dict.fromkeys(spectra.groups.tolist()))
        if len(names) > 1:
            raise ValueError("Os espectros vêm de mais de um experimento; faça a ligação de um experimento por vez.")
        label = str(names[0])
    match = match_times(spectra.times, ref_times_min, mode, radius_min, max_distance_min)
    data = dataset_from_match(match, lambda idx: spectra.X[idx], spectra.wavelengths, Y, analytes, units, label)
    return data, match
