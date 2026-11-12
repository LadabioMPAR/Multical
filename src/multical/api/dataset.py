"""The data container shared by importers, calibration and prediction."""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray                              # spectra, (n_samples, n_wavelengths)
    wavelengths: np.ndarray                    # (n_wavelengths,)
    Y: Optional[np.ndarray] = None             # reference values, (n_samples, n_analytes)
    analytes: List[str] = field(default_factory=list)
    units: str = ""
    sample_ids: Optional[List[str]] = None
    groups: Optional[np.ndarray] = None        # experiment/batch label per sample
    times: Optional[np.ndarray] = None

    def __post_init__(self):
        self.X = np.atleast_2d(np.asarray(self.X, dtype=float))
        self.wavelengths = np.asarray(self.wavelengths, dtype=float).ravel()
        n, p = self.X.shape
        if self.wavelengths.size != p:
            raise ValueError(
                f"Os espectros têm {p} pontos, mas o eixo de comprimentos de onda tem {self.wavelengths.size}."
            )
        if self.Y is not None:
            self.Y = np.asarray(self.Y, dtype=float)
            if self.Y.ndim == 1:
                self.Y = self.Y.reshape(-1, 1)
            if self.Y.shape[0] != n:
                raise ValueError(
                    f"Há {n} espectros e {self.Y.shape[0]} linhas de valores medidos; os números precisam ser iguais."
                )
            if not self.analytes:
                self.analytes = [f"analito_{i + 1}" for i in range(self.Y.shape[1])]
            if len(self.analytes) != self.Y.shape[1]:
                raise ValueError(
                    f"Foram dados {len(self.analytes)} nomes de analitos para {self.Y.shape[1]} colunas de valores."
                )
        self.analytes = [str(a).strip() for a in self.analytes]
        if self.sample_ids is None:
            self.sample_ids = [f"amostra_{i + 1}" for i in range(n)]
        for name in ("sample_ids", "groups", "times"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"'{name}' tem {len(value)} itens para {n} espectros.")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=float)

    @property
    def n_samples(self):
        return self.X.shape[0]

    def subset(self, index):
        """New Dataset with the selected samples (boolean mask or index array)."""
        index = np.asarray(index)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        return Dataset(
            X=self.X[index],
            wavelengths=self.wavelengths.copy(),
            Y=None if self.Y is None else self.Y[index],
            analytes=list(self.analytes),
            units=self.units,
            sample_ids=[self.sample_ids[i] for i in index],
            groups=None if self.groups is None else self.groups[index],
            times=None if self.times is None else self.times[index],
        )

    @staticmethod
    def concat(parts):
        """Stack datasets (for instance one per experiment) that share the spectral range and the analytes."""
        parts = list(parts)
        if not parts:
            raise ValueError("Nenhum conjunto de dados foi informado.")
        first = parts[0]
        for part in parts[1:]:
            if part.wavelengths.size != first.wavelengths.size or not np.allclose(part.wavelengths, first.wavelengths, atol=1e-6):
                raise ValueError("Os conjuntos têm uma faixa espectral diferente e não podem ser reunidos.")
            if (part.Y is None) != (first.Y is None) or part.analytes != first.analytes:
                raise ValueError("Os conjuntos não têm os mesmos analitos e não podem ser reunidos.")
        if len(parts) == 1:
            return first

        def joined(name):
            values = [getattr(part, name) for part in parts]
            return None if any(v is None for v in values) else np.concatenate(values)

        return Dataset(
            X=np.vstack([part.X for part in parts]),
            wavelengths=first.wavelengths.copy(),
            Y=None if first.Y is None else np.vstack([part.Y for part in parts]),
            analytes=list(first.analytes),
            units=first.units,
            sample_ids=[sample for part in parts for sample in part.sample_ids],
            groups=joined("groups"),
            times=joined("times"),
        )

    @classmethod
    def from_legacy(cls, x0, absor0, analytes=None, units=""):
        """Build from the matrices used by the scripts: `absor0` has the wavelengths in row 0."""
        absor0 = np.asarray(absor0, dtype=float)
        return cls(X=absor0[1:], wavelengths=absor0[0], Y=x0, analytes=list(analytes or []), units=units)

    def to_legacy(self):
        """Return (x0, absor0) as expected by `MulticalEngine.run`."""
        return self.Y, np.vstack([self.wavelengths, self.X])
