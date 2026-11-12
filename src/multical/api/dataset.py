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

    @classmethod
    def from_legacy(cls, x0, absor0, analytes=None, units=""):
        """Build from the matrices used by the scripts: `absor0` has the wavelengths in row 0."""
        absor0 = np.asarray(absor0, dtype=float)
        return cls(X=absor0[1:], wavelengths=absor0[0], Y=x0, analytes=list(analytes or []), units=units)

    def to_legacy(self):
        """Return (x0, absor0) as expected by `MulticalEngine.run`."""
        return self.Y, np.vstack([self.wavelengths, self.X])
