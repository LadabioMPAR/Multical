"""
Reader for the text format used so far: one pair of files per experiment.

- reference file: one row per sample, optional first column with the time, then
  one column per analyte;
- spectra file: header row "Time wl1 wl2 ...", then one row per sample with the
  time in the first column.

Rows are matched by position, as in the existing scripts.
"""
import os

import numpy as np

from ..api.dataset import Dataset
from ..api.registry import IMPORTERS, Importer, register


def read_txt_pairs(pairs, analytes=None, units=""):
    """`pairs` is a list of (reference_file, spectra_file). Returns a Dataset."""
    X_parts, Y_parts, ids, groups, times = [], [], [], [], []
    wavelengths = None
    for ref_path, spec_path in pairs:
        for path in (ref_path, spec_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        name = os.path.splitext(os.path.basename(spec_path))[0]

        ref = np.loadtxt(ref_path, ndmin=2)
        ref_values = ref[:, 1:] if ref.shape[1] > 1 else ref

        with open(spec_path, "r") as handle:
            header = handle.readline().split()
        wl = np.array([float(v) for v in header[1:]])
        spec = np.loadtxt(spec_path, skiprows=1, ndmin=2)
        spectra, spec_times = spec[:, 1:], spec[:, 0]

        if spectra.shape[1] != wl.size:
            raise ValueError(f"{spec_path}: o cabeçalho tem {wl.size} comprimentos de onda e as linhas têm {spectra.shape[1]} valores.")
        if spectra.shape[0] != ref_values.shape[0]:
            raise ValueError(
                f"{os.path.basename(spec_path)} tem {spectra.shape[0]} espectros e "
                f"{os.path.basename(ref_path)} tem {ref_values.shape[0]} linhas de valores medidos."
            )
        if wavelengths is None:
            wavelengths = wl
        elif wl.size != wavelengths.size or not np.allclose(wl, wavelengths, atol=1e-1):
            raise ValueError(f"{os.path.basename(spec_path)} tem uma faixa espectral diferente dos arquivos anteriores.")

        X_parts.append(spectra)
        Y_parts.append(ref_values)
        times.append(spec_times)
        groups.extend([name] * spectra.shape[0])
        ids.extend(f"{name}_{i + 1}" for i in range(spectra.shape[0]))

    if not X_parts:
        raise ValueError("Nenhum arquivo foi informado.")
    return Dataset(
        X=np.vstack(X_parts), wavelengths=wavelengths, Y=np.vstack(Y_parts),
        analytes=list(analytes or []), units=units, sample_ids=ids,
        groups=np.array(groups), times=np.concatenate(times),
    )


@register(IMPORTERS)
class TxtPairImporter(Importer):
    id = "txt_pair"
    label = "Par de arquivos de texto (valores medidos + espectros)"
    extensions = (".txt",)

    def read(self, sources, **options):
        return read_txt_pairs(sources, **options)
