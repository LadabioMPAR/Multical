"""
Reader for .spp files: one file holds the whole time series of an experiment
(one spectrum every couple of minutes, thousands per file).

Layout, worked out from sample files (there is no public specification):

- the file is an OLE compound file;
- the stream "CONTENTS" is a property set: number of points, first x value and
  step, number of spectra collected, units, and the names of the data streams;
- the data stream is spectra x points float64 values, one spectrum after the
  other; the time stream is one float64 per spectrum, in seconds;
- the file is created with room for a fixed number of spectra, so only the
  first "collected" ones are real; the rest is zeros.

Spectra are returned with the x axis in increasing order, as in the text files
used so far (the instrument stores it from the highest value to the lowest).
"""
import os
import struct

import numpy as np

from ..api.dataset import Dataset
from ..api.pairing import dataset_from_match, match_times
from ..api.registry import IMPORTERS, Importer, register
from ._cfb import CompoundFile, CompoundFileError

# Property ids inside CONTENTS.
_P_CODEPAGE, _P_X_UNIT, _P_TIME_UNIT, _P_POINTS = 1, 6, 7, 9
_P_STEP, _P_START, _P_ALLOCATED, _P_COLLECTED = 11, 12, 14, 15
_P_Y_UNIT, _P_DATA_STREAM, _P_TIME_STREAM = 17, 18, 19

_VT_I2, _VT_I4, _VT_R8, _VT_BSTR, _VT_UI4, _VT_STREAM = 2, 3, 5, 8, 19, 66
# The clock of the instrument is a 32-bit millisecond counter; some files store
# it after it wrapped around, as negative seconds.
_CLOCK_WRAP_SECONDS = 2 ** 32 / 1000.0


def _properties(data):
    """Decode the property set of CONTENTS into {id: value}."""
    data = bytes(data)
    try:
        raw = {}
        for section in range(struct.unpack_from("<I", data, 24)[0]):
            base = struct.unpack_from("<I", data, 44 + 20 * section)[0]
            count = struct.unpack_from("<I", data, base + 4)[0]
            for i in range(count):
                prop_id, offset = struct.unpack_from("<II", data, base + 8 + 8 * i)
                pos = base + offset
                kind = struct.unpack_from("<I", data, pos)[0]
                if kind == _VT_I2:
                    raw[prop_id] = struct.unpack_from("<h", data, pos + 4)[0]
                elif kind == _VT_I4:
                    raw[prop_id] = struct.unpack_from("<i", data, pos + 4)[0]
                elif kind == _VT_UI4:
                    raw[prop_id] = struct.unpack_from("<I", data, pos + 4)[0]
                elif kind == _VT_R8:
                    raw[prop_id] = struct.unpack_from("<d", data, pos + 4)[0]
                elif kind in (_VT_BSTR, _VT_STREAM):
                    length = struct.unpack_from("<I", data, pos + 4)[0]
                    if pos + 8 + length > len(data):
                        raise CompoundFileError("propriedades corrompidas")
                    raw[prop_id] = data[pos + 8:pos + 8 + length]
    except struct.error:
        raise CompoundFileError("propriedades corrompidas") from None
    encoding = "utf-16-le" if raw.get(_P_CODEPAGE, 1200) == 1200 else "cp1252"
    return {
        key: value.decode(encoding, "replace").split("\x00")[0] if isinstance(value, bytes) else value
        for key, value in raw.items()
    }


class _SppFile:
    """An open .spp file with its description decoded."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        try:
            self._cf = CompoundFile(path)
        except CompoundFileError as error:
            raise ValueError(f"{self.name} não é um arquivo .spp válido ({error}).") from None
        try:
            self._describe()
        except CompoundFileError as error:
            self._cf.close()
            raise ValueError(f"{self.name} não pôde ser lido: {error}.") from None
        except Exception:
            self._cf.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cf.close()

    def _describe(self):
        if not self._cf.has_stream("CONTENTS"):
            raise CompoundFileError("não tem a descrição dos espectros (não parece ser um .spp)")
        props = _properties(self._cf.read("CONTENTS"))
        self.n_points = int(props.get(_P_POINTS, 0))
        self.start = float(props.get(_P_START, 0.0))
        self.step = float(props.get(_P_STEP, 0.0))
        self.n_allocated = int(props.get(_P_ALLOCATED, 0))
        self.x_unit = props.get(_P_X_UNIT, "")
        self.y_unit = props.get(_P_Y_UNIT, "")
        self.time_unit = props.get(_P_TIME_UNIT, "")
        self._data = props.get(_P_DATA_STREAM) or "prop18"
        self._time = props.get(_P_TIME_STREAM) or "prop19"

        collected = int(props.get(_P_COLLECTED, 0))
        row_bytes = 8 * self.n_points
        if self.n_points <= 0 or self.step == 0 or not self._cf.has_stream(self._data):
            collected = 0
        else:
            stored = self._cf.stream_size(self._data) // row_bytes
            if self._cf.has_stream(self._time):
                stored = min(stored, self._cf.stream_size(self._time) // 8)
            else:
                stored = 0
            collected = max(min(collected, stored), 0)
        self.n_spectra = collected
        self._row_bytes = row_bytes

    def require_spectra(self):
        if self.n_spectra == 0:
            raise ValueError(f"{self.name} não contém espectros gravados.")

    def wavelengths(self):
        """x axis in increasing order."""
        axis = self.start + self.step * np.arange(self.n_points)
        return axis[::-1].copy() if self.step < 0 else axis

    def times(self):
        """Seconds since the start of the acquisition, one per spectrum."""
        times = np.frombuffer(self._cf.read(self._time, 0, 8 * self.n_spectra), dtype="<f8").astype(float)
        times[times < 0] += _CLOCK_WRAP_SECONDS
        return times

    def spectra(self, indices=None):
        """Spectra as rows, x axis in increasing order. `indices` selects rows (0-based)."""
        if indices is None:
            block = self._cf.read(self._data, 0, self._row_bytes * self.n_spectra)
            X = np.frombuffer(block, dtype="<f8").reshape(self.n_spectra, self.n_points)
        else:
            X = np.empty((len(indices), self.n_points))
            for row, index in enumerate(indices):
                X[row] = np.frombuffer(self._cf.read(self._data, self._row_bytes * int(index), self._row_bytes), dtype="<f8")
        if self.step < 0:
            X = X[:, ::-1]
        return np.ascontiguousarray(X, dtype=float)


def spp_info(path):
    """What a .spp file holds, without loading the spectra (fast even for large files)."""
    with _SppFile(path) as spp:
        has_data = spp.n_spectra > 0
        axis = spp.wavelengths() if has_data else np.array([])
        return {
            "n_spectra": spp.n_spectra,
            "n_allocated": spp.n_allocated,
            "n_points": spp.n_points if has_data else 0,
            "x_first": float(axis[0]) if has_data else None,
            "x_last": float(axis[-1]) if has_data else None,
            "x_step": abs(spp.step) if has_data else None,
            "x_unit": spp.x_unit,
            "y_unit": spp.y_unit,
            "time_unit": spp.time_unit,
            "times": spp.times() if has_data else np.array([]),
        }


def read_spp(path, indices=None, group=None):
    """
    Read one .spp file into a Dataset (no reference values).

    `indices` loads only the given spectra (0-based positions in the file),
    which avoids reading hundreds of megabytes when only a few are needed.
    """
    with _SppFile(path) as spp:
        spp.require_spectra()
        if indices is not None:
            indices = np.atleast_1d(np.asarray(indices))
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)
            indices = indices.astype(int)
            if indices.size and (indices.min() < 0 or indices.max() >= spp.n_spectra):
                raise ValueError(f"{spp.name} tem {spp.n_spectra} espectros; as posições pedidas vão além disso.")
        positions = np.arange(spp.n_spectra) if indices is None else indices
        label = group if group is not None else os.path.splitext(spp.name)[0]
        return Dataset(
            X=spp.spectra(indices),
            wavelengths=spp.wavelengths(),
            sample_ids=[f"{label}_{i + 1}" for i in positions],
            groups=np.array([label] * positions.size),
            times=spp.times()[positions],
        )


def read_spp_with_reference(path, ref_times_min, Y, analytes=None, units="",
                            mode="exact", radius_min=None, max_distance_min=None, group=None):
    """
    Read from a .spp file only the spectra that go with a table of measured values.

    `ref_times_min` are the sampling times in minutes since the first spectrum;
    `mode` is "exact" (closest spectrum) or "window" (mean of the spectra within
    `radius_min` minutes). Returns `(dataset, match)` as `api.pair_with_reference`.
    """
    with _SppFile(path) as spp:
        spp.require_spectra()
        match = match_times(spp.times(), ref_times_min, mode, radius_min, max_distance_min)
        used = np.unique(np.concatenate(match.indices)).astype(int) if match.matched.any() else np.array([], dtype=int)
        loaded = spp.spectra(used) if used.size else np.empty((0, spp.n_points))
        label = group if group is not None else os.path.splitext(spp.name)[0]
        data = dataset_from_match(
            match, lambda idx: loaded[np.searchsorted(used, idx)], spp.wavelengths(), Y, analytes, units, label,
        )
    return data, match


def read_spp_files(paths):
    """Read several .spp files (one experiment each) into a single Dataset."""
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not paths:
        raise ValueError("Nenhum arquivo foi informado.")
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    parts = []
    for path, stem in zip(paths, stems):
        # Files are often all called "Untitled": tell them apart by their folder.
        label = stem if stems.count(stem) == 1 else f"{os.path.basename(os.path.dirname(os.path.abspath(path)))}/{stem}"
        parts.append(read_spp(path, group=label))
    return Dataset.concat(parts)


@register(IMPORTERS)
class SppImporter(Importer):
    id = "spp"
    label = "Série temporal do espectrômetro (.spp)"
    extensions = (".spp",)

    def read(self, sources, **options):
        return read_spp_files(sources, **options)
