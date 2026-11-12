"""
The state shared by the screens: which experiments were imported, which samples
are in use, the configuration and the last result. No Qt in here.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from multical import api
from multical.io.reference import read_reference_table
from multical.io.spp import read_spp, read_spp_with_reference, spp_info
from multical.io.txt_pair import read_txt_pairs, read_txt_spectra


@dataclass
class Experiment:
    name: str
    spectra_path: str
    reference_path: str
    kind: str                       # "spp": time series linked by time; "txt": text pair linked row by row
    n_spectra: int = 0              # spectra in the file
    n_linked: int = 0               # samples with measured values and a spectrum
    error: str = ""                 # why the experiment could not be used ("" when it is fine)
    problems: List[str] = field(default_factory=list)   # samples left without a spectrum


def default_experiment_name(spectra_path):
    stem = os.path.splitext(os.path.basename(spectra_path))[0]
    if stem.lower() in ("untitled", "sem título", "sem titulo"):
        stem = os.path.basename(os.path.dirname(os.path.abspath(spectra_path))) or stem
    return stem


def load_spectra(path, every=1, taken=()):
    """
    Spectra of one file for estimating (no measured values), one in every `every`.
    Times are returned in minutes whatever the file type; `taken` are experiment
    names already in use, so two files called "Untitled" stay apart.
    """
    name = default_experiment_name(path)
    base, n = name, 2
    while name in taken:
        name, n = f"{base} ({n})", n + 1
    every = max(int(every), 1)
    if str(path).lower().endswith(".spp"):
        count = spp_info(path)["n_spectra"]
        data = read_spp(path, indices=np.arange(0, count, every) if every > 1 else None, group=name)
        data.times = data.times / 60.0
        data.sample_ids = [f"{name}_{t:.1f}min".replace(".", ",") for t in data.times]
        return data
    data = read_txt_spectra(path, group=name)
    return data.subset(np.arange(0, data.n_samples, every)) if every > 1 else data


class Project:
    def __init__(self):
        self.experiments: List[Experiment] = []
        self.match_mode = "exact"
        self.radius_min = 5.0
        self.analyte_names: List[str] = []
        self.units = "g/L"
        self.dataset: Optional[api.Dataset] = None      # every linked sample
        self.excluded = set()                           # sample ids taken out by the user
        self.config = api.RunConfig()
        self.result: Optional[api.CalibrationResult] = None
        self.advanced = False

    # ---- import ----------------------------------------------------------
    def add_experiment(self, spectra_path, reference_path, name=None):
        kind = "spp" if str(spectra_path).lower().endswith(".spp") else "txt"
        name = (name or "").strip() or default_experiment_name(spectra_path)
        taken = {e.name for e in self.experiments}
        base, n = name, 2
        while name in taken:
            name, n = f"{base} ({n})", n + 1
        self.experiments.append(Experiment(name, str(spectra_path), str(reference_path), kind))
        self.rebuild()

    def remove_experiment(self, index):
        del self.experiments[index]
        self.rebuild()

    def _load(self, exp):
        if exp.kind == "spp":
            exp.n_spectra = spp_info(exp.spectra_path)["n_spectra"]
            times, Y, header = read_reference_table(exp.reference_path)
            data, match = read_spp_with_reference(
                exp.spectra_path, times, Y, mode=self.match_mode, radius_min=self.radius_min, group=exp.name)
            exp.problems = match.problems()
            return data, header
        pair = read_txt_pairs([(exp.reference_path, exp.spectra_path)])
        exp.n_spectra = pair.n_samples
        data = api.Dataset(
            X=pair.X, wavelengths=pair.wavelengths, Y=pair.Y,
            sample_ids=[f"{exp.name}_{t:g}min" for t in pair.times],
            groups=np.array([exp.name] * pair.n_samples), times=pair.times)
        return data, None

    def rebuild(self):
        """Read every experiment again and assemble the calibration set."""
        self.result = None
        parts, header = [], None
        for exp in self.experiments:
            exp.error, exp.problems, exp.n_linked = "", [], 0
            try:
                data, names = self._load(exp)
                if parts:
                    first = parts[0]
                    if data.Y.shape[1] != first.Y.shape[1]:
                        raise ValueError(f"A tabela tem {data.Y.shape[1]} colunas de valores medidos; "
                                         f"a do primeiro experimento tem {first.Y.shape[1]}.")
                    if data.wavelengths.size != first.wavelengths.size or not np.allclose(data.wavelengths, first.wavelengths, atol=1e-6):
                        raise ValueError("Os espectros têm uma faixa espectral diferente da do primeiro experimento.")
                elif names:
                    header = names
                exp.n_linked = data.n_samples
                parts.append(data)
            except (ValueError, OSError) as error:
                exp.error = str(error)
        if not parts:
            self.dataset = None
            return
        count = parts[0].Y.shape[1]
        if len(self.analyte_names) != count:
            self.analyte_names = list(header) if header and len(header) == count else [f"analito_{i + 1}" for i in range(count)]
        for part in parts:
            part.analytes, part.units = list(self.analyte_names), self.units
        self.dataset = api.Dataset.concat(parts)
        self.excluded &= set(self.dataset.sample_ids)
        if self.config.analytes is not None:
            self.config.analytes = [a for a in self.config.analytes if a in self.analyte_names] or None

    def set_analyte_names(self, names):
        names = [n.strip() for n in names if n.strip()]
        if self.dataset is None or len(names) != len(self.dataset.analytes) or len(set(names)) != len(names):
            return False
        self.analyte_names = names
        self.dataset.analytes = list(names)
        self.config.analytes = None
        self.result = None
        return True

    def set_units(self, units):
        self.units = units.strip()
        if self.dataset is not None:
            self.dataset.units = self.units
        self.result = None

    # ---- samples in use --------------------------------------------------
    def active_dataset(self):
        """The imported samples minus the ones the user took out."""
        if self.dataset is None:
            return None
        keep = np.array([s not in self.excluded for s in self.dataset.sample_ids])
        return self.dataset.subset(keep)

    def set_excluded(self, sample_ids):
        sample_ids = set(sample_ids)
        if sample_ids != self.excluded:
            self.excluded = sample_ids
            self.result = None

    # ---- configuration ---------------------------------------------------
    def set_recipe(self, steps):
        self.config.pretreatment = [dict(id=s["id"], params=dict(s.get("params", {}))) for s in steps]
        self.result = None

    def largest_complexity(self):
        """Highest number of latent variables the data and the validation scheme allow (None: no limit applies)."""
        data = self.active_dataset()
        config = self.config.resolved()
        model = api.MODELS.get(config.model["id"])
        if data is None or not model.sweep:
            return None
        splitter = api.SPLITTERS.get(config.validation["id"])
        rng = np.random.RandomState(config.seed)
        folds = list(splitter.split(data.n_samples, groups=data.groups, X=data.X, Y=data.Y, rng=rng, **config.validation["params"]))
        if not folds:
            return None
        return max(min(min(len(train) for train, _ in folds) - 1, data.X.shape[1]), 0)
