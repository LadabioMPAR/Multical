"""
The state shared by the screens: which experiments were imported, which samples
are in use, the configuration and the last result. No Qt in here.

A project can be saved to a small text file (.mcproj, JSON) and opened again.
The file keeps what the user chose (which files, how samples are linked, the
names of the analytes, the samples taken out, the recipe, the model and the
validation); the spectra themselves stay in their own files and are read again
when the project is opened. A trained model is not kept in the project: export
it, or train again after opening.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from multical import api
from multical.io.reference import read_reference_table
from multical.io.spp import read_spp, read_spp_with_reference, spp_info
from multical.io.txt_pair import read_txt_spectra

PROJECT_FORMAT = 1
PROJECT_EXTENSION = ".mcproj"
LINKED_BY = {"time": "ligado pelo tempo", "order": "ligado pela ordem das linhas"}


@dataclass
class Experiment:
    name: str
    spectra_path: str
    reference_path: str
    kind: str                       # "spp": instrument time series; "txt": text file with one spectrum per row
    n_spectra: int = 0              # spectra in the file
    n_linked: int = 0               # samples with measured values and a spectrum
    linked_by: str = ""             # "time" or "order" (see LINKED_BY); "" while the experiment is not usable
    error: str = ""                 # why the experiment could not be used ("" when it is fine)
    note: str = ""                  # something the user should know about an experiment that is in use
    problems: List[str] = field(default_factory=list)   # samples left without a spectrum
    header: Optional[List[str]] = None                  # column names found in the table of measured values


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
        self.path: Optional[str] = None                 # file the project was saved to or opened from
        self._saved = None                              # what that file holds, to tell whether something changed
        self.reset()

    def reset(self):
        """Back to an empty, unnamed project."""
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
        self.path = None
        self._saved = self.to_dict()

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
        """Read one experiment and link its samples to spectra. Returns a Dataset with measured values."""
        times, Y, exp.header = read_reference_table(exp.reference_path)
        if exp.kind == "spp":
            exp.n_spectra = spp_info(exp.spectra_path)["n_spectra"]
            data, match = read_spp_with_reference(
                exp.spectra_path, times, Y, mode=self.match_mode, radius_min=self.radius_min, group=exp.name)
            exp.problems, exp.linked_by = match.problems(), "time"
            return data

        spectra = read_txt_spectra(exp.spectra_path, group=exp.name)
        exp.n_spectra = spectra.n_samples
        if spectra.n_samples == times.size:
            # One spectrum per sample, as in the files used so far: linked row by row.
            exp.linked_by = "order"
            if not np.allclose(spectra.times, times, atol=1e-6):
                exp.note = "os tempos do arquivo de espectros não coincidem com os da tabela"
            return api.Dataset(
                X=spectra.X, wavelengths=spectra.wavelengths, Y=Y,
                sample_ids=[f"{exp.name}_{t:g}min" for t in times],
                groups=np.array([exp.name] * times.size), times=times)
        # A whole series in text: linked by the time, which these files keep in minutes.
        series = api.Dataset(spectra.X, spectra.wavelengths, times=spectra.times * 60.0, groups=spectra.groups)
        data, match = api.pair_with_reference(series, times, Y, mode=self.match_mode, radius_min=self.radius_min)
        exp.problems, exp.linked_by = match.problems(), "time"
        return data

    def rebuild(self):
        """Read every experiment again and assemble the calibration set."""
        self.result = None
        parts, first = [], None
        for exp in self.experiments:
            exp.error, exp.note, exp.problems, exp.n_linked, exp.linked_by, exp.header = "", "", [], 0, "", None
            try:
                data = self._load(exp)
                if first is not None:
                    if data.Y.shape[1] != first.Y.shape[1]:
                        raise ValueError(f"A tabela tem {data.Y.shape[1]} colunas de valores medidos; "
                                         f"a do primeiro experimento tem {first.Y.shape[1]}.")
                    if data.wavelengths.size != first.wavelengths.size or not np.allclose(data.wavelengths, first.wavelengths, atol=1e-6):
                        raise ValueError("Os espectros têm uma faixa espectral diferente da do primeiro experimento.")
                    known = parts[0][0].header
                    if exp.header and known and [h.lower() for h in exp.header] != [h.lower() for h in known]:
                        raise ValueError(
                            f"As colunas da tabela ({', '.join(exp.header)}) não são as mesmas, na mesma ordem, "
                            f"da tabela do primeiro experimento ({', '.join(known)}).")
                else:
                    first = data
                exp.n_linked = data.n_samples
                parts.append((exp, data))
            except (ValueError, OSError) as error:
                exp.error, exp.linked_by = str(error), ""
        if not parts:
            self.dataset = None
            return
        count = first.Y.shape[1]
        if len(self.analyte_names) != count:
            header = next((exp.header for exp, _ in parts if exp.header), None)
            self.analyte_names = list(header) if header and len(set(header)) == count else [f"analito_{i + 1}" for i in range(count)]
        for _, part in parts:
            part.analytes, part.units = list(self.analyte_names), self.units
        self.dataset = api.Dataset.concat([part for _, part in parts])
        self.excluded &= set(self.dataset.sample_ids)
        if self.config.analytes is not None:
            self.config.analytes = [a for a in self.config.analytes if a in self.analyte_names] or None

    # ---- analytes --------------------------------------------------------
    def analyte_summary(self):
        """One dict per analyte: column of the table, name and range of the measured values in use."""
        if self.dataset is None:
            return []
        Y = self.dataset.Y
        return [{"column": j + 2, "name": name, "low": float(np.nanmin(Y[:, j])), "high": float(np.nanmax(Y[:, j]))}
                for j, name in enumerate(self.dataset.analytes)]

    def set_analyte_names(self, names):
        names = [n.strip() for n in names]
        if (self.dataset is None or len(names) != len(self.dataset.analytes)
                or not all(names) or len({n.lower() for n in names}) != len(names)):
            return False
        if names != self.analyte_names:
            chosen = self.config.analytes
            if chosen is not None:      # keep the same analytes selected, under their new names
                self.config.analytes = [new for old, new in zip(self.analyte_names, names) if old in chosen] or None
            self.analyte_names = names
            self.dataset.analytes = list(names)
            self.result = None
        return True

    def set_analyte_name(self, index, name):
        names = list(self.analyte_names)
        if not 0 <= index < len(names):
            return False
        names[index] = name
        return self.set_analyte_names(names)

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

    # ---- saving and opening ----------------------------------------------
    def to_dict(self, folder=None):
        """
        What the user chose, as plain data. With `folder` (where the project file
        goes), each data file is also written relative to it, so the project still
        opens after its folder is moved or copied together with the data.
        """
        def location(path):
            entry = {"path": os.path.abspath(path)}
            if folder:
                try:
                    entry["relative"] = os.path.relpath(entry["path"], folder).replace("\\", "/")
                except ValueError:      # another drive: there is no relative path
                    pass
            return entry

        return {
            "format": PROJECT_FORMAT,
            "experiments": [{"name": e.name, "spectra": location(e.spectra_path), "reference": location(e.reference_path)}
                            for e in self.experiments],
            "matching": {"mode": self.match_mode, "radius_min": self.radius_min},
            "analytes": list(self.analyte_names),
            "units": self.units,
            "excluded": sorted(self.excluded),
            "config": self.config.to_dict(),
            "advanced": bool(self.advanced),
        }

    @property
    def dirty(self):
        """True when something chosen by the user is not in the project file yet."""
        return self.to_dict() != self._saved

    @property
    def is_empty(self):
        return not self.experiments

    def save(self, path):
        path = str(path)
        if not path.lower().endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        folder = os.path.dirname(os.path.abspath(path))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(folder), handle, indent=2, ensure_ascii=False)
        self.path = path
        self._saved = self.to_dict()
        return path

    def open(self, path):
        """Replace this project with the one in `path` and read its data files again."""
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or "experiments" not in data:
                raise ValueError("não tem a lista de experimentos")
            if data.get("format", 0) > PROJECT_FORMAT:
                raise ValueError("foi salvo por uma versão mais nova do Multical")
            folder = os.path.dirname(os.path.abspath(path))

            def resolve(entry):
                relative = entry.get("relative")
                if relative and os.path.isfile(os.path.join(folder, relative)):
                    return os.path.normpath(os.path.join(folder, relative))
                return entry["path"]

            experiments = []
            for item in data["experiments"]:
                spectra = resolve(item["spectra"])
                kind = "spp" if spectra.lower().endswith(".spp") else "txt"
                experiments.append(Experiment(str(item["name"]), spectra, resolve(item["reference"]), kind))
            config = api.RunConfig.from_dict(data.get("config", {}))
            config.resolved()       # refuses methods that are not installed (for instance a missing plugin)
            matching = data.get("matching", {})
            mode = matching.get("mode", "exact")
            if mode not in api.MATCH_MODES:
                raise ValueError(f"modo de ligação desconhecido: {mode!r}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"O arquivo não pôde ser aberto como projeto do Multical ({error}).") from None

        self.reset()
        self.experiments = experiments
        self.match_mode, self.radius_min = mode, float(matching.get("radius_min", 5.0))
        self.analyte_names = [str(name) for name in data.get("analytes", [])]
        self.units = str(data.get("units", "g/L"))
        self.config = config
        self.advanced = bool(data.get("advanced", False))
        self.excluded = set(data.get("excluded", []))
        kept = set(self.excluded)
        self.rebuild()
        if self.dataset is None:
            self.excluded = kept    # the data files are missing for now: do not forget the user's choices
        self.path = str(path)
        self._saved = self.to_dict()
        return self
