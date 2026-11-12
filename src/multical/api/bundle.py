"""
Model file (.multical): everything needed to predict new samples.

A zip with `manifest.json` (readable description: analytes, unit, the recipe,
the model and its validation error) and `arrays.npz` (numbers: spectral grid,
state of each pretreatment step, scaling of X, fitted model per analyte).
Only numbers are stored, never pickled objects, so the file stays readable
across versions and is safe to open.
"""
import datetime
import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import RunConfig
from .recipe import normalize_recipe
from .registry import PRETREATMENTS, Pretreatment

BUNDLE_FORMAT = 1
EXTENSION = ".multical"


@dataclass
class SavedAnalyte:
    name: str
    model_state: dict
    k: Optional[int]
    rmsecv: float
    r2cv: float
    y_min: Optional[float] = None     # range of the measured values used in the calibration
    y_max: Optional[float] = None


@dataclass
class SavedModel:
    """What `load_model` returns; `api.predict` accepts it in place of a CalibrationResult."""
    config: RunConfig
    analytes: List[str]
    units: str
    wavelengths_in: Optional[np.ndarray]     # grid expected from new spectra (None: unknown, old model files)
    pretreatment_states: List[dict]
    x_mean: Optional[np.ndarray]
    x_std: Optional[np.ndarray]
    per_analyte: Dict[str, SavedAnalyte]
    created: str = ""
    n_samples: int = 0
    domain: Optional[dict] = None             # see domain.py; None: the file has no such summary
    wavelengths_pre: Optional[np.ndarray] = None   # grid after pretreatment, when the input grid is unknown


def _pack(prefix, state, arrays):
    for key, value in state.items():
        value = np.asarray(value)
        if value.dtype.kind not in "biuf":
            raise ValueError(f"O estado '{key}' não é numérico e não pode ser guardado no arquivo de modelo.")
        arrays[f"{prefix}__{key}"] = value


def _unpack(prefix, arrays):
    state = {}
    for name in arrays.files:
        if name.startswith(prefix + "__"):
            value = arrays[name]
            state[name[len(prefix) + 2:]] = value.item() if value.ndim == 0 else value
    return state


def save_model(result, path):
    """Write the model of a CalibrationResult to `path` (adds .multical when missing)."""
    path = str(path)
    if not path.lower().endswith(EXTENSION):
        path += EXTENSION
    arrays = {"wavelengths_in": np.asarray(result.wavelengths_in, dtype=float)}
    if result.x_mean is not None:
        arrays["x_mean"], arrays["x_std"] = np.asarray(result.x_mean), np.asarray(result.x_std)
    for i, state in enumerate(result.pretreatment_states):
        _pack(f"pre{i}", state, arrays)
    for j, name in enumerate(result.analytes):
        _pack(f"model{j}", result.per_analyte[name].model_state, arrays)
    if getattr(result, "domain", None):
        _pack("domain", result.domain, arrays)
    manifest = {
        "format": BUNDLE_FORMAT,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "analytes": list(result.analytes),
        "units": result.units,
        "n_samples": len(result.sample_ids),
        "n_pretreatment_steps": len(result.pretreatment_states),
        "config": result.config.to_dict(),
        "validation": {
            name: {"k": res.best_k, "rmsecv": res.best_rmsecv, "r2cv": res.best_r2cv,
                   "y_min": float(np.min(result.Y[:, j])), "y_max": float(np.max(result.Y[:, j]))}
            for j, (name, res) in enumerate(result.per_analyte.items())
        },
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        archive.writestr("arrays.npz", buffer.getvalue())
    return path


def load_model(path):
    """Read a .multical file written by `save_model`."""
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            arrays = np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False)
    except (zipfile.BadZipFile, KeyError, ValueError) as error:
        raise ValueError(f"O arquivo não é um modelo do Multical válido ({error}).") from None
    if manifest.get("format", 0) > BUNDLE_FORMAT:
        raise ValueError("Este modelo foi criado por uma versão mais nova do Multical.")
    analytes = list(manifest["analytes"])
    per_analyte = {}
    for j, name in enumerate(analytes):
        info = manifest["validation"][name]
        per_analyte[name] = SavedAnalyte(name, _unpack(f"model{j}", arrays), info["k"], info["rmsecv"], info["r2cv"],
                                         info.get("y_min"), info.get("y_max"))
    return SavedModel(
        config=RunConfig.from_dict(manifest["config"]),
        analytes=analytes,
        units=manifest.get("units", ""),
        wavelengths_in=arrays["wavelengths_in"],
        pretreatment_states=[_unpack(f"pre{i}", arrays) for i in range(manifest["n_pretreatment_steps"])],
        x_mean=arrays["x_mean"] if "x_mean" in arrays.files else None,
        x_std=arrays["x_std"] if "x_std" in arrays.files else None,
        per_analyte=per_analyte,
        created=manifest.get("created", ""),
        n_samples=manifest.get("n_samples", 0),
        domain=_unpack("domain", arrays) or None,
    )


class _NumbersOnly(pickle.Unpickler):
    """Reads pickles made only of numbers, lists, dicts and numpy arrays; anything else is refused."""
    _ALLOWED = {"_reconstruct", "_frombuffer", "ndarray", "dtype", "scalar"}

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" and name in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"conteúdo não permitido ({module}.{name})")


def load_legacy_pls(path, recipe, analytes=None, units=""):
    """
    Read a PLS model saved by the scripts (`train_and_save_model_pls`, a .pkl file).

    Those files do not store the pretreatment, so the `recipe` used when the
    model was made must be given; only steps that keep no state can be used.
    """
    try:
        with open(path, "rb") as handle:
            data = _NumbersOnly(handle).load()
        if data["type"] != "PLS":
            raise ValueError("o tipo do modelo não é PLS")
        entries = list(data["analytes"])
        grid = np.asarray(data["wavelengths"], dtype=float).ravel()
    except (pickle.UnpicklingError, KeyError, TypeError, EOFError, ValueError, AttributeError, ImportError) as error:
        raise ValueError(f"O arquivo não é um modelo PLS antigo do Multical ({error}).") from None
    steps = normalize_recipe(recipe)
    for step in steps:
        item = PRETREATMENTS.get(step["id"])
        if type(item).fit is not Pretreatment.fit:
            raise ValueError(f"O passo '{item.label}' depende dos dados de calibração, que o modelo antigo não guarda.")
    names = list(analytes) if analytes else [f"analito_{i + 1}" for i in range(len(entries))]
    if len(names) != len(entries):
        raise ValueError(f"O modelo tem {len(entries)} analito(s); foram dados {len(names)} nomes.")
    per_analyte = {}
    for name, entry in zip(names, entries):
        state = {"B": np.asarray(entry["B_pls"], dtype=float), "y_mean": float(np.ravel(entry["Ymed"])[0]),
                 "y_std": float(np.ravel(entry["Ysig"])[0]), "k": int(entry["k"])}
        # Only a single number is trusted as this analyte's validation error: run_calibration.py stores a
        # row of the whole error table here (errors of every analyte for one k), which is not that.
        rmsecv = np.ravel(entry.get("rmsecv")) if entry.get("rmsecv") is not None else np.array([])
        error = float(rmsecv[0]) if rmsecv.size == 1 else float("nan")
        per_analyte[name] = SavedAnalyte(name, state, int(entry["k"]), error, float("nan"))
    return SavedModel(
        config=RunConfig(pretreatment=steps), analytes=names, units=units, wavelengths_in=None,
        pretreatment_states=[{} for _ in steps],
        x_mean=np.asarray(entries[0]["Xmed"], dtype=float).ravel(), x_std=np.asarray(entries[0]["Xsig"], dtype=float).ravel(),
        per_analyte=per_analyte, wavelengths_pre=grid,
    )
