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
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import RunConfig

BUNDLE_FORMAT = 1
EXTENSION = ".multical"


@dataclass
class SavedAnalyte:
    name: str
    model_state: dict
    k: Optional[int]
    rmsecv: float
    r2cv: float


@dataclass
class SavedModel:
    """What `load_model` returns; `api.predict` accepts it in place of a CalibrationResult."""
    config: RunConfig
    analytes: List[str]
    units: str
    wavelengths_in: np.ndarray
    pretreatment_states: List[dict]
    x_mean: Optional[np.ndarray]
    x_std: Optional[np.ndarray]
    per_analyte: Dict[str, SavedAnalyte]
    created: str = ""
    n_samples: int = 0


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
    manifest = {
        "format": BUNDLE_FORMAT,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "analytes": list(result.analytes),
        "units": result.units,
        "n_samples": len(result.sample_ids),
        "n_pretreatment_steps": len(result.pretreatment_states),
        "config": result.config.to_dict(),
        "validation": {
            name: {"k": res.best_k, "rmsecv": res.best_rmsecv, "r2cv": res.best_r2cv}
            for name, res in result.per_analyte.items()
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
        per_analyte[name] = SavedAnalyte(name, _unpack(f"model{j}", arrays), info["k"], info["rmsecv"], info["r2cv"])
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
    )
