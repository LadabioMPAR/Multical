"""Everything needed to repeat a calibration, storable as JSON."""
import json
from dataclasses import dataclass, field
from typing import List, Optional

from .params import resolve_params
from .recipe import normalize_recipe
from .registry import MODELS, SELECTORS, SPLITTERS

FORMAT_VERSION = 2     # 2: variable selection; written only when one is chosen, so older programs still read the rest
SELECTION_RULES = ("ftest", "haaland", "min")


def default_pretreatment():
    """Recipe used when none is chosen: Savitzky-Golay, 15-point window, 2nd degree, 1st derivative."""
    return [{"id": "sg", "params": {"radius": 7, "poly_order": 2, "deriv": 1}}]


@dataclass
class RunConfig:
    # [{"id": ..., "params": {...}}]; pass [] to use the spectra as they are
    pretreatment: List[dict] = field(default_factory=default_pretreatment)
    model: dict = field(default_factory=lambda: {"id": "pls", "params": {}})
    validation: dict = field(default_factory=lambda: {"id": "kfold", "params": {}})
    selection: str = "ftest"          # "ftest": Osten F-test; "haaland": Haaland & Thomas F-test; "min": lowest validation error
    seed: Optional[int] = 0           # seed for schemes that draw samples at random
    analytes: Optional[List[str]] = None   # None: every analyte in the data
    # {"id": ..., "params": {...}}: choose, per analyte, which points of the spectrum the model uses; None: all of them
    variable_selection: Optional[dict] = None

    def resolved(self):
        """Validated copy with every parameter filled in (defaults included)."""
        if self.selection not in SELECTION_RULES:
            raise ValueError(f"Regra de escolha desconhecida: {self.selection!r}. Use 'ftest', 'haaland' ou 'min'.")
        model = MODELS.get(self.model["id"])
        splitter = SPLITTERS.get(self.validation["id"])
        chosen = None
        if self.variable_selection:
            selector = SELECTORS.get(self.variable_selection["id"])
            chosen = {"id": selector.id,
                      "params": resolve_params(selector.params, self.variable_selection.get("params"), owner=selector.label)}
        return RunConfig(
            pretreatment=normalize_recipe(self.pretreatment),
            model={"id": model.id, "params": resolve_params(model.params, self.model.get("params"), owner=model.label)},
            validation={
                "id": splitter.id,
                "params": resolve_params(splitter.params, self.validation.get("params"), owner=splitter.label),
            },
            selection=self.selection,
            seed=None if self.seed is None else int(self.seed),
            analytes=None if self.analytes is None else [str(a) for a in self.analytes],
            variable_selection=chosen,
        )

    def to_dict(self):
        out = {
            "format": 2 if self.variable_selection else 1,
            "pretreatment": self.pretreatment,
            "model": self.model,
            "validation": self.validation,
            "selection": self.selection,
            "seed": self.seed,
            "analytes": self.analytes,
        }
        if self.variable_selection:
            out["variable_selection"] = self.variable_selection
        return out

    @classmethod
    def from_dict(cls, data):
        version = data.get("format", FORMAT_VERSION)
        if version > FORMAT_VERSION:
            raise ValueError(f"Configuração no formato {version}, mais novo que o suportado ({FORMAT_VERSION}).")
        return cls(
            pretreatment=list(data.get("pretreatment", [])),
            model=dict(data.get("model", {"id": "pls", "params": {}})),
            validation=dict(data.get("validation", {"id": "kfold", "params": {}})),
            selection=data.get("selection", "ftest"),
            seed=data.get("seed", 0),
            analytes=data.get("analytes"),
            variable_selection=data.get("variable_selection") or None,
        )

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))
