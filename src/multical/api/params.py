"""
Parameter descriptions.

Every method (pretreatment, model, validation scheme) declares its parameters as
a list of `Param`. The GUI builds its forms from these descriptions, and the
same descriptions validate values coming from a saved configuration.
"""
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

KINDS = ("int", "float", "bool", "choice", "ranges")


@dataclass(frozen=True)
class Param:
    name: str                      # keyword passed to the method
    label: str                     # text shown next to the field
    kind: str                      # one of KINDS
    default: Any
    min: Optional[float] = None    # inclusive lower bound (int/float)
    max: Optional[float] = None    # inclusive upper bound (int/float)
    choices: Optional[Sequence] = None
    help: str = ""                 # plain-language explanation
    advanced: bool = False         # True: shown only in "Avançado" mode

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Param {self.name!r}: tipo desconhecido {self.kind!r}")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"Param {self.name!r}: 'choice' exige a lista 'choices'")

    def coerce(self, value):
        """Return `value` converted to this parameter's type, or raise ValueError."""
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            raise ValueError(f"{self.label}: esperado verdadeiro ou falso, recebido {value!r}.")

        if self.kind == "choice":
            if value in self.choices:
                return value
            options = ", ".join(str(c) for c in self.choices)
            raise ValueError(f"{self.label}: {value!r} não é uma opção válida ({options}).")

        if self.kind == "ranges":
            return self._coerce_ranges(value)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.label}: esperado um número, recebido {value!r}.")
        if not math.isfinite(value):
            raise ValueError(f"{self.label}: o valor precisa ser um número finito.")
        if self.kind == "int":
            if float(value) != int(value):
                raise ValueError(f"{self.label}: esperado um número inteiro, recebido {value!r}.")
            value = int(value)
        else:
            value = float(value)
        if self.min is not None and value < self.min:
            raise ValueError(f"{self.label}: {value} está abaixo do mínimo permitido ({self.min}).")
        if self.max is not None and value > self.max:
            raise ValueError(f"{self.label}: {value} está acima do máximo permitido ({self.max}).")
        return value

    def _coerce_ranges(self, value):
        try:
            pairs = [(float(lo), float(hi)) for lo, hi in value]
        except (TypeError, ValueError):
            raise ValueError(f"{self.label}: esperado uma lista de faixas (início, fim).") from None
        for lo, hi in pairs:
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
                raise ValueError(f"{self.label}: a faixa {lo}–{hi} é inválida (o início deve ser menor que o fim).")
        return [[lo, hi] for lo, hi in pairs]


def resolve_params(specs, given=None, owner=""):
    """Fill defaults and validate `given` against `specs`. Returns a new dict."""
    given = dict(given or {})
    known = {p.name for p in specs}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"{owner or 'Método'}: parâmetro(s) desconhecido(s): {', '.join(unknown)}.")
    return {p.name: p.coerce(given.get(p.name, p.default)) for p in specs}
