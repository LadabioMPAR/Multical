"""
Example of a Multical plugin: one extra pretreatment.

How to add a method of your own:
1. copy this file with another name into the plugin folder;
2. change the function and the description in the decorator;
3. open Multical again. The method appears in the list of pretreatments
   ("Avançado" mode), with its form, and is stored in the model file.

Models and validation schemes are added the same way, with a class and
`@register(MODELS)` or `@register(SPLITTERS)`; see src/multical/api/models.py
and src/multical/api/splitters.py for the built-in ones.
"""
import numpy as np

from multical.api import Param, pretreatment


@pretreatment(
    id="log10", label="Logaritmo do sinal (exemplo de plugin)", group="Transformações",
    params=[Param("offset", "Deslocamento somado antes do logaritmo", "float", default=1.0, min=0.0,
                  help="Evita o logaritmo de zero ou de números negativos: o menor valor do espectro vira este número.")],
)
def log10(X, wavelengths, offset):
    shifted = X - X.min(axis=1, keepdims=True) + offset
    if np.any(shifted <= 0):
        raise ValueError("O deslocamento precisa ser maior que zero para calcular o logaritmo.")
    return np.log10(shifted), wavelengths
