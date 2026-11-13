"""
Modelo de plugin: pré-tratamento COM estado (uma classe com fit e apply).

Use esta forma quando a correção depende do conjunto de calibração (uma média,
um espectro de referência...). `fit` aprende com os espectros de calibração e
devolve o que precisa ser lembrado; `apply` usa isso em quaisquer espectros,
inclusive nas amostras novas estimadas meses depois com o modelo salvo.
Para ativar, copie este arquivo para a pasta "plugins".
"""
import numpy as np

from multical.api import PRETREATMENTS, Param, Pretreatment, register


@register(PRETREATMENTS)
class SubtrairMedia(Pretreatment):
    id = "ex_subtrair_media"
    label = "Subtrair o espectro médio da calibração (exemplo)"
    group = "Exemplos"
    params = (
        Param("dividir", "Dividir também pelo desvio padrão", "bool", default=False,
              help="Deixa todos os pontos do espectro com o mesmo peso."),
    )

    def fit(self, X, wavelengths, dividir):
        # Chamado uma vez, com todos os espectros de calibração.
        # O estado é um dicionário só com números e matrizes de números: ele vai para o arquivo de modelo.
        desvio = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
        return {"media": X.mean(axis=0), "desvio": np.where(desvio == 0, 1.0, desvio)}

    def apply(self, X, wavelengths, state, dividir):
        # Chamado com os espectros de calibração e, depois, com espectros novos.
        # Nunca recalcule a média aqui: use a que veio em `state`.
        corrigido = X - state["media"]
        if dividir:
            corrigido = corrigido / state["desvio"]
        return corrigido, wavelengths
