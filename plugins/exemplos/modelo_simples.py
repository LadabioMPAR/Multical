"""
Modelo de plugin: modelo SEM número de complexidade (fit e predict).

O programa cria um modelo por analito. `fit` recebe os espectros de treino e os
valores medidos de UM analito e devolve o estado; `predict` usa o estado para
estimar. Para ativar, copie este arquivo para a pasta "plugins".
"""
import numpy as np

from multical.api import MODELS, Model, Param, register


@register(MODELS)
class Ridge(Model):
    id = "ex_ridge"
    label = "Regressão ridge (exemplo)"
    scale_x = True      # pede ao programa para padronizar X (média 0, desvio 1) com o conjunto de calibração
    params = (
        Param("alfa", "Penalidade (alfa)", "float", default=1.0, min=1e-6,
              help="Valores maiores deixam o modelo mais simples e menos sensível a ruído."),
    )

    def fit(self, X, y, alfa):
        # X: (amostras de treino x pontos), já pré-tratado e padronizado.
        # y: vetor (amostras de treino,) na unidade original; nada foi feito com ele.
        y = np.asarray(y, dtype=float)
        y_media = float(y.mean())
        # Há muito mais pontos do que amostras: resolve-se o sistema pequeno (amostras x amostras).
        pesos = np.linalg.solve(X @ X.T + alfa * np.eye(X.shape[0]), y - y_media)
        return {"coeficientes": X.T @ pesos, "y_media": y_media}

    def predict(self, state, X):
        # Devolve um valor por espectro, na unidade original: vetor (amostras,).
        return X @ state["coeficientes"] + state["y_media"]
