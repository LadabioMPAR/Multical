"""
Modelo de plugin: modelo que deixa o programa ESCOLHER um ajuste pela validação.

`candidates` devolve a lista de ajustes a testar; o programa treina e valida
cada um e fica com o de menor erro. Para ativar, copie este arquivo para a
pasta "plugins".
"""
import numpy as np

from multical.api import MODELS, Model, Param, register


@register(MODELS)
class RidgeAutomatico(Model):
    id = "ex_ridge_auto"
    label = "Regressão ridge com penalidade escolhida pela validação (exemplo)"
    scale_x = True
    params = (
        Param("escolher", "Escolher a penalidade automaticamente", "bool", default=True),
        Param("alfa", "Penalidade (alfa)", "float", default=1.0, min=1e-6, advanced=True,
              help="Usada quando a escolha automática está desligada."),
    )

    def candidates(self, escolher, alfa):
        # Recebe todos os parâmetros. Devolve None quando há um único ajuste, ou uma lista de
        # (descrição mostrada ao usuário, {parâmetro: valor}) com os ajustes a testar.
        if not escolher:
            return None
        return [(f"alfa={valor:g}".replace(".", ","), {"alfa": valor}) for valor in (0.01, 0.1, 1.0, 10.0, 100.0)]

    def fit(self, X, y, escolher, alfa):
        # fit recebe todos os parâmetros, com os valores do ajuste que está sendo testado.
        y = np.asarray(y, dtype=float)
        y_media = float(y.mean())
        pesos = np.linalg.solve(X @ X.T + alfa * np.eye(X.shape[0]), y - y_media)
        return {"coeficientes": X.T @ pesos, "y_media": y_media}

    def predict(self, state, X):
        return X @ state["coeficientes"] + state["y_media"]
