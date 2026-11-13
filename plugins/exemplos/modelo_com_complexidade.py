"""
Modelo de plugin: modelo COM um número de complexidade (como as variáveis
latentes do PLS). O usuário informa o máximo; o programa testa de 1 até o
máximo e escolhe o melhor. Para ativar, copie este arquivo para a pasta "plugins".
"""
import numpy as np

from multical.api import MODELS, Model, Param, register


def _distancias(A, B):
    quadrados = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(quadrados, 0.0))


@register(MODELS)
class Vizinhos(Model):
    id = "ex_vizinhos"
    label = "Vizinhos mais próximos (exemplo)"
    scale_x = True
    sweep = "max_k"                               # nome do parâmetro que dá o máximo a testar
    sweep_label = ("vizinho", "vizinhos")         # o que o número conta (singular, plural), para os textos da tela
    params = (
        Param("max_k", "Máximo de vizinhos", "int", default=10, min=1, advanced=True,
              help="O programa testa de 1 até este número e escolhe o melhor."),
    )

    def fit(self, X, y, k):
        # Atenção: com `sweep`, fit NÃO recebe max_k; recebe `k`, a complexidade a usar.
        return {"X": np.array(X, dtype=float), "y": np.asarray(y, dtype=float), "k": int(k)}

    def predict(self, state, X):
        k = int(state["k"])
        mais_proximos = np.argsort(_distancias(X, state["X"]), axis=1)[:, :k]
        return state["y"][mais_proximos].mean(axis=1)

    def cv_path(self, X_train, y_train, X_val, max_k):
        # Opcional. Devolve as estimativas para todas as complexidades de uma vez:
        # matriz (max_k, amostras de validação), linha 0 = complexidade 1.
        # Sem este método o programa chama fit e predict uma vez para cada k: mesmo resultado, mais devagar.
        ordem = np.argsort(_distancias(X_val, X_train), axis=1)[:, :max_k]
        soma = np.cumsum(np.asarray(y_train, dtype=float)[ordem], axis=1)
        return (soma / np.arange(1, max_k + 1)).T
