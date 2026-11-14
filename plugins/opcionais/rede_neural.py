"""
Plugin opcional: rede neural (ANN) como modelo. Para ativar, copie este arquivo
para a pasta "plugins".

O treino é o de `multical.models.ann_torch.ANNModel` e precisa do PyTorch
(`pip install torch`), que não vem no programa instalado: treine a partir de uma
cópia do repositório. Do treino ficam guardados só números (os pesos de cada
camada e as médias e desvios usados para padronizar), e a estimativa é
calculada aqui, sem o PyTorch. Por isso um modelo de rede neural já treinado
pode ser usado em qualquer instalação que tenha este arquivo na pasta
"plugins".

Como em todo o programa, é treinada uma rede por analito.
"""
import numpy as np

from multical.api import MODELS, Model, Param, register

ATIVACOES = ("relu", "tanh", "sigmoid", "leaky_relu", "silu", "elu")


def _ativar(codigo, h):
    nome = ATIVACOES[int(codigo)]
    if nome == "relu":
        return np.maximum(h, 0.0)
    if nome == "tanh":
        return np.tanh(h)
    if nome == "sigmoid":
        return 1.0 / (1.0 + np.exp(-h))
    if nome == "leaky_relu":
        return np.where(h > 0, h, 0.01 * h)
    if nome == "silu":
        return h / (1.0 + np.exp(-h))
    return np.where(h > 0, h, np.expm1(np.minimum(h, 0.0)))      # elu


@register(MODELS)
class RedeNeural(Model):
    id = "ann"
    label = "Rede neural (treino exige PyTorch)"
    scale_x = False         # a rede padroniza X e y por conta própria, com as amostras de treino
    params = (
        Param("hidden_units", "Neurônios por camada", "int", default=50, min=1),
        Param("n_layers", "Camadas escondidas", "int", default=1, min=1),
        Param("epochs", "Épocas de treino", "int", default=200, min=1,
              help="Quantas vezes o treino passa por todas as amostras."),
        Param("activation", "Função de ativação", "choice", default="relu", choices=ATIVACOES, advanced=True),
        Param("learning_rate", "Taxa de aprendizado", "float", default=0.001, min=0.0001, advanced=True),
        Param("batch_size", "Amostras por passo do treino", "int", default=32, min=1, advanced=True),
        Param("early_stopping", "Parar quando deixar de melhorar", "bool", default=False, advanced=True,
              help="Separa 10% das amostras de treino e para quando o erro nelas deixa de cair."),
        Param("patience", "Épocas sem melhora antes de parar", "int", default=20, min=1, advanced=True),
    )

    def fit(self, X, y, hidden_units, n_layers, epochs, activation, learning_rate, batch_size, early_stopping, patience):
        try:
            import torch
            from multical.models.ann_torch import ANNModel
        except ImportError:
            raise ValueError(
                "Para treinar a rede neural é preciso o PyTorch, que não está disponível aqui (ele não vem no "
                "programa instalado). Treine a partir de uma cópia do repositório, depois de `pip install torch`. "
                "Modelos de rede neural já treinados podem ser usados para estimar amostras sem ele.") from None
        X = np.asarray(X, dtype=float)
        rede = ANNModel()
        rede.fit_predict(
            X, np.asarray(y, dtype=float).reshape(-1, 1), X[:1], hidden_units=hidden_units, n_layers=n_layers,
            activation=activation, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
            device=torch.device("cpu"), early_stopping=early_stopping, patience=patience)
        camadas = [m for m in rede.model.model if isinstance(m, torch.nn.Linear)]
        estado = {
            "camadas": len(camadas), "ativacao": ATIVACOES.index(activation),
            "x_media": np.asarray(rede.scaler_x.mean_, dtype=float), "x_desvio": np.asarray(rede.scaler_x.scale_, dtype=float),
            "y_media": float(rede.scaler_y.mean_[0]), "y_desvio": float(rede.scaler_y.scale_[0]),
        }
        for numero, camada in enumerate(camadas):
            estado[f"pesos{numero}"] = camada.weight.detach().cpu().numpy()       # (saídas, entradas)
            estado[f"vies{numero}"] = camada.bias.detach().cpu().numpy()
        return estado

    def predict(self, state, X):
        h = (np.asarray(X, dtype=float) - state["x_media"]) / state["x_desvio"]
        ultima = int(state["camadas"]) - 1
        for numero in range(ultima + 1):
            h = h @ np.asarray(state[f"pesos{numero}"], dtype=float).T + np.atleast_1d(state[f"vies{numero}"])
            if numero < ultima:
                h = _ativar(state["ativacao"], h)
        return h.ravel() * state["y_desvio"] + state["y_media"]
