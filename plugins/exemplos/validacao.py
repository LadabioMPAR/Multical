"""
Modelo de plugin: esquema de validação (como dividir as amostras em treino e
validação). Para ativar, copie este arquivo para a pasta "plugins".
"""
import numpy as np

from multical.api import SPLITTERS, Param, Splitter, register


@register(SPLITTERS)
class PartesPorExperimento(Splitter):
    id = "ex_partes_por_experimento"
    label = "Partes sorteadas dentro de cada experimento (exemplo)"
    kind = "cv"             # "cv": toda amostra é validada; "holdout": uma única divisão, só parte das amostras valida
    needs_groups = True     # avisa o programa de que este esquema precisa saber o experimento de cada amostra
    params = (
        Param("partes", "Número de partes", "int", default=4, min=2,
              help="Cada experimento é dividido neste número de partes, e cada parte valida uma vez."),
    )

    def split(self, n, groups=None, X=None, Y=None, rng=None, partes=4):
        # n: número de amostras; groups: experimento de cada amostra; rng: gerador de sorteios já com a semente do projeto.
        # Use sempre `rng` para sortear, senão o resultado muda a cada execução.
        rng = rng if rng is not None else np.random.RandomState(0)
        groups = np.asarray(groups)
        parte = np.empty(n, dtype=int)
        for nome in np.unique(groups):
            posicoes = rng.permutation(np.flatnonzero(groups == nome))
            parte[posicoes] = np.arange(posicoes.size) % partes
        for numero in range(partes):
            validacao = np.flatnonzero(parte == numero)
            if validacao.size:
                # Cada item devolvido é um par (posições de treino, posições de validação), contadas a partir de 0.
                yield np.flatnonzero(parte != numero), validacao
