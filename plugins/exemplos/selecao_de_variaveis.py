"""
Modelo de plugin: seleção de variáveis (escolher, para cada analito, quais
pontos do espectro o modelo usa). Para ativar, copie este arquivo para a pasta
"plugins".
"""
import numpy as np

from multical.api import SELECTORS, Param, Selector, register


@register(SELECTORS)
class MaisCorrelacionados(Selector):
    id = "ex_mais_correlacionados"
    label = "Pontos mais correlacionados com o valor medido (exemplo)"
    params = (
        Param("menor_fracao", "Menor fração de pontos a testar", "float", default=0.1, min=0.01, max=1.0,
              help="O programa testa desde esta fração até o espectro inteiro."),
        Param("tentativas", "Quantas frações testar", "int", default=5, min=2),
    )

    def select(self, X, y, error, rng, progress, menor_fracao, tentativas):
        # X: espectros de calibração já pré-tratados, formato (amostras, pontos).
        # y: valores medidos DESTE analito, formato (amostras,). O programa chama select uma vez por analito.
        # error(escolha): erro de validação do modelo usando só os pontos de `escolha` (quanto menor, melhor).
        # rng: gerador de sorteios já com a semente do projeto (não usado aqui).
        # progress(fração, texto): avisa a tela de quanto já foi feito.
        n_pontos = X.shape[1]
        desvio = X.std(axis=0)
        desvio[desvio == 0] = 1.0
        correlacao = np.abs((X - X.mean(axis=0)).T @ (y - y.mean())) / (desvio * (y.std() or 1.0) * len(y))
        ordem = np.argsort(-correlacao)                  # do ponto mais correlacionado para o menos

        melhor, melhor_erro = np.ones(n_pontos, dtype=bool), np.inf
        for numero, fracao in enumerate(np.linspace(menor_fracao, 1.0, tentativas), start=1):
            escolha = np.zeros(n_pontos, dtype=bool)
            escolha[ordem[:max(2, int(round(fracao * n_pontos)))]] = True
            erro = error(escolha)
            if erro < melhor_erro:
                melhor, melhor_erro = escolha, erro
            progress(numero / tentativas, f"{numero} de {tentativas} frações testadas")

        # Devolva um vetor de verdadeiro/falso com um item por ponto (ou as posições dos pontos escolhidos).
        return melhor
