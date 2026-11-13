"""
Modelo de plugin: pré-tratamento SEM estado (uma função).

Use esta forma quando o resultado de cada espectro depende só dele mesmo e dos
ajustes escolhidos. Para ativar, copie este arquivo para a pasta "plugins".
"""
from multical.api import Param, pretreatment


@pretreatment(
    id="ex_um_a_cada_n",                               # identificador único, sem espaços; não mude depois de salvar modelos
    label="Manter um ponto a cada N (exemplo)",        # texto que aparece na lista
    group="Exemplos",                                  # título do grupo na lista
    params=[
        Param("passo", "Manter um ponto a cada", "int", default=2, min=1,
              help="2 mantém metade dos pontos do espectro, 3 mantém um terço, e assim por diante."),
    ],
)
def um_a_cada_n(X, wavelengths, passo):
    # X: matriz (amostras x pontos); wavelengths: vetor com um valor por ponto.
    # Como o número de pontos muda, o eixo devolvido precisa mudar junto.
    return X[:, ::passo], wavelengths[::passo]
