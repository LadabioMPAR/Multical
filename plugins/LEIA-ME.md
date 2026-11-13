# Como acrescentar métodos ao Multical (plugins)

Um plugin é um arquivo `.py` colocado nesta pasta. Quando o programa abre, ele
executa cada arquivo, e o que o arquivo registrar aparece nas listas do
programa, no modo **Avançado**: pré-tratamentos na tela 3, modelos, validações
e seleções de variáveis na tela 4. O formulário de ajustes, a gravação no projeto e no arquivo de
modelo e a exportação já vêm prontos; você escreve só o cálculo.

Este documento descreve exatamente o que cada tipo de método recebe e o que
precisa devolver. A pasta `exemplos` tem um arquivo pronto para cada caso.

## 1. Passo a passo

1. Copie o exemplo mais parecido da pasta `exemplos` para esta pasta (`plugins`).
2. Troque o `id`, o `label`, os parâmetros e o cálculo.
3. Confira: rode o autoteste (seção 9). Ele executa todos os métodos
   registrados, inclusive o seu, e diz o que está fora do formato.
4. Feche e abra o Multical. O método aparece nas listas do modo Avançado.

Regras da pasta:

- Só os arquivos `.py` **diretamente** nesta pasta são carregados. A pasta
  `exemplos` não é lida; por isso os exemplos não aparecem no programa.
- Arquivos cujo nome começa com `_` são ignorados (útil para desativar um plugin).
- Se um arquivo der erro ao carregar, o programa abre normalmente e mostra o
  nome do arquivo e o erro na tela inicial.

Onde fica esta pasta:

| Situação | Pasta |
|---|---|
| Programa instalado | `plugins`, ao lado de `Multical.exe` |
| Cópia do repositório, aberta com `py abrir_multical.py` | `plugins`, na raiz do repositório |
| Outro lugar | qualquer pasta indicada na variável de ambiente `MULTICAL_PLUGINS` |

Um plugin é código Python comum, com acesso total ao computador. Use apenas
arquivos de quem você confia.

## 2. Os dados que circulam

Em todo o programa os dados têm a mesma forma:

| Nome | O que é | Formato |
|---|---|---|
| `X` | espectros | matriz `numpy` de números reais, **uma linha por amostra e uma coluna por ponto do espectro**: formato `(amostras, pontos)` |
| `wavelengths` | eixo espectral | vetor `(pontos,)`, um valor por coluna de `X`, em ordem crescente |
| `y` | valores medidos de **um** analito | vetor `(amostras,)`, na unidade original (g/L, por exemplo) |
| `Y` | valores medidos de todos os analitos | matriz `(amostras, analitos)` |
| `groups` | experimento de cada amostra | vetor `(amostras,)` de textos, ou `None` |

Não altere as matrizes recebidas: devolva matrizes novas.

## 3. Parâmetros (`Param`)

Cada ajuste que o usuário pode mudar é um `Param`. O programa monta o campo na
tela, confere os limites e entrega o valor já convertido como argumento da sua
função, com o nome dado em `name`.

```python
Param(name, label, kind, default, min=None, max=None, choices=None, help="", advanced=False)
```

| `kind` | Campo na tela | Valor que a função recebe |
|---|---|---|
| `"int"` | número inteiro | `int` |
| `"float"` | número com casas decimais | `float` |
| `"bool"` | caixa de marcar | `True` ou `False` |
| `"choice"` | lista de opções (`choices=("a", "b")`) | o item escolhido de `choices` |
| `"ranges"` | faixas do eixo, como `4400–7500; 8000–9000` | lista de pares `[[4400.0, 7500.0], [8000.0, 9000.0]]` (vazia: nada informado) |

- `label` é o texto ao lado do campo; `help` aparece quando o mouse para sobre ele.
- `min` e `max` valem para `int` e `float`.
- `advanced=True` esconde o campo de quem não ligou o modo Avançado.
- Todo parâmetro precisa de um `default` que funcione sem o usuário mexer.

## 4. Pré-tratamento

O programa aplica a receita **uma vez ao conjunto de calibração inteiro**
(todas as amostras em uso, antes de separar treino e validação) e, mais tarde,
às amostras novas, **em grupos de até 50 espectros**. Por isso:

> O resultado de um espectro não pode depender dos outros espectros que vieram
> junto com ele. O que depende do conjunto de calibração (uma média, um espectro
> de referência) tem de ser aprendido em `fit` e guardado no estado.

### 4.1 Sem estado: uma função

```python
@pretreatment(id="...", label="...", group="...", params=[...])
def meu_tratamento(X, wavelengths, <um argumento por Param>):
    return X_novo, wavelengths_novo
```

Devolve **dois** itens:

- `X_novo`: formato `(amostras, pontos_novo)`, com o **mesmo número de linhas** recebido e sem valores ausentes;
- `wavelengths_novo`: vetor `(pontos_novo,)`. Se o número de pontos não mudou, devolva o `wavelengths` recebido.

Exemplo: `exemplos/pretratamento_simples.py`.

### 4.2 Com estado: uma classe

```python
@register(PRETREATMENTS)
class MeuTratamento(Pretreatment):
    id = "..."; label = "..."; group = "..."; params = (...)

    def fit(self, X, wavelengths, <params>):          # só com os espectros de calibração
        return {"nome": numero_ou_matriz, ...}

    def apply(self, X, wavelengths, state, <params>):  # calibração e amostras novas
        return X_novo, wavelengths_novo
```

- `fit` devolve o **estado**: um dicionário cujas chaves são textos e cujos
  valores são números ou matrizes `numpy` de números. Nada de textos, listas
  de objetos ou funções: o estado é gravado no arquivo de modelo.
- `apply` recebe esse mesmo dicionário em `state`. Quando o modelo é reaberto,
  um número isolado volta como número do Python e uma matriz volta como matriz.

Exemplo: `exemplos/pretratamento_com_estado.py`.

## 5. Modelo

O programa cria **um modelo por analito**. Para cada analito ele chama `fit`
várias vezes: uma para cada divisão da validação (só com as amostras de treino)
e uma última com todas as amostras, que é o modelo guardado.

O `X` que o modelo recebe já passou pela receita de pré-tratamento. Se a classe
tiver `scale_x = True`, ele também já vem padronizado (cada ponto com média 0
e desvio padrão 1, calculados no conjunto de calibração inteiro), e as amostras
novas chegam a `predict` padronizadas do mesmo jeito. O `y` chega **sem
nenhuma transformação**: se o método precisar centrar ou padronizar `y`, faça
isso em `fit`, guarde a média e o desvio no estado e desfaça em `predict`.

### 5.1 Modelo simples

```python
@register(MODELS)
class MeuModelo(Model):
    id = "..."; label = "..."; scale_x = True; params = (...)

    def fit(self, X, y, <params>):        # X: (amostras de treino, pontos); y: (amostras de treino,)
        return {...}                       # estado, com as mesmas regras da seção 4.2

    def predict(self, state, X):           # X: (amostras, pontos), o mesmo número de pontos do treino
        return estimativas                 # vetor (amostras,), na unidade original
```

Exemplo: `exemplos/modelo_simples.py`.

### 5.2 Modelo com um número de complexidade

Para métodos em que se escolhe "quantos" (variáveis latentes, componentes,
vizinhos). O usuário informa o máximo e o programa testa de 1 até ele.

```python
class MeuModelo(Model):
    sweep = "max_k"                               # nome do Param (inteiro) que dá o máximo
    sweep_label = ("vizinho", "vizinhos")         # singular e plural, usados nos textos e gráficos
    params = (Param("max_k", "Máximo de vizinhos", "int", default=10, min=1), ...)

    def fit(self, X, y, k, <os outros params>): ...       # recebe k, não max_k
    def predict(self, state, X): ...
    def cv_path(self, X_train, y_train, X_val, max_k, <os outros params>):   # opcional
        return matriz                              # formato (max_k, amostras de validação)
```

- `fit` é chamado com `k=` a complexidade a usar. O parâmetro nomeado em
  `sweep` **não** é passado a `fit`.
- `cv_path` é opcional. Ele devolve de uma vez as estimativas para todas as
  complexidades: a linha `0` é a complexidade 1, a linha `max_k - 1` é a
  complexidade `max_k`. Sem ele, o programa chama `fit` e `predict` para cada
  `k`, com o mesmo resultado e mais devagar.
- O programa reduz o máximo quando há poucas amostras: ele nunca passa do
  menor conjunto de treino menos 1, nem do número de pontos do espectro.
- `cv_folds(self, X, y, folds, max_k, <os outros params>)` também é opcional e
  só importa para a seleção de variáveis (seção 7), que valida o modelo
  milhares de vezes: recebe todas as amostras e a lista de pares (treino,
  validação) e devolve uma matriz `(max_k, amostras)` com a estimativa de cada
  amostra quando ela ficou de fora. Sem ele, o programa chama `cv_path` para
  cada par. Se escrever um, ele precisa dar os mesmos números de `cv_path`.
- A complexidade é escolhida pelo teste F (a mais simples que não perde
  precisão) ou pelo menor erro, conforme a escolha do usuário.

Exemplo: `exemplos/modelo_com_complexidade.py`.

### 5.3 Modelo com ajustes escolhidos pela validação

```python
    def candidates(self, <todos os params>):
        return [("descrição mostrada", {"param": valor}), ...]   # ou None, para um único ajuste
```

O programa treina e valida cada item da lista, chamando `fit` com os
parâmetros do usuário **substituídos** pelos do item, e fica com o de menor
erro de validação. Vale só para modelos sem `sweep`.

Exemplo: `exemplos/modelo_com_ajustes.py`.

## 6. Validação

```python
@register(SPLITTERS)
class MinhaValidacao(Splitter):
    id = "..."; label = "..."; params = (...)
    kind = "cv"             # ou "holdout"
    needs_groups = False    # True: o esquema precisa do experimento de cada amostra

    def split(self, n, groups=None, X=None, Y=None, rng=None, <params>):
        yield posicoes_de_treino, posicoes_de_validacao
```

O que `split` recebe:

| Argumento | Conteúdo |
|---|---|
| `n` | número de amostras em uso |
| `groups` | experimento de cada amostra, vetor `(n,)`; `None` se não houver |
| `X` | espectros já pré-tratados, `(n, pontos)` |
| `Y` | valores medidos dos analitos escolhidos, `(n, analitos)` |
| `rng` | gerador de sorteios (`numpy.random.RandomState`) já com a semente do projeto |

O que devolve: uma sequência de pares `(treino, validação)`, um por divisão.
Cada item do par é um vetor de **posições inteiras entre 0 e n − 1**. Use
`yield` para cada par, ou devolva uma lista de pares.

- Em cada par, nenhuma amostra pode estar nos dois lados, e nenhum lado pode ficar vazio.
- `kind = "cv"`: espera-se que toda amostra apareça na validação de alguma
  divisão. `kind = "holdout"`: uma única divisão; o erro é calculado só nas
  amostras de validação.
- Para sortear, use sempre `rng` (por exemplo `rng.permutation(n)`). Assim a
  mesma semente repete o mesmo resultado.

Exemplo: `exemplos/validacao.py`.

## 7. Seleção de variáveis

Escolhe, **para cada analito**, quais pontos do espectro o modelo usa. O
programa chama `select` uma vez por analito, depois do pré-tratamento, e o
modelo desse analito passa a ser validado, treinado e usado só com os pontos
escolhidos. A escolha é gravada no arquivo de modelo.

```python
@register(SELECTORS)
class MinhaSelecao(Selector):
    id = "..."; label = "..."; params = (...)

    def select(self, X, y, error, rng, progress, <params>):
        return escolha
```

O que `select` recebe:

| Argumento | Conteúdo |
|---|---|
| `X` | espectros de calibração já pré-tratados, `(amostras, pontos)`; já padronizados se o modelo escolhido tiver `scale_x = True` |
| `y` | valores medidos do analito, `(amostras,)` |
| `error` | função: `error(escolha)` devolve o **erro de validação** (na unidade do analito) do modelo escolhido pelo usuário usando só os pontos de `escolha`; quanto menor, melhor |
| `rng` | gerador de sorteios (`numpy.random.RandomState`) já com a semente do projeto |
| `progress` | função: `progress(fração, "texto")` avisa a tela de quanto já foi feito (fração de 0 a 1) |

O que devolve: um vetor de verdadeiro/falso com **um item por ponto**
(`True`: o ponto é usado), formato `(pontos,)`, ou um vetor com as posições
inteiras dos pontos escolhidos. Pelo menos um ponto precisa ser escolhido.

- `error(escolha)` aceita os mesmos dois formatos. Para modelos com número de
  complexidade, o erro devolvido é o menor entre as complexidades testadas.
  Uma escolha vazia, ou que o modelo não consegue calcular, devolve infinito.
- `error.curve(escolha)` devolve o erro de cada complexidade (ou ajuste)
  testado; `error.folds` é a lista de pares (treino, validação) em uso e
  `error.wavelengths` é o eixo.
- Não é preciso tratar o cancelamento: se o usuário clicar em Cancelar, a
  próxima chamada de `error` interrompe o cálculo.
- Para sortear, use sempre `rng`.
- O erro mostrado na tela de resultados é calculado de novo com os pontos
  devolvidos. Como a escolha olhou esse mesmo erro, ele tende a parecer menor
  do que será com amostras novas; o programa avisa o usuário disso.

Exemplo: `exemplos/selecao_de_variaveis.py`.

## 8. Erros e mensagens

- Para recusar um ajuste ou um dado, use `raise ValueError("frase para o usuário")`.
  A frase aparece em vermelho na tela, sem derrubar o programa.
- Se o método devolver algo fora do formato (número de linhas diferente, eixo
  com tamanho errado, estado com texto, estimativas com formato errado,
  escolha de pontos com tamanho errado ou vazia), o
  programa mostra uma mensagem com o nome do método e o que era esperado.
- Qualquer outro erro aparece como "Erro inesperado", com o trecho do código.

## 9. Como testar

O autoteste roda todos os pré-tratamentos, modelos, validações e seleções de
variáveis registrados
em dados de exemplo e grava um relatório:

| Situação | Comando |
|---|---|
| Programa instalado | `Multical.exe --autoteste relatorio.txt` |
| Cópia do repositório | `py abrir_multical.py --autoteste` |

Além dos formatos, ele confere que cada pré-tratamento dá o mesmo resultado
para um espectro sozinho e em grupo (a regra da seção 4). Pré-tratamentos cujo
ajuste padrão não serve para os dados de exemplo aparecem como "não testados".

Para experimentar à mão, em Python, a partir da raiz do repositório (com a
pasta `src` no caminho do Python: `set PYTHONPATH=src`, ou depois de
`pip install -e .`):

```python
import numpy as np
from multical import api

api.load_plugins(["plugins/exemplos"])              # aqui, os exemplos; use ["plugins"] para os seus arquivos
rng = np.random.RandomState(0)
X = rng.normal(size=(30, 200)).cumsum(axis=1)       # 30 espectros de 200 pontos
dados = api.Dataset(X, np.arange(200.0), X[:, [50, 120]], analytes=["a", "b"], units="g/L")

config = api.RunConfig(pretreatment=[{"id": "ex_subtrair_media", "params": {}}],
                       model={"id": "ex_ridge", "params": {"alfa": 0.5}},
                       validation={"id": "kfold", "params": {}})
resultado = api.run_calibration(dados, config)
print(resultado.per_analyte["a"].best_rmsecv)
print(api.predict(resultado, X[:3], dados.wavelengths))
```

## 10. Cuidados

- **`id` único e definitivo.** O `id` vai para os projetos e para os arquivos
  de modelo. Se ele mudar, ou se o plugin for removido, esses arquivos deixam
  de abrir, com a mensagem de que o método não existe.
- **Dois plugins com o mesmo `id`**: o segundo é recusado e aparece como erro na tela inicial.
- **Nada de estado fora do dicionário.** Não guarde resultados em variáveis do
  arquivo ou em atributos da classe: o modelo salvo só leva o que `fit` devolveu.
- **Tempo de cálculo.** `fit` é chamado muitas vezes (analitos × divisões ×
  ajustes testados), e muito mais vezes quando há seleção de variáveis (cada
  chamada de `error` é uma validação inteira). O programa não trava enquanto
  calcula, e o usuário pode cancelar.
- **Outras bibliotecas.** No programa instalado só existem `numpy`, `scipy`,
  `scikit-learn` e `matplotlib`. Outras bibliotecas só funcionam na cópia do
  repositório, com o Python onde foram instaladas.
- **Leitores de arquivo** (novos formatos de espectro) ainda não entram por
  plugin: a tela de importação lê `.spp` e `.txt`.
