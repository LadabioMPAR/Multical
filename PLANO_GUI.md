# Plano — Multical com interface gráfica (desktop Windows, .exe)

Levantamento feito em 2026-11-11 por leitura do código (nada foi executado nem alterado).
Convenções: `arquivo:linha` aponta a evidência; `[não verificado]` marca o que não pude confirmar no repositório.

**Resumo em 6 linhas**
1. O repositório local está igual ao `origin/main`, mas ~5 meses de trabalho (SVR, PSO, LOEXO, KS) existem só na pasta de trabalho, sem commit.
2. A GUI atual (`gui_main.py`) não serve de base: quebra ao final de toda execução e oferece SPA/PCR que o motor não calcula.
3. Recomendação de stack: **PySide6 + matplotlib embutido + PyInstaller (onedir)**, com formulários gerados a partir de esquemas de parâmetros.
4. A extensibilidade vem de quatro registros (importadores, pré-tratamentos, modelos, validações); a GUI só lê os registros.
5. O leitor de `.spp` está pronto (seção 6). Cada arquivo é a série temporal de um experimento, então a ligação com os valores medidos é pelo tempo.
6. Antes de qualquer tela é preciso uma camada de API sem gráficos/prints acoplados, validada por teste de equivalência numérica contra o motor atual.

---

## 1. Estado do repositório

**Local vs. remoto.** Após `git fetch`, `main` e `origin/main` estão no mesmo commit `7646a74` (2026-06-08): 0 à frente, 0 atrás. O remoto não tem nada mais novo que o local.

**Sem commit (risco: a única cópia está nesta pasta do OneDrive).**
- 4 arquivos rastreados modificados (161 linhas adicionadas):
  - `src/multical/core/saving.py:73` e `:133` — `train_and_save_model_svr` / `load_and_predict_svr` (salvar e carregar SVR) só existem aqui.
  - `src/multical/models/svr.py:14` — parâmetros `degree` e `coef0`.
  - `src/multical/var_selection/pso.py:5-7` — `seed`, `return_history`, `lambda_sparsity`, `p_init`.
  - `scripts/calibration/run_variable_selection.py`.
- 42 entradas não rastreadas: ~31 scripts, pastas `figures*/`, `data_fig*/`, `doe_plan/`, dois `.zip` e `RESUMO_DESCOBERTAS_AD.md` (não lido).

**Scripts não rastreados cuja funcionalidade a GUI precisa herdar.**
| Script | O que aproveitar |
|---|---|
| `scripts/calibration/run_calibration_svr.py` | `_fold_indices` (`:187`), CV de SVR (`:209`), busca em grade (`:240`) |
| `scripts/calibration/run_loexo_model_comparison.py` | dicionário de funções `predict_*(A_tr, y_tr, A_te)` (`:205`) e `loexo(exps, predict_fn)` (`:216`) — já é, na prática, o contrato de modelo e de validação por experimento |
| `scripts/calibration/run_ks_split_svr.py` | `_select_ks` Kennard-Stone/SPXY (`:46`), `make_split` (`:77`) |
| `scripts/calibration/run_loexo_svr.py` | seleção aninhada VIP + ajuste (`select_and_tune`, `:78`) |
| `scripts/plotting/plot_influence_*.py`, `diagnose_ad.py` | diagnóstico de domínio de aplicabilidade `[não verificado: conteúdo não lido]` |

**Ramos remotos.** `homero_testes`, `Dan-teste` e `martin-teste` são ancestrais de `main` (0 commits exclusivos, 2024). `artur---teste` tem 6 commits exclusivos (último em 2025-12-03) em `main.py`, `.idea/` e `workspace.json`, sobre a estrutura antiga; nada aproveitável para a GUI. Nenhum ramo contém leitor de `.sp`; `homero_testes` tinha importadores `txt/spc/xlsx` com pandas (`arquivos/spec.py`), úteis só como referência.

**Outros fatos que afetam o plano.**
- `data/` (745 MB) e `results*/` são ignorados pelo git (`.gitignore:161-162`): não há dado versionado para testes.
- Não existe `pyproject.toml`/`setup.py`; o pacote não é instalável. Os imports misturam relativo (`src/multical/core/engine.py:6-12`) e absoluto `from src.multical...` (`analysis.py:5`, `core/outliers.py:4-5`, `var_selection/pso.py:3`, `sa.py:2`, `utils.py:2-3`, `vip.py:4`); a GUI depende de `sys.path.append(os.getcwd())` (`gui_main.py:10`).
- `requirements.txt` não lista `torch`, usado em `src/multical/models/ann_torch.py:2`.
- `docs/Import.rst:4` e `docs/arquivos.rst:10` documentam módulos que não existem em `main`.
- Python disponível na máquina: 3.14.4 (via `py`); não há `venv` no projeto.

**Recomendação antes de começar (não executei nada disso).**
1. Fazer commit do trabalho pendente em um ramo (por exemplo `consolidacao-svr-loexo`) e enviar ao remoto.
2. Decidir quais scripts de artigo ficam em `scripts/` e quais vão para uma pasta de arquivo; a GUI só depende dos cinco da tabela acima.
3. Criar o ramo `gui` a partir desse ponto.

---

## 2. Stack

**Toolkit (3 opções).**
| Critério | PySide6 (Qt) | tkinter + ttkbootstrap/CustomTkinter | NiceGUI em janela nativa |
|---|---|---|---|
| Reaproveita `gui_main.py` | não | em parte | não |
| Assistente em etapas e tabelas editáveis | prontos (`QStackedWidget`, `QTableView`) | manual; tabela editável não é nativa | bons |
| Matplotlib embutido com zoom | backend QtAgg oficial | backend TkAgg oficial | via imagem/HTML, menos direto |
| Cálculo sem travar a janela | `QThread` + sinais | `threading` + `after()` manual | assíncrono |
| Empacotar em .exe | hooks maduros no PyInstaller | o mais leve | servidor local + webview; mais frágil |
| Curva para quem não domina interfaces | média | baixa | baixa a média |
| Licença | LGPL | livre | MIT |

**Empacotamento (2 opções).** PyInstaller (maduro, bem documentado para Qt/numpy/scipy/sklearn) vs. Nuitka (executável compilado, build mais lento e mais difícil de depurar).

**Impacto do torch.** `ann_torch.py` puxaria o PyTorch para dentro do .exe, com aumento de tamanho da ordem de centenas de MB `[estimativa; medir]`. O próprio `run_loexo_model_comparison.py:17` e `:254` exclui a ANN por haver poucos lotes. ANN fica fora do executável padrão e entra como plugin opcional (seção 4).

**Recomendação: PySide6 + matplotlib (QtAgg) + PyInstaller em modo pasta (`--onedir`)**, com instalador simples (Inno Setup) que cria o atalho.
- A curva do Qt é compensada pela decisão central deste plano: **os formulários são gerados a partir do esquema de parâmetros de cada método**. Você escreve um componente de formulário uma vez, não uma tela por método.
- Modo pasta em vez de arquivo único: abre mais rápido e dispara menos alarmes de antivírus.
- `src/multical/var_selection/pso.py:2` usa `multiprocessing.Pool`; no .exe isso exige `multiprocessing.freeze_support()` no ponto de entrada.
- Ambiente de build fixado em uma versão de Python com rodas estáveis para PySide6, scikit-learn e PyInstaller; compatibilidade com 3.14 `[não verificado]`, por isso sugiro 3.12 ou 3.13.
- Sem assinatura digital o Windows SmartScreen mostra aviso na primeira execução; registrar isso no guia do usuário.

---

## 3. Arquitetura em camadas

```
src/multical/                núcleo de cálculo (existente; os scripts continuam usando)
  api/                       NOVO — camada de serviço, sem pyplot, sem print, sem escrita em disco
    params.py                Param: descrição de um parâmetro (rótulo, tipo, padrão, limites, ajuda, avançado)
    registry.py              registros IMPORTERS, PRETREATMENTS, MODELS, SPLITTERS + decoradores
    dataset.py               Dataset(X, wavelengths, Y, analytes, units, sample_ids, groups, times)
    config.py                RunConfig <-> JSON (receita de pré-tratamento, modelo, validação, semente)
    calibrate.py             run_calibration(dataset, config, progress, cancel) -> CalibrationResult
    predict.py               predict(bundle, dataset) -> PredictionResult
    results.py               dataclasses de resultado (métricas por k, predições de CV, k escolhido, T²/Q)
    plots.py                 funções puras: recebem resultado, devolvem matplotlib.figure.Figure
    bundle.py                pacote de modelo .multical (seção 7) + leitura dos .pkl antigos
  io/                        NOVO — sp_reader.py, txt_pair_reader.py, reference_table.py
src/multical_gui/            NOVO — PySide6; só conhece multical.api
  app.py, wizard.py, worker.py (QThread), pages/, widgets/param_form.py, widgets/plot_canvas.py
plugins/                     pasta ao lado do .exe; métodos adicionados sem recompilar
tests/                       equivalência numérica, leitores, contrato dos registros
```

Regra de dependência: `multical_gui` → `multical.api` → núcleo. O núcleo nunca importa a GUI nem `pyplot`. Os scripts atuais continuam chamando o núcleo como hoje e podem migrar para `multical.api` aos poucos.

**O que precisa ser desacoplado, arquivo por arquivo.**
| Arquivo | Problema | Evidência |
|---|---|---|
| `core/engine.py` | cálculo, gráficos e escrita de arquivos na mesma função | pyplot `:4`; gráficos `:463-615`; `plt.show` `:504`, `:615`; `np.savetxt` `:349-373`; 14 `print` |
| `core/engine.py` | só PLS é calculado; `pcr_model`/`spa_model` são importados e nunca chamados | `:9-10`, `:122-124`, `:290`, `:315` |
| `core/engine.py` | parâmetros sem efeito: `optkini`, `lini`, `unid`, `outlier`, `frac_test`; `RMSEtest` nunca preenchido | `:18`, `:89`, `:116-117` |
| `core/engine.py` | sorteio sem semente (holdout e CV aleatória não reprodutíveis) | `:155`, `:166` |
| `core/engine.py` | devolve tupla de 9 itens e não devolve as predições de CV nem os espectros pré-tratados; o script recalcula tudo | `:86`; `scripts/calibration/run_calibration.py:159`, `:219`, `:274`, `:310`, `:353-355` |
| `preprocessing/pipeline.py` | despacho por `if/elif` de strings; gráficos e `print` dentro | `:44-163`; `:23-32`, `:165-174`; `:36`, `:42` |
| `preprocessing/pipeline.py` | `AutoScale`/`MeanCenter` usam a média do conjunto recebido, sem guardar a da calibração | `:53`, `:58` |
| `core/saving.py` | modelo salvo não contém a receita de pré-tratamento; a inferência exige repeti-la à mão | `:24-28`; `scripts/inference/run_inference.py:68` |
| `core/saving.py` | grade de comprimentos de onda diferente só gera aviso | `:153-154`, `:187-188` |
| `core/saving.py` | formato pickle/joblib (frágil entre versões) | `:69-70`, `:129` |
| `core/outliers.py`, `analysis.py`, `inference.py`, `var_selection/vip.py` | cálculo e gráfico juntos | `outliers.py:15`; `analysis.py:2`; `inference.py:281`; `vip.py:2` |
| `models/pls.py` | método `compute_outlier_metrics` definido duas vezes | `:73`, `:168` |

**Por que não evoluir `gui_main.py`** (pela leitura do código; não executei):
- Desempacota 6 valores de `engine.run`, que devolve 9 → erro ao fim de toda execução (`gui_main.py:323` vs. `engine.py:86`).
- Oferece SPA e PCR (`gui_main.py:158`) que o motor não calcula (linha acima na tabela).
- Pré-tratamento digitado como lista Python e avaliado com `eval` (`gui_main.py:202-207`, `:301`).
- Cálculo na thread da janela (`:321`); resultado é só "veja a pasta" (`:329`); não salva modelo nem faz previsão.

---

## 4. Contrato de extensão

Quatro registros. A GUI monta listas e formulários lendo os registros; nenhum método é citado por nome no código da interface.

```python
@dataclass
class Param:
    name: str; label: str
    kind: str            # 'int' | 'float' | 'bool' | 'choice' | 'ranges'
    default: object
    min: float = None; max: float = None; choices: list = None
    help: str = ""       # texto leigo mostrado ao lado do campo
    advanced: bool = False   # True => só aparece no modo Avançado

class Pretreatment:      # registro PRETREATMENTS
    id: str; label: str; group: str; params: list[Param]
    def fit(self, X, wl, **p) -> dict: return {}          # estado aprendido na calibração (opcional)
    def apply(self, X, wl, state, **p) -> tuple[X, wl]

class Model:             # registro MODELS
    id: str; label: str; params: list[Param]
    sweep: str = None    # nome do parâmetro de complexidade varrido de 1..max (ex.: 'k'); None => sem varredura
    grid: dict = None    # grade de hiperparâmetros (ex.: SVR)
    def fit(self, X, Y, **p) -> dict                       # estado serializável
    def predict(self, state, X) -> Y
    def cv_path(self, Xtr, Ytr, Xval, max_k) -> Y[k]       # opcional: predições para todos os k de uma vez

class Splitter:          # registro SPLITTERS
    id: str; label: str; params: list[Param]
    kind: str            # 'cv' (vários folds) | 'holdout' (uma divisão)
    needs_groups: bool = False
    def split(self, n, groups=None, X=None, Y=None, rng=None, **p) -> Iterator[(train_idx, val_idx)]

class Importer:          # registro IMPORTERS
    id: str; label: str; extensions: list[str]
    def read(self, paths) -> Dataset
```

**Como o que já existe se encaixa.**
| Item | Onde está hoje | Encaixe |
|---|---|---|
| PLS | `models/pls.py:4`, `nipals` `:8`, `predict_model` `:110` | `Model`, `sweep='k'`; `cv_path` reaproveita o caminho rápido de `engine.py:243-259` |
| PCR | `models/pcr.py:4` | `Model`, `sweep='k'`; hoje ajusta e prediz numa só chamada, precisa separar em `fit`/`predict` |
| SPA | `models/spa.py:72` (`cini` como parâmetro) | `Model`, `sweep='k'`; `cini` e `optkini` como `Param(advanced=True)` |
| SVR | `models/svr.py:8`, parâmetros `:14` | `Model`, `grid` vindo de `run_calibration_svr.py:240` |
| ANN | `models/ann_torch.py:53`, parâmetros `:64-67` | `Model` distribuído em `plugins/` (exige torch) |
| Ridge, ElasticNet, kNN, RF | `run_loexo_model_comparison.py:81-193` | já têm a forma `predict(A_tr, y_tr, A_te)`; viram `Model` com adaptador de poucas linhas |
| MA, AutoScale, MeanCenter | `preprocessing/pipeline.py:44`, `:50`, `:55` | `Pretreatment`; os dois últimos com `fit` (guardam a média/desvio da calibração) |
| SG, Loess, Deriv, SNV | `pipeline.py:61`, `:84`, `:97`, `:106` | `Pretreatment` sem estado |
| EMSC, MSC | `pipeline.py:110`, `:115` | `Pretreatment`; se usam espectro médio de referência, precisam de `fit` `[não verificado: scatter_correction.py não lido]` |
| BLCtr, Cut | `pipeline.py:120`, `:142` | `Pretreatment`; `Cut` usa `Param(kind='ranges')` |
| k-fold venetian / random / consecutive | `engine.py:213-214`, `:165-170`; `run_calibration_svr.py:187` | `Splitter(kind='cv')`, com `rng` semeado |
| Leave-one-out | `engine.py:139-140` | `Splitter(kind='cv')` |
| Holdout | `engine.py:148-158` | `Splitter(kind='holdout')` |
| Leave-one-experiment-out | `run_loexo_model_comparison.py:216` | `Splitter(kind='cv', needs_groups=True)` |
| Kennard-Stone / SPXY | `run_ks_split_svr.py:46`, `:77` | `Splitter(kind='holdout')`, recebe `X` (e `Y` no SPXY) |
| Escolha de k pelo teste F de Osten | `engine.py:379-432`; `var_selection/utils.py:114` | regra de seleção para modelos com `sweep`; mínimo de RMSECV para modelos com `grid` |
| Leitura de pares `.txt` | `gui_main.py:335-383`; `run_calibration.py:109` | `Importer` legado |

**Como adicionar um método novo** (exemplo: um pré-tratamento):
```python
# plugins/log10.py  — ou src/multical/preprocessing/log10.py
import numpy as np
from multical.api import pretreatment, Param

@pretreatment(id="log10", label="Logaritmo da absorbância", group="Transformações",
              params=[Param("eps", "Deslocamento", "float", default=1e-6, min=0, advanced=True,
                            help="Somado antes do logaritmo para evitar log de zero.")])
def log10(X, wl, eps):
    return np.log10(X + eps), wl
```
1. Criar um arquivo com a função ou classe e o decorador do registro correspondente.
2. Declarar os parâmetros em `params`.
3. Salvar em `plugins/` (usuário do .exe) ou no pacote (desenvolvedor). Ao abrir o programa, o método aparece na lista, com formulário, validação de limites, gravação em `config.json` e inclusão no pacote de modelo — sem abrir nenhum arquivo de `multical_gui/`.
4. Um teste de contrato (`tests/test_registry.py`) roda todo item registrado em dados sintéticos e falha se a forma de saída estiver errada.

Plugins executam código arbitrário: carregar só de fonte confiável.

---

## 5. Pipeline de telas

Barra lateral fixa com as etapas numeradas (feita, atual, bloqueada), botões **Voltar / Avançar** no rodapé e um interruptor **Avançado** no topo. Cada tela tem um painel de formulário à esquerda e um gráfico à direita. Uma etapa só libera a seguinte quando suas validações passam.

**Tela 0 — Início.** Objetivo: escolher o que fazer.
```
┌──────────────────────────────────────────────────────────────┐
│  Multical                                                    │
│  ┌────────────────────────┐   ┌────────────────────────┐     │
│  │  Criar um modelo       │   │  Usar um modelo pronto │     │
│  │  (tenho espectros e    │   │  (quero prever novas   │     │
│  │   valores medidos)     │   │   amostras)            │     │
│  └────────────────────────┘   └────────────────────────┘     │
│  Projetos recentes: fermentacao_out.mcproj · ...             │
└──────────────────────────────────────────────────────────────┘
```

**Tela 1 — Importar.** Objetivo: carregar espectros e ligá-los aos valores de referência.
- Entradas: um `.spp` por experimento (ou par `.txt` legado); tabela de referência com tempo + analitos por experimento. A tela oferece a escolha "tempo exato" ou "janela" (com o raio em minutos). Ver seção 6: o wireframe abaixo ainda supõe um espectro por arquivo e será refeito na fase 2.
- Padrões: todos os arquivos selecionados; analitos = colunas da planilha; unidade `g/L` (`run_calibration.py:42`).
- Validações (texto leigo): "3 arquivos têm faixa espectral diferente dos demais e foram desmarcados"; "12 espectros não têm valor medido: servem para previsão, não para criar o modelo"; "Só há 8 amostras com valor medido; o modelo tende a ser pouco confiável" (limiar a definir, pergunta 5).
- Avançado: interpolar para grade comum; unidade do eixo x; coluna de experimento/lote; coluna de tempo.
```
│ [Adicionar .sp…] [Adicionar pasta…]   [Gerar planilha-modelo] [Carregar planilha…] │
│ ┌──┬───────────────┬────────┬───────┬──────┬──────┬──────┬────────────┐           │
│ │✓ │ Arquivo       │ Pontos │ Faixa │ cb   │ gl   │ xy   │ Experimento│           │
│ │✓ │ exp4_030.sp   │ 3101   │ …     │ 1.59 │ 7.74 │ 4.48 │ exp4       │           │
│ │✓ │ exp4_060.sp   │ 3101   │ …     │ —    │ —    │ —    │ exp4   ⚠   │           │
│ └──┴───────────────┴────────┴───────┴──────┴──────┴──────┴────────────┘           │
│ 46 espectros · 34 com valor medido · 3 analitos                       [Avançar ▶] │
```

**Tela 2 — Inspecionar espectros.** Objetivo: ver os dados e tirar o que está claramente errado.
- Entradas: conjunto importado. Padrões: todos os espectros sobrepostos, cor por experimento.
- Validações: sinaliza espectro saturado, plano, com valor ausente ou duplicado; nada é removido sem clique do usuário.
- Avançado: colorir por analito; aba PCA e aba Lambert-Beer (hoje em `analysis.py:94` e `:33`).
```
│ Colorir por: [Experimento ▾]      │  absorbância                         │
│ Amostras (46)                     │   ╱╲      ╱╲╲                        │
│ ✓ exp4_030                        │  ╱  ╲────╱   ╲___                    │
│ ✓ exp4_060                        │ ─────────────────────── comprimento  │
│ ☐ exp5_120  ⚠ saturado            │  Clique em uma curva para destacar   │
│ [Excluir marcadas] [Restaurar]    │                                      │
```

**Tela 3 — Pré-tratar.** Objetivo: aplicar a correção dos espectros com visualização imediata.
- Entradas: uma **receita** (lista ordenada de passos). Padrão: "Receita do laboratório" definida por você (pergunta 5); sem ela, nenhum passo.
- Validações: "A janela do filtro (15 pontos) é maior que a região cortada"; "O corte removeu todos os pontos".
- Avançado: adicionar, remover e reordenar passos; parâmetros de cada passo (formulário gerado do `Param`); salvar como nova receita.
- Substitui a caixa de texto com sintaxe Python de `gui_main.py:202-207`.
```
│ Receita: [Laboratório – NIR fermentação ▾]   │  Antes  ┄┄┄┄   Depois ────   │
│ 1 ▸ Cortar região      4400 – 7500     [⋮]  │    ╱╲    ╱╲                   │
│ 2 ▸ Savitzky-Golay     1ª derivada     [⋮]  │ ──╱──╲──╱──╲────              │
│ [+ Adicionar passo]        (Avançado)       │                               │
│ [Salvar receita…]                           │ Amostra exibida: [todas ▾]    │
```

**Tela 4 — Modelo e validação.** Objetivo: treinar e testar o modelo com um clique.
- Entradas: analitos a modelar. Padrões hoje usados nos scripts: PLS, até 15 variáveis latentes, k-fold 5 venetian, teste F ligado (`run_calibration.py:35`, `:39`, `:57-58`, `:78`). Se houver 3 ou mais experimentos, sugerir "deixar um experimento de fora" (pergunta 4).
- Validações: "Há mais variáveis latentes (15) do que amostras de treino permitem (11)"; "A validação por experimento precisa da coluna Experimento".
- Avançado: escolha do modelo e seus parâmetros; tipo de validação e parâmetros; regra de escolha (teste F ou menor erro); semente; seleção de variáveis (VIP/PSO, fase 6).
- Execução em segundo plano com barra de progresso e botão Cancelar.
```
│ Analitos: ☑ cb  ☑ gl  ☑ xy                                         │
│ Como testar o modelo?  (•) Automático (recomendado)  ( ) Escolher   │
│ ── Avançado ───────────────────────────────────────────────────    │
│ Modelo: [PLS ▾]   Máx. variáveis latentes: [15]                    │
│ Validação: [k-fold ▾]  Partes: [5]  Tipo: [venetian ▾]             │
│                    [ Treinar modelo ]   ▓▓▓▓▓░░░ 62%  [Cancelar]   │
```

**Tela 5 — Resultados e diagnóstico.** Objetivo: dizer em linguagem simples se o modelo presta e por quê.
- Entradas: resultado da etapa 4. Padrão: resumo por analito com erro típico na unidade original.
- Validações: semáforo por analito (limites definidos por você, pergunta 5); lista de amostras suspeitas por T² de Hotelling e resíduo Q (hoje em `core/outliers.py:7`).
- Avançado: erro por número de variáveis latentes, estatística F, tabela completa de métricas, R² de calibração e de validação.
```
│ ● gl  Erro típico ±0,42 g/L · R² validação 0,97 · 6 variáveis latentes │
│ ● cb  Erro típico ±0,11 g/L · R² validação 0,91                        │
│ ▲ xy  Erro típico ±0,80 g/L · R² validação 0,74  (atenção)             │
│ ┌ Previsto × Medido ─────────┐ ┌ Amostras suspeitas ───────────┐      │
│ │      ·  · ╱                │ │ exp7_240  longe das demais    │      │
│ │   · ·╱·                    │ │ exp5_120  espectro atípico    │      │
│ └────────────────────────────┘ └ [Excluir e treinar de novo] ──┘      │
```

**Tela 6 — Exportar.** Objetivo: salvar o modelo e os resultados em um lugar só.
- Entradas: pasta de destino e nome da execução. Padrão: tudo marcado, pasta `execucoes/AAAA-MM-DD_HHMM_<modelo>/` dentro do projeto.
- Validações: "Já existe uma execução com esse nome"; "Sem permissão de escrita nesta pasta".
- Avançado: formato das figuras (PNG 300 dpi / SVG / PDF); arquivos `.txt` no formato antigo.
```
│ Salvar em: [C:\...\projeto\execucoes\2026-11-11_1432_PLS]  [Procurar…] │
│ ☑ Modelo para uso em rotina (.multical)   ☑ Planilha de resultados     │
│ ☑ Figuras                                 ☑ Relatório (PDF)            │
│ ☑ Configuração para reproduzir por script                              │
│                                   [ Exportar ]   [Abrir pasta]         │
```

**Tela 7 — Prever novas amostras.** Objetivo: obter concentrações de espectros novos com um modelo salvo.
- Entradas: arquivo `.multical` (ou `.pkl` antigo); arquivos `.spp`.
- Padrões: pré-tratamento aplicado automaticamente a partir do pacote do modelo (hoje é repetido à mão, `run_inference.py:68`).
- Validações: "Estes espectros cobrem uma faixa diferente da usada no modelo" bloqueia a previsão (hoje só há aviso, `saving.py:187-188`); cada amostra recebe o selo "dentro/fora do que o modelo conhece" (T²/Q).
- Avançado: interpolar para a grade do modelo; comparar com valores medidos, se houver.
```
│ Modelo: fermentacao_PLS.multical  (cb, gl, xy · g/L · criado em 11/11/2026) │
│ ┌───────────────┬───────────┬───────────┬───────────┬───────────────┐      │
│ │ Amostra       │ cb (g/L)  │ gl (g/L)  │ xy (g/L)  │ Confiança     │      │
│ │ lote12_030    │ 1,52 ±0,1 │ 8,10 ±0,4 │ 4,6 ±0,8  │ ● dentro      │      │
│ │ lote12_060    │ 1,49 ±0,1 │ 9,02 ±0,4 │ 5,0 ±0,8  │ ▲ fora        │      │
│ └───────────────┴───────────┴───────────┴───────────┴───────────────┘      │
│                                            [Exportar planilha]             │
```

---

## 6. Importação de `.spp`

**Situação (2026-11-12): leitor pronto** em `src/multical/io/spp.py` (contêiner em `src/multical/io/_cfb.py`), sem dependência nova, registrado como importador `spp`. A extensão é `.spp`, não `.sp`. As hipóteses anteriores desta seção (PerkinElmer, biblioteca `specio`, um espectro por arquivo) estavam erradas e foram removidas.

**Formato**, decifrado a partir de 6 arquivos de exemplo (não há especificação pública):
- O arquivo é um documento composto OLE. Dentro dele, o bloco `CONTENTS` é um conjunto de propriedades: número de pontos, primeiro valor do eixo x e passo, espectros reservados e espectros gravados, unidades (`cm-1`, `Secs`, `Abs`) e os nomes dos blocos de dados.
- Bloco de dados: espectros × pontos em `float64`, um espectro após o outro. Bloco de tempos: um `float64` por espectro, em segundos desde o início da aquisição.
- **Um arquivo guarda a série temporal inteira de um experimento**: nos exemplos, 8801 pontos (10000 → 1200 cm⁻¹, passo 1), um espectro a cada 120 s, até 2000 ou 2200 espectros (142–156 MB).
- O arquivo nasce com espaço para o total reservado; só os primeiros "gravados" são reais, o resto é zero. O leitor devolve apenas os gravados.
- Em um dos exemplos o relógio aparece como segundos negativos (contador de milissegundos de 32 bits que deu a volta); o leitor soma 2³²/1000 s e o tempo volta a começar em ~2 s, como nos outros arquivos.
- Arquivo `.spp.dcs` ao lado: resumo da aquisição; não é necessário para ler os espectros.

**O que o leitor entrega.**
- `spp_info(caminho)`: quantos espectros, faixa, unidades e tempos, sem carregar os espectros (~0,01 s).
- `read_spp(caminho, indices=None)`: `Dataset` com o eixo x em ordem crescente (1200 → 10000), como nos arquivos de texto atuais; `indices` carrega só os espectros pedidos. Arquivo inteiro de 2000 espectros: ~0,2 s.
- `read_spp_files([...])`: vários experimentos em um `Dataset`; arquivos com o mesmo nome (`Untitled.spp`) são distinguidos pela pasta.
- Arquivo sem espectros gravados, arquivo que não é `.spp` e arquivo truncado dão mensagem clara em português.

**Verificação.**
- Feita: os 3 arquivos com dados (1762, 2000 e 1010 espectros) são lidos; os valores conferem com uma segunda leitura independente dos bytes; tempos crescentes com intervalo mediano de 120,01 s; bandas da água em 5200 e 6900 cm⁻¹ no lugar esperado; ordem e passo do eixo iguais aos das exportações em texto já existentes (`data/jao_espectros.txt`: 10000 → 1200, passo 1).
- **Não feita:** comparação com os mesmos espectros exportados em texto pelo software do equipamento. Nenhuma exportação em `data/` corresponde aos experimentos dos exemplos. Uma exportação de um desses arquivos fecharia a verificação.
- Testes automáticos: `tests/test_spp.py` usa arquivos `.spp` sintéticos pequenos gerados por `tests/_spp_writer.py` (os reais são grandes e não vão para o repositório). Com `MULTICAL_SPP_SAMPLES=<pasta>` os mesmos testes percorrem arquivos reais.
- Arquivos de exemplo sem dados: `marcio 2.spp` (142 MB só de zeros) e dois de 5 kB.

**Associação com os valores de referência (decidido em 2026-11-12, implementado).** Como o arquivo é uma série temporal, a ligação é pelo tempo, não pelo nome do arquivo:
- Tempo dos espectros: segundos desde o primeiro espectro, como gravado no `.spp`. Tempo da tabela de referência: **sempre em minutos**, no mesmo relógio.
- Tabela de referência por experimento, como os `*_refe.txt` atuais: coluna de tempo + uma coluna por analito. Leitor em `src/multical/io/reference.py` (`read_reference_table`): texto ou CSV, separado por tabulação, ponto e vírgula, vírgula ou espaços; linha de nomes opcional; vírgula decimal aceita. `.xlsx` não é lido (exigiria `openpyxl`); a mensagem pede para salvar como CSV.
- O usuário escolhe entre dois modos (`api.MATCH_MODES`):
  - **Tempo exato** (`exact`, padrão): o espectro mais próximo do tempo da amostra. A amostra fica sem espectro se o mais próximo estiver além do intervalo usual entre espectros (2 min nos exemplos), ou seja, fora do período gravado.
  - **Janela** (`window`): a média de todos os espectros a até `raio` minutos do tempo da amostra; o raio é informado pelo usuário. Janela sem nenhum espectro deixa a amostra de fora.
- Amostras sem espectro não entram no conjunto e são listadas em frases prontas para a tela (`TimeMatch.problems()`).
- Funções: `api.match_times`, `api.pair_with_reference` (espectros já em memória), `read_spp_with_reference` (lê do `.spp` só os espectros necessários: 0,07 s para 13 amostras em um arquivo de 142 MB) e `Dataset.concat` para reunir os experimentos antes de calibrar.
- Os espectros sem valor de referência (a grande maioria) ficam disponíveis para previsão ao longo do tempo (tela 7).
- Em aberto: valor medido ausente para um analito em uma linha (hoje a tabela precisa estar completa); raio em minutos (e não em número de espectros).

**Conversão para o formato interno.** O importador devolve `Dataset`. Para o núcleo atual: `absor0 = vstack([wavelengths, X])` e `x0 = Y`, como em `gui_main.py:378-381`. Grades diferentes entre arquivos: recusar com mensagem clara por padrão; interpolação para grade comum só no modo Avançado.

---

## 7. Exportação

Tudo em uma pasta por execução (`execucoes/AAAA-MM-DD_HHMM_<modelo>/`):

| Arquivo | Conteúdo | Formato |
|---|---|---|
| `modelo.multical` | manifesto (versão do Multical, analitos, unidade, grade espectral, receita de pré-tratamento com estado, modelo e parâmetros, validação, métricas, limites de T²/Q) + matrizes | zip com `manifest.json` + `.npz`. PLS fica só numérico (os mesmos campos de `saving.py:55-64`), sem pickle. Modelos scikit-learn entram via joblib com a versão registrada (`saving.py:129`) |
| `resultados.xlsx` | abas Resumo, Métricas por k, Predições de validação, Amostras suspeitas, Configuração | `.xlsx` + `.csv` equivalentes |
| `figuras/` | previsto × medido, erro × variáveis latentes, teste F, T² × Q, espectros antes/depois | PNG 300 dpi (como `engine.py:500`, `:611`); SVG/PDF opcionais |
| `relatorio.pdf` | uma página por analito com resumo, figuras e avisos | PDF gerado com matplotlib (sem dependência nova) |
| `config.json` + `reproduzir.py` | configuração completa (com semente) e script de 10 linhas que chama `multical.api` e reproduz os mesmos números | JSON + Python |
| `Erro_cal.txt`, `Erro_cv.txt`, `minimos.txt` | compatibilidade com os scripts atuais (`engine.py:352-373`) | texto, opcional |
| `previsoes.xlsx` (tela 7) | amostra, valor previsto por analito, ± RMSECV, selo dentro/fora | `.xlsx` + `.csv` |

O leitor de `.multical` também abre os `.pkl` antigos (`saving.py:133`, `:171`), pedindo a receita de pré-tratamento, que eles não guardam.

---

## 8. Roteiro em fases

| Fase | Entregável | Arquivos | Depende de | Risco principal | Pronto quando |
|---|---|---|---|---|---|
| **0 Preparação** | trabalho pendente em commit; pacote instalável; números de referência; leitor `.spp` provado | `pyproject.toml`; imports relativos nos 6 arquivos da seção 1; `tests/fixtures/`; `tests/golden/` | respostas 1, 7, 8 | formato `.sp` diferente do esperado | `pip install -e .` funciona; os 5 scripts principais rodam sem mudança de resultado; os `.spp` de exemplo são lidos e conferidos |
| **1 API sem interface** | `multical.api` com `Dataset`, registros, `run_calibration`, `plots`; PLS + pré-tratamentos atuais + k-fold/holdout | `src/multical/api/*`, `src/multical/io/*`; núcleo intocado ou só refatorado internamente | fase 0; resposta 4 | divergência numérica ao separar cálculo de gráfico | teste de equivalência passa (abaixo); `import multical.api` não carrega `matplotlib.pyplot` |
| **2 MVP calibrar** | telas 0–6 com PLS; execução em segundo plano; exportação | `src/multical_gui/*` | fase 1; resposta 5 | usabilidade para leigos | um colega sem treinamento vai do `.spp` ao `resultados.xlsx` sem ajuda; um .exe de teste abre em máquina sem Python |
| **3 Prever** | tela 7; pacote `.multical`; leitura de `.pkl` antigo; selo dentro/fora | `api/bundle.py`, `api/predict.py`, `pages/predict.py` | fase 2 | pré-tratamentos com estado mal reproduzidos | previsão pela GUI = `load_and_predict_pls` do script nos mesmos dados (1e-10) |
| **4 Extensibilidade provada** | SVR, PCR, SPA, LOEXO, Kennard-Stone, receitas; pasta `plugins/` | só registros e adaptadores; **nenhuma mudança em `multical_gui/`** | fase 3 | contrato insuficiente para SVR (grade) | `git diff --stat` da fase mostra zero linhas em `src/multical_gui/`; um plugin de exemplo aparece na GUI |
| **5 Distribuição** | instalador Windows; guia do usuário de 2 páginas | `packaging/multical.spec`, `packaging/installer.iss`, `docs/` | fase 4; resposta 6 | antivírus/SmartScreen; tamanho | instala e roda em 2 máquinas limpas; PSO (multiprocessing) funciona no .exe |
| **6 Extras** | seleção de variáveis (VIP/PSO), plugin ANN, domínio de aplicabilidade, documentação atualizada | `var_selection/*` via registro novo; `plugins/ann.py` | fase 5 | tempo de PSO na GUI | cada item com seu teste de equivalência contra o script correspondente |

**Teste de equivalência numérica (fase 1, `tests/test_equivalencia.py`).**
- Dados: subconjunto pequeno de `data/splits/` copiado para `tests/fixtures/` (hoje `data/` não é versionado; resposta 8).
- Configuração: a de `scripts/calibration/run_calibration.py` — PLS, 15 variáveis latentes, k-fold 5 venetian, `Cut 4400–7500` + `SG(7,2,1)` (`:35-69`). Venetian não tem sorteio (`engine.py:213-214`), então o teste é determinístico.
- Caminho A: `MulticalEngine.run` com backend `Agg`. Caminho B: `multical.api.run_calibration`.
- Critério: `RMSECV_conc`, `RMSEcal_conc` e `best_k` iguais com `rtol=1e-10`; predições do modelo salvo iguais às de `load_and_predict_pls`.
- Para validações com sorteio, a semente é injetada nos dois caminhos (`engine.py:155`, `:166` usam o gerador global).

---

## 9. Perguntas abertas

**Relatos de 2026-11-14: LOESS travando o programa e gráfico Q × T² ausente.**
- **LOESS.** Causa reproduzida: `preprocessing/loess.py` ajusta um polinômio por ponto e por espectro (3,6 s por espectro de 8.801 pontos; cerca de 5 min para 79), e a prévia da tela 3 roda na própria janela, que ficava congelada (“não está respondendo”) até terminar. Não foi um erro do programa em si, e sim a janela parada por minutos; o relato do usuário foi de “crash”, e não vi a tela dele.
  - Correção: `api/pretreatments.loess_rows` calcula os pesos de cada ponto uma vez e aplica a todos os espectros juntos. Mesmo resultado da função original (diferença relativa máxima de 3e-11 em 112 combinações de eixo, fração e grau; `np.polyfit` multiplica os resíduos pelos pesos, então o ajuste é ponderado pelo quadrado deles, e isso foi mantido). 79 espectros × 8.801 pontos: 3,7 s com fração 0,2 (10,6 s com fração 1). A função original não foi alterada; os scripts continuam iguais.
  - A prévia mostra o cursor de espera enquanto calcula.
  - **Limite que continua:** a prévia ainda roda na janela. Um passo lento (o LOESS com fração perto de 1, ou um plugin lento) ainda a deixa parada por alguns segundos. Na tela de estimar amostras, o LOESS custa cerca de 3 s a cada 50 espectros.
- **Q × T².** O programa já calculava T² e Q para avisar de espectros diferentes ao estimar amostras, mas não mostrava o gráfico. Agora:
  - Tela 5, aba **“Amostras atípicas (Q × T²)”**, visível para todos: as amostras de calibração do modelo de cada analito, com os limites e o nome das que ficam além deles; o quadro ao lado lista essas amostras. Para PLS é o *Influence Plot* dos scripts (`core/outliers.evaluate_and_plot_outliers`: escores do PLS do analito com o número de variáveis latentes escolhido, limites de 95%), conferido linha a linha. Para os outros modelos, que não têm variáveis latentes, usa-se o resumo em componentes principais que já julga os espectros novos (`api/domain.py`, limites de 99%).
  - Tela “Estimar amostras”, aba **“Q × T²”**: os espectros estimados em relação aos limites do modelo.
  - Exportação: `amostras_q_t2.csv`, `figuras/q_x_t2_<analito>.png` e uma linha por analito no relatório.
  - `api.influence(result, analito)` devolve os valores.
- **Testes:** 150 passam; 1 pulado (precisa de `MULTICAL_SPP_SAMPLES`).

**Pedido de 2026-11-13: adicionar vários experimentos de uma vez.**
- Botão **“Adicionar vários arquivos…”** na tela de importação: uma única janela de escolha, com seleção múltipla, para os espectros e as tabelas de todos os experimentos. O botão de um experimento por vez continua.
- `src/multical_gui/file_pairs.py` separa os arquivos pelo conteúdo (`.spp` e textos com 30 colunas ou mais são espectros; textos com 2 a 29 colunas de números são tabelas; o resto é ignorado) e sugere os pares pelo nome: palavras e números em comum e, no empate, o que os nomes têm de diferente. Um par só é sugerido quando cada arquivo é a única melhor opção do outro; na dúvida (por exemplo `exp8_nonda` e `exp8_inf` para um só `exp8_refe`), a linha fica em “não adicionar”. Um arquivo de espectros e uma tabela, sozinhos, são tomados como par.
- **A sugestão nunca é aplicada sem conferência:** a janela “Conferir os experimentos” mostra cada arquivo de espectros com a tabela sugerida (trocável) e o nome do experimento (editável), lista os arquivos sem tabela e as tabelas sem uso, e recusa a mesma tabela em duas linhas. Espectros que já estão no projeto ficam de fora.
- Os dados são lidos uma vez para o lote inteiro (`Project.add_experiments`), não uma vez por experimento.
- Verificado nos nomes reais de `data/` (só leitura dos nomes e das primeiras linhas): `exp4..9_nonda` com `exp4..9_refe` e `abs02..05` com `x02..05` saem certos.
- **Limite:** a janela de escolha do Windows só seleciona arquivos de uma mesma pasta. Experimentos em pastas separadas (como os `.spp` chamados “Untitled” em uma pasta por experimento) continuam sendo adicionados um por vez.

**Fase 6, segunda parte (2026-11-13): plugin de rede neural e README.**
- **Plugin opcional** `plugins/opcionais/rede_neural.py` (id `ann`), inativo até ser copiado para `plugins`. O treino chama `multical.models.ann_torch.ANNModel.fit_predict` (PyTorch, em CPU) com os valores de um analito; do resultado ficam só números (pesos e viés de cada camada, médias e desvios), e a estimativa é calculada em numpy. Assim o arquivo `.multical` de uma rede treinada na cópia do repositório abre e estima no programa instalado, que não traz o PyTorch. Sem PyTorch, “Treinar modelo” mostra uma mensagem explicando.
- O build passa a copiar também a pasta `plugins/opcionais`.
- **Verificado** (`tests/test_ann_plugin.py`; PyTorch 2.15.0 CPU instalado nesta máquina em 2026-11-13, a pedido do usuário): o plugin carrega sem PyTorch; a estimativa confere com a conta feita à mão para as seis funções de ativação; o treino dá as estimativas de `ANNModel.fit_predict` (diferença máxima de 8e-7 em 64 × 8.801 pontos, porque a rede calcula em precisão simples e o plugin em dupla), com e sem parada antecipada; o modelo salvo é relido e usado com o PyTorch escondido; o treino sem PyTorch é recusado com a mensagem. Os testes de treino são pulados onde o PyTorch não está instalado.
- **Tempos** (dados sintéticos de 80 × 8.801 pontos, ajustes padrão): 2,6 s por treino; calibração inteira de 3 analitos com 5 partes em 29 s; arquivo de modelo de 5,6 MB.
- **Pela tela** (dados sintéticos): com o plugin carregado, a rede aparece na lista de modelos com seus oito ajustes, treina, mostra o resultado e exporta um modelo que repete as estimativas.
- **No executável** (o de `dist/`, com o arquivo copiado para `plugins` só durante o teste): o plugin carrega sem erro e o treino é recusado com a mensagem do PyTorch; o autoteste acusa essa falha em “Modelos e validações”, como o tutorial avisa.
- **Não verificado:** abrir, no executável, um `.multical` de rede neural e estimar amostras pela tela (o cálculo usa só numpy e foi testado em Python com o PyTorch escondido); a rede nos dados reais de `data/`.
- **README** reescrito (em inglês, como o anterior): as três formas de uso (programa, API, scripts), o que o programa faz, instalação, como gerar o programa para Windows, exemplo da API e testes. O exemplo de código do README é executado por um teste.
- **Testes:** 134 passam; 1 pulado (precisa de `MULTICAL_SPP_SAMPLES`). Sem o PyTorch instalado, 2 a mais são pulados.

**Fase 6, primeira parte (2026-11-13): seleção de variáveis por analito.**
- **Decisão do usuário:** a seleção é feita por analito, com o modelo do próprio analito (e não com o PLS conjunto que `run_pso` e `run_simulated_annealing` otimizam quando recebem `target_idx`).
- **Registro novo `SELECTORS`** (`api/registry.py`, contrato `Selector.select(X, y, error, rng, progress, **params)`), ao lado dos outros quatro; aceita plugins. Métodos embutidos em `api/selection.py`: `vip` (com a busca do limite de 0,1 a 1,4 de `scripts/calibration/run_variable_selection.py`, ou limite fixo), `pso` e `sa`. Os sorteios são feitos na mesma ordem do código original.
- **Motor** (`api/calibrate.py`): `RunConfig.variable_selection` (`None` = espectro inteiro); para cada analito, a seleção roda depois do pré-tratamento e a validação e o modelo final usam só os pontos escolhidos. O resultado guarda `selected` (um verdadeiro/falso por ponto) e `full_rmsecv` (erro com todos os pontos, pela mesma regra de escolha da complexidade). O erro que os métodos minimizam é o menor RMSECV entre as complexidades testadas, como no código original; o erro **mostrado** é o da complexidade escolhida pela regra do projeto (teste F ou mínimo).
- **Velocidade:** durante a busca, o PLS é validado por `PLSModel.cv_folds`, que calcula o mesmo NIPALS a partir dos produtos entre espectros (amostras × amostras). Em 80 amostras × 5.179 pontos: 11 ms por avaliação contra 186 ms, com diferença relativa de 5e-12 no erro. Cronometrado de ponta a ponta em dados sintéticos de 80 × 8.801 pontos, um analito, ajustes padrão: VIP 2,3 s, recozimento 3,2 s, PSO (30 partículas × 50 rodadas) 22 s. A validação que vai para o relatório continua sendo a de `cv_path`, idêntica à do código original. Para SVR com escolha automática de ajustes a busca é lenta (cada avaliação treina 24 ajustes × divisões); não foi acelerada.
- **Arquivos:** o `.multical` guarda a escolha de cada analito; quando há seleção, o formato do modelo e o da configuração passam a 2, para que uma versão antiga do programa recuse o arquivo em vez de usá-lo errado. Sem seleção, os arquivos continuam no formato 1. A exportação ganha `pontos_escolhidos.csv`, `figuras/pontos_escolhidos_<analito>.png` e três colunas no `resumo.csv`.
- **Interface:** tela 4, modo Avançado, linha “Seleção de variáveis” com os ajustes do método; a caixa de ajustes avançados agora rola quando não cabe; a barra de progresso mostra o tempo restante estimado. Tela 5: número de pontos na linha do analito, aba “Pontos usados” e aviso de que o erro de validação fica otimista depois da seleção. Tela “Estimar amostras”: a descrição do modelo diz quantos pontos ele usa.
- **Equivalência verificada** (`tests/test_selection.py`, dados sintéticos de 30 × 150): o erro de uma escolha é o de `calculate_rmsecv_fast` com uma coluna de Y; os VIP são os de `calculate_vip`; `vip`, `pso` (com e sem penalidade/partida esparsa) e `sa` devolvem **a mesma escolha** do código original com a mesma semente.
- **Nos arquivos reais** `data/exp4..9_nonda.txt` + `_refe.txt` (79 amostras × 8.801 pontos, 3 analitos, receita padrão, PLS, só leitura): com validação em 5 partes, o R² de validação passa de 0,77 / 0,43 / 0,39 (espectro inteiro) para 0,91 / 0,85 / 0,83 com VIP (4 s), 0,79 / 0,48 / 0,43 com recozimento (7 s) e 0,79 / 0,48 / 0,50 com PSO (63 s). Com “deixar um experimento de fora”, o VIP dá −0,00 / 0,33 / 0,07: o ganho visto na validação em partes quase some quando o experimento inteiro fica de fora, que é o caso avisado na tela de resultados.
- **Não feito / não verificado:** a seleção não foi rodada em arquivos `.spp` reais (não há tabela de valores medidos para eles); o novo executável não foi gerado; o domínio de aplicabilidade continua calculado com o espectro inteiro, mesmo quando o modelo usa parte dele; o erro depois da seleção não é corrigido por validação aninhada (só há o aviso).
- **Testes:** 128 passam, 1 a mais com `MULTICAL_SPP_SAMPLES` definido.

**Pedidos de 2026-11-13, implementados.**
- **Espectros em `.txt`** (como os `*_nonda.txt`), além de `.spp`, na tela de importação. Regra de ligação: se o arquivo tem um espectro para cada linha da tabela de valores medidos, liga-se pela ordem das linhas (como os scripts fazem); se tem a série inteira, liga-se pelo tempo, em minutos, com a mesma escolha entre tempo exato e janela dos `.spp`. A coluna "Situação" mostra qual das duas foi usada e avisa quando, na ligação pela ordem, os tempos dos dois arquivos não coincidem. A tabela de valores medidos passa a ser lida do mesmo jeito nos dois casos (`read_reference_table`).
- **Analitos identificáveis.** O número de analitos é o número de colunas da tabela, qualquer que seja. O quadro "Analitos" da tela de importação mostra, para cada coluna, a posição na tabela, o nome (editável com duplo clique) e o menor e o maior valor medido. Nomes vazios ou repetidos são recusados. Se as tabelas de dois experimentos têm linha de nomes e elas diferem (outra ordem, outros nomes), o experimento é recusado com a explicação. Continua havendo uma única unidade para todos os analitos.
- **Salvar e reabrir projetos** (`.mcproj`, JSON; `Project.save`/`Project.open` em `src/multical_gui/state.py`). Menu Arquivo (Novo, Abrir, Abrir recente, Salvar, Salvar como), botão "Abrir um projeto salvo" e lista de projetos recentes na tela inicial, `*` no título quando há alterações e pergunta antes de descartá-las. O arquivo guarda as escolhas (experimentos, modo de ligação, nomes dos analitos, unidade, amostras retiradas, receita, modelo, validação, modo Avançado), com o caminho de cada arquivo de dados absoluto e relativo à pasta do projeto, de modo que projeto e dados podem ser movidos juntos. **Não guarda** os espectros, o modelo treinado nem o estado da tela "Estimar amostras". Arquivo de dados ausente aparece como erro no experimento; arquivo de projeto inválido, de versão mais nova ou que pede um método de plugin ausente é recusado sem alterar o projeto aberto.
- **Tutorial de plugins com os formatos** (`plugins/LEIA-ME.md`): forma de `X`, do eixo, de `y`, de `Y` e de `groups`; o valor que cada tipo de `Param` entrega; o que recebem e devolvem os pré-tratamentos (com e sem estado), os modelos (simples, com número de complexidade, com ajustes testados pela validação) e as validações; regras do estado; erros; como testar. `plugins/exemplos/` traz um arquivo pronto para cada caso, e os testes carregam e executam cada um, além do trecho em Python do próprio tutorial.
  - O motor agora **confere o que cada método devolve** (`api/contract.py`) e diz, com o nome do método, o que estava fora do formato (número de linhas, tamanho do eixo, estado não numérico, formato das estimativas, divisões inválidas).
  - O autoteste ganhou a verificação de pré-tratamentos, inclusive a de que um espectro dá o mesmo resultado sozinho e em grupo.
  - A tela inicial lista os plugins que falharam ao carregar.
- **Testes:** 112 passam (`py -m unittest discover -s tests -t .`), 1 a mais com `MULTICAL_SPP_SAMPLES` definido.

**Decisões registradas em 2026-11-12.**
- **Pergunta 1:** resolvida. Arquivos `.spp` de exemplo recebidos e leitor pronto (seção 6). Falta apenas, como conferência final, uma exportação em texto de um dos mesmos arquivos.
- **Pergunta 2:** interface em português.
- **Pergunta 3:** a premissa "um espectro por arquivo" era falsa; cada `.spp` é a série temporal de um experimento. A ligação com os valores medidos é pelo tempo: `.spp` em segundos desde o primeiro espectro, tabela de referência sempre em minutos, e o usuário escolhe entre tempo exato e janela com raio informado (seção 6). Implementado em `api/pairing.py`.
- **Pergunta 5 (em parte):** receita padrão = apenas Savitzky-Golay com janela de 15 pontos, polinômio de 2º grau e 1ª derivada. Implementado como padrão de `RunConfig` (`api/config.py`, `default_pretreatment()`); `pretreatment=[]` continua significando "sem pré-tratamento". Ainda em aberto: validação recomendada e limites dos avisos amarelo/vermelho.
- **Fase 2 (MVP calibrar): implementada** em `src/multical_gui/` (PySide6-Essentials 6.12, que funciona no Python 3.14 desta máquina). Abrir com `py abrir_multical.py`.
  - Telas: Início, Importar dados, Inspecionar espectros, Pré-tratamento, Modelo e validação, Resultados, Exportar. Barra lateral com as etapas, que só abrem quando a anterior está completa; interruptor **Avançado**; treino em segundo plano com progresso e Cancelar.
  - Importar: um experimento = arquivo de espectros (`.spp`, ou o par `.txt` atual) + tabela de valores medidos; escolha entre tempo exato e janela com raio em minutos; nomes dos analitos e unidade editáveis.
  - Formulários gerados dos `Param` (`widgets.py`, `ParamForm`): pré-tratamentos, modelos e validações aparecem a partir dos registros, sem código de tela por método.
  - API nova usada pelas telas: `api/bundle.py` (arquivo `.multical`: zip com `manifest.json` + `arrays.npz`, sem pickle; `save_model`/`load_model`, e `api.predict` aceita o modelo carregado), `api/export.py` (pasta com modelo, tabelas CSV com `;` e vírgula decimal, figuras PNG 300 dpi, relatório PDF, `config.json`), `api/assessment.py` (frases e semáforo).
  - A interface força o esquema de cores claro, porque no modo escuro do Windows os textos ficavam ilegíveis.
  - Testes: `tests/test_gui.py` percorre a interface sem abrir janela (importar → excluir amostra → receita → treinar → resultados → exportar → recarregar o modelo), além de cancelamento e erros. `tests/test_export.py` cobre o arquivo de modelo e a exportação.
  - Conferido também com os dados reais `data/exp4..9_*.txt` (79 amostras): importação 0,4 s, tela de pré-tratamento 1,3 s, treino 2,6 s.
- **Ficou fora da fase 2** (em relação ao desenho da seção 5):
  - Tela 7 "Usar um modelo pronto": feita na fase 3 (abaixo).
  - Salvar/reabrir projeto (`.mcproj`) e lista de projetos recentes: feito em 2026-11-13.
  - Amostras suspeitas por T² de Hotelling e resíduo Q; por enquanto a tela lista as amostras com erro de validação acima de 2,5 vezes o erro típico.
  - `resultados.xlsx` (exigiria `openpyxl`) e `reproduzir.py`; saem CSV e `config.json`.
  - Receitas nomeadas salvas pelo usuário; sugestão automática de validação por experimento (o esquema ainda não está nos registros, fase 4).
- **Limites provisórios, a definir pelo laboratório (pergunta 5):** semáforo por R² de validação (verde ≥ 0,90, amarelo ≥ 0,70), aviso com menos de 20 amostras, mínimo de 6 amostras em uso. Todos em `api/assessment.py` e `pages/inspect_page.py`.
- **Observação nos dados reais:** só com o Savitzky-Golay padrão, a região ruidosa abaixo de ~4000 cm⁻¹ domina o sinal tratado (gl: R² de validação 0,43; cb 0,77; xy 0,39). O corte 4400–7500 usado em `run_calibration.py` não faz parte do padrão escolhido; dá para adicioná-lo no modo Avançado.
- **Fase 3 (prever): implementada.** Na tela inicial, "Usar um modelo pronto" abre a tela **Estimar amostras**.
  - Tela: abre um `.multical`, mostra o que ele estima (analitos, unidade, erro típico e faixa calibrada de cada um), recebe arquivos `.spp` ou texto de espectros, estima em segundo plano (progresso e Cancelar), mostra tabela e gráfico ao longo do tempo e salva a tabela em CSV. "Usar um espectro a cada N" acelera séries longas.
  - Selo por amostra, com dois motivos possíveis (`api/prediction.py`, `Prediction.notes`):
    - **espectro diferente dos usados para criar o modelo**: distâncias T² de Hotelling e resíduo Q em um resumo por componentes principais dos espectros de calibração como o modelo os vê (`api/domain.py`). Os limites vêm das próprias amostras de calibração, cada uma avaliada por um resumo feito sem ela.
    - **valor fora da faixa calibrada**: estimativa além do menor/maior valor medido na calibração, com folga de 10% da faixa.
  - O resumo e a faixa são gravados no `.multical`; modelos sem eles continuam funcionando, sem o selo.
  - `api.predict_report` faz a estimativa em blocos; `api.export_predictions` grava o CSV.
  - Modelos antigos `.pkl`: `api.load_legacy_pls(caminho, receita)` lê os PLS salvos pelos scripts sem executar código do arquivo (só números e matrizes são aceitos) e exige a receita, que o `.pkl` não guarda. **A tela ainda não abre `.pkl`**; ela explica isso ao usuário.
  - Conferido nos dados reais (somente leitura): o `results/model_calibration.pkl` lido pela API dá as mesmas estimativas do `load_and_predict_pls` (diferença relativa 4e-15); das 14 amostras de teste de `data/splits`, nenhuma é marcada como fora; na série `exp_04_inf.txt` (438 espectros, 6 s) tudo fica dentro até 720 min, que é onde termina a calibração desse experimento, e os espectros depois de ~930 min são marcados; um `.spp` inteiro de 1010 espectros leva 14 s.
  - Parâmetros provisórios do selo (em `api/domain.py` e `api/prediction.py`): até 10 componentes ou 95% da variância, confiança de 99%, folga de 10% na faixa.
- **Erro encontrado em script existente (não corrigido):** `scripts/calibration/run_calibration.py:363` passa a matriz inteira de erros como `rmsecv_list`; o `.pkl` guarda, para o analito i, a linha de erros de todos os analitos com i+1 variáveis latentes, e não o erro daquele analito. Por isso `load_legacy_pls` trata o erro típico desses arquivos como desconhecido.
- **Fase 4 (extensibilidade provada): implementada.**
  - Métodos novos, todos só por registro: modelos **PCR**, **SPA** e **SVR** (`api/models.py`); validações **deixar um experimento de fora** e **Kennard-Stone/SPXY** (`api/splitters.py`).
  - O contrato precisou de dois acréscimos antes disso (`api/registry.py`): `sweep_label` (o que o número varrido conta: variáveis latentes, componentes principais, pontos do espectro) e `candidates()` (ajustes a testar pela validação; o motor fica com o de menor erro). O SVR usa `candidates()` para escolher C, epsilon e largura entre 24 combinações.
  - A interface citava PLS pelo nome em três pontos (título de aba, cabeçalho de tabela e um aviso); passaram a vir do modelo. Depois dessa correção, os cinco métodos novos e o plugin de exemplo foram adicionados **sem nenhuma alteração em `src/multical_gui/`** (árvore idêntica antes e depois, conferida por hash) e funcionaram pela tela (`tests/test_gui.py`, classe `MethodsFromTheRegistries`). Só depois ajustei duas coisas de aparência vistas nas capturas: largura da primeira coluna da tabela e parênteses vazios no resumo.
  - Equivalência com o código existente (`tests/test_methods.py`): PCR contra `pcr_model` (modo de normalização conjunta), SPA contra `spa_model` (`teste_switch=0`: pontos escolhidos no espectro como está, normalização só com o treino) e SVR contra `SVRModel.fit_predict` (kernels rbf e linear), todas com tolerância de 1e-6 ou menor. O PCR usa SVD em vez da matriz X'X, que com 8801 pontos não caberia na memória.
  - O SVR é guardado no `.multical` como números (vetores de suporte e coeficientes) e previsto pelo próprio Multical, sem objeto do scikit-learn dentro do arquivo.
  - **Plugins** (`api/plugins.py`): todo `.py` das pastas em `MULTICAL_PLUGINS` (e da pasta `plugins` ao lado do executável, no programa empacotado) é executado uma vez ao abrir; o que ele registrar aparece nas listas. Nada é carregado da pasta atual; plugin com erro não derruba o programa e fica em `api.PLUGIN_ERRORS`. `abrir_multical.py` aponta para a pasta `plugins/` do repositório, que traz `exemplo_log10.py`.
  - Nos dados reais `exp4..9` (79 amostras, corte 4400–7500 + Savitzky-Golay), k-fold 5: SVR 4,6 s, PLS e PCR 1 s, SPA 1,5 s. Validando por experimento (`loexo`) o R² de validação fica perto de zero ou negativo em todos os modelos, isto é, nenhum deles prevê bem um experimento que não participou do treino.
- **Ficou fora da fase 4:** receitas nomeadas salvas pelo usuário; Ridge, ElasticNet, kNN e RF de `run_loexo_model_comparison.py`; sugestão automática de validação por experimento; mostrar na tela os plugins que falharam; a caixa "Escolha da complexidade" continua visível para modelos em que ela não se aplica (SVR).
- **Fase 5 (distribuição): implementada em parte.**
  - `py packaging/build.py` gera `dist/Multical/` (PyInstaller 6.22, modo pasta, sem console), copia a pasta `plugins` e o guia para junto do executável, roda o autoteste do próprio executável e grava `dist/Multical-0.1.0-win64.zip`. Medido nesta máquina: pasta de 267 MB, zip de 117 MB; o build leva alguns minutos.
  - **Autoteste embutido** (`Multical.exe --autoteste relatorio.txt`, `src/multical_gui/selftest.py`): processos em paralelo (o que o PSO vai precisar), todos os modelos e validações registrados, modelo/tabelas/figuras/PDF, a janela e a pasta de plugins. O build para se o executável não passar. No executável gerado aqui, as cinco verificações passaram em 15 s, inclusive um plugin colocado na pasta ao lado do `.exe`.
  - **Instalador:** `packaging/installer.iss` (instala só para o usuário atual, sem senha de administrador; atalhos no menu Iniciar e na área de trabalho; em português). Compilado em 2026-11-13 com o Inno Setup 6.7.3 (instalado pelo winget, por usuário, com a sua autorização): `dist/Multical-0.1.0-instalador.exe`, 77 MB. O `build.py` compila o instalador sempre que encontra o Inno Setup.
  - **Instalador testado nesta máquina:** instalação silenciosa em uma pasta temporária, autoteste do programa instalado (cinco verificações, tudo certo), guia e pasta de plugins presentes, e desinstalação em seguida: a pasta, a entrada de desinstalação e o atalho do menu Iniciar sumiram. A primeira tentativa de rodar o instalador recém-compilado também foi recusada pelo Controle de Aplicativo do Windows; minutos depois ele rodou.
  - **Guia do usuário:** `docs/guia_do_usuario.html` (com capturas feitas a partir de dados sintéticos), aberto pelo botão "Abrir o guia do usuário" da tela inicial e incluído na pasta do programa.
  - No pacote, o plugin de exemplo vai como `_exemplo_log10.py` (desativado) com um `LEIA-ME.txt`.
  - **Dois defeitos de interface achados e corrigidos nesta fase:** (1) ao clicar em "Usar um modelo pronto", o foco do teclado podia ir para a barra lateral, que devolvia o programa à tela inicial sem mostrar isso (barra em "Início", botão Voltar sumido); (2) a largura mínima da janela era de 1627 px, maior que a tela de muitos notebooks; agora é de cerca de 1120 × 620. Ambos têm teste em `tests/test_gui.py`.
  - **Bloqueio do Windows observado:** logo depois do segundo build, o Windows recusou a primeira execução do `.exe` recém-gerado (erro 4551, "Uma política de Controle de Aplicativo bloqueou este arquivo"); minutos depois o mesmo arquivo rodou normalmente. O `build.py` agora espera e tenta de novo algumas vezes, e `py packaging/build.py --reaproveitar` refaz só a verificação e o zip. A proteção não foi desligada nem contornada. Risco para a distribuição: em máquinas com esse controle ativo, um executável sem assinatura digital pode ser recusado, e isso só se resolve com assinatura ou com a liberação por quem administra as máquinas.
  - **Não feito / não verificado:** instalar e rodar em outra máquina limpa (o critério da fase pedia duas; só esta foi testada); o comportamento do antivírus e do SmartScreen; assinatura digital; ícone próprio; o PSO em si dentro do `.exe` (ele ainda não está ligado à interface; só o mecanismo de processos em paralelo foi testado). O Python do build foi o 3.14 desta máquina, e não 3.12/3.13 como o plano sugeria.
- **Testes:** 95 passam (`py -m unittest discover -s tests -t .`), 1 a mais com `MULTICAL_SPP_SAMPLES` definido.

**Decisões registradas em 2026-11-11.**
- **Pergunta 2 (em parte):** os colegas vão tanto calibrar modelos quanto prever com modelos prontos. A tela inicial mantém as duas entradas com o mesmo peso e as fases 2 e 3 seguem na ordem do roteiro. Idioma da interface: ainda em aberto.
- **Pergunta 4a:** a normalização conjunta (calibração + amostras de validação) vale só dentro da validação cruzada; na validação externa e na previsão, usam-se apenas os parâmetros da calibração. Na prática o motor atual já normaliza X com o conjunto de calibração inteiro em cada fold, porque treino + validação do fold somam todas as amostras (`engine.py:233-235`); esse comportamento é mantido e os números de referência não mudam. Pendência: o modo holdout (`'Val'`, `engine.py:148-158`) hoje também normaliza com tudo junto; fica como está até você dizer se ele conta como validação externa.
- **Pergunta 4c:** `AutoScale` e `MeanCenter` passam a guardar a média e o desvio da calibração (método `fit` do contrato da seção 4) e a reaplicá-los na validação externa e na previsão. O `pipeline.py` atual não é alterado, para não mudar o resultado dos scripts.
- **Pergunta 4b:** um PLS por analito, tanto na validação quanto no modelo salvo. O erro mostrado passa a ser o do modelo que é de fato salvo. Consequência: o RMSECV da API nova não coincide com o de `MulticalEngine.run` (PLS conjunto); o teste de equivalência da fase 1 compara com `calculate_rmsecv_fast` chamado com uma única coluna de Y e com `train_and_save_model_pls` + `load_and_predict_pls`. Atenção: `calculate_rmsecv_fast(..., target_idx=j)` **não** é PLS por analito; ele ajusta o PLS conjunto e só isola o erro do analito j (`var_selection/utils.py:52`, `:95-96`). A seleção de variáveis por PSO/SA, que usa `target_idx`, otimiza portanto um modelo conjunto.
- **Fase 1:** concluída. `src/multical/api/` e `src/multical/io/txt_pair.py`; 34 testes passam (`py -m unittest discover -s tests -t .`). Nos arquivos reais `data/splits/*_cal.txt` (65 amostras), a API reproduz o código atual por analito com diferença relativa de 1e-15.
- **Pergunta 8:** resolvida sem versionar dados reais; os testes de referência usam dados sintéticos (`tests/_cases.py`).
- **Fase 0:** concluída no ramo `gui`; o leitor de `.spp` entrou em 2026-11-12.

1. **`.sp`**: pode fornecer os 5 itens da seção 6? → destrava a fase 0 e a tela 1.
2. **Quem faz o quê**: os colegas vão criar modelos ou principalmente prever com modelos que você cria? Interface só em pt-BR? → define a ordem das fases 2 e 3 e a tela inicial.
3. **Um espectro por `.sp`** e concentrações anotadas em planilha: confirma? Existe a noção de experimento/lote para cada amostra? → define a tabela de referência e a validação por experimento.
4. **Comportamentos numéricos do motor atual — manter (equivalência com resultados já publicados) ou corrigir na API nova?** → define o alvo do teste de equivalência.
   - a) A normalização de X em cada fold usa treino e validação juntos (`engine.py:233-235`).
   - b) O k é escolhido com PLS de todos os analitos juntos (`engine.py:178-182`, `:244`), mas o modelo salvo é um PLS por analito (`saving.py:34-49`); o erro reportado pode não corresponder ao modelo salvo.
   - c) `AutoScale`/`MeanCenter` não guardam o estado da calibração (`pipeline.py:53`, `:58`).
5. **Padrões para leigos**: qual receita de pré-tratamento e qual validação viram "recomendado"? Quais limites de erro/R² e de número de amostras acendem o aviso amarelo e o vermelho? → destrava as telas 3, 4 e 5.
6. **ANN/torch** precisa estar no executável padrão ou pode ser plugin opcional? Qual versão de Python usar no build? → define tamanho do .exe e a fase 5.
7. **Estrutura**: a GUI pode morar no mesmo repositório (`src/multical_gui/`), com `pyproject.toml` e imports relativos no núcleo? Isso altera como os scripts são chamados. → destrava a fase 0.
8. **Dados de teste**: posso versionar um subconjunto pequeno de `data/` em `tests/fixtures/` (hoje `data/` é ignorado, `.gitignore:161`)? → destrava o teste de equivalência.
