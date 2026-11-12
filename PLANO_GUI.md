# Plano — Multical com interface gráfica (desktop Windows, .exe)

Levantamento feito em 2026-11-11 por leitura do código (nada foi executado nem alterado).
Convenções: `arquivo:linha` aponta a evidência; `[não verificado]` marca o que não pude confirmar no repositório.

**Resumo em 6 linhas**
1. O repositório local está igual ao `origin/main`, mas ~5 meses de trabalho (SVR, PSO, LOEXO, KS) existem só na pasta de trabalho, sem commit.
2. A GUI atual (`gui_main.py`) não serve de base: quebra ao final de toda execução e oferece SPA/PCR que o motor não calcula.
3. Recomendação de stack: **PySide6 + matplotlib embutido + PyInstaller (onedir)**, com formulários gerados a partir de esquemas de parâmetros.
4. A extensibilidade vem de quatro registros (importadores, pré-tratamentos, modelos, validações); a GUI só lê os registros.
5. O maior bloqueio é o `.sp`: não há leitor nem arquivo de exemplo no repositório.
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
- Entradas: arquivos/pasta `.sp` (ou par `.txt` legado); planilha de referência.
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
- Entradas: arquivo `.multical` (ou `.pkl` antigo); arquivos `.sp`.
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

## 6. Importação de `.sp`

**Bloqueio.** Não há leitor nem arquivo `.sp` no repositório ou nos ramos remotos. O formato não pode ser confirmado sem exemplos.

**O que pedir a você (exatamente):**
1. 3 arquivos `.sp` reais, de dias ou experimentos diferentes.
2. Os mesmos 3 espectros exportados em texto/CSV pelo software do equipamento (gabarito para validar o leitor).
3. Marca, modelo e software do espectrômetro.
4. Unidade do eixo x (nm ou cm⁻¹) e se o eixo y é absorbância ou transmitância.
5. A planilha onde hoje são anotadas as concentrações medidas, como está.

**Como confirmar o formato** (fase 0, com os exemplos): abrir os primeiros bytes. Hipótese mais provável: `.sp` da PerkinElmer, que existe em variante binária (assinatura `PEPE`, estrutura em blocos) e em variante texto (cabeçalho `PE …` e seção `#DATA`) `[não verificado: conhecimento geral, sem arquivo para testar]`. Se for outro fabricante, esta seção é refeita.

**Opções de leitura.**
| Opção | Prós | Contras |
|---|---|---|
| Biblioteca existente (candidata: `specio`) | pronta; cobre outros formatos | manutenção e compatibilidade com Python atual `[não verificado]`; mais uma dependência no .exe |
| Leitor próprio em `multical/io/sp_reader.py` | sem dependência; controlado; testável contra o gabarito | escrever e manter ~100–150 linhas; cobre só as variantes que o laboratório usa |
| Exportar texto pelo software do equipamento | funciona já | passo manual a cada medida; contraria o objetivo |

Recomendação: testar a biblioteca nos 3 exemplos; se ler corretamente e instalar no Python do build, usar; caso contrário, leitor próprio. Em ambos os casos o teste automático compara com o gabarito do item 2 (tolerância 1e-6).

**Associação com os valores de referência.** Premissa a confirmar: um `.sp` contém um espectro, enquanto o formato atual guarda um experimento inteiro por arquivo, com as linhas de concentração na mesma ordem das linhas de espectro (`gui_main.py:345-365`; o motor só confere a contagem, `engine.py:66-68`).
- Chave de ligação: nome do arquivo sem extensão.
- Planilha de referência (`.xlsx` ou `.csv`): coluna `arquivo`, uma coluna por analito, colunas opcionais `experimento` e `tempo`.
- Botão **Gerar planilha-modelo**: cria a planilha já com os nomes dos arquivos importados; o colega preenche no Excel e recarrega.
- Relatório de ligação na tela: espectros sem valor, valores sem espectro, nomes duplicados.
- Ler `.xlsx` exige `openpyxl` (dependência nova; `.csv` não exige).

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
| **0 Preparação** | trabalho pendente em commit; pacote instalável; números de referência; leitor `.sp` provado | `pyproject.toml`; imports relativos nos 6 arquivos da seção 1; `tests/fixtures/`; `tests/golden/` | respostas 1, 7, 8 | formato `.sp` diferente do esperado | `pip install -e .` funciona; os 5 scripts principais rodam sem mudança de resultado; 3 `.sp` lidos batem com o gabarito |
| **1 API sem interface** | `multical.api` com `Dataset`, registros, `run_calibration`, `plots`; PLS + pré-tratamentos atuais + k-fold/holdout | `src/multical/api/*`, `src/multical/io/*`; núcleo intocado ou só refatorado internamente | fase 0; resposta 4 | divergência numérica ao separar cálculo de gráfico | teste de equivalência passa (abaixo); `import multical.api` não carrega `matplotlib.pyplot` |
| **2 MVP calibrar** | telas 0–6 com PLS; execução em segundo plano; exportação | `src/multical_gui/*` | fase 1; resposta 5 | usabilidade para leigos | um colega sem treinamento vai do `.sp` ao `resultados.xlsx` sem ajuda; um .exe de teste abre em máquina sem Python |
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

**Decisões registradas em 2026-11-11.**
- **Pergunta 2 (em parte):** os colegas vão tanto calibrar modelos quanto prever com modelos prontos. A tela inicial mantém as duas entradas com o mesmo peso e as fases 2 e 3 seguem na ordem do roteiro. Idioma da interface: ainda em aberto.
- **Pergunta 4a:** a normalização conjunta (calibração + amostras de validação) vale só dentro da validação cruzada; na validação externa e na previsão, usam-se apenas os parâmetros da calibração. Na prática o motor atual já normaliza X com o conjunto de calibração inteiro em cada fold, porque treino + validação do fold somam todas as amostras (`engine.py:233-235`); esse comportamento é mantido e os números de referência não mudam. Pendência: o modo holdout (`'Val'`, `engine.py:148-158`) hoje também normaliza com tudo junto; fica como está até você dizer se ele conta como validação externa.
- **Pergunta 4c:** `AutoScale` e `MeanCenter` passam a guardar a média e o desvio da calibração (método `fit` do contrato da seção 4) e a reaplicá-los na validação externa e na previsão. O `pipeline.py` atual não é alterado, para não mudar o resultado dos scripts.
- **Pergunta 4b:** um PLS por analito, tanto na validação quanto no modelo salvo. O erro mostrado passa a ser o do modelo que é de fato salvo. Consequência: o RMSECV da API nova não coincide com o de `MulticalEngine.run` (PLS conjunto); o teste de equivalência da fase 1 compara com `calculate_rmsecv_fast` chamado com uma única coluna de Y e com `train_and_save_model_pls` + `load_and_predict_pls`. Atenção: `calculate_rmsecv_fast(..., target_idx=j)` **não** é PLS por analito; ele ajusta o PLS conjunto e só isola o erro do analito j (`var_selection/utils.py:52`, `:95-96`). A seleção de variáveis por PSO/SA, que usa `target_idx`, otimiza portanto um modelo conjunto.
- **Fase 1:** concluída. `src/multical/api/` e `src/multical/io/txt_pair.py`; 34 testes passam (`py -m unittest discover -s tests -t .`). Nos arquivos reais `data/splits/*_cal.txt` (65 amostras), a API reproduz o código atual por analito com diferença relativa de 1e-15.
- **Pergunta 8:** resolvida sem versionar dados reais; os testes de referência usam dados sintéticos (`tests/_cases.py`).
- **Fase 0:** concluída no ramo `gui`, exceto o leitor de `.sp` (pergunta 1).

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
