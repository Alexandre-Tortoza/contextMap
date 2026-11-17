# Handoff — `refactoring/visual_perception` (PR #695)

> Estado em 2026-10-03. Branch `refactoring/visual_perception`, base `dev`, head `81db3689`,
> tudo em `origin`. Nada pendente no working tree.

Este documento existe para quem continua. Ele diz **o que está medido**, **o que falta**,
**o que já foi decidido e não precisa ser rediscutido**, e **três armadilhas de método** que
custaram tempo nesta sessão.

---

## 1. Estado medido

|  | base `4089223` | head `81db3689` |
|---|---:|---:|
| `pytest` | 0 failed, 8570 passed | **1 failed, 8662 passed**, 18 skipped |
| `mypy src tests` | limpo | **limpo** (808 arquivos) |
| `ruff check` / `format --check` | limpo | **limpo** |
| violações de boundary | não enforced | **38** (33 public-api, 5 private) |

A única falha é `tests/architecture/test_boundaries.py`, vermelha pelas 38 violações
declaradas. **É o único item que bloqueia o merge.**

Comandos que reproduzem:

```bash
pytest -q -p no:randomly -p no:cacheprovider --basetemp="$HOME/.cache/cm2/bt"
mypy src tests
ruff check src tests && ruff format --check src tests
```

O pipeline **executa ponta a ponta**, o que não acontecia antes nesta branch:

```text
preflight: {"ok": true, "problems": []}
executors.missing: ingestion      # por projeto: precisa de um IngestionRequest, não de config
```

---

## 2. O que falta — único bloqueio, 38 violações

`pytest tests/architecture/test_boundaries.py` imprime a lista completa. Ela se divide em duas
naturezas, e só a segunda pede decisão.

### 2.1 Repoint mecânico de import (~30)

Um consumidor alcança o submódulo em vez da raiz pública da capability dona, e **o nome já está
exposto na raiz** (ou basta acrescentá-lo ao `__all__` do owner). São:

```text
visual_perception  ->  region_perception.refinement.region_refinement        4
                       region_perception.grounding.region_grounding          4
                       region_perception.region_evidence                     3
                       region_perception.discovery.region_discovery          3
                       region_perception.artifacts.mask_store                2
                       region_perception.ports                               1
                       region_perception.region_semantic_hint                1
                       region_perception.region_eligibility                  1
                       region_perception.discovery.region_normalization      1
                       semantic_scoring.contracts                            2
                       semantic_scoring.ports                                1
                       visual_perception.visual_features.visual_evidence     2
                       visual_perception.semantic_interpretation             1
```

Já fiz exatamente esse movimento 23 vezes nesta branch (commit `5e8edab8`). Padrão: trocar o
caminho do import; onde o nome não estiver no `__all__` do owner, acrescentá-lo lá — **não**
criar um atalho.

### 2.2 Precisa decisão de ownership (8)

```text
region_perception  ->  visual_perception.artifacts.perception_serialization  6
                       visual_perception.evidence_identity                   2
```

São o codec e os geradores de identidade de **região** morando em `visual_perception`:
`encode_region`/`decode_region` de `Region2D`, e `region_id_for`,
`grounding_region_id_for`, `refined_region_id_for`. Por ownership (§3) pertencem a
`region_perception`.

É a mesma forma do move que deu certo em `5e8edab8` (`region_geometry.py`), com um detalhe: os
geradores de id são o assunto do **passo 3 do ADR** (ver §4), então mover agora pode conflitar
com aquela decisão. Duas saídas defensáveis:

- mover o codec agora (6 violações) e deixar `evidence_identity` para o passo 3; ou
- declarar as 8 numa baseline que só aperta, e resolvê-las com o passo 3.

### 2.3 As 5 `private-cross-module-import`

```text
semantic_scoring/adapters/{clip,alphaclip}_text.py  ->  visual_features.adapters._feature_values
                                                        visual_features.adapters.{clip,alphaclip}
visual_perception/adapters/sam2.py                  ->  region_perception.adapters.prepared_images
```

As torres de texto precisam do espaço de embedding da torre visual correspondente, e `sam2.py`
nunca migrou (ver §3). Resolver as duas primeiras provavelmente significa expor o espaço de
embedding pela raiz de `visual_features`; a terceira vai junto com a migração do `sam2.py`.

---

## 3. Dívida declarada (não bloqueia, se aceita como declarada)

- **`sam2.py` nunca migrou.** É o único adapter que satisfaz `RegionRefinement` e continua em
  `visual_perception/adapters/`; o catálogo aponta para lá.
- **Buraco residual da #545.** `_check_pinned_upstream` inspeciona só dependências de nós
  *tocados*, então um stage pinado a jusante de um stage variado, que nenhuma variável toca,
  nunca é examinado. Nenhuma fase atual exercita. Corrigir exige uma regra em `experiments.py`.
- **`visual_perception/docs/` tem 33 documentos de assunto** que não seguiram o código.
- **`experiments/*/commands.sh`** em quatro diretórios de campanha usam os ids de componente
  anteriores ao rename. São registros de comandos que rodaram contra campanhas reais, então
  reescrevê-los mentiria sobre o que executou (§23). Não resolvem mais, e esse é o estado
  honesto de um registro cujo schema mudou embaixo dele.
- **`BackendScore` é reduzido a `float`** na normalização, perdendo `name` e `semantics` antes
  de chegar à auditoria.
- **`region_perception.__all__` tem 74 símbolos.** Medido contra os imports reais, nenhum está
  morto — mas só sete têm consumidor cross-capability em `src/`; o resto chega pela fachada
  `visual_perception` ou pela suíte da própria capability, que ainda vive em
  `tests/visual_perception/`. Estreitar exige mover os testes primeiro.

---

## 4. Decisões já fechadas — não rediscutir

Estão em [`docs/perception-dag-decomposition.md`](docs/perception-dag-decomposition.md), que é
decisão fechada, não especulação. Resumo do que **não** é mais pergunta:

### O critério que substitui "raiz = gera artifact"

```text
Capability   ownership de conceitos e API pública; é o que justifica o pacote na raiz
Stage        unidade de execução cuja saída precisa ser selecionável no DAG (decisão de catálogo)
Artifact     identidade imutável e endereçável da saída de uma execução de stage
Evidence     dado com provenance, que pode estar dentro de um artifact sem ser um
View         composição derivada; não cria verdade nova e não tem produtor
Cache        otimização abaixo do contrato do DAG; nunca é dependência downstream
```

A convenção antiga era falsa: `shared` e `model_runtime` são pacotes de topo e não produzem
artifact nenhum.

Promoção a stage exige **as quatro**: (1) um consumidor precisa selecionar aquela saída como
upstream; (2) a saída precisa ser variada ou pinada independentemente; (3) reexecutar não é
equivalente a reutilizar; (4) existe consumidor **hoje**.

Artifact vs cache **não** é determinismo — é se a identidade daquela execução precisa ser
selecionada, citada ou pinada. Determinismo é pré-condição de *cache*, não critério de
artifact.

### As cinco decisões por componente

| Componente | Decisão | Estado |
|---|---|---|
| `region_perception` | stage de topo, com executor | **feito** (`81db3689`) |
| `visual_features` | submódulo; embeddings como evidence, sem artifact próprio | feito |
| `semantic_interpretation` | stage de topo, com claims + `scene_context` | **passo 4** |
| `semantic_scoring` | stage consumindo claims + evidência visual | parcial |
| `visual_perception` | stage, estreitado; `PerceptionResult` vira view | **passo 4** |

### O shape do passo 1, e por que esse

O stage de região roda **só a descoberta**; grounding e refinamento são componentes do stage
`visual_perception`. Não é concessão: o catálogo já declarava `visual_features.dense` e
`semantic_interpretation.interpreter` como componentes daquele stage. Capability diz de quem é
o conceito; pertinência ao stage diz onde roda.

A alternativa (o stage de região rodando os três) mudaria o preset da percepção, que entra na
identidade de run e no manifest, e regravaria todos os goldens. Este shape não muda nada disso
**e** entrega o ganho de ablação: trocar backend de feature deixa o digest e o conteúdo do
artifact de região idênticos, então o **SAM2 não roda de novo**.

### `scene_context` fica com as claims

Mesmo backend, mesma `SemanticInferenceProvenance`, e as requisições de região são
condicionadas nele (#529). Com uma assimetria registrada no ADR: `scene_interpretation` depende
só de `image`, não de regiões, então um artifact com upstream de região faz o contexto de cena
herdar dependência que ele não tem. Desperdício limitado (uma chamada por frame, não por
região) e aceito; a condição que o promove está escrita lá.

---

## 5. Decisões abertas — do mantenedor

1. **As 8 violações de §2.2**: mover o codec agora, ou declarar baseline e resolver com o passo 3.
2. **Passo 3 — a cadeia de identidades.** `RegionId` deriva de `PerceptionResultId`, que deriva
   do id do run **fundido**; e `perception_result_id` é campo declarado em **31 contratos** em
   `sensor_association`, `semantic_mapping`, `semantic_fusion` e `evaluation`. Ou é redefinido
   por produtor — e deixa de ser chave compartilhada —, ou é substituído a jusante por
   `region_id` + `source_observation_id`, que claims e features já carregam. É o maior custo da
   migração e determina a ordem do resto.
3. **Passo 4** — quebrar o `PerceptionRunArtifact` em `VisualEvidenceArtifact` +
   `SemanticInterpretationArtifact`, e `PerceptionResult` virando view. Depende de 2.
4. **Contrato de release.** O cenário vivo é **1.0.6**; o contrato de release do v0.1.0 continua
   sendo **1.0.5**, e os dois documentos de release dizem isso explicitamente, com
   `test_release_scope.py` exigindo que a divergência fique escrita enquanto existir. Se o
   release for refeito contra 1.0.6, é decisão de release, não de refatoração.

---

## 6. Três armadilhas que custaram tempo

### 6.1 Baseline em worktree é inválida neste repo

O pacote está instalado em modo editable apontando para o checkout primário, então uma
`git worktree` de outro commit roda os **testes daquele commit contra o `src/` do branch
atual**. Isso produziu "618 falhas na base" e deltas de "582 corrigidos / 1 regredido" que
foram publicados na descrição da MR antes de alguém notar. A base real está **verde**.

```bash
# correto
PYTHONPATH=<worktree>/src pytest -q
MYPYPATH=<worktree>/src PYTHONPATH=<worktree>/src mypy src tests
# conferir sempre
cd <worktree> && python -c "import contextmap; print(contextmap.__file__)"
```

E comparar **por nome de teste**, nunca por contagem — lembrando que um teste que falha nos dois
lados por motivos diferentes não conta como regressão nessa comparação, o que esconde quebra
nova.

### 6.2 Um golden re-gravado esconde a mudança que motivou a re-gravação

O defeito central desta branch não foi um conjunto de bugs — foi método. Dois commits deletaram
módulos e escreveram substitutos mais pobres, e cada caso ficou invisível porque o teste que o
protegia não coletava. O padrão apareceu **quatro vezes**: `Region2D`, o `mask-index.jsonl`, os
diagnósticos de feature, o `SemanticScorer`. E o sinal estava sempre lá — um golden re-gravado,
com justificativa que raciocinava sobre a mudança errada.

As restaurações se provam: os 173 goldens de normalização e os dois testes de identidade byte a
byte do artifact de percepção voltaram a bater com a gravação anterior à decomposição **sem
re-gravar nenhum**. É o teste que vale.

> **Um digest que muda é uma pergunta, não um passo.**

### 6.3 Com várias sessões no mesmo checkout

- O **cache do mypy mente**: 9 erros reportados como pré-existentes eram `.mypy_cache` obsoleto.
  `rm -rf .mypy_cache` antes de acreditar.
- **Cross-talk de pytest**: 27 falhas e 57 erros evaporaram numa reexecução. Use
  `-p no:cacheprovider --basetemp=<dir próprio>`.
- **`/tmp` é recurso compartilhado** e estourou (EDQUOT) com suítes concorrentes. Escreva saída
  grande fora dele (`$HOME/.cache/...`); `--basetemp` sozinho não resolve se a saída também vai
  para `/tmp`.
- **Nunca `git stash`** em trabalho paralelo: numa sessão anterior um `stash` capturou trabalho
  em voo de outro agente e o `pop` falhou em silêncio.

---

## 7. Commits desta fase

```text
09aeb191  artifact de região como fronteira tipada e atômica; dois pacotes sem consumidor removidos
978ab2c9  cinco contratos restaurados (port do scorer, enums do manifest, iter_scores, ancoragem, entry)
8b03ea3b  o ADR: critério capability/stage/artifact/evidence/view/cache
fa557d58  Region2D restaurado — provado byte a byte
9087da8f  docs da capability alinhados ao contrato real
1e1c20d2  mypy 217 -> 0; checagem de tipos devolvida a ~170 módulos
28e5bc6e  chaves públicas de configuração migradas; examples/v0.2.0 congelado
0acdceea  cenário 1.0.6, com 1.0.5 preservado como contrato de release por escrito
be34d87b  diagnósticos de feature restaurados (ABSTAINED, __DEBUG__, quatro streams)
5e8edab8  geometria de região movida para o owner — 61 -> 38 violações
81db3689  o executor de região; pipeline executa ponta a ponta
```

Guardas de arquitetura acrescentados nesta fase, que impedem a reincidência dos defeitos:

- `tests/architecture/test_runtime_type_guards.py` — recusa um `isinstance` cujo tipo só existe
  sob `TYPE_CHECKING`, e um tipo diferido substituído por `None`. Validado vermelho contra os
  dois defeitos originais, com `ruff` passando no estado quebrado: é exatamente o que lint e
  mypy não veem.
- `tests/architecture/test_compatibility_facade.py` — todo nome do `__all__` resolve, nenhuma
  chave duplicada no mapa lazy, e os contratos movidos são o **mesmo objeto** nos dois caminhos
  de import.
