# Experimentos, matrizes de ablação e execução

Comparar modelos, políticas, canais de evidência ou estágios opcionais do DAG só faz sentido quando **tudo, exceto a variável declarada, é idêntico**. Este módulo define o manifesto de experimento (`contextmap.evaluation.experiments`), a execução controlada (`contextmap.evaluation.experiment_runner`) e o esqueleto comum dos drivers de experimento sobre dados reais (`contextmap.evaluation.experiment_driver`, ver [Drivers de experimento e retenção dos braços](#drivers-de-experimento-e-retenção-dos-braços)).

Resolver uma configuração em um DAG é responsabilidade do `runtime` (`resolve_plan`), do qual este módulo não depende; gerar os arms de um experimento a partir de configurações resolvidas é trabalho das fases (#527, ver [`experiment-phases.md`](experiment-phases.md)). A avaliação **recebe topologias já resolvidas** e apenas verifica que elas diferem só como declarado; o executor de cada arm é injetado (`ArmExecutor`).

## Manifesto

`ExperimentManifest` (schema `contextmap.experiment/v2`) é imutável, versionado e hasheado, e amarra:

| Campo | Conteúdo |
|---|---|
| `selection` | reference set (`id`, `version`, `digest`), scheme, split e a **lista ordenada exata** de amostras |
| `evaluated_stage` | o estágio cuja qualidade todo relatório mede |
| `base_configuration_digest` | digest da configuração efetiva base/default |
| `variables` | as variáveis sob teste, com tipo, estágios que podem tocar, valores e baseline |
| `mode` e `arms` | a matriz de ablação e, para cada célula, a **topologia resolvida** (estágios, backend, modelo, digest de configuração e, opcionalmente, a própria configuração campo a campo, dependências e o run de estágio que computa cada estágio junto com os irmãos, `execution`) |
| artefatos upstream | por estágio, o `ArtifactIdentity` (com digest `sha256:<64 hex>` válido) **reaproveitado em vez de recalculado** |
| `fixed_controls` | valores constantes entre arms (versão de prompt, seed, variante de evidência, …) |
| `quality_metrics` / `resource_capture` | métricas de qualidade e de performance exigidas em todo relatório (política de captura de runtime, memória, storage e throughput) |
| `registry` | identidade do registro de métricas ([`metrics.md`](metrics.md)) |
| `repetitions_per_sample` | inferências repetidas por amostra: evidência correlacionada, **nunca** novas amostras físicas |

## Só as variáveis declaradas variam

O manifesto só se constrói se for uma comparação controlada:

1. **Matriz.** Os arms são exatamente as células de `ablation_cells()`: `one_at_a_time` (baseline mais um arm por variável e valor não baseline) ou `full_factorial` (todas as combinações); ou, em `selected`, o baseline mais uma lista explícita de células distintas. O baseline segura todos os valores baseline.
2. **Diferenças declaradas.** Cada diferença entre um arm e o baseline precisa ser coberta por uma variável que o arm atribui a um valor não baseline, **de um tipo que a permita**:

   | Diferença | Tipo de variável exigido |
   |---|---|
   | estágio inserido, removido ou dependência religada | `topology` |
   | run de estágio (`execution`) de um estágio | `topology` |
   | backend ou modelo de um estágio | `backend` |
   | digest de configuração (estágio sem configuração registrada campo a campo) | `backend`, `policy`, `configuration` ou `evidence_channels`, **sem** `configuration_fields` (um campo declarado não pode ser verificado num digest opaco) |
   | campo de configuração registrado | uma variável desses tipos que **declara o campo** em `configuration_fields` (ver [Braços pareados](#braços-pareados-matched-arms)) |
   | artifact pinado de um estágio | qualquer variável que toque o estágio |
   | capability de um estágio | nunca (troque removendo e inserindo) |

   Uma mudança fora dessas regras é `undeclared change`, e a mensagem lista **todas** as identidades divergentes com os dois valores. Uma variável com valor não baseline que **não muda nenhum** dos estágios que toca também é recusada.
3. **Entrada imutável.** Toda dependência upstream de um estágio tocado por uma variável é um **artifact pinado**, então todos os arms a compartilham comprovadamente. Um arm não pode trocar o artifact pinado de um estágio não tocado.
4. **Pontos de variação de um run variado são pinados pela identidade (#660).** Um `TopologyStage` pode nomear o run de estágio que o computa junto com os irmãos (`execution`): o run de percepção computa descoberta, grounding, refino, features e interpretação de uma vez. Quando uma variável toca um estágio de um run, o run inteiro roda de novo em todo arm, e os irmãos não tocados não podem ser um artifact reaproveitado. Eles ficam **pinados pela identidade**: a regra 2 já exige backend e configuração iguais em todo arm, então uma dependência upstream num run variado dispensa o artifact. Um run que nenhuma variável toca continua exigindo o artifact pinado, como antes.
5. **Um artifact pinado é uma computação.** Todo arm que reaproveita um artifact o reaproveita para os mesmos estágios, com a mesma implementação. Pinar um run de estágio enquanto uma variável muda um estágio dentro dele faria um arm reaproveitar a saída de outro (o arm com grounding reaproveitaria o run de percepção sem grounding) e é recusado.

`ExperimentVariable.kind` cobre os tipos de variação pedidos: backends (Qwen × Gemini, backends de Region Discovery, CLIP × AlphaCLIP, Point Representation off × geométrico × PTv3), políticas (fusão uniforme × ciente de qualidade), subconjuntos de canais (Semantic Fusion e Entity Resolution) e topologia.

### Ablação de estágio do DAG

```text
Arm baseline:  DINO artifact X  ->  Sensor Association
Arm enhanced:  DINO artifact X  ->  FeatureResolutionEnhancement  ->  Sensor Association
```

A variável `feature_resolution` (`topology`) toca `feature_resolution_enhancement` e `sensor_association`. O manifesto só é válido se o estágio DINO estiver pinado com o **mesmo** artifact X nos dois arms; a comparação registra o artifact compartilhado.

## Validação contra reference set e registro

`validate_experiment_manifest()` confere:

- a seleção cita a versão/digest do reference set e **reproduz** exatamente a lista ordenada do split; o papel do split declarado confere;
- um experimento de **tuning nunca usa o split held-out** (`test`): hiperparâmetros não são ajustados em dados de avaliação retidos;
- o manifesto cita o registro de métricas usado; cada métrica existe, é de qualidade (ou de performance, em `resource_capture`), pertence ao `evaluated_stage` e o reference set tem os schemas de anotação que ela exige (`MetricCompatibilityError` caso contrário).

## Execução

`run_experiment(manifest, executor=…, registry=…, reference_set=…, root=…)`:

- exige um `ValidatedReferenceSet` ([`reference-integrity.md`](reference-integrity.md)): a integridade do reference set é pré-requisito, e valida o manifesto **antes** de executar qualquer arm;
- exige um `root` novo: reexecutar cria outra identidade, nunca sobrescreve;
- executa o baseline primeiro, depois os demais; um arm que falha **não interrompe** os outros.

```text
<root>/
├── experiment.json            # o manifesto, com digest
├── arms/<arm_id>/run.json     # run manifest do arm
├── arms/<arm_id>/report.json  # relatório no envelope comum (só arms concluídos)
└── comparison.json            # manifesto de comparação
```

O **run manifest** de cada arm preserva a topologia resolvida, backend, modelo, digest de configuração, os artifacts por estágio, a identidade do reference set e do registro, e o resultado. Não há timestamps: a mesma entrada produz os mesmos bytes.

### Arms que falham ou ficam indisponíveis

| Situação | Registro |
|---|---|
| o executor levanta `ArmUnavailableError` (backend, modelo ou API externa indisponível) | `unavailable`; **nenhum fallback** para outro backend |
| o executor levanta outro erro | `failed`, com tipo e mensagem |
| o resultado contradiz o manifesto | `failed` com `invalid_result` e o motivo |

Um resultado é inválido quando: avalia outro estágio; cita outro reference set ou registro; seu digest de configuração não é o da topologia do arm; não cita este manifesto exato entre as entradas; é inconsistente com o registro; não traz uma métrica de qualidade ou de recursos declarada; não reporta o artifact de cada estágio; ou **não reaproveitou um artifact pinado**.

Um arm que não concluiu não tem métricas, aparece na comparação com `comparable: false` e torna a comparação **incompleta** (`complete: false`). `require_complete_comparison()` recusa uma comparação incompleta.

## Comparação

O `ComparisonManifest` lista:

- os arms (assignments, digest da topologia, digest de configuração e artifact por estágio, status, falha e o pareamento com o baseline), as variáveis e os controles fixos;
- `matching`: quantos pares estão `matched`, `invalid` e `incomplete`, e os arms excluídos das métricas ([Braços pareados](#braços-pareados-matched-arms));
- os **artifacts compartilhados**: os que todo arm concluído reporta idênticos para o mesmo estágio (a prova de que só o declarado difere; em uma ablação de DAG, o artifact upstream compartilhado);
- cada métrica declarada **lado a lado**, por estrato, só para os arms comparáveis (concluídos e não inválidos): valor, status (`value`/`not_applicable`/`unsupported`), `sample_count` e a diferença **para o baseline na mesma métrica**; `same_population` indica se todos os valores vieram do mesmo número de amostras e é `false` quando algum valor não informa `sample_count`, porque população desconhecida não é evidência de população comum;
- `physical_sample_count` (amostras físicas distintas) e `repetitions_per_sample`, separados.

**Não há score geral nem vencedor.** Métricas distintas nunca são agregadas, qualidade e performance ficam em `kind` diferentes, e nada ranqueia os arms; a decisão de manter, adiar ou promover uma configuração é humana e explícita.

## Braços pareados (matched arms)

Uma ablação que diz variar um fator (prompt, vista, contexto de cena, orçamento visual, backend, tarefa nativa) só é interpretável se os arms não diferirem em mais nada. O pareamento é verificado em dois momentos: **no plano**, quando o manifesto é construído (falha com `ExperimentError`), e **na execução**, quando a comparação é construída (o par fica `invalid` e sai das métricas). Nenhum dos dois agrega em silêncio.

### No plano: campos de configuração declarados

Um digest de configuração opaco não distingue "mudou o prompt" de "mudou o prompt **e** a revisão do modelo". Por isso:

- `StageImplementation.configuration` guarda a configuração efetiva do estágio **campo a campo** (JSON). Quando presente, `configuration_digest` é o digest exatamente desses campos (`StageImplementation.from_configuration()` o calcula; um digest que não confere é recusado), então nenhum campo pode variar fora do que está registrado. Um estágio registra a configuração campo a campo em todos os arms ou em nenhum.
- `ExperimentVariable.configuration_fields` são os caminhos pontuados que a variável declara mudar nos estágios que toca. Um caminho cobre a si e tudo abaixo dele (`interpreter` cobre `interpreter.qwen.revision`). Só variáveis `backend`, `policy`, `configuration` e `evidence_channels` declaram campos; uma variável `topology` não.
- Um campo que diverge do baseline sem uma variável atribuída de forma diferente, que toque o estágio e declare o campo, é `undeclared change`. Folhas são comparadas pelo JSON canônico (`1`, `1.0` e `true` diferem) e uma lista é uma folha só, com ordem (a política ordenada de vistas).
- `arm_differences(variables, reference, arm)` devolve cada diferença planejada entre dois arms como `ArmDifference` (estágio, `DifferenceKind`, campo, valor de referência e valor em JSON canônico, variáveis que a declaram). O manifesto a usa para recusar o que não é declarado, e a comparação a usa para registrar o que é.

Exemplo de ablação de prompt no estágio semântico:

```json
{
  "request_policy": {
    "prompt_policy": "region/v1",
    "view_policy": ["tight_crop"],
    "scene_context": "none",
    "output_schema": "semantic-response/1"
  },
  "interpreter": {
    "backend": "qwen",
    "qwen": {"model": "Qwen/Qwen3-VL-4B-Instruct", "revision": "<sha>", "max_new_tokens": 256, "temperature": 0.0}
  }
}
```

| Variável | Tipo | `configuration_fields` | Pode mudar |
|---|---|---|---|
| `prompt_policy` | `policy` | `request_policy.prompt_policy` | só o prompt |
| `view_policy` | `evidence_channels` | `request_policy.view_policy` | só a lista ordenada de vistas |
| `semantic_backend` | `backend` | `interpreter` | backend, modelo, revisão e parâmetros do intérprete; nunca a `request_policy` |

Com isso, uma ablação de prompt que também muda `interpreter.qwen.revision`, `request_policy.output_schema` ou `interpreter.qwen.max_new_tokens` é recusada nomeando o campo e os dois valores; e uma troca Qwen × Gemini só é válida se prompt, vistas, contexto e schema de saída forem os mesmos. A política de conceito do SAM3, o contexto de cena, o orçamento de tokens/resolução e a tarefa nativa são declarados do mesmo jeito, como campos da configuração do estágio que os consome.

Onde cada identidade exigida pelo #545 está registrada e como é verificada:

| Identidade | Registro | Verificação |
|---|---|---|
| reference set congelado | `SelectionBinding.reference_set` (id, versão, digest); `ReproducibilityMetadata.reference_set` de cada relatório | um manifesto tem uma única seleção; relatório com outro reference set → arm `failed` (`invalid_result`) |
| dataset/seleção e observações físicas | `SelectionBinding` (scheme, split, lista ordenada de amostras); cada `ReferenceSample` guarda seus `observation_ids` | compartilhados por construção; a sequência (ingestão) é artifact pinado reaproveitado ou estágio não afetado com o mesmo conteúdo |
| calibração/pose | `ReferenceSample.calibration_ids` (com `content_hash` no reference set); artifacts de ingestão e de estimação de estado | idem: pinados ou não afetados com o mesmo conteúdo |
| artifacts upstream | `TopologyStage.artifact` (plano) e `StageArtifact` (execução) | pinado igual no plano e reaproveitado na execução; estágio não afetado com o mesmo `kind` + `digest` |
| pontos de variação de um run variado | `TopologyStage.execution` e a implementação do estágio (plano); `StageArtifact` (execução) | backend e configuração iguais em todo arm no plano; na execução continuam estágios não afetados, então cada um precisa reportar a **própria** saída (a auditoria da descoberta, por exemplo), comparada por `kind` + `digest` |
| backend/modelo | `StageImplementation.backend_id`, `backend_version`, `model` | só uma variável `backend` |
| revisão/checkpoint, schema de saída/parser, geração, política de prompt, de vistas, contexto de cena, orçamento visual, tarefa nativa | campos de `StageImplementation.configuration` | só a variável que declara o campo |
| evaluator, código e schemas de anotação | `ReproducibilityMetadata` do relatório | iguais entre o arm e o baseline |

### Na execução: cada arm contra o baseline

`build_comparison()` compara cada arm concluído com o baseline e grava o resultado em `ComparisonArm.match` (`ArmMatch`; o baseline não tem):

| `MatchStatus` | Quando |
|---|---|
| `matched` | ambos concluíram e o que foi executado difere só onde os fatores declarados permitem |
| `invalid` | ambos concluíram, mas um estágio **não afetado** pelos fatores (nem tocado, nem a jusante de um tocado) produziu ou reaproveitou outro conteúdo, ou o relatório vem de outro evaluator, outra versão de código ou outros schemas de anotação |
| `incomplete` | o arm ou o baseline não concluiu (erro, OOM, indisponível); o motivo fica registrado |

Artifacts de estágios não afetados são comparados por `kind` + `digest`: uma reexecução determinística tem outro id e o mesmo conteúdo, e uma reexecução não determinística (ou sobre outra sequência) invalida o par — o estágio deveria ter sido pinado. Um arm `invalid` mantém `run.json` e `report.json`, mas **não entra nas métricas**; `comparable` passa a ser "concluído e não inválido", e `require_complete_comparison()` recusa o par com as identidades divergentes.

`ArmMatch` preserva `factors` (a atribuição dos fatores que diferem do baseline), `declared_differences` (do plano), `undeclared_differences` (da execução, com os dois valores), `matched_sample_count` e `reason`. `ComparisonArm` traz também `stage_configurations` (digest de configuração por estágio) e `stage_artifacts` (id e digest do que cada estágio produziu ou reaproveitou). `comparison.json` ganha `matching`: contagens `matched`/`invalid`/`incomplete` e `excluded_arm_ids`.

### Inferência repetida não vira amostra

`matched_sample_count` conta **amostras físicas** da seleção (`physical_sample_count`); é zero fora de `matched`. Variantes de prompt ou de vista sobre o mesmo frame ou região e as `repetitions_per_sample` são inferência repetida sobre a mesma observação física: três prompts × três repetições sobre duas amostras continuam duas amostras pareadas. A mesma regra vale dentro dos estágios: no relatório de Semantic Interpretation, `repeat_index > 0` mede estabilidade e não entra na qualidade, e `compare_evidence_variants()` pareia variantes por observação e região física; em Semantic Fusion, `group_by_physical_observation()` agrupa por observação física, e runs com outra configuração (outro prompt) são outra variante de inferência, nunca outra observação.

### Arms que falham continuam no resultado

Um arm que falhou, estourou memória ou ficou indisponível aparece em `arms` com status e motivo, com o par `incomplete` e seus `factors` e `declared_differences`; nunca é removido para fabricar um conjunto pareado. A comparação fica `complete: false`.

## Fases

As fases do #521 declaram fatores sobre a configuração do runtime e são expandidas em arms deste manifesto por [`experiment-phases.md`](experiment-phases.md); o modo `selected` roda o baseline e só as células escolhidas.

## Matriz de capacidades

Quais backends e composições podem entrar num arm, e quais operações se comparam diretamente, está em [`capability-matrix.md`](capability-matrix.md) (#522): um manifesto só deve combinar arestas `supported` e comparar entradas do mesmo grupo de comparação.

## Drivers de experimento e retenção dos braços

Uma execução real sobre dados reais é feita por **drivers**: scripts de pesquisa sob `experiments/<experimento>/` que montam à mão o `StageRequest` de um estágio, ou de um braço de um estágio, e chamam o executor. Cada rodada copiava os drivers e editava neles as mesmas três coisas: o número do run embutido no diretório de saída, nas `location` dos `ArtifactRef` e no `config_digest`; o caminho do workspace; e o runner que mede tempo e memória de um processo filho (achados EV-05 e EV-06 da auditoria 2026-09, issue #620). `contextmap.evaluation.experiment_driver` é a versão única e testada dessas três coisas, e só delas. O executor, as políticas e as entradas de cada estágio continuam no driver, porque são o assunto de cada experimento. Não há framework, registry nem plugin.

As cópias históricas em `experiments/*/scripts*` e os reports existentes ficam **intactos**, como evidência do que cada rodada executou. Só experimentos novos usam o esqueleto.

### Layout de um run: `DriverRun`

| Valor | Para `DriverRun(experiment="e2e-real", run_number=3, outputs_root=R)` |
|---|---|
| `run_id` | `run-0003` |
| `output_dir(nome)` | `R/e2e-real/run-0003/<nome>`, o diretório que o writer da capability cria e finaliza |
| `location(nome)` | `e2e-real/run-0003/<nome>`, relativo a `R`: a `location` do `ArtifactRef` e o `run_dir` que o report registra |
| `config_digest(nome)` | `e2e-real/run-0003/<nome>` |

`<nome>` é o id do estágio ou, quando vários braços do mesmo estágio rodam lado a lado, o id do braço. O layout é o da runtime (`<workspace>/<dataset>/run-NNNN/<estágio>`, ver [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)) com `R` como workspace: `StageRequest.run_number()` lê o número de volta do `output_dir`, e `StageRequest.directory_of()` abre a `location`. O experimento, o nome e o número do run são validados: um nome com separador, `.` ou `..`, um número que não é inteiro positivo e uma raiz relativa são recusados.

Uma rodada nova muda **só `run_number`**. O `config_digest` é um rótulo, porque um driver montado à mão não tem documento de configuração efetiva para hashear. Ele depende do run, para que uma reexecução tenha outra identidade. Depende também do nome do artifact porque a identidade de execução da runtime é estágio + `config_digest` + hashes das entradas: dois braços do mesmo estágio sobre as mesmas entradas, diferentes só na configuração, teriam a mesma identidade com um rótulo por run. Os drivers do run real canônico usavam um rótulo por run, o que bastava lá porque cada estágio era diferente.

Esboço de um driver novo:

```python
from pathlib import Path

from contextmap.evaluation import DriverRun
from contextmap.runtime import ArtifactRef, StageRequest

RUN = DriverRun.for_driver(Path(__file__), experiment="e2e-real", run_number=6)

request = StageRequest(
    stage_id="sensor_association",
    inputs={
        "trajectory": (
            ArtifactRef(
                stage_id="state_estimation",
                contract="StateEstimationRunArtifact",
                artifact_id=TRAJECTORY_ARTIFACT_ID,
                content_hash=TRAJECTORY_CONTENT_HASH,
                location=RUN.location("state_estimation"),
            ),
        ),
        # ... as demais entradas do estágio
    },
    components={},
    config_digest=RUN.config_digest("sensor_association"),
    output_dir=RUN.output_dir("sensor_association"),
    workspace=RUN.outputs_root,
)
```

### Retenção dos braços sob `outputs/`

Os artifacts dos braços de um experimento são **retidos**. Sem eles, o experimento pode ser reexecutado, mas não reinspecionado: auditar as associações de um braço custaria a execução inteira de novo. Foi o que aconteceu com `experiments/sensor-association-scaling-20260925/`, cujo `report.json` aponta para diretórios temporários que não existem mais.

A convenção é a do run real canônico (`experiments/e2e-real-canonical-run-20260923/`):

- **Onde.** Sob o `outputs/` do checkout que contém o driver, a raiz padrão de `DriverRun.for_driver()`. O checkout é reconhecido pelo layout, nunca por um caminho absoluto de máquina: é o pai do ancestral mais próximo do driver chamado `experiments`, desde que esse pai tenha um `pyproject.toml`. Um driver fora disso não tem raiz padrão e precisa de uma explícita.
- **Diretório temporário só com raiz explícita.** `outputs_root=` aceita outra raiz, por exemplo um diretório temporário para um smoke run. Nada escolhe um diretório temporário implicitamente.
- **O que o report registra.** O `run_dir` de cada braço é `run.location(<braço>)`, relativo ao `outputs/`. Ele continua válido em outra máquina que tenha os artifacts no mesmo lugar e não expõe caminho pessoal.
- **Tempo de vida.** `outputs/` é ignorado pelo Git: os artifacts não são versionados nem entram no bundle leve do experimento (ver `experiments/` em [`docs/ARTIFACTS.md`](../../../../docs/ARTIFACTS.md)) e duram enquanto o `outputs/` daquele checkout durar. Um *worktree* do Git é outro checkout, e o `outputs/` dele some junto com ele. Um experimento cujos braços precisam ser retidos roda do checkout principal ou recebe explicitamente o `outputs/` dele.
- **Contrato.** Nenhum contrato público de artifact muda: cada braço é um artifact comum do seu estágio, finalizado pelo writer da capability.

### Medição por processo filho: `measure_child_process()`

`measure_child_process(command)` executa um comando e devolve um `ChildProcessMeasurement` com o código de saída, o wall time (relógio monotônico, incluindo a partida do interpretador do filho), o pico de RSS em KiB e a saída capturada. Um filho que falha é medido, não levantado. Um comando vazio, um comando que nem começa e uma plataforma sem unidade conhecida são erro (`ExperimentDriverError`).

O pico vem de `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, o método do perfil de recursos do #181. Esse valor é um **máximo corrente sobre todos os filhos que um processo já esperou**: lido de um pai de vida longa, ele atribuiria a cada estágio ou braço o pico de todos os anteriores. Por isso a função nunca o lê no processo de quem chama. Um processo de medição novo executa o comando como **único** filho e reporta a leitura, e esse processo recusa medir se já tiver esperado qualquer outro filho. Chamar a função várias vezes do mesmo pai, como faz um orquestrador de braços, é seguro.

- O pico é o do **maior processo isolado** entre o filho e os descendentes que ele esperou, não a soma de uma árvore de processos.
- `ru_maxrss` é KiB só no Linux (no macOS é byte). Em outra plataforma a função recusa em vez de inflar o pico 1024 vezes.
- `to_record()` devolve exatamente os campos que os dois runners anteriores registravam: `returncode`, `wall_time_seconds` com 3 casas e `peak_rss_mb` = KiB / 1024 com 1 casa. Os números novos continuam comparáveis com os do #181 e do #564.
- O comando herda o ambiente e o diretório de trabalho. Para rodar um braço sob o `src/` de outra revisão, a variável vai no próprio comando (`["env", "PYTHONPATH=<src da revisão>", python, "arm.py", ...]`): `env` se substitui pelo alvo, que continua sendo o filho medido.

## Limitações e lacunas

- **Sem ligação com o runtime.** A topologia é a resolvida que se fornece. Ao gerar arms a partir de configurações resolvidas (#527), a configuração registrada de um estágio deve ser a do plano do runtime (`PlannedStage.component_configs`: backend e parâmetros de cada ponto de variação) mais a política de requisição semântica quando ela existir na configuração (#544), sem mudar as regras de verificação.
- **Sem executor real.** O executor é injetado. A conexão com o reaproveitamento de artifacts imutáveis e com os backends reais virá com #527/#528.
- **Observações físicas por arm não estão no envelope.** O relatório comum não lista as observações que cada arm avaliou (elas só aparecem dentro de relatórios de estágio, por exemplo `SemanticSampleReport.source_observation_id`). A identidade física é garantida pela seleção única do manifesto e pela sequência pinada ou não afetada com o mesmo conteúdo.
- **A configuração executada por estágio não é reportada.** O executor não devolve a configuração com que cada estágio rodou (a impressão digital de estágio do runtime, `ReuseKey`); o plano é verificado campo a campo e a execução, pelo digest da topologia citado no relatório e pelo conteúdo dos artifacts.
- **Variantes dentro de um relatório.** `compare_evidence_variants()` não confere a configuração do backend entre variantes: `SemanticSampleReport` guarda só a `BackendProvenance` (com a impressão digital da configuração), não os campos. Ablações de prompt e de evidência que precisam dessa garantia são arms de um experimento.
- **Fases.** A identidade de fase de um experimento fatorado chega com o #527.
- O registro de métricas v1 não tem uma métrica de custo de modelo externo; ela entra com uma nova versão do registro quando for necessária.
