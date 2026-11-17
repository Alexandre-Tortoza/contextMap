# Executores de estágio

Um executor roda **uma** capability para **um** estágio do DAG e grava só no diretório que a runtime lhe entrega. Ele é fino: abre as entradas por `StageRequest.directory_of(ref)`, chama o serviço público da capability com as políticas científicas com que foi construído, entrega o `StageRequest.output_dir` ao writer da capability e devolve um `ArtifactRef` com o `location`. Não calcula caminho, não aloca identidade e não decide ciência: todo limiar e toda política vêm de quem constrói o executor.

O módulo é `contextmap.runtime.executors` e é importado **explicitamente**, nunca por `import contextmap.runtime`: ele depende da raiz pública de cada capability que executa, e o pacote da runtime não pode carregar nenhuma delas (um teste de arquitetura garante). Só raízes públicas, nunca um backend.

## Executores existentes

| Estágio | Executor | Entradas | Políticas recebidas na construção |
|---|---|---|---|
| `ingestion` | `IngestionStageExecutor` (em `ingestion_service.py`) | — | o `IngestionRequest`; o `output_dir` e o `artifact_id` vêm do estágio |
| `region_perception` | `RegionPerceptionExecutor` | `sequence` | `RegionDiscoveryPlan`, grounding/refinamento opcionais e preparo de imagem |
| `visual_perception` | `VisualPerceptionExecutor` | `sequence`, `regions` | factories de features densas e regionais; pooling regional opcional |
| `semantic_interpretation` | `SemanticInterpretationExecutor` | `sequence`, `regions` | intérprete e política resolvida de requests, views e contexto |
| `semantic_scoring` (opcional) | `SemanticScoringExecutor` | `regions`, `visual_evidence`, `semantic_interpretation` | factory do scorer, provenance e template de texto |
| `state_estimation` | `StateEstimationExecutor` | `sequence` | o estimador composto e a prontidão que reporta para a geometria |
| `geometric_mapping` | `GeometricMappingExecutor` | `sequence`, `trajectory` | política de pose, correção de movimento, agregação opcional |
| `sensor_association` | `SensorAssociationExecutor` | `sequence`, `regions`, `trajectory`, `geometry`; `visual_evidence` opcional | candidatos, oclusão, tolerâncias e política de pose; opcionalmente, a interpolação dos canais densos (`dense_sampling`, #651) |
| `point_representation` (opcional) | `PointRepresentationExecutor` | `geometry`, `association` | a política de centros (`observation-support-anchor-v1`) e o encoder composto (descritor determinístico ou PTv3) |
| `semantic_fusion` | `SemanticFusionExecutor` | `sequence`, `association`, `regions`, `semantic_interpretation`, `geometry`; `visual_evidence` e `semantic_scores` opcionais | política de suporte e a política de acumulação selecionada: baseline ou ciente de qualidade |
| `semantic_mapping` | `SemanticMappingExecutor` | `fusion`, `geometry` | política de materialização e o `SemanticMapId` |
| `entity_resolution` | `EntityResolutionExecutor` | `entities` | política de retrieval, o `MatchEvidenceBuilder` (canal de geometria obrigatório; semântico, temporal, aparência e representação só quando configurados) e a política de decisão conservadora |
| `spatial_relations` | `SpatialRelationsExecutor` | `entities`, `geometry` | convenções de eixo, política de candidatos e resumo de geometria, sempre presentes; avaliadores geométrico e de contato só quando configurados |
| `context_map` | `ContextMapExecutor` | `sequence`, `geometry`, `entities`, `relations` | o `up_axis` de `spatial_relations` (quando disponível) e a política de assembly |

### `RegionPerceptionExecutor`

Materializa a seleção física em imagens preparadas, constrói o backend regional por run e
pede `discover_audited()`. Passes, normalização, merges, rejeições e hints são regras da
capability regional; o executor encaminha a auditoria e qualifica os IDs por produtor e
observação. Grounding faz um request por query e imagem. Refinamento consome as propostas
aceitas, sob a política da capability, preservando caixas e máscaras como evidências distintas.
`assemble_regions` define a ordem descoberta → grounding → refinamento.

O `RegionPerceptionRunWriter` externaliza máscaras na fronteira do produtor, publica
`frames.jsonl`, `regions.jsonl`, auditoria e streams de grounding/refinamento. Todo frame
selecionado tem preparo de imagem e outcome explícito; falhas preservam evidências anteriores
sem inventar regiões. Provenance do backend construído deve coincidir com a do plano antes
do primeiro frame. A publicação é atômica e o scratch é removido mesmo em caso de falha.
Ver [descoberta](../../region_perception/docs/region-discovery.md),
[grounding](../../region_perception/docs/region-grounding.md),
[refinamento](../../region_perception/docs/region-refinement.md) e
[máscaras](../../region_perception/docs/mask_store.md).

### `VisualPerceptionExecutor` (#507)

Abre `regions` pelo `RegionPerceptionRunReader`, exige cobertura da seleção e rematerializa
cada imagem sem alterar a geometria ou constraints do preparo regional. As factories de
features recebem `FeatureBuildScope` com o payload sink do `VisualEvidenceRunWriter`.
Features densas preservam geometria de amostragem nos diagnósticos; pooling regional reutiliza
o payload denso do frame, sem uma segunda inferência. Um extrator condicionado à máscara
recebe um `mask_source` que resolve o payload regional explicitamente. Caixa sem máscara
fica inelegível para esse canal, sem máscara fabricada.

O artifact visual guarda features, diagnósticos, inelegibilidades e metadados de preparo.
Ele cita `region_artifact_id` e as regiões consumidas, sem copiar máscaras, auditoria ou
claims dos outros produtores. IDs de features pertencem ao run visual e à observação física.
O sink libera payloads temporários ao terminar o último consumidor de cada frame.

### `SemanticInterpretationExecutor`

Seleciona `sequence` e `regions` e constrói seu intérprete sobre o scratch de views.
A capability semântica materializa as views declaradas e cria requests com produtor
semântico, observação física e região explícitos. O adapter recebe os bytes hash-verificados,
prompt e política; o executor chama o port `interpret(request)` diretamente.

`SemanticInterpretationRunWriter` preserva claims/contexto, requests, resposta nativa,
views, parsing e falhas de chamada. Cada request regional tem seu próprio desfecho: uma falha
não apaga respostas de outras regiões do frame. A seleção, o produtor regional, as políticas,
provenance e código permanecem na linhagem do artifact semântico. A ciência de interpretação
é da [capability semântica](../../semantic_interpretation/docs/README.md).

### `SemanticScoringExecutor` (#652)

Abre os artifacts regional, visual e semântico selecionados. Exige mesma sequência,
seleção física e `region_artifact_id`; claims e features precisam referenciar as regiões
publicadas. A factory constrói o scorer sobre o feature store visual. Claims regionais são
pontuadas com suas features, sem modificar qualquer produtor. `SemanticScoringRunArtifact`
preserva as três identidades de origem e é uma entrada opcional explícita da fusão.
O cosseno, a compatibilidade do espaço e a semântica de scores são regras do scorer.

### `SensorAssociationExecutor` e os canais densos (#651)

Regiões, preparo de imagem e máscaras vêm do artifact regional. Cada `AssociationFrameInput`
nomeia `region_artifact_id`, observação física, imagem preparada e regiões, além das entradas
geométricas/temporais. O serviço recebe `association_run_id` próprio; `SpatialObservation`
preserva região, observação física, geometria, pose, calibração e provenance, sem claims ou
refs de features acopladas.

Sem `sensor_association.dense_sampling`, o executor usa apenas o canal geométrico.
Com a política selecionada, abre também `visual_evidence`, exige cobertura física e produtor
regional compatíveis, declara canais por estágio e espaço de embedding e reconstitui
`DenseFeatureMap` pela geometria persistida. O payload permanece lazy: amostras guardam índices
e pesos, e `gather()` carrega os vetores. Ausência de feature densa ou geometria necessária
é erro explícito, sem canal vazio.

O run grava associações densas e suas células separadamente. Suas lineage lists são
`region_artifact_ids` e `visual_artifact_ids` opcionais. As amostras densas não têm consumidor
a jusante no canônico atual; ligar o canal permite medir ancoragem/amostragem sem mudar a fusão.

### `PointRepresentationExecutor` (#663)

Publica o `PointRepresentationRunArtifact` do run, como saída derivada. O executor só compõe as peças públicas de Point Representation: a política de centros escolhe as âncoras a partir das observações espaciais do run de `association` (uma por observação, [seleção de centros](../../point_representation/docs/center-selection.md)); o `RepresentationService` extrai o suporte de cada âncora no mapa de `geometry`, sob a política de suporte do encoder, e chama o encoder; o `PointRepresentationRunWriter` grava o run no `output_dir`, com `association_context_id` (o run de associação), `center_selection_policy_id` e a referência ao mapa, que nunca é copiado. Quais centros, qual suporte e qual vetor são decisões da capability; um suporte que o encoder não representa é um `FailedSupport` explícito, e uma falha do backend (por exemplo `PTv3RuntimeUnavailableError`) derruba o estágio, sem fallback. Antes de escolher qualquer centro ou criar qualquer saída, o executor compara o `geometric_map_id` do manifest de `association` com o `map_id` do mapa de `geometry` e recusa a divergência com `ExecutorError`, que nomeia os dois mapas e o run de associação, sem publicar nada: a checagem por observação da política de centros não vê uma associação sem observações.

O executor é um `ConfiguredStageExecutor`: sua `configuration_identity` cobre a `EncoderIdentity` (versão do adaptador, fingerprint da configuração resolvida, com o dispositivo, e o hash do checkpoint), o fingerprint do `RepresentationSpace` (que inclui a política de suporte) e a política de centros, e o `run_id` combina `StageRequest.identity()` com ela. O run grava `code_version`, então o executor só é composto com uma identidade de código.

Nenhum estágio consome o run (#657): a fusão não tem entrada `representation`, e ligar o estágio não muda a identidade de nenhum outro estágio, que é reusado como antes. Uma `ContextRun` com o estágio ligado grava o run como membro; o `ContextBuild` só fornece à materialização os estágios de contexto que ela consome, então não fornece nem recomputa a representação.

### `SemanticFusionExecutor`: a política de acumulação (#655)

O executor roda exatamente a política de acumulação com que foi construído, `BaselineAccumulationPolicy` ou `QualityAwareAccumulationPolicy`, e despacha pelo tipo da política, pela API pública da Semantic Fusion (`accumulate_baseline_evidence` ou `accumulate_quality_aware_evidence`), nunca por um nome de backend; uma política de outro tipo é recusada com `ExecutorError`, sem cair na baseline. Suportes, agrupamento por observação física e escrita do run são os mesmos nos dois casos, então os dois braços do experimento B da #197 leem as mesmas entradas e diferem só na acumulação.

Com a política ciente de qualidade, o executor:

1. confere, logo depois de abrir os runs de associação e antes de ler a evidência regional/semântica ou construir qualquer suporte, que cada run mediu a qualidade sob as definições que a política declara (`manifest.definitions["quality"]` contra `definitions_version`), pela regra da própria política (`QualityAwareAccumulationPolicy.require_definitions`); um run com outras definições é recusado com `ExecutorError` que nomeia o run e as duas versões, nunca uma falha no meio da acumulação;
2. lê a `ObservationQuality` de cada observação que entrou em algum suporte com `SensorAssociationRunReader.quality()`, do run de associação que produziu a observação;
3. acumula cada suporte com `accumulate_quality_aware_evidence`. Um componente que a associação não mediu (por exemplo a reprojeção sem correspondências de referência) recebe o fator neutro da política, com o motivo que a associação registrou.

A baseline não lê qualidade nenhuma, e o caminho dela não mudou: o run de fusão baseline da cadeia sintética tem a mesma identidade e o mesmo hash de conteúdo de antes da #655. O run de fusão não ganha campo novo: `policies.fusion_policy_id` e `policies.fusion_configuration_fingerprint` do manifest já nomeiam a política que rodou e o fingerprint dela (rampas, fator neutro, `definitions_version` e a configuração baseline embutida), `evidence_identities.observation_quality` lista as definições da qualidade lida e cada `FusedEvidence` guarda seu `weighting`. Os parâmetros da política são parâmetros do componente `semantic_fusion.accumulation`, então entram no `config_digest` do estágio e, por ele, no `run_id`.

## Seleção de observações

Os três produtores aplicam `StageRequest.observation_selection`: decodificam e resolvem a
seleção canônica da Ingestion antes de carregar modelos ou publicar saídas. Cada manifest
preserva `selection_id` e observações físicas; consumidores recusam seleção sem cobertura
regional. `iter_frames()` e `frame(source_observation_id)` evitam hidratar constraints de
todas as imagens ao mesmo tempo. Associação herda a seleção regional. Estado e geometria
continuam na seleção da fundação. Ver [configuração](configuration.md#inputsobservation_selection).

## Identidade

O `run_id` que o writer grava vem de `StageRequest.identity()`: o estágio, o `config_digest` e o hash de conteúdo exato de cada entrada. Uma execução idêntica publica a **mesma** identidade e, portanto, o mesmo conteúdo; qualquer mudança de configuração ou de entrada muda a identidade.

O `config_digest` cobre a configuração declarada de cada stage. A `configuration_identity`
completa seus defaults e algoritmos efetivos: regional inclui provenance/discovery identity,
queries e refinamento selecionados; visual inclui backends/pooling; semântico inclui intérprete
e política de request/views. O runner incorpora essa identidade na chave de reuso, e cada
produtor a combina em seu próprio `run_id`. Mudar só a política semântica reutiliza regiões,
features e associação; recomputa o produtor semântico e consumidores de seu novo conteúdo.

O `run_index` vem de `StageRequest.run_number()`. O `content_hash` do `ArtifactRef` é o
digest do inventário contratual (`inventory_digest`), não uma identidade fundida da percepção.

## Cardinalidade

Um stage publica exatamente um artifact. Entradas comuns recebem um run; entradas
`multiple` de `SemanticFusionExecutor` recebem associações, regiões, interpretações semânticas
e evidências visuais como execuções distintas. Cada associação deve citar produtores regionais
selecionados; cada consumidor visual/semântico deve citar esses mesmos produtores. Runs
repetidos por identidade são deduplicados, e o agrupamento científico usa a observação física,
sem tratar inferências repetidas como aquisições independentes. Uma entrada de cardinalidade
única com vários runs é recusada, sem escolher um silenciosamente.

## Topologia

Para que os executores tenham o que precisam, o DAG `canonical/1` dá `sequence` à fusão (os instantes de aquisição das observações físicas vêm da sequência) e `geometry` ao mapeamento semântico e às relações espaciais (o suporte 3D das entidades vem do mapa).

## Como um executor chega a existir

A [composition root](composition.md) liga factories e políticas pela `EffectiveConfig`.
`compose_executors` monta os três produtores independentemente: ausência de runtime regional
ou semântico bloqueia seu stage, sem transformar o executor visual em um monólito parcial.
Features dependem apenas de seus backends configurados. Cada falha de composição preserva
seu componente e motivo no preflight; não há fallback entre backends.

Os demais executores são montados quando suas capabilities/políticas e identidades obrigatórias
estão disponíveis. `executor_identities` deriva código e identidade do mapa semântico para os
stages que os gravam. `ingestion` e `pose_ingestion` continuam precisando de
`IngestionRequest` concreto, vindo do comando de ingestão ou de injeção explícita.

## O que ainda não existe

- **Um consumidor do run de `point_representation`.** O estágio roda (#663), mas nem a fusão nem a Entity Resolution recebem o run: consumir a evidência 3D é um contrato novo, fora de escopo (#657). `pose_ingestion` (issue #555) é o outro estágio opcional do canônico sem composição automática: como `ingestion`, ele sempre precisa de um `IngestionStageExecutor` injetado explicitamente (ver [`composition.md`](composition.md#bridge-de-pose-auxiliar-em-state_estimation-issue-555)).
- **`entity_resolution.appearance`** e **`entity_resolution.representation`**, quando selecionados, precisam de um `FeatureVectorSource`/`RepresentationVectorSource` que só existe vinculado a runs de percepção/representação já abertos: `compose_executors` os obtém de um `provider` fornecido pelo chamador (o mesmo mecanismo de `RuntimeProvider` já usado por `sam3`/`qwen`/`gemini`, explícito ou declarado em `resources.providers`), nunca os inventa.

## `ContextMapExecutor`

Monta o `ContextMapArtifact` -- o produto público do repositório -- por referência aos runs de Entity Resolution e Spatial Relations, que já decidiram toda entidade, relação e evidência. É fino de propósito: traduz os manifests reais de `sequence`, `geometry`, `entities` e `relations` no que `contextmap.artifact.assemble_context_map_with_metrics()` precisa (essa função já é a dona da regra de tradução, identificada por `CONTEXT_MAP_ASSEMBLY_POLICY_ID`) e chama essa função e `write_context_map_with_metrics()` sem alterá-las -- nenhuma inferência semântica nova aqui. `source_sequences`/`time_bounds` vêm direto do manifest do mapa geométrico; `bounds` vem do `Bounds3D` que o próprio mapa já computou; os `relation_predicates` vêm de ler de verdade as relações do run; a identidade de linhagem da `SEQUENCE` usa `artifact_digest()`, o mesmo helper que `assembly.py` já usa para as outras entradas; e `upstream_locations` do writer vem da própria linhagem que o mapa montado já cita, filtrada ao que `ArtifactKind.is_structural` diz. O frame do mapa usa `contextmap.artifact.estimator_local_map_frame()` (métrico, destro, ancorado no estimador -- a única declaração de frame que Solution 1 usa hoje, nomeada como política versionada em vez de assumida em silêncio); seu `up_direction` é o `up_axis` que `spatial_relations` já compôs, nunca derivado de novo.

`ArtifactKind.SEQUENCE` não é uma dependência estrutural (`ArtifactKind.is_structural`), então `ContextMapArtifactWriter` nunca abre nem verifica o input `sequence` por conta própria -- ele só entra na lineage citada, sem localização, e `assemble_context_map()` não pode abri-lo diretamente porque `contextmap.artifact` não depende de `contextmap.ingestion` (fronteira de arquitetura testada). Por isso o próprio `ContextMapExecutor.execute()` abre `sequence` com `SequenceArtifactReader` e confere, antes de montar a lineage, que `sequence.manifest.artifact_id` e a seleção correspondente batem com `geometry_manifest.sequence_artifact_id`/`selection_id` -- sem essa checagem, um `sequence` incorreto passado pelo chamador publicaria uma lineage cujo `artifact_id` (do manifest da geometria) e `content_identity` (o digest de um diretório diferente) nomeariam dois artifacts distintos, sem nenhum outro lugar do sistema notando.

Depois de gravar, o executor roda `validate_context_map_artifact(level=ValidationLevel.FULL)` sobre o artifact, com as localizações dos upstreams e da sequência (#499). Um mapa que não sai `VERIFIED` derruba o estágio com os achados da validação: o run falha, nada é indexado para reuso, e um `ContextBuild` nunca termina com sucesso sobre um mapa inválido. A regra de validação continua sendo do artifact; o executor só se recusa a publicar como sucesso o que ela não verifica.
