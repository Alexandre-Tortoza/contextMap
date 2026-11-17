# Canonical Pipeline do ContextMap2

Este documento é o mapa operacional do pipeline do ContextMap2. O objetivo não é apenas mostrar a ordem dos módulos, mas tornar explícito **onde cada transformação acontece**, **quais dados entram e saem**, **qual capability é dona da decisão** e **onde procurar no código quando um comportamento precisa ser alterado ou um gargalo precisa ser investigado**.

Para ownership e dependências entre capabilities, consulte [architecture.md](architecture.md). Para os contratos públicos, consulte [CONTRACTS.md](CONTRACTS.md). Para persistência, imutabilidade e lineage, consulte [ARTIFACTS.md](ARTIFACTS.md). Para composição dos executores e configuração, consulte [runtime-composition.md](runtime-composition.md).

A regra de leitura é importante: o diagrama abaixo mistura o **fluxo científico completo** com o **estado real da implementação**. Uma capability pode estar implementada e testável isoladamente sem possuir executor automático no runtime global. Essas duas coisas são distinguidas explicitamente ao longo do documento.

## Pipeline detalhada

O runtime seleciona artifacts imutáveis por identidade e hash e compõe seus executores.
A percepção tem três branches produtoras: regiões finais; features visuais; interpretação
semântica. Scoring junta visual e semântica apenas por seleção explícita. A associação exige
regiões e pode amostrar features densas, sem depender da interpretação semântica.

```mermaid
flowchart TD
    RAW["Fonte registrada"] --> ING["Ingestion<br/>SequenceArtifact"]
    ING --> RP["Region Perception<br/>regiões, máscaras e auditorias"]
    ING --> SE["State Estimation<br/>Trajectory"]
    SE --> GM["Geometric Mapping<br/>GeometricMapArtifact"]
    RP --> VP["Visual Perception<br/>VisualEvidenceArtifact"]
    RP --> SI["Semantic Interpretation<br/>SemanticInterpretationArtifact"]
    RP --> SA["Sensor Association<br/>SpatialObservation"]
    VP -. denso opcional .-> SA
    SE --> SA
    GM --> SA
    VP --> SS["Semantic Scoring<br/>SemanticScoringArtifact"]
    SI --> SS
    SA --> SF["Semantic Fusion<br/>FusedEvidence"]
    SI --> SF
    SS -. score opcional .-> SF
    GM --> PTR["Point Representation<br/>opcional"]
    PTR -. evidência opcional .-> SF
    SF --> SM["Semantic Mapping<br/>Entity"]
    SM --> ER["Entity Resolution<br/>ResolvedEntity"]
    ER --> SR["Spatial Relations<br/>Relation"]
    GM --> CM["ContextMapArtifact"]
    ER --> CM
    SR --> CM
```

Cada caixa de percepção publica seu próprio artifact. O `PerceptionResult` pode ser montado
por um leitor como view de uma seleção de artifacts, mas não é persistido. Regiões e claims
se encontram por `region_artifact_id`, `source_observation_id` e `region_id`; o run de
interpretação distingue inferências repetidas da mesma observação.

## Estado de execução atual

A capability de domínio existe para todos os blocos de Ingestion até Spatial Relations, e o pacote `contextmap.artifact` já implementa schema, montagem, escrita, leitura, validação e bundle do mapa final. O que ainda não está uniforme é a **orquestração automática**.

| Estágio | Capability implementada | Executor automático no runtime | Observação |
| --- | --- | --- | --- |
| Ingestion | sim | não por `compose_executors()` | `IngestionStageExecutor` existe, mas requer um `IngestionRequest` da execução e é injetado explicitamente. |
| Region Perception | sim | sim | Executa descoberta e, quando selecionados, grounding e refinamento; publica regiões finais, máscaras e auditoria por frame. |
| Visual Perception | sim | sim | Consome o artifact regional selecionado e publica somente features densas e por região, sem copiar regiões ou claims. |
| Semantic Interpretation | sim | sim | Consome o artifact regional selecionado e publica claims, contexto de cena, views, requests e falhas. |
| Semantic Scoring | sim | sim, quando selecionado | Consome os artifacts regional, visual e semântico explícitos; publica scores com linhagem separada. |
| State Estimation | sim | sim | ExternalPose e FAST-LIO são selecionáveis. |
| Geometric Mapping | sim | sim | Executor usa lookup, motion-correction policy e agregação configurada. |
| Sensor Association | sim | sim | Usa regiões do produtor regional e, com `sensor_association.dense_sampling` selecionado, amostra features densas do produtor visual. Não consome claims. |
| Point Representation | sim | sim, quando habilitado e com identidade do código | Estágio opcional e derivado (#663): centros pela política `observation-support-anchor-v1`, descritor determinístico sem runtime ou PTv3 com o `PTv3Runtime` de um provider. Nenhum estágio consome o run (#657). |
| Semantic Fusion | sim | sim | O executor global roda a acumulação selecionada, baseline ou quality-aware (#655), sem entrada de Point Representation; a configuração recusa o canal `point_representation` (#657). A quality-aware lê a `ObservationQuality` do run de associação e recusa, antes de fundir, um run medido sob outras definições de qualidade. |
| Semantic Mapping | sim | sim, com identidade do código | Compõe com `code_version`, `semantic_map_id` e `code_digest`, que não são valor de configuração; a CLI (`--code-identity`) e `Runtime` (`code_identity`) os derivam da identidade do código e do dataset. |
| Entity Resolution | sim | sim | Aparência e representação exigem vector sources fornecidos pelo runtime/provider quando ativadas. |
| Spatial Relations | sim | sim | Avaliadores geométrico e de contato são opcionais por configuração. |
| ContextMapArtifact | sim | sim | `ContextMapExecutor` monta o artifact por referência aos runs de Entity Resolution e Spatial Relations; `assemble_context_map_with_metrics()` e `ContextMapArtifactWriter` continuam existindo como capability standalone, sem lógica nova no executor. |
| Spatial Support (derivado, fora do DAG) | sim | **não**, e por desenho | Não é nó da topologia: nada a jusante consome o artifact e o consumidor é o viewer, fora do repositório. O caminho de produção é o comando `contextmap derive-spatial-support`. A agregação voxel que ele consome também não tem estágio. |

O preset global é:

- `canonical/1`: Ingestion → Region Perception → Visual Perception e Semantic Interpretation independentes, em paralelo com State Estimation → Geometric Mapping → Sensor Association → Point Representation opcional → Semantic Fusion → Semantic Mapping → Entity Resolution → Spatial Relations → `context_map`. `semantic_scoring` é opcional e consome visual + semântico com o regional selecionado. O cenário de avaliação 1.0.7 congela essa topologia sem alterar os snapshots históricos 1.0.5/1.0.6.

## Contexto incremental

A partir da v0.1.1, um mapa contextual pode crescer sem refazer a geometria: uma **fundação espacial** fixa (sequência, trajetória e mapa) recebe execuções de contexto (`ContextRun`, percepção e associação sobre uma seleção de observações), acumuladas numa `ContextBranch`; cada `ContextBuild` congela uma revisão da branch e materializa, da Semantic Fusion ao `ContextMapArtifact`, um snapshot imutável. Só percepção e associação rodam a cada incremento; `state_estimation` e `geometric_mapping` rodam uma vez, na fundação. Ciclo, identidades, CLI e limitações em [runtime-composition.md](runtime-composition.md#contexto-incremental-v011-decisão-de-arquitetura).

## 0. Runtime & Configuration

**Função.** Transformar configuração declarativa em uma execução reproduzível. Runtime não possui ciência de percepção, geometria, fusão ou relações; ele seleciona implementações, valida dependências, conecta artifacts e registra lifecycle.

**Recebe.** `EffectiveConfig`, preset, seleção de backends/policies, recursos, providers, secrets via ambiente e artifacts já existentes ou fornecidos.

**Faz.** `components/` define variation points, backends, stages e presets; `composition.py` constrói backends e policies concretos; `pipeline.py` resolve DAG, preflight, inputs e ordem; `reuse.py` decide reuso por identidade; `runs.py` e `lifecycle.py` registram execução, falhas, cancelamento e retomada; `executors.py` adapta um stage do runtime ao serviço público da capability.

**Entrega.** `StageRequest`, `ArtifactRef`, plano de execução, journal e artifacts produzidos ou reutilizados.

**Onde procurar.**

| Problema | Arquivo principal |
| --- | --- |
| Stage não aparece ou depende do artifact errado | `src/contextmap/runtime/components/presets.py` |
| Backend/policy não compõe | `src/contextmap/runtime/composition.py` |
| Preflight, dependências, ordem do DAG | `src/contextmap/runtime/pipeline.py` |
| Executor chama a capability de forma errada | `src/contextmap/runtime/executors.py` |
| Reuso/recompute inesperado | `src/contextmap/runtime/reuse.py` |
| Run, journal, resume, lifecycle | `src/contextmap/runtime/runs.py`, `lifecycle.py` |

Detalhes: [runtime docs](../src/contextmap/runtime/docs/README.md), [composition](../src/contextmap/runtime/docs/composition.md) e [executors](../src/contextmap/runtime/docs/executors.md).

## 1. Ingestion

**Função.** Converter formatos de borda em uma sequência canônica independente de ROS ou dataset. Ingestion é dona de observações de sensor, calibração canônica, sincronização, provenance da fonte e persistência do `SequenceArtifact`.

**Recebe de.** Fonte registrada e configuração da execução. Adapters ROS 1/ROS 2 ou específicos convertem os dados de origem.

**Entradas principais.** RGB, LiDAR, IMU, poses externas, timestamps, intrínsecos, extrínsecos estáticos, tópicos/canais e janela temporal.

**Fluxo interno.**

1. `SourceAdapter` lê a fonte e produz `SourceObservation` e `CalibrationSet`.
2. As observações são validadas sem corrigir silenciosamente payloads, frames ou timestamps inválidos.
3. `synchronize()` agrupa por uma modalidade de referência usando nearest-within-tolerance e registra decisões e eventos descartados. Clocks diferentes nunca são comparados implicitamente.
4. Provenance e identidades de conteúdo/configuração são calculadas.
5. `SequenceArtifactWriter` publica observações, calibração, diagnostics e provenance de forma atômica.
6. `resolve_selection()` fornece subsets determinísticos para os stages downstream.

**Entrega para.** Visual Perception, State Estimation, Geometric Mapping, Sensor Association e Semantic Fusion.

**Onde procurar.**

| Problema | Arquivo / documento |
| --- | --- |
| Parsing de ROS/dataset | `ingestion/adapters/`, [adapters.md](../src/contextmap/ingestion/docs/adapters.md) |
| Frames/timestamps inválidos | `ingestion/source_observation_validation.py` |
| RGB e LiDAR não alinham no tempo | `ingestion/observation_synchronization.py`, [synchronization.md](../src/contextmap/ingestion/docs/synchronization.md) |
| Intrínsecos/extrínsecos incorretos | `ingestion/calibration/`, [calibration.md](../src/contextmap/ingestion/docs/calibration.md) |
| Artifact ou replay | `ingestion/sequence_artifact/`, `sequence_observation_selection.py` |

## 2. Percepção regional, visual e semântica

`region_perception` prepara a imagem e executa descoberta, grounding e refinamento conforme
os componentes selecionados. Ela publica a geometria final e as máscaras, com auditoria de
candidatos, passes, requests e decisões. O `RegionId` é local ao artifact regional.

`visual_perception` consome essa geometria congelada e extrai features densas e por região.
`VisualEvidenceArtifact` contém features, espaço de embedding, payloads lazy, amostragem e
diagnostics. Uma ablação de extrator visual reutiliza o run regional.

`semantic_interpretation` consome a mesma região congelada, prepara views exatas, executa
interpretação de cena e região e publica claims, contexto, prompts, respostas e falhas em
`SemanticInterpretationArtifact`. Uma ablação visual reutiliza esse artifact; mudar o
interpretador produz outro run sem alterar o artifact visual. O scoring combina os três
produtores selecionados e publica apenas scores e linhagem. O runtime não troca backend por
fallback silencioso quando o selecionado falha.

Detalhes de contratos e linhagem: [decisão de decomposição](perception-dag-decomposition.md),
[ARTIFACTS.md](ARTIFACTS.md) e as APIs públicas de `region_perception`,
`visual_perception` e `semantic_interpretation`.

## 3. State Estimation

**Função.** Produzir a trajetória dinâmica `T_map_body(t)` usada para colocar medições no mapa e projetar geometria no timestamp das imagens.

**Recebe de.** Ingestion: `ExternalPoseMeasurement` ou LiDAR + IMU, além de calibração estática quando o backend exige.

**Fluxo interno.**

1. O backend declara `geometry_requirements()`.
2. `StaticFrameGraph` resolve relações estáticas necessárias.
3. `run_geometry_preflight()` bloqueia combinações inviáveis antes do estimador.
4. `ExternalPoseEstimator` valida/publica medições externas, ou `FastLioEstimator` constrói um job LiDAR-inercial para `FastLioRunner`.
5. O resultado é normalizado em `PoseEstimate[]` e `Trajectory`.
6. Gaps, frames, clock e provenance permanecem explícitos.
7. `StateEstimationRunWriter` persiste o run.

**Entrega para.** Geometric Mapping, Sensor Association e Evaluation.

**Onde procurar.**

| Problema | Arquivo |
| --- | --- |
| Pose importada inválida | `state_estimation/backends/external_pose.py` |
| FAST-LIO não inicia/diverge | `backends/fast_lio.py`, `fast_lio_process.py`, `fast_lio_wrapper.py` |
| Extrínseco ou frame ausente | `state_estimation/frame_graph.py`, `state_estimation/preflight/` |
| Pose no timestamp errada | `state_estimation/lookup.py` |
| Drift / ATE / RPE | capability de Evaluation, não o contrato de State Estimation |

Detalhes: [State Estimation](../src/contextmap/state_estimation/docs/README.md), [backends](../src/contextmap/state_estimation/docs/backends.md) e [lookup](../src/contextmap/state_estimation/docs/lookup.md).

## 4. Geometric Mapping

**Função.** Transformar LiDAR do frame local do sensor para um frame global persistente, mantendo referências estáveis e lineage até cada medição.

**Recebe de.** Ingestion fornece os scans e a calibração; State Estimation fornece a trajetória.

**Fluxo interno.**

1. `assemble_geometry_inputs()` resolve layout XYZ, payload hash, estado de motion correction, `T_map_body(t)` e `T_body_source`.
2. Scans incompatíveis são rejeitados com motivo explícito antes de transformar pontos.
3. `transform_scan()` aplica `P_map = T_map_body(t) · T_body_source · P_source`.
4. Apenas coordenadas NaN/Inf são descartadas nessa etapa. Não há filtro de distância, denoising ou semântica escondida.
5. `MapAccumulator` grava em fluxo. O baseline mantém cada medição; `ScanVoxelPolicy` pode agregar apenas dentro do mesmo scan.
6. `PackedGeometry` oferece acesso aleatório, lookup por `GeometryReference`, scan index e trace da transformação.
7. `GeometricMapArtifactWriter` publica o mapa.

**Limite importante.** O módulo registra o estado de motion correction, mas **não aplica deskew**; todos os pontos de um scan usam a pose resolvida para um instante.

**Entrega para.** Sensor Association, Point Representation, Semantic Fusion, Semantic Mapping, Entity Resolution, Spatial Relations e Artifact.

**Onde procurar.**

| Problema | Arquivo |
| --- | --- |
| Scan rejeitado | `geometric_mapping/inputs.py` |
| Point cloud deslocada/rotacionada | `transformation.py`, depois conferir trajetória e extrínsecos upstream |
| Mapa muito grande | `accumulation.py`, política de agregação |
| Consulta espacial lenta | `geometry_storage.py` / acesso espacial |
| Erro de lineage de ponto | transform trace e serialization do módulo |

Detalhes: [Geometric Mapping](../src/contextmap/geometric_mapping/docs/README.md), [inputs](../src/contextmap/geometric_mapping/docs/inputs.md), [transformation](../src/contextmap/geometric_mapping/docs/transformation.md) e [accumulation](../src/contextmap/geometric_mapping/docs/accumulation.md).

## 5. Sensor Association

**Função.** Ancorar evidência visual 2D em geometria 3D persistente. É aqui que uma região/máscara deixa de ser apenas geometria de imagem e passa a ter suporte espacial no mapa.

**Recebe de.**

- Geometric Mapping: `GeometryBlockSource` / `GeometricMapArtifact`.
- State Estimation: trajetória e política de lookup.
- Ingestion: calibração, modelo de câmera e imagem original.
- Visual Perception: `PreparedImage`, `Region2D`, claims e features.

**Fluxo interno.**

1. `FrameProjector` resolve `T_map_body(t_rgb)` e `T_body_camera`; a translação composta é o centro óptico no frame do mapa.
2. `select_candidate_geometry()` escolhe, por `GeometryBlockSource.iter_blocks()`, a geometria dentro da esfera `max_range_m` em torno desse centro, carregando coordenadas autoritativas **e** o índice global de cada linha. Com `max_range_m = None` o candidato é o mapa inteiro, o baseline. O que o recorte garante **depende da `DepthMetric` da câmera**, porque é essa a grandeza que a oclusão compara: sob `RAY_RANGE` (fisheye, MEI) a preservação é **exata**, pois todo excluído está mais longe que todo retido na grandeza comparada; sob `OPTICAL_AXIS` (pinhole) **não é**, porque a regra compara `z` e `z ≤ range`, então um excluído pode ter `z` menor que um retido e, na mesma janela de células, levar embora um suporte. O erro nesse caso só torna um retido **menos** ocluído — entre os retidos, nunca descarta geometria que o mapa inteiro associou, mas pode **acrescentar**: um retido liberado que cai na máscara de uma região vira `ASSOCIATED`, evidência que o mapa inteiro teria rejeitado como ocluída. O diagnóstico `range_limit_candidate_support_count`, por frame e somado no run, conta o suporte associado fundo o bastante para isso; zero é conclusivo. As contagens por frame registram a população avaliada. Ver [`sensor_association/docs/projection_chain.md`](../src/contextmap/sensor_association/docs/projection_chain.md), que detalha os dois casos, e o par adversarial em `tests/sensor_association/test_candidate_equivalence.py`.
3. O modelo de câmera projeta 3D para pixel na imagem crua.
4. `raw_to_prepared_transform()` reproduz crop/resize da percepção para levar o pixel ao mesmo espaço em que as máscaras vivem.
5. Pontos são classificados como behind-camera, outside-image, outside-valid-support ou in-support.
6. `resolve_visibility()` aplica a política conservadora de oclusão por suporte de profundidade local.
7. `associate_regions()` testa pertencimento às máscaras. Máscaras persistidas são carregadas sob demanda via `MaskStoreReader`.
8. `build_spatial_observations()` cria uma `SpatialObservation` por região, com `GeometryReference`, referências de claims/features e provenance.
9. Quando mapas densos são fornecidos, `sample_dense_features()` calcula índices/pesos nearest ou bilinear sem duplicar vetores por ponto.
10. `derive_observation_quality()` mede profundidade, ângulo, borda, visibilidade, densidade, offset temporal e reprojeção quando existe referência confiável.
11. Diagnostics de calibração, tempo e reprojeção permanecem separados da evidência semântica.
12. Cada frame concluído é entregue ao `FrameSink` da transação de escrita, persistido e **liberado**; o `SensorAssociationOutcome` carrega identidade, políticas, rejeições e `frame_count`, nunca os frames. O pico de memória é `estado estático do mapa + um frame + buffers do writer`, não `frames x estado por frame`.
13. O run é publicado atomicamente como `SensorAssociationRunArtifact`; sair da transação sem `finalize()` não deixa artifact algum.

**Estado do executor global.** Por padrão (perfil canônico) `SensorAssociationExecutor` não declara canal denso: o stage automático usa o caminho de associação geométrica/máscara. Com o componente opcional `sensor_association.dense_sampling` selecionado (#651), ele declara um canal por feature densa da run de percepção, identificado por `<estágio>@<embedding_space_id>`, e reconstrói o `DenseFeatureMap` de cada frame pela geometria que a run gravou em `metrics/feature-extraction.jsonl`. As amostras densas não têm consumidor a jusante: Semantic Fusion, Semantic Mapping e Entity Resolution não as leem.

**Onde procurar.**

| Sintoma | Código |
| --- | --- |
| Projeção 3D→2D deslocada | `frame_projection.py`, `camera_models.py`, `image_transform.py` |
| Problema só depois de crop/resize | `image_transform.py` |
| Pontos do fundo recebem label da frente | `visibility.py` e `OcclusionPolicy` |
| Máscara não recebe pontos | `membership.py`; confirmar `MaskStoreReader` e dimensões |
| Feature densa amostrada no patch errado | `dense_sampling.py` |
| Quality estranha | `quality.py`, `quality_derivation.py` |
| Reprojeção/tempo suspeitos | `diagnostics.py` |

Detalhes: [Sensor Association](../src/contextmap/sensor_association/docs/README.md), [projection chain](../src/contextmap/sensor_association/docs/projection_chain.md), [visibility](../src/contextmap/sensor_association/docs/visibility.md) e [membership](../src/contextmap/sensor_association/docs/membership.md).

## 6. Point Representation

**Função.** Produzir uma representação numérica opcional da estrutura 3D local. Ela complementa geometria, não a substitui e não contém label.

**Recebe de.** Geometric Mapping, a geometria; Sensor Association, as observações espaciais de que a política de centros escolhe as âncoras. O run registra a associação usada (`association_context_id`) e a política (`center_selection_policy_id`); nenhuma feature, máscara ou claim da associação entra no vetor.

**Fluxo interno.**

1. Selecionar centros `GeometryReference`: no runtime, `observation-support-anchor-v1`, uma âncora por observação espacial, o membro do suporte mais próximo do centróide (uma heurística; ver [seleção de centros](../src/contextmap/point_representation/docs/center-selection.md)).
2. `SupportExtractor` recupera ponto, vizinhança por raio ou k-nearest.
3. `CoordinatePreparation` centraliza e/ou normaliza escala conforme a política.
4. `PointEncoder` recebe `PreparedSupport`.
5. O backend pode ser o descritor geométrico determinístico ou PTv3/Pointcept.
6. `RepresentationService` mantém falhas de suporte explícitas.
7. O run persiste `PointRepresentation`, `RepresentationSpace` e payloads lazy.

**Estado do runtime.** O estágio é opcional e desligado em `canonical/1`. Habilitado, `compose_executors` compõe o `PointRepresentationExecutor` (#663), que publica o `PointRepresentationRunArtifact` do run; o PTv3 recebe o runtime de um provider. Nenhum estágio consome o run (#657): é saída derivada, que a `ContextRun` grava e o `ContextBuild` não fornece à materialização.

**Onde procurar.** `point_representation/support.py` para custo de neighborhood query; `backends/geometric_descriptor.py` para baseline; `backends/ptv3/` para o backend aprendido; `service.py` para execução.

Detalhes: [Point Representation](../src/contextmap/point_representation/docs/README.md) e [support extraction](../src/contextmap/point_representation/docs/support-extraction.md).

## 7. Semantic Fusion

**Função.** Acumular múltiplas observações sobre suporte espacial compatível sem criar identidade persistente de objeto.

**Recebe de.** Sensor Association fornece `SpatialObservation`; Visual Perception fornece a evidência referenciada; Geometric Mapping fornece as referências espaciais; Ingestion fornece timestamps; Point Representation é um canal opcional.

**Fluxo interno.**

1. `group_by_physical_observation()` separa frame físico de execução de inferência. Repetir um modelo três vezes no mesmo frame continua sendo uma observação física.
2. `build_fusion_supports()` conecta observações cuja geometria possui Jaccard acima do limiar e usa componentes conexas como `FusionSupport`.
3. Cada observação vira `EvidenceContribution`.
4. A política baseline agrupa claims por uma chave tipográfica de label, mantém stances `SUPPORTING`, `CONFLICTING`, `AMBIGUOUS` e `ABSTAINING` e registra incerteza.
5. Canais permanecem tipados: claims, scores, visual features, observation quality, geometry support e point representation não são concatenados num score único.
6. A política quality-aware opcional adiciona peso de contribuição derivado somente da qualidade mensurável da vista e não apaga evidência.
7. `FusedEvidence` é persistida em `SemanticFusionRunArtifact`.

**Estado do executor global.** O executor roda a política de acumulação selecionada, baseline ou quality-aware, sem substituir uma pela outra (#655). O estágio não tem entrada de Point Representation, e a configuração recusa o canal `point_representation` da acumulação: consumir a evidência 3D é um contrato novo, fora de escopo (#657). Com a quality-aware, ele lê a `ObservationQuality` de cada observação do run de associação que a produziu e recusa, antes de construir qualquer suporte, um run de associação cuja versão de definições de qualidade não é a da política. O perfil canônico continua na baseline.

**Onde procurar.**

| Problema | Código |
| --- | --- |
| Mesmo frame contado várias vezes | `semantic_fusion/grouping.py` |
| Observações do mesmo objeto não entram no mesmo suporte | `support_construction.py`, limiar Jaccard |
| Regiões diferentes são fundidas demais | `support_construction.py`; revisar geometria e política |
| Contradições/empates inesperados | `uncertainty.py` |
| Hipótese formada ou label agrupado de forma inesperada | `hypotheses.py` |
| Quality weight inesperado | `quality_aware.py` |
| Custo da construção de suporte | `geometry_overlap.py` |
| Canal opcional desaparece | conferir `BaselineAccumulationPolicy.channels` e inputs |

Detalhes: [Semantic Fusion](../src/contextmap/semantic_fusion/docs/README.md), [support](../src/contextmap/semantic_fusion/docs/support.md) e [accumulation](../src/contextmap/semantic_fusion/docs/accumulation.md).

## 8. Semantic Mapping

**Função.** Materializar `FusedEvidence` como entidades persistentes e consultáveis, mantendo geometria, semântica, evidência e tempo separados.

**Recebe de.** Semantic Fusion e Geometric Mapping.

**Fluxo interno.**

1. A seleção de `FusionOutcome` é explícita.
2. `materialize_entities()` aplica a baseline `one-support-one-entity-v1`.
3. `summarize_geometry()` mantém o conjunto exato de `GeometryReference` como autoridade e deriva bounds, centroid, extent, estatísticas, orientação opcional e diagnostics.
4. `semantic_state_from_fused_evidence()` preserva todas as hipóteses, alternativas, sinais e incertezas.
5. Evidence links mantêm identidade/digest dos artifacts upstream.
6. `summarize_temporal_state()` deriva first/last seen, frames físicos, inferências e lifecycle.
7. Candidatos inválidos viram `CandidateRejection`; não desaparecem.
8. O run persiste `Entity` em `SemanticMappingRunArtifact`.

**Fronteira central.** Dois `FusionSupport` diferentes continuam duas `Entity`, mesmo que tenham o mesmo label e estejam adjacentes. Decidir se são o mesmo objeto pertence exclusivamente a Entity Resolution.

**Estado do runtime.** `SemanticMappingExecutor` compõe automaticamente quando o chamador fornece `code_version`, `semantic_map_id` e `code_digest`, que não são valor de configuração; a CLI e `Runtime` os derivam da identidade do código do run (`--code-identity`) e do dataset. Sem eles, o artifact continua precisando ser fornecido para o stage seguinte, e o preflight diz o que falta.

**Onde procurar.** `semantic_mapping/materialization.py` orquestra; os contratos estão em `entity.py`, `geometry.py`, `semantic_state.py`, `evidence.py` e `temporal.py`; as regras versionadas em `geometry_summary.py`, `state_mapping.py`, `primary_hypothesis.py` e `interaction_radius.py`; o run persistido no package `run_artifact/`. O [mapa de código](../src/contextmap/semantic_mapping/docs/code-map.md) declara o dono de cada arquivo.

Detalhes: [Semantic Mapping](../src/contextmap/semantic_mapping/docs/README.md) e [materialization](../src/contextmap/semantic_mapping/docs/materialization.md).

## 9. Entity Resolution

**Função.** Decidir se entidades de origem representam o mesmo objeto físico, objetos distintos ou um caso não resolvido, sem mutar as entidades originais.

**Recebe de.** Semantic Mapping. Os canais opcionais podem ler geometria, features visuais e representações 3D pelos owners correspondentes.

**Fluxo interno.**

1. `EntitySpatialIndex` e `retrieve_candidate_sets()` reduzem o all-pairs com regras espaciais/temporais interpretáveis.
2. `candidate_pairs` enumera cada par plausível uma vez.
3. Gates impedem comparação entre mapas/frames incompatíveis.
4. Cada canal produz evidência separada:
   - geometria: bounds, suporte compartilhado, distância e orientação;
   - semântica: compatibilidade de hipóteses/atributos;
   - temporal: histórico de observação;
   - aparência: protótipos de features visuais dentro do mesmo `EmbeddingSpace`;
   - representação 3D: protótipos dentro do mesmo `RepresentationSpace`.
5. `EntityMatchEvidence` preserva todos os canais sem score agregado.
6. `ConservativeResolutionPolicy` decide `MATCH`, `DISTINCT` ou `UNRESOLVED`, lendo status dos canais.
7. `materialize_resolved_entities()` forma componentes conexas apenas de `MATCH`.
8. Contradições de transitividade impedem merge do componente e ficam explícitas.
9. `detect_split_candidates()` pode diagnosticar over-merge, mas nunca divide automaticamente.
10. O run persiste decisões, evidência e `ResolvedEntity`.

**Entrega para.** Spatial Relations e montagem do ContextMap.

**Onde procurar.**

| Sintoma | Código |
| --- | --- |
| Número de pares explode | `entity_resolution/retrieval.py` / `candidate-retrieval.md` |
| Objetos iguais nunca viram candidatos | política de retrieval antes dos comparadores |
| False merge por geometria | `geometry_comparison.py` e `resolution_policy.py` |
| Labels incompatíveis tratadas errado | `semantic_comparison.py` |
| Aparência parece ignorada | `appearance_comparison.py`, espaço e vector source |
| Representação 3D parece ignorada | `representation_comparison.py` |
| Match transitivo perigoso | `resolved_entity.py` |
| Entidade deveria ser dividida | `split_detection.py`, lembrando que é diagnóstico only |

Detalhes: [Entity Resolution](../src/contextmap/entity_resolution/docs/README.md), [candidate retrieval](../src/contextmap/entity_resolution/docs/candidate-retrieval.md) e [resolution policy](../src/contextmap/entity_resolution/docs/resolution-policy.md).

## 10. Spatial Relations

**Função.** Derivar relações espaciais entre entidades resolvidas usando medições explícitas e uma taxonomia versionada.

**Recebe de.** Entity Resolution fornece entidades resolvidas; Semantic Mapping fornece os resumos espaciais associados aos membros; Geometric Mapping resolve pontos quando predicados de contato precisam deles.

**Fluxo interno.**

1. O leitor de Entity Resolution produz geometrias resolvidas coerentes.
2. `FrameConventions` declara `up_axis` e `forward_axis`; nada é inferido silenciosamente.
3. `generate_relation_candidates()` reduz os pares via bounds, proximidade e precondições direcionais.
4. Predicados geométricos medem `NEXT_TO`, `ABOVE`, `IN_FRONT_OF`, `INSIDE` e `INTERSECTS`.
5. Predicados de contato medem `TOUCHING`, `ON_TOP_OF` e `LEANING_AGAINST`.
6. Afirmações relacionais upstream podem entrar como evidência de observação, mas nunca decidem sozinhas.
7. `decide_relations()` produz `SUPPORTED`, `REJECTED` ou `UNRESOLVED`.
8. Consistência estrutural impede relações direcionais contraditórias.
9. Inversos e simétricos são derivados depois da decisão, para não medir duas vezes o mesmo fato.
10. O run persiste `Relation`, `RelationEvidence` e decisões.

**Onde procurar.** `candidates.py` para recuperação; `geometric_predicates.py`; `contact_predicates.py`; `observation_evidence.py`; `decision.py`; `taxonomy.py`.

Detalhes: [Spatial Relations](../src/contextmap/spatial_relations/docs/README.md), [taxonomy](../src/contextmap/spatial_relations/docs/taxonomy.md) e [decision](../src/contextmap/spatial_relations/docs/decision.md).

## 11. Context Map Artifact

**Função.** Compor o produto público final sem repetir a ciência dos módulos upstream. O mapa final referencia a geometria autoritativa, incorpora as entidades resolvidas e relações e fecha provenance/lineage.

**Recebe de.** Geometric Mapping, Entity Resolution, Spatial Relations e metadata de criação.

**Fluxo interno.**

1. `assemble_context_map()` abre os runs reais e verifica que a lineage fecha.
2. Entidades resolvidas e relações são traduzidas para os contratos do `ContextMap` sem refazer decisões.
3. O estado semântico é traduzido preservando ambiguidade; hipóteses de label apontam para a evidência fundida que realmente as originou.
4. Relações apontam para a evidência e geometria que as sustentam.
5. `ContextMap.__post_init__` valida referências, capabilities declaradas e fechamento de provenance.
6. `ContextMapArtifactWriter` grava manifest e tabelas deterministicamente e publica atomicamente.
7. `ContextMapArtifactReader` permite leitura sem runtime de modelo.
8. `validate_context_map_artifact()` verifica integridade e dependências.
9. `export_bundle()` pode fechar dependências num bundle portátil.

**Estado do runtime.** `context_map` é um estágio real de `canonical/1`, com `ContextMapExecutor` composto automaticamente pelo `compose_executors()`; nenhuma orquestração manual fora do DAG é necessária.

**Onde procurar.** `artifact/serialization/assembly.py`, `writer.py`, `reader.py`, `validation.py`, `bundle.py`.

Detalhes: [Artifact](../src/contextmap/artifact/docs/README.md), [assembly](../src/contextmap/artifact/docs/assembly.md) e [storage layout](../src/contextmap/artifact/docs/storage-layout.md).

## 12. Evaluation

Evaluation é transversal e **não altera o pipeline**. Ela lê artifacts imutáveis, reference sets e anotações para medir qualidade, regressão, custo e reprodutibilidade.

Ela possui protocolos específicos para Region Discovery, Feature Extraction, Semantic Interpretation, State Estimation, Geometric Mapping, Sensor Association, Point Representation, Semantic Fusion, Semantic Mapping, Entity Resolution e Spatial Relations. Resultados de qualidade e performance permanecem separados; ausência de anotação ou métrica não aplicável nunca vira zero.

Use Evaluation quando a pergunta for “esta alteração melhorou?”, não para executar a transformação científica em si.

Detalhes: [Evaluation](../src/contextmap/evaluation/docs/README.md).

## Consumidores derivados do artefato

Depois que o `ContextMapArtifact` existe, uma derivação pode produzir outra representação **sem**
ser etapa do pipeline. Hoje há uma, `spatial_support`, e ela não é numerada aqui porque não
pertence à cadeia que produz o mapa.

**Spatial Support** junta as entidades de um `ContextMapArtifact` finalizado aos voxels ocupados
de um `VoxelAggregationArtifact` do mesmo mapa bruto, usando o `membership()` da agregação para
atribuir cada ponto de uma entidade ao voxel em que cai. Cada `SpatialCell` é um endereço
espacial cuja identidade é função pura da grade, **nunca** do índice de um ponto: é essa
propriedade que faz a cor semântica sobreviver à simplificação da geometria para centroides na
apresentação. Um voxel não vira entidade; ele acumula suporte.

Não há executor nem entrada no catálogo. Produção:

```bash
contextmap derive-spatial-support \
  --context-map DIR --voxel-aggregation DIR --raw-geometry DIR \
  --output DIR --run-id ID [--code-version ID]
```

Detalhes: [spatial_support](../src/contextmap/spatial_support/docs/README.md); o artifact em
[ARTIFACTS.md](ARTIFACTS.md) e os contratos em [CONTRACTS.md](CONTRACTS.md).

## Mapa rápido: sintoma → etapa → código

Esta tabela existe para que o documento funcione como índice de manutenção e investigação.

| Sintoma ou mudança desejada | Etapa mais provável | Comece por |
| --- | --- | --- |
| Máscaras demorando para aparecer | Visual Perception / Region Discovery | `discovery.py`, config de passes/tiling e backend SAM/Florence |
| Máscaras boas, mas artifact demora a finalizar | Visual Perception / persistência | `mask_store.py`, `run_artifact.py` |
| Máscara está deslocada após resize/tile | Visual Perception / coordinate remap | `image_preparation.py`, `discovery.py`, `normalization.py` |
| Embeddings densos muito lentos | Visual Perception / Feature Extraction | backend DINO + `feature_store.py` |
| Pose tem gaps ou offset temporal | State Estimation | `lookup.py`, backend e preflight |
| Mapa LiDAR fica duplicado/deslocado | Geometric Mapping | `inputs.py`, `transformation.py`; depois pose/calibração upstream |
| 3D projeta no pixel errado | Sensor Association | `frame_projection.py`, `camera_models.py`, `image_transform.py` |
| Objetos do fundo recebem evidência da frente | Sensor Association / visibility | `visibility.py` |
| Região tem máscara, mas zero suporte 3D | Sensor Association / membership | `membership.py`, máscara persistida, visibilidade e calibração |
| Feature densa chega na célula errada | Sensor Association / dense sampling | `dense_sampling.py` |
| Support extraction 3D domina o tempo | Point Representation | `support.py` e implementação de `GeometrySource` |
| Mesma imagem está contando como várias evidências | Semantic Fusion | `grouping.py` |
| Observações do mesmo lugar não fundem | Semantic Fusion | `support.py`, Jaccard e geometria de associação |
| Labels entram em conflito ou empate estranho | Semantic Fusion | `accumulation.py` |
| Muitas entidades duplicadas | Semantic Mapping / Entity Resolution | primeiro `materialization.py`, depois `retrieval.py` e comparadores |
| False merge de entidades | Entity Resolution | comparadores + `resolution_policy.py` |
| Entity Resolution está O(N²) | Entity Resolution / candidate retrieval | `retrieval.py`, `EntitySpatialIndex` |
| Relações demais ou faltando | Spatial Relations | `candidates.py`, depois evaluator do predicado |
| ABOVE/IN_FRONT_OF invertidos | Spatial Relations / frame conventions | `frame_conventions.py`, `taxonomy.py` |
| ContextMap não fecha lineage | Artifact | `serialization/assembly.py`, `structural_dependencies.py`, `validation.py` |
| O mapa perde a cor quando a apresentação usa centroides de voxel | Spatial Support (derivado) | `spatial_support/service.py`, e o `aggregate_index` de cada `SpatialCell` |
| Mudança funciona visualmente, mas não sabemos se melhorou | Evaluation | evaluator do estágio + reference set/experiment manifest |

## Regras de ownership que evitam procurar no lugar errado

- Ingestion normaliza fonte e calibração; não projeta em câmera.
- Visual Perception produz evidência de frame; não cria identidade persistente.
- State Estimation produz pose; não possui geometria persistente.
- Geometric Mapping possui a geometria e suas referências; não possui semântica.
- Sensor Association liga 2D a 3D; não funde observações nem decide labels.
- Point Representation descreve estrutura 3D; não substitui XYZ nem classifica.
- Semantic Fusion acumula evidência; não decide identidade de objeto.
- Spatial Support endereça suporte semântico no espaço; não é etapa do pipeline, não decide label e não possui a grade.
- Semantic Mapping materializa entidades; não resolve duplicatas entre suportes.
- Entity Resolution decide identidade; não cria relações espaciais.
- Spatial Relations deriva relações; não corrige entidades.
- Artifact compõe e persiste o produto final; não reexecuta ciência upstream.
- Evaluation mede; não muda outputs do pipeline.

Quando uma alteração exige quebrar uma dessas fronteiras, trate-a como mudança arquitetural e atualize [architecture.md](architecture.md), [module-api.md](module-api.md), [CONTRACTS.md](CONTRACTS.md) e os documentos internos afetados no mesmo PR.
