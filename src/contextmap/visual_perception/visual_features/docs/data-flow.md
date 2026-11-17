# Data Flow: visual_features

O caminho real dos dados pela implementação. Nenhuma etapa aqui é planejada: todas existem no
código citado.

## Caminho principal, escopo denso

```text
PreparedImage (visual_perception)
        │
        ▼
FeatureExtractor.extract(image)                    ports.py
   required_scope() == DENSE                        adapters/{dinov2,dinov3,siglip2}
   [único ponto com SDK: transformers, torch]
        │
        ▼
DenseFeatureMap                             region_feature_pooling.py
   ├── feature: VisualFeature (escopo DENSE)
   ├── sampling: DenseFeatureSampling    <- descreve o mapeamento pixel↔célula
   ├── source_artifact_id
   └── enhancement: provenance do realce | None
        │
        ▼
FeatureStoreWriter.write(feature, observation_id, array)   feature_payload_store.py
   serializa .npy, calcula hash, registra FeaturePayloadEntry
        │
        ▼
feature-index.jsonl  +  um arquivo .npy por feature
```

## Caminho principal, escopo de região

Duas rotas chegam a uma feature de região, e elas não são equivalentes:

```text
ROTA A — pooling sobre a grade densa
  DenseFeatureMap + Region2D congelada
        ▼
  pool_region_feature(region, dense_map, array)      region_feature_pooling.py
    1. map_box_to_grid_cells: geometria da região -> células da grade
    2. _local_mask_bounds: janela local da máscara
    3. média ponderada pela máscara, preservando norma L2
       política: mask_weighted_mean_preserve_l2_v2
    4. nenhuma célula na região -> EmptyRegionSupportError (nunca vetor de zeros)
        ▼
  RegionPoolingResult
    ├── feature: VisualFeature (escopo REGION)
    ├── diagnostics: RegionPoolingDiagnostics
    └── provenance: RegionPoolingProvenance (com a política nomeada)

ROTA B — extrator que já produz por região
  PreparedImage + Region2D congeladas
        ▼
  FeatureExtractor.extract(image, regions)           adapters/{clip,alphaclip}
    required_scope() == REGION
        ▼
  VisualFeature (escopo REGION), direto do modelo
```

Na rota A a região nunca é alterada: o pooling mapeia sua geometria e agrega. Na rota B o modelo
recebe a região e produz o vetor por conta própria. Ambas registram proveniência, e o
`embedding_space_id` distingue de qual rota e de qual modelo a feature veio.

## Caminho paralelo: identidade do espaço de embedding

```text
EmbeddingSpace(family, model, version, dimension, checkpoint, layer, normalization)
        │
        ▼
embedding_space_fingerprint()  -> cobre os SETE campos
        │
        ▼
embedding_space_id  gravado em cada VisualFeature
        │
        ▼
ensure_compatible_embedding_spaces(a, b)  antes de qualquer comparação
   fingerprints diferentes -> EmbeddingSpaceMismatchError
```

É o que impede comparar um vetor DINOv2 de 768 dimensões com um CLIP de 768 dimensões. A
dimensão coincide; o espaço, não.

## Caminho paralelo: proveniência

```text
extractor.backend_provenance()  ──┐
                                  ├──> VisualFeature.provenance
pooling_backend_provenance()  ────┘     (POOLING_BACKEND_ID, POOLING_BACKEND_VERSION,
                                         POOLING_POLICY)
```

Uma feature agregada carrega **duas** proveniências encadeadas: a do modelo que produziu a grade
densa e a do pooling que a agregou. Perder a segunda tornaria duas políticas de agregação
indistinguíveis no artifact.

## Caminho paralelo: realce de resolução (opcional, sem backend hoje)

```text
DenseFeatureMap nativo
        ▼
enhance_feature_resolution(source, enhancer=...)      feature_enhancement.py
   enhancer.enhance(source)  -> DenseFeatureMap candidato
        ▼
   dez validações antes de aceitar:
     lineage nomeia exatamente o mapa de origem
     proveniência bate com o backend selecionado
     grade / imagem de origem / embedding space coerentes com a saída
     aumento estrito de ao menos um eixo, sem redução do outro
     feature_id, payload_reference distintos da origem
     normalização diferente exige embedding space distinto
        ▼
DenseFeatureMap realçado, com identidade PRÓPRIA e lineage completa
```

O estágio é opcional e **não há `FeatureResolutionEnhancement` concreto no repositório**. A
fronteira existe para que qualquer implementação futura não possa produzir uma feature realçada
indistinguível da nativa.

## Para onde as features vão

```text
VisualFeature (escopo REGION ou GLOBAL)
        ├──> semantic_scoring: comparação com a hipótese do claim
        │      (o scorer seleciona separadamente features e claims)
        │
        └──> VisualEvidenceArtifact, pelo VisualPerceptionExecutor
```

## O que não existe neste fluxo

Para evitar leitura otimista do diagrama:

- **Não há artifact próprio produzido.** Este pacote é componente do stage `visual_perception`,
  não um stage. O que persiste é o `FeatureStoreWriter` (payloads) e o `VisualEvidenceRunWriter` (o
  artifact do stage). O writer de run que existia aqui foi removido por não ter consumidor.
- **Não há realce em execução.** O port não tem implementação.
- **Não há semântica.** Nenhuma etapa aqui atribui rótulo a uma feature.
- **O array não trafega no contrato.** Entre capabilities circula `payload_reference`; o array só
  é carregado por quem precisa dele, pelo `FeatureStoreReader`.
