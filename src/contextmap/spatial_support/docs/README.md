# Spatial Support

A capability `spatial_support` deriva um **suporte espacial semântico persistente** a partir
de um `ContextMap` finalizado e de uma agregação voxel (geometry-only) do seu mapa bruto.

## Problema

A cor que o viewer mostra não vem de RGB por ponto: ela é derivada do suporte semântico por
entidade, indexado por `geometry_id`, que é um **índice posicional na nuvem bruta**. Quando a
apresentação usa centroides de voxel, cada centroide tem um índice de transporte local do
agregado, que não corresponde ao índice do ponto bruto — então a coloração semântica é
desligada e o mapa "perde a cor".

## Decisão arquitetural

> **Nenhuma informação semântica persistente importante deve depender exclusivamente do
> índice de um ponto de uma point cloud concreta.** Atributos semânticos devem sobreviver à
> simplificação da geometria (*semantic attributes must survive geometry simplification*).

O índice do ponto continua existindo como proveniência de uma observação, mas não é a chave
primária da memória semântica. A semântica é agregada sobre um **suporte espacial** com IDs
estáveis derivados do espaço e da grade.

## Responsabilidades

- o endereço espacial e o suporte que se acumula nele;
- a atribuição de cada ponto bruto de uma entidade ao voxel em que ele cai, com a identidade
  versionada dessa regra;
- o run persistido que carrega esse suporte, imutável e auditável.

## O que este módulo explicitamente não possui

- a entidade (é de `artifact`) nem a grade ou a agregação voxel (são de `geometric_mapping`);
- nenhuma política de hipótese semântica: um voxel acumula suporte e nunca vira entidade;
- nenhum estágio de runtime, nenhum backend, nenhum ponto de variação;
- nenhuma produção da agregação voxel que consome.

## Modelo

- `SpatialCell` — um **endereço espacial**, não uma entidade. Identidade
  `spatial-cell:<grid_fingerprint>:<kx>_<ky>_<kz>`, função pura da grade e da chave do voxel.
  Carrega `centroid_m`, `bounds`, `point_count` (resumo geométrico) e `entity_support[]`.
  O `aggregate_index` é a posição do voxel na agregação de origem — o **índice de transporte**
  sob o qual a geometria de apresentação serve o centroide. É por ele (nunca por índice de
  ponto bruto) que o viewer alinha uma linha de suporte ao ponto renderizado.
- `EntitySupport` — `(entity_id, support_weight)`, onde `support_weight` é a **contagem de
  pontos** da entidade que caem na célula. É uma medida de observações, nunca uma
  probabilidade, e não é misturada com scores (respeita a separação de semânticas do AGENTS.md).
- `SemanticSpatialSupport` — todas as células ocupadas de uma grade que alguma entidade
  suporta, ordenadas por chave de voxel.

A separação Evidence/Belief/Knowledge é preservada: o ponto bruto é evidência; o voxel é um
endereço em que várias evidências são agregadas; a entidade (crença) é referenciada, nunca
substituída.

## Junção

`build_semantic_spatial_support` reutiliza `VoxelAggregation.membership()` de
`geometric_mapping` — que mapeia cada índice de ponto bruto ao voxel que o resume — e atribui
os pontos de cada entidade às células correspondentes. A escolha do voxel é decidida pela
geometria (via `membership`), não por um novo re-binning.

## Artefato

`SemanticSpatialSupportArtifact` é um artefato **derivado** e imutável (padrão
`AtomicRunDirectory`, inventário com SHA-256), separado do `ContextMap` e da agregação, que
ele só lê. `derive_semantic_spatial_support_artifact` abre os três artefatos de origem,
verifica os hashes do próprio `ContextMap` e exige que tanto ele quanto a agregação nomeiem
exatamente o conteúdo do mapa bruto fornecido, não só o mesmo `map_id`, antes de persistir o
suporte. Nenhum estágio canônico o consome; o consumidor é o viewer (LODs por voxel). Schema
`0.2.0`; um run de schema anterior é recusado ao abrir, sem camada de migração. Ver
[ARTIFACTS.md](../../../../docs/ARTIFACTS.md) e [data-flow](data-flow.md).

## Estado

A capability **não é estágio do DAG**: `src/contextmap/runtime/{catalog,composition,pipeline,config}.py`
não a mencionam, e nada em `src/contextmap/` depende dela. O caminho de produção suportado é o
comando

```bash
contextmap derive-spatial-support \
  --context-map DIR --voxel-aggregation DIR --raw-geometry DIR \
  --output DIR --run-id ID [--code-version ID]
```

Produzir a agregação voxel que ele consome continua fora do runtime — é de
`geometric_mapping`, e `experiments/semantic-voxel-color-20260930/derive_voxel_support.py` é
quem a produz para o bundle real.

## Estrutura do código

```text
src/contextmap/spatial_support/
├── __init__.py       fachada pública (quinze nomes)
├── models.py         o vocabulário do suporte (folha)
├── policy.py         a identidade versionada da regra
├── service.py        a junção
├── derivation.py     seleção dos três upstreams e linhagem
└── run_artifact/     o run persistido: layout, manifest, writer, reader
```

Camadas: `models`, `policy` ← `service` ← `derivation`; e `models`, `policy` ←
`run_artifact/layout` ← `manifest` ← {`writer`, `reader`} ← `derivation`. `run_artifact/` é
interno; `derivation.py` é o único módulo que abre artifact de outra capability. As duas regras
são verificadas em `tests/architecture/test_spatial_support_layering.py`.

## Superfície pública

```text
SCHEMA_VERSION
EntityCellInput  EntitySupport  EntityVoxelSupportPolicy
SemanticSpatialSupport  SemanticSpatialSupportError
SpatialCell  SpatialCellId  spatial_cell_id_for
build_semantic_spatial_support  derive_semantic_spatial_support_artifact
SpatialSupportArtifactError  SpatialSupportArtifactReader
SpatialSupportArtifactWriter  SpatialSupportManifest  SpatialSupportRunId
```

`SemanticSpatialSupport` mantém o nome longo porque ali não é redundância: a capability é o
suporte espacial, e a coisa é o suporte espacial **semântico**. Os tipos de persistência
encurtam para `SpatialSupport*`.

## Módulos consumidos

`shared` (primitivas e `AtomicRunDirectory`), `artifact` (`ContextEntityId`,
`ContextMapArtifactReader`), `geometric_mapping` (grade, agregação, geometria bruta),
`ingestion` (`FrameId`, só como tipo).

## Módulos que consomem este

Nenhum dentro de `src/contextmap/`, por desenho. Fora: o comando de CLI e o viewer Nuxt, que
vive fora deste repositório e alinha a cor por `aggregate_index`.

## Como ler esta documentação

| Pergunta | Documento |
| --- | --- |
| por que a capability existe, o que entra e o que sai | este README |
| quem possui o quê, que dependência é permitida, quais invariantes arquiteturais | [architecture.md](architecture.md) |
| qual arquivo abrir para uma responsabilidade | [code-map.md](code-map.md) |
| o que acontece com os dados, etapa por etapa | [data-flow.md](data-flow.md) |
| qual teste protege qual invariante | [testing.md](testing.md) |

## Fronteiras

`spatial_support` é um consumidor derivado do produto público: depende apenas das APIs
públicas de `artifact` e `geometric_mapping`, da identidade de frame de `ingestion` (só como
tipo) e de `shared`. Nada depende de `spatial_support`, então não há ciclo com `artifact`.

## Limitações

- a ordenação das células por chave de voxel é garantia emprestada de `VoxelAggregation`;
- o `aggregate_index` existe para um consumidor fora do repositório, então o contrato de
  transporte não tem teste do lado de quem o consome;
- a política não tem parâmetro: ela versiona o que a junção computa, não sobre o que;
- a agregação voxel a montante não tem produtor no pipeline.

## Fora de escopo (por ora)

- Hipóteses semânticas e `embedding_summary` por célula (M3/M5): são políticas científicas e
  exigem avaliação registrada antes da implementação (AGENTS.md §22/§23).
- `observation_count` por `(célula, entidade)`: exige leitura por ponto da observação de
  origem; entra num próximo incremento.
- Multi-LOD e consumo pelo viewer: milestones seguintes.
