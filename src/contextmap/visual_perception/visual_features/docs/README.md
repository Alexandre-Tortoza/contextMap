# Visual Features

## Propósito

Extrair **embeddings visuais** de uma imagem preparada: mapas densos sobre a grade da imagem e
vetores por região.

Uma `VisualFeature` é evidência de **estrutura visual**, não de identidade semântica. Ela não diz
o que a coisa é — diz como ela se parece, num espaço de embedding nomeado. O significado só
aparece depois, quando `semantic_scoring` compara a feature com uma hipótese de
`semantic_interpretation`.

## Responsabilidades

| Responsabilidade | Arquivo |
|---|---|
| Contratos de feature e de espaço de embedding | `visual_evidence.py` |
| Associação determinística entre região congelada e grade densa | `region_feature_pooling.py` |
| Persistência e leitura aleatória dos payloads numéricos | `feature_payload_store.py` |
| Fronteira validada do realce de resolução | `feature_enhancement.py` |
| Pontos de substituição dos backends | `ports.py` |
| Adapters concretos dos modelos | `adapters/` |

## Não responsabilidades

- **Descoberta de região** — é de `region_perception`. Aqui a região chega congelada, e o pooling
  nunca a altera.
- **Interpretação semântica** — é de `semantic_interpretation`. Uma feature não carrega rótulo.
- **Scoring** entre feature e claim — é de `semantic_scoring`.
- **Fusão** — é de `semantic_fusion`.
- **Escolher o backend** — é do composition root, em `runtime`.

## Entradas e saídas

**Entra:** uma `PreparedImage`; no escopo de região, também as `Region2D` congeladas.

**Sai:** `VisualFeature` com proveniência completa — escopo, identidade do espaço de embedding,
shape, dtype, normalização e uma **referência** ao payload. O array numérico nunca viaja inline no
contrato (#48); ele é persistido pelo feature store e carregado sob demanda.

## Obrigatório e opcional

Obrigatório: um extrator densa/global ou de região, conforme o escopo que o estágio pede. Cada
adapter **declara** seu escopo em `required_scope()`, e é isso que decide se ele recebe apenas a
imagem ou também as regiões.

Opcional: o realce de resolução (`FeatureResolutionEnhancement`). **Não há implementação
concreta dele no repositório hoje** — existe o port e existe a fronteira validada que qualquer
implementação terá de respeitar.

## Escopos e o que cada adapter produz

| Adapter | Escopo declarado | Acesso |
|---|---|---|
| `dinov2` | `DENSE` | Transformers local |
| `dinov3` | `DENSE` | Transformers local |
| `siglip2` | `DENSE`, `GLOBAL` | Transformers local |
| `clip` | `GLOBAL`, `REGION` | Transformers local |
| `alphaclip` | `REGION` | Transformers local |

`FeatureScope` tem exatamente três valores: `DENSE`, `GLOBAL`, `REGION`.

As torres de texto de CLIP e AlphaCLIP vivem em `contextmap.semantic_scoring.adapters`, não
aqui: elas codificam hipótese, não imagem. O embedding space que cada uma declara continua
sendo definido por esta capability.

## Fluxo de alto nível

```text
PreparedImage
    -> FeatureExtractor.extract()         [adapter: SDK do modelo]
    -> DenseFeatureMap  (grade densa + DenseFeatureSampling descrevendo o mapeamento)
    -> pool_region_feature(region, dense_map)
         mapeia a geometria da região para células da grade
         pondera pela máscara, preserva norma L2
    -> VisualFeature (escopo REGION) + RegionPoolingDiagnostics
    -> FeatureStoreWriter persiste o array; a feature guarda só a referência
```

A política de pooling é nomeada e versionada: `mask_weighted_mean_preserve_l2_v2`. Ela entra na
proveniência, de modo que duas execuções com políticas diferentes sejam distinguíveis.

## Estrutura do código

```text
visual_features/
├── visual_evidence.py         VisualFeature, EmbeddingSpace, FeatureScope, PoolingMethod
├── region_feature_pooling.py  DenseFeatureMap, pooling região↔grade, diagnósticos
├── feature_payload_store.py   writer, índice JSONL e reader de acesso aleatório
├── feature_enhancement.py     fronteira validada do realce (10 invariantes)
├── ports.py                   FeatureExtractor, FeatureResolutionEnhancement
└── adapters/                  dinov2, dinov3, siglip2, clip, alphaclip (+ torres de texto)
```

Não há `artifacts/`: este pacote é componente do stage `visual_perception`, e é o artifact
daquele stage que publica as features.

## API pública

```python
from contextmap.visual_perception.visual_features import (
    VisualFeature,
    FeatureId,
    FeatureScope,
    PoolingMethod,
    EmbeddingSpace,
    EmbeddingSpaceMismatchError,
    encode_embedding_space,
    decode_embedding_space,
    embedding_space_fingerprint,
    ensure_compatible_embedding_spaces,
    ensure_compatible_features,
    FeatureExtractor,
    FeatureResolutionEnhancement,
    FeatureResolutionEnhancementError,
    enhance_feature_resolution,
    FeatureStoreWriter,
    FeatureStoreReader,
    FeaturePayloadEntry,
    DenseFeatureMap,
    DenseFeatureSampling,
    pool_region_feature,
    POOLING_POLICY,
)
```

Os contratos de `region_feature_pooling` resolvem sob demanda (`__getattr__`), para que importar
a capability não torne o NumPy dependência de runtime.

### Os contratos que importam

**`VisualFeature`** — `feature_id`, `scope`, `embedding_space_id`, `shape`, `dtype`,
`payload_reference`, `provenance`, `region_id`, `normalization`. Note o que **não** existe: não há
campo com o vetor. A feature é metadado leve mais uma referência.

**`EmbeddingSpace`** — `family`, `model`, `version`, `dimension`, `checkpoint`, `layer`,
`normalization`. Os sete campos entram no fingerprint, e é por isso que mesma dimensão **não**
significa mesmo espaço: dois modelos de 768 dimensões produzem fingerprints distintos, e
`ensure_compatible_embedding_spaces` recusa a comparação.

## Direção decidida

[`docs/perception-dag-decomposition.md`](../../../../../docs/perception-dag-decomposition.md) decide que esta capability **permanece
submódulo** de `visual_perception`, sem stage e sem artifact próprio. Os embeddings continuam
evidence com proveniência completa: `FeaturePayloadEntry` já persiste escopo, espaço, shape,
dtype, normalização, referência de payload, SHA-256, tamanho e proveniência, carregados
preguiçosamente e nunca inline no contrato.

Com uma consequência que vale registrar: quando as claims saírem para o stage de
`semantic_interpretation`, `semantic_scoring` passará a precisar selecionar os embeddings como um
upstream próprio — ele compara a hipótese de um claim **contra uma feature**, e precisa dos dois
lados. Isso **não** faz deste pacote um stage: o artifact de `visual_perception`, renomeado para
`VisualEvidenceArtifact`, é a saída selecionável que os contém. Promover `visual_features` à raiz
produziria o mesmo DAG com um nome a mais.

## Documentos relacionados

- [`architecture.md`](architecture.md) — ownership, boundaries e dependências verificáveis
- [`code-map.md`](code-map.md) — responsabilidade por arquivo
- [`data-flow.md`](data-flow.md) — o caminho real dos dados
- [`testing.md`](testing.md) — responsabilidade → teste, e os invariantes
- `../../visual_perception/docs/feature_store.md` — layout em disco e garantias de integridade
- `../../visual_perception/docs/embedding_space.md`, `dense_region_association.md`,
  `feature_resolution_enhancement.md`, `feature-extraction.md` e os documentos por adapter
  (`dinov2.md`, `dinov3.md`, `clip.md`, `alphaclip.md`, `siglip2.md`)

Esses continuam sob `visual_perception/docs/`: a decomposição moveu o código e deixou a
documentação de assunto atrás.

## Limitações

Dívida real desta branch, registrada em vez de escondida:

- **`FeatureResolutionEnhancement` é um port sem implementação concreta.** A fronteira validada
  existe e tem dez invariantes; backend que a satisfaça, não.
- **Upstream regional explícito:** regiões e máscaras pertencem a `region_perception`. Extratores recebem as regiões finais e o resolver do artifact selecionado; o artifact visual conserva somente a referência ao produtor regional.
- **Este pacote não é uma capability de topo nem um stage.** É um submódulo de
  `visual_perception`, que no catálogo são os componentes `visual_features.dense` e
  `visual_features.region` do stage `visual_perception` — e sua saída vai para o
  `VisualEvidenceArtifact`. Consumidores de outras capabilities importam pela fachada
  `contextmap.visual_perception`, que reexporta `VisualFeature`, `FeatureId` e `FeatureScope`.

Resolvido nesta branch: `VisualFeaturesRunWriter`/`Reader` e o manifest de run foram removidos.
Sem stage próprio não havia quem produzisse nem quem lesse o artifact — zero consumidores em
`src/` e em `tests/` —, e o reader devolvia `Sequence[dict]` em vez de contrato tipado. Era
estrutura sem demanda, contra AGENTS.md §14.
