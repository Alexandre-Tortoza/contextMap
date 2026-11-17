# Code Map: visual_features

Responsabilidade por arquivo. Serve como contrato para reviews: se uma função nova não cabe na
responsabilidade declarada do arquivo, isso é discussão de ownership, não detalhe de implementação.

---

## visual_evidence.py — 252 linhas

**Responsabilidade:** contratos da feature como evidência e identidade do espaço de embedding.

**Entrada → saída:** nada (só contratos) → `VisualFeature`, `FeatureId`, `FeatureScope`,
`PoolingMethod`, `EmbeddingSpace`, codecs, `embedding_space_fingerprint`, as duas verificações de
compatibilidade.

**Conceitos:** `FeatureScope` = `DENSE | GLOBAL | REGION`; `PoolingMethod` = `AVERAGE | MAX |
ATTENTION`; `EmbeddingSpace` com sete campos, todos no fingerprint — mesma dimensão não é mesmo
espaço; `EMBEDDING_SPACE_SCHEMA_VERSION = "0.1.0"`; `VisualFeature` carrega `payload_reference`,
nunca o array.

**Não deve ter:** array numérico, código de modelo, lógica de pooling, persistência.

---

## region_feature_pooling.py — 632 linhas

**Responsabilidade:** associar deterministicamente uma região congelada à grade densa, e agregar.

**Entrada → saída:** `Region2D` + `DenseFeatureMap` + array → `RegionPoolingResult` com
`VisualFeature` de escopo `REGION`, diagnósticos e proveniência.

**Conceitos:** `POOLING_POLICY = "mask_weighted_mean_preserve_l2_v2"`, versionada e presente na
proveniência; `DenseFeatureSampling` descreve o mapeamento, e é por isso que o pooling **não tem
ramo** específico de DINO, CLIP ou realce; `map_box_to_grid_cells`; `EmptyRegionSupportError`
quando nenhuma célula cai na região; `FeatureResolutionEnhancementProvenance` com a lineage do
realce.

**Não deve ter:** ramo por backend, alteração da região recebida, nem o array na identidade
persistida. O NumPy é importado dentro de `pool_region_feature`, deliberadamente.

---

## feature_payload_store.py — 439 linhas

**Responsabilidade:** persistir o array de cada feature e reencontrá-lo por acesso aleatório.

**Entrada → saída:** `VisualFeature` + `SourceObservationId` + array → `FeaturePayloadEntry` e um
arquivo `.npy`; na leitura, referência → array validado contra shape e dtype.

**Conceitos:** `FEATURE_INDEX_FILENAME = "feature-index.jsonl"` e
`FEATURE_INDEX_SCHEMA_VERSION = "0.1.0"`; contêiner `.npy` nativo, que preserva shape, dtype e
byte order não nativo; um arquivo por feature, o que dá acesso aleatório trivial; hash de conteúdo
por entrada; `_resolve_within_root` impede referência que escape da raiz; recusa de chave
`(observação, feature)` e de `payload_reference` duplicados.

**Não deve ter:** regra de pooling, semântica, nem decisão de quais features extrair.

**Nota:** a docstring aponta para `visual_perception/docs/feature_store.md`, que descreve o layout
em disco e as garantias de integridade — documento que continua sob o owner anterior.

---

## feature_enhancement.py — 129 linhas

**Responsabilidade:** a fronteira validada do realce opcional de resolução.

**Entrada → saída:** `DenseFeatureMap` de origem + `FeatureResolutionEnhancement` →
`DenseFeatureMap` realçado, ou `FeatureResolutionEnhancementError`.

**Conceitos:** dez invariantes, e todos existem pela mesma razão — um realce que reaproveite
identidade, payload ou embedding space da origem tornaria a feature realçada indistinguível da
nativa no artifact. Verifica: lineage presente e nomeando exatamente o mapa de origem;
proveniência batendo com o backend selecionado; grade, imagem de origem e embedding space
coerentes com a saída; aumento estrito de ao menos uma dimensão sem redução da outra;
`feature_id`, `payload_reference` e — sob normalização diferente — `embedding_space_id` distintos.

**Não deve ter:** implementação de realce. O estágio consome e produz o mesmo contrato; o
aprendizado é detalhe de backend.

**Histórico:** este módulo foi reduzido a um stub de 26 linhas durante a decomposição, perdendo as
dez validações, e foi restaurado. O teste que o protege não era coletável, então a perda passou
silenciosa.

---

## ports.py — 89 linhas

**Responsabilidade:** os dois pontos de substituição.

**Entrada → saída:** `FeatureExtractor`: `PreparedImage` (+ regiões, se o escopo pede) →
`Sequence[VisualFeature]`, com `backend_provenance()` e `required_scope()`.
`FeatureResolutionEnhancement`: `DenseFeatureMap` → `DenseFeatureMap`.

**Conceitos:** `required_scope()` é o que decide se o extrator recebe regiões, em vez de uma
assinatura obrigatória comum a todos.

**Não deve ter:** construção de backend nem import de SDK.

---

## adapters/ — 3.809 linhas em 7 módulos

**Responsabilidade:** ligar um modelo concreto ao port; único lugar da capability onde um SDK pode
ser importado.

| Arquivo | Linhas | Escopo / função |
|---|---:|---|
| `alphaclip.py` | 917 | `REGION` |
| `siglip2.py` | 711 | `DENSE`, `GLOBAL` — região explicitamente não implementada |
| `clip.py` | 684 | `GLOBAL`, `REGION` |
| `dinov2.py` | 617 | `DENSE` |
| `dinov3.py` | 550 | `DENSE` |
| `_feature_values.py` | 45 | validação numérica privada, compartilhada pelos adapters daqui |

**Não deve ter:** regra de domínio. Um adapter traduz entre contrato e SDK, declara seu escopo e
reporta sua proveniência.

**Onde não estão:** as torres de texto (`clip_text.py`, `alphaclip_text.py`) foram para
`contextmap.semantic_scoring.adapters`. Elas codificam hipótese, não imagem, e enquanto estavam
aqui fechavam o ciclo `visual_features ↔ semantic_scoring`. Cada uma declara o embedding space
da torre visual correspondente, que continua definido neste pacote.

---

## artifacts/ — removido

Havia aqui um `run_manifest.py`, um `run_writer.py` e um `run_reader.py` (146 linhas) que
pretendiam persistir e reler uma execução de extração de features. Foram removidos nesta branch:

- **zero consumidores** em `src/` e em `tests/` — o runtime persiste pelo `VisualEvidenceRunWriter`
  (metadados) e pelo `FeatureStoreWriter` (payloads);
- `read_features()` devolvia `Sequence[dict]`, não `Sequence[VisualFeature]`;
- `run_writer.write_feature(feature_dict)` era homônimo de `FeatureStoreWriter.write_feature`,
  sem relação com ele — a colisão de nome convidava ao erro.

Por AGENTS.md §14, estrutura sem consumidor sai.
