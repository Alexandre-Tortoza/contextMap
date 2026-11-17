# Diagnósticos auditáveis de Feature Extraction

Este documento descreve `src/contextmap/visual_perception/feature_diagnostics.py` e sua integração atômica com `VisualEvidenceRunWriter`.

## Separação entre contrato, métrica e debug

```text
outputs/features/                         payloads contratuais + índice
metrics/feature-extraction.jsonl          métricas contratuais, quando há diagnóstico
debug/30-feature-extraction/              inspeção humana, nunca dependência downstream
```

`metrics/feature-extraction.jsonl` é contratual e existe **quando o run tem ao menos um diagnóstico de feature**; o nível de debug nunca decide se ele existe. Todo run do runtime com um extrator denso o tem: DINOv2, DINOv3 e SigLIP2 denso entregam um diagnóstico por feature densa produzida (ver "Diagnóstico entregue pelo extrator denso"). Um run sem diagnóstico (um extrator que não os entrega, como os de região hoje) não tem o arquivo, e nenhuma feature densa dele pode ser reaberta como mapa espacial. Desabilitar debug não remove `VisualFeature`, payload, índice nem métricas de status/timing/memória. Consumidores nunca dependem de `__DEBUG__/`.

## Schema comum

`FeatureExtractionDiagnostic` registra de forma backend-agnostic:

- observação e imagem preparada (referência/dimensões);
- stage, status e backend/model/checkpoint/configuração;
- feature, escopo, shape, dtype, normalização e payload;
- `EmbeddingSpace` completo e fingerprint;
- preprocessing ordenado;
- timing, memória, warnings, falha ou abstention.

`DenseFeatureDiagnostic` acrescenta o run dono da feature (`source_artifact_id`), grid, origem (`origin_x`, `origin_y`), stride, suporte e transformação espacial. Origem, stride e suporte precisam ser finitos (NaN/Inf são rejeitados) e stride e suporte precisam ser positivos: as mesmas regras de `DenseFeatureSampling`, para que o que foi persistido sempre reconstrua um sampling válido. Essa geometria é contratual: ela vai em `metrics/feature-extraction.jsonl` (chave `dense`, com o tamanho da imagem preparada) em qualquer nível de debug, e reconstrói exatamente o `DenseFeatureSampling` do mapa. Sem ela o payload denso não vira mapa espacial reaberto de um run; uma execução real com DINOv3 mostrou que antes a geometria só existia em `__DEBUG__/` e sem a origem, obrigando o leitor a supor `(0, 0)`. `RegionFeatureDiagnostic` acrescenta região/box, view de suporte, mask reference/hash, transformação e estatísticas de pooling quando aplicáveis.

Falha ou abstention pode existir sem inventar `feature_id`/payload (e, por isso, não é conferida contra nenhuma feature). Eventos bem-sucedidos ou com warning exigem metadata completa. Estatísticas de pooling são um grupo coerente: todas presentes ou todas ausentes.

## Vínculo com a feature persistida

`metrics/feature-extraction.jsonl` é contratual, então um registro que apenas confere por hash não basta: ele pode pertencer a outro payload ou ser incompatível com ele. Antes de publicar o run, `VisualEvidenceRunWriter.finalize()` exige que todo diagnostic `SUCCEEDED` ou `WARNING` descreva **exatamente uma** `VisualFeature` dos resultados do próprio run, localizada por `source_observation_id` + `feature_id`. Sem correspondência, ou com mais de um diagnostic para a mesma feature (a geometria ficaria ambígua), o `finalize()` falha com `VisualEvidenceRunError` e nenhum diretório de run é criado.

Para a feature encontrada, o writer compara:

| Diagnostic | `VisualFeature` |
| --- | --- |
| `scope` | `scope` |
| `output_shape` | `shape` |
| `dtype` | `dtype` |
| `normalization` | `normalization` |
| `payload_reference` | `payload_reference` |
| `backend` | `provenance` |
| fingerprint de `embedding_space` | `embedding_space_id` |
| `region.region_id` (escopo `REGION`) | `region_id` |

Para o escopo `DENSE`, o writer também reconstrói o `DenseFeatureMap` exatamente como um consumidor de `metrics/` o faria (`DenseFeatureSampling` a partir da geometria persistida, mais a feature do resultado). Assim o shape precisa ser `(altura, largura, canais)` e `(grid_height, grid_width)` precisa ser igual a `feature.shape[:2]`; uma grade transposta é outro mapa e é rejeitada. Além disso, `dense.source_artifact_id` precisa ser `str(run_id)` do run que persiste a feature: a identidade da feature combina o run visual, a observação física, o produtor de features e o índice de emissão.

O payload continua opcional por feature: o diagnostic se vincula à feature do resultado, e, quando o payload existe, `finalize()` já o cruza com essa mesma feature (ver [`visual-evidence-artifact.md`](visual-evidence-artifact.md)). `VisualEvidenceRunReader.verify_integrity()` segue conferindo apenas o inventário (presença, tamanho e hash); a garantia semântica vale para todo run finalizado por este writer e não é refeita na leitura.

## Diagnóstico entregue pelo extrator denso (#651)

O sink que um extrator recebe (`FeatureBuildScope.payload_sink` no runtime; `VisualEvidenceRunWriter` o satisfaz) tem dois canais: `add_feature_payload()` e `add_feature_diagnostic()`. Um extrator denso chama os dois para cada feature que produz: o payload e, com `dense_extraction_diagnostic()`, um evento `SUCCEEDED` cuja geometria `dense` é exatamente o `DenseFeatureSampling` que o próprio adapter calculou, com o estágio (`stage_id`), a imagem preparada, o `EmbeddingSpace` completo e a identidade da feature. O evento não tem timing, então uma extração idêntica se descreve igual. O port público `FeatureExtractor` não muda: `extract()` continua devolvendo só a `VisualFeature`.

`dense_feature_map_of(feature, dense, source_image_width=..., source_image_height=...)` é a única regra que transforma essa geometria de volta num `DenseFeatureMap`: o writer valida o diagnóstico com ela antes de publicar, e `VisualEvidenceRunReader.dense_feature_map()` reabre o mapa com ela (via `dense_feature_map_from_record`, o inverso da codificação da chave `dense`). Ver [`visual-evidence-artifact.md`](visual-evidence-artifact.md).

## Níveis de debug

### `none`

Persiste apenas `metrics/feature-extraction.jsonl` (que inclui a geometria densa), além dos outputs contratuais normais. Nenhum diretório `__DEBUG__/` é criado.

### `standard`

Adiciona:

```text
debug/30-feature-extraction/
├── feature-summary.json
├── feature-map-meta.json              # quando há feature densa
├── embedding-spaces.json
├── extraction-events.jsonl
├── dense/                             # previews fornecidos pelo produtor
└── regions/<region-id>/
    ├── feature-meta.json
    └── support-preview.*              # quando fornecido
```

O summary agrupa contagens por escopo, status e backend, além de duração total e pico máximo, permitindo comparar DINOv2, DINOv3, CLIP, AlphaCLIP ou fakes pelo mesmo schema.

### `full`

Inclui tudo de `standard`, previews marcados como full e `regions/<id>/pooling.json` com política, células contribuintes, peso e cobertura.

## Previews

`FeatureDiagnosticPreview` recebe bytes já renderizados pelo produtor e um caminho descritivo iniciado por `dense/`, `regions/` ou `diagnostics/`. O writer não interpreta imagem e não transforma canais arbitrários de embedding em heatmaps. Paths absolutos/escape, duplicatas e previews acima de 5 MiB são rejeitados.

A existência de previews é opcional: só devem ser produzidos quando houver método determinístico e significado científico claro. Payload denso bruto nunca deve ser copiado para debug.

## Imutabilidade e inventário

`VisualEvidenceRunWriter.add_feature_diagnostic()` e `add_feature_preview()` apenas enfileiram conteúdo. `finalize()` escreve tudo no diretório temporário, inclui cada arquivo no `file_inventory`, verifica hashes e só então publica o run por rename atômico. Após finalização, novos diagnósticos/previews são rejeitados.

A integração de diagnostics não exige campos dedicados no `RunArtifactManifest`; o `file_inventory` aberto já representa esses arquivos. O schema global do `VisualEvidenceArtifact` está em `0.6.0` por mudanças em outros contratos; o layout de diagnostics não mudou, e o runtime passar a gravar `metrics/feature-extraction.jsonl` (#651) é aditivo.

Ver [`feature_store.md`](feature_store.md) para payloads contratuais e [`visual-evidence-artifact.md`](visual-evidence-artifact.md) para o artefato completo.
