# Persistência de payload de feature

Este documento descreve `src/contextmap/visual_perception/visual_features/feature_payload_store.py`: onde e como o array numérico de um `VisualFeature` (referenciado apenas por `payload_reference`, nunca inline — `contracts.md`, #48) é persistido, indexado, e carregado sob demanda.

## Formato: `.npy`, não Parquet/pickle

Cada payload é um arquivo `.npy` (formato nativo do NumPy) — preserva shape/dtype/valores exatos (incluindo byte order não nativo, já codificado na própria string de dtype), não exige nenhuma dependência de runtime nova além do NumPy, e dá acesso aleatório trivial já que cada feature é seu próprio arquivo. `allow_pickle=False` em toda leitura/escrita: nenhum payload pode conter código Python arbitrário.

## NumPy nunca na fronteira pública, mas livre internamente

`docs/shared-primitives.md` estabelece que primitivas públicas compartilhadas não devem exigir `numpy.ndarray` como identidade do contrato — e `VisualFeature` já obedece isso (`shape`/`dtype` são primitivos). Este módulo é a exceção esperada: cálculos/implementação internos de uma capability podem usar NumPy livremente. Para minimizar o alcance dessa dependência, `import numpy as np` só acontece **dentro** de `FeatureStoreWriter.write()` e `FeatureStoreReader.load()` — abrir um run, listar `feature_keys()`, ou ler um `FeaturePayloadEntry` nunca importa NumPy.

## Layout no artefato de run

```text
outputs/
└── features/
    ├── feature-index.jsonl              # header de schema + um FeaturePayloadEntry por linha
    └── frame-000120/
        ├── dense-<feature-id>.npy
        └── region-0001-<feature-id>.npy
```

O caminho exato de cada payload é `feature.payload_reference` — o mesmo valor já presente no `VisualFeature` persistido em `outputs/results.jsonl`. Este módulo nunca inventa o caminho; ele apenas grava no caminho que o produtor da feature já declarou.

## Metadados sem carregar o array

`FeatureStoreReader.open(root)` lê apenas `feature-index.jsonl`. A primeira linha identifica explicitamente `record_type="feature_index"` e `schema_version`; versões desconhecidas são rejeitadas antes de decodificar entradas. `entry(source_observation_id, feature_id)` devolve um `FeaturePayloadEntry` (shape, dtype, embedding_space_id, hash, proveniência) sem tocar o arquivo `.npy`. Só `load(source_observation_id, feature_id)` lê e decodifica o payload — e persistir um payload é opt-in por feature: um `VisualFeature` sem `add_feature_payload()` correspondente ainda aparece normalmente em `outputs/results.jsonl`, só não tem array carregável nesse run.

A chave do store é `(source_observation_id, feature_id)` e preserva explicitamente a aquisição física e a identidade da feature. O `FeatureId` combina run visual, observação, produtor e índice. Writer e reader rejeitam chaves duplicadas e colisões de `payload_reference`; nenhuma entrada pode sobrescrever outra silenciosamente.

## Integridade

`load()` verifica, nesta ordem: arquivo presente; hash de conteúdo (`sha256`, lido em blocos por `contextmap.shared.hash_file`, a mesma regra do inventário de todo run artifact) igual ao indexado; arquivo `.npy` válido (erro decodificando -> `FeaturePayloadIntegrityError`); shape decodificado igual ao indexado; dtype decodificado igual ao indexado. Qualquer divergência levanta `FeaturePayloadIntegrityError` com diagnóstico legível, nunca um array silenciosamente incorreto.

`write()` valida shape/dtype do array recebido contra os metadados do `VisualFeature` **antes** de gravar — um array incompatível com o que a feature declara nunca chega a ser persistido.

## Fronteira do artefato

Toda referência de caminho é resolvida e validada como estritamente dentro do root do feature store (`_resolve_within_root`) — um `payload_reference` malicioso ou malformado (ex. contendo `../`) nunca escreve/lê fora do artefato.

## Integração com `VisualEvidenceArtifact`

`VisualEvidenceRunWriter.add_feature_payload(feature, source_observation_id, array)` grava imediatamente o array em staging, sem reter todos os payloads em memória. A identidade da observação deve coincidir com `feature.source_observation_id`. `finalize()` exige que cada payload resolva para exatamente um `VisualFeature` de `outputs/results.jsonl` na mesma observação e compara scope, embedding space, shape, dtype, normalização e referência. O índice e os hashes entram no inventário antes da publicação atômica do artifact, descrita em [`visual-evidence-artifact.md`](visual-evidence-artifact.md).

`VisualEvidenceRunReader.feature_store()` abre o `FeatureStoreReader` de um run; quando nenhum payload foi persistido, devolve um reader vazio (não é erro).

O contrato `VisualEvidenceArtifact/0.1.0` declara o artifact regional selecionado; regiões e máscaras continuam pertencendo ao produtor regional.
