# Persistência de payload de máscara

Este documento descreve `src/contextmap/region_perception/artifacts/mask_store.py`: onde e como a máscara de um `Region2D` (referenciada apenas por `mask_reference`, nunca inline em `outputs/regions.jsonl` — `architecture.md`, #378) é persistida, indexada e carregada sob demanda.

## Problema, decisão e justificativa (#378)

- **Problema**: `encode_region()` inlinava `mask.to_dict()` — um array JSON de um inteiro por pixel cobrindo a imagem inteira, mesmo quando a caixa delimitadora da região era pequena. Uma máscara 640x480 custava cerca de 0,92 MB de JSON por região; um frame com dezenas de regiões virava uma única linha de dezenas de MB, e reabrir o run materializava cada máscara como `tuple[bool, ...]` em memória.
- **Justificativa**: `outputs/regions.jsonl` deve escalar com metadados de região (KB por frame), e payloads de pixel devem ser compactos e carregados sob demanda, do mesmo jeito que payloads de feature (`feature_store.py`) já são.
- **Entrada**: `Region2D.mask` (um `InlineMask` cobrindo a imagem inteira) para cada região publicada em `RegionPerceptionRunWriter`.
- **Processamento**: bit-packing (`numpy.packbits`) do array booleano completo — não recorte pela caixa delimitadora — persistido como `.npy`, indexado por `(source_observation_id, region_id)`.
- **Saída**: `outputs/masks/<source_observation_id>/<region_id>.npy` (payload compacto) + `outputs/masks/mask-index.jsonl` (hash, shape, referência); a região em `outputs/regions.jsonl` carrega só `mask_reference`.
- **Métrica**: tamanho de `encode_region()`/`outputs/regions.jsonl` deixa de escalar com resolução de imagem; round trip bit-exato via hash de conteúdo.
- **Teste**: `tests/visual_perception/test_mask_store.py` (writer/reader/integridade), `tests/unit/region_perception/test_artifacts.py`, `tests/unit/region_perception/test_mask_externalization.py`.

## Por que bit-packing sem recorte pela caixa delimitadora

A issue sugeria RLE ou um recorte bit-packed pela caixa delimitadora como alternativas compactas. Bit-packing da imagem inteira foi escolhido por ser a correção menor e mais simples que já resolve o alvo numérico pedido:

- 8x de redução sobre um byte por pixel (640x480 empacotado = 38.400 bytes, independente do conteúdo) e cerca de 24x sobre o array JSON que substitui;
- mover o payload para fora de `outputs/regions.jsonl` já domina a redução pedida (o arquivo volta a escalar só com metadados) — o bit-packing reduz ainda mais o `outputs/masks/` sem depender do formato do conteúdo (ao contrário de RLE);
- mantém o shape do payload idêntico a `(image_height, image_width)`: nenhuma contabilidade de offset de recorte, nenhuma transformação de coordenada para inverter na leitura, e nenhuma divergência com o invariante de imagem inteira que `InlineMask`/`_validate_geometry_bounds` já impõe (`region_perception/region_geometry.py`).

Um recorte pela caixa delimitadora foi considerado e rejeitado como complexidade desnecessária para o tamanho que esta issue efetivamente pede (AGENTS.md §13/§14, KISS/YAGNI). `numpy.unpackbits` (limitado à contagem exata de bits original) garante round trip bit-exato.

## NumPy nunca na fronteira pública, mas livre internamente

Como em `feature_store.py`, `import numpy as np` só acontece **dentro** de `MaskStoreWriter.write()` e `MaskStoreReader.load()` — abrir um run, listar `region_keys()`, ou ler um `MaskPayloadEntry` nunca importa NumPy.

## Layout no artefato de run

```text
outputs/
└── masks/
    ├── mask-index.jsonl              # header de schema + um MaskPayloadEntry por linha
    └── frame-0001/
        └── region-0001.npy
```

O caminho de cada payload é `<source_observation_id>/<region_id>.npy`, relativo à raiz do mask store; `Region2D.mask_reference` grava o caminho completo relativo ao artefato (`outputs/masks/<source_observation_id>/<region_id>.npy`).

## Metadados sem carregar o payload

`MaskStoreReader.open(root)` lê apenas `mask-index.jsonl`. A primeira linha identifica explicitamente `record_type="mask_index"` e `schema_version`; versões desconhecidas são rejeitadas antes de decodificar entradas. `entry(source_observation_id, region_id)` devolve um `MaskPayloadEntry` (width, height, hash, tamanho) sem tocar o arquivo `.npy`. Só `load(source_observation_id, region_id)` lê, verifica o hash e desempacota o payload.

A chave do store é `(source_observation_id, region_id)`. IDs publicados são qualificados por
run regional e observação; a chave física preserva também a consulta exigida pela associação.
Writer e reader rejeitam chaves duplicadas e colisões de `payload_reference`.

## Integridade

`load()` verifica, nesta ordem: arquivo presente; hash de conteúdo (`sha256`) igual ao indexado; arquivo `.npy` válido (erro decodificando -> `MaskPayloadIntegrityError`); contagem de bits desempacotados igual a `width * height`. Qualquer divergência levanta `MaskPayloadIntegrityError` com diagnóstico legível, nunca uma máscara silenciosamente incorreta.

## Fronteira do artefato

Toda referência de caminho é resolvida e validada como estritamente dentro do root do mask store (`_resolve_within_root`) — um `payload_reference` malicioso ou malformado (ex. contendo `../`) nunca escreve/lê fora do artefato.

## Integração com `RegionPerceptionRunArtifact`

`RegionPerceptionRunWriter.externalize_region_mask(source_observation_id, region)` grava pixels
no staging e devolve a região com `mask=None` e referência relativa ao artifact. A repetição
com os mesmos pixels é idempotente; pixels diferentes para a mesma região são recusados.
`write_region()` exige que a referência pertença à região e observação, e ao store do produtor.
Nenhum vetor de pixels fica retido pelo writer entre frames.

`finalize()` exige que máscaras regionais staged tenham um consumidor publicado e valida
prompts/execuções antes da publicação atômica. Constraints `valid_region` e `exclusion_regions`
de `PreparedImage` também são externalizadas; não são máscaras de regiões descobertas.
O schema regional atual é `0.2.0`.

O writer e `RegionPerceptionRunReader.mask_resolver()` satisfazem `RegionMaskResolver`:
`resolve_mask(reference) -> InlineMask`. `region_mask_pixels(region, resolver)` resolve uma
máscara inline ou referenciada; uma caixa sem máscara retorna `None`. Referência ausente,
externa ou corrompida é um erro explícito. Cada leitura verifica o payload, sem cache implícito.

`RegionPerceptionRunReader.read_regions()` não carrega pixels.
`iter_frames()` hidrata constraints de um frame por vez; `frame(source_observation_id)` hidrata
somente a observação solicitada. Features e claims permanecem em artifacts próprios, sem
copiar as máscaras regionais.

## Consumo por Sensor Association

`associate_regions(resolution, regions, region_artifact_id=..., sequence_artifact_id=...,
mask_loader=...)` seleciona as regiões e o produtor explicitamente.
`RegionPerceptionRunReader.mask_store()` satisfaz a porta local `RegionMaskLoader`, com
`load(source_observation_id, region_id) -> InlineMask`. O runtime entrega esse reader à associação;
a ausência de máscara é registrada como `SkipReason.NO_INLINE_MASK`, sem fabricar suporte 3D.

Os testes atuais de externalização, referência, memória, staging e integridade estão em
`tests/unit/region_perception/test_mask_externalization.py` e
`tests/unit/region_perception/test_regional_execution_streams.py`.
