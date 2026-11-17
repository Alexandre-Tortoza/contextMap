# Cor semântica sobrevivendo à simplificação por voxel (Fase 3)

Fecha o caminho `spatial_support` → viewer identificado na Fase 1: a "perda de cor" no viewer
**não** era bug de associação (projeção/membership medidos a 0 px / 100% em
`../rgb-lidar-association-validation-20260930`), e sim a coloração keiar pelo `geometry_id` =
índice posicional na nuvem bruta, que não sobrevive à apresentação por centroide de voxel.

## O que já existia (nada disso foi alterado)

- `contextmap.spatial_support` completo: `SpatialCell` com `aggregate_index` (índice de
  transporte do voxel) + `entity_support`, `build_semantic_spatial_support`,
  `derive_semantic_spatial_support_artifact`, writer/reader do artefato.
- **O viewer (`contextmap2-viewer`) inteiro — cliente E servidor — já consome isto:**
  - `app/composables/useVoxelGeometrySupport.ts` busca
    `/api/v1/maps/{id}/geometry/voxel/{source}/support`, keyed por `aggregate_index`;
  - `server/api/v1/maps/[contextMapId]/geometry/voxel/[sourceId]/support.get.ts` +
    `server/utils/spatial-support-artifact.ts` localizam o `SemanticSpatialSupportArtifact` por
    identidade (`source_map_id` + `run_id` da agregação voxel na lineage) e servem
    `outputs/cells.jsonl`;
  - 404 (nenhum artefato configurado) = "sem coloração", nunca erro.

## O gap real

Nenhum `SemanticSpatialSupportArtifact` (nem agregação voxel) tinha sido **produzido** para os
mapas em disco, e o viewer usa `NUXT_SPATIAL_SUPPORT_ARTIFACTS_ROOT` (default vazio). Sem o
artefato, o endpoint responde 404 e o voxel fica sem cor semântica.

## O que este experimento faz

`derive_voxel_support.py`, para `workspace/frame-10_29-full-run` (mapa real com entidades):

1. produz uma **agregação voxel** (`InterScanVoxelPolicy`, cell 0,10 m) via
   `VoxelAggregationArtifactWriter` — a fonte voxel do viewer;
2. **deriva** o `SemanticSpatialSupportArtifact` via `derive_semantic_spatial_support_artifact`;
3. verifica integridade e imprime as três envs do viewer.

## Resultado (ver `results.txt`)

- Agregação voxel: 426.576 pts → **30.491 voxels**.
- Suporte semântico: **818 células, 28 entidades cobertas, `verify_integrity()==[]`**.
- **Contrato do viewer batido**: `run_id` da agregação e `source_map_id` casam entre voxel e
  support; `outputs/cells.jsonl` com `{aggregate_index, entity_support:[{entity_id,
  support_weight}]}`; schema `0.1.0`.

Ou seja: apontando as três envs para estes artefatos, os centroides de voxel são coloridos por
entidade — **a cor sobrevive à simplificação**. O viewer **não precisa de mudança de código**.

## Rodar

```bash
# do repo root; python 3.12 com numpy (mise 3.12.14)
python experiments/semantic-voxel-color-20260930/derive_voxel_support.py
# depois, no viewer (contextmap2-viewer):
cp .env.semantic-voxel .env   # ou exporte as três NUXT_*_ROOT impressas
npm run preview               # ver memória viewer_run_infra (não use nuxt dev)
```

Artefatos vão para `outputs/semantic-voxel-color-20260930/` (gitignored, imutáveis; remova antes
de re-derivar).

## Fora de escopo

- Wiring de CLI/build (`context build` derivar/publicar automaticamente): o caminho de build está
  em WIP no `milestone/canonical-2`; deixado como follow-up limpo. A derivação em si (biblioteca)
  já é coberta por `tests/end_to_end/test_semantic_spatial_support_derivation.py`.
- Multi-LOD (cell 0,05 m além de 0,10 m): trivial (repetir com outro `cell_m`), não necessário
  para provar o loop.
- Validação visual real no navegador (lançar o viewer): infra pesada; a verificação aqui é no
  nível do contrato que o servidor do viewer exige.
