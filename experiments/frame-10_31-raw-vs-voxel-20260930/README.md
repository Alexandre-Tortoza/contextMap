# Frame 10:31 — duas runs completas: mapa RAW vs VOXEL, com ego-mask

Duas runs de pipeline **completas** (ingestão → context_map) do frame em **10:31 (631s)**,
ambas com a **ego-mask** (overlay `examples/v0.2.0/corridor-02-image-masks.toml`), diferindo
apenas no **mapa geométrico**: RAW (todos os pontos) vs VOXEL (`ScanVoxelPolicy`, 0,10 m).

Compartilha o que a imagem determina (state_estimation, visual_perception, semantic_scoring),
pois a imagem e a ego-mask são idênticas nos dois arms — isola a variável "mapa raw vs voxel"
e roda a percepção (GPU) só uma vez.

## Resultado

| | RAW | VOXEL |
|---|---:|---:|
| pontos do mapa | 2.322.324 | 402.626 |
| entidades | 38 | 33 |
| labels de veículo/carrinho | **0** | **0** |
| context_map_id | 13d4158495464f76887ac73ec0a150ad | d6aff901dcf9f8a82369a9cc1525bcd6 |

- **Ego-mask funcionou end-to-end:** na percepção do frame 55, **44 regiões rejeitadas por
  `exclusion_overlap`** (o rig) + **16 por `outside_valid_region`** (vinheta fisheye), restando
  83 regiões limpas. Nenhuma entidade final é o carrinho (contraste com `frame-10_29-full-run`,
  que tinha `entity-0001 "robotic vehicle"` + wheels/tire/dashboard/off-road).
- Frame 10:31 = índice de imagem **55** na janela ingerida (header-ts 1724621714.985 =
  recording-second 630.985 ≈ 631,0s).

## Ambiente (não commitado)

- Python com ML deps: `~/.cache/contextmap2-ablation/venv/bin/python` (sam2, torch 2.8+cu126).
- Provider real (SAM2/Qwen): módulo `ablation_providers` em
  `~/.cache/contextmap2-ablation/providers` (ver `providers.toml`). Base de config:
  `experiments/canonical-2-empirical-validation-20260929/config/base.template.toml` (params) +
  `examples/v0.2.0/canonical-config.toml` (seleções de backend). As 3 etapas de cauda
  (semantic_mapping/entity_resolution/spatial_relations) são construídas direto no driver com
  os valores canônicos do `e2e-real-canonical-run` (nenhum config commitado carrega esses params).

## Reproduzir

```bash
PY=~/.cache/contextmap2-ablation/venv/bin/python
PROV=~/.cache/contextmap2-ablation/providers
BASE=experiments/canonical-2-empirical-validation-20260929/config/base.template.toml
D=experiments/frame-10_31-raw-vs-voxel-20260930

# 0. ingestão da janela de 631s (câmera+lidar+imu) -> SEQUENCE_ID / POSE_SEQUENCE_ID
PYTHONPATH=src $PY experiments/canonical-2-empirical-validation-20260929/scripts/ingest_window_fresh.py \
  --bag datasets/corridor-02/corridor-02/corridor02.bag --pose-file datasets/corridor-02/corridor-02-gt.txt \
  --outputs-root workspace/frame-10_31-runs-20260930 --start-offset-s 631 --end-offset-s 634 --padding-s 3

# 1..3. foundation (raw+voxel) -> percepção (frame 55, GPU) -> downstream (raw+voxel -> context_map)
for STAGE in foundation perception downstream; do
  PYTHONPATH=src:$PROV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 $PY $D/run_two_arms.py \
    --workspace workspace/frame-10_31-runs-20260930 \
    --sequence-id <SEQ> --pose-id <POSE> \
    --config examples/v0.2.0/canonical-config.toml --config $BASE \
    --config $D/providers.toml --config examples/v0.2.0/corridor-02-image-masks.toml \
    --frame-index 55 --stage $STAGE
done
```

## Ver no viewer

Cada mapa é colorido **nativamente** por entidade (as entidades referenciam os índices do
próprio mapa; não precisa do artefato spatial_support da Fase 3, que era para apresentar um
mapa RAW *como* voxel). No `contextmap2-viewer`:

```bash
cp .env.frame-10_31 .env    # aponta CONTEXT/UPSTREAM roots para workspace/frame-10_31-runs-20260930
npm run preview             # os dois mapas (…-raw e …-voxel) aparecem no catálogo lado a lado
```

Artefatos em `workspace/frame-10_31-runs-20260930/` (gitignored, imutáveis).

## Gotcha: locator do context_map após reorganizar

O `context_map/manifest.json` grava o `locator` da geometria como caminho RELATIVO ao dir do
context_map (ex.: `../geometric_mapping`). Se os artefatos forem MOVIDOS depois da run, esse
locator quebra e o viewer não resolve a geometria (`geometry/manifest` retorna `chunks:[]`,
`overview:null`) → o mapa não plota. Como o locator é imutável (entra no hash), a correção é
re-gerar só o context_map com os inputs já nos caminhos finais do bundle
(`fix_context_map.py`), para o locator virar `../geometric_mapping` (irmão), como nos outros
bundles (frame-10_29-full-run). Melhor ainda: rodar a pipeline já com o output no layout final
para não precisar reorganizar.
