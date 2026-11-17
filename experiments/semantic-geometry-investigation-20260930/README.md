# semantic-geometry-investigation-20260930

Investigação read-only: a semântica altera a geometria? Projeção/calibração, máscaras,
explosão de labels e promoção a entidades, no run `workspace-old/frame-10_31-{raw,voxel}`
(imagem `camera_1_image_raw-000055`). Conclusões em `REPORT.md`.

Nenhum código de produção nem artefato do run foi alterado. A associação é reimplementada em
numpy (`scripts/assoc.py`) e validada contra o artefato (0/83 regiões divergentes) antes de
qualquer variação.

## Ambiente

- `PY=~/.cache/contextmap2-audit/venv/bin/python` (numpy, scipy, cv2, torch, transformers,
  `contextmap` editável).
- `p6_sam2_rerun.py` usa `~/.cache/contextmap2-ablation/venv/bin/python` (sam2) e GPU.
- `HF_HUB_OFFLINE=1`; `SCRATCH=<diretório temporário>` para os scripts que escrevem fora do bundle.

## Ordem

```bash
cd experiments/semantic-geometry-investigation-20260930/scripts
$PY p2_geometry_snapshots.py                      # snapshots 01..08, deltas, hashes
SCRATCH=... PYTHONPATH=../../../src $PY p2_test_e_semantics_off.py   # teste E (reconstrói os mapas)
$PY p3_map_consistency.py                         # fantasmas do mapa, varredura de offset
$PY p3_time_offset_deskew.py                      # offset GT<->header isolado do deskew
$PY p3_extrinsic_direction.py                     # sentido de laser_to_imu
$PY p4_calibration_photometric.py                 # critério fotométrico do V1 com GT
$PY p4_validate_reimpl.py                         # reimplementação == artefato
$PY p5_association_ablation.py                    # calibração x geometria x oclusão
$PY p5_occlusion_anatomy.py
$PY p4_overlays.py                                # figuras f01..f06
SCRATCH=... PYTHONPATH=../../../src ~/.cache/contextmap2-ablation/venv/bin/python p6_sam2_rerun.py
$PY p6_mask_quality.py
$PY p7_label_provenance.py                        # results/p7_label_provenance.csv
$PY p9_label_redundancy.py
$PY p10_entity_overlap.py
$PY p2_render_viewer_effect.py                    # f08
cd ../../.. && PYTHONPATH=src:experiments/semantic-geometry-investigation-20260930/scripts \
  $PY -m pytest -q -p no:cacheprovider experiments/semantic-geometry-investigation-20260930/diagnostic_tests
```

## Saídas

- `results/*.json|csv` — uma saída por script (nome = script).
- `figures/` — f01 RGB; f02 scan local projetado base vs corrigido (+ recortes); f03 visíveis
  do pipeline; f04 máscaras numeradas; f05 contagem de sobreposição; f06 pontos por máscara;
  f07 recortes enviados ao VLM; f08 efeito da cobertura do viewer (topo).
- `diagnostic_tests/` — testes A–E (pulam sem os artefatos do run).
