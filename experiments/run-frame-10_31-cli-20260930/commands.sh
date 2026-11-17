#!/usr/bin/env bash
# Run de dois frames do corridor-02 (imagens 55 = 10:31 e 61), braços raw e voxel, só pelo CLI.
# Executar da raiz do repositório. Ambiente: o venv de ablação (torch/sam2/rosbags) e os providers
# de SAM2/Qwen (ver a memória "Full pipeline run recipe").
set -euo pipefail

E=experiments/run-frame-10_31-cli-20260930
W=workspace/run-frame-10_31          # run store: sequências, runs, branches
M=workspace                          # maps root: workspace/<branch>/<NNNNNN>
PY=~/.cache/contextmap2-ablation/venv/bin/python
export PYTHONPATH="src:$HOME/.cache/contextmap2-ablation/providers"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HEAD + sha256 do diff local (rastreado e staged) e dos fontes não rastreados de src/, gravado
# antes da run:
#   echo "$(git rev-parse --short HEAD)+local.$( { git diff HEAD --binary; git ls-files -o \
#     --exclude-standard -z src | sort -z | xargs -0 sha256sum; } | sha256sum | cut -c1-12)"
# As fundações (run-0001, run-0003) foram gravadas com a3a175f+local.51d175b60ae9, que cobria só `git diff`;
# as context runs e os builds, depois do prompt v2, com a identidade atual deste arquivo.
CODE=$(cat $E/code-identity.txt)

STACK=(
  -c examples/v0.2.0/canonical-config.toml
  -c experiments/canonical-2-empirical-validation-20260929/config/base.template.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml
  -c examples/v0.2.0/corridor-02-image-masks.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml
  -c $E/run.toml
)
# TOML não tem null: a cena herdada das camadas de baixo é removida explicitamente.
NO_SCENE=(--set 'components.visual_perception.semantic_interpretation.qwen.prompt_policy.scene=null')
NO_MODELS=(--set 'resources.providers={}')   # etapas que não rodam percepção não carregam modelos
CALIBRATION=experiments/correction-benchmark-10_31-20260930/calibration/refined-calibration.json

# 1. Sequência (câmera/LiDAR/IMU): mesma janela, correção de relógio e sincronização do E0,
#    calibração refinada (C2).
$PY -m contextmap ingest "${STACK[@]}" --source datasets/corridor-02/corridor-02/corridor02.bag \
  --sequence-name corridor-02 --topic rgb=/camera_1/image_raw --topic lidar=/velodyne_points \
  --topic imu=/imu/data --required rgb --required lidar --required imu \
  --clock-id corridor-02-header --sync-reference image --sync-tolerance-ns 50000000 \
  --on-problems warn --timestamp-policy $E/timestamp-policy.json \
  --window-clock "ros1_bag:datasets/corridor-02/corridor-02/corridor02.bag:recording_time" \
  --window-start 1724621712.6603158 --window-end 1724621721.6603158 \
  --calibration $CALIBRATION --output-dir $W/sequences/corridor-02/frame-10_31-refined --json

# 2. GT de pose (TUM) como sequência de pose.
$PY -m contextmap ingest "${STACK[@]}" --set components.ingestion.source_adapter.backend=pose_file \
  --source datasets/corridor-02/corridor-02-gt.txt --sequence-name corridor-02-pose \
  --topic pose=corridor-02-gt.txt --required pose --clock-id corridor-02-header \
  --sync-reference external_pose --sync-tolerance-ns 50000000 --on-problems warn \
  --timestamp-policy $E/timestamp-policy.json --calibration $CALIBRATION \
  --extra format=tum --extra parent_frame=map --extra body_frame=epson --extra pose_role=ground_truth \
  --output-dir $W/sequences/corridor-02-pose/frame-10_31-gt --json

# 3. Fundação raw (trajetória + mapa com deskew) -> run-0001.
#    catalog-foundation.json lista as duas sequências publicadas acima.
$PY -m contextmap run "${STACK[@]}" "${NO_MODELS[@]}" "${NO_SCENE[@]}" --workspace $W \
  --catalog $E/catalog-foundation.json --select ingestion=<SEQ> --select pose_ingestion=<POSE> \
  --stage state_estimation --stage geometric_mapping --code-identity "$CODE"

# 4. Fundação voxel: a mesma trajetória, selecionada, e o mapa com voxel.toml -> run-0003.
#    (run-0002 é o registro de uma tentativa bloqueada no pré-voo, sem artefato.)
$PY -m contextmap run "${STACK[@]}" -c $E/voxel.toml "${NO_MODELS[@]}" "${NO_SCENE[@]}" \
  --workspace $W --catalog $E/catalog-foundation-voxel.json --select ingestion=<SEQ> \
  --select state_estimation=<TRAJ> --stage geometric_mapping --code-identity "$CODE"

# 5. Branches: o nome é o map_id e reserva $M/<nome>. As branches foram recriadas duas vezes:
#    as context runs com os prompts region-interaction/v1 (run-0004) e /v2 (run-0005) falharam no
#    parse (ver README); os registros de branch e as pastas reservadas vazias saíram do workspace,
#    os runs ficaram no run store, fora de qualquer branch.
$PY -m contextmap context branch create frame-10_31-raw --from-run run-0001 "${STACK[@]}" \
  "${NO_MODELS[@]}" --workspace $W --maps-root $M --json > $E/branch-frame-10_31-raw.json
$PY -m contextmap context branch create frame-10_31-voxel --from-run run-0003 "${STACK[@]}" \
  "${NO_MODELS[@]}" --workspace $W --maps-root $M --json > $E/branch-frame-10_31-voxel.json

# 6. Percepção + associação no braço raw, um frame por processo (a GPU de 8 GB degrada o 2º frame
#    de um mesmo processo: memória "perception frame-major GPU residency") -> run-0006 (55) e
#    run-0007 (61), com a identidade de código a3a175f+local.63077e630372.
for FRAME in 000055 000061; do
  $PY -m contextmap context run --branch frame-10_31-raw "${STACK[@]}" "${NO_SCENE[@]}" \
    --set "inputs.observation_selection={\"kind\":\"explicit_ids\",\"observation_ids\":[\"camera_1_image_raw-$FRAME\"]}" \
    --workspace $W --maps-root $M --code-identity "$CODE" --json > $E/context-run-raw-${FRAME#0000}.json
done

# 7. Braço voxel: seleciona a percepção do mesmo frame do braço raw (labels idênticas) e só associa
#    -> run-0009 (55) e run-0010 (61). O catálogo vem do registro da context run raw. run-0008 é
#    uma tentativa bloqueada de antes de o `context run` aceitar --select (ele o ignorava).
#    Identidade de código a partir daqui: a3a175f+local.04d61c01a05c.
declare -A RAW_RUN=([000055]=run-0006 [000061]=run-0007)
for FRAME in 000055 000061; do
  PERC=$(python3 $E/scripts/perception_catalog.py $W ${RAW_RUN[$FRAME]} $E/catalog-perception-$FRAME.json)
  $PY -m contextmap context run --branch frame-10_31-voxel "${STACK[@]}" "${NO_MODELS[@]}" "${NO_SCENE[@]}" \
    --set "inputs.observation_selection={\"kind\":\"explicit_ids\",\"observation_ids\":[\"camera_1_image_raw-$FRAME\"]}" \
    --catalog $E/catalog-perception-$FRAME.json --select visual_perception=$PERC \
    --workspace $W --maps-root $M --code-identity "$CODE" --json > $E/context-run-voxel-${FRAME#0000}.json
done

# 8. Um build por branch: os dois frames fundidos -> $M/<branch>/000001 (run-0012 raw, run-0013
#    voxel). run-0011 é um build bloqueado antes de run.toml desligar semantic_scoring.
for BRANCH in frame-10_31-raw frame-10_31-voxel; do
  $PY -m contextmap context build --branch $BRANCH "${STACK[@]}" "${NO_MODELS[@]}" "${NO_SCENE[@]}" \
    --workspace $W --maps-root $M --code-identity "$CODE" --json > $E/context-build-${BRANCH#frame-10_31-}.json
  $PY -m contextmap validate $M/$BRANCH/000001 --full --json > $E/validate-$BRANCH.json
done

# 9. Relatórios: semântica por run de percepção e métricas por braço.
python3 $E/scripts/semantic_summary.py $W/corridor-02/run-0006/visual_perception > $E/semantic-summary-v3-frame-55.json
python3 $E/scripts/semantic_summary.py $W/corridor-02/run-0007/visual_perception > $E/semantic-summary-v3-frame-61.json
python3 $E/scripts/run_report.py $W raw=run-0012 voxel=run-0013 > $E/report.json
