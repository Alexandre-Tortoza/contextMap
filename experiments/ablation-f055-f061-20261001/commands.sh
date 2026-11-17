#!/usr/bin/env bash
# Ablação f055/f061 (corridor-02, 10:31): os comandos exatos, na ordem em que rodaram.
# Executar da raiz do repositório, com o venv de ablação e os providers locais (env.sh).
set -euo pipefail
source experiments/ablation-f055-f061-20261001/env.sh

# Branch por map_id, sempre sobre a fundação voxel 0,10 m (run-0003), e um processo por frame.
branch() { $PY -m contextmap context branch create "$1" --from-run run-0003 "${STACK[@]}" "${@:2}" \
  "${NO_MODELS[@]}" --workspace $W --maps-root $M --json > $E/logs/branch-$1.json; }
perceive() {  # perceive <map_id> <frame> <camadas...>: percepção + associação novas
  with_gpu_peak $E/logs/$1-$2 $PY -m contextmap context run --branch "$1" "${STACK[@]}" "${@:3}" \
    "${NO_SCENE[@]}" --set "$(select_frame $2)" --workspace $W --maps-root $M --code-identity "$CODE" \
    --json > $E/logs/context-run-$1-$2.json
}
reuse() {  # reuse <map_id> <frame> <camadas...>: seleciona a percepção da run anterior, só associa
  local perc; perc=$(python3 -c "import json;print(json.load(open('$P/catalog-perception-$2.json'))['entries'][0]['artifact_id'])")
  with_gpu_peak $E/logs/$1-$2 $PY -m contextmap context run --branch "$1" "${STACK[@]}" "${@:3}" \
    "${NO_MODELS[@]}" "${NO_SCENE[@]}" --set "$(select_frame $2)" --catalog $P/catalog-perception-$2.json \
    --select visual_perception=$perc --workspace $W --maps-root $M --code-identity "$CODE" \
    --json > $E/logs/context-run-$1-$2.json
}
build() {
  with_gpu_peak $E/logs/$1-build $PY -m contextmap context build --branch "$1" "${STACK[@]}" "${@:2}" \
    "${NO_MODELS[@]}" "${NO_SCENE[@]}" --workspace $W --maps-root $M --code-identity "$CODE" \
    --json > $E/logs/context-build-$1.json
  $PY -m contextmap validate $M/$1/000001 --full --json > $E/logs/validate-$1.json
}
arm() {  # arm <map_id> <perceive|reuse> <camadas...>
  local map=$1 mode=$2; shift 2
  branch "$map" "$@"
  for frame in 000055 000061; do $mode "$map" $frame "$@"; done
  build "$map" "$@"
}
OUTLINE_OFF=(--set 'components.visual_perception.semantic_interpretation.qwen.view_policy.mask_outline=null')

# B0: SAM2 + Qwen v3 (mask_outline_crop), percepção das runs 0006/0007 reusada.
arm corridor02-f055-f061-sam2-qwen-baseline-vox10 reuse
# A1: só o piso de área 1536 px na normalização do SAM2.
arm corridor02-f055-f061-sam2-qwen-minarea1536-vox10 perceive -c $E/config/minarea1536.toml
# F1-od / F1-rp: Florence-2 <OD> / <REGION_PROPOSAL> + Florence-2 semântico + regra por caixa.
arm corridor02-f055-f061-florenceod-florencesem-boxdepth-vox10 perceive \
  -c $E/config/florence-od.toml -c $E/config/florence-semantic.toml -c $E/config/box-membership.toml
arm corridor02-f055-f061-florencerp-florencesem-boxdepth-vox10 perceive \
  -c $E/config/florence-rp.toml -c $E/config/florence-semantic.toml -c $E/config/box-membership.toml
# F1q: as mesmas caixas <REGION_PROPOSAL>, intérprete Qwen v3 com tight_crop.
arm corridor02-f055-f061-florencerp-qwen-boxdepth-vox10 perceive \
  -c $E/config/florence-rp.toml -c $E/config/qwen-tight.toml -c $E/config/box-membership.toml "${OUTLINE_OFF[@]}"
# S1: regiões SAM2 do B0 (determinístico), Qwen v3 com tight_crop em vez de mask_outline_crop.
arm corridor02-f055-f061-sam2-qwentight-vox10 perceive -c $E/config/qwen-tight.toml "${OUTLINE_OFF[@]}"
# R1: B0 com os predicados topológicos/de suporte já implementados (inside, on_top_of, above).
arm corridor02-f055-f061-sam2-qwen-baseline-relinside-vox10 reuse -c $E/config/relations-topology.toml

# Human Search: SAM3 text_prompt, só percepção (sem mapa), e o controle positivo com door/window.
for frame in 000055 000061; do
  with_gpu_peak $E/logs/human-search-$frame $PY -m contextmap run "${STACK[@]}" -c $E/config/human-search.toml \
    -c $E/config/florence-semantic.toml --workspace $W --catalog $P/catalog-foundation.json \
    --select ingestion=4a4d895e0e4e4731ad91326114607296 --stage visual_perception \
    --set "$(select_frame $frame)" --code-identity "$CODE" --json > $E/logs/human-search-$frame.json
done
with_gpu_peak $E/logs/human-search-control-000055 $PY -m contextmap run "${STACK[@]}" \
  -c $E/config/human-search.toml -c $E/config/florence-semantic.toml \
  --set 'components.visual_perception.region_discovery.sam3.prompts=["door","window"]' \
  --workspace $W --catalog $P/catalog-foundation.json --select ingestion=4a4d895e0e4e4731ad91326114607296 \
  --stage visual_perception --set "$(select_frame 000055)" --code-identity "$CODE" \
  --json > $E/logs/human-search-control-000055.json

# Métricas (anotação: rascunho de modelo, pendente de revisão humana) e inspeção no Viewer.
$PY $E/scripts/arm_metrics.py $W corridor02-f055-f061-sam2-qwen-baseline-vox10 \
  corridor02-f055-f061-sam2-qwentight-vox10 corridor02-f055-f061-sam2-qwen-minarea1536-vox10 \
  corridor02-f055-f061-florenceod-florencesem-boxdepth-vox10 corridor02-f055-f061-florencerp-florencesem-boxdepth-vox10 \
  corridor02-f055-f061-florencerp-qwen-boxdepth-vox10 corridor02-f055-f061-sam2-qwen-baseline-relinside-vox10 \
  > $E/results/metrics.json
# Viewer (outro terminal): cd ~/Projects/contextmap2-viewer && HOST=127.0.0.1 PORT=3001 npx nuxt preview --dotenv .env.ablation-f055-f061
node $E/scripts/viewer_check.mjs http://127.0.0.1:3001/ $E/results/viewer \
  corridor02-f055-f061-sam2-qwen-baseline-vox10 corridor02-f055-f061-sam2-qwen-minarea1536-vox10 \
  corridor02-f055-f061-florencerp-qwen-boxdepth-vox10 corridor02-f055-f061-sam2-qwentight-vox10
