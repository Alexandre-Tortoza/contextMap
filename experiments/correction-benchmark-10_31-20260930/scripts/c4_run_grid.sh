#!/usr/bin/env bash
# Roda os braços SAM2 do C4 em sequência, um processo por braço (GPU de 8 GB; #689).
# E0-sam2 (0/0) reproduz o E0; E2 = defaults do SAM2 (0,8/0,95); E3-g = grade do plano.
# A célula E3-g 0,80×0,95 repete de propósito a configuração do E2: serve de checagem de
# determinismo (mesmo digest -> mesmas máscaras).
# Uso (da raiz do repositório): bash experiments/correction-benchmark-10_31-20260930/scripts/c4_run_grid.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-$HOME/.cache/contextmap2-ablation/venv/bin/python}"
PROV="${PROV:-$HOME/.cache/contextmap2-ablation/providers}"
ARM="experiments/correction-benchmark-10_31-20260930/scripts/c4_sam2_arm.py"
export PYTHONPATH="src:$PROV" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm() {
  local id="$1" iou="$2" stab="$3"
  if [[ -f "outputs/correction-benchmark-10_31-20260930/c4/$id/arm.json" ]]; then
    echo "skip $id (já existe)"
    return
  fi
  echo "== $id iou=$iou stab=$stab"
  "$PY" "$ARM" --arm-id "$id" --pred-iou "$iou" --stability "$stab" --force 2>&1 | grep -v -i warning | tail -2
}

run_arm "E0-sam2-iou0.00-stab0.00" 0.0 0.0
run_arm "E2-iou0.80-stab0.95" 0.8 0.95
for iou in 0.50 0.60 0.70 0.80; do
  for stab in 0.80 0.90 0.95; do
    run_arm "E3-g-iou${iou}-stab${stab}" "$iou" "$stab"
  done
done
echo "grid done"
