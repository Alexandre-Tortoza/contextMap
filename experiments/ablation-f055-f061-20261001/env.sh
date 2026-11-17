# Ambiente comum dos braços (source a partir da raiz do repositório).
E=experiments/ablation-f055-f061-20261001
P=experiments/run-frame-10_31-cli-20260930            # run anterior: run.toml, catálogos de percepção
W=workspace/run-frame-10_31                            # run store (fundação voxel = run-0003)
M=workspace                                            # maps root: workspace/<map_id>/<NNNNNN>
PY=~/.cache/contextmap2-ablation/venv/bin/python
export PYTHONPATH="src:$HOME/.cache/contextmap2-ablation/providers"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CODE=$(cat $E/code-identity.txt)
STACK=(
  -c examples/v0.2.0/canonical-config.toml
  -c experiments/canonical-2-empirical-validation-20260929/config/base.template.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml
  -c examples/v0.2.0/corridor-02-image-masks.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml
  -c $P/run.toml
)
NO_SCENE=(--set 'components.visual_perception.semantic_interpretation.qwen.prompt_policy.scene=null')
NO_MODELS=(--set 'resources.providers={}')
select_frame() {  # observation_selection de um frame
  echo "inputs.observation_selection={\"kind\":\"explicit_ids\",\"observation_ids\":[\"camera_1_image_raw-$1\"]}"
}
# Pico de memória da GPU de um processo: amostra o nvidia-smi a cada 200 ms enquanto o comando roda.
with_gpu_peak() {  # with_gpu_peak <arquivo-de-saída> <comando...>
  local out=$1; shift
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$out.gpu" &
  local sampler=$!
  local start=$(date +%s.%N)
  "$@"; local code=$?
  local end=$(date +%s.%N)
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  python3 -c "import sys;v=[int(x) for x in open('$out.gpu') if x.strip()];print('{\"wall_s\": %.2f, \"gpu_peak_mib\": %d, \"gpu_baseline_mib\": %d, \"exit\": %d}'%($end-$start, max(v) if v else -1, min(v) if v else -1, $code))" > "$out.cost.json"
  rm -f "$out.gpu"
  return $code
}
