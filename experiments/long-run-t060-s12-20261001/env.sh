# Ambiente da run longa (source a partir da raiz do repositório).
L=experiments/long-run-t060-s12-20261001
P=experiments/run-frame-10_31-cli-20260930          # run.toml, timestamp-policy da run frame-10_31
W=workspace/run-t060-s12                             # run store novo
M=workspace                                          # maps root
PY=~/.cache/contextmap2-ablation/venv/bin/python
export PYTHONPATH="src:$HOME/.cache/contextmap2-ablation/providers"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CODE=$(cat $L/code-identity.txt)
STACK=(
  -c examples/v0.2.0/canonical-config.toml
  -c experiments/canonical-2-empirical-validation-20260929/config/base.template.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml
  -c examples/v0.2.0/corridor-02-image-masks.toml
  -c experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml
  -c $P/run.toml
)
SOTA=(-c $L/config/sota.toml -c $L/config/ptv3.json)
HUMAN=(-c $L/config/human.toml)
NO_SCENE=(--set 'components.visual_perception.semantic_interpretation.qwen.prompt_policy.scene=null')
NO_MODELS=(--set 'resources.providers={}')
CALIBRATION=experiments/correction-benchmark-10_31-20260930/calibration/refined-calibration.json
MAP=corridor02-t060-s12-sam2t-dinov3-qwen8b-top8-ptv3-sam3human-vox10
