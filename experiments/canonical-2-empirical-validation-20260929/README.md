# canonical/2 empirical validation (2026-09-29, PR #688 item 5)

A reexecutable protocol that validates the `canonical/2` pipeline and the PR #688 round-1 fixes
(#1 embedding-space scorer, #2/#3 fusion provenance) on real corridor-02 data, with a controlled
`dense_region_pooling` ON/OFF ablation. It also records a real runtime limitation (issue #689) as
an experimental condition, keeping the scientific result and the operational limitation separate.

- **`REPORT.md`** — results, split into Measured / Observed limitation / Not yet measured.
- **`metrics.json`** — the numbers (`scripts/collect_metrics.py` output).
- **`manifests/experiment.json`** — the single verifiable source: commit, dataset, window, frame
  selection, arms, models/revisions, executed chain, and the run identities that prove lineage.
- **`manifests/frames.json`**, **`manifests/environment.json`** — the 4 selected frames; the machine.
- **`calibration/`** — how the calibration was rebuilt from the dataset YAMLs, and its identity.
- **`config/`** — `base.template.toml` (identities only; no paths, no weights) + the two arm layers.
- **`scripts/`** — the drivers.

## What it does not contain

No model weights, no machine-local paths, no provider module. The config declares model
**identities** (checkpoint + revision); the physical checkpoint/HF-cache/dataset locations and the
SAM2/Qwen `RuntimeProvider` module are the environment's responsibility. Fill them with `--set`
overrides or a private config layer; `scripts/preflight.py` refuses leftover placeholders.

## Conditions (from `manifests/experiment.json`)

- Commit `6788b830d2b9890be8c57de287dd37da23b83158`, branch `milestone/canonical-2`.
- Dataset corridor-02; window 2:30–2:35 (±5 s padding); 4 frames every 6th from the window start.
- Two arms differing ONLY by `dense_region_pooling`: ON (canonical/2) vs OFF (canonical/1).
- Models: SAM2.1-hiera-tiny, DINOv2-base, CLIP-ViT-L/14, Qwen3-VL-4B (4-bit); device cuda.
- **#689 workaround**: perception is run one process per frame (multi-frame degrades on 8 GB).

## Reproduce

Prerequisites the environment supplies (not committed): the dataset, the model checkpoints in the
HF cache, and a `RuntimeProvider` module exposing `load_sam2`/`load_qwen`. Set these as overrides:

```bash
BUNDLE=experiments/canonical-2-empirical-validation-20260929
COMMON=(--config $BUNDLE/config/base.template.toml \
  --set resources.workspace=<WS> \
  --set inputs.sequence=<SEQ_ID> \
  --set 'resources.providers={"visual_perception.region_discovery":"<mod>:load_sam2","visual_perception.semantic_interpretation":"<mod>:load_qwen"}')

# 0. Ingest the window with the rebuilt calibration (produces <SEQ_ID> and the pose sequence).
python $BUNDLE/scripts/ingest_window_fresh.py --bag <BAG> --pose-file <POSE> --outputs-root <WS>

# (state_estimation + geometric_mapping: run the executors as in the canonical run, into <WS>/pipeline.)

# 1. Preflight (refuses placeholders / local paths / divergent identities).
python $BUNDLE/scripts/preflight.py "${COMMON[@]}" --config $BUNDLE/config/pooling-on.toml

# 2. Ablation arms (per-frame, isolated processes — the #689 workaround).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $BUNDLE/scripts/run_arm.py \
  "${COMMON[@]}" --config $BUNDLE/config/pooling-on.toml  --sequence-id <SEQ_ID> --out-prefix pool_on
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $BUNDLE/scripts/run_arm.py \
  "${COMMON[@]}" --config $BUNDLE/config/pooling-off.toml --sequence-id <SEQ_ID> --out-prefix pool_off

# 3. Downstream on one ON frame (scoring -> association -> fusion): exercises #1/#2/#3 on real data.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $BUNDLE/scripts/run_downstream.py \
  "${COMMON[@]}" --config $BUNDLE/config/pooling-on.toml --sequence-id <SEQ_ID> \
  --perception-subdir pool_on_f0 --state-id <STATE_ID> --geo-id <GEO_ID> --out-prefix f0on

# 4. Collect metrics.
python $BUNDLE/scripts/collect_metrics.py <WS> $BUNDLE/metrics.json
```

The run identities in `manifests/experiment.json` are from this machine's execution and prove the
lineage of the artifacts the committed `metrics.json` was read from; they are not re-derivable
without the dataset. A fresh reproduction produces new identities but should reproduce the ablation.
