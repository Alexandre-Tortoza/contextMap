# canonical/2 empirical validation — report

Real corridor-02 run (RTX 3060, 8 GB) at commit `6788b83`, validating the canonical/2 pipeline and
the PR #688 round-1 fixes with a controlled `dense_region_pooling` ON/OFF ablation. Fresh
calibration identity (the canonical SequenceArtifact was absent); see `calibration/`. Machine
identities and full lineage are in `manifests/experiment.json`; numbers in `metrics.json`.

## Measured

- **Region discovery identical across arms**: 357 SAM2 regions over the 4 frames, ON and OFF alike
  (84–94 per frame).
- **DINO pooling is the only added evidence**: pooling ON adds **357 DINO pooled region features**
  (one per region); OFF adds **0**.
- **DINO pooling coverage = 1.0** for all 357 pooled features.
- **CLIP semantic scoring** (ON, frame 0): **66 raw CLIP cosines**, distribution min 0.157 /
  p50 0.258 / max 0.355, kept as a raw score in [-1, 1] (never a probability).
- **0 CLIP/DINO scorer collisions** (#1): the CLIP scorer scored the region claims while 88 DINO
  pooled REGION features were present in the same run; **0 unscored region claims**.
- **Semantic scoring lineage preserved** (#2/#3): the fusion `semantic_scores` channel carries the
  CLIP region scorer identity, and `fusion.lineage.semantic_scoring_run_ids` records the scoring run;
  the fusion writer's self-consistency check passed.
- **2D→3D survival**: 91 regions → 91 spatial observations (frame 0).
- **Fusion**: 46 supports / 62 contributions; `visual_features` carries BOTH the CLIP and DINO
  embedding spaces (DINO appearance evidence survives to fusion).

## Observed limitation

- **Multi-frame perception degrades on the RTX 3060 (8 GB)**: in a single run only the first frame
  yields regions; later frames return 0 (no error). Each frame succeeds in an isolated process.
- **Process isolation used as a workaround**: perception was run one process per frame. This is an
  experimental condition, not a defect in the results.
- **Tracked by issue #689** (hypothesis: frame-major executor keeps SAM2+DINO+CLIP+Qwen
  GPU-resident, no unload between stages).

## Not yet measured

- **Regions without valid 3D association** — the complete metric (only the frame-0 count of
  91 regions → 91 spatial observations → 46 supports is recorded so far).
- **Intra-entity vs inter-entity DINO separation** — needs entity resolution, and appearance-based
  association is off pending calibration (canonical/2 doc). Deferred.

## Note on the 266 vs 267 claim difference

Qwen produced 266 region claims (ON) vs 267 (OFF). This is **Qwen non-determinism, tracked in
#556**, not an effect of `dense_region_pooling`: the pooling stage runs after region interpretation
and does not change the claims. It is recorded here as an observation only.

## Baseline for #689

This process-isolated result is a useful baseline: once #689 moves execution to stage-major, that
run can be checked for semantic equivalence to this workaround without re-asking the scientific
question of the ablation.
