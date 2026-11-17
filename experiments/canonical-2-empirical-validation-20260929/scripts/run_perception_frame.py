"""Run visual_perception over ONE image frame, in this process (canonical/2 validation, #688 item 5).

One frame per process is the #689 workaround: multi-frame perception degrades after the first
frame on the 8 GB device, while each frame succeeds in isolation. `run_arm.py` calls this once
per selected frame.

The perception backends (SAM2/Qwen providers, CLIP/DINOv2 self-loading) come entirely from the
resolved configuration's `resources.providers`; nothing about the machine is hardcoded here.

Usage:
  run_perception_frame.py --config base.toml [--config arm.toml] --workspace DIR
      --sequence-id ID --frame-index N [--stride 6] [--count 1] --out SUBDIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contextmap.ingestion import (
    SequenceArtifactReader,
    SourceObservationId,
    encode_selection,
)
from contextmap.ingestion.sequence_observation_selection import ExplicitIdsSelection
from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
from contextmap.runtime.composition import compose_executors
from contextmap.visual_perception import PerceptionRunReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", action="append", required=True, help="config layer(s), lowest first"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out", required=True, help="output subdir under <workspace>/pipeline")
    args = parser.parse_args()

    effective = resolve_effective_config(files=list(args.config))
    executors = compose_executors(
        effective,
        module_available=lambda _m: True,
        environ={},
        code_version="canonical-2-validation",
    )
    executor = executors["visual_perception"]

    seq_location = f"sequences/corridor-02/{args.sequence_id}"
    image_ids = [
        e.observation_id
        for e in SequenceArtifactReader(args.workspace / seq_location).iter_index()
        if e.modality == "image"
    ]
    start = args.frame_index
    chosen = image_ids[start : start + args.stride * args.count : args.stride][: args.count]
    print("PRESET", effective.config.pipeline.preset)
    print("SELECTED_IMAGES", chosen)

    out = args.workspace / "pipeline" / args.out
    ref = executor.execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={
                "sequence": (
                    ArtifactRef(
                        stage_id="ingestion",
                        contract="SequenceArtifact",
                        artifact_id=args.sequence_id,
                        content_hash="sha256:placeholder",
                        location=seq_location,
                    ),
                )
            },
            components={},
            config_digest=f"canonical-2-validation-{args.out}",
            output_dir=out,
            workspace=args.workspace,
            observation_selection=encode_selection(
                ExplicitIdsSelection(
                    observation_ids=frozenset(SourceObservationId(i) for i in chosen)
                )
            ),
        )
    )
    reader = PerceptionRunReader(out)
    print("PERCEPTION_ARTIFACT_ID", ref.artifact_id)
    print("verify_integrity", reader.verify_integrity())
    for r in reader.iter_results():
        scene = 0 if r.scene_context is None else len(r.scene_context.claims)
        print(
            f"frame {r.source_observation_id}: regions={len(r.regions)} "
            f"features={len(r.features)} region_claims={len(r.claims)} scene_claims={scene}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
