r"""Run state_estimation (external_pose) over a LiDAR sequence and its GT pose sequence.

The map's trajectory comes from the ground-truth pose (issue #555's auxiliary pose bridge):
State Estimation merges the pose sequence into the LiDAR sequence's timeline and adopts it as the
trajectory (``policies.trajectory_mode = "allow_ground_truth"``). This drives the composed
``StateEstimationExecutor`` directly, so the two sequences this script's sibling
``ingest_corridor02.py`` produced can be consumed without a workspace journal or a catalog file.

Usage:
    python run_trajectory.py --sequence-dir DIR --pose-dir DIR --workspace DIR \
        -c config.toml [--profile canonical/2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contextmap.ingestion import SequenceArtifactReader
from contextmap.runtime import compose, resolve_effective_config
from contextmap.runtime.artifacts import ArtifactRef, inventory_digest
from contextmap.runtime.components.contracts import SEQUENCE
from contextmap.runtime.executors import StateEstimationExecutor
from contextmap.runtime.pipeline import StageRequest


def _sequence_ref(stage_id: str, sequence_dir: Path, workspace: Path) -> ArtifactRef:
    """Build the input handle for a sequence artifact, located relative to the workspace."""
    reader = SequenceArtifactReader(sequence_dir)
    location = sequence_dir.resolve().relative_to(workspace.resolve()).as_posix()
    return ArtifactRef(
        stage_id=stage_id,
        contract=SEQUENCE,
        artifact_id=str(reader.manifest.artifact_id),
        content_hash=inventory_digest(reader.manifest.file_inventory),
        location=location,
    )


def main(argv: list[str] | None = None) -> int:
    """Estimate the trajectory and print its artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sequence-dir", type=Path, required=True, help="LiDAR SequenceArtifact")
    parser.add_argument("--pose-dir", type=Path, required=True, help="pose SequenceArtifact")
    parser.add_argument(
        "--workspace", type=Path, required=True, help="root the sequences and trajectory live under"
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    options = parser.parse_args(argv)

    profile_kwargs = {"profile": options.profile} if options.profile else {}
    effective = resolve_effective_config(
        files=options.config, overrides=options.set, **profile_kwargs
    )
    composed = compose(effective, stages=["state_estimation"])
    if composed.state_estimator is None:
        raise SystemExit("state_estimation did not compose an estimator; check the config")
    allow_ground_truth = effective.config.policies.trajectory_mode == "allow_ground_truth"

    workspace: Path = options.workspace
    output_dir = workspace / "state_estimation"
    request = StageRequest(
        stage_id="state_estimation",
        inputs={
            "sequence": (_sequence_ref("ingestion", options.sequence_dir, workspace),),
            "pose_sequence": (_sequence_ref("pose_ingestion", options.pose_dir, workspace),),
        },
        components={},
        config_digest="pointcloud-raw-vs-voxel",
        output_dir=output_dir,
        workspace=workspace,
    )
    executor = StateEstimationExecutor(
        composed.state_estimator, allow_ground_truth_trajectory=allow_ground_truth
    )
    reference = executor.execute(request)
    print("TRAJECTORY_DIR=" + str(output_dir.resolve()))
    print("TRAJECTORY_ID=" + reference.artifact_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
