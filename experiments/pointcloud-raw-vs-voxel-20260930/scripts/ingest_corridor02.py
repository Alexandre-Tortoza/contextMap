r"""Ingest corridor-02 (LiDAR only) plus its ground-truth pose, for the point-cloud test.

The canonical corridor-02 ingestion used an uncommitted driver with an external calibration and a
timestamp correction (``experiments/e2e-real-ingestion-corridor-02-20260923/README.md``); its
``SequenceArtifact`` was deleted with ``outputs/``, and ``contextmap ingest`` loads no external
calibration, which Geometric Mapping needs (``T_body_source`` comes from the calibration). This
script rebuilds just what a **geometry-only** map needs, from the dataset that is on disk again:

- a main ``SequenceArtifact`` of only ``/velodyne_points`` (no image, no IMU: a raw/voxel point
  cloud needs neither), with a ``CalibrationSet`` whose one static transform places the LiDAR in
  the body frame (``epson -> cmu_rc1_velodyne``, from ``corridor-02-extrinsics.yaml``);
- the auxiliary pose ``SequenceArtifact`` of ``corridor-02-gt.txt`` (``pose_file`` adapter), the
  one State Estimation's pose bridge merges.

Both are ingested on the header clock (``corridor-02-header``); the LiDAR header stamps and the GT
pose timestamps already share that epoch, so no offset correction is needed here (the canonical
correction only mattered for windowing on the bag's *recording* clock). ``--window-s START END``
reads a short recording-time slice for a fast smoke; omit it to ingest the whole bag.

Usage:
    python ingest_corridor02.py --dataset datasets/corridor-02 --output-root <dir> \
        [--window-s 60 90]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from rosbags.rosbag1 import Reader

from contextmap.ingestion import (
    CalibrationSet,
    FrameId,
    RigidTransform,
    SourceAdapter,
    SourceAdapterConfig,
    SourceTopicMapping,
    SourceWindow,
    SynchronizationConfig,
)
from contextmap.ingestion.adapters.pose_file_source_adapter import PoseFileSourceAdapter
from contextmap.ingestion.adapters.ros.ros1_bag_source_adapter import Ros1BagSourceAdapter
from contextmap.runtime import (
    ExecutionEvent,
    IngestionRequest,
    IngestionService,
    ValidationPolicy,
)

HEADER_CLOCK = "corridor-02-header"
"""The one physical robot clock LiDAR and pose share (``e2e-real-ingestion`` README)."""

BODY_FRAME = "epson"
"""The IMU/body frame the GT trajectory is expressed in (``ingest_window.py``)."""

LIDAR_FRAME = "cmu_rc1_velodyne"
"""``header.frame_id`` of every ``/velodyne_points`` message in the bag."""

LIDAR_TOPIC = "/velodyne_points"
SEQUENCE_NAME = "corridor-02"
POSE_SEQUENCE_NAME = "corridor-02-pose"
SYNC_TOLERANCE_NS = 100_000_000  # 100 ms; loose, LiDAR is the only synced modality here.

POSE_FILE_SETTINGS: dict[str, object] = {
    "format": "tum",
    "parent_frame": "map",
    "body_frame": BODY_FRAME,
    "pose_role": "ground_truth",
}
"""What ``corridor-02-gt.txt`` is; the file itself states no frame or role."""

_NS_PER_S = 1_000_000_000


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion ordered ``(x, y, z, w)``.

    ``RigidTransform`` uses ``p_parent = rotate(rotation, p_child) + translation`` with the
    quaternion ordered ``(x, y, z, w)`` (``ingestion/calibration.py``).
    """
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _read_laser_to_imu(extrinsics_yaml: Path) -> np.ndarray:
    """Read the 4x4 ``laser_to_imu`` transform from the dataset's extrinsics YAML.

    ``laser_to_imu`` maps a point in the LiDAR frame to the IMU (body) frame, which is exactly
    ``T_body_source`` for ``body=epson`` and ``source=cmu_rc1_velodyne``.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    document = yaml.load(extrinsics_yaml.read_text(encoding="utf-8"))
    data = document["laser_to_imu"]["data"]
    return np.array(data, dtype=np.float64).reshape(4, 4)


def _calibration(extrinsics_yaml: Path) -> CalibrationSet:
    """Build the minimal geometry calibration: the LiDAR->body static transform."""
    matrix = _read_laser_to_imu(extrinsics_yaml)
    rotation = _matrix_to_quaternion_xyzw(matrix[:3, :3])
    translation = tuple(float(value) for value in matrix[:3, 3])
    transform = RigidTransform(
        parent_frame=FrameId(BODY_FRAME),
        child_frame=FrameId(LIDAR_FRAME),
        translation=translation,  # type: ignore[arg-type]
        rotation=rotation,
    )
    return CalibrationSet(entries={}, static_transforms=(transform,))


def _build_adapter(config: SourceAdapterConfig) -> SourceAdapter:
    """Build the adapter of a request: this script is the composition root of its two sources."""
    if config.source_type == "ros1_bag":
        return Ros1BagSourceAdapter(config)
    if config.source_type == "pose_file":
        return PoseFileSourceAdapter(config)
    raise ValueError(f"only ros1_bag and pose_file are ingested, not {config.source_type!r}")


class _PhasePrinter:
    """Print each ingestion phase on stderr, as ``contextmap ingest`` does."""

    def __init__(self, name: str) -> None:
        self._name = name

    def emit(self, event: ExecutionEvent) -> None:
        """Print the event kind, skipping the periodic progress."""
        if event.kind != "ingestion.progress":
            print(f"{self._name}: {event.kind}", file=sys.stderr)


def _bag_window(bag: Path, topics: SourceTopicMapping, window_s: tuple[float, float] | None):
    """Resolve the recording-time window to read, or ``None`` for the whole bag."""
    if window_s is None:
        return None
    clock = SourceAdapterConfig(
        source_type="ros1_bag", path=str(bag), topics=topics
    ).resolved_window_clock_id()
    with Reader(bag) as reader:
        first_s = int(reader.start_time) / _NS_PER_S
    return SourceWindow(
        clock_id=clock, start_seconds=first_s + window_s[0], end_seconds=first_s + window_s[1]
    )


def _ingest(service: IngestionService, request: IngestionRequest) -> None:
    """Preflight then run one ingestion; exit non-zero on any problem or failure."""
    preflight = service.preflight(request)
    if not preflight.ok:
        for problem in preflight.problems:
            print(f"{request.sequence_name}: {problem}", file=sys.stderr)
        raise SystemExit(1)
    result = service.run(request, event_sink=_PhasePrinter(request.sequence_name))
    if result.failure is not None or result.status != "completed":
        print(f"{request.sequence_name} {result.status}: {result.failure}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Ingest the LiDAR sequence and the pose sequence; print their locations."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="datasets/corridor-02")
    parser.add_argument("--output-root", type=Path, required=True, help="where the sequences go")
    parser.add_argument(
        "--window-s",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="recording-time slice, seconds from the first message; omit for the whole bag",
    )
    options = parser.parse_args(argv)

    dataset: Path = options.dataset
    bag = dataset / "corridor-02" / "corridor02.bag"
    pose_file = dataset / "corridor-02-gt.txt"
    extrinsics = dataset / "corridor-02-extrinsics.yaml"
    for path in (bag, pose_file, extrinsics):
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")

    calibration = _calibration(extrinsics)
    lidar_topics = SourceTopicMapping(lidar=LIDAR_TOPIC)
    window = _bag_window(bag, lidar_topics, tuple(options.window_s) if options.window_s else None)

    sequence_id, pose_id = uuid4().hex, uuid4().hex
    sequence_dir = options.output_root / "sequences" / SEQUENCE_NAME / sequence_id
    pose_dir = options.output_root / "sequences" / POSE_SEQUENCE_NAME / pose_id

    bag_request = IngestionRequest(
        source_type="ros1_bag",
        source_path=str(bag),
        sequence_name=SEQUENCE_NAME,
        output_dir=str(sequence_dir),
        topics=lidar_topics,
        synchronization=SynchronizationConfig(
            reference_modality="lidar", tolerance_nanoseconds=SYNC_TOLERANCE_NS
        ),
        required_topics=frozenset({"lidar"}),
        timestamp_clock_id=HEADER_CLOCK,
        window=window,
        calibration=calibration,
        validation=ValidationPolicy(on_problems="warn"),
        hash_source=False,  # 24 GB bag; the smoke and the map do not need the source digest.
        artifact_id=sequence_id,
        extra={},
    )
    pose_request = IngestionRequest(
        source_type="pose_file",
        source_path=str(pose_file),
        sequence_name=POSE_SEQUENCE_NAME,
        output_dir=str(pose_dir),
        topics=SourceTopicMapping(pose=pose_file.name),
        synchronization=SynchronizationConfig(
            reference_modality="external_pose", tolerance_nanoseconds=SYNC_TOLERANCE_NS
        ),
        required_topics=frozenset({"pose"}),
        timestamp_clock_id=HEADER_CLOCK,
        calibration=calibration,
        validation=ValidationPolicy(on_problems="warn"),
        hash_source=True,
        artifact_id=pose_id,
        extra=POSE_FILE_SETTINGS,
    )

    service = IngestionService(_build_adapter)
    _ingest(service, bag_request)
    _ingest(service, pose_request)

    print("SEQUENCE_DIR=" + str(sequence_dir.resolve()))
    print("POSE_DIR=" + str(pose_dir.resolve()))
    print("SEQUENCE_ID=" + sequence_id)
    print("POSE_ID=" + pose_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
