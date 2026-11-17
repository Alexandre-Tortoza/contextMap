"""Fresh corridor-02 windowed ingestion with a calibration rebuilt from the dataset YAMLs.

The canonical SequenceArtifact (720a486d…) was absent on this machine, so the reproducible
`experiments/corridor-02-window-ingestion-20260927` command (which reads calibration/config/anchors
from it) cannot run. This driver rebuilds the same geometry with a NEW calibration identity and
reuses that command's tested window/correction/request machinery. Calibration decisions are in
../calibration/ (README.md + manifest.json). MEI intrinsics and extrinsics below are the exact
dataset YAML values (PyYAML is not assumed present); the manifest records their sha256.

Usage:
  ingest_window_fresh.py --bag PATH --pose-file PATH --outputs-root DIR
      [--start-offset-s 150] [--end-offset-s 155] [--padding-s 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments/corridor-02-window-ingestion-20260927/scripts"))

import ingest_window as iw  # noqa: E402

from contextmap.ingestion import (  # noqa: E402
    CalibrationSet,
    ConstantOffsetCorrection,
    SourceAdapterConfig,
    SourceTopicMapping,
    SynchronizationConfig,
    TimestampPolicy,
)
from contextmap.ingestion.calibration import (  # noqa: E402
    CalibrationEntry,
    CalibrationProvenance,
    MeiCameraModel,
    RigidTransform,
    compute_content_hash,
)
from contextmap.ingestion.canonical_observations import (  # noqa: E402
    CalibrationReferenceId,
    FrameId,
    SensorId,
)
from contextmap.ingestion.source_timestamp_normalization import (  # noqa: E402
    EVENT_CLOCK_HEADER_STAMP,
    WINDOW_CLOCK_RECORDING_TIME,
)
from contextmap.runtime import IngestionService, ValidationPolicy  # noqa: E402
from contextmap.shared import SourceTimestamp  # noqa: E402

CLOCK = "corridor-02-header"
TOPICS = SourceTopicMapping(rgb="/camera_1/image_raw", lidar="/velodyne_points", imu="/imu/data")
CAM, LIDAR, IMU = SensorId("camera_1_image_raw"), SensorId("velodyne_points"), SensorId("imu_data")
CAM_FRAME, LIDAR_FRAME, BODY = FrameId("d"), FrameId("cmu_rc1_velodyne"), FrameId("epson")

# Exact values from datasets/corridor-02/*.yaml (hashes recorded in ../calibration/manifest.json).
_MEI = dict(
    width=640,
    height=480,
    gamma1=761.6791126513665,
    gamma2=678.7676595096547,
    u0=317.73997456826834,
    v0=233.50210820810244,
    xi=1.5629934283981157,
    k1=-0.07083665039247518,
    k2=0.3094574364801575,
    p1=0.0004607746627279861,
    p2=0.00015704097191542886,
)
_RGB_CAMERA_TO_IMU = [
    -0.01578308,
    -0.01124397,
    0.99981222,
    0.17313616,
    -0.99975463,
    -0.01536642,
    -0.01595499,
    0.0476788,
    0.01554293,
    -0.99981871,
    -0.01099868,
    -0.01933398,
    0,
    0,
    0,
    1,
]
_LASER_TO_IMU = [
    0.999954875339377,
    0.0,
    0.009499857104811,
    0,
    0.0,
    1.0,
    0.0,
    0,
    -0.009499857104811,
    0.0,
    0.999954875339377,
    0,
    0,
    0,
    0,
    1,
]


def _transform(matrix16: list[float], child: FrameId) -> RigidTransform:
    m = np.array(matrix16, dtype=float).reshape(4, 4)
    q = Rotation.from_matrix(m[:3, :3]).as_quat()  # (x, y, z, w)
    return RigidTransform(
        parent_frame=BODY,
        child_frame=child,
        translation=(float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
        rotation=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def _entry(sensor: SensorId, frame: FrameId, model: MeiCameraModel | None) -> CalibrationEntry:
    calib_id = CalibrationReferenceId(f"{sensor}-calib")
    return CalibrationEntry(
        calibration_id=calib_id,
        sensor_id=sensor,
        frame_id=frame,
        camera_model=model,
        provenance=CalibrationProvenance(
            source_type="dataset",
            source_path="datasets/corridor-02",
            conversions_applied=("rebuilt fresh from YAMLs (#688 item 5)",),
        ),
        content_hash=compute_content_hash(sensor_id=sensor, frame_id=frame, camera_model=model),
    )


def build_calibration() -> CalibrationSet:
    mei = MeiCameraModel(
        width=_MEI["width"],
        height=_MEI["height"],
        fx=_MEI["gamma1"],
        fy=_MEI["gamma2"],
        cx=_MEI["u0"],
        cy=_MEI["v0"],
        xi=_MEI["xi"],
        distortion_coefficients=(_MEI["k1"], _MEI["k2"], _MEI["p1"], _MEI["p2"]),
    )
    entries = {
        e.calibration_id: e
        for e in (
            _entry(CAM, CAM_FRAME, mei),
            _entry(LIDAR, LIDAR_FRAME, None),
            _entry(IMU, BODY, None),
        )
    }
    return CalibrationSet(
        entries=entries,
        static_transforms=(
            _transform(_RGB_CAMERA_TO_IMU, CAM_FRAME),
            _transform(_LASER_TO_IMU, LIDAR_FRAME),
        ),
    )


def build_canonical() -> iw.CanonicalIngestion:
    policy = TimestampPolicy(
        event_clock=EVENT_CLOCK_HEADER_STAMP,
        window_clock=WINDOW_CLOCK_RECORDING_TIME,
        correction=ConstantOffsetCorrection.from_anchors(
            source_time=SourceTimestamp(seconds=1646000057, nanoseconds=997985442, clock_id=CLOCK),
            reference_time=SourceTimestamp(
                seconds=1724621416, nanoseconds=410417577, clock_id="corridor-02-recording-time"
            ),
        ),
    )
    return iw.CanonicalIngestion(
        artifact_id="fresh-yaml-calibration",
        topics=TOPICS,
        required_topics=frozenset({"rgb", "lidar", "imu"}),
        timestamp_clock_id=CLOCK,
        synchronization=SynchronizationConfig(
            reference_modality="image", tolerance_nanoseconds=50_000_000
        ),
        validation=ValidationPolicy(on_problems="warn"),
        hash_source=True,
        timestamp_policy=policy,
        calibration=build_calibration(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--pose-file", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--start-offset-s", type=float, default=150.0)
    parser.add_argument("--end-offset-s", type=float, default=155.0)
    parser.add_argument("--padding-s", type=float, default=5.0)
    args = parser.parse_args()
    args.outputs_root.mkdir(parents=True, exist_ok=True)

    spec = iw.WindowSpec(
        start_offset_s=args.start_offset_s, end_offset_s=args.end_offset_s, padding_s=args.padding_s
    )
    canonical = build_canonical()
    assert canonical.correction.offset_nanoseconds == 78621358412432135, "offset mismatch"
    window = iw.resolve_window(
        spec,
        first_message_ns=iw.bag_first_message_recording_ns(args.bag),
        correction=canonical.correction,
        window_clock_id=SourceAdapterConfig(
            source_type="ros1_bag", path=str(args.bag), topics=TOPICS
        ).resolved_window_clock_id(),
    )

    seq_id, pose_id = uuid4().hex, uuid4().hex
    seq_loc = Path("sequences", iw.SEQUENCE_NAME, seq_id)
    pose_loc = Path("sequences", iw.POSE_SEQUENCE_NAME, pose_id)
    requests = (
        iw.bag_request(
            args.bag,
            canonical,
            window.read_window,
            output_dir=args.outputs_root / seq_loc,
            artifact_id=seq_id,
        ),
        iw.pose_request(
            args.pose_file, canonical, output_dir=args.outputs_root / pose_loc, artifact_id=pose_id
        ),
    )
    service = IngestionService(iw.build_adapter)
    for request in requests:
        pf = service.preflight(request)
        if not pf.ok:
            for problem in pf.problems:
                print(f"PREFLIGHT {request.sequence_name}: {problem}", file=sys.stderr)
            return 1
    for request in requests:
        result = service.run(request, event_sink=iw._PhasePrinter(request.sequence_name))
        if result.failure is not None or result.status != "completed":
            print(f"{request.sequence_name} {result.status}: {result.failure}", file=sys.stderr)
            return 1

    quality = iw.window_quality(
        args.outputs_root / seq_loc,
        args.outputs_root / pose_loc,
        start_ns=window.start_ns,
        end_ns=window.end_ns,
    )
    print("SEQUENCE_ID", seq_id)
    print("POSE_SEQUENCE_ID", pose_id)
    print("QUALITY", quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
