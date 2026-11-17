"""Pose-interpolation deskew (C3): a platform in translation and rotation observing a wall.

Every scan is *simulated* from the known truth: point ``i`` of a scan is measured at
``t_ref + tau_i`` (``t_ref`` is the scan's timestamp, the start of the acquisition), so a
mapping that places the whole scan with the pose at ``t_ref`` bends the wall, and a correct
deskew puts every point back on it. The trajectory moves at constant velocity about a fixed
axis, which linear and spherical interpolation reproduce exactly, and the point times are
exact multiples of the time bin, so the only remaining error is the payload's float32.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from contextmap.geometric_mapping import (
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputError,
    GeometryMappingPlan,
    InputRejectionReason,
    MapId,
    MotionCorrectionEvidence,
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
    MotionCorrectionState,
    PointTimeReference,
    PointTimeUnit,
    PoseInterpolationDeskewPolicy,
    ScanDisposition,
    accumulate_plan,
    build_geometry_mapping_plan,
    transform_scans_to_map_frame,
    verify_motion_correction,
)
from contextmap.geometric_mapping.deskew import deskew_configuration_fingerprint
from contextmap.ingestion import (
    CalibrationSet,
    FrameId,
    FullSequenceSelection,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SequenceSelectionResult,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.shared import (
    Quaternion,
    SourceTimestamp,
    Vector3,
    invert_rigid,
    normalize_quaternion,
    rotate_vector,
)
from contextmap.state_estimation import (
    EstimatorProvenance,
    LookupPolicy,
    PoseEstimate,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryId,
    TrajectoryProvenance,
    calibration_identity,
    pose_estimate_id_for,
)

# --- Fixture: the truth and its simulated measurements -------------------------------------

MS = 1_000_000
CLOCK_ID = "fixture:header"
SEQUENCE_ID = SequenceArtifactId("sequence-deskew")
TRAJECTORY_ID = TrajectoryId("run-0001--trajectory")
SCAN_START_NS = 200 * MS
SCAN_DURATION_MS = 100
POINT_COUNT = 400
POSE_PERIOD_MS = 20
TRAJECTORY_END_MS = 600
WALL_X_M = 6.0
NOISE_M = 0.002
VELOCITY_M_S: Vector3 = (2.0, 0.3, 0.0)
ROTATION_AXIS: Vector3 = (0.05, 0.0, 1.0)
ANGULAR_RATE_RAD_S = 0.6


def axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
    """Unit quaternion of a rotation of ``angle_rad`` about ``axis``."""
    norm = math.sqrt(sum(component * component for component in axis))
    sin = math.sin(angle_rad / 2.0)
    return normalize_quaternion(
        (axis[0] / norm * sin, axis[1] / norm * sin, axis[2] / norm * sin, math.cos(angle_rad / 2))
    )


STATIC: tuple[Vector3, Quaternion] = ((0.4, -0.1, 0.8), axis_angle((0.2, 0.1, 1.0), 0.3))
CALIBRATION = CalibrationSet(
    entries={},
    static_transforms=(
        RigidTransform(
            parent_frame=FrameId("body"),
            child_frame=FrameId("lidar"),
            translation=STATIC[0],
            rotation=STATIC[1],
        ),
    ),
)


def true_pose(time_ns: int) -> tuple[Vector3, Quaternion]:
    """``T_map_body`` of the platform: constant velocity and constant rotation rate."""
    seconds = time_ns / 1e9
    translation = (
        VELOCITY_M_S[0] * seconds,
        VELOCITY_M_S[1] * seconds,
        VELOCITY_M_S[2] * seconds,
    )
    return translation, axis_angle(ROTATION_AXIS, ANGULAR_RATE_RAD_S * seconds)


def through(transform: tuple[Vector3, Quaternion], point: Vector3) -> Vector3:
    """Apply ``T_parent_child`` to a point of the child frame."""
    translation, rotation = transform
    rotated = rotate_vector(rotation, point)
    return (
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    )


def seen_from_sensor(point_m: Vector3, time_ns: int) -> Vector3:
    """A map point expressed in the sensor frame of the platform at ``time_ns``."""
    body_pose = true_pose(time_ns)
    to_body = invert_rigid(translation=body_pose[0], rotation=body_pose[1])
    to_sensor = invert_rigid(translation=STATIC[0], rotation=STATIC[1])
    return through(to_sensor, through(to_body, point_m))


def wall_point(index: int) -> Vector3:
    """Point ``index`` of the wall ``x = WALL_X_M``, off the plane by a known alternating noise."""
    return (
        WALL_X_M + NOISE_M * (-1) ** index,
        -2.0 + 4.0 * (index % 20) / 19,
        0.2 + 1.8 * (index // 20) / 19,
    )


def point_offset_ms(index: int, point_count: int = POINT_COUNT) -> int:
    """When point ``index`` is measured after the scan's timestamp, an exact number of ms."""
    return index * SCAN_DURATION_MS // point_count


def observe_wall(
    header_ns: int = SCAN_START_NS, point_count: int = POINT_COUNT
) -> tuple[list[Vector3], list[float]]:
    """The wall as the moving sensor measures it: coordinates and time offsets in seconds."""
    points: list[Vector3] = []
    times_s: list[float] = []
    for index in range(point_count):
        offset_ms = point_offset_ms(index, point_count)
        points.append(seen_from_sensor(wall_point(index), header_ns + offset_ms * MS))
        times_s.append(offset_ms / 1000)
    return points, times_s


_TIME_CODES = {PointFieldDataType.FLOAT32: "f", PointFieldDataType.UINT32: "I"}


def make_scan(
    points: Sequence[Vector3],
    times: Sequence[float],
    *,
    observation_id: str = "scan-0000",
    header_ns: int = SCAN_START_NS,
    time_field: str = "time",
    time_type: PointFieldDataType = PointFieldDataType.FLOAT32,
) -> LidarObservation:
    """A float32 scan with the packed layout of a spinning LiDAR, time last and unaligned."""
    fields = (
        PointFieldDescriptor(name="x", offset_bytes=0, data_type=PointFieldDataType.FLOAT32),
        PointFieldDescriptor(name="y", offset_bytes=4, data_type=PointFieldDataType.FLOAT32),
        PointFieldDescriptor(name="z", offset_bytes=8, data_type=PointFieldDataType.FLOAT32),
        PointFieldDescriptor(
            name="intensity", offset_bytes=12, data_type=PointFieldDataType.FLOAT32
        ),
        PointFieldDescriptor(name="ring", offset_bytes=16, data_type=PointFieldDataType.UINT16),
        PointFieldDescriptor(name=time_field, offset_bytes=18, data_type=time_type),
    )
    record = f"<4fH{_TIME_CODES[time_type]}"
    data = b"".join(
        struct.pack(record, x, y, z, 10.0, index % 16, time)
        for index, ((x, y, z), time) in enumerate(zip(points, times, strict=True))
    )
    seconds, nanoseconds = divmod(header_ns, 1_000_000_000)
    return LidarObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("lidar0"),
        frame_id=FrameId("lidar"),
        timestamp=SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/wall"),
        point_count=len(points),
        point_step_bytes=22,
        fields=fields,
        data=data,
    )


def wall_scan() -> LidarObservation:
    """The default scan: the wall seen during 100 ms of motion."""
    points, times = observe_wall()
    return make_scan(points, times)


def make_trajectory(*, missing_ms: range = range(0)) -> Trajectory:
    """The true trajectory sampled every 20 ms, without the samples in ``missing_ms``."""
    poses: list[PoseEstimate] = []
    for time_ms in range(0, TRAJECTORY_END_MS + 1, POSE_PERIOD_MS):
        if time_ms in missing_ms:
            continue
        translation, rotation = true_pose(time_ms * MS)
        seconds, nanoseconds = divmod(time_ms * MS, 1_000_000_000)
        poses.append(
            PoseEstimate(
                estimate_id=pose_estimate_id_for(trajectory_id=TRAJECTORY_ID, index=len(poses)),
                timestamp=SourceTimestamp(
                    seconds=seconds, nanoseconds=nanoseconds, clock_id=CLOCK_ID
                ),
                parent_frame=FrameId("map"),
                child_frame=FrameId("body"),
                translation_m=translation,
                orientation=rotation,
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(
                    source_observation_ids=(SourceObservationId(f"pose-{time_ms}"),)
                ),
            )
        )
    return Trajectory(
        trajectory_id=TRAJECTORY_ID,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=tuple(poses),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fake", backend_version="0"),
            sequence_artifact_id=SEQUENCE_ID,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(CALIBRATION),
            code_version="test",
        ),
    )


def selection_of(scans: Sequence[LidarObservation]) -> SequenceSelectionResult:
    """The scans as the resolved full selection of the fixture sequence."""
    selection = FullSequenceSelection()
    return SequenceSelectionResult(
        sequence_artifact_id=SEQUENCE_ID,
        selection=selection,
        selection_id=selection_identity(SEQUENCE_ID, selection),
        observations=tuple(scans),
    )


INTERPOLATED = LookupPolicy.interpolated(max_interpolation_gap_ns=100 * MS)
ACCEPT_ALL = MotionCorrectionPolicy(raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT)

# sha256 do geometry.bin do mapa sem deskew desta cena, medido com o código de antes do deskew
# existir: sem a política selecionada, a geometria persistida não pode mudar um byte.
GEOMETRY_WITHOUT_DESKEW_SHA256 = "525ade1bb1c1b26cd3a0f61c3b9ff3e6fb89e54a68749ebadc5726894fc7bf82"

# --- Helpers of the tests ------------------------------------------------------------------

POLICY = PoseInterpolationDeskewPolicy(
    time_field="time",
    time_unit=PointTimeUnit.SECONDS,
    reference=PointTimeReference.SCAN_TIMESTAMP,
    time_bin_ns=1 * MS,
)


def assemble(
    scans: Sequence[LidarObservation],
    *,
    deskew: PoseInterpolationDeskewPolicy | None = POLICY,
    trajectory: Trajectory | None = None,
    pose_lookup: LookupPolicy = INTERPOLATED,
    policy: MotionCorrectionPolicy = ACCEPT_ALL,
    motion_correction: dict[SourceObservationId, MotionCorrectionRecord] | None = None,
) -> GeometryMappingPlan:
    return build_geometry_mapping_plan(
        sequence=selection_of(scans),
        calibration=CALIBRATION,
        trajectory=trajectory if trajectory is not None else make_trajectory(),
        pose_lookup=pose_lookup,
        motion_correction_policy=policy,
        motion_correction=motion_correction,
        deskew=deskew,
    )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _map_points(plan: GeometryMappingPlan) -> np.ndarray:
    (scan,) = transform_scans_to_map_frame(plan)
    return np.frombuffer(scan.map_coordinates_m, dtype=np.float64).reshape(-1, 3)


def _sensor_points(observation: LidarObservation) -> np.ndarray:
    record = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8],
            "itemsize": observation.point_step_bytes,
        }
    )
    points = np.frombuffer(observation.data, dtype=record, count=observation.point_count)
    return np.column_stack([points[name].astype(np.float64) for name in "xyz"])


def _rejections(plan: GeometryMappingPlan) -> dict[str, InputRejectionReason]:
    return {str(item.observation_id): item.reason for item in plan.rejections}


# --- Motion compensation -------------------------------------------------------------------


def test_a_moving_platform_bends_a_wall_that_deskew_flattens_back_to_the_noise() -> None:
    scan = wall_scan()

    raw = np.abs(_map_points(assemble([scan], deskew=None))[:, 0] - WALL_X_M)
    corrected = np.abs(_map_points(assemble([scan]))[:, 0] - WALL_X_M)

    # 2 m/s e 0,6 rad/s durante 100 ms: sem deskew a parede se afasta do plano muito além do ruído.
    assert float(np.median(raw)) > 10 * NOISE_M
    assert float(corrected.max()) <= NOISE_M + 1e-5


def test_each_point_is_re_expressed_in_the_sensor_frame_at_the_scan_timestamp() -> None:
    # p' = T_bs^-1 · T_mb(t_ref)^-1 · T_mb(t_ref + tau) · T_bs · p é o ponto do mundo visto do
    # sensor em t_ref: a verdade simulada dá a resposta sem repetir a álgebra da implementação.
    plan = assemble([wall_scan()])

    corrected = _sensor_points(plan.inputs[0].observation)
    expected = np.array(
        [seen_from_sensor(wall_point(i), SCAN_START_NS) for i in range(POINT_COUNT)]
    )

    assert float(np.abs(corrected - expected).max()) < 1e-5


def test_points_of_the_bin_looked_up_at_the_scan_timestamp_are_left_byte_identical() -> None:
    # τ = 0 é a identidade exata; os pontos a menos de meio bin dividem o bin com eles e
    # são lidos no mesmo instante, com erro de tempo abaixo de meio bin.
    points, _ = observe_wall()
    times = [0.0 if index % 2 else 0.0004 for index in range(POINT_COUNT)]
    scan = make_scan(points, times)

    item = assemble([scan]).inputs[0]

    assert item.observation.data == scan.data
    record = item.motion_correction.record
    assert record.state is MotionCorrectionState.CORRECTED
    assert record.evidence is not None and record.evidence.payload_hash == _sha256(scan.data)


def test_the_evidence_names_the_corrected_payload_and_verifies_against_it() -> None:
    scan = wall_scan()

    item = assemble([scan]).inputs[0]

    record = item.motion_correction.record
    assert record.state is MotionCorrectionState.CORRECTED
    assert record.per_point_timing_available
    assert record.evidence is not None
    assert record.evidence.producer == "pose-interpolation-deskew-v1"
    assert record.evidence.trajectory_id == TRAJECTORY_ID
    assert record.evidence.configuration_fingerprint == deskew_configuration_fingerprint(
        POLICY, INTERPOLATED
    )
    assert record.evidence.payload_hash == item.payload_hash == _sha256(item.observation.data)
    assert item.observation.data != scan.data
    assert verify_motion_correction(record, item.observation) == []
    # O payload bruto continua sendo o da sequência; a evidência não o descreve.
    assert verify_motion_correction(record, scan) != []
    # Mesma observação física: só as coordenadas mudam, nunca a identidade, o layout ou o tempo.
    assert dataclasses.replace(item.observation, data=scan.data) == scan


def test_the_configuration_fingerprint_changes_with_the_policy_and_the_pose_lookup() -> None:
    base = deskew_configuration_fingerprint(POLICY, INTERPOLATED)

    assert base == deskew_configuration_fingerprint(dataclasses.replace(POLICY), INTERPOLATED)
    assert base.startswith("sha256:")
    changed = [
        deskew_configuration_fingerprint(
            dataclasses.replace(POLICY, time_bin_ns=2 * MS), INTERPOLATED
        ),
        deskew_configuration_fingerprint(dataclasses.replace(POLICY, time_field="t"), INTERPOLATED),
        deskew_configuration_fingerprint(
            POLICY, LookupPolicy.interpolated(max_interpolation_gap_ns=200 * MS)
        ),
    ]
    assert base not in changed and len(set(changed)) == 3


def test_a_point_instant_without_an_acceptable_pose_rejects_the_scan_explicitly() -> None:
    # A pose do cabeçalho (200 ms) é exata, mas de 220 a 400 ms o intervalo entre poses
    # (180 ms) excede a tolerância de 100 ms: a cauda do scan não tem pose aceitável.
    trajectory = make_trajectory(missing_ms=range(240, 400, POSE_PERIOD_MS))
    scan = wall_scan()

    without = assemble([scan], deskew=None, trajectory=trajectory)
    plan = assemble([scan], trajectory=trajectory)

    assert [str(item.observation_id) for item in without.inputs] == ["scan-0000"]
    assert plan.inputs == ()
    assert _rejections(plan) == {"scan-0000": InputRejectionReason.DESKEW_REJECTED}
    assert "interpolation_gap" in plan.rejections[0].detail


def test_a_time_bin_is_never_looked_up_after_the_last_instant_it_serves() -> None:
    # Regressão medida no corridor-02: o último ponto (τ = 100,84 ms) arredondava para 101 ms,
    # depois da amostra de pose seguinte, e caía num gap onde nenhum ponto foi medido.
    # Aqui o último ponto está em 299,9 ms, antes da pose de 300 ms; o bin (100 ms) daria
    # 300,3 ms, dentro do intervalo 300→500 ms, que excede a tolerância.
    header_ns = SCAN_START_NS + 300_000
    points, times = observe_wall(header_ns)
    last = len(points) - 1
    points[last] = seen_from_sensor(wall_point(last), header_ns + 99_600_000)
    times[last] = 0.0996
    trajectory = make_trajectory(missing_ms=range(320, 500, POSE_PERIOD_MS))

    plan = assemble([make_scan(points, times, header_ns=header_ns)], trajectory=trajectory)

    assert plan.rejections == ()
    corrected = _sensor_points(plan.inputs[0].observation)
    expected = seen_from_sensor(wall_point(last), header_ns)
    assert float(np.abs(corrected[last] - np.array(expected)).max()) < 1e-5


def test_a_scan_already_declared_corrected_is_never_corrected_twice() -> None:
    scan = wall_scan()
    declared = MotionCorrectionRecord(
        observation_id=scan.observation_id,
        state=MotionCorrectionState.CORRECTED,
        per_point_timing_available=True,
        evidence=MotionCorrectionEvidence(
            producer="upstream-deskew",
            trajectory_id=TRAJECTORY_ID,
            payload_hash=_sha256(scan.data),
        ),
    )

    plan = assemble([scan], motion_correction={scan.observation_id: declared})

    assert _rejections(plan) == {"scan-0000": InputRejectionReason.DESKEW_REJECTED}
    assert "already" in plan.rejections[0].detail


def test_deskew_needs_an_interpolated_pose_lookup() -> None:
    with pytest.raises(GeometryInputError, match="interpolated"):
        assemble([wall_scan()], pose_lookup=LookupPolicy.nearest(max_time_delta_ns=10 * MS))


@pytest.mark.parametrize(
    ("field", "time_type"),
    [("stamp", PointFieldDataType.FLOAT32), ("time", PointFieldDataType.UINT32)],
)
def test_a_scan_whose_time_field_is_missing_or_not_floating_point_is_rejected(
    field: str, time_type: PointFieldDataType
) -> None:
    points, _ = observe_wall()
    scan = make_scan(points, [0] * POINT_COUNT, time_field=field, time_type=time_type)

    plan = assemble([scan])

    assert _rejections(plan) == {"scan-0000": InputRejectionReason.DESKEW_REJECTED}
    assert "'time'" in plan.rejections[0].detail


def test_a_finite_point_with_a_non_finite_time_rejects_the_scan() -> None:
    points, times = observe_wall()
    times[7] = math.nan

    plan = assemble([make_scan(points, times)])

    assert _rejections(plan) == {"scan-0000": InputRejectionReason.DESKEW_REJECTED}
    assert "not finite" in plan.rejections[0].detail


def test_a_point_without_finite_coordinates_is_left_as_it_was_and_dropped_downstream() -> None:
    points, times = observe_wall()
    points[3] = (math.nan, math.nan, math.nan)
    times[3] = math.nan
    scan = make_scan(points, times)

    plan = assemble([scan])

    item = plan.inputs[0]
    assert item.observation.data[3 * 22 : 4 * 22] == scan.data[3 * 22 : 4 * 22]
    (transformed,) = transform_scans_to_map_frame(plan)
    assert transformed.dropped_non_finite_count == 1


def test_the_plan_records_the_deskew_it_was_assembled_under() -> None:
    assert assemble([wall_scan()]).policy.deskew == POLICY
    assert assemble([wall_scan()], deskew=None).policy.deskew is None


def test_the_policy_is_explicit_and_validated() -> None:
    with pytest.raises(TypeError):
        PoseInterpolationDeskewPolicy()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="time_field"):
        dataclasses.replace(POLICY, time_field="")
    with pytest.raises(ValueError, match="time_bin_ns"):
        dataclasses.replace(POLICY, time_bin_ns=0)


def test_accumulated_deskewed_scans_are_recorded_corrected_with_the_corrected_payload() -> None:
    plan = assemble([wall_scan()])

    accumulated = accumulate_plan(
        plan,
        map_id=MapId("deskew--run-0001"),
        sink=io.BytesIO(),
        aggregation=None,
        configuration_fingerprint=None,
        code_version="test",
    )

    (record,) = accumulated.scans
    evidence = plan.inputs[0].motion_correction.record.evidence
    assert record.motion_correction is MotionCorrectionState.CORRECTED
    assert evidence is not None and record.payload_hash == evidence.payload_hash


# --- Regression: without deskew nothing changes --------------------------------------------


def test_without_deskew_the_inputs_and_the_geometry_are_unchanged(tmp_path: Path) -> None:
    scan = wall_scan()
    plan = assemble([scan], deskew=None)

    item = plan.inputs[0]
    assert item.observation is scan
    assert item.payload_hash == _sha256(scan.data)
    assert item.motion_correction.record.state is MotionCorrectionState.UNKNOWN
    GeometricMapArtifactWriter(
        output_dir=tmp_path / "run",
        sequence_name="deskew-fixture",
        run_id=GeometricMapRunId("run-0001"),
        run_index=1,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    geometry = (tmp_path / "run" / "outputs" / "geometry.bin").read_bytes()
    assert hashlib.sha256(geometry).hexdigest() == GEOMETRY_WITHOUT_DESKEW_SHA256


def test_the_mapping_fingerprint_records_the_deskew_policy_only_when_it_is_selected() -> None:
    """A deskewed map is another map: its configuration fingerprint must say so (C3 wiring).

    Without deskew the recorded configuration keeps its previous shape, so maps built before the
    corrector existed keep their fingerprint.
    """
    from contextmap.geometric_mapping import mapping_configuration_fingerprint
    from contextmap.geometric_mapping.run_artifact import encode_mapping_configuration

    raw = assemble([wall_scan()], deskew=None)
    deskewed = assemble([wall_scan()])
    coarser = assemble([wall_scan()], deskew=dataclasses.replace(POLICY, time_bin_ns=5 * MS))

    assert "deskew" not in encode_mapping_configuration(raw, None)
    assert encode_mapping_configuration(deskewed, None)["deskew"]["policy_id"] == POLICY.policy_id
    fingerprints = {
        mapping_configuration_fingerprint(plan, None) for plan in (raw, deskewed, coarser)
    }
    assert len(fingerprints) == 3
