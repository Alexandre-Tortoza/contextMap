"""A camera-body refinement is versioned data applied once to the base calibration (C2).

The corridor-02 values below are the base calibration of the E0 benchmark run (the MEI camera and
the ``T_epson_d``/``T_epson_cmu_rc1_velodyne`` extrinsics rebuilt from the dataset YAMLs), so the
identities pinned here are the ones E0 recorded.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationRefinement,
    CalibrationSet,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    RefinementValidation,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    apply_calibration_refinement,
    camera_rotation_correction,
    load_calibration_refinement,
    rigid_transform_digest,
    validate_calibration_set,
)
from contextmap.ingestion.calibration import (
    CAMERA_OPTICAL_ROTATION_CONVENTION,
    MAX_ROTATION_CORRECTION_DEG,
    REFINEMENT_DOCUMENT_SCHEMA,
    compute_content_hash,
    decode_calibration_refinement_document,
    decode_calibration_set,
    encode_calibration_refinement_document,
    encode_calibration_set,
)
from contextmap.runtime.ingestion_service import _calibration_identity
from contextmap.sensor_association import camera_projection_for
from contextmap.shared import (
    Vector3,
    invert_rigid,
    quaternion_to_rotation_matrix,
    rotate_vector,
)
from contextmap.state_estimation import calibration_identity


def test_calibration_package_preserves_the_pre_split_public_constants() -> None:
    """The package facade retains the former calibration module's public contract."""
    assert REFINEMENT_DOCUMENT_SCHEMA == "calibration-refinement/1"
    assert CAMERA_OPTICAL_ROTATION_CONVENTION
    assert MAX_ROTATION_CORRECTION_DEG == 10.0


BODY, CAMERA, LIDAR = FrameId("epson"), FrameId("d"), FrameId("cmu_rc1_velodyne")
_PROVENANCE = CalibrationProvenance(
    source_type="dataset",
    source_path="datasets/corridor-02",
    conversions_applied=("rebuilt fresh from YAMLs (#688 item 5)",),
)
_MEI = MeiCameraModel(
    width=640,
    height=480,
    fx=761.6791126513665,
    fy=678.7676595096547,
    cx=317.73997456826834,
    cy=233.50210820810244,
    xi=1.5629934283981157,
    distortion_coefficients=(
        -0.07083665039247518,
        0.3094574364801575,
        0.0004607746627279861,
        0.00015704097191542886,
    ),
)
_T_EPSON_D = RigidTransform(
    parent_frame=BODY,
    child_frame=CAMERA,
    translation=(0.17313616, 0.0476788, -0.01933398),
    rotation=(0.5026385421736821, -0.5028457380724255, 0.5050125794996385, -0.4893495214782934),
)
_T_EPSON_VELODYNE = RigidTransform(
    parent_frame=BODY,
    child_frame=LIDAR,
    translation=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.004749982138040742, 0.0, 0.9999887187712111),
)
# Identidades que o E0 (workspace-old/frame-10_31-raw) registrou para essa mesma calibração.
_E0_CALIBRATION_IDENTITY = "sha256:f7d3288e82305a34960dee8b841d625c7de643a506cb5706395a9f9bbe3242d6"
_E0_CALIBRATION_SOURCE_HASH = (
    "sha256:4fe08c293d66def337da052c6742d63f669b23ce851ca62afa339b820fcfae48"
)
_V1_CORRECTION = (4.25, -2.25, 0.0)


def _entry(
    sensor: str, frame: FrameId, model: MeiCameraModel | PinholeCameraModel | None
) -> CalibrationEntry:
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId(f"{sensor}-calib"),
        sensor_id=SensorId(sensor),
        frame_id=frame,
        camera_model=model,
        provenance=_PROVENANCE,
        content_hash=compute_content_hash(
            sensor_id=SensorId(sensor), frame_id=frame, camera_model=model
        ),
    )


def _corridor_base() -> CalibrationSet:
    entries = (
        _entry("camera_1_image_raw", CAMERA, _MEI),
        _entry("velodyne_points", LIDAR, None),
        _entry("imu_data", BODY, None),
    )
    return CalibrationSet(
        entries={entry.calibration_id: entry for entry in entries},
        static_transforms=(_T_EPSON_D, _T_EPSON_VELODYNE),
    )


def _refinement(
    base: CalibrationSet | None = None,
    *,
    refinement_id: str = "corridor-02-camera-body-refinement-test",
    yaw_deg: float = 4.0,
    pitch_deg: float = -1.75,
    roll_deg: float = 0.0,
    child_frame: FrameId = CAMERA,
    metrics: dict[str, float] | None = None,
) -> CalibrationRefinement:
    base = _corridor_base() if base is None else base
    target = next(t for t in base.static_transforms if t.child_frame == child_frame)
    return CalibrationRefinement(
        refinement_id=refinement_id,
        parent_frame=target.parent_frame,
        child_frame=target.child_frame,
        base_transform_digest=rigid_transform_digest(target),
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        convention=CAMERA_OPTICAL_ROTATION_CONVENTION,
        method="multi-frame photometric consistency grid search",
        estimator="experiments/correction-benchmark-10_31-20260930/calibration/estimate_refinement.py",
        code_identity="sha256:" + "0" * 64,
        pose_source_kind="ground_truth_tum",
        pose_source_digest="sha256:" + "1" * 64,
        source_files={"corridor-02-extrinsics.yaml": "sha256:" + "2" * 64},
        validation=RefinementValidation(
            fit_ids=("camera_1_image_raw-000000",),
            held_out_ids=("camera_1_image_raw-000027",),
            metrics={"held_out_inconsistency_base": 0.19177} if metrics is None else metrics,
        ),
    )


def _p4_rotation_correction(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Literal copy of ``p4_calibration_photometric.rotation_correction`` (the estimator's)."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rotation: np.ndarray = rz @ rx @ ry
    return rotation


def _matrix(transform: RigidTransform) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(quaternion_to_rotation_matrix(transform.rotation))
    matrix[:3, 3] = transform.translation
    return matrix


def _camera_transform(calibration: CalibrationSet) -> RigidTransform:
    return next(t for t in calibration.static_transforms if t.child_frame == CAMERA)


def _in_camera(transform: RigidTransform, point_body: Vector3) -> Vector3:
    """``p_camera = T_body_camera^-1 p_body``."""
    translation, rotation = invert_rigid(
        translation=transform.translation, rotation=transform.rotation
    )
    x, y, z = rotate_vector(rotation, point_body)
    return (x + translation[0], y + translation[1], z + translation[2])


def _to_body(transform: RigidTransform, point_camera: Vector3) -> Vector3:
    x, y, z = rotate_vector(transform.rotation, point_camera)
    t = transform.translation
    return (x + t[0], y + t[1], z + t[2])


# --- convenção e composição -------------------------------------------------------------------


def test_the_correction_is_the_estimators_rz_rx_ry_rotation() -> None:
    for angles in ((4.25, -2.25, 0.0), (4.0, -1.75, 0.0), (-3.0, 2.0, 1.5)):
        yaw, pitch, roll = angles
        quaternion = camera_rotation_correction(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

        assert np.allclose(
            np.array(quaternion_to_rotation_matrix(quaternion)),
            _p4_rotation_correction(*angles),
            atol=1e-12,
        )


def test_the_refined_extrinsic_applies_the_correction_to_base_camera_coordinates() -> None:
    """``p_camera = Rcorr p_camera_base`` for every body point, as p4 samples the texture."""
    base = _corridor_base()
    refined = apply_calibration_refinement(base, _refinement(base))
    correction = _p4_rotation_correction(4.0, -1.75, 0.0)
    rng = np.random.default_rng(3)

    for point in rng.uniform(-5.0, 5.0, size=(20, 3)):
        body: Vector3 = (float(point[0]), float(point[1]), float(point[2]))
        base_camera = np.array(_in_camera(_T_EPSON_D, body))
        refined_camera = np.array(_in_camera(_camera_transform(refined), body))

        assert np.allclose(refined_camera, correction @ base_camera, atol=1e-12)


def test_the_refined_extrinsic_matches_the_v1_lidar_to_camera_composition() -> None:
    """V1 ``load_calibration``: ``T_camera_lidar = C @ T_camera_lidar_base`` (translation too)."""
    base = _corridor_base()
    refined = apply_calibration_refinement(
        base,
        _refinement(base, yaw_deg=_V1_CORRECTION[0], pitch_deg=_V1_CORRECTION[1]),
    )
    correction = np.eye(4)
    correction[:3, :3] = _p4_rotation_correction(*_V1_CORRECTION)
    lidar_in_body = _matrix(_T_EPSON_VELODYNE)
    v1 = correction @ np.linalg.inv(_matrix(_T_EPSON_D)) @ lidar_in_body

    ours = np.linalg.inv(_matrix(_camera_transform(refined))) @ lidar_in_body

    assert np.allclose(ours, v1, atol=1e-12)


def test_only_the_rotation_of_the_target_transform_changes() -> None:
    base = _corridor_base()
    refinement = _refinement(base)

    refined = apply_calibration_refinement(base, refinement)

    camera = _camera_transform(refined)
    assert camera.translation == _T_EPSON_D.translation
    assert camera.rotation != _T_EPSON_D.rotation
    assert math.isclose(math.hypot(*camera.rotation), 1.0, abs_tol=1e-12)
    assert refined.static_transforms[1] == _T_EPSON_VELODYNE
    assert refined.entries == base.entries
    assert refined.refinements == (refinement,)
    assert base.refinements == ()
    assert validate_calibration_set(refined) == []


def test_positive_yaw_moves_the_pixel_right_and_positive_pitch_moves_it_up() -> None:
    pinhole = PinholeCameraModel(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    camera_entry = _entry("front_camera", CAMERA, pinhole)
    base = CalibrationSet(
        entries={camera_entry.calibration_id: camera_entry}, static_transforms=(_T_EPSON_D,)
    )
    projection = camera_projection_for(camera_entry)
    ahead = _to_body(_T_EPSON_D, (0.0, 0.0, 3.0))

    def pixel(calibration: CalibrationSet) -> np.ndarray:
        point = np.array([_in_camera(_camera_transform(calibration), ahead)])
        pixels: np.ndarray = projection.project(point).pixels[0]
        return pixels

    yawed = pixel(apply_calibration_refinement(base, _refinement(base, yaw_deg=2.0, pitch_deg=0)))
    pitched = pixel(
        apply_calibration_refinement(base, _refinement(base, yaw_deg=0.0, pitch_deg=2.0))
    )

    assert np.allclose(pixel(base), (320.0, 240.0))
    assert yawed[0] > 320.0 + 10.0
    assert math.isclose(yawed[1], 240.0, abs_tol=1e-9)
    assert pitched[1] < 240.0 - 10.0
    assert math.isclose(pitched[0], 320.0, abs_tol=1e-9)


def test_the_v1_correction_reproduces_the_investigations_known_pixel_shift() -> None:
    """Test D: map point 481273 moves from (120.5, 76.3) to (142.1, 88.6) px in frame 55."""
    # results/test_d_known_pixel.json da investigação.
    point_camera_m_base: Vector3 = (-0.5953975128816129, -0.5324752096799714, 0.6647451469491727)
    pixel_base = (120.49001521288031, 76.32646656922697)
    pixel_with_v1_correction = (142.1443172641207, 88.59224713318386)
    base = _corridor_base()
    refined = apply_calibration_refinement(
        base, _refinement(base, yaw_deg=_V1_CORRECTION[0], pitch_deg=_V1_CORRECTION[1])
    )
    projection = camera_projection_for(
        base.entries[CalibrationReferenceId("camera_1_image_raw-calib")]
    )
    body = _to_body(_T_EPSON_D, point_camera_m_base)

    base_pixel = projection.project(np.array([_in_camera(_T_EPSON_D, body)])).pixels[0]
    refined_pixel = projection.project(
        np.array([_in_camera(_camera_transform(refined), body)])
    ).pixels[0]

    assert np.allclose(base_pixel, pixel_base, atol=1e-6)
    assert np.allclose(refined_pixel, pixel_with_v1_correction, atol=1e-6)


# --- recusas ----------------------------------------------------------------------------------


def test_applying_the_same_refinement_twice_is_refused() -> None:
    base = _corridor_base()
    refinement = _refinement(base)
    refined = apply_calibration_refinement(base, refinement)

    with pytest.raises(CalibrationError, match="already"):
        apply_calibration_refinement(refined, refinement)
    with pytest.raises(CalibrationError, match="digest"):
        apply_calibration_refinement(
            refined, dataclasses.replace(refinement, refinement_id="another-id")
        )


def test_a_refinement_estimated_on_another_base_transform_is_refused() -> None:
    base = _corridor_base()
    other = dataclasses.replace(_refinement(base), base_transform_digest="sha256:" + "f" * 64)

    with pytest.raises(CalibrationError, match="digest"):
        apply_calibration_refinement(base, other)


def test_a_refinement_whose_target_transform_is_missing_is_refused() -> None:
    base = _corridor_base()
    missing = dataclasses.replace(_refinement(base), child_frame=FrameId("camera_2"))

    with pytest.raises(CalibrationError, match="no static transform"):
        apply_calibration_refinement(base, missing)


def test_a_correction_above_the_limit_is_refused() -> None:
    base = _corridor_base()

    with pytest.raises(CalibrationError, match="limit"):
        apply_calibration_refinement(
            base, _refinement(base, yaw_deg=MAX_ROTATION_CORRECTION_DEG + 0.5)
        )
    with pytest.raises(CalibrationError, match="finite"):
        apply_calibration_refinement(base, _refinement(base, roll_deg=math.nan))


def test_an_unknown_convention_is_refused() -> None:
    base = _corridor_base()
    other = dataclasses.replace(_refinement(base), convention="p_camera = Ry(yaw) @ p_base")

    with pytest.raises(CalibrationError, match="convention"):
        apply_calibration_refinement(base, other)


def test_a_correction_of_a_frame_that_is_not_a_calibrated_camera_is_refused() -> None:
    """The yaw/pitch/roll axes are the optical axes; a LiDAR frame has none."""
    base = _corridor_base()

    with pytest.raises(CalibrationError, match="camera"):
        apply_calibration_refinement(base, _refinement(base, child_frame=LIDAR))


def test_validation_reports_duplicate_ids_and_missing_targets() -> None:
    base = _corridor_base()
    refinement = _refinement(base)
    orphan = dataclasses.replace(refinement, child_frame=FrameId("camera_2"))

    problems = validate_calibration_set(
        dataclasses.replace(base, refinements=(refinement, refinement, orphan))
    )

    assert any("duplicate refinement_id" in problem for problem in problems)
    assert any("camera_2" in problem for problem in problems)


# --- serialização -----------------------------------------------------------------------------


def test_a_refined_calibration_round_trips() -> None:
    base = _corridor_base()
    refined = apply_calibration_refinement(base, _refinement(base))

    decoded = decode_calibration_set(json.loads(json.dumps(encode_calibration_set(refined))))

    assert decoded.refinements == refined.refinements
    assert decoded.static_transforms == refined.static_transforms
    assert calibration_identity(decoded) == calibration_identity(refined)


def test_a_refined_calibration_is_persisted_in_the_sequence_artifact(tmp_path: Path) -> None:
    base = _corridor_base()
    refined = apply_calibration_refinement(base, _refinement(base))
    writer = SequenceArtifactWriter(
        output_dir=tmp_path / "sequence",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-e1"),
    )
    writer.set_calibration(refined)
    writer.finalize()

    reader = SequenceArtifactReader(tmp_path / "sequence")
    persisted = reader.read_calibration()
    record = json.loads((tmp_path / "sequence/calibration/calibration.json").read_text("utf-8"))

    assert persisted is not None
    assert [item["refinement_id"] for item in record["refinements"]] == [
        "corridor-02-camera-body-refinement-test"
    ]
    assert persisted.refinements == refined.refinements
    assert calibration_identity(persisted) == calibration_identity(refined)
    assert reader.verify_integrity() == []


def test_a_calibration_without_refinements_is_encoded_as_before() -> None:
    """No ``refinements`` key: E0 records and new unrefined records stay byte-compatible."""
    record = encode_calibration_set(_corridor_base())

    assert "refinements" not in record
    assert decode_calibration_set(record).refinements == ()


def test_the_refinement_document_round_trips(tmp_path: Path) -> None:
    refinement = _refinement()
    document = encode_calibration_refinement_document(refinement)
    path = tmp_path / "refinement.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert document["schema"] == REFINEMENT_DOCUMENT_SCHEMA
    assert decode_calibration_refinement_document(document) == refinement
    assert load_calibration_refinement(path) == refinement


def test_a_document_with_another_schema_or_an_unknown_key_is_refused(tmp_path: Path) -> None:
    document = encode_calibration_refinement_document(_refinement())
    path = tmp_path / "refinement.json"
    path.write_text(json.dumps({**document, "schema": "calibration-refinement/2"}), "utf-8")

    with pytest.raises(CalibrationError, match="schema"):
        load_calibration_refinement(path)
    with pytest.raises(CalibrationError, match="rotation_corection"):
        decode_calibration_refinement_document({**document, "rotation_corection": {}})
    with pytest.raises(CalibrationError, match="limit"):
        decode_calibration_refinement_document(
            {**document, "rotation_correction": {**document["rotation_correction"], "yaw_deg": 45}}
        )


# --- identidade -------------------------------------------------------------------------------


def test_a_calibration_without_refinements_keeps_the_identity_e0_recorded() -> None:
    base = _corridor_base()

    assert calibration_identity(base) == _E0_CALIBRATION_IDENTITY
    assert _calibration_identity(base) == _E0_CALIBRATION_SOURCE_HASH


def test_the_identity_changes_with_a_refinement_but_not_with_its_evidence() -> None:
    base = _corridor_base()
    refinement = _refinement(base)
    refined = apply_calibration_refinement(base, refinement)
    # Mesmos transforms e entradas, mas sem o registro do refinamento: a identidade precisa
    # distinguir uma extrínseca refinada de uma editada à mão com os mesmos números.
    hand_edited = dataclasses.replace(refined, refinements=())
    other_evidence = apply_calibration_refinement(
        base,
        dataclasses.replace(
            refinement,
            method="another description",
            code_identity="sha256:" + "9" * 64,
            validation=RefinementValidation(
                fit_ids=(), held_out_ids=(), metrics={"held_out_inconsistency_base": 0.5}
            ),
        ),
    )
    renamed = apply_calibration_refinement(
        base, dataclasses.replace(refinement, refinement_id="renamed")
    )

    for identity in (calibration_identity, _calibration_identity):
        assert identity(refined) != identity(base)
        assert identity(refined) != identity(hand_edited)
        assert identity(refined) != identity(renamed)
        assert identity(refined) == identity(other_evidence)


def test_an_adapter_merge_keeps_the_refinement_of_a_provided_calibration() -> None:
    """A refined calibration handed to ingestion keeps its refinement record through the merge.

    The ROS adapters merge the provided calibration with the entries the source holds; dropping the
    refinement kept the corrected transform but lost which correction produced it, and with it the
    part of the calibration identity that names the refinement.
    """
    from contextmap.ingestion.adapters.ros import ros_adapter_operations as _ros_operations
    from contextmap.state_estimation import calibration_identity

    base = _corridor_base()
    refined = apply_calibration_refinement(base, _refinement(base))

    merged = _ros_operations.merge_calibration(refined, ())

    assert merged is not None
    assert merged.refinements == refined.refinements
    assert calibration_identity(merged) == calibration_identity(refined)
