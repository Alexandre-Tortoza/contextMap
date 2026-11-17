"""Metrics that judge a camera-body calibration against images and annotations (C2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from contextmap.evaluation.camera_body_calibration import (
    MIN_SAMPLES_PER_POINT,
    PhotometricGroup,
    photometric_inconsistency,
    pixel_displacement,
    relative_improvement,
    reprojection_residual,
)
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    FrameId,
    PinholeCameraModel,
    SensorId,
    camera_rotation_correction,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.sensor_association import CameraProjection, camera_projection_for
from contextmap.shared import quaternion_to_rotation_matrix

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SHAPE = (480, 640)


def _projection() -> CameraProjection:
    model = PinholeCameraModel(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("front_camera-calib"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=model,
        provenance=CalibrationProvenance(source_type="manual", source_path="test"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            camera_model=model,
        ),
    )
    return camera_projection_for(entry)


def _constant_textures(*values: float) -> tuple[np.ndarray, ...]:
    return tuple(np.full(SHAPE, value) for value in values)


def _group(
    points: list[list[tuple[float, float, float]]],
    textures: tuple[np.ndarray, ...],
    visible: np.ndarray | None = None,
) -> PhotometricGroup:
    camera_points = np.array(points, dtype=float)
    return PhotometricGroup(
        camera_points_m=camera_points,
        visible=np.ones(camera_points.shape[:2], dtype=bool) if visible is None else visible,
        textures=textures,
    )


# --- inconsistência fotométrica ---------------------------------------------------------------


def test_the_inconsistency_is_the_variance_of_the_texture_a_point_samples_across_frames() -> None:
    ahead = (0.0, 0.0, 2.0)
    group = _group([[ahead, ahead, ahead]], _constant_textures(0.0, 1.0, 2.0))

    result = photometric_inconsistency([group], projection=_projection(), correction=IDENTITY)

    assert MIN_SAMPLES_PER_POINT == 3
    assert result.point_count == 1
    assert math.isclose(result.value, 2.0 / 3.0)


def test_a_point_with_fewer_samples_than_the_minimum_does_not_count() -> None:
    ahead = (0.0, 0.0, 2.0)
    outside = (10.0, 0.0, 1.0)  # u = 5320 px: fora da imagem, a amostra se perde
    visible = np.array([[True, True, True], [True, True, False], [True, True, True]])
    group = _group(
        [[ahead, ahead, ahead], [ahead, ahead, ahead], [ahead, ahead, outside]],
        _constant_textures(0.0, 1.0, 2.0),
        visible,
    )

    result = photometric_inconsistency([group], projection=_projection(), correction=IDENTITY)

    assert result.point_count == 1
    assert math.isclose(result.value, 2.0 / 3.0)


def test_groups_are_pooled_by_point_not_averaged_by_group() -> None:
    ahead = (0.0, 0.0, 2.0)
    varying = _group([[ahead, ahead, ahead]], _constant_textures(0.0, 1.0, 2.0))
    steady = _group([[ahead] * 3, [ahead] * 3], _constant_textures(5.0, 5.0, 5.0))

    result = photometric_inconsistency(
        [varying, steady], projection=_projection(), correction=IDENTITY
    )

    assert result.point_count == 3
    assert math.isclose(result.value, (2.0 / 3.0) / 3.0)


def test_the_true_correction_makes_every_point_sample_its_own_texture_again() -> None:
    """Base coordinates are ``Rcorr^T p``: only ``p = Rcorr p_base`` recovers the scene."""
    projection = _projection()
    rng = np.random.default_rng(11)
    world = np.column_stack(
        [rng.uniform(-1.0, 1.0, 400), rng.uniform(-0.7, 0.7, 400), rng.uniform(3.0, 5.0, 400)]
    )
    centres = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.0]])
    true_points = world[:, None, :] - centres[None, :, :]
    pixels = [
        np.round(projection.project(true_points[:, frame]).pixels).astype(int) for frame in range(3)
    ]
    keys = [pixel[:, 1] * SHAPE[1] + pixel[:, 0] for pixel in pixels]
    # Um ponto por pixel em todo frame: senão dois pontos disputam a mesma amostra.
    unique = np.ones(len(world), dtype=bool)
    for key in keys:
        _, first, counts = np.unique(key, return_index=True, return_counts=True)
        once = np.zeros(len(world), dtype=bool)
        once[first[counts == 1]] = True
        unique &= once
    albedo = rng.uniform(-3.0, 3.0, len(world))
    textures = []
    for pixel in pixels:
        texture = rng.uniform(-3.0, 3.0, SHAPE)
        texture[pixel[unique, 1], pixel[unique, 0]] = albedo[unique]
        textures.append(texture)
    correction = camera_rotation_correction(yaw_deg=3.0, pitch_deg=-2.0, roll_deg=0.0)
    rotation = np.array(quaternion_to_rotation_matrix(correction))
    base_points = true_points[unique] @ rotation  # linhas: Rcorr^T p
    group = PhotometricGroup(
        camera_points_m=base_points,
        visible=np.ones(base_points.shape[:2], dtype=bool),
        textures=tuple(textures),
    )

    at_truth = photometric_inconsistency([group], projection=projection, correction=correction)
    at_base = photometric_inconsistency([group], projection=projection, correction=IDENTITY)

    assert at_truth.point_count == int(unique.sum()) > 300
    assert at_truth.value < 1e-20
    assert at_base.value > 1.0


def test_a_group_whose_arrays_disagree_is_refused() -> None:
    ahead = (0.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="visible"):
        _group([[ahead, ahead, ahead]], _constant_textures(0, 1, 2), np.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError, match="texture"):
        _group([[ahead, ahead, ahead]], _constant_textures(0, 1))


def test_no_point_with_enough_samples_is_an_error_not_a_zero() -> None:
    ahead = (0.0, 0.0, 2.0)
    group = _group(
        [[ahead, ahead, ahead]],
        _constant_textures(0, 1, 2),
        np.array([[True, False, True]]),
    )

    with pytest.raises(ValueError, match="no point"):
        photometric_inconsistency([group], projection=_projection(), correction=IDENTITY)


# --- deslocamento em pixel --------------------------------------------------------------------


def test_the_pixel_displacement_summarizes_points_projected_by_both() -> None:
    first = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.nan, np.nan]])
    second = np.array([[3.0, 4.0], [6.0, 8.0], [0.0, 1.0], [1.0, 1.0]])

    result = pixel_displacement(first, second)

    assert result.compared_count == 3
    assert result.excluded_count == 1
    assert math.isclose(result.median_px, 5.0)
    assert math.isclose(result.p10_px, 1.8)
    assert math.isclose(result.p90_px, 9.0)


def test_a_pixel_displacement_with_nothing_to_compare_is_refused() -> None:
    with pytest.raises(ValueError, match="no point"):
        pixel_displacement(np.full((2, 2), np.nan), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        pixel_displacement(np.zeros((2, 2)), np.zeros((3, 2)))


# --- resíduo de reprojeção --------------------------------------------------------------------


def test_the_reprojection_residual_measures_annotated_correspondences() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.0, -1.0]])
    observed = np.array([[323.0, 244.0], [370.0, 240.0], [320.0, 240.0]])

    result = reprojection_residual(
        projection=_projection(), points_camera_m=points, observed_pixels=observed
    )

    assert result.correspondence_count == 2
    assert result.unprojectable_count == 1
    assert math.isclose(result.median_px, 2.5)
    assert math.isclose(result.p90_px, 4.5)


# --- melhoria relativa ------------------------------------------------------------------------


def test_the_relative_improvement_of_a_lower_is_better_metric() -> None:
    assert math.isclose(relative_improvement(0.19177, 0.13293), 0.30682588517493874)
    assert relative_improvement(1.0, 1.5) == -0.5
    with pytest.raises(ValueError, match="positive"):
        relative_improvement(0.0, 0.1)
