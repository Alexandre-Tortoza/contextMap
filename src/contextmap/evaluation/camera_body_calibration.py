"""Objective measures of a camera-body calibration and of a refinement of it (C2).

A rotation error of the camera extrinsic is invisible in the map and in the camera model alone:
it shows up only where the two meet. Four measures make it comparable between calibrations:

* **Photometric inconsistency** — the same map point, seen by several frames, should sample
  the same local texture in each; a wrong camera rotation lands it on different surfaces. It
  is the definition of the semantic-geometry investigation's estimator
  (``p4_calibration_photometric.py``), fixed here so that an estimation and its validation
  measure the same thing.
* **Pixel displacement** — how far two calibrations put the same points apart in the image.
* **Reprojection residual** — distance between annotated 2D↔3D correspondences and the
  projection of their 3D side.
* **Relative improvement** — the validation gain of a lower-is-better measure.

Nothing here chooses a correction or alters an artifact: the experiment supplies inputs and
reads the numbers. See ``src/contextmap/evaluation/docs/camera-body-calibration.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.sensor_association import CameraProjection
from contextmap.shared import Quaternion, quaternion_to_rotation_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

MIN_SAMPLES_PER_POINT = 3
"""Texture samples a point needs, across a group's frames, to enter the inconsistency."""


@dataclass(frozen=True, kw_only=True, eq=False)
class PhotometricGroup:
    """A short stretch of frames and the map points it measures the calibration with.

    Visibility is decided once, under the base calibration, by the caller; a correction only
    changes where each visible point reads the texture, so every candidate correction is
    judged on the same observations.

    Attributes:
        camera_points_m: ``(N, F, 3)`` each point in each frame's base camera optical frame,
            in meters.
        visible: ``(N, F)`` booleans: frame ``f`` sees point ``n``.
        textures: ``F`` arrays ``(H, W)``, the normalized local texture of each frame, in the
            raw image's pixel grid.
    """

    camera_points_m: NDArray[Any]
    visible: NDArray[Any]
    textures: tuple[NDArray[Any], ...]

    def __post_init__(self) -> None:
        """Validate that the arrays describe the same points and frames.

        Raises:
            ValueError: If ``camera_points_m`` is not ``(N, F, 3)``, ``visible`` is not
                ``(N, F)``, or there is not one 2D texture per frame.
        """
        points = self.camera_points_m
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"camera_points_m must have shape (N, F, 3), got {points.shape}")
        if self.visible.shape != points.shape[:2]:
            raise ValueError(
                f"visible must have shape {points.shape[:2]}, got {self.visible.shape}"
            )
        if len(self.textures) != points.shape[1]:
            raise ValueError(
                f"expected one texture per frame ({points.shape[1]}), got {len(self.textures)}"
            )
        if any(texture.ndim != 2 for texture in self.textures):
            raise ValueError("every texture must be a 2D (H, W) array")


@dataclass(frozen=True, kw_only=True)
class PhotometricInconsistency:
    """Mean per-point texture variance over the points with enough samples.

    Attributes:
        value: Mean, over the counted points, of the population variance of the texture
            values each point sampled, in squared texture units. Lower is better.
        point_count: Points with at least :data:`MIN_SAMPLES_PER_POINT` samples.
    """

    value: float
    point_count: int


@dataclass(frozen=True, kw_only=True)
class PixelDisplacement:
    """How far two projections of the same points are apart, in raw-image pixels.

    Attributes:
        compared_count: Points with a finite pixel in both projections.
        excluded_count: Points missing from at least one of them.
        median_px: Median displacement.
        p10_px: 10th percentile (linear interpolation).
        p90_px: 90th percentile (linear interpolation).
    """

    compared_count: int
    excluded_count: int
    median_px: float
    p10_px: float
    p90_px: float


@dataclass(frozen=True, kw_only=True)
class ReprojectionResidual:
    """Distance between annotated pixels and the projection of their 3D points.

    Attributes:
        correspondence_count: Correspondences whose 3D point the camera model projects.
        unprojectable_count: Correspondences whose 3D point it cannot project; they are
            reported, never silently dropped into the statistics.
        median_px: Median residual, in raw-image pixels.
        p90_px: 90th percentile residual (linear interpolation), in raw-image pixels.
    """

    correspondence_count: int
    unprojectable_count: int
    median_px: float
    p90_px: float


def photometric_inconsistency(
    groups: Sequence[PhotometricGroup],
    *,
    projection: CameraProjection,
    correction: Quaternion,
) -> PhotometricInconsistency:
    """Measure how consistently map points sample texture across frames under a correction.

    Each visible point is moved by ``p_camera = Rcorr p_camera_base``, projected, rounded to
    the nearest pixel and, when that pixel lies in the texture, sampled. A point with at least
    :data:`MIN_SAMPLES_PER_POINT` samples contributes the variance of its samples. Points are
    pooled across groups: the result is the total variance over the total count, not a mean
    of group means.

    Args:
        groups: The frame stretches and their points.
        projection: The camera model of every frame.
        correction: ``Rcorr``, e.g. from
            :func:`~contextmap.ingestion.camera_rotation_correction`; the identity measures the
            base calibration.

    Returns:
        The pooled inconsistency and how many points it covers.

    Raises:
        ValueError: If no point has enough samples, since an empty measure is not zero.
    """
    import numpy as np

    rotation = np.array(quaternion_to_rotation_matrix(correction), dtype=np.float64)
    total = 0.0
    count = 0
    for group in groups:
        samples = np.full(group.visible.shape, np.nan)
        for frame, texture in enumerate(group.textures):
            rows = np.flatnonzero(group.visible[:, frame])
            points = group.camera_points_m[rows, frame].astype(np.float64) @ rotation.T
            column, row, inside = _texture_pixels(projection, points, texture.shape)
            samples[rows[inside], frame] = texture[row[inside], column[inside]]
        enough = np.sum(~np.isnan(samples), axis=1) >= MIN_SAMPLES_PER_POINT
        if enough.any():
            total += float(np.sum(np.nanvar(samples[enough], axis=1)))
            count += int(enough.sum())
    if count == 0:
        raise ValueError(
            f"no point has {MIN_SAMPLES_PER_POINT} texture samples; the inconsistency is undefined"
        )
    return PhotometricInconsistency(value=total / count, point_count=count)


def _texture_pixels(
    projection: CameraProjection, points_camera_m: NDArray[Any], shape: tuple[int, ...]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Nearest texture pixel of each point and whether it exists.

    O arredondamento é ``np.round`` (metade para o par) e o teste de limites é sobre o índice
    inteiro, exatamente como o estimador da investigação, para que os números se reproduzam.
    """
    import numpy as np

    pixels = projection.project(points_camera_m).pixels
    inside = np.isfinite(pixels).all(axis=1)
    column = np.zeros(len(pixels), dtype=np.int64)
    row = np.zeros(len(pixels), dtype=np.int64)
    column[inside] = np.round(pixels[inside, 0]).astype(np.int64)
    row[inside] = np.round(pixels[inside, 1]).astype(np.int64)
    inside &= (column >= 0) & (column < shape[1]) & (row >= 0) & (row < shape[0])
    return column, row, inside


def pixel_displacement(first: NDArray[Any], second: NDArray[Any]) -> PixelDisplacement:
    """Summarize the displacement between two projections of the same points.

    Args:
        first: ``(N, 2)`` pixels of one projection; ``NaN`` where a point was not projected.
        second: ``(N, 2)`` pixels of the other, in the same point order.

    Returns:
        The displacement statistics over the points projected by both.

    Raises:
        ValueError: If the shapes differ or no point is projected by both.
    """
    import numpy as np

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
        raise ValueError(
            f"both projections must have the same (N, 2) shape, got {first.shape} and "
            f"{second.shape}"
        )
    both = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    if not both.any():
        raise ValueError("no point is projected by both calibrations")
    distance = np.linalg.norm(second[both] - first[both], axis=1)
    return PixelDisplacement(
        compared_count=int(both.sum()),
        excluded_count=int((~both).sum()),
        median_px=float(np.median(distance)),
        p10_px=float(np.percentile(distance, 10)),
        p90_px=float(np.percentile(distance, 90)),
    )


def reprojection_residual(
    *,
    projection: CameraProjection,
    points_camera_m: NDArray[Any],
    observed_pixels: NDArray[Any],
) -> ReprojectionResidual:
    """Measure annotated 2D↔3D correspondences against the calibration under test.

    Args:
        projection: The camera model.
        points_camera_m: ``(N, 3)`` the 3D side of each correspondence, already expressed in
            the camera optical frame by the calibration under test, in meters.
        observed_pixels: ``(N, 2)`` the annotated raw-image pixel of each correspondence.

    Returns:
        The residual statistics over the projectable correspondences.

    Raises:
        ValueError: If the arrays disagree in length or no correspondence is projectable.
    """
    import numpy as np

    observed = np.asarray(observed_pixels, dtype=np.float64)
    points = np.asarray(points_camera_m, dtype=np.float64)
    if observed.shape != (points.shape[0], 2):
        raise ValueError(
            f"observed_pixels must have shape ({points.shape[0]}, 2), got {observed.shape}"
        )
    projected = projection.project(points)
    usable = projected.projectable
    if not usable.any():
        raise ValueError("no correspondence is projectable by the camera model")
    residual = np.linalg.norm(projected.pixels[usable] - observed[usable], axis=1)
    return ReprojectionResidual(
        correspondence_count=int(usable.sum()),
        unprojectable_count=int((~usable).sum()),
        median_px=float(np.median(residual)),
        p90_px=float(np.percentile(residual, 90)),
    )


def relative_improvement(baseline: float, candidate: float) -> float:
    """Return ``(baseline - candidate) / baseline`` for a lower-is-better measure.

    Args:
        baseline: The measure under the base calibration; positive.
        candidate: The same measure under the candidate calibration.

    Returns:
        The fraction of the baseline removed; negative when the candidate is worse.

    Raises:
        ValueError: If ``baseline`` is not a positive finite number or ``candidate`` is not
            finite.
    """
    if not (math.isfinite(baseline) and baseline > 0):
        raise ValueError(f"baseline must be a positive finite number, got {baseline!r}")
    if not math.isfinite(candidate):
        raise ValueError(f"candidate must be finite, got {candidate!r}")
    return (baseline - candidate) / baseline
