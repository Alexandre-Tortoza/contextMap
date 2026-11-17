"""Contrato de projeção calibrada e comportamento comum dos três modelos.

Pontos estão no frame óptico, em metros; pixels têm centros em coordenadas inteiras.
O modelo declara seu domínio de visão e não recorta projeções inválidas para a imagem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationReferenceId,
    FisheyeCameraModel,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    camera_model_kind,
)
from contextmap.sensor_association.camera.distortion import lazy_numpy, sublevel_limit

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class CameraIdentity:
    """Which calibrated camera produced a projection.

    Attributes:
        calibration_id: The calibration entry the camera was built from.
        content_hash: Hash of that entry's canonical values.
        camera_frame: The optical frame points must be expressed in.
        camera_model_kind: ``"pinhole"``, ``"fisheye"`` or ``"mei"``.
        image_size: ``(width, height)`` of the raw image, in pixels.
    """

    calibration_id: CalibrationReferenceId
    content_hash: str
    camera_frame: FrameId
    camera_model_kind: str
    image_size: tuple[int, int]


@dataclass(frozen=True, kw_only=True, eq=False)
class PixelProjection:
    """The projection of ``N`` camera-frame points, with the camera that made it.

    Attributes:
        camera: The calibrated camera that produced the projection.
        pixels: ``(N, 2)`` raw-image pixels; ``NaN`` where a point is not projectable.
        projectable: ``(N,)`` ``True`` where the point lies in the model's viewing domain.
        depth_m: ``(N,)`` signed ``z`` of each point in the camera frame, in meters.
        range_m: ``(N,)`` distance from the optical center to each point, in meters.
    """

    camera: CameraIdentity
    pixels: NDArray[Any]
    projectable: NDArray[Any]
    depth_m: NDArray[Any]
    range_m: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate that the arrays describe the same points.

        Raises:
            ValueError: If the shapes disagree or ``projectable`` is not true
                exactly where the pixel is finite.
        """
        import numpy as np

        if self.projectable.ndim != 1:
            raise ValueError(f"projectable must have shape (N,), got {self.projectable.shape}")
        count = self.projectable.shape[0]
        if self.pixels.shape != (count, 2):
            raise ValueError(f"pixels must have shape ({count}, 2), got {self.pixels.shape}")
        if self.depth_m.shape != (count,) or self.range_m.shape != (count,):
            raise ValueError(f"depth_m and range_m must have shape ({count},)")
        if not np.array_equal(np.isfinite(self.pixels).all(axis=1), self.projectable):
            raise ValueError("projectable must be true exactly where the pixel is finite")


@runtime_checkable
class CameraProjection(Protocol):
    """Capability port: project camera-frame points to pixels and back."""

    @property
    def identity(self) -> CameraIdentity:
        """The calibrated camera this projection was built from."""
        ...

    def project(self, points_camera_m: NDArray[Any]) -> PixelProjection:
        """Project points expressed in the camera optical frame.

        Args:
            points_camera_m: ``(N, 3)`` finite points, in meters.

        Returns:
            The pixels, the depth and range of each point, and which points the
            model could project.

        Raises:
            ValueError: If the input is not a finite ``(N, 3)`` array.
        """
        ...

    def unproject(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Return the unit viewing ray of each pixel, in the camera optical frame.

        Args:
            pixels: ``(N, 2)`` finite raw-image pixels.

        Returns:
            ``(N, 3)`` unit vectors.

        Raises:
            ValueError: If the input is not a finite ``(N, 2)`` array or a pixel
                has no viewing ray under the model.
        """
        ...

    def in_image(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Tell which pixels fall inside the raw image extent.

        Args:
            pixels: ``(N, 2)`` pixels; ``NaN`` is never inside.

        Returns:
            ``(N,)`` booleans.
        """
        ...

    def max_ray_angle_rad(
        self, *, u_bounds: tuple[float, float], v_bounds: tuple[float, float]
    ) -> float:
        """Bound the angle to the optical axis of every ray the model sends into a pixel box.

        Args:
            u_bounds: ``(u_min, u_max)`` of the closed box, in raw pixels.
            v_bounds: ``(v_min, v_max)`` of the closed box, in raw pixels.

        Returns:
            An angle, in radians, that no ray of the model's viewing domain landing in the
            box exceeds. It is never above the domain's own limit.

        Raises:
            ValueError: If a bound is not finite or the box is empty.
        """
        ...


def _as_points(value: NDArray[Any]) -> NDArray[Any]:
    """Require finite camera-frame points with shape (N, 3)."""
    np = lazy_numpy()
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera_m must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_camera_m must be finite")
    return points  # type: ignore[no-any-return]


def _as_pixels(value: NDArray[Any], *, finite: bool) -> NDArray[Any]:
    """Require finite image pixels with shape (N, 2)."""
    np = lazy_numpy()
    pixels = np.asarray(value, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}")
    if finite and not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite")
    return pixels  # type: ignore[no-any-return]


class CalibratedProjection:
    """Shared plumbing: identity, validation, pixel extent and result assembly.

    Subclasses own the model math: the viewing domain, the forward projection of
    in-domain points and the inverse ray of a pixel.
    """

    def __init__(
        self,
        entry: CalibrationEntry,
        model: PinholeCameraModel | FisheyeCameraModel | MeiCameraModel,
    ) -> None:
        """Record the camera identity and the intrinsics shared by every model.

        Args:
            entry: The validated calibration entry the camera was built from.
            model: Its camera model, which fixes the kind and the raw image size.
        """
        self._identity = CameraIdentity(
            calibration_id=entry.calibration_id,
            content_hash=entry.content_hash,
            camera_frame=entry.frame_id,
            camera_model_kind=camera_model_kind(model),
            image_size=(model.width, model.height),
        )
        self._fx, self._fy, self._cx, self._cy = model.fx, model.fy, model.cx, model.cy

    @property
    def identity(self) -> CameraIdentity:
        """The calibrated camera this projection was built from."""
        return self._identity

    def project(self, points_camera_m: NDArray[Any]) -> PixelProjection:
        """Project points, reporting the ones outside the viewing domain."""
        np = lazy_numpy()
        points = _as_points(points_camera_m)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        range_m = np.sqrt(x * x + y * y + z * z)
        projectable = self._in_domain(x, y, z, range_m)
        pixels = np.full((points.shape[0], 2), np.nan)
        if projectable.any():
            pixels[projectable] = self._to_pixels(
                x[projectable], y[projectable], z[projectable], range_m[projectable]
            )
        return PixelProjection(
            camera=self._identity,
            pixels=pixels,
            projectable=projectable,
            depth_m=z.copy(),
            range_m=range_m,
        )

    def unproject(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Return unit viewing rays; a pixel with no ray is an error, not a guess."""
        np = lazy_numpy()
        array = _as_pixels(pixels, finite=True)
        xd = (array[:, 0] - self._cx) / self._fx
        yd = (array[:, 1] - self._cy) / self._fy
        rays, has_ray = self._to_rays(xd, yd)
        if not has_ray.all():
            first = array[int(np.argmin(has_ray))]
            raise ValueError(
                f"{int((~has_ray).sum())} pixel(s) have no viewing ray under the "
                f"{self._identity.camera_model_kind} model, e.g. {tuple(float(v) for v in first)}"
            )
        return rays  # type: ignore[no-any-return]

    def in_image(self, pixels: NDArray[Any]) -> NDArray[Any]:
        """Tell which pixels fall inside ``[-0.5, size - 0.5)`` on both axes."""
        array = _as_pixels(pixels, finite=False)
        width, height = self._identity.image_size
        u, v = array[:, 0], array[:, 1]
        return (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)  # type: ignore[no-any-return]

    def max_ray_angle_rad(
        self, *, u_bounds: tuple[float, float], v_bounds: tuple[float, float]
    ) -> float:
        """Bound the ray angle over a pixel box, from its farthest corner and the domain.

        The farthest point of the box from the principal point on the normalized plane is a
        corner, since that plane is an axis-aligned scaling of the pixels. A ray landing in the
        box therefore has a distorted normalized radius of at most that corner's, ``rho``, and
        of at least the model's lower bound at its angle, so its angle lies where that bound
        is at most ``rho``, inside the viewing domain. Without tangential terms the lower bound
        is the exact radius, and the result is the angle of the farthest corner's ray, or the
        domain's limit when the box reaches past it.
        """
        np = lazy_numpy()
        bounds = np.array((u_bounds, v_bounds), dtype=np.float64)
        if not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
            raise ValueError(f"the pixel box must be finite and ordered, got {bounds.tolist()}")
        u_reach = float(np.abs((bounds[0] - self._cx) / self._fx).max())
        v_reach = float(np.abs((bounds[1] - self._cy) / self._fy).max())
        return sublevel_limit(
            self._radius_lower_bound, math.hypot(u_reach, v_reach), self._domain_limit_rad()
        )

    def _apply_intrinsics(self, xd: NDArray[Any], yd: NDArray[Any]) -> NDArray[Any]:
        """Apply focal lengths and principal point to normalized-plane coordinates."""
        np = lazy_numpy()
        pixels: NDArray[Any] = np.column_stack((self._fx * xd + self._cx, self._fy * yd + self._cy))
        return pixels

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Select nonzero rays inside the calibrated injective projection domain."""
        raise NotImplementedError

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Project camera-frame coordinates through the model and camera intrinsics."""
        raise NotImplementedError

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Invert normalized pixels into rays and report which inverses are valid."""
        raise NotImplementedError

    def _domain_limit_rad(self) -> float:
        """Supremum of the angle to the optical axis inside the viewing domain."""
        raise NotImplementedError

    def _radius_lower_bound(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Lower bound of the distorted normalized radius of every ray at ``angle``."""
        raise NotImplementedError
