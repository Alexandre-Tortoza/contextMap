"""Modelo omnidirecional unificado (Geyer-Daniilidis / Mei), na parametrização do CamOdoCal.

O domínio de visão termina no horizonte da esfera do modelo ou, se vier antes, no ângulo em que
a distorção radial para de crescer. Os termos tangenciais são confiados dentro desse domínio.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import CalibrationEntry, MeiCameraModel
from contextmap.sensor_association.camera.distortion import (
    COSINE_MARGIN,
    angle_at_radius,
    distortion_jacobian,
    growth_limit,
    lazy_numpy,
    tangential_offset,
    tangential_reach,
    undistort,
)
from contextmap.sensor_association.camera.projection import CalibratedProjection

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MeiProjection(CalibratedProjection):
    """Unified omnidirectional model as parameterized by CamOdoCal (Geyer-Daniilidis / Mei)."""

    def __init__(self, entry: CalibrationEntry, model: MeiCameraModel) -> None:
        """Build the projection of a validated MEI calibration entry."""
        super().__init__(entry, model)
        self._xi = model.xi
        self._k1, self._k2, self._p1, self._p2 = model.distortion_coefficients
        # Cosseno mínimo do ângulo com o eixo: acima do horizonte da esfera (xi > 1) a
        # projeção dobra de volta; para xi < 1 o denominador z + xi * alcance zera antes.
        self._horizon_cosine = -min(self._xi, 1 / self._xi) if self._xi > 0 else 0.0
        # A dobra da distorção, quando vem antes do horizonte, encerra o domínio sem margem:
        # nela o raio distorcido é finito.
        self._fold_rad = self._distortion_fold()
        self._cosine_bound = self._horizon_cosine + COSINE_MARGIN
        if self._fold_rad is not None:
            self._cosine_bound = max(self._cosine_bound, math.cos(self._fold_rad))
        self._max_plane_radius = float(
            self._plane_radius_at(lazy_numpy().array([self._domain_limit_rad()]))[0]
        )

    def _plane_radius_at(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Normalized-plane radius ``sin / (cos + xi)`` of the sphere point at ``angle``.

        It grows with the angle up to the horizon. There, for ``xi <= 1``, the denominator
        vanishes and rounding can leave it negative: the radius is unbounded, never a negative
        value that would fit in any box.
        """
        np = lazy_numpy()
        shifted = np.cos(angle) + self._xi
        return np.divide(  # type: ignore[no-any-return]
            np.sin(angle), shifted, out=np.full_like(shifted, np.inf), where=shifted > 0
        )

    def _radial_radius_at(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Distorted normalized radius of a ray at ``angle``, by the radial term alone."""
        m = self._plane_radius_at(angle)
        m2 = m * m
        return m * (1 + m2 * (self._k1 + m2 * self._k2))  # type: ignore[no-any-return]

    def _distortion_fold(self) -> float | None:
        """Angle to the optical axis, before the horizon, where the distorted radius stops growing.

        The radial term maps the normalized-plane radius ``m`` to ``m (1 + k1 m^2 + k2 m^4)``,
        whose derivative is ``1 + 3 k1 m^2 + 5 k2 m^4``; past its first zero two angles share a
        distorted radius and a point far off the axis folds back into the image, even next to
        the principal point. Since ``m`` grows with the angle up to the horizon, the scan runs
        over the angle.

        Returns:
            The fold angle, in radians, or ``None`` when the radius grows up to the horizon.
        """
        np = lazy_numpy()

        def growth(angle: NDArray[Any]) -> NDArray[Any]:
            """Evaluate the local growth criterion used to locate the injective domain limit."""
            m2 = self._plane_radius_at(angle) ** 2
            # No horizonte com xi <= 1, m é infinito: 0 * inf vira NaN, que não conta como dobra.
            with np.errstate(invalid="ignore", over="ignore"):
                return 1 + m2 * (3 * self._k1 + m2 * 5 * self._k2)  # type: ignore[no-any-return]

        return growth_limit(growth, math.acos(self._horizon_cosine))

    def _radial_inverse(self, radius: NDArray[Any]) -> NDArray[Any]:
        """Normalized-plane radius, inside the domain, whose radial term reaches ``radius``."""
        angle = angle_at_radius(self._radial_radius_at, radius, self._domain_limit_rad())
        return self._plane_radius_at(angle)

    def _distort(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Apply the radial-tangential distortion on the normalized image plane."""
        r2 = x * x + y * y
        radial = self._k1 * r2 + self._k2 * r2 * r2
        tx, ty = tangential_offset(self._p1, self._p2, x, y)
        return x + (x * radial + tx), y + (y * radial + ty)

    def _jacobian(
        self, x: NDArray[Any], y: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """The distortion's symmetric Jacobian at undistorted normalized ``(x, y)``."""
        r2 = x * x + y * y
        scale = 1 + r2 * (self._k1 + r2 * self._k2)
        scale_slope = self._k1 + 2 * self._k2 * r2
        return distortion_jacobian(scale, scale_slope, (self._p1, self._p2), x, y)

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Select nonzero rays inside the calibrated injective projection domain."""
        return z > self._cosine_bound * range_m  # type: ignore[no-any-return]

    def _domain_limit_rad(self) -> float:
        """Return the calibrated injective domain limit in radians."""
        return math.acos(self._horizon_cosine) if self._fold_rad is None else self._fold_rad

    def _radius_lower_bound(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Bound the normalized projected radius at each optical-axis angle."""
        m = self._plane_radius_at(angle)
        reach = tangential_reach(self._p1, self._p2)
        return lazy_numpy().abs(self._radial_radius_at(angle)) - reach * (m * m)  # type: ignore[no-any-return]

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Project camera-frame coordinates through the model and camera intrinsics."""
        shifted = z + self._xi * range_m
        xd, yd = self._distort(x / shifted, y / shifted)
        return self._apply_intrinsics(xd, yd)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Invert normalized pixels into rays and report which inverses are valid."""
        np = lazy_numpy()
        x, y, converged = undistort(
            self._radial_inverse, self._distort, self._jacobian, self._max_plane_radius, xd, yd
        )
        rho2 = x * x + y * y
        # O inverso radial leva um pixel além do raio que o domínio alcança (horizonte ou
        # dobra) até o limite do domínio, que project recusa: ele não tem raio.
        has_ray = converged & (np.sqrt(rho2) < self._max_plane_radius)
        discriminant = 1 + (1 - self._xi * self._xi) * rho2
        scale = (self._xi + np.sqrt(np.maximum(discriminant, 0.0))) / (1 + rho2)
        return np.column_stack((scale * x, scale * y, scale - self._xi)), has_ray
