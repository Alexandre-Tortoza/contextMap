"""Modelo fisheye equidistante (Kannala-Brandt).

O raio distorcido é ``theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)`` para o ângulo
``theta`` ao eixo óptico. O domínio de visão termina onde esse raio para de crescer: passado
esse ponto dois ângulos compartilham o mesmo pixel e a projeção deixa de ser injetiva. Não há
termo tangencial.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import CalibrationEntry, FisheyeCameraModel
from contextmap.sensor_association.camera.distortion import (
    angle_at_radius,
    growth_limit,
    lazy_numpy,
)
from contextmap.sensor_association.camera.projection import CalibratedProjection

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FisheyeProjection(CalibratedProjection):
    """Equidistant (Kannala-Brandt) fisheye.

    The radius is ``theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)`` for the
    angle ``theta`` to the optical axis.
    """

    def __init__(self, entry: CalibrationEntry, model: FisheyeCameraModel) -> None:
        """Build the projection of a validated fisheye calibration entry."""
        super().__init__(entry, model)
        self._k = model.distortion_coefficients
        self._theta_max = self._radius_growth_limit()
        self._theta_d_max = float(self._theta_d(lazy_numpy().array([self._theta_max]))[0])

    def _theta_d(self, theta: NDArray[Any]) -> NDArray[Any]:
        """Evaluate the equidistant radial distortion polynomial at each ray angle."""
        k1, k2, k3, k4 = self._k
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))  # type: ignore[no-any-return]

    def _radius_growth_limit(self) -> float:
        """Largest angle up to which the radius keeps growing with the angle.

        ``d(theta_d)/d(theta) = 1 + 3 k1 t^2 + 5 k2 t^4 + 7 k3 t^6 + 9 k4 t^8``; past its
        first zero two angles share a radius, so the projection is no longer injective.
        Falls back to ``pi`` when it never vanishes in ``(0, pi)``.
        """
        k1, k2, k3, k4 = self._k

        def growth(theta: NDArray[Any]) -> NDArray[Any]:
            """Evaluate the local growth criterion used to locate the injective domain limit."""
            t2 = theta * theta
            return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))  # type: ignore[no-any-return]

        limit = growth_limit(growth, math.pi)
        return math.pi if limit is None else limit

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Select nonzero rays inside the calibrated injective projection domain."""
        np = lazy_numpy()
        theta = np.arctan2(np.hypot(x, y), z)
        return (range_m > 0) & (theta < self._theta_max)  # type: ignore[no-any-return]

    def _domain_limit_rad(self) -> float:
        """Return the calibrated injective domain limit in radians."""
        return self._theta_max

    def _radius_lower_bound(self, angle: NDArray[Any]) -> NDArray[Any]:
        # Sem termo tangencial, o raio normalizado é exatamente theta_d.
        """Bound the normalized projected radius at each optical-axis angle."""
        return self._theta_d(angle)

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Project camera-frame coordinates through the model and camera intrinsics."""
        np = lazy_numpy()
        rho = np.hypot(x, y)
        theta_d = self._theta_d(np.arctan2(rho, z))
        # Sobre o eixo (rho = 0) a direção é indefinida e o raio do pixel é zero.
        scale = np.divide(theta_d, rho, out=np.zeros_like(rho), where=rho > 0)
        return self._apply_intrinsics(scale * x, scale * y)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Invert normalized pixels into rays and report which inverses are valid."""
        np = lazy_numpy()
        radius = np.hypot(xd, yd)
        has_ray = radius <= self._theta_d_max
        theta = angle_at_radius(self._theta_d, radius, self._theta_max)
        scale = np.divide(np.sin(theta), radius, out=np.zeros_like(radius), where=radius > 0)
        return np.column_stack((scale * xd, scale * yd, np.cos(theta))), has_ray
