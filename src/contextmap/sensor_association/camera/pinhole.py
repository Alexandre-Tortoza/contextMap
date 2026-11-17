"""Modelo pinhole com distorção OpenCV radial-tangencial opcional.

Cobre ``plumb_bob`` e o modelo racional de oito coeficientes. O domínio de visão é o maior
disco em torno do eixo óptico em que a distorção inteira, termos tangenciais incluídos,
continua injetiva, e para o modelo racional ele termina antes de qualquer zero do denominador.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import CalibrationEntry, DistortionModel, PinholeCameraModel
from contextmap.sensor_association.camera.distortion import (
    COSINE_MARGIN,
    angle_at_radius,
    distortion_jacobian,
    growth_limit,
    lazy_numpy,
    smallest_jacobian_determinant,
    tangential_offset,
    tangential_reach,
    undistort,
)
from contextmap.sensor_association.camera.projection import CalibratedProjection

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DENOMINATOR_MARGIN = 1e-9
"""Smallest value of the rational pinhole's denominator ``D`` (``D(0) = 1``) kept in the domain.

Next to a zero of ``D`` the distorted radius ``r N / D`` grows without bound. Ending the domain
where ``D`` falls to the margin, on the near side of the pole, keeps that radius finite and keeps
the rounding of the projection's own evaluation of ``D`` far from crossing the pole.
"""


def _pinhole_coefficients(model: PinholeCameraModel) -> tuple[float, ...]:
    """Pad to OpenCV's ``(k1, k2, p1, p2, k3, k4, k5, k6)``."""
    coefficients = model.distortion_coefficients
    return (*coefficients, *([0.0] * (8 - len(coefficients))))


class PinholeProjection(CalibratedProjection):
    """Pinhole with optional OpenCV radial/tangential (plumb_bob or rational) distortion."""

    def __init__(self, entry: CalibrationEntry, model: PinholeCameraModel) -> None:
        """Build the projection of a validated pinhole calibration entry."""
        super().__init__(entry, model)
        self._distorted = model.distortion_model is not DistortionModel.NONE
        self._k = _pinhole_coefficients(model)
        _, _, p1, p2, *_ = self._k
        self._tangential_coefficients = (p1, p2)
        # Maior raio do plano normalizado (sem distorção) dentro do domínio; infinito quando
        # nada dobra nem há polo, e então o domínio é o hemisfério da frente, como no pinhole
        # ideal.
        self._max_radius = self._injective_radius() if self._distorted else math.inf

    def _injective_radius(self) -> float:
        """Radius of the widest disk around the axis on which the distortion is injective.

        The distortion ``F(x) = x N(s) / D(s) + t(x)``, ``s = |x|^2``, is a gradient, so its
        Jacobian ``J`` is symmetric (:func:`distortion_jacobian`). ``J`` is the identity on the
        axis; on a disk where ``det J > 0`` no eigenvalue crosses zero, so ``J`` stays positive
        definite, and ``(b - a) . (F(b) - F(a)) > 0`` for any two points of the disk: ``F`` is
        injective. Past the first radius where ``det J`` vanishes somewhere around the circle
        the map folds, and a point far off the axis lands back in the image, even on the
        principal point. Without tangential terms ``det J = g (r g)'``, whose first zero is
        where the radial term stops growing. At a zero of ``D``, a pole of the rational model,
        the distorted radius jumps from plus to minus infinity and a point past it lands
        mirrored; the disk stops where ``D`` falls to ``_DENOMINATOR_MARGIN``. The smallest
        ``det J`` around each circle has a closed form (:func:`smallest_jacobian_determinant`),
        and the scan runs over the angle to the optical axis, ``r = tan(angle)``.

        Returns:
            The limit radius, or ``inf`` when the distortion is injective over the whole front
            hemisphere.
        """
        np = lazy_numpy()
        twist = math.hypot(*self._tangential_coefficients)

        def growth(angle: NDArray[Any]) -> NDArray[Any]:
            """Evaluate the local growth criterion used to locate the injective domain limit."""
            r = np.tan(angle)
            s = r * r
            numerator, denominator, numerator_slope, denominator_slope = self._radial_terms(s)
            # Perto de 90 graus as potências de s podem estourar: inf e NaN não contam como dobra.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                slope_sign = numerator * denominator + 2 * s * (
                    numerator_slope * denominator - numerator * denominator_slope
                )
                determinant = smallest_jacobian_determinant(
                    slope_sign / (denominator * denominator), numerator / denominator, r * twist
                )
            return np.where(denominator > _DENOMINATOR_MARGIN, determinant, 0.0)  # type: ignore[no-any-return]

        limit = growth_limit(growth, math.pi / 2)
        return math.inf if limit is None else math.tan(limit)

    def _radial_terms(
        self, s: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """``N(s)``, ``D(s)`` and their derivatives, at squared undistorted radius ``s``."""
        k1, k2, _, _, k3, k4, k5, k6 = self._k
        numerator = 1 + s * (k1 + s * (k2 + s * k3))
        denominator = 1 + s * (k4 + s * (k5 + s * k6))
        numerator_slope = k1 + s * (2 * k2 + s * 3 * k3)
        denominator_slope = k4 + s * (2 * k5 + s * 3 * k6)
        return numerator, denominator, numerator_slope, denominator_slope

    def _radial_scale(self, r2: NDArray[Any]) -> NDArray[Any]:
        """The radial factor ``N(r^2) / D(r^2)`` at squared undistorted normalized radius ``r2``."""
        numerator, denominator, _, _ = self._radial_terms(r2)
        return numerator / denominator  # type: ignore[no-any-return]

    def _jacobian(
        self, x: NDArray[Any], y: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """The distortion's symmetric Jacobian at undistorted normalized ``(x, y)``."""
        numerator, denominator, numerator_slope, denominator_slope = self._radial_terms(
            x * x + y * y
        )
        scale_slope = (numerator_slope * denominator - numerator * denominator_slope) / (
            denominator * denominator
        )
        return distortion_jacobian(
            numerator / denominator, scale_slope, self._tangential_coefficients, x, y
        )

    def _radial_radius_at(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Distorted normalized radius of a ray at ``angle``, by the radial term alone."""
        r = lazy_numpy().tan(angle)
        return r * self._radial_scale(r * r)  # type: ignore[no-any-return]

    def _radial_inverse(self, radius: NDArray[Any]) -> NDArray[Any]:
        """Undistorted normalized radius in the domain whose radial term reaches ``radius``."""
        angle = angle_at_radius(self._radial_radius_at, radius, self._domain_limit_rad())
        return lazy_numpy().tan(angle)  # type: ignore[no-any-return]

    def _distort(self, x: NDArray[Any], y: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Apply the radial-tangential distortion on the normalized image plane."""
        radial = self._radial_scale(x * x + y * y)
        tx, ty = tangential_offset(*self._tangential_coefficients, x, y)
        return x * radial + tx, y * radial + ty

    def _in_domain(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        # z > margem * alcance também exclui a origem (0 > 0 é falso).
        """Select nonzero rays inside the calibrated injective projection domain."""
        in_front = z > COSINE_MARGIN * range_m
        if math.isinf(self._max_radius):
            return in_front  # type: ignore[no-any-return]
        # Raio x/z, y/z comparado sem dividir: z > 0 onde o ponto está na frente.
        return in_front & (lazy_numpy().hypot(x, y) < self._max_radius * z)  # type: ignore[no-any-return]

    def _domain_limit_rad(self) -> float:
        """Return the calibrated injective domain limit in radians."""
        return math.atan(self._max_radius)

    def _radius_lower_bound(self, angle: NDArray[Any]) -> NDArray[Any]:
        """Bound the normalized projected radius at each optical-axis angle."""
        np = lazy_numpy()
        r = np.tan(angle)
        if not self._distorted:
            return r  # type: ignore[no-any-return]
        reach = tangential_reach(*self._tangential_coefficients)
        return np.abs(self._radial_radius_at(angle)) - reach * (r * r)  # type: ignore[no-any-return]

    def _to_pixels(
        self, x: NDArray[Any], y: NDArray[Any], z: NDArray[Any], range_m: NDArray[Any]
    ) -> NDArray[Any]:
        """Project camera-frame coordinates through the model and camera intrinsics."""
        xn, yn = x / z, y / z
        if self._distorted:
            xn, yn = self._distort(xn, yn)
        return self._apply_intrinsics(xn, yn)

    def _to_rays(self, xd: NDArray[Any], yd: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Invert normalized pixels into rays and report which inverses are valid."""
        np = lazy_numpy()
        if self._distorted:
            xn, yn, converged = undistort(
                self._radial_inverse, self._distort, self._jacobian, self._max_radius, xd, yd
            )
            # Um pixel além do que o domínio alcança fica no limite do domínio, que project
            # recusa: ele não tem raio.
            has_ray = converged & (np.hypot(xn, yn) < self._max_radius)
        else:
            xn, yn, has_ray = xd, yd, np.ones(xd.shape, dtype=bool)
        rays = np.column_stack((xn, yn, np.ones_like(xn)))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True), has_ray
