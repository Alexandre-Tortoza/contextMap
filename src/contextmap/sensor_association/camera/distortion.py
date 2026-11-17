"""Numérica de distorção compartilhada pelos modelos de câmera.

Resolve domínio injetivo, inversão radial-tangencial e limites do raio normalizado.
Não conhece calibração, identidade de câmera ou pixels; NumPy é carregado sob demanda.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

COSINE_MARGIN = 1e-9
"""Margin, in cosine of the angle to the optical axis, kept from the domain limit.

It keeps the projected coordinates finite next to the boundary, where the
denominators of the models vanish.
"""

_UNDISTORT_ITERATIONS = 100
_UNDISTORT_STEP_TOLERANCE = 1e-13
"""Newton step, on the normalized plane, below which the inverse stops refining."""
_NEWTON_STEP_HALVINGS = 50
"""Halvings of a Newton step before the inverse gives up improving a pixel."""
_UNDISTORT_RESIDUAL_TOLERANCE = 1e-9
"""Normalized-plane residual above which an undistorted pixel is rejected as ray-less."""

_RADIUS_SCAN_STEPS = 31416
"""Samples of the angle scan for a radius growth limit.

About 1e-4 rad over the fisheye's ``[0, pi]`` and half that over the pinhole's ``[0, pi/2]``.
"""

_BISECTION_STEPS = 64

_ANGLE_BOUND_ULPS = 4
"""ULPs added to a ray-angle bound, so that rounding can only widen it and never narrow it."""


def lazy_numpy() -> Any:
    """Import NumPy on first use, so importing the capability contracts stays cheap.

    Returns:
        The ``numpy`` module.
    """
    import numpy as np

    return np


def growth_limit(growth: Callable[[NDArray[Any]], NDArray[Any]], upper: float) -> float | None:
    """Largest angle to the optical axis up to which a projection stays injective.

    Past the first zero of ``growth`` the projection folds: two rays share a pixel. The first
    non-positive sample of a scan over ``[0, upper]`` is refined by bisection, and the angle
    returned stays on the positive side.

    Args:
        growth: Positive exactly while the projection stays injective, as a function of the
            angle; positive at zero. The radius' slope for the fisheye and the MEI, the
            smallest Jacobian determinant around the circle for the pinhole.
        upper: The end of the scan, in radians.

    Returns:
        The limit angle, in radians, or ``None`` when ``growth`` never stops being positive.
    """
    np = lazy_numpy()
    angle = np.linspace(0.0, upper, _RADIUS_SCAN_STEPS)
    stalled = np.flatnonzero(growth(angle) <= 0)
    if stalled.size == 0:
        return None
    low, high = float(angle[stalled[0] - 1]), float(angle[stalled[0]])
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        if float(growth(np.array([middle]))[0]) > 0:
            low = middle
        else:
            high = middle
    return low


def angle_at_radius(
    radius_at: Callable[[NDArray[Any]], NDArray[Any]], radius: NDArray[Any], upper: float
) -> NDArray[Any]:
    """Invert, by bisection, a radius that grows with the angle to the optical axis.

    Bisection stays robust next to the end of the domain, where Newton would diverge: the
    radius' slope tends to zero at a fold and to infinity at a pole.

    Args:
        radius_at: The distorted normalized radius at an angle, increasing on ``[0, upper]``.
        radius: ``(N,)`` radii to reach.
        upper: The end of the viewing domain, in radians.

    Returns:
        ``(N,)`` angles in ``[0, upper]``; next to ``upper`` for a radius the domain does not
        reach, which the caller has to reject.
    """
    np = lazy_numpy()
    low = np.zeros_like(radius)
    high = np.full_like(radius, upper)
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        below = radius_at(middle) < radius
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    return (low + high) / 2  # type: ignore[no-any-return]


def tangential_reach(p1: float, p2: float) -> float:
    """Largest ``|t| / r^2`` of the OpenCV tangential term ``t`` at normalized radius ``r``.

    ``t = (2 p1 x y + p2 (r^2 + 2 x^2), p1 (r^2 + 2 y^2) + 2 p2 x y)``; since ``2 |x y| <= r^2``
    and ``r^2 + 2 x^2 <= 3 r^2``, its components are at most ``(|p1| + 3 |p2|) r^2`` and
    ``(3 |p1| + |p2|) r^2``.
    """
    return math.hypot(abs(p1) + 3 * abs(p2), 3 * abs(p1) + abs(p2))


def sublevel_limit(
    radius_at: Callable[[NDArray[Any]], NDArray[Any]], level: float, upper: float
) -> float:
    """Largest angle in ``[0, upper]`` whose radius lower bound does not exceed ``level``.

    A scan over ``[0, upper]`` finds the last sample at or below ``level`` -- angle zero always
    is -- and a bisection towards the next sample keeps its outer end, so the result can only
    err wide. The scan has the resolution of the domain limits: a lower bound that is not
    monotone could dip below ``level`` strictly between two samples, which it would not see.

    Args:
        radius_at: A lower bound of the distorted normalized radius of every ray at a given
            angle to the optical axis; zero at angle zero.
        level: The normalized radius to stay within.
        upper: The end of the model's viewing domain, in radians.

    Returns:
        The angle, in radians, with a few ULPs of conservative margin, never above ``upper``.
    """
    np = lazy_numpy()
    angle = np.linspace(0.0, upper, _RADIUS_SCAN_STEPS)
    # Perto do limite do domínio os denominadores se anulam: inf e NaN ficam fora do nível.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        last = int(np.flatnonzero(radius_at(angle) <= level)[-1])
        if last == angle.shape[0] - 1:
            return upper
        low, high = float(angle[last]), float(angle[last + 1])
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2
            if float(radius_at(np.array([middle]))[0]) <= level:
                low = middle
            else:
                high = middle
    return min(upper, high + _ANGLE_BOUND_ULPS * math.ulp(high))


def tangential_offset(
    p1: float, p2: float, x: NDArray[Any], y: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """The OpenCV tangential term at undistorted normalized ``(x, y)``, shared by CamOdoCal."""
    r2 = x * x + y * y
    return 2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def distortion_jacobian(
    scale: NDArray[Any],
    scale_slope: NDArray[Any],
    tangential: tuple[float, float],
    x: NDArray[Any],
    y: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Jacobian ``[[a, b], [b, c]]`` of ``(x, y) -> (x, y) g(r^2) + t(x, y)``.

    The map is the gradient of ``G(r^2) / 2 + r^2 (p2 x + p1 y)``, with ``G' = g``, so its
    Jacobian is that potential's Hessian, which is symmetric.

    Args:
        scale: The radial factor ``g`` at ``r^2 = x^2 + y^2``.
        scale_slope: Its derivative with respect to ``r^2``.
        tangential: ``(p1, p2)``.
        x: Undistorted normalized ``x``.
        y: Undistorted normalized ``y``.

    Returns:
        ``(a, b, c)``.
    """
    p1, p2 = tangential
    a = scale + 2 * x * x * scale_slope + 2 * p1 * y + 6 * p2 * x
    b = 2 * x * y * scale_slope + 2 * p1 * x + 2 * p2 * y
    c = scale + 2 * y * y * scale_slope + 6 * p1 * y + 2 * p2 * x
    return a, b, c


def smallest_jacobian_determinant(
    radial_slope: NDArray[Any], scale: NDArray[Any], twist: NDArray[Any]
) -> NDArray[Any]:
    """Smallest determinant of the radial-tangential Jacobian around a circle ``r``.

    In the frame of the radial and azimuthal directions of a point, the Jacobian of
    :func:`distortion_jacobian` is ``[[A + 6 q c, 2 q s], [2 q s, B + 2 q c]]``, with ``c``
    and ``s`` the cosine and sine of the angle from ``(p2, p1)`` to the point's direction. Its
    determinant ``16 q^2 c^2 + 2 q (A + 3 B) c + A B - 4 q^2`` is a convex quadratic in ``c``,
    whose minimum over ``[-1, 1]`` is at the clipped vertex.

    Args:
        radial_slope: ``A = d(r g)/dr``, the radial term's stretch along the radius.
        scale: ``B = g``, its stretch across the radius.
        twist: ``q = r hypot(p1, p2)``, the tangential term's reach.

    Returns:
        The smallest determinant over the circle; ``A B`` without tangential terms.
    """
    np = lazy_numpy()
    spread = radial_slope + 3 * scale
    vertex = np.divide(-spread, 16 * twist, out=np.zeros_like(twist), where=twist > 0)
    c = np.clip(vertex, -1.0, 1.0)
    squared_twist = twist * twist
    determinant: NDArray[Any] = (
        16 * squared_twist * c * c + 2 * twist * spread * c + radial_slope * scale
    ) - 4 * squared_twist
    return determinant


def undistort(
    radial_inverse: Callable[[NDArray[Any]], NDArray[Any]],
    distort: Callable[[NDArray[Any], NDArray[Any]], tuple[NDArray[Any], NDArray[Any]]],
    jacobian: Callable[
        [NDArray[Any], NDArray[Any]], tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
    ],
    max_radius: float,
    xd: NDArray[Any],
    yd: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Invert a radial-tangential distortion of the normalized plane inside the viewing domain.

    The start is the exact inverse of the radial term alone, which keeps the direction and grows
    over the domain: a bisection of the angle, robust next to a fold or a pole. Newton's method
    with the distortion's Jacobian then brings in the tangential term, each step halved until it
    stays in the domain and lowers the residual. The iteration therefore never lands on a branch
    past a fold or a pole, and still converges next to them, where a fixed point with an
    identity Jacobian stalls (fold) or diverges (pole).

    Args:
        radial_inverse: The undistorted radius, inside the domain, whose radial term reaches a
            distorted radius; the domain's limit for a radius it does not reach.
        distort: The whole distortion.
        jacobian: Its symmetric Jacobian, as :func:`distortion_jacobian` returns it.
        max_radius: The undistorted radius the domain stays short of.
        xd: Distorted normalized ``x``.
        yd: Distorted normalized ``y``.

    Returns:
        The undistorted coordinates and a mask of the pixels whose residual is
        within tolerance; the others have no trustworthy inverse.
    """
    np = lazy_numpy()
    rho = np.hypot(xd, yd)
    scale = np.divide(radial_inverse(rho), rho, out=np.zeros_like(rho), where=rho > 0)
    x, y = scale * xd, scale * yd
    fx_, fy_ = distort(x, y)
    rx, ry = fx_ - xd, fy_ - yd
    # Na borda do domínio o Jacobiano fica singular: um passo infinito ou NaN nunca melhora o
    # resíduo, então nunca é aceito.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(_UNDISTORT_ITERATIONS):
            residual = np.hypot(rx, ry)
            a, b, c = jacobian(x, y)
            determinant = a * c - b * b
            step_x = (b * ry - c * rx) / determinant
            step_y = (b * rx - a * ry) / determinant
            fraction = np.ones_like(residual)
            for _ in range(_NEWTON_STEP_HALVINGS):
                next_x, next_y = x + fraction * step_x, y + fraction * step_y
                fx_, fy_ = distort(next_x, next_y)
                next_rx, next_ry = fx_ - xd, fy_ - yd
                better = (np.hypot(next_x, next_y) < max_radius) & (
                    np.hypot(next_rx, next_ry) < residual
                )
                if better.all():
                    break
                fraction = np.where(better, fraction, fraction / 2)
            moved = np.where(better, fraction * np.hypot(step_x, step_y), 0.0)
            x, y = np.where(better, next_x, x), np.where(better, next_y, y)
            rx, ry = np.where(better, next_rx, rx), np.where(better, next_ry, ry)
            if not (moved >= _UNDISTORT_STEP_TOLERANCE).any():
                break
    converged = (np.abs(rx) < _UNDISTORT_RESIDUAL_TOLERANCE) & (
        np.abs(ry) < _UNDISTORT_RESIDUAL_TOLERANCE
    )
    return x, y, converged
