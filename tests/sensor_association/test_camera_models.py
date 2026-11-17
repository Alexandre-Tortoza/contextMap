import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

import contextmap.sensor_association.camera as camera
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationReferenceId,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    FrameId,
    MeiCameraModel,
    PinholeCameraModel,
    SensorId,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.sensor_association import (
    CameraIdentity,
    CameraProjection,
    PixelProjection,
    camera_projection_for,
)
from contextmap.sensor_association.camera.fisheye import FisheyeProjection
from contextmap.sensor_association.camera.mei import MeiProjection
from contextmap.sensor_association.camera.pinhole import PinholeProjection

FX, FY, CX, CY = 500.0, 480.0, 320.0, 240.0
WIDTH, HEIGHT = 640, 480
PLUMB_BOB = (-0.28, 0.07, 0.0002, -0.0001, 0.0)
RATIONAL = (-0.3, 0.09, 0.0003, -0.0002, -0.01, 0.02, 0.005, -0.001)
FISHEYE_K = (0.01, -0.004, 0.0007, -0.0001)
MEI_K = (-0.0708, 0.3095, 0.00046, 0.000157)
MEI_XI = 1.563


def _entry(model: CameraModel | None, *, name: str = "camera") -> CalibrationEntry:
    return CalibrationEntry(
        calibration_id=CalibrationReferenceId(f"{name}-calib"),
        sensor_id=SensorId(name),
        frame_id=FrameId(f"{name}_optical"),
        camera_model=model,
        provenance=CalibrationProvenance(source_type="dataset", source_path="fixtures/cal.yaml"),
        content_hash=compute_content_hash(
            sensor_id=SensorId(name), frame_id=FrameId(f"{name}_optical"), camera_model=model
        ),
    )


def _pinhole_model(
    model: DistortionModel = DistortionModel.NONE, coefficients: tuple[float, ...] = ()
) -> PinholeCameraModel:
    return PinholeCameraModel(
        width=WIDTH,
        height=HEIGHT,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        distortion_model=model,
        distortion_coefficients=coefficients,
    )


def _fisheye_model(k: tuple[float, float, float, float] = FISHEYE_K) -> FisheyeCameraModel:
    return FisheyeCameraModel(
        width=WIDTH, height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY, distortion_coefficients=k
    )


def _mei_model(xi: float = MEI_XI, k: tuple[float, float, float, float] = MEI_K) -> MeiCameraModel:
    return MeiCameraModel(
        width=WIDTH, height=HEIGHT, fx=FX, fy=FY, cx=CX, cy=CY, xi=xi, distortion_coefficients=k
    )


def _points(*rows: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.array(rows, dtype=np.float64)


def _at_angle(
    angle_deg: float, *, range_m: float = 2.0, azimuth_deg: float = 0.0
) -> tuple[float, float, float]:
    """A point ``angle_deg`` off the optical axis, ``azimuth_deg`` from ``x`` towards ``y``."""
    theta, azimuth = math.radians(angle_deg), math.radians(azimuth_deg)
    off_axis = range_m * math.sin(theta)
    return (off_axis * math.cos(azimuth), off_axis * math.sin(azimuth), range_m * math.cos(theta))


# --- Trusted formulas, written scalar and independent of the implementation ---


def _reference_distortion(k: tuple[float, ...], x: Any, y: Any) -> tuple[Any, Any]:
    """OpenCV plumb_bob / rational_polynomial on the normalized plane: (k1, k2, p1, p2, k3, ...)."""
    padded = (*k, *([0.0] * (8 - len(k))))
    k1, k2, p1, p2, k3, k4, k5, k6 = padded
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def _reference_pinhole(
    p: tuple[float, float, float], k: tuple[float, ...] = (0.0,) * 8
) -> tuple[float, float]:
    """OpenCV pinhole through :func:`_reference_distortion`."""
    xd, yd = _reference_distortion(k, p[0] / p[2], p[1] / p[2])
    return FX * xd + CX, FY * yd + CY


def _reference_fisheye(
    p: tuple[float, float, float], k: tuple[float, float, float, float]
) -> tuple[float, float]:
    """OpenCV equidistant model, valid for any direction (theta from the axis)."""
    rho = math.hypot(p[0], p[1])
    theta = math.atan2(rho, p[2])
    theta_d = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    if rho == 0:
        return CX, CY
    return FX * theta_d * p[0] / rho + CX, FY * theta_d * p[1] / rho + CY


def _reference_mei(
    p: tuple[float, float, float], xi: float, k: tuple[float, float, float, float]
) -> tuple[float, float]:
    """CamOdoCal CataCamera::spaceToPlane."""
    z = p[2] + xi * math.sqrt(sum(c * c for c in p))
    x, y = p[0] / z, p[1] / z
    k1, k2, p1, p2 = k
    r2 = x * x + y * y
    radial = k1 * r2 + k2 * r2 * r2
    xd = x + x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y + y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return FX * xd + CX, FY * yd + CY


# --- Exact, hand-checkable cases --------------------------------------------


def test_an_ideal_pinhole_projects_by_similar_triangles() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    result = projection.project(_points((0.5, -0.25, 2.0), (0.0, 0.0, 3.0)))

    np.testing.assert_allclose(result.pixels, [[445.0, 180.0], [320.0, 240.0]], atol=1e-12)
    assert result.projectable.tolist() == [True, True]
    np.testing.assert_allclose(result.depth_m, [2.0, 3.0])
    np.testing.assert_allclose(result.range_m, [math.sqrt(4.3125), 3.0])


def test_an_ideal_equidistant_camera_maps_the_angle_linearly_to_the_radius() -> None:
    projection = camera_projection_for(_entry(_fisheye_model((0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)))

    np.testing.assert_allclose(result.pixels[0], [FX * math.pi / 4 + CX, CY], atol=1e-9)
    np.testing.assert_allclose(result.pixels[1], [CX, FY * math.pi / 2 + CY], atol=1e-9)


def test_a_unified_model_with_no_mirror_shift_is_a_pinhole() -> None:
    unified = camera_projection_for(_entry(_mei_model(xi=0.0, k=(0.0, 0.0, 0.0, 0.0))))
    pinhole = camera_projection_for(_entry(_pinhole_model()))
    points = _points((0.5, -0.25, 2.0), (-1.0, 0.7, 4.0))

    np.testing.assert_allclose(unified.project(points).pixels, pinhole.project(points).pixels)


def test_a_unified_model_with_a_unit_mirror_shift_projects_stereographically() -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)))

    tan_eighth = math.tan(math.pi / 8)
    np.testing.assert_allclose(
        result.pixels,
        [[FX + CX, CY], [FX * tan_eighth + CX, CY], [CX, CY]],
        atol=1e-9,
    )


# --- Agreement with the published formulas ----------------------------------

_FRONT_POINTS = [
    (0.3, -0.2, 1.5),
    (-1.1, 0.6, 2.4),
    (0.9, 0.8, 1.2),
    (-0.05, 0.02, 0.7),
    (2.0, -1.5, 3.0),
]


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [
        (DistortionModel.NONE, ()),
        (DistortionModel.PLUMB_BOB, PLUMB_BOB),
        (DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL),
    ],
)
def test_the_pinhole_matches_the_opencv_distortion_formulas(
    model: DistortionModel, coefficients: tuple[float, ...]
) -> None:
    projection = camera_projection_for(_entry(_pinhole_model(model, coefficients)))

    result = projection.project(_points(*_FRONT_POINTS))

    expected = [_reference_pinhole(p, coefficients) for p in _FRONT_POINTS]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)


def test_the_fisheye_matches_the_opencv_equidistant_formula_even_beyond_ninety_degrees() -> None:
    projection = camera_projection_for(_entry(_fisheye_model()))
    points = [*_FRONT_POINTS, _at_angle(95.0), _at_angle(-110.0)]

    result = projection.project(_points(*points))

    expected = [_reference_fisheye(p, FISHEYE_K) for p in points]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)
    assert result.projectable.all()


def test_the_unified_model_matches_the_camodocal_formula_beyond_ninety_degrees() -> None:
    projection = camera_projection_for(_entry(_mei_model()))
    points = [*_FRONT_POINTS, _at_angle(95.0), _at_angle(-120.0)]

    result = projection.project(_points(*points))

    expected = [_reference_mei(p, MEI_XI, MEI_K) for p in points]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)
    assert result.projectable.all()


# --- The viewing domain is explicit, never clipped --------------------------


def test_a_pinhole_cannot_project_what_is_beside_or_behind_the_camera() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    result = projection.project(
        _points(
            (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1e-12), (1.0, 0.0, 1e-3)
        )
    )

    assert result.projectable.tolist() == [False, False, False, False, True]
    assert np.isnan(result.pixels[:4]).all()
    assert np.isfinite(result.pixels[4]).all()
    np.testing.assert_allclose(result.depth_m, [-1.0, 0.0, 0.0, 1e-12, 1e-3])


def test_a_distorted_pinhole_stops_where_the_distorted_radius_stops_growing() -> None:
    # d(r (1 + k1 r^2))/dr = 1 + 3 k1 r^2 se anula em r = sqrt(1 / (-3 k1)), cerca de 47,5 graus.
    k1 = -0.28
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (k1, 0.0, 0.0, 0.0, 0.0)))
    )
    limit_deg = math.degrees(math.atan(math.sqrt(1 / (-3 * k1))))

    result = projection.project(_points(_at_angle(limit_deg - 1.0), _at_angle(limit_deg + 1.0)))

    assert result.projectable.tolist() == [True, False]
    assert np.isnan(result.pixels[1]).all()


def test_a_point_beyond_the_fold_never_lands_on_the_principal_point() -> None:
    """The audit's reproduction: past the fold the radius shrinks back to zero.

    With ``k1 = -0.28`` alone, ``r (1 + k1 r^2)`` returns to zero at ``r = sqrt(1 / 0.28)``,
    about 62.1 degrees off the axis, so the unguarded polynomial sent that point exactly to the
    principal point, inside the image.
    """
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.0, 0.0, 0.0, 0.0)))
    )
    folded_deg = math.degrees(math.atan(math.sqrt(1 / 0.28)))

    result = projection.project(_points(_at_angle(folded_deg)))

    assert not result.projectable[0]
    assert np.isnan(result.pixels).all()
    assert not projection.in_image(result.pixels)[0]


def test_a_distorted_pinhole_gives_no_ray_beyond_its_fold() -> None:
    """``unproject`` must stay inside the same domain ``project`` accepts.

    With ``k2 > 0`` the distorted radius grows again past the fold, so a pixel beyond the
    fold's peak has a preimage only on that outer branch; returning it would hand out a ray
    that ``project`` refuses.
    """
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.03, 0.0, 0.0, 0.0)))
    )

    with pytest.raises(ValueError, match="no viewing ray"):
        projection.unproject(np.array([[CX + 0.9 * FX, CY]]))


RATIONAL_POLE = (0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
"""Denominator ``1 - r^2``: a pole at ``r = 1``, 45 degrees off the axis, and no fold at all."""


def _rational_pinhole(coefficients: tuple[float, ...] = RATIONAL_POLE) -> CameraProjection:
    return camera_projection_for(
        _entry(_pinhole_model(DistortionModel.RATIONAL_POLYNOMIAL, coefficients))
    )


def _first_positive_root(polynomial: Polynomial) -> float:
    roots = polynomial.roots()
    real = roots[np.abs(roots.imag) < 1e-12].real
    positive = real[real > 0]
    return float(positive.min()) if positive.size else math.inf


def _rational_domain_limit_deg(coefficients: tuple[float, ...]) -> float:
    """The first zero of ``D`` or of ``d(r N / D)/dr``, from polynomial roots in ``s = r^2``."""
    k1, k2, _, _, k3, k4, k5, k6 = coefficients
    numerator, denominator = Polynomial([1, k1, k2, k3]), Polynomial([1, k4, k5, k6])
    s = Polynomial([0, 1])
    growth = numerator * denominator + 2 * s * (
        numerator.deriv() * denominator - numerator * denominator.deriv()
    )
    limit_s = min(_first_positive_root(denominator), _first_positive_root(growth))
    return math.degrees(math.atan(math.sqrt(limit_s)))


def test_a_point_past_the_pole_of_a_rational_pinhole_never_lands_in_the_image() -> None:
    """The audit's reproduction: past the pole the radial term changes sign.

    With ``k4 = -1`` alone, ``D = 1 - r^2`` vanishes 45 degrees off the axis while
    ``d(r / D)/dr`` stays positive, so the fold never narrows the domain. At 70 degrees
    ``r / (1 - r^2) = -0.42``: the point landed inside the image, mirrored across the
    principal point.
    """
    projection = _rational_pinhole()

    result = projection.project(_points(_at_angle(45.0 + 1e-7), _at_angle(45.1), _at_angle(70.0)))

    assert result.projectable.tolist() == [False, False, False]
    assert np.isnan(result.pixels).all()
    assert not projection.in_image(result.pixels).any()


def test_a_point_just_short_of_the_pole_projects_far_out_and_back() -> None:
    projection = _rational_pinhole()
    near = _at_angle(44.9)

    result = projection.project(_points(near))

    assert result.projectable[0]
    np.testing.assert_allclose(result.pixels[0], _reference_pinhole(near, RATIONAL_POLE))
    assert (result.pixels[0, 0] - CX) / FX > 143.0
    ray = projection.unproject(result.pixels)
    np.testing.assert_allclose(ray[0], np.array(near) / 2.0, atol=1e-12)


def test_the_domain_stops_short_of_the_pole_while_the_radius_is_finite() -> None:
    """The domain ends where ``D`` falls to ``1e-9``: radius about ``1e9``, never the pole."""
    projection = _rational_pinhole()
    angles = [45.0 - 10.0**-exponent for exponent in range(1, 15)]

    result = projection.project(_points(*(_at_angle(angle) for angle in angles)))

    radius = (result.pixels[result.projectable, 0] - CX) / FX
    assert result.projectable[0]
    assert not result.projectable[-1]
    assert ((radius > 0) & (radius < 1.1e9)).all()


def test_no_pixel_of_a_rational_pinhole_unprojects_past_its_pole() -> None:
    """Past the pole ``r / (1 - r^2)`` climbs back from minus infinity, mirrored.

    So the pixel ``(0, 24)``, in the image, is also the image of a ray about 61 degrees off the
    axis on the opposite side; handing out that ray gives a point ``project`` must refuse.
    """
    projection = _rational_pinhole()
    pixel = np.array([[0.0, 24.0]])

    ray = projection.unproject(pixel)

    assert _ray_angle(ray)[0] < math.pi / 4
    np.testing.assert_allclose(projection.project(ray * 3.0).pixels, pixel, atol=1e-6)


@pytest.mark.parametrize(
    "coefficients",
    [
        # Dobra em 63,7 graus antes do polo em 74,3 graus: o comportamento da SA-03.
        (-0.3, 0.09, 0.0, 0.0, -0.01, 0.02, 0.005, -0.001),
        # Polo em 45 graus, e a expressão de crescimento só se anula além dele, em 60,4 graus.
        (0.0, 0.1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        RATIONAL_POLE,
    ],
    ids=["fold-first", "pole-first", "pole-only"],
)
def test_a_rational_pinhole_ends_at_its_pole_or_its_fold_whichever_comes_first(
    coefficients: tuple[float, ...],
) -> None:
    projection = _rational_pinhole(coefficients)
    limit_deg = _rational_domain_limit_deg(coefficients)

    result = projection.project(_points(_at_angle(limit_deg - 0.5), _at_angle(limit_deg + 0.5)))
    domain = projection.max_ray_angle_rad(**_EVERYWHERE)

    assert result.projectable.tolist() == [True, False]
    assert domain == pytest.approx(math.radians(limit_deg), abs=1e-6)


TANGENTIAL = (0.0, 0.0, 0.05, 0.0, 0.0)
"""``p1`` alone: towards ``-y`` the map is ``y + 3 p1 y^2``, which comes back to zero."""

TANGENTIAL_FOLD_DEG = math.degrees(math.atan(1 / (6 * 0.05)))
"""73.3 degrees: towards ``-y`` the Jacobian is ``diag(1 - 2 p1 r, 1 - 6 p1 r)``."""

MIXED = (-0.28, 0.0, 0.01, 0.005, 0.0)
"""Radial fold at 47.5 degrees, which the tangential terms bring about one degree closer."""


def _smallest_reference_determinant(
    coefficients: tuple[float, ...], x: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """Smallest ``det J`` over the points, ``J`` by central differences of the reference."""
    step = 1e-7
    right = _reference_distortion(coefficients, x + step, y)
    left = _reference_distortion(coefficients, x - step, y)
    up = _reference_distortion(coefficients, x, y + step)
    down = _reference_distortion(coefficients, x, y - step)
    dx = [(a - b) / (2 * step) for a, b in zip(right, left, strict=True)]
    dy = [(a - b) / (2 * step) for a, b in zip(up, down, strict=True)]
    return float((dx[0] * dy[1] - dy[0] * dx[1]).min())


def _jacobian_fold_deg(coefficients: tuple[float, ...]) -> float:
    """First angle off the axis whose circle holds a point with ``det J <= 0``, else 90 degrees.

    Around 3600 azimuths; a scan in half degrees is refined by bisection.
    """
    azimuth = np.linspace(0.0, 2 * math.pi, 3600, endpoint=False)

    def folded(angle_deg: float) -> bool:
        radius = math.tan(math.radians(angle_deg))
        x, y = radius * np.cos(azimuth), radius * np.sin(azimuth)
        return _smallest_reference_determinant(coefficients, x, y) <= 0

    first = next((a for a in np.arange(0.5, 90.0, 0.5) if folded(float(a))), None)
    if first is None:
        return 90.0
    low, high = float(first) - 0.5, float(first)
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (low, middle) if folded(middle) else (middle, high)
    return high


def test_a_point_past_the_tangential_fold_never_lands_in_the_image() -> None:
    """The audit's reproduction: past its fold the tangential term brings a ray back to the center.

    Towards ``-y``, ``y + 3 p1 y^2`` returns to zero at ``y = -1 / (3 p1)``: that point, 81.5
    degrees off the axis, landed exactly on the principal point.
    """
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, TANGENTIAL))
    )

    result = projection.project(
        _points(
            _at_angle(TANGENTIAL_FOLD_DEG + 0.5, azimuth_deg=-90.0), (0.0, -1 / (3 * 0.05), 1.0)
        )
    )

    assert result.projectable.tolist() == [False, False]
    assert np.isnan(result.pixels).all()
    assert not projection.in_image(result.pixels).any()


@pytest.mark.parametrize(
    ("model", "coefficients", "fold_deg"),
    [
        (DistortionModel.PLUMB_BOB, TANGENTIAL, TANGENTIAL_FOLD_DEG),
        (DistortionModel.PLUMB_BOB, MIXED, _jacobian_fold_deg(MIXED)),
        (DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL, _jacobian_fold_deg(RATIONAL)),
        (DistortionModel.PLUMB_BOB, PLUMB_BOB, 90.0),
    ],
    ids=["tangential", "mixed", "rational", "plumb_bob"],
)
def test_the_pinhole_domain_ends_where_the_distortion_jacobian_first_degenerates(
    model: DistortionModel, coefficients: tuple[float, ...], fold_deg: float
) -> None:
    """The domain is the widest disk around the axis on which ``det J > 0``, tangential included.

    There ``J``, symmetric, stays positive definite, which makes the distortion injective. A
    normal calibration keeps its domain: none for plumb_bob, and for the rational fixture the
    radial fold of 63.73 degrees moves by two hundredths of a degree.
    """
    projection = camera_projection_for(_entry(_pinhole_model(model, coefficients)))

    domain = projection.max_ray_angle_rad(**_EVERYWHERE)

    assert domain == pytest.approx(math.radians(fold_deg), abs=1e-5)


def test_a_pinhole_with_strong_tangential_terms_round_trips_up_to_its_fold() -> None:
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, TANGENTIAL))
    )
    near = [
        _at_angle(TANGENTIAL_FOLD_DEG - 0.5, azimuth_deg=azimuth)
        for azimuth in (-90.0, -45.0, 0.0, 90.0, 180.0)
    ]

    result = projection.project(_points(*near))
    rays = projection.unproject(result.pixels)
    on_axis = projection.unproject(np.array([[CX, CY]]))

    assert result.projectable.all()
    np.testing.assert_allclose(rays, _points(*near) / 2.0, atol=1e-9)
    np.testing.assert_allclose(on_axis, [[0.0, 0.0, 1.0]], atol=1e-12)


def test_a_fisheye_stops_where_the_radius_stops_growing_with_the_angle() -> None:
    # d(theta_d)/d(theta) = 1 + 3 k1 theta^2 se anula em theta = sqrt(1 / (-3 k1)).
    projection = camera_projection_for(_entry(_fisheye_model((-0.3, 0.0, 0.0, 0.0))))
    limit_deg = math.degrees(math.sqrt(1 / 0.9))

    result = projection.project(_points(_at_angle(limit_deg - 1.0), _at_angle(limit_deg + 1.0)))

    assert result.projectable.tolist() == [True, False]
    assert np.isnan(result.pixels[1]).all()


def test_an_undistorted_fisheye_sees_everything_except_the_exact_opposite_direction() -> None:
    projection = camera_projection_for(_entry(_fisheye_model((0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(179.0), (0.0, 0.0, -1.0)))

    assert result.projectable.tolist() == [True, False]


@pytest.mark.parametrize(
    ("xi", "inside_deg", "outside_deg"),
    [
        # xi > 1: a projeção dobra de volta além de acos(-1 / xi), cerca de 129,8 graus.
        (MEI_XI, 128.8, 130.8),
        # xi < 1: o denominador de normalização se anula em acos(-xi).
        (0.5, 119.0, 121.0),
        (0.0, 89.0, 91.0),
    ],
)
def test_a_unified_model_stops_at_its_horizon(
    xi: float, inside_deg: float, outside_deg: float
) -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=xi, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(inside_deg), _at_angle(outside_deg)))

    assert result.projectable.tolist() == [True, False]
    assert np.isfinite(result.pixels[0]).all()
    assert np.isnan(result.pixels[1]).all()


MEI_FOLD = (-0.3, 0.0, 0.0, 0.0)
"""With ``xi = 1``, ``m = tan(angle / 2)`` and ``m (1 - 0.3 m^2)`` folds at ``m^2 = 1 / 0.9``."""

MEI_FOLD_DEG = math.degrees(2 * math.atan(math.sqrt(1 / 0.9)))
"""About 93.0 degrees, far inside the 180 degrees of the horizon at ``xi = 1``."""


def test_a_point_past_the_fold_of_a_unified_model_never_lands_in_the_image() -> None:
    """The audit's reproduction: past the fold the distorted radius shrinks back.

    At 120 degrees ``m = tan(60) = 1.73`` and ``m (1 - 0.3 m^2) = 0.17``: the point landed near
    the principal point, although it is 120 degrees off the optical axis.
    """
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=MEI_FOLD)))

    result = projection.project(
        _points(_at_angle(MEI_FOLD_DEG + 0.5), _at_angle(120.0), _at_angle(150.0))
    )

    assert result.projectable.tolist() == [False, False, False]
    assert np.isnan(result.pixels).all()
    assert not projection.in_image(result.pixels).any()


def test_a_unified_model_projects_as_before_short_of_its_fold() -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=MEI_FOLD)))
    points = [_at_angle(80.0), _at_angle(MEI_FOLD_DEG - 0.5), _at_angle(-60.0)]

    result = projection.project(_points(*points))

    assert result.projectable.all()
    expected = [_reference_mei(p, 1.0, MEI_FOLD) for p in points]
    np.testing.assert_allclose(result.pixels, expected, atol=1e-9)


def test_a_unified_model_unprojects_up_to_its_fold_and_no_further() -> None:
    """Every pixel inside the fold's circle has a ray short of the fold, and none past it.

    ``m (1 - 0.3 m^2)`` peaks at 0.7027, so a pixel at normalized radius 0.70 has its ray at
    90 degrees, next to the fold, and one at 0.71 has no ray in the domain at all.
    """
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=MEI_FOLD)))
    grid = _pixel_grid()
    radius = np.hypot((grid[:, 0] - CX) / FX, (grid[:, 1] - CY) / FY)
    edge = np.array([[CX + 0.70 * FX, CY], [CX, CY - 0.70 * FY]])
    inside = np.vstack((grid[radius < 0.70], edge))

    rays = projection.unproject(inside)

    assert (_ray_angle(rays) < math.radians(MEI_FOLD_DEG)).all()
    np.testing.assert_allclose(_ray_angle(rays[-2:]), math.pi / 2, atol=1e-9)
    np.testing.assert_allclose(projection.project(rays * 3.0).pixels, inside, atol=1e-6)
    with pytest.raises(ValueError, match="no viewing ray"):
        projection.unproject(np.array([[CX + 0.71 * FX, CY]]))


def test_a_unit_mirror_shift_sees_everything_except_the_exact_opposite_direction() -> None:
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points(_at_angle(179.0), (0.0, 0.0, -1.0)))

    assert result.projectable.tolist() == [True, False]


def test_the_point_on_the_negative_axis_never_folds_back_to_the_principal_point() -> None:
    projection = camera_projection_for(_entry(_mei_model(k=(0.0, 0.0, 0.0, 0.0))))

    result = projection.project(_points((0.0, 0.0, -1.0)))

    assert not result.projectable[0]
    assert np.isnan(result.pixels).all()


def test_the_input_must_be_finite_camera_frame_points() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite"):
        projection.project(_points((0.0, float("nan"), 1.0)))
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        projection.project(np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        projection.project(np.zeros(3))


# --- Round trip through unproject -------------------------------------------


_CAMERAS = [
    pytest.param(_pinhole_model(), id="pinhole"),
    pytest.param(_pinhole_model(DistortionModel.PLUMB_BOB, PLUMB_BOB), id="plumb_bob"),
    pytest.param(_pinhole_model(DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL), id="rational"),
    pytest.param(
        _pinhole_model(DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL_POLE), id="rational-pole"
    ),
    pytest.param(_fisheye_model(), id="fisheye"),
    pytest.param(_mei_model(), id="mei"),
    pytest.param(_mei_model(xi=0.6), id="mei-small-xi"),
]


def _pixel_grid() -> NDArray[np.float64]:
    us = np.linspace(0.0, WIDTH - 1.0, 9)
    vs = np.linspace(0.0, HEIGHT - 1.0, 7)
    grid = np.array([(u, v) for u in us for v in vs], dtype=np.float64)
    return grid


@pytest.mark.parametrize("model", _CAMERAS)
def test_unprojecting_a_pixel_and_projecting_the_ray_returns_the_same_pixel(
    model: CameraModel,
) -> None:
    projection = camera_projection_for(_entry(model))
    pixels = _pixel_grid()

    rays = projection.unproject(pixels)
    back = projection.project(rays * 3.0)

    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
    assert back.projectable.all()
    np.testing.assert_allclose(back.pixels, pixels, atol=1e-6)


def test_a_pixel_beyond_the_horizon_of_the_model_has_no_ray() -> None:
    # O raio normalizado 1 / sqrt(xi^2 - 1) é o horizonte do modelo unificado.
    xi = 1.5
    projection = camera_projection_for(
        _entry(_mei_model(xi=xi, k=(0.0, 0.0, 0.0, 0.0))),
    )
    horizon_px = FX / math.sqrt(xi * xi - 1)

    with pytest.raises(ValueError, match="no viewing ray"):
        projection.unproject(np.array([[CX + horizon_px * 1.01, CY]]))
    np.testing.assert_allclose(
        projection.unproject(np.array([[CX, CY]])), [[0.0, 0.0, 1.0]], atol=1e-12
    )


def test_unproject_rejects_pixels_that_are_not_finite() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite"):
        projection.unproject(np.array([[float("nan"), 1.0]]))


# --- Bound of the ray angle over an image box -------------------------------

_FULL_IMAGE = {"u_bounds": (-0.5, WIDTH - 0.5), "v_bounds": (-0.5, HEIGHT - 0.5)}
_EVERYWHERE = {"u_bounds": (-1e300, 1e300), "v_bounds": (-1e300, 1e300)}
"""A box no ray of the viewing domain misses: its bound is the domain's own limit."""


def _ray_angle(rays: NDArray[np.float64]) -> NDArray[np.float64]:
    angles: NDArray[np.float64] = np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2])
    return angles


def _image_border(step_px: float = 0.5) -> NDArray[np.float64]:
    """Pixel centers along the four edges of the image, corners included."""
    us = np.arange(0.0, WIDTH - 1.0 + step_px / 2, step_px)
    vs = np.arange(0.0, HEIGHT - 1.0 + step_px / 2, step_px)
    edges = [
        np.column_stack((us, np.zeros_like(us))),
        np.column_stack((us, np.full_like(us, HEIGHT - 1.0))),
        np.column_stack((np.zeros_like(vs), vs)),
        np.column_stack((np.full_like(vs, WIDTH - 1.0), vs)),
    ]
    return np.vstack(edges)


def test_the_ray_angle_bound_of_an_ideal_pinhole_is_its_farthest_corner() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    corner = math.atan(math.hypot(320.5 / FX, 240.5 / FY))
    assert corner <= bound <= corner + 1e-12
    # O canto, não a metade do campo de visão horizontal.
    assert bound > math.atan(320.5 / FX) + 0.1


def test_a_radially_distorted_pinhole_is_bounded_by_the_ray_of_its_farthest_corner() -> None:
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.07, 0.0, 0.0, 0.0)))
    )

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    farthest = float(_ray_angle(projection.unproject(np.array([[-0.5, -0.5]])))[0])
    assert farthest - 1e-12 <= bound <= farthest + 1e-9


def test_a_pinhole_whose_fold_lies_inside_the_image_is_bounded_by_the_fold() -> None:
    # Com k1 = -0,28 sozinho o raio distorcido chega no máximo a 0,727 antes de dobrar,
    # e o canto está a 0,81: a imagem alcança a dobra, então o limite é o domínio.
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, (-0.28, 0.0, 0.0, 0.0, 0.0)))
    )

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.atan(math.sqrt(1 / 0.84)), abs=1e-9)


def test_a_fisheye_whose_valid_circle_ends_inside_the_image_is_bounded_by_its_domain() -> None:
    # theta_d cresce até sqrt(1 / 0,9) rad, onde vale 0,703; o canto está a 0,81.
    projection = camera_projection_for(_entry(_fisheye_model((-0.3, 0.0, 0.0, 0.0))))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.sqrt(1 / 0.9), abs=1e-9)


def test_a_unified_model_whose_horizon_is_inside_the_image_is_bounded_by_the_horizon() -> None:
    wide = MeiCameraModel(
        width=WIDTH,
        height=HEIGHT,
        fx=200.0,
        fy=200.0,
        cx=CX,
        cy=CY,
        xi=MEI_XI,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
    )
    projection = camera_projection_for(_entry(wide))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.acos(-1 / MEI_XI), abs=1e-9)
    assert bound > math.pi / 2


def test_a_unified_model_whose_fold_is_inside_the_image_is_bounded_by_the_fold() -> None:
    # O raio distorcido chega no máximo a 0,703 antes de dobrar, e o canto está a 0,81: a
    # imagem alcança a dobra, então o limite é o domínio, não o horizonte a 180 graus.
    projection = camera_projection_for(_entry(_mei_model(xi=1.0, k=MEI_FOLD)))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    assert bound == pytest.approx(math.radians(MEI_FOLD_DEG), abs=1e-9)


@pytest.mark.parametrize("model", _CAMERAS)
def test_no_ray_the_model_sends_into_the_image_exceeds_the_bound(model: CameraModel) -> None:
    """Sound for every model, tangential terms included, and still close to the border."""
    projection = camera_projection_for(_entry(model))

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    border = _ray_angle(projection.unproject(_image_border()))
    assert border.max() <= bound
    assert bound - border.max() < 0.01


def test_a_pinhole_with_strong_tangential_terms_has_a_bound_short_of_its_fold() -> None:
    """Without a tangential fold in the domain the bound degenerated to 90 degrees."""
    projection = camera_projection_for(
        _entry(_pinhole_model(DistortionModel.PLUMB_BOB, TANGENTIAL))
    )

    bound = projection.max_ray_angle_rad(**_FULL_IMAGE)

    border = _ray_angle(projection.unproject(_image_border()))
    assert border.max() <= bound < math.radians(TANGENTIAL_FOLD_DEG)


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [
        (DistortionModel.PLUMB_BOB, PLUMB_BOB),
        (DistortionModel.RATIONAL_POLYNOMIAL, RATIONAL),
        (DistortionModel.PLUMB_BOB, TANGENTIAL),
    ],
    ids=["plumb_bob", "rational", "tangential"],
)
def test_the_image_is_injective_under_the_pinhole_and_round_trips(
    model: DistortionModel, coefficients: tuple[float, ...]
) -> None:
    """The audit's metrics over the image, from 41 x 31 pixels, borders included.

    No pixel lacks a ray (the invalid share of the image is zero), the smallest ``det J`` at
    the undistorted points is positive, and the round trip error stays below ``1e-6`` pixel.
    """
    projection = camera_projection_for(_entry(_pinhole_model(model, coefficients)))
    us, vs = np.linspace(-0.5, WIDTH - 0.5, 41), np.linspace(-0.5, HEIGHT - 0.5, 31)
    pixels = np.array([(u, v) for u in us for v in vs], dtype=np.float64)

    rays = projection.unproject(pixels)
    back = projection.project(rays * 3.0)

    undistorted_x, undistorted_y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    assert _smallest_reference_determinant(coefficients, undistorted_x, undistorted_y) > 0
    assert back.projectable.all()
    assert float(np.abs(back.pixels - pixels).max()) < 1e-6


def test_the_box_must_be_finite_and_ordered() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    with pytest.raises(ValueError, match="finite and ordered"):
        projection.max_ray_angle_rad(u_bounds=(10.0, 0.0), v_bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="finite and ordered"):
        projection.max_ray_angle_rad(u_bounds=(0.0, 1.0), v_bounds=(0.0, float("nan")))


def test_the_bound_narrows_with_the_box() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))

    full = projection.max_ray_angle_rad(**_FULL_IMAGE)
    central = projection.max_ray_angle_rad(u_bounds=(219.5, 419.5), v_bounds=(139.5, 339.5))

    assert central == pytest.approx(math.atan(math.hypot(100.5 / FX, 100.5 / FY)), abs=1e-12)
    assert central < full


# --- Pixel domain -----------------------------------------------------------


def test_a_pixel_is_inside_when_its_center_is_within_the_image_extent() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))
    pixels = np.array(
        [
            [0.0, 0.0],
            [-0.5, -0.5],
            [WIDTH - 1.0, HEIGHT - 1.0],
            [WIDTH - 0.5, 10.0],
            [10.0, HEIGHT - 0.5],
            [-0.6, 5.0],
            [float("nan"), 5.0],
        ]
    )

    assert projection.in_image(pixels).tolist() == [True, True, True, False, False, False, False]


# --- Selection and provenance from canonical calibration only ---------------


@pytest.mark.parametrize(
    ("model", "expected_type", "kind"),
    [
        (_pinhole_model(), PinholeProjection, "pinhole"),
        (_fisheye_model(), FisheyeProjection, "fisheye"),
        (_mei_model(), MeiProjection, "mei"),
    ],
)
def test_the_projection_is_selected_solely_by_the_canonical_model_kind(
    model: CameraModel, expected_type: type, kind: str
) -> None:
    entry = _entry(model)

    projection = camera_projection_for(entry)

    assert isinstance(projection, expected_type)
    assert isinstance(projection, CameraProjection)
    assert projection.identity == CameraIdentity(
        calibration_id=entry.calibration_id,
        content_hash=entry.content_hash,
        camera_frame=entry.frame_id,
        camera_model_kind=kind,
        image_size=(WIDTH, HEIGHT),
    )


def test_every_projection_result_names_the_calibration_that_produced_it() -> None:
    projection = camera_projection_for(_entry(_mei_model()))

    result = projection.project(_points((0.0, 0.0, 1.0)))

    assert isinstance(result, PixelProjection)
    assert result.camera == projection.identity


def test_an_entry_without_a_camera_model_cannot_be_projected() -> None:
    with pytest.raises(ValueError, match="camera model"):
        camera_projection_for(_entry(None, name="lidar"))


def test_an_invalid_calibration_is_rejected_instead_of_projected() -> None:
    broken = dataclasses.replace(_pinhole_model(), fx=float("nan"))

    with pytest.raises(CalibrationError, match="fx"):
        camera_projection_for(_entry(broken))


def test_the_projection_result_keeps_its_arrays_consistent() -> None:
    projection = camera_projection_for(_entry(_pinhole_model()))
    good = projection.project(_points((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))

    with pytest.raises(ValueError, match="projectable"):
        dataclasses.replace(good, projectable=np.array([True, True]))
    with pytest.raises(ValueError, match="shape"):
        dataclasses.replace(good, depth_m=np.zeros(3))


def test_the_camera_models_know_no_dataset_by_name() -> None:
    package = Path(camera.__file__).parent
    modules = sorted(package.glob("*.py"))

    assert modules, f"no camera module found under {package}"
    for module in modules:
        source = module.read_text(encoding="utf-8").lower()
        assert "corridor" not in source, module.name
        assert "kalibr" not in source, module.name
