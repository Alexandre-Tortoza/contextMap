"""Properties of range culling for ``RAY_RANGE`` cameras (fisheye, MEI).

``docs/projection_chain.md`` states why culling by range is exact for these cameras: the
occlusion rule compares the distance to the optical centre, so every excluded element is
strictly farther than every retained one in the compared quantity, an excluded element can never
have been the nearest support of a retained one, and everything about a retained element is
what the full map gives it. These properties check both statements over random scenes, camera
positions and radii, including a radius equal to the range of one of the points, one or two
units in the last place either side of it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st
from projection_builders import (
    ArrayGeometrySource,
    make_calibration,
    map_point_for_camera_point,
    project_frame,
)

from contextmap.ingestion import CalibrationSet, FisheyeCameraModel, MeiCameraModel
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    OcclusionPolicy,
    VisibilityState,
)
from contextmap.sensor_association.projection_chain.candidate_geometry import (
    select_candidate_geometry,
)
from contextmap.sensor_association.projection_chain.frame_projection import FrameProjection
from contextmap.sensor_association.projection_chain.visibility import resolve_visibility
from contextmap.shared import Vector3

IDENTITY = (0.0, 0.0, 0.0, 1.0)
# Grade grossa: janelas largas fazem os pontos da cena disputarem o suporte uns dos outros.
COARSE = OcclusionPolicy(
    cell_size_px=64, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)


def _fisheye() -> CalibrationSet:
    return make_calibration(
        model=FisheyeCameraModel(
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
        )
    )


def _mei() -> CalibrationSet:
    return make_calibration(
        model=MeiCameraModel(
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            xi=1.0,
            distortion_coefficients=(0.0, 0.0, 0.0, 0.0),
        )
    )


CALIBRATIONS = {"fisheye": _fisheye(), "mei": _mei()}

# Coordenadas de cena em metros: zero (com sinal) ou de um nanômetro a 40 m. Abaixo disso o
# quadrado de uma coordenada chega aos subnormais e some, e a distância deixa de ter sentido
# físico: um sensor não resolve 1e-150 m.
coordinates = (
    st.sampled_from([0.0, -0.0])
    | st.floats(min_value=1e-9, max_value=40.0)
    | st.floats(min_value=-40.0, max_value=-1e-9)
)
camera_points = st.lists(st.tuples(coordinates, coordinates, coordinates), min_size=2, max_size=12)
body_positions = st.tuples(coordinates, coordinates, coordinates)


def _scene(points: list[Vector3], body: Vector3) -> list[Vector3]:
    """Camera-frame points placed in the map, with the body (and the camera) at ``body``."""
    placed = []
    for point in points:
        mapped = map_point_for_camera_point(point)
        placed.append((mapped[0] + body[0], mapped[1] + body[1], mapped[2] + body[2]))
    return placed


def _project(
    scene: list[Vector3], body: Vector3, calibration: CalibrationSet, radius: float | None
) -> FrameProjection:
    return project_frame(
        scene,
        calibration=calibration,
        poses=((0, body, IDENTITY), (100_000_000, body, IDENTITY)),
        candidate_policy=CandidateGeometryPolicy(max_range_m=radius),
    )


@st.composite
def radii(draw: st.DrawFn, ranges: list[float]) -> float:
    """A radius at the range of one point, a few ulps either side of it, or anywhere."""
    at_a_point = draw(st.sampled_from(ranges))
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        at_a_point = math.nextafter(at_a_point, draw(st.sampled_from([math.inf, 0.0])))
    radius = draw(st.just(at_a_point) | st.floats(min_value=0.01, max_value=100.0))
    return max(radius, math.ulp(0.0))


def _outcomes(frame: FrameProjection) -> dict[int, tuple[Any, ...]]:
    """Every evaluated element's outcome, keyed by its global map index, never by row."""
    resolution = resolve_visibility(frame, COARSE)
    outcomes = {}
    for row in range(frame.candidate_count):
        found = resolution.correspondence(row, visible_state=VisibilityState.VISIBLE_UNASSIGNED)
        outcomes[int(frame.global_indices[row])] = (
            found.visibility,
            found.raw_pixel,
            found.prepared_pixel,
            found.camera_depth_m,
            found.support_depth_m,
        )
    return outcomes


@given(
    points=camera_points,
    body=body_positions,
    camera=st.sampled_from(sorted(CALIBRATIONS)),
    data=st.data(),
)
def test_every_excluded_element_is_strictly_farther_than_every_retained_one(
    points: list[Vector3], body: Vector3, camera: str, data: st.DataObject
) -> None:
    scene = _scene(points, body)
    full = _project(scene, body, CALIBRATIONS[camera], None)
    ranges = {
        int(index): float(value)
        for index, value in zip(full.global_indices, full.camera_range_m, strict=True)
    }
    radius = data.draw(radii(sorted(ranges.values())))

    culled = _project(scene, body, CALIBRATIONS[camera], radius)

    kept = {int(index) for index in culled.global_indices}
    dropped = set(ranges) - kept
    # A grandeza comparada é a que a oclusão usa para RAY_RANGE: a distância ao centro óptico.
    if kept and dropped:
        assert min(ranges[i] for i in dropped) > max(ranges[i] for i in kept)


@given(
    points=camera_points,
    body=body_positions,
    camera=st.sampled_from(sorted(CALIBRATIONS)),
    data=st.data(),
)
def test_a_retained_element_is_evaluated_exactly_as_in_the_full_map(
    points: list[Vector3], body: Vector3, camera: str, data: st.DataObject
) -> None:
    scene = _scene(points, body)
    full = _project(scene, body, CALIBRATIONS[camera], None)
    radius = data.draw(radii(sorted(float(value) for value in full.camera_range_m)))

    culled = _project(scene, body, CALIBRATIONS[camera], radius)

    # Pixel, profundidade, visibilidade e o suporte que decidiu a oclusão: nada muda para quem fica.
    baseline = _outcomes(full)
    retained = _outcomes(culled)
    assert retained == {index: baseline[index] for index in retained}


# A decisão de pertencimento à esfera deve ser exata também na fronteira.


# Estes contraexemplos reproduzem o desacordo entre a distância ao quadrado em ponto
# flutuante e o teste exato. O primeiro está dentro da esfera exata e era excluído:
@example(
    point=(34.687225220099236, 0.0, 5.606270737171492), centre=(-48.0, 19.757140234641625, 43.75)
)
# Fora da esfera exata, e antes candidato:
@example(point=(0.0, 3.0, 41.0), centre=(0.0, 0.0, 0.0))
@given(point=body_positions, centre=body_positions)
def test_the_candidates_are_the_elements_within_the_range_of_the_camera(
    point: Vector3, centre: Vector3
) -> None:
    radius = math.dist(point, centre)
    source = ArrayGeometrySource([point], calibration=_fisheye())

    selection = select_candidate_geometry(
        source, camera_center_m=centre, policy=CandidateGeometryPolicy(max_range_m=radius or 1.0)
    )

    exact = sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(point, centre, strict=True))
    within = exact <= Fraction(radius or 1.0) ** 2
    assert np.array_equal(selection.cloud.global_indices, [0] if within else [])
