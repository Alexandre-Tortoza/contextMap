"""Properties of the rigid-transform primitives: a transform and its inverse cancel out.

Every capability chains ``compose_rigid``/``invert_rigid`` (calibration, poses, the map chain
of each point), so a point carried from ``A`` to ``B`` and back must land where it started,
for any rotation (near the identity, half turns, either sign of the quaternion) and any
translation, from signed zero and subnormals to planetary distances.
"""

from __future__ import annotations

import math

from hypothesis import example, given
from hypothesis import strategies as st

from contextmap.shared import (
    Quaternion,
    Vector3,
    compose_rigid,
    invert_rigid,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_angle_between,
    quaternion_norm,
    rotate_vector,
)
from property_strategies import finite_vectors, rotations

IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)
# Até dez mil quilômetros: coordenadas de mapa em metros, inclusive ECEF.
translations = finite_vectors(bound=1e7)
points = finite_vectors(bound=1e7)
# O erro de arredondamento de uma rotação cresce com a magnitude do que ela carrega; a
# tolerância é relativa a ela, com uma parcela absoluta para os subnormais.
RELATIVE_TOLERANCE = 1e-12


def _apply(translation: Vector3, rotation: Quaternion, point: Vector3) -> Vector3:
    rotated = rotate_vector(rotation, point)
    return (
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    )


def _close(actual: Vector3, expected: Vector3, *scales: Vector3) -> bool:
    magnitude = 1.0 + sum(math.hypot(*scale) for scale in scales)
    return math.dist(actual, expected) <= RELATIVE_TOLERANCE * magnitude


@given(translation=translations, rotation=rotations(), point=points)
def test_a_point_carried_from_a_to_b_and_back_lands_where_it_started(
    translation: Vector3, rotation: Quaternion, point: Vector3
) -> None:
    inverse_translation, inverse_rotation = invert_rigid(translation=translation, rotation=rotation)

    carried = _apply(translation, rotation, point)
    back = _apply(inverse_translation, inverse_rotation, carried)

    assert _close(back, point, translation, point)


@given(translation=translations, rotation=rotations())
def test_a_transform_composed_with_its_inverse_is_the_identity_on_either_side(
    translation: Vector3, rotation: Quaternion
) -> None:
    inverse_translation, inverse_rotation = invert_rigid(translation=translation, rotation=rotation)

    for outer, inner in (
        ((translation, rotation), (inverse_translation, inverse_rotation)),
        ((inverse_translation, inverse_rotation), (translation, rotation)),
    ):
        composed_translation, composed_rotation = compose_rigid(
            outer_translation=outer[0],
            outer_rotation=outer[1],
            inner_translation=inner[0],
            inner_rotation=inner[1],
        )
        assert _close(composed_translation, (0.0, 0.0, 0.0), translation)
        assert quaternion_angle_between(composed_rotation, IDENTITY) <= 1e-12
        assert is_unit_quaternion(composed_rotation)


@given(
    outer_translation=translations,
    outer_rotation=rotations(),
    inner_translation=translations,
    inner_rotation=rotations(),
    point=points,
)
def test_a_composed_transform_carries_a_point_like_the_two_in_sequence(
    outer_translation: Vector3,
    outer_rotation: Quaternion,
    inner_translation: Vector3,
    inner_rotation: Quaternion,
    point: Vector3,
) -> None:
    translation, rotation = compose_rigid(
        outer_translation=outer_translation,
        outer_rotation=outer_rotation,
        inner_translation=inner_translation,
        inner_rotation=inner_rotation,
    )

    in_sequence = _apply(
        outer_translation, outer_rotation, _apply(inner_translation, inner_rotation, point)
    )

    assert _close(
        _apply(translation, rotation, point),
        in_sequence,
        outer_translation,
        inner_translation,
        point,
    )


@given(translation=translations, rotation=rotations())
def test_inverting_twice_gives_the_transform_back(
    translation: Vector3, rotation: Quaternion
) -> None:
    once = invert_rigid(translation=translation, rotation=rotation)
    twice_translation, twice_rotation = invert_rigid(translation=once[0], rotation=once[1])

    assert twice_rotation == rotation
    assert _close(twice_translation, translation, translation)


# --- Normalizing gives a unit quaternion or refuses (#614, #647) -----------------------------


# O contraexemplo reduzido da #614: a norma subnormal devolvia (0.5, 1.0, 0.0, 0.0) (#647).
@example(quaternion=(5e-324, 1e-323, 0.0, 0.0))
# Componentes finitos cuja norma passa do maior float: a divisão por inf devolvia zeros.
@example(quaternion=(1e308, 1e308, 1e308, 1e308))
@given(quaternion=st.tuples(*(st.floats(allow_nan=False, allow_infinity=False) for _ in range(4))))
def test_normalizing_a_quaternion_gives_a_unit_quaternion_or_refuses(
    quaternion: Quaternion,
) -> None:
    try:
        normalized = normalize_quaternion(quaternion)
    except ValueError:
        return
    assert is_unit_quaternion(normalized), (quaternion, normalized, quaternion_norm(normalized))
