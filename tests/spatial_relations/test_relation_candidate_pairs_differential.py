"""Differential gate for the candidate sweep of Spatial Relations (sweep and prune).

Candidate generation only measures the pairs ``_neighbor_pairs`` enumerates. A pair the sweep
misses is never a candidate nor an exclusion, only one more ``pairs_not_enumerated``, so a lost
relation would leave no trace in the artifact. The sweep is therefore compared with the exhaustive
O(N^2) enumeration of its own definition (a pair within ``reach`` on every axis), and with the gap
the candidate stage measures, on random scenes with fixed seeds, on the collinear scenes where
the choice of the sweep axis degenerates, and on boundary scenes. That the sweep also stays
sub-quadratic is gated in ``test_relation_scaling.py``.

The boundary scenes land gaps on the last float within ``reach`` and on the first beyond it, with
faces on both sides of zero, where ``upper face + reach`` and the measured ``lower face - upper
face`` round differently. The sweep once stopped on the sum and dropped pairs whose measured gap
was exactly the reach (#641); ``test_relation_candidate_pairs_properties.py`` walks that boundary
float by float.
"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Callable
from itertools import combinations

import pytest
from relation_builders import entity_ref
from relation_scene import Scene

from contextmap.geometric_mapping import Bounds3D
from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    FrameConventions,
    RelationPredicate,
    generate_relation_candidates,
    predicate_spec,
)
from contextmap.spatial_relations.candidates import _neighbor_pairs

Pair = tuple[int, int]

REACHES = (0.0, 0.4, 1.5)
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)


def _geometries(boxes: list[tuple[Vector3, Vector3]]) -> tuple[EntityGeometry, ...]:
    scene = Scene()
    for index, (low, high) in enumerate(boxes):
        scene.add_box(f"e{index:05d}", low, high)
    return tuple(scene.geometry(f"e{index:05d}") for index in range(len(boxes)))


def _box_around(center: Vector3, half: Vector3) -> tuple[Vector3, Vector3]:
    low = (center[0] - half[0], center[1] - half[1], center[2] - half[2])
    high = (center[0] + half[0], center[1] + half[1], center[2] + half[2])
    return low, high


@functools.cache
def _random_scene(seed: int, count: int = 150) -> tuple[EntityGeometry, ...]:
    """Boxes of every shape in a room: small, flat on some axis, and a few that span it."""
    rng = random.Random(seed)
    boxes = []
    for index in range(count):
        center = (rng.uniform(0.0, 12.0), rng.uniform(0.0, 8.0), rng.uniform(0.0, 3.0))
        if index % 25 == 0:
            # Poucos suportes grandes e planos (piso, parede) cruzam a varredura inteira.
            half: Vector3 = (rng.uniform(3.0, 6.0), rng.uniform(2.0, 4.0), 0.0)
        else:
            # Um eixo de extensão zero é um caso real (plano ou linha), não só um canto teórico.
            half = (
                0.0 if rng.random() < 0.2 else rng.uniform(0.02, 0.6),
                0.0 if rng.random() < 0.2 else rng.uniform(0.02, 0.6),
                0.0 if rng.random() < 0.2 else rng.uniform(0.02, 0.6),
            )
        boxes.append(_box_around(center, half))
    return _geometries(boxes)


LINES: dict[str, Vector3] = {
    "along-x": (1.0, 0.0, 0.0),
    "along-y": (0.0, 1.0, 0.0),
    "along-z": (0.0, 0.0, 1.0),
    # Os centros se espalham igualmente nos três eixos: a escolha do eixo cai no desempate.
    "diagonal": (1.0, 1.0, 1.0),
    # Todos os centros coincidem: nenhum eixo espalha, e as caixas se aninham.
    "coincident": (0.0, 0.0, 0.0),
}


@functools.cache
def _collinear_scene(line: str, count: int = 80) -> tuple[EntityGeometry, ...]:
    """Boxes of random sizes whose centers lie exactly on one line.

    Coordinates are multiples of 1/8, so every center is exactly on the line and the spread of
    two axes is exactly zero, or exactly tied, instead of rounding noise deciding the sweep axis.
    Touching faces are exact too, which is what a reach of zero has to keep.
    """
    rng = random.Random(f"collinear-{line}")
    direction = LINES[line]
    boxes = []
    for _ in range(count):
        position = rng.randrange(0, 240) / 8.0
        center = (position * direction[0], position * direction[1], position * direction[2])
        half = (rng.randrange(0, 5) / 8.0, rng.randrange(0, 5) / 8.0, rng.randrange(0, 5) / 8.0)
        boxes.append(_box_around(center, half))
    return _geometries(boxes)


BOUNDARY_REACHES = (0.4, 0.5, 1.5)
BOUNDARY_PAIRS = 40


def _last_face_within(upper_face: float, reach: float) -> float:
    """The largest lower face whose gap from ``upper_face``, as measured, is within ``reach``."""
    face = upper_face + reach
    while face - upper_face > reach:
        face = math.nextafter(face, -math.inf)
    while math.nextafter(face, math.inf) - upper_face <= reach:
        face = math.nextafter(face, math.inf)
    return face


@functools.cache
def _boundary_scene(seed: int, reach: float) -> tuple[EntityGeometry, ...]:
    """Pairs of boxes whose gap along ``x`` is the last float within ``reach`` or the first beyond.

    The boxes ``2k`` and ``2k + 1`` form a pair and share their ``y`` and ``z`` intervals, so the
    pair's gap is exactly its separation along ``x``. Upper faces straddle zero: across a change
    of exponent, and where the sum cancels, ``upper face + reach`` rounds to another float than the
    gap does. The centers spread along ``x`` more than along the other axes, so ``x`` is also the
    sweep axis.
    """
    rng = random.Random(f"boundary-{seed}")
    boxes = []
    for index in range(BOUNDARY_PAIRS):
        upper_face = rng.uniform(-1.5, 1.0)
        lower_face = _last_face_within(upper_face, reach)
        if index % 2:
            lower_face = math.nextafter(lower_face, math.inf)
        low_y, low_z = rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.5)
        high_y, high_z = low_y + 0.5, low_z + 0.5
        boxes.append(
            ((upper_face - rng.uniform(0.05, 0.5), low_y, low_z), (upper_face, high_y, high_z))
        )
        boxes.append(
            ((lower_face, low_y, low_z), (lower_face + rng.uniform(0.05, 0.5), high_y, high_z))
        )
    return _geometries(boxes)


# O contraexemplo mínimo da #641 e o seu espelho em x: o gap medido é 0.5, igual ao alcance.
MINIMAL_BOUNDARY: dict[str, list[tuple[Vector3, Vector3]]] = {
    "minimal": [
        ((0.0, 0.0, 0.0), (0.337290862214726, 1.0, 1.0)),
        ((0.837290862214726, 0.0, 0.0), (1.5, 1.0, 1.0)),
    ],
    "mirrored": [
        ((-0.337290862214726, 0.0, 0.0), (0.0, 1.0, 1.0)),
        ((-1.5, 0.0, 0.0), (-0.837290862214726, 1.0, 1.0)),
    ],
}


# --- The exhaustive definitions ------------------------------------------------------------------


def _separation_m(first: Bounds3D, second: Bounds3D, axis: int) -> float:
    """How far apart two boxes are along one axis; negative when they overlap on it."""
    return max(first.minimum_m[axis], second.minimum_m[axis]) - min(
        first.maximum_m[axis], second.maximum_m[axis]
    )


def _within_reach_on_every_axis(first: Bounds3D, second: Bounds3D, reach: float) -> bool:
    return all(_separation_m(first, second, axis) <= reach for axis in range(3))


def _gap_m(first: Bounds3D, second: Bounds3D) -> float:
    """The Euclidean gap between two boxes: what the candidate stage compares with a radius."""
    separations = [max(0.0, _separation_m(first, second, axis)) for axis in range(3)]
    return math.sqrt(sum(separation * separation for separation in separations))


def _brute_force(
    geometries: tuple[EntityGeometry, ...], keep: Callable[[Bounds3D, Bounds3D], bool]
) -> set[Pair]:
    return {
        (first, second)
        for first, second in combinations(range(len(geometries)), 2)
        if keep(geometries[first].bounds, geometries[second].bounds)
    }


def _swept(geometries: tuple[EntityGeometry, ...], reach: float) -> set[Pair]:
    pairs = list(_neighbor_pairs(list(geometries), reach))
    # Cada par sai uma vez, ordenado: é o que torna ``pairs_not_enumerated`` uma contagem.
    assert all(first < second for first, second in pairs)
    assert len(set(pairs)) == len(pairs)
    return set(pairs)


def _assert_agrees_with_brute_force(geometries: tuple[EntityGeometry, ...], reach: float) -> None:
    swept = _swept(geometries, reach)

    within_every_axis = _brute_force(
        geometries, lambda a, b: _within_reach_on_every_axis(a, b, reach)
    )
    within_gap = _brute_force(geometries, lambda a, b: _gap_m(a, b) <= reach)
    # A definição documentada da varredura, exatamente: nem par a mais, nem a menos.
    assert swept == within_every_axis
    # E o que importa cientificamente: nenhum par dentro do alcance medido pelo gap se perde.
    assert within_gap <= swept


# --- The sweep agrees with the exhaustive enumeration --------------------------------------------


@pytest.mark.parametrize("seed", [20260926, 1, 2, 3, 4, 5])
def test_the_sweep_enumerates_exactly_the_pairs_the_exhaustive_definition_does(seed: int) -> None:
    geometries = _random_scene(seed)

    for reach in REACHES:
        _assert_agrees_with_brute_force(geometries, reach)


@pytest.mark.parametrize("line", sorted(LINES))
def test_the_sweep_is_exact_when_every_center_lies_on_one_line(line: str) -> None:
    geometries = _collinear_scene(line)

    for reach in REACHES:
        _assert_agrees_with_brute_force(geometries, reach)


@pytest.mark.parametrize("case", sorted(MINIMAL_BOUNDARY))
def test_a_pair_whose_gap_is_exactly_the_reach_is_enumerated(case: str) -> None:
    geometries = _geometries(MINIMAL_BOUNDARY[case])

    assert _gap_m(geometries[0].bounds, geometries[1].bounds) == 0.5
    assert _swept(geometries, 0.5) == {(0, 1)}
    _assert_agrees_with_brute_force(geometries, 0.5)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("reach", BOUNDARY_REACHES)
def test_the_sweep_is_exact_when_gaps_land_on_the_reach(seed: int, reach: float) -> None:
    _assert_agrees_with_brute_force(_boundary_scene(seed, reach), reach)


def test_the_boundary_scenes_hold_pairs_that_upper_face_plus_reach_would_prune() -> None:
    """Meaningful only with pairs within reach whose lower face is past ``upper face + reach``."""
    for seed in (1, 2, 3):
        for reach in BOUNDARY_REACHES:
            geometries = _boundary_scene(seed, reach)
            pairs = [
                (geometries[2 * index].bounds, geometries[2 * index + 1].bounds)
                for index in range(BOUNDARY_PAIRS)
            ]
            assert any(
                _gap_m(upper, lower) <= reach and lower.minimum_m[0] > upper.maximum_m[0] + reach
                for upper, lower in pairs
            )
            assert any(_gap_m(upper, lower) > reach for upper, lower in pairs)


def test_the_scenes_exercise_both_kept_and_pruned_pairs() -> None:
    """The differential is only meaningful if the scenes have near and far pairs at every reach."""
    for geometries in (_random_scene(20260926), _collinear_scene("along-y")):
        total = len(geometries) * (len(geometries) - 1) // 2
        for reach in REACHES:
            swept = len(_swept(geometries, reach))
            assert 0 < swept < total


# --- What the candidate set records follows the enumeration --------------------------------------


def test_every_enumerated_pair_is_accounted_for_and_only_the_others_are_uncounted() -> None:
    geometries = _random_scene(20260926)
    predicates = (RelationPredicate.NEXT_TO, RelationPredicate.ABOVE)
    policy = CandidatePolicy(
        predicates=predicates, proximity_radius_m=0.4, directional_radius_m=1.5
    )
    entities = {entity_ref(number): geometry for number, geometry in enumerate(geometries, 1)}

    result = generate_relation_candidates(entities, policy=policy, conventions=CONVENTIONS)

    enumerated = _brute_force(geometries, lambda a, b: _within_reach_on_every_axis(a, b, 1.5))
    total = len(geometries) * (len(geometries) - 1) // 2
    assert result.pairs_not_enumerated == total - len(enumerated)
    # Cada par enumerado vira, por predicado e direção, um candidato ou uma exclusão: listada ou
    # contada num resumo de grupo acima do teto.
    directions = sum(2 if predicate_spec(item).directed else 1 for item in predicates)
    summarized = sum(summary.count for summary in result.exclusion_summaries)
    assert result.exclusion_summaries
    assert (
        len(result.candidates) + len(result.exclusions) + summarized == len(enumerated) * directions
    )
