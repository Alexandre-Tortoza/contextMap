"""The semantic interaction channel: does the object fall within the subject's interaction radius?

The radius is a per-subject semantic property (derived upstream from the interpreter's factor and
the entity's measured size); this rule compares it with the measured bounds gap, in its own channel,
leaving the geometric channel's arithmetic free of semantics.
"""

from __future__ import annotations

import pytest
from relation_builders import entity_ref
from relation_scene import Scene

from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    AxisDirection,
    FrameConventions,
    Quantity,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceStatus,
    RelationPredicate,
)
from contextmap.spatial_relations.semantic_predicates import (
    SemanticPredicatePolicy,
    evaluate_interaction_candidate,
)

Box = tuple[Vector3, Vector3]
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
POLICY = SemanticPredicatePolicy(boundary_tolerance_m=0.02)
UNIT: Box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _shift(box: Box, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Box:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 + dx, y0 + dy, z0 + dz), (x1 + dx, y1 + dy, z1 + dz))


def _geometries(subject: Box, obj: Box) -> tuple[EntityGeometry, EntityGeometry]:
    scene = Scene()
    scene.add_box("subject", *subject)
    scene.add_box("object", *obj)
    return scene.geometry("subject"), scene.geometry("object")


def _evaluate(subject: Box, obj: Box, *, radius_m: float) -> RelationEvidence:
    subject_geometry, object_geometry = _geometries(subject, obj)
    return evaluate_interaction_candidate(
        subject_entity_ref=entity_ref(1),
        subject_geometry=subject_geometry,
        subject_interaction_radius_m=radius_m,
        object_entity_ref=entity_ref(2),
        object_geometry=object_geometry,
        policy=POLICY,
        conventions=CONVENTIONS,
    )


def _quantity(items: tuple[Quantity, ...], name: str) -> float:
    values: dict[str, float] = {item.name: item.value for item in items}
    return values[name]


def test_an_object_within_the_radius_supports_the_relation() -> None:
    evidence = _evaluate(UNIT, _shift(UNIT, dx=1.3), radius_m=1.0)  # gap 0.3 < 1.0

    assert evidence.status is RelationEvidenceStatus.SUPPORTS
    assert evidence.channel is RelationEvidenceChannel.SEMANTIC
    assert evidence.predicate is RelationPredicate.WITHIN_INTERACTION_RANGE
    assert _quantity(evidence.measurements, "bounds_gap") == pytest.approx(0.3)


def test_an_object_beyond_the_radius_conflicts() -> None:
    evidence = _evaluate(UNIT, _shift(UNIT, dx=2.3), radius_m=0.5)  # gap 1.3 > 0.5 + tol

    assert evidence.status is RelationEvidenceStatus.CONFLICTS


def test_a_gap_within_tolerance_of_the_radius_is_ambiguous() -> None:
    evidence = _evaluate(UNIT, _shift(UNIT, dx=1.5), radius_m=0.5)  # gap 0.5 == radius, within tol

    assert evidence.status is RelationEvidenceStatus.AMBIGUOUS
    assert evidence.caveats  # an undecided record must say why


def test_the_radius_is_recorded_as_the_threshold_compared_with() -> None:
    evidence = _evaluate(UNIT, _shift(UNIT, dx=1.3), radius_m=1.0)

    assert _quantity(evidence.thresholds, "interaction_radius") == pytest.approx(1.0)


def test_a_nonpositive_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="subject_interaction_radius_m"):
        _evaluate(UNIT, _shift(UNIT, dx=1.3), radius_m=0.0)


def test_the_direction_evaluated_is_the_one_given() -> None:
    evidence = _evaluate(UNIT, _shift(UNIT, dx=1.3), radius_m=1.0)

    assert evidence.subject_entity_ref == entity_ref(1)
    assert evidence.object_entity_ref == entity_ref(2)
