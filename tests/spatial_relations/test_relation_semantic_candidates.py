"""Candidate generation for ``WITHIN_INTERACTION_RANGE``: directed, gated by the interaction reach.

The interaction reach is a single number (the greatest radius any entity declares) that the caller
derives from the entities' semantic radii; candidate generation still reads only geometry.
"""

from __future__ import annotations

import pytest
from relation_builders import entity_ref
from relation_scene import Scene

from contextmap.entity_resolution import ResolvedEntityReference
from contextmap.semantic_mapping import EntityGeometry
from contextmap.shared import Vector3
from contextmap.spatial_relations import (
    AxisDirection,
    CandidateExclusionReason,
    CandidatePolicy,
    FrameConventions,
    RelationPredicate,
    RelationState,
    decide_relations,
    generate_relation_candidates,
)
from contextmap.spatial_relations.semantic_predicates import (
    SemanticPredicatePolicy,
    evaluate_interaction_candidates,
)

Box = tuple[Vector3, Vector3]
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)
UNIT: Box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
WITHIN = RelationPredicate.WITHIN_INTERACTION_RANGE


def _shift(box: Box, dx: float) -> Box:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 + dx, y0, z0), (x1 + dx, y1, z1))


def _entities(*boxes: Box) -> dict[ResolvedEntityReference, EntityGeometry]:
    scene = Scene()
    for number, (low, high) in enumerate(boxes, start=1):
        scene.add_box(f"e{number}", low, high)
    return {entity_ref(n): scene.geometry(f"e{n}") for n in range(1, len(boxes) + 1)}


def _policy(interaction: float) -> CandidatePolicy:
    return CandidatePolicy(
        predicates=(WITHIN,),
        proximity_radius_m=0.5,
        directional_radius_m=2.0,
        interaction_radius_m=interaction,
    )


def test_a_pair_within_the_interaction_reach_is_a_candidate_in_both_directions() -> None:
    result = generate_relation_candidates(
        _entities(UNIT, _shift(UNIT, 1.3)),  # gap 0.3 <= 1.0
        policy=_policy(1.0),
        conventions=CONVENTIONS,
    )

    assert len(result.candidates) == 2  # directed: both (1->2) and (2->1)
    assert all(item.predicate is WITHIN for item in result.candidates)
    pairs = {(item.subject_entity_ref, item.object_entity_ref) for item in result.candidates}
    assert pairs == {(entity_ref(1), entity_ref(2)), (entity_ref(2), entity_ref(1))}


def test_a_pair_beyond_the_interaction_reach_is_excluded() -> None:
    result = generate_relation_candidates(
        _entities(UNIT, _shift(UNIT, 2.3)),  # gap 1.3 > 1.0
        policy=_policy(1.0),
        conventions=CONVENTIONS,
    )

    assert not result.candidates
    assert result.exclusions
    assert all(
        item.reason is CandidateExclusionReason.BEYOND_INTERACTION_RADIUS
        for item in result.exclusions
    )


def test_the_reach_is_the_interaction_radius_not_the_proximity_one() -> None:
    # gap 0.8 exceeds the proximity reach (0.5) but is within the interaction reach (1.0).
    result = generate_relation_candidates(
        _entities(UNIT, _shift(UNIT, 1.8)),  # gap 0.8
        policy=_policy(1.0),
        conventions=CONVENTIONS,
    )

    assert len(result.candidates) == 2


def test_the_policy_requires_the_interaction_radius_when_the_predicate_is_selected() -> None:
    with pytest.raises(ValueError, match="interaction_radius_m is required"):
        CandidatePolicy(predicates=(WITHIN,), proximity_radius_m=0.5, directional_radius_m=2.0)


def _evaluate(candidate_set: object, entities: object, radii: dict[object, float]) -> tuple:
    return evaluate_interaction_candidates(
        candidate_set,  # type: ignore[arg-type]
        entities=entities,  # type: ignore[arg-type]
        radii=radii,  # type: ignore[arg-type]
        policy=SemanticPredicatePolicy(boundary_tolerance_m=0.02),
        conventions=CONVENTIONS,
    )


def test_a_supported_interaction_candidate_becomes_a_relation() -> None:
    entities = _entities(UNIT, _shift(UNIT, 1.3))  # gap 0.3 <= radius 1.0
    candidate_set = generate_relation_candidates(
        entities, policy=_policy(1.0), conventions=CONVENTIONS
    )
    evidence = _evaluate(candidate_set, entities, {entity_ref(1): 1.0, entity_ref(2): 1.0})

    assert len(evidence) == 2  # both directions

    result = decide_relations(candidate_set, evidence)
    within = [item for item in result.relations if item.predicate is WITHIN]
    assert len(within) == 2
    assert all(item.state is RelationState.SUPPORTED for item in within)


def test_a_subject_without_a_radius_produces_no_semantic_evidence() -> None:
    entities = _entities(UNIT, _shift(UNIT, 1.3))
    candidate_set = generate_relation_candidates(
        entities, policy=_policy(1.0), conventions=CONVENTIONS
    )

    evidence = _evaluate(candidate_set, entities, {entity_ref(1): 1.0})  # only entity 1 has one

    assert len(evidence) == 1
    assert evidence[0].subject_entity_ref == entity_ref(1)
