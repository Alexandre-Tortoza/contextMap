"""A derived interaction reach: candidate generation reaches as far as the largest declared radius.

A configuration cannot know the entities' radii in advance, so it may ask for the reach to be
derived. ``resolve_interaction_reach`` then sets it to the greatest radius any resolved entity
declares, which is exactly the contract of ``interaction_radius_m``: no pair whose subject could
reach its object is dropped before it is measured. The resolved policy is what a run records.
"""

from __future__ import annotations

import pytest
from relation_builders import entity_ref
from relation_scene import Scene

from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    FrameConventions,
    RelationPredicate,
    generate_relation_candidates,
    resolve_interaction_reach,
)

WITHIN = RelationPredicate.WITHIN_INTERACTION_RANGE
NEXT_TO = RelationPredicate.NEXT_TO
CONVENTIONS = FrameConventions(
    map_frame="map", up_axis=AxisDirection.POSITIVE_Z, forward_axis=AxisDirection.POSITIVE_X
)


def _derived(*predicates: RelationPredicate) -> CandidatePolicy:
    return CandidatePolicy(
        predicates=predicates or (NEXT_TO, WITHIN),
        proximity_radius_m=0.6,
        directional_radius_m=2.0,
        derive_interaction_radius=True,
    )


def test_a_derived_reach_is_a_valid_configuration_before_it_is_resolved() -> None:
    policy = _derived()

    assert policy.derive_interaction_radius is True
    assert policy.interaction_radius_m is None


def test_deriving_a_reach_for_an_unselected_predicate_is_refused() -> None:
    with pytest.raises(ValueError, match="derive_interaction_radius"):
        CandidatePolicy(
            predicates=(NEXT_TO,),
            proximity_radius_m=0.6,
            directional_radius_m=2.0,
            derive_interaction_radius=True,
        )


def test_a_derived_reach_and_a_fixed_one_have_different_identities() -> None:
    fixed = CandidatePolicy(
        predicates=(NEXT_TO, WITHIN),
        proximity_radius_m=0.6,
        directional_radius_m=2.0,
        interaction_radius_m=4.2,
    )

    resolved = resolve_interaction_reach(_derived(), {entity_ref(1): 4.2})

    assert len({_derived().fingerprint(), fixed.fingerprint(), resolved.fingerprint()}) == 3


def test_the_reach_is_the_largest_declared_radius() -> None:
    resolved = resolve_interaction_reach(
        _derived(), {entity_ref(1): 1.5, entity_ref(2): 4.2, entity_ref(3): 0.3}
    )

    assert resolved.interaction_radius_m == 4.2
    assert resolved.derive_interaction_radius is True
    assert resolved.predicates == _derived().predicates


def test_without_any_radius_the_interaction_predicate_is_not_generated() -> None:
    resolved = resolve_interaction_reach(_derived(), {})

    assert WITHIN not in resolved.predicates
    assert resolved.predicates == (NEXT_TO,)
    assert (resolved.derive_interaction_radius, resolved.interaction_radius_m) == (False, None)


def test_without_any_radius_a_policy_with_only_the_interaction_predicate_is_refused() -> None:
    with pytest.raises(ValueError, match="no resolved entity declares an interaction radius"):
        resolve_interaction_reach(_derived(WITHIN), {})


def test_a_fixed_reach_is_left_as_configured() -> None:
    fixed = CandidatePolicy(
        predicates=(NEXT_TO, WITHIN),
        proximity_radius_m=0.6,
        directional_radius_m=2.0,
        interaction_radius_m=3.0,
    )

    assert resolve_interaction_reach(fixed, {entity_ref(1): 9.0}) is fixed


def test_an_unresolved_derived_reach_never_generates_interaction_candidates() -> None:
    scene = Scene()
    scene.add_box("e1", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    scene.add_box("e2", (1.3, 0.0, 0.0), (2.3, 1.0, 1.0))
    entities = {entity_ref(n): scene.geometry(f"e{n}") for n in (1, 2)}

    with pytest.raises(ValueError, match="interaction_radius_m"):
        generate_relation_candidates(entities, policy=_derived(WITHIN), conventions=CONVENTIONS)


def test_a_run_records_the_resolved_reach_and_that_it_was_derived() -> None:
    import dataclasses

    from relation_run_fixture import POLICIES

    from contextmap.spatial_relations import run_artifact

    derived = dataclasses.replace(
        POLICIES.candidate,
        predicates=(*POLICIES.candidate.predicates, WITHIN),
        derive_interaction_radius=True,
    )
    resolved = resolve_interaction_reach(derived, {entity_ref(1): 4.2})

    record = run_artifact._policies_record(dataclasses.replace(POLICIES, candidate=resolved))
    plain = run_artifact._policies_record(POLICIES)

    assert record["candidate"]["parameters"]["interaction_radius_m"] == 4.2
    assert record["candidate"]["parameters"]["interaction_reach"] == "max_entity_radius"
    assert "interaction_radius_m" not in plain["candidate"]["parameters"]
    assert "interaction_reach" not in plain["candidate"]["parameters"]
