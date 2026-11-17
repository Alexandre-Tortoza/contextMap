"""The interaction radius is derived from the primary hypothesis's factor and the measured size.

``interaction_radius_m = 0.5 * max(extent_m) * clamp(factor)``. The factor is the interpreter's
semantic judgement, fused onto the hypothesis; the meters come from the entity's measured bounds, so
the interpreter never fabricates a metric quantity. The attribute is added only when a primary
hypothesis carries a factor, and never replaces the ``class`` attribute.
"""

from evidence_builders import ClaimSpec, View
from mapping_fusion import fuse

from contextmap.semantic_fusion import FusedEvidence
from contextmap.semantic_mapping import AttributeOrigin, EntityAttribute, EntitySemanticState
from contextmap.semantic_mapping.interaction_radius import (
    INTERACTION_RADIUS_DERIVATION_ID,
    InteractionRadiusPolicy,
    attach_interaction_radius,
)
from contextmap.semantic_mapping.state_mapping import semantic_state_from_fused_evidence

_EXTENT = (2.0, 1.0, 0.5)


def _attached_to(fused: FusedEvidence) -> EntitySemanticState:
    state = semantic_state_from_fused_evidence(fused)
    # The factor is fused onto the hypothesis; materialization reads the primary's and passes it in.
    factor = fused.hypotheses[0].interaction_factor if fused.hypotheses else None
    return attach_interaction_radius(state, extent_m=_EXTENT, interaction_factor=factor)


def _attached(interaction_factor: float | None) -> EntitySemanticState:
    claim = ClaimSpec("door", confidence=0.9, interaction_factor=interaction_factor)
    return _attached_to(fuse([View("run-a", "frame-0120", (claim,))])[1])


def _radius(state: EntitySemanticState) -> EntityAttribute | None:
    return next((item for item in state.attributes if item.name == "interaction_radius_m"), None)


def test_the_radius_is_half_the_largest_side_scaled_by_the_factor() -> None:
    state = _attached(0.5)

    radius = _radius(state)
    assert radius is not None
    assert radius.origin is AttributeOrigin.DERIVED
    assert radius.derivation_id == InteractionRadiusPolicy().derivation_id
    assert float(radius.value) == 0.5 * 2.0 * 0.5  # 0.5 * max(2, 1, 0.5) * 0.5 = 0.5
    assert radius.evidence  # cites the claims that support the primary hypothesis


def test_a_state_without_a_factor_keeps_only_the_class_attribute() -> None:
    state = _attached(None)

    assert _radius(state) is None
    assert [item.name for item in state.attributes] == ["class"]


def test_the_factor_is_clamped_before_scaling() -> None:
    state = _attached(50.0)

    radius = _radius(state)
    assert radius is not None
    assert float(radius.value) == 0.5 * 2.0 * 5.0  # factor clamped to 5.0


def test_the_class_and_radius_attributes_coexist_sorted_by_name() -> None:
    state = _attached(1.0)

    assert [item.name for item in state.attributes] == ["class", "interaction_radius_m"]


def test_the_policy_id_and_fingerprint_appear_in_the_radius_provenance() -> None:
    radius = _radius(_attached(0.5))

    policy = InteractionRadiusPolicy()
    assert radius is not None
    assert INTERACTION_RADIUS_DERIVATION_ID == "interaction-radius-from-factor-v1"
    assert policy.policy_id == INTERACTION_RADIUS_DERIVATION_ID
    assert policy.derivation_id == f"{policy.policy_id}@{policy.fingerprint()}"
    assert radius.derivation_id == policy.derivation_id


def test_the_fingerprint_changes_with_the_clamp_range() -> None:
    assert (
        InteractionRadiusPolicy().fingerprint()
        != InteractionRadiusPolicy(factor_max=3.0).fingerprint()
    )


def test_a_custom_policy_clamps_with_its_own_range() -> None:
    fused = fuse(
        [View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=50.0),))]
    )[1]
    state = semantic_state_from_fused_evidence(fused)

    attached = attach_interaction_radius(
        state,
        extent_m=_EXTENT,
        interaction_factor=fused.hypotheses[0].interaction_factor,
        policy=InteractionRadiusPolicy(factor_max=2.0),
    )

    radius = _radius(attached)
    assert radius is not None
    assert float(radius.value) == 0.5 * 2.0 * 2.0


def test_the_radius_cites_only_the_claims_that_declared_the_factor() -> None:
    fused = fuse(
        [
            View(
                "run-a",
                "frame-0120",
                (
                    ClaimSpec("door", confidence=0.9, interaction_factor=1.0, claim_id="declares"),
                    ClaimSpec("door", confidence=0.8, claim_id="silent"),
                ),
            ),
            View("run-a", "frame-0130", (ClaimSpec("door", confidence=0.9, claim_id="other"),)),
        ]
    )[1]

    state = _attached_to(fused)

    radius = _radius(state)
    class_attribute = next(item for item in state.attributes if item.name == "class")
    assert radius is not None
    assert [str(ref.claim_id) for ref in radius.evidence] == ["declares"]
    assert len(class_attribute.evidence) == 3
