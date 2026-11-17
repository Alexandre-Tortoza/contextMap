"""What the claims stated about the object reaches the entity, and so does the support per label.

Two things Semantic Fusion keeps on a fused hypothesis must survive materialization for the map to
use them later:

* every ``(name, value)`` the supporting claims stated (their ``category`` and ``attributes``)
  becomes an ``OBSERVED`` attribute derived by ``claim-attribute-v1``, citing exactly those claims,
  whether or not a primary hypothesis is exposed. Nothing is aggregated: two values of one name
  are two attributes and neither wins;
* every hypothesis keeps the distinct physical observations that support it, so a later stage can
  count independent support after identity resolution without the fused evidence at hand.
"""

from __future__ import annotations

import dataclasses

from mapping_builders import make_entity, make_hypothesis, make_semantic_state
from mapping_fusion import ClaimSpec, View, fuse

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import EvidenceReference, FusedEvidence
from contextmap.semantic_interpretation import HypothesisRole
from contextmap.semantic_mapping import (
    CLAIM_ATTRIBUTE_DERIVATION_ID,
    CLASS_ATTRIBUTE_DERIVATION_ID,
    AmbiguityState,
    AttributeOrigin,
    EntityAttribute,
    EntitySemanticState,
    semantic_state_from_fused_evidence,
)
from contextmap.semantic_mapping.serialization import decode_entity, encode_entity

BLUE_METAL_DOOR = ClaimSpec(
    "blue metal door", category="door", attributes=(("color", "blue"), ("material", "metal"))
)


def _state(views: list[View]) -> tuple[FusedEvidence, EntitySemanticState]:
    _, fused = fuse(views)
    return fused, semantic_state_from_fused_evidence(fused)


def _observed(state: EntitySemanticState) -> list[EntityAttribute]:
    return [item for item in state.attributes if item.origin is AttributeOrigin.OBSERVED]


def _supporting_claims(fused: FusedEvidence) -> tuple[EvidenceReference, ...]:
    (hypothesis,) = fused.hypotheses
    return tuple(
        EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim_id)
        for item in hypothesis.evidence
    )


def test_a_blue_metal_door_keeps_its_label_and_its_observed_attributes() -> None:
    fused, state = _state([View("run-a", "frame-0120", (BLUE_METAL_DOOR,))])

    assert [item.label for item in state.hypotheses] == ["blue metal door"]
    observed = _observed(state)
    assert [(item.name, item.value) for item in observed] == [
        ("category", "door"),
        ("color", "blue"),
        ("material", "metal"),
    ]
    assert all(item.derivation_id == CLAIM_ATTRIBUTE_DERIVATION_ID for item in observed)
    assert all(item.evidence == _supporting_claims(fused) for item in observed)


def test_the_class_attribute_is_still_derived_from_the_primary_hypothesis() -> None:
    _, state = _state([View("run-a", "frame-0120", (BLUE_METAL_DOOR,))])

    (derived,) = [item for item in state.attributes if item.origin is AttributeOrigin.DERIVED]
    assert (derived.name, derived.value, derived.derivation_id) == (
        "class",
        "blue metal door",
        CLASS_ATTRIBUTE_DERIVATION_ID,
    )


def test_two_claims_with_different_values_give_two_attributes_and_no_winner() -> None:
    _, state = _state(
        [
            View("run-a", "frame-0120", (ClaimSpec("door", attributes=(("color", "blue"),)),)),
            View("run-a", "frame-0130", (ClaimSpec("door", attributes=(("color", "gray"),)),)),
        ]
    )

    colors = [item for item in _observed(state) if item.name == "color"]
    assert [item.value for item in colors] == ["blue", "gray"]
    assert colors[0].evidence != colors[1].evidence
    assert all(len(item.evidence) == 1 for item in colors)


def test_observed_attributes_do_not_need_a_primary_hypothesis() -> None:
    _, state = _state(
        [
            View(
                "run-a",
                "frame-0120",
                (
                    ClaimSpec("door", attributes=(("color", "blue"),)),
                    ClaimSpec(
                        "wall", role=HypothesisRole.ALTERNATIVE, attributes=(("color", "white"),)
                    ),
                ),
            )
        ]
    )

    assert state.ambiguity_state is AmbiguityState.AMBIGUOUS
    assert state.primary_hypothesis is None
    assert [(item.name, item.value) for item in state.attributes] == [
        ("color", "blue"),
        ("color", "white"),
    ]


def test_an_entity_without_claim_attributes_has_only_the_class_attribute() -> None:
    _, state = _state([View("run-a", "frame-0120", (ClaimSpec("door"),))])

    assert [item.name for item in state.attributes] == ["class"]


# --- supporting physical observations per hypothesis ---------------------------------------------


def test_each_hypothesis_lists_the_distinct_physical_observations_that_support_it() -> None:
    fused, state = _state(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-a", "frame-0130", (ClaimSpec("door"),)),
        ]
    )

    (hypothesis,) = state.hypotheses
    assert hypothesis.supporting_physical_observations == (
        SourceObservationId("frame-0120"),
        SourceObservationId("frame-0130"),
    )
    assert hypothesis.supporting_physical_observations == fused.supporting_physical_observations(
        hypothesis.hypothesis_id
    )


def test_repeated_inference_over_one_frame_is_one_supporting_observation() -> None:
    _, state = _state(
        [
            View("run-a", "frame-0120", (ClaimSpec("door"),)),
            View("run-b", "frame-0120", (ClaimSpec("door"),)),
        ]
    )

    (hypothesis,) = state.hypotheses
    assert hypothesis.supporting_physical_observations == (SourceObservationId("frame-0120"),)


def test_a_hypothesis_exposes_the_typographic_key_fusion_grouped_it_by() -> None:
    assert make_hypothesis(label="Metal  Door").label_key == "metal door"


# --- serialization -------------------------------------------------------------------------------


def test_observed_attributes_and_support_survive_the_entity_record() -> None:
    _, state = _state(
        [
            View("run-a", "frame-0120", (BLUE_METAL_DOOR,)),
            View("run-a", "frame-0130", (BLUE_METAL_DOOR,)),
        ]
    )
    entity = make_entity(semantic_state=state)

    assert decode_entity(encode_entity(entity)) == entity


def test_a_record_without_the_support_field_still_decodes() -> None:
    entity = make_entity(semantic_state=make_semantic_state())
    record = encode_entity(entity)

    assert "supporting_physical_observations" not in record["semantic_state"]["hypotheses"][0]
    assert decode_entity(record) == entity


def test_the_support_field_is_written_when_it_is_known() -> None:
    hypothesis = dataclasses.replace(
        make_hypothesis(), supporting_physical_observations=(SourceObservationId("frame-0120"),)
    )
    entity = make_entity(semantic_state=make_semantic_state((hypothesis,)))

    (record,) = encode_entity(entity)["semantic_state"]["hypotheses"]
    assert record["supporting_physical_observations"] == ["frame-0120"]
    assert decode_entity(encode_entity(entity)) == entity
