"""Attributes the claims stated are compared, but a divergence is not a conflict by default (D6).

``claim-attribute-v1`` attributes are what an interpreter *said* about one view: ``color=blue`` in
one frame and ``color=gray`` in the next is as likely lighting as two objects. The semantic
channel records every comparison, but a divergence is at most neutral unless the policy declares
the attribute relevant to identity; and even then two entities that share one of the stated values
do not conflict. Attributes of any other derivation keep the rule they always had.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from resolution_entity_builders import attribute, entity_at, hypotheses_of

from contextmap.entity_resolution import (
    BASELINE_REFINEMENT_MODIFIERS,
    SEMANTIC_COMPATIBILITY_POLICY_ID,
    EvidenceStatus,
    SemanticCompatibilityPolicy,
    SemanticEvidence,
    compare_semantics,
)
from contextmap.semantic_mapping import CLAIM_ATTRIBUTE_DERIVATION_ID, Entity, EntityAttribute

POLICY = SemanticCompatibilityPolicy(refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS)


def _stated(name: str, value: str, claim: str = "claim-0001") -> EntityAttribute:
    return attribute(name, value, derivation_id=CLAIM_ATTRIBUTE_DERIVATION_ID, claim=claim)


def _door(entity_id: str, *attributes: EntityAttribute) -> Entity:
    return entity_at(entity_id, hypotheses=hypotheses_of("door"), attributes=attributes)


def _attribute_status(evidence: SemanticEvidence) -> EvidenceStatus:
    (finding,) = [item for item in evidence.findings if item.rule_id == "attribute-compatibility"]
    return finding.status


def test_a_stated_color_that_differs_is_recorded_but_is_not_a_conflict() -> None:
    evidence = compare_semantics(
        _door("a", _stated("color", "blue")), _door("b", _stated("color", "gray")), POLICY
    )

    assert evidence.measurement is not None
    (comparison,) = evidence.measurement.attribute_comparisons
    assert (comparison.name, comparison.value_a, comparison.value_b, comparison.same) == (
        "color",
        "blue",
        "gray",
        False,
    )
    assert _attribute_status(evidence) is EvidenceStatus.NEUTRAL
    assert evidence.status is EvidenceStatus.SUPPORTING


def test_a_declared_stated_attribute_conflicts_when_nothing_is_shared() -> None:
    declared = SemanticCompatibilityPolicy(
        refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS, identity_attributes=("color",)
    )

    evidence = compare_semantics(
        _door("a", _stated("color", "blue")), _door("b", _stated("color", "gray")), declared
    )

    assert _attribute_status(evidence) is EvidenceStatus.CONFLICTING


def test_a_declared_attribute_with_a_shared_stated_value_does_not_conflict() -> None:
    declared = SemanticCompatibilityPolicy(
        refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS, identity_attributes=("color",)
    )

    evidence = compare_semantics(
        _door("a", _stated("color", "blue"), _stated("color", "gray", claim="claim-0002")),
        _door("b", _stated("color", "Blue")),
        declared,
    )

    assert _attribute_status(evidence) is EvidenceStatus.NEUTRAL


def test_attributes_of_other_derivations_keep_their_rule() -> None:
    evidence = compare_semantics(
        _door("a", attribute("material", "wood")),
        _door("b", attribute("material", "metal")),
        POLICY,
    )

    assert _attribute_status(evidence) is EvidenceStatus.CONFLICTING


@pytest.mark.parametrize(
    "names", [("",), ("Color",), ("color", "color"), ("material", "color"), (" color",)]
)
def test_identity_attributes_are_declared_normalized_sorted_and_unique(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="identity_attributes"):
        SemanticCompatibilityPolicy(refinement_modifiers=(), identity_attributes=names)


def test_declaring_no_identity_attribute_keeps_the_fingerprint_of_before() -> None:
    before = json.dumps(
        {
            "policy_id": SEMANTIC_COMPATIBILITY_POLICY_ID,
            "refinement_modifiers": list(BASELINE_REFINEMENT_MODIFIERS),
            "related_label_pairs": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    assert POLICY.fingerprint() == f"sha256:{hashlib.sha256(before.encode()).hexdigest()}"
    assert (
        SemanticCompatibilityPolicy(
            refinement_modifiers=BASELINE_REFINEMENT_MODIFIERS, identity_attributes=("color",)
        ).fingerprint()
        != POLICY.fingerprint()
    )
