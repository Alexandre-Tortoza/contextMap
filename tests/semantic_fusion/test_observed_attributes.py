"""The claim's category and attributes survive fusion, on the hypothesis the claim supports.

The label stays the grouping key (``baseline-evidence-accumulation-v1``); what a claim says *about*
the object -- its ``category`` and its named ``attributes`` -- travels with the hypothesis as
observations, each with the exact claims that stated it. Nothing is aggregated into one value:
``color=blue`` and ``color=gray`` from two views are two observations, and neither wins.
"""

from __future__ import annotations

import pytest
from evidence_builders import ClaimSpec, View, build_scenario, reference_to

from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedAttribute,
    FusedEvidence,
    FusedHypothesis,
    FusedHypothesisId,
    HypothesisEvidence,
    accumulate_baseline_evidence,
)
from contextmap.semantic_fusion.serialization import decode_fused_evidence, encode_fused_evidence
from contextmap.semantic_interpretation import ClaimId, HypothesisRole


def _accumulate(views: list[View], *, abstention: frozenset[str] = frozenset()) -> FusedEvidence:
    scenario = build_scenario(views)
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        policy=BaselineAccumulationPolicy(abstention_labels=abstention),
    )


def _pairs(hypothesis: FusedHypothesis) -> list[tuple[str, str]]:
    return [(item.name, item.value) for item in hypothesis.observed_attributes]


def test_category_and_attributes_of_the_claim_reach_its_hypothesis() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (
            ClaimSpec(
                "blue metal door",
                category="door",
                attributes=(("color", "blue"), ("material", "metal")),
            ),
        ),
    )
    scenario = build_scenario([view])

    (hypothesis,) = _accumulate([view]).hypotheses

    assert hypothesis.label == "blue metal door"
    assert _pairs(hypothesis) == [("category", "door"), ("color", "blue"), ("material", "metal")]
    claim = reference_to(scenario, view, 0)
    assert all(item.evidence == (claim,) for item in hypothesis.observed_attributes)


def test_divergent_values_are_kept_side_by_side_with_their_own_claims() -> None:
    first = View("run-a", "frame-0120", (ClaimSpec("door", attributes=(("color", "blue"),)),))
    second = View("run-a", "frame-0130", (ClaimSpec("door", attributes=(("color", "gray"),)),))
    scenario = build_scenario([first, second])

    (hypothesis,) = _accumulate([first, second]).hypotheses

    assert _pairs(hypothesis) == [("color", "blue"), ("color", "gray")]
    blue, gray = hypothesis.observed_attributes
    assert blue.evidence == (reference_to(scenario, first, 0),)
    assert gray.evidence == (reference_to(scenario, second, 0),)


def test_the_same_value_from_two_views_is_one_observation_citing_both_claims() -> None:
    first = View("run-a", "frame-0120", (ClaimSpec("door", attributes=(("color", "blue"),)),))
    second = View("run-a", "frame-0130", (ClaimSpec("door", attributes=(("color", "blue"),)),))
    scenario = build_scenario([first, second])

    ((attribute,),) = [item.observed_attributes for item in _accumulate([first, second]).hypotheses]

    assert (attribute.name, attribute.value) == ("color", "blue")
    assert attribute.evidence == tuple(
        sorted(
            (reference_to(scenario, first, 0), reference_to(scenario, second, 0)),
            key=lambda ref: (ref.contribution_id, ref.claim_id or ""),
        )
    )


def test_an_attribute_stays_with_the_hypothesis_its_claim_proposes() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (
            ClaimSpec("door", attributes=(("color", "blue"),)),
            ClaimSpec("wall", role=HypothesisRole.ALTERNATIVE, attributes=(("color", "white"),)),
        ),
    )

    door, wall = _accumulate([view]).hypotheses

    assert (door.label, _pairs(door)) == ("door", [("color", "blue")])
    assert (wall.label, _pairs(wall)) == ("wall", [("color", "white")])


def test_the_interaction_factor_keeps_its_own_typed_treatment_and_is_not_listed_again() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (ClaimSpec("door", interaction_factor=0.5, attributes=(("color", "blue"),)),),
    )

    (hypothesis,) = _accumulate([view]).hypotheses

    assert hypothesis.interaction_factor is not None
    assert hypothesis.interaction_factor.value == 0.5
    assert _pairs(hypothesis) == [("color", "blue")]


def test_scalar_values_are_written_canonically_and_empty_values_are_not_observations() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (
            ClaimSpec(
                "door",
                attributes=(
                    ("count", 3),
                    ("height_m", 2.5),
                    ("is_open", True),
                    ("note", "   "),
                    ("state", None),
                ),
            ),
        ),
    )

    (hypothesis,) = _accumulate([view]).hypotheses

    assert _pairs(hypothesis) == [("count", "3"), ("height_m", "2.5"), ("is_open", "true")]


def test_an_abstention_contributes_no_attribute() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (ClaimSpec("unknown", category="thing", attributes=(("color", "blue"),)),),
    )

    fused = _accumulate([view], abstention=frozenset({"unknown"}))

    assert fused.hypotheses == ()


def test_a_claim_without_category_or_attributes_adds_nothing() -> None:
    (hypothesis,) = _accumulate([View("run-a", "frame-0120", (ClaimSpec("door"),))]).hypotheses

    assert hypothesis.observed_attributes == ()


# --- serialization -------------------------------------------------------------------------------


def test_observed_attributes_survive_serialization() -> None:
    view = View(
        "run-a",
        "frame-0120",
        (ClaimSpec("door", category="door", attributes=(("color", "blue"),)),),
    )
    fused = _accumulate([view])

    assert decode_fused_evidence(encode_fused_evidence(fused)) == fused


def test_a_hypothesis_without_observations_is_encoded_exactly_as_before() -> None:
    fused = _accumulate([View("run-a", "frame-0120", (ClaimSpec("door"),))])

    (record,) = encode_fused_evidence(fused)["hypotheses"]

    assert set(record) == {"hypothesis_id", "label", "evidence"}


# --- contract ------------------------------------------------------------------------------------


def _supporting(claim: str) -> HypothesisEvidence:
    return HypothesisEvidence(
        contribution_id=EvidenceContributionId("contribution-a"),
        claim_id=ClaimId(claim),
        stance=EvidenceStance.SUPPORTING,
        role=HypothesisRole.PRIMARY,
    )


def _reference(claim: str) -> EvidenceReference:
    return EvidenceReference(
        contribution_id=EvidenceContributionId("contribution-a"), claim_id=ClaimId(claim)
    )


def test_an_attribute_must_cite_a_claim_that_supports_the_hypothesis() -> None:
    with pytest.raises(ValueError, match="does not support"):
        FusedHypothesis(
            hypothesis_id=FusedHypothesisId("hypothesis-0001"),
            label="door",
            evidence=(_supporting("claim-1"),),
            observed_attributes=(
                FusedAttribute(name="color", value="blue", evidence=(_reference("claim-9"),)),
            ),
        )


def test_attributes_must_be_listed_canonically() -> None:
    blue = FusedAttribute(name="color", value="blue", evidence=(_reference("claim-1"),))
    metal = FusedAttribute(name="material", value="metal", evidence=(_reference("claim-1"),))

    with pytest.raises(ValueError, match="observed_attributes"):
        FusedHypothesis(
            hypothesis_id=FusedHypothesisId("hypothesis-0001"),
            label="door",
            evidence=(_supporting("claim-1"),),
            observed_attributes=(metal, blue),
        )


@pytest.mark.parametrize(
    ("name", "value", "evidence"),
    [
        ("", "blue", True),
        ("color", " ", True),
        ("color", "blue", False),
    ],
)
def test_an_attribute_needs_a_name_a_value_and_evidence(
    name: str, value: str, evidence: bool
) -> None:
    with pytest.raises(ValueError):
        FusedAttribute(
            name=name, value=value, evidence=(_reference("claim-1"),) if evidence else ()
        )


def test_an_attribute_must_name_the_claim_not_only_the_view() -> None:
    with pytest.raises(ValueError, match="claim"):
        FusedAttribute(
            name="color",
            value="blue",
            evidence=(EvidenceReference(contribution_id=EvidenceContributionId("contribution-a")),),
        )
