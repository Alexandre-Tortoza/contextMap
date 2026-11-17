"""The interpreter's interaction factor survives fusion, aggregated on the fused hypothesis.

The factor is a semantic property of the object the label names, so a fused hypothesis carries one
value. Each physical observation counts once (the median of the factors its own claims declared) and
the observations are then aggregated by median, so a frame with many claims weighs the same as one
with a single claim. The aggregate names the claims and observations that declared the factor.
It is not support for the label and is never folded into a signal, and it stays
on the fused hypothesis rather than each evidence item so it does not leak into every downstream
copy of the evidence.
"""

import dataclasses

import pytest
from evidence_builders import ClaimSpec, View, build_scenario

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    INTERACTION_FACTOR_AGGREGATION_POLICY_ID,
    BaselineAccumulationPolicy,
    EvidenceChannel,
    EvidenceContributionId,
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
    FusedHypothesis,
    FusedHypothesisId,
    FusedInteractionFactor,
    HypothesisEvidence,
    accumulate_baseline_evidence,
)
from contextmap.semantic_fusion.serialization import decode_fused_evidence, encode_fused_evidence
from contextmap.semantic_interpretation import ClaimId, HypothesisRole


def _accumulate(views: list[View]) -> FusedEvidence:
    scenario = build_scenario(views)
    return accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
        policy=BaselineAccumulationPolicy(channels=frozenset({EvidenceChannel.SEMANTIC_CLAIMS})),
    )


def _factors(fused: FusedEvidence) -> list[float | None]:
    return [
        None if hypothesis.interaction_factor is None else hypothesis.interaction_factor.value
        for hypothesis in fused.hypotheses
    ]


def test_the_factor_is_carried_on_the_fused_hypothesis() -> None:
    fused = _accumulate(
        [View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=0.5),))]
    )

    assert _factors(fused) == [0.5]


def test_a_hypothesis_without_a_factor_carries_none() -> None:
    fused = _accumulate([View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9),))])

    assert _factors(fused) == [None]


def test_the_factor_is_the_median_across_the_physical_observations() -> None:
    fused = _accumulate(
        [
            View(
                "run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=0.2),)
            ),
            View(
                "run-a", "frame-0130", (ClaimSpec("door", confidence=0.9, interaction_factor=0.8),)
            ),
            View(
                "run-a", "frame-0140", (ClaimSpec("door", confidence=0.9, interaction_factor=0.5),)
            ),
        ]
    )

    assert _factors(fused) == [0.5]


def test_the_factor_survives_serialization() -> None:
    fused = _accumulate(
        [View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=1.25),))]
    )

    restored = decode_fused_evidence(encode_fused_evidence(fused))

    assert restored == fused
    assert _factors(restored) == [1.25]


def test_a_nonpositive_claim_factor_is_dropped_rather_than_aggregated() -> None:
    fused = _accumulate(
        [View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=0.0),))]
    )

    assert _factors(fused) == [None]


def test_the_fused_hypothesis_rejects_a_nonpositive_factor() -> None:
    with pytest.raises(ValueError, match="interaction_factor"):
        FusedHypothesis(
            hypothesis_id=FusedHypothesisId("hypothesis-0001"),
            label="door",
            evidence=(
                HypothesisEvidence(
                    contribution_id=EvidenceContributionId("contribution"),
                    claim_id=ClaimId("claim"),
                    stance=EvidenceStance.SUPPORTING,
                    role=HypothesisRole.PRIMARY,
                ),
            ),
            interaction_factor=FusedInteractionFactor(
                value=-1.0,
                aggregation_policy_id=INTERACTION_FACTOR_AGGREGATION_POLICY_ID,
                physical_observation_ids=(SourceObservationId("frame"),),
                evidence=(
                    EvidenceReference(
                        contribution_id=EvidenceContributionId("contribution"),
                        claim_id=ClaimId("claim"),
                    ),
                ),
                minimum=-1.0,
                maximum=-1.0,
            ),
        )


def test_an_observation_with_three_claims_weighs_the_same_as_one_with_a_single_claim() -> None:
    fused = _accumulate(
        [
            View(
                "run-a",
                "frame-0120",
                tuple(
                    ClaimSpec("door", confidence=0.9, interaction_factor=2.0, claim_id=f"c-{i}")
                    for i in range(3)
                ),
            ),
            View(
                "run-a", "frame-0130", (ClaimSpec("door", confidence=0.9, interaction_factor=1.0),)
            ),
        ]
    )

    # Por claim a mediana seria 2.0; por observação física são duas amostras (2.0 e 1.0).
    assert _factors(fused) == [1.5]


def test_the_factor_names_only_the_claims_and_observations_that_declared_it() -> None:
    fused = _accumulate(
        [
            View(
                "run-a",
                "frame-0120",
                (
                    ClaimSpec("door", confidence=0.9, interaction_factor=2.0, claim_id="declares"),
                    ClaimSpec("door", confidence=0.8, claim_id="silent"),
                ),
            ),
            View("run-a", "frame-0130", (ClaimSpec("door", confidence=0.9),)),
        ]
    )

    factor = fused.hypotheses[0].interaction_factor
    assert factor is not None
    assert [str(ref.claim_id) for ref in factor.evidence] == ["declares"]
    assert factor.physical_observation_ids == (SourceObservationId("frame-0120"),)
    assert factor.aggregation_policy_id == INTERACTION_FACTOR_AGGREGATION_POLICY_ID
    assert (factor.minimum, factor.maximum) == (2.0, 2.0)


def test_the_factor_reports_its_dispersion() -> None:
    fused = _accumulate(
        [
            View(
                "run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=0.2),)
            ),
            View(
                "run-a", "frame-0130", (ClaimSpec("door", confidence=0.9, interaction_factor=0.8),)
            ),
        ]
    )

    factor = fused.hypotheses[0].interaction_factor
    assert factor is not None
    assert (factor.minimum, factor.maximum, factor.value) == (0.2, 0.8, 0.5)


def test_the_factor_rejects_observations_that_are_not_the_ones_of_its_cited_claims() -> None:
    fused = _accumulate(
        [View("run-a", "frame-0120", (ClaimSpec("door", confidence=0.9, interaction_factor=2.0),))]
    )
    hypothesis = fused.hypotheses[0]
    factor = hypothesis.interaction_factor
    assert factor is not None
    contradictory = dataclasses.replace(
        factor, physical_observation_ids=(SourceObservationId("frame-999"),)
    )

    with pytest.raises(ValueError, match="physical_observation_ids"):
        dataclasses.replace(
            fused,
            hypotheses=(dataclasses.replace(hypothesis, interaction_factor=contradictory),),
        )
