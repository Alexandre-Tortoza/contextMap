"""Belief after resolution: existence is not belief (``resolved-belief-min-support-v1``).

A resolved entity keeps every hypothesis of its members. The belief rule adds a separate,
versioned judgement on top of that union: per label (under Semantic Fusion's typographic key) it
counts the *distinct* physical observations that support it, across every member, and compares the
count with a declared minimum. The union itself never changes, the ambiguity records never change,
and no score is read: support is a count of frames.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from itertools import pairwise

import pytest
from mapping_builders import make_evidence_item, make_hypothesis
from resolution_builders import decision_between
from resolution_entity_builders import entity_at, fused_id

from contextmap.entity_resolution import (
    MATERIALIZATION_POLICY_ID,
    RESOLVED_BELIEF_RULE_ID,
    EntityResolutionRunId,
    LabelSupport,
    MaterializationError,
    ResolutionOutcome,
    ResolvedBeliefPolicy,
    ResolvedEntity,
    ResolvedSemanticBelief,
    SupportStatus,
    derive_resolved_belief,
    materialization_policy,
    materialize_resolved_entities,
)
from contextmap.entity_resolution._codec import from_record, to_record
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity, EntityHypothesis

RUN = EntityResolutionRunId("resolution-run-0001")
POLICY = ResolvedBeliefPolicy(min_supporting_physical_observations=2)


def _hypothesis(
    label: str, frames: Sequence[str], *, number: int, index: int = 1
) -> EntityHypothesis:
    return dataclasses.replace(
        make_hypothesis(
            f"hypothesis-{index:04d}",
            label,
            fused_evidence_id=fused_id(number),
            evidence=(
                make_evidence_item(
                    contribution=f"contribution--support-{number:06d}--spatial-a",
                    claim=f"claim-{number:04d}-{index:04d}",
                ),
            ),
        ),
        supporting_physical_observations=tuple(SourceObservationId(item) for item in frames),
    )


def _entity(
    name: str, number: int, labels: dict[str, Sequence[str]], *, seconds: Sequence[int]
) -> Entity:
    """One entity whose hypotheses list the frames that support each label."""
    return entity_at(
        name,
        (0.0, 0.0, 0.0),
        support_number=number,
        seconds=seconds,
        hypotheses=tuple(
            _hypothesis(label, frames, number=number, index=index)
            for index, (label, frames) in enumerate(sorted(labels.items()), start=1)
        ),
    )


def _merge(*entities: Entity, policy: ResolvedBeliefPolicy | None = POLICY) -> ResolvedEntity:
    decisions = [
        decision_between(first.reference, second.reference, ResolutionOutcome.MATCH)
        for first, second in pairwise(entities)
    ]
    materialized = materialize_resolved_entities(
        entities, decisions, resolution_run_id=RUN, belief_policy=policy
    )
    (resolved,) = materialized.resolved.entities
    return resolved


def _belief(resolved: ResolvedEntity) -> ResolvedSemanticBelief:
    belief = resolved.semantic_state.belief
    assert belief is not None
    return belief


# --- the four statuses ---------------------------------------------------------------------------


def test_a_single_frame_never_yields_a_primary_label() -> None:
    """B1 regression: with one frame every entity has one observation, so nothing is supported."""
    entities = [
        _entity("a", 1, {"door": ["frame-0055"]}, seconds=[55]),
        _entity("b", 2, {"metal pole": ["frame-0055"], "wall": ["frame-0055"]}, seconds=[55]),
    ]
    separated = [dataclasses.replace(item) for item in entities]

    materialized = materialize_resolved_entities(
        separated, [], resolution_run_id=RUN, belief_policy=POLICY
    )

    beliefs = [_belief(item) for item in materialized.resolved.entities]
    assert [item.support_status for item in beliefs] == [SupportStatus.TENTATIVE] * 2
    assert all(item.primary_label is None for item in beliefs)
    # Nada é descartado: toda hipótese continua visível com a sua contagem.
    assert sorted(
        label for item in beliefs for group in item.label_support for label in group.labels
    ) == [
        "door",
        "metal pole",
        "wall",
    ]


def test_the_same_label_seen_in_two_frames_is_supported_and_primary() -> None:
    resolved = _merge(
        _entity("a", 1, {"door": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"]}, seconds=[20]),
    )

    belief = _belief(resolved)
    assert belief.support_status is SupportStatus.SUPPORTED
    assert belief.primary_label == "door"
    (door,) = belief.label_support
    assert door.supporting_physical_observations == (
        SourceObservationId("frame-0010"),
        SourceObservationId("frame-0020"),
    )


def test_two_labels_that_both_reach_the_minimum_are_contested_without_a_primary() -> None:
    resolved = _merge(
        _entity("a", 1, {"door": ["frame-0010"], "wall": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"], "wall": ["frame-0020"]}, seconds=[20]),
    )

    belief = _belief(resolved)
    assert belief.support_status is SupportStatus.CONTESTED
    assert belief.primary_label is None
    assert [len(item.supporting_physical_observations) for item in belief.label_support] == [2, 2]


def test_only_one_label_reaching_the_minimum_is_supported_and_the_others_stay_visible() -> None:
    resolved = _merge(
        _entity("a", 1, {"door": ["frame-0010"], "wall": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"]}, seconds=[20]),
    )

    belief = _belief(resolved)
    assert belief.support_status is SupportStatus.SUPPORTED
    assert belief.primary_label == "door"
    assert [group.labels for group in belief.label_support] == [("door",), ("wall",)]
    # O registro de ambiguidade dos membros não muda: a crença é estado derivado separado.
    assert resolved.semantic_state.ambiguity_state.value == "ambiguous"


def test_an_entity_without_hypotheses_has_insufficient_evidence() -> None:
    (resolved,) = materialize_resolved_entities(
        [entity_at("a", support_number=1, hypotheses=())],
        [],
        resolution_run_id=RUN,
        belief_policy=POLICY,
    ).resolved.entities

    belief = _belief(resolved)
    assert belief.support_status is SupportStatus.INSUFFICIENT_EVIDENCE
    assert (belief.primary_label, belief.label_support, belief.semantic_entropy_bits) == (
        None,
        (),
        0.0,
    )


# --- counting ------------------------------------------------------------------------------------


def test_the_same_physical_observation_in_two_members_counts_once() -> None:
    resolved = _merge(
        _entity("a", 1, {"door": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0010"]}, seconds=[10]),
    )

    belief = _belief(resolved)
    assert belief.support_status is SupportStatus.TENTATIVE
    (door,) = belief.label_support
    assert door.supporting_physical_observations == (SourceObservationId("frame-0010"),)


def test_labels_that_differ_only_typographically_are_counted_together() -> None:
    resolved = _merge(
        _entity("a", 1, {"Door": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"]}, seconds=[20]),
    )

    belief = _belief(resolved)
    (group,) = belief.label_support
    assert (group.label_key, group.labels) == ("door", ("Door", "door"))
    assert belief.support_status is SupportStatus.SUPPORTED
    assert belief.primary_label in group.labels


@pytest.mark.parametrize(
    ("counts", "expected_bits"),
    [
        ((1,), 0.0),
        ((1, 1), 1.0),
        ((2, 2), 1.0),
        ((2, 1, 1), 1.5),
        ((3, 1), 0.8112781244591328),
    ],
)
def test_the_entropy_is_over_the_counts_of_observations(
    counts: tuple[int, ...], expected_bits: float
) -> None:
    labels = [f"label-{index}" for index in range(len(counts))]
    hypotheses = tuple(
        _hypothesis(
            label, [f"frame-{frame:04d}" for frame in range(count)], number=1, index=index + 1
        )
        for index, (label, count) in enumerate(zip(labels, counts, strict=True))
    )

    belief = derive_resolved_belief(hypotheses, POLICY)

    assert math.isclose(belief.semantic_entropy_bits, expected_bits, abs_tol=1e-12)


def test_a_minimum_of_one_supports_a_lone_label_from_a_single_frame() -> None:
    belief = derive_resolved_belief(
        (_hypothesis("door", ["frame-0010"], number=1),),
        ResolvedBeliefPolicy(min_supporting_physical_observations=1),
    )

    assert (belief.support_status, belief.primary_label) == (SupportStatus.SUPPORTED, "door")


# --- no policy, no belief ------------------------------------------------------------------------


def test_without_a_policy_nothing_changes() -> None:
    entities = [
        _entity("a", 1, {"door": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"]}, seconds=[20]),
    ]

    resolved = _merge(*entities, policy=None)

    assert resolved.semantic_state.belief is None
    assert "belief" not in to_record(resolved)["semantic_state"]
    assert resolved.provenance.policy == materialization_policy()


def test_the_belief_is_recorded_and_round_trips() -> None:
    resolved = _merge(
        _entity("a", 1, {"door": ["frame-0010"]}, seconds=[10]),
        _entity("b", 2, {"door": ["frame-0020"]}, seconds=[20]),
    )

    record = to_record(resolved)

    assert record["semantic_state"]["belief"]["support_status"] == "supported"
    assert from_record(ResolvedEntity, record) == resolved


def test_the_materialization_policy_names_the_belief_rule_and_its_minimum() -> None:
    with_belief = materialization_policy(belief=POLICY)
    other_minimum = materialization_policy(
        belief=ResolvedBeliefPolicy(min_supporting_physical_observations=3)
    )

    assert with_belief.policy_id == MATERIALIZATION_POLICY_ID
    assert len({with_belief, other_minimum, materialization_policy()}) == 3


def test_a_hypothesis_without_recorded_support_is_refused_not_guessed() -> None:
    entity = entity_at("a", support_number=1, hypotheses=(make_hypothesis(),))

    with pytest.raises(MaterializationError, match="supporting physical observations"):
        materialize_resolved_entities([entity], [], resolution_run_id=RUN, belief_policy=POLICY)


# --- contract ------------------------------------------------------------------------------------


def _support(label: str, frames: Sequence[str]) -> LabelSupport:
    return LabelSupport(
        label_key=label,
        labels=(label,),
        supporting_physical_observations=tuple(SourceObservationId(item) for item in frames),
    )


def test_a_belief_whose_status_disagrees_with_its_counts_is_refused() -> None:
    with pytest.raises(ValueError, match="support_status"):
        ResolvedSemanticBelief(
            rule_id=RESOLVED_BELIEF_RULE_ID,
            min_supporting_physical_observations=2,
            support_status=SupportStatus.SUPPORTED,
            primary_label="door",
            label_support=(_support("door", ["frame-0010"]),),
            semantic_entropy_bits=0.0,
        )


def test_a_primary_label_outside_a_supported_belief_is_refused() -> None:
    with pytest.raises(ValueError, match="primary_label"):
        ResolvedSemanticBelief(
            rule_id=RESOLVED_BELIEF_RULE_ID,
            min_supporting_physical_observations=2,
            support_status=SupportStatus.TENTATIVE,
            primary_label="door",
            label_support=(_support("door", ["frame-0010"]),),
            semantic_entropy_bits=0.0,
        )


def test_an_entropy_that_is_not_the_one_of_the_counts_is_refused() -> None:
    with pytest.raises(ValueError, match="semantic_entropy_bits"):
        ResolvedSemanticBelief(
            rule_id=RESOLVED_BELIEF_RULE_ID,
            min_supporting_physical_observations=2,
            support_status=SupportStatus.TENTATIVE,
            primary_label=None,
            label_support=(_support("door", ["frame-0010"]), _support("wall", ["frame-0010"])),
            semantic_entropy_bits=0.5,
        )


@pytest.mark.parametrize("minimum", [0, -1])
def test_the_policy_minimum_is_a_count_of_at_least_one(minimum: int) -> None:
    with pytest.raises(ValueError, match="min_supporting_physical_observations"):
        ResolvedBeliefPolicy(min_supporting_physical_observations=minimum)
