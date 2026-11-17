"""The map's semantic state carries observed attributes and, when derived, the belief.

Both are optional additions of schema 0.2.0: an entity without attributes and without a belief is
written exactly as a 0.1.0 entity, so maps that do not use them keep their bytes. When a belief
exists, the state says whether the label is ``tentative``, ``supported``, ``contested`` or has
``insufficient_evidence``, exposes a primary label only when supported, and each label says how
many distinct physical observations support it -- a count, never a score.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from context_map_builders import (
    FUSION_ARTIFACT_ID,
    context_map,
    entity,
    entity_capabilities,
    fused_origin,
    hypothesis,
    metadata,
    origin,
    policy,
    upstream_record,
)

from contextmap.artifact import (
    AmbiguityStatus,
    ContextAttribute,
    ContextAttributeOrigin,
    ContextSemanticState,
    ContextSupportStatus,
    DerivationKind,
    LabelHypothesis,
    ReferenceIntegrityError,
    context_map_from_record,
    context_map_to_record,
)


def _stated(name: str, value: str, *, artifact_id: str = FUSION_ARTIFACT_ID) -> ContextAttribute:
    return ContextAttribute(
        name=name,
        value=value,
        origin=origin(
            DerivationKind.MULTIVIEW_FUSED,
            upstream_record(artifact_id, "fused--support-000001"),
            policy_ref=policy("claim-attribute"),
        ),
    )


def _counted(label: str, count: int) -> LabelHypothesis:
    return replace(hypothesis(label), supporting_physical_observation_count=count)


def _believed(
    support_status: ContextSupportStatus,
    counts: dict[str, int],
    *,
    primary_label: str | None = None,
    status: AmbiguityStatus | None = None,
) -> ContextSemanticState:
    labels = sorted(counts)
    return ContextSemanticState(
        status=status
        or (
            AmbiguityStatus.INSUFFICIENT_EVIDENCE
            if not labels
            else AmbiguityStatus.UNAMBIGUOUS
            if len(labels) == 1
            else AmbiguityStatus.AMBIGUOUS
        ),
        hypotheses=tuple(_counted(label, counts[label]) for label in labels),
        support_status=support_status,
        primary_label=primary_label,
    )


# --- valid states --------------------------------------------------------------------------------


def test_a_supported_state_exposes_its_primary_label_and_the_counts() -> None:
    state = _believed(ContextSupportStatus.SUPPORTED, {"door": 2, "wall": 1}, primary_label="door")

    assert state.primary_label == "door"
    assert [item.supporting_physical_observation_count for item in state.hypotheses] == [2, 1]


@pytest.mark.parametrize(
    ("support_status", "counts"),
    [
        (ContextSupportStatus.TENTATIVE, {"door": 1}),
        (ContextSupportStatus.CONTESTED, {"door": 2, "wall": 2}),
        (ContextSupportStatus.INSUFFICIENT_EVIDENCE, {}),
    ],
)
def test_other_support_statuses_expose_no_primary_label(
    support_status: ContextSupportStatus, counts: dict[str, int]
) -> None:
    assert _believed(support_status, counts).primary_label is None


def test_observed_attributes_are_kept_in_canonical_order() -> None:
    state = ContextSemanticState(
        status=AmbiguityStatus.UNAMBIGUOUS,
        hypotheses=(hypothesis("door"),),
        attributes=(
            _stated("color", "blue"),
            _stated("color", "gray"),
            _stated("material", "metal"),
        ),
    )

    assert [(item.name, item.value) for item in state.attributes] == [
        ("color", "blue"),
        ("color", "gray"),
        ("material", "metal"),
    ]


@pytest.mark.parametrize(
    "attributes",
    [
        (
            replace(_stated("interaction_radius_m", "1.5"), derivation_id="radius-v1@sha256:aaa"),
            replace(_stated("interaction_radius_m", "1.5"), derivation_id="radius-v1@sha256:bbb"),
        ),
        (
            replace(
                _stated("interaction_radius_m", "1.5"),
                attribute_origin=ContextAttributeOrigin.DERIVED,
                derivation_id="radius-v1@sha256:aaa",
            ),
            _stated("interaction_radius_m", "1.5"),
        ),
    ],
)
def test_equal_attribute_values_with_distinct_provenance_round_trip(
    attributes: tuple[ContextAttribute, ...],
) -> None:
    state = ContextSemanticState(
        status=AmbiguityStatus.UNAMBIGUOUS,
        hypotheses=(hypothesis("door"),),
        attributes=attributes,
    )
    original = _map_with(state)

    assert (
        context_map_from_record(
            json.loads(json.dumps(context_map_to_record(original), sort_keys=True))
        )
        == original
    )


# --- invalid states ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (
            {"support_status": ContextSupportStatus.TENTATIVE, "primary_label": "door"},
            "primary_label",
        ),
        ({"support_status": ContextSupportStatus.SUPPORTED}, "primary_label"),
        (
            {"support_status": ContextSupportStatus.SUPPORTED, "primary_label": "window"},
            "primary_label",
        ),
    ],
)
def test_a_primary_label_needs_a_supported_state_and_one_of_its_labels(
    fields: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ContextSemanticState(
            status=AmbiguityStatus.UNAMBIGUOUS, hypotheses=(_counted("door", 2),), **fields
        )


def test_a_primary_label_without_a_belief_is_refused() -> None:
    with pytest.raises(ValueError, match="primary_label"):
        ContextSemanticState(
            status=AmbiguityStatus.UNAMBIGUOUS,
            hypotheses=(hypothesis("door"),),
            primary_label="door",
        )


def test_counts_exist_exactly_when_a_belief_exists() -> None:
    with pytest.raises(ValueError, match="supporting_physical_observation_count"):
        ContextSemanticState(status=AmbiguityStatus.UNAMBIGUOUS, hypotheses=(_counted("door", 1),))
    with pytest.raises(ValueError, match="supporting_physical_observation_count"):
        ContextSemanticState(
            status=AmbiguityStatus.UNAMBIGUOUS,
            hypotheses=(hypothesis("door"),),
            support_status=ContextSupportStatus.TENTATIVE,
        )


@pytest.mark.parametrize("count", [0, -1])
def test_a_count_is_at_least_one(count: int) -> None:
    with pytest.raises(ValueError, match="supporting_physical_observation_count"):
        _counted("door", count)


def test_a_primary_label_has_the_largest_count() -> None:
    with pytest.raises(ValueError, match="primary_label"):
        _believed(ContextSupportStatus.SUPPORTED, {"door": 1, "wall": 3}, primary_label="door")


def test_a_contested_state_keeps_at_least_two_hypotheses() -> None:
    with pytest.raises(ValueError, match="contested"):
        _believed(ContextSupportStatus.CONTESTED, {"door": 2})


def test_insufficient_evidence_has_no_hypothesis_and_the_others_have_some() -> None:
    with pytest.raises(ValueError, match="insufficient_evidence"):
        _believed(
            ContextSupportStatus.INSUFFICIENT_EVIDENCE,
            {"door": 1},
            status=AmbiguityStatus.UNAMBIGUOUS,
        )
    with pytest.raises(ValueError, match="tentative"):
        ContextSemanticState(
            status=AmbiguityStatus.INSUFFICIENT_EVIDENCE,
            hypotheses=(),
            support_status=ContextSupportStatus.TENTATIVE,
        )


def test_attributes_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="attributes"):
        ContextSemanticState(
            status=AmbiguityStatus.UNAMBIGUOUS,
            hypotheses=(hypothesis("door"),),
            attributes=(_stated("material", "metal"), _stated("color", "blue")),
        )


@pytest.mark.parametrize(("name", "value"), [("", "blue"), ("color", "  ")])
def test_an_attribute_needs_a_name_and_a_value(name: str, value: str) -> None:
    with pytest.raises(ValueError):
        _stated(name, value)


# --- records -------------------------------------------------------------------------------------


def _map_with(state: ContextSemanticState) -> Any:
    return context_map(
        metadata=metadata(capabilities=entity_capabilities()),
        entities=(entity(semantic_state=state),),
    )


def test_a_state_without_belief_or_attributes_is_written_exactly_as_before() -> None:
    record = context_map_to_record(
        _map_with(
            ContextSemanticState(
                status=AmbiguityStatus.UNAMBIGUOUS,
                hypotheses=(hypothesis("door", fused_origin("a")),),
            )
        )
    )

    (written,) = record["entities"]
    assert set(written["semantic_state"]) == {"status", "hypotheses"}
    assert set(written["semantic_state"]["hypotheses"][0]) == {"label", "origin"}


def test_belief_and_attributes_round_trip_through_plain_json() -> None:
    state = replace(
        _believed(ContextSupportStatus.SUPPORTED, {"door": 2, "wall": 1}, primary_label="door"),
        attributes=(_stated("color", "blue"), _stated("material", "metal")),
    )
    original = _map_with(state)

    record = json.loads(json.dumps(context_map_to_record(original), sort_keys=True))

    assert record["entities"][0]["semantic_state"]["support_status"] == "supported"
    assert context_map_from_record(record) == original


def test_an_attribute_origin_must_be_in_the_lineage() -> None:
    state = ContextSemanticState(
        status=AmbiguityStatus.UNAMBIGUOUS,
        hypotheses=(hypothesis("door"),),
        attributes=(_stated("color", "blue", artifact_id="semantic-fusion--unknown"),),
    )

    with pytest.raises(ReferenceIntegrityError, match="attribute"):
        _map_with(state)
