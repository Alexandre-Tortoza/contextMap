"""The two-layer comparison of incremental context builds pre-registered in issue #501.

Layer 1 is exact: geometry, counts and states that no semantic backend reaches. Layer 2 is
statistical: agreement of claims and entity semantics, whose partition effect is tested against
the rerun noise with a seeded paired cluster bootstrap. Everything here runs on small, explicit
fixtures; the end-to-end reading of real build artifacts is covered in ``tests/end_to_end/``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from contextmap.evaluation import (
    BuildCounts,
    ClaimKey,
    ClusterTerms,
    EquivalenceDecision,
    GeometryKey,
    LayerOneOutcome,
    LayerOneState,
    claim_agreement,
    compare_layer_one,
    decide_equivalence,
    encode_incremental_context,
    judge_downstream_only,
    judge_layer_one,
    judge_repeated_inference,
    paired_agreement,
    partition_effect,
    stage_dispositions,
)
from contextmap.runtime import (
    ArtifactRef,
    ExecutionRecord,
    ReuseDecision,
    StageRecord,
    write_execution_record,
)

A = GeometryKey(("g-001", "g-002"))
B = GeometryKey(("g-003",))
C = GeometryKey(("g-004", "g-005", "g-006"))


def _claim(frame: str, hypothesis: str, region: str | None = "region-0001") -> ClaimKey:
    return (frame, region, hypothesis, "object", None, 0.9)


def _state(**changes: object) -> LayerOneState:
    base = LayerOneState(
        map_id="map-1",
        physical_observations=3,
        inference_results=3,
        excluded_observations=1,
        supports=(A, B),
        contributions=((A, "frame-1", "region-0001"), (B, "frame-2", "region-0001")),
        entities=(A, B),
        rejections=(),
        resolved_entities=((A, (A,)), (B, (B,))),
        relations=(("next_to", A, B, "supported", ()),),
        capabilities=("entities", "geometry", "relations"),
        regions=(("frame-1", (((1.0, 2.0, 3.0, 4.0, None, True),),)),),
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


# --- Agreement terms --------------------------------------------------------------------


def test_claim_agreement_is_the_jaccard_of_the_canonical_claim_keys_summed_over_frames() -> None:
    # A mesma definição dos 29/90 do v0.1.0: interseção sobre união das chaves de todos os frames.
    left = {
        "frame-1": frozenset({_claim("frame-1", "door"), _claim("frame-1", "wall")}),
        "frame-2": frozenset(),
        "frame-3": frozenset({_claim("frame-3", "chair")}),
    }
    right = {
        "frame-1": frozenset({_claim("frame-1", "wall"), _claim("frame-1", "floor")}),
        "frame-2": frozenset(),
        "frame-3": frozenset(),
    }

    terms = claim_agreement(left, right)

    assert terms.clusters == ("frame-1", "frame-2", "frame-3")
    assert terms.agreeing == (1, 0, 0)
    assert terms.compared == (3, 0, 1)
    assert terms.value == pytest.approx(1 / 4)


def test_a_frame_perceived_by_only_one_side_still_counts_its_claims_as_disagreement() -> None:
    terms = claim_agreement({"frame-1": frozenset({_claim("frame-1", "door")})}, {})

    assert terms.clusters == ("frame-1",)
    assert (terms.agreeing, terms.compared) == ((0,), (1,))
    assert terms.value == 0.0


def test_paired_agreement_compares_only_the_keys_both_sides_have() -> None:
    left = {A: "unambiguous", B: "ambiguous", C: "conflicting"}
    right = {A: "unambiguous", B: "unambiguous"}

    terms = paired_agreement(left, right)

    # C não tem par: fica de fora do denominador, em vez de contar como discordância.
    assert terms.clusters == (str(A), str(B), str(C))
    assert terms.agreeing == (1, 0, 0)
    assert terms.compared == (1, 1, 0)
    assert terms.value == pytest.approx(1 / 2)


def test_an_agreement_with_nothing_to_compare_has_no_value() -> None:
    assert paired_agreement({A: "x"}, {B: "x"}).value is None
    assert claim_agreement({"frame-1": frozenset()}, {"frame-1": frozenset()}).value is None


def test_label_sets_are_compared_as_sets() -> None:
    left = {A: frozenset({"door", "wall"})}
    right = {A: frozenset({"wall", "door"})}

    assert paired_agreement(left, right).value == 1.0


# --- Partition effect -------------------------------------------------------------------


def _terms(agreeing: tuple[int, ...], compared: tuple[int, ...]) -> ClusterTerms:
    clusters = tuple(f"c{index:03d}" for index in range(len(agreeing)))
    return ClusterTerms(clusters=clusters, agreeing=agreeing, compared=compared)


def test_the_partition_effect_is_the_mean_agreement_with_the_chunks_minus_the_rerun_agreement() -> (
    None
):
    full = _terms((1, 1, 1, 1), (1, 1, 1, 1))
    half = _terms((1, 1, 0, 0), (1, 1, 1, 1))

    effect = partition_effect(full, full, half, resamples=200, seed=7, confidence=0.95)

    assert effect.w1_c == 1.0 and effect.w2_c == 1.0 and effect.w1_w2 == 0.5
    assert effect.point == pytest.approx(0.5)


def test_identical_agreement_everywhere_gives_a_degenerate_interval_at_zero() -> None:
    same = _terms((3, 2, 5), (3, 2, 5))

    effect = partition_effect(same, same, same, resamples=500, seed=1, confidence=0.95)

    assert effect.point == 0.0
    assert (effect.lower, effect.upper) == (0.0, 0.0)
    assert effect.undefined_resamples == 0


def test_the_bootstrap_is_reproducible_from_its_seed() -> None:
    w1_c = _terms((1, 0, 1, 1, 0, 1), (1, 1, 1, 1, 1, 1))
    w2_c = _terms((1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 1, 1))
    w1_w2 = _terms((1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1))

    first = partition_effect(w1_c, w2_c, w1_w2, resamples=1000, seed=11, confidence=0.95)
    again = partition_effect(w1_c, w2_c, w1_w2, resamples=1000, seed=11, confidence=0.95)

    assert first == again
    assert first.lower is not None and first.upper is not None and first.point is not None
    assert first.lower <= first.point <= first.upper
    assert (first.seed, first.resamples, first.confidence) == (11, 1000, 0.95)


def test_one_cluster_can_only_resample_itself() -> None:
    w1_c = _terms((2,), (4,))
    w1_w2 = _terms((3,), (4,))

    effect = partition_effect(w1_c, w1_c, w1_w2, resamples=100, seed=3, confidence=0.95)

    assert effect.point == pytest.approx(-0.25)
    assert (effect.lower, effect.upper) == pytest.approx((-0.25, -0.25))


def test_the_three_pairs_are_resampled_on_the_same_clusters() -> None:
    # Pareado: o mesmo cluster sorteado entra nos três pares. Com agr(W1,C) = agr(W2,C) =
    # agr(W1,W2) cluster a cluster, toda reamostragem dá exatamente zero.
    terms = _terms((1, 0, 3, 0), (1, 2, 3, 4))

    effect = partition_effect(terms, terms, terms, resamples=300, seed=5, confidence=0.95)

    assert (effect.lower, effect.upper) == (0.0, 0.0)


def test_pairs_over_different_clusters_are_aligned_on_their_union() -> None:
    left = ClusterTerms(clusters=("a", "b"), agreeing=(1, 1), compared=(1, 1))
    right = ClusterTerms(clusters=("b", "c"), agreeing=(1, 0), compared=(1, 1))

    effect = partition_effect(left, left, right, resamples=50, seed=2, confidence=0.9)

    assert effect.clusters == 3
    assert effect.point == pytest.approx(1.0 - 0.5)


def test_a_resample_with_nothing_to_compare_is_counted_and_left_out_of_the_interval() -> None:
    terms = _terms((1, 0), (1, 0))

    effect = partition_effect(terms, terms, terms, resamples=400, seed=9, confidence=0.95)

    # P(sortear só o cluster vazio duas vezes) = 1/4: uma fração das reamostragens é indefinida.
    assert 0 < effect.undefined_resamples < 400
    assert (effect.lower, effect.upper) == (0.0, 0.0)


def test_an_effect_with_nothing_to_compare_is_not_measurable() -> None:
    empty = _terms((0, 0), (0, 0))

    effect = partition_effect(empty, empty, empty, resamples=10, seed=1, confidence=0.95)

    assert effect.point is None and effect.lower is None and effect.upper is None
    assert decide_equivalence(effect, delta=0.1) is EquivalenceDecision.NOT_MEASURABLE


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"resamples": 0, "seed": 1, "confidence": 0.95}, "resamples"),
        ({"resamples": 10, "seed": 1, "confidence": 1.0}, "confidence"),
        ({"resamples": 10, "seed": -1, "confidence": 0.95}, "seed"),
    ],
)
def test_an_impossible_bootstrap_is_refused(options: dict[str, float], message: str) -> None:
    terms = _terms((1,), (1,))

    with pytest.raises(ValueError, match=message):
        partition_effect(terms, terms, terms, **options)  # type: ignore[arg-type]


def test_misaligned_terms_are_refused() -> None:
    with pytest.raises(ValueError, match="one count per cluster"):
        ClusterTerms(clusters=("a", "b"), agreeing=(1,), compared=(1, 1))
    with pytest.raises(ValueError, match="cannot exceed"):
        ClusterTerms(clusters=("a",), agreeing=(2,), compared=(1,))


# --- Decision rule ----------------------------------------------------------------------


def _effect(lower: float, upper: float) -> object:
    terms = _terms((1,), (1,))
    effect = partition_effect(terms, terms, terms, resamples=1, seed=0, confidence=0.95)
    return dataclasses.replace(effect, point=(lower + upper) / 2, lower=lower, upper=upper)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-0.05, 0.03, EquivalenceDecision.EQUIVALENT_WITHIN_DELTA),
        (-0.10, 0.00, EquivalenceDecision.EQUIVALENT_WITHIN_DELTA),
        (-0.30, -0.01, EquivalenceDecision.REJECTS_INVARIANCE),
        # A regra de rejeição vem primeiro, como no pré-registro: um IC inteiro abaixo de zero
        # rejeita mesmo quando o limite inferior também está dentro de δ.
        (-0.05, -0.01, EquivalenceDecision.REJECTS_INVARIANCE),
        (-0.20, 0.05, EquivalenceDecision.INCONCLUSIVE),
    ],
)
def test_the_pre_registered_decision_rule(
    lower: float, upper: float, expected: EquivalenceDecision
) -> None:
    assert decide_equivalence(_effect(lower, upper), delta=0.10) is expected  # type: ignore[arg-type]


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="delta"):
        decide_equivalence(_effect(0.0, 0.0), delta=-0.1)  # type: ignore[arg-type]


# --- Layer 1 ----------------------------------------------------------------------------


def test_equal_states_compare_equal_in_every_dimension() -> None:
    comparison = compare_layer_one(_state(), _state())

    assert comparison.equal
    assert all(item.equal for item in comparison.dimensions)
    assert comparison.region_frames_differing == ()


def test_a_divergent_dimension_is_named_with_what_each_side_has_alone() -> None:
    comparison = compare_layer_one(_state(), _state(entities=(A, C)))

    assert not comparison.equal
    (entities,) = [item for item in comparison.dimensions if item.name == "entities"]
    assert not entities.equal
    assert (entities.only_left, entities.only_right) == (1, 1)
    assert entities.left_examples == (str(B),)
    assert entities.right_examples == (str(C),)
    assert all(item.equal for item in comparison.dimensions if item.name != "entities")


def test_a_scalar_dimension_reports_both_values() -> None:
    comparison = compare_layer_one(_state(), _state(inference_results=6))

    (results,) = [item for item in comparison.dimensions if item.name == "inference_results"]
    assert (results.left_examples, results.right_examples) == (("3",), ("6",))


def test_contributions_are_compared_as_a_multiset() -> None:
    doubled = _state().contributions * 2

    comparison = compare_layer_one(_state(), _state(contributions=doubled))

    (contributions,) = [item for item in comparison.dimensions if item.name == "contributions"]
    assert (contributions.only_left, contributions.only_right) == (0, 2)


def test_regions_are_a_diagnostic_that_never_decides_equality() -> None:
    other = (("frame-1", (((1.0, 2.0, 3.0, 5.0, None, True),),)),)

    comparison = compare_layer_one(_state(), _state(regions=other))

    assert comparison.equal
    assert comparison.region_frames_differing == ("frame-1",)


def test_the_examples_of_a_large_divergence_are_capped() -> None:
    many = tuple(GeometryKey((f"g-{index:05d}",)) for index in range(100))

    comparison = compare_layer_one(_state(), _state(entities=many))

    (entities,) = [item for item in comparison.dimensions if item.name == "entities"]
    assert entities.only_right == 100
    assert len(entities.right_examples) == 20


def test_a_geometry_key_renders_as_a_short_digest_of_its_ids() -> None:
    assert str(A).startswith("geometry:") and str(A).endswith("x2")
    assert str(A) != str(GeometryKey(("g-001",)))
    assert GeometryKey(("g-002", "g-001")) == A  # a ordem dos ids não é identidade


# --- H1 ---------------------------------------------------------------------------------


def test_h1_holds_when_the_chunked_state_equals_both_whole_runs() -> None:
    verdict = judge_layer_one(w1=_state(), w2=_state(), chunked=_state())

    assert verdict.outcome is LayerOneOutcome.HOLDS
    assert verdict.precondition.equal


def test_a_chunked_state_that_differs_from_equal_whole_runs_is_a_partition_divergence() -> None:
    verdict = judge_layer_one(w1=_state(), w2=_state(), chunked=_state(entities=(A,)))

    assert verdict.outcome is LayerOneOutcome.PARTITION_DIVERGENCE
    assert not verdict.chunked_w1.equal and not verdict.chunked_w2.equal


def test_whole_runs_that_differ_break_the_precondition_whatever_the_chunks_do() -> None:
    verdict = judge_layer_one(w1=_state(), w2=_state(supports=(A,)), chunked=_state())

    assert verdict.outcome is LayerOneOutcome.PRECONDITION_FAILED


# --- Scenarios 2 and 3 ------------------------------------------------------------------


def _counts(physical: int, inference: int) -> BuildCounts:
    return BuildCounts(
        physical_observations=physical,
        inference_results=inference,
        supports=1,
        contributions=1,
        excluded_observations=0,
        semantic_entities=1,
        rejected_candidates=0,
        resolved_entities=1,
        relations=0,
        relations_by_state={},
    )


def test_repeated_inference_keeps_the_frames_and_adds_the_results_of_both_runs() -> None:
    verdict = judge_repeated_inference(
        w1=_counts(100, 100), w2=_counts(100, 99), both=_counts(100, 199)
    )

    assert verdict.holds
    assert verdict.physical_observations == (100, 100, 100)
    assert verdict.inference_results == (100, 99, 199)


@pytest.mark.parametrize(
    "both",
    [_counts(200, 200), _counts(100, 100), _counts(100, 201)],
    ids=["new-frames", "lost-results", "extra-results"],
)
def test_repeated_inference_that_counts_new_frames_or_miscounts_results_does_not_hold(
    both: BuildCounts,
) -> None:
    verdict = judge_repeated_inference(w1=_counts(100, 100), w2=_counts(100, 100), both=both)

    assert not verdict.holds


def _ref(stage: str) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage,
        contract="X",
        artifact_id=f"{stage}-1",
        content_hash="sha256:1",
        location=stage,
    )


def test_the_dispositions_of_a_run_come_from_its_execution_record(tmp_path: Path) -> None:
    record = ExecutionRecord(
        plan_digest="sha256:plan",
        order=("semantic_fusion", "semantic_mapping"),
        stages=(
            StageRecord(stage_id="semantic_fusion", inputs={}, output=_ref("semantic_fusion")),
            StageRecord(
                stage_id="semantic_mapping",
                inputs={},
                output=_ref("semantic_mapping"),
                decision=ReuseDecision(kind="reused", reason="same key"),
            ),
        ),
        reused={
            "visual_perception": (_ref("visual_perception"),),
            "ingestion": (_ref("ingestion"),),
        },
    )
    write_execution_record(record, tmp_path)

    assert stage_dispositions(tmp_path) == {
        "ingestion": "provided",
        "visual_perception": "provided",
        "semantic_fusion": "executed",
        "semantic_mapping": "reused",
    }


def test_a_downstream_policy_change_that_reruns_no_context_stage_holds() -> None:
    provided = {"visual_perception": "provided", "sensor_association": "provided"}
    downstream = {"semantic_fusion": "executed", "context_map": "executed"}

    verdict = judge_downstream_only(
        baseline={**provided, **downstream},
        changed={**provided, **downstream},
        baseline_build="sha256:a",
        changed_build="sha256:b",
    )

    assert verdict.holds
    assert verdict.recomputed_context_stages == ()
    assert verdict.identities_differ


def test_a_build_that_reruns_a_context_stage_is_not_downstream_only() -> None:
    verdict = judge_downstream_only(
        baseline={"visual_perception": "provided", "sensor_association": "provided"},
        changed={"visual_perception": "executed", "sensor_association": "provided"},
        baseline_build="sha256:a",
        changed_build="sha256:b",
    )

    assert not verdict.holds
    assert verdict.recomputed_context_stages == ("changed:visual_perception",)


def test_two_builds_with_one_identity_did_not_change_a_policy() -> None:
    same = {"visual_perception": "provided", "sensor_association": "provided"}

    verdict = judge_downstream_only(
        baseline=same, changed=same, baseline_build="sha256:a", changed_build="sha256:a"
    )

    assert not verdict.holds and not verdict.identities_differ


def test_every_verdict_encodes_to_json_with_geometry_as_its_short_digest() -> None:
    verdict = judge_layer_one(w1=_state(), w2=_state(), chunked=_state(entities=(A,)))
    terms = _terms((1, 0), (1, 1))
    effect = partition_effect(terms, terms, terms, resamples=10, seed=1, confidence=0.95)

    document = encode_incremental_context(
        {
            "verdict": verdict,
            "effect": effect,
            "decision": decide_equivalence(effect, delta=0.1),
            "states": {A: frozenset({"door"})},
        }
    )

    assert json.loads(json.dumps(document)) == document
    assert document["verdict"]["outcome"] == "partition_divergence"
    dimensions = document["verdict"]["chunked_w1"]["dimensions"]
    entities = [item for item in dimensions if item["name"] == "entities"]
    assert entities[0]["right_examples"] == [str(B)]
    assert document["effect"]["seed"] == 1
    assert document["decision"] == "equivalent_within_delta"
    assert document["states"] == {str(A): ["door"]}


def test_a_value_the_report_does_not_know_is_refused() -> None:
    with pytest.raises(TypeError, match="object"):
        encode_incremental_context(object())


def test_a_build_without_its_context_evidence_is_not_downstream_only() -> None:
    verdict = judge_downstream_only(
        baseline={"visual_perception": "provided"},
        changed={"visual_perception": "provided"},
        baseline_build="sha256:a",
        changed_build="sha256:b",
    )

    assert not verdict.holds
