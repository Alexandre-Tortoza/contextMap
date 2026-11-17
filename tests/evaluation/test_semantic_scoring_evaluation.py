"""Reference evaluation of semantic scorer support (#530, #658), on real domain types."""

from collections.abc import Sequence
from dataclasses import fields

import pytest
from reference_set_builders import make_valid_manifest

from contextmap.evaluation.annotations import AnnotationFamily
from contextmap.evaluation.metrics import (
    EvaluationStage,
    MetricKind,
    default_metric_registry,
)
from contextmap.evaluation.report_adapters import semantic_scoring_evaluation_report
from contextmap.evaluation.report_schema import MetricStatus
from contextmap.evaluation.semantic_interpretation import SemanticAnnotation
from contextmap.evaluation.semantic_scoring import (
    ClaimLabel,
    ScoredRegion,
    ScorerDiagnostics,
    ScorerSupportQuality,
    SemanticScoringEvaluationContext,
    SemanticScoringEvaluationError,
    SemanticScoringEvaluationPolicy,
    SemanticScoringEvaluationReport,
    encode_semantic_scoring_report,
    evaluate_semantic_scoring,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_scoring import ScoreId, SemanticScore, SemanticScoreType
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import FeatureId

OBSERVATION = SourceObservationId("frame-1")
RESULT = SemanticInterpretationRunId("result-1")
SEMANTICS = AnnotationFamily.SEMANTICS.schema
REGIONS = AnnotationFamily.REGIONS.schema
INTERPRETER = SemanticInferenceProvenance(
    backend=BackendProvenance(
        backend_id="qwen_semantic",
        capability="semantic_interpreter",
        provider="qwen",
        model="qwen3-vl",
        version="1",
    ),
    task_identity="region",
    prompt_template_id="region/v1",
    output_schema_version="semantic-response/1",
)
CLIP = BackendProvenance(
    backend_id="clip_region_semantic_scorer",
    capability="semantic_scorer",
    provider="openai",
    model="openai/clip-vit-large-patch14",
    version="1",
    configuration_fingerprint="sha256:clip",
)
ALPHACLIP = BackendProvenance(
    backend_id="alphaclip_semantic_scorer",
    capability="semantic_scorer",
    provider="alphaclip",
    model="ViT-L/14",
    version="1",
    configuration_fingerprint="sha256:alphaclip",
)
CLIP_SPACE = "clip:openai/clip-vit-large-patch14"
ALPHACLIP_SPACE = "alphaclip:ViT-L/14"
POLICY = SemanticScoringEvaluationPolicy(policy_id="scoring-support/test", minimum_class_count=1)
CHAIR_NOT_TABLE = SemanticAnnotation(
    acceptable_hypotheses=("chair",), rejected_hypotheses=("table", "stool")
)

# (hipótese, valor do score) por claim; a primeira claim é a primária do VLM. None: sem score.
ClaimSpec = tuple[str, float | None]


def _claim(claim_id: str, hypothesis: str, region: str, role: HypothesisRole) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(claim_id),
        source_observation_id=OBSERVATION,
        semantic_interpretation_run_id=RESULT,
        hypothesis=hypothesis,
        role=role,
        provenance=INTERPRETER,
        region_id=RegionId(region),
    )


def _score(
    claim: SemanticClaim,
    value: float,
    *,
    scorer: BackendProvenance = CLIP,
    space: str = CLIP_SPACE,
    calibrated_probability: float | None = None,
) -> SemanticScore:
    return SemanticScore(
        score_id=ScoreId(f"{scorer.backend_id}:{claim.claim_id}"),
        claim_id=claim.claim_id,
        feature_id=FeatureId(f"feature-{claim.region_id}"),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=value,
        calibrated_probability=calibrated_probability,
        embedding_space_id=space,
        source_observation_id=claim.source_observation_id,
        region_id=claim.region_id,
        provenance=scorer,
    )


def _region(
    claims: Sequence[ClaimSpec],
    *,
    region: str = "region-1",
    annotation: SemanticAnnotation | None = CHAIR_NOT_TABLE,
    repeat_index: int = 0,
) -> ScoredRegion:
    built = tuple(
        _claim(
            f"{region}-claim-{index}",
            hypothesis,
            region,
            HypothesisRole.PRIMARY if index == 0 else HypothesisRole.ALTERNATIVE,
        )
        for index, (hypothesis, _) in enumerate(claims)
    )
    scores = tuple(
        _score(claim, value)
        for claim, (_, value) in zip(built, claims, strict=True)
        if value is not None
    )
    return ScoredRegion(
        claims=built, scores=scores, annotation=annotation, repeat_index=repeat_index
    )


def _context(
    *,
    schemas: tuple[str, ...] = (REGIONS, SEMANTICS),
    join: str | None = "reference-region-join/test",
) -> SemanticScoringEvaluationContext:
    return SemanticScoringEvaluationContext(
        evaluation_id="scoring-eval-1",
        reference_set_version="2.0.0",
        selection_id="test",
        semantic_interpretation_artifact_id="perception-run-1",
        region_artifact_id="regions-run-1",
        visual_evidence_artifact_id="visual-run-1",
        scoring_run_ids=("scoring-run-clip",),
        annotation_schemas=schemas,
        region_join_policy_id=join,
    )


def _evaluate(
    *regions: ScoredRegion,
    policy: SemanticScoringEvaluationPolicy = POLICY,
    context: SemanticScoringEvaluationContext | None = None,
) -> SemanticScoringEvaluationReport:
    return evaluate_semantic_scoring(
        context=_context() if context is None else context,
        policy=policy,
        regions=regions,
    )


def _only(blocks: Sequence[ScorerSupportQuality]) -> ScorerSupportQuality:
    assert len(blocks) == 1
    return blocks[0]


# --------------------------------------------------------------------------- labels


def test_scores_are_labeled_by_the_reference_and_unassessed_claims_stay_outside() -> None:
    annotation = SemanticAnnotation(
        acceptable_hypotheses=("Office Chair",), rejected_hypotheses=("table",)
    )

    report = _evaluate(
        _region((("office  chair", 0.30), ("TABLE", 0.10), ("lamp", 0.90)), annotation=annotation)
    )

    quality = _only(report.quality)
    labels = {sample.hypothesis: sample.label for sample in quality.samples}
    assert labels == {
        "office  chair": ClaimLabel.POSITIVE,
        "TABLE": ClaimLabel.NEGATIVE,
        "lamp": ClaimLabel.UNASSESSED,
    }
    assert (quality.positive_count, quality.negative_count, quality.unassessed_count) == (1, 1, 1)
    assert quality.positive_scores.median == 0.30
    assert quality.negative_scores.maximum == 0.10
    # "lamp" tem o maior score, mas não anotado nunca é negativo: o ranking só vê as rotuladas.
    assert quality.top1_accuracy.value == 1.0
    assert quality.auroc.value == 1.0
    assert quality.auroc.sample_count == 2


def test_a_claim_without_a_score_is_left_out_and_never_scored_as_zero() -> None:
    report = _evaluate(_region((("chair", 0.2), ("table", None), ("stool", 0.3))))

    quality = _only(report.quality)
    assert (quality.claim_count, quality.scored_claim_count, quality.unscored_claim_count) == (
        3,
        2,
        1,
    )
    assert quality.negative_count == 1
    assert quality.negative_scores.minimum == 0.3
    # Um zero fabricado para "table" ficaria abaixo de 0.2 e levaria o AUROC a 0.5.
    assert quality.auroc.value == 0.0


def test_an_unknown_target_keeps_its_rejected_concepts_as_negatives_but_has_no_positive() -> None:
    unknown = SemanticAnnotation(abstention_expected=True, rejected_hypotheses=("table",))

    report = _evaluate(_region((("chair", 0.4), ("table", 0.2)), annotation=unknown))

    quality = _only(report.quality)
    assert (quality.positive_count, quality.negative_count, quality.unassessed_count) == (0, 1, 1)
    assert quality.top1_accuracy.value is None
    assert quality.auroc.value is None


def test_the_calibrated_probability_is_never_read() -> None:
    chair, table = (
        _claim("c-1", "chair", "region-1", HypothesisRole.PRIMARY),
        _claim("c-2", "table", "region-1", HypothesisRole.ALTERNATIVE),
    )
    region = ScoredRegion(
        claims=(chair, table),
        scores=(_score(chair, 0.2, calibrated_probability=0.99), _score(table, 0.5)),
        annotation=CHAIR_NOT_TABLE,
    )

    quality = _only(_evaluate(region).quality)

    assert quality.positive_scores.maximum == 0.2
    assert quality.auroc.value == 0.0


# ----------------------------------------------------------------- scorers and spaces


def test_two_embedding_spaces_are_two_blocks_and_never_pooled() -> None:
    chair = _claim("c-1", "chair", "region-1", HypothesisRole.PRIMARY)
    table = _claim("c-2", "table", "region-1", HypothesisRole.ALTERNATIVE)
    region = ScoredRegion(
        claims=(chair, table),
        scores=(
            _score(chair, 0.3),
            _score(table, 0.1),
            _score(chair, 0.8, scorer=ALPHACLIP, space=ALPHACLIP_SPACE),
            _score(table, 0.9, scorer=ALPHACLIP, space=ALPHACLIP_SPACE),
        ),
        annotation=CHAIR_NOT_TABLE,
    )

    report = _evaluate(region)

    blocks = {block.embedding_space_id: block for block in report.quality}
    assert set(blocks) == {CLIP_SPACE, ALPHACLIP_SPACE}
    assert blocks[CLIP_SPACE].scorer == CLIP
    assert blocks[ALPHACLIP_SPACE].scorer == ALPHACLIP
    for block in blocks.values():
        assert (block.positive_count, block.negative_count) == (1, 1)
    assert blocks[CLIP_SPACE].auroc.value == 1.0
    assert blocks[ALPHACLIP_SPACE].auroc.value == 0.0


def test_one_claim_scored_twice_by_the_same_scorer_and_space_is_refused() -> None:
    chair = _claim("c-1", "chair", "region-1", HypothesisRole.PRIMARY)

    with pytest.raises(ValueError, match="more than one score"):
        ScoredRegion(claims=(chair,), scores=(_score(chair, 0.3), _score(chair, 0.4)))


# ------------------------------------------------------------------ AUROC and AUPRC


def test_auroc_and_auprc_are_not_applicable_below_the_declared_class_minimum() -> None:
    regions = (
        _region((("chair", 0.9), ("table", 0.6)), region="region-1"),
        _region((("chair", 0.4), ("table", 0.1)), region="region-2"),
    )

    quality = _only(
        _evaluate(
            *regions,
            policy=SemanticScoringEvaluationPolicy(
                policy_id="scoring-support/test", minimum_class_count=3
            ),
        ).quality
    )

    for metric in (quality.auroc, quality.auprc):
        assert metric.value is None
        assert metric.sample_count == 4
        assert metric.not_applicable_reason is not None
        assert "3" in metric.not_applicable_reason
        assert "2 positive" in metric.not_applicable_reason


def test_pooled_auroc_and_auprc_match_a_small_case_computed_by_hand() -> None:
    regions = (
        _region((("chair", 0.9), ("table", 0.6)), region="region-1"),
        _region((("chair", 0.4), ("table", 0.1)), region="region-2"),
    )

    quality = _only(
        _evaluate(
            *regions,
            policy=SemanticScoringEvaluationPolicy(
                policy_id="scoring-support/test", minimum_class_count=2
            ),
        ).quality
    )

    # Positivos {0.9, 0.4}, negativos {0.6, 0.1}: 3 dos 4 pares ordenados, AUROC = 0.75.
    assert quality.auroc.value == pytest.approx(0.75)
    # Limiares 0.9, 0.6, 0.4, 0.1: AP = 0.5 * 1 + 0.5 * 2/3 = 5/6.
    assert quality.auprc.value == pytest.approx(5 / 6)
    # Dentro de cada região a ordem é perfeita; só o limiar global erra.
    assert quality.top1_accuracy.value == 1.0
    assert quality.pairwise_accuracy.value == 1.0


def test_a_tied_pair_counts_half_in_the_pooled_auroc() -> None:
    quality = _only(_evaluate(_region((("chair", 0.5), ("table", 0.5)))).quality)

    assert quality.auroc.value == 0.5
    assert quality.auprc.value == 0.5


# -------------------------------------------------------------------------- ranking


def test_ranking_declares_ties_and_skips_regions_without_both_classes() -> None:
    regions = (
        _region((("chair", 0.5), ("table", 0.5)), region="tied"),
        _region((("chair", 0.7), ("lamp", 0.9)), region="no-negative"),
        _region((("chair", 0.3),), region="single-alternative"),
        _region((("chair", 0.2), ("stool", 0.6), ("table", 0.1)), region="negative-on-top"),
        _region((("chair", 0.8), ("table", 0.1)), region="correct"),
    )

    quality = _only(_evaluate(*regions).quality)

    assert quality.ranking_region_count == 3
    # Empate no topo entre positiva e negativa: no denominador, nunca como acerto.
    assert quality.top1_tie_count == 1
    assert quality.top1_accuracy.value == pytest.approx(1 / 3)
    assert quality.top1_accuracy.sample_count == 3
    # Pares empatados valem 1/2: (0.5 + 0.5 + 1.0) / 3.
    assert quality.pairwise_tie_count == 1
    assert quality.pairwise_accuracy.value == pytest.approx(2 / 3)


def test_ranking_without_any_eligible_region_is_not_applicable_with_its_reason() -> None:
    quality = _only(_evaluate(_region((("chair", 0.7), ("lamp", 0.9)))).quality)

    for metric in (quality.top1_accuracy, quality.pairwise_accuracy):
        assert metric.value is None
        assert metric.sample_count == 0
        assert metric.not_applicable_reason is not None
        assert "positive and a negative" in metric.not_applicable_reason


# -------------------------------------------------------------------------- repeats


def test_repeats_of_a_region_are_not_new_samples() -> None:
    claims = (("chair", 0.6), ("table", 0.2))

    report = _evaluate(_region(claims), _region(claims, repeat_index=1))

    quality = _only(report.quality)
    assert (quality.positive_count, quality.negative_count) == (1, 1)
    assert quality.auroc.sample_count == 2
    assert report.repeat_region_count == 1
    assert report.region_count == 1


def test_the_same_region_twice_in_one_repeat_is_refused() -> None:
    claims = (("chair", 0.6),)

    with pytest.raises(SemanticScoringEvaluationError, match="region"):
        _evaluate(_region(claims), _region(claims))


# --------------------------------------------------------------------- diagnostics


def test_agreement_with_the_vlm_primary_is_a_diagnostic_never_a_quality_metric() -> None:
    # A primária do VLM ("lamp") tem o maior score, mas não está anotada.
    report = _evaluate(_region((("lamp", 0.9), ("chair", 0.5), ("table", 0.1))))

    diagnostics = _only_diagnostics(report)
    assert diagnostics.primary_agreement.value == 1.0
    assert diagnostics.primary_agreement.sample_count == 1
    assert diagnostics.scores.count == 3
    assert "primary_agreement" not in {field.name for field in fields(ScorerSupportQuality)}


def test_primary_agreement_needs_no_reference_and_declares_ties() -> None:
    report = _evaluate(
        _region((("lamp", 0.4), ("chair", 0.4)), region="tied", annotation=None),
        _region((("lamp", 0.1), ("chair", 0.8)), region="disagree", annotation=None),
        _region((("lamp", 0.9),), region="primary-only", annotation=None),
        context=_context(schemas=(REGIONS,), join=None),
    )

    diagnostics = _only_diagnostics(report)
    assert diagnostics.primary_agreement.sample_count == 2
    assert diagnostics.primary_agreement.value == 0.0
    assert diagnostics.primary_agreement_tie_count == 1


def _only_diagnostics(report: SemanticScoringEvaluationReport) -> ScorerDiagnostics:
    assert len(report.diagnostics) == 1
    return report.diagnostics[0]


# ------------------------------------------------------------ no reference semantics


def test_without_a_semantics_family_every_support_metric_is_not_applicable() -> None:
    report = _evaluate(
        _region((("chair", 0.6), ("table", 0.2)), annotation=None),
        context=_context(schemas=(REGIONS,), join=None),
    )

    quality = _only(report.quality)
    assert (quality.positive_count, quality.negative_count) == (0, 0)
    assert quality.unassessed_count == 2
    for metric in (
        quality.top1_accuracy,
        quality.pairwise_accuracy,
        quality.auroc,
        quality.auprc,
    ):
        assert metric.value is None
        assert metric.not_applicable_reason is not None
        assert SEMANTICS in metric.not_applicable_reason
    # A distribuição sem rótulo continua como diagnóstico.
    assert _only_diagnostics(report).scores.count == 2


def test_an_annotated_run_region_needs_the_semantics_family_and_a_named_join() -> None:
    region = _region((("chair", 0.6),))

    with pytest.raises(SemanticScoringEvaluationError, match="join"):
        _evaluate(region, context=_context(join=None))
    with pytest.raises(SemanticScoringEvaluationError, match="semantics"):
        _evaluate(region, context=_context(schemas=(REGIONS,)))


def test_only_region_claims_of_one_region_form_a_scored_region() -> None:
    scene = SemanticClaim(
        claim_id=ClaimId("scene"),
        source_observation_id=OBSERVATION,
        semantic_interpretation_run_id=RESULT,
        hypothesis="corridor",
        role=HypothesisRole.PRIMARY,
        provenance=INTERPRETER,
    )
    first = _claim("c-1", "chair", "region-1", HypothesisRole.PRIMARY)
    other = _claim("c-2", "table", "region-2", HypothesisRole.PRIMARY)

    with pytest.raises(ValueError, match="region claim"):
        ScoredRegion(claims=(scene,))
    with pytest.raises(ValueError, match="one region"):
        ScoredRegion(claims=(first, other))
    with pytest.raises(ValueError, match="claim"):
        ScoredRegion(claims=(first,), scores=(_score(other, 0.1),))


def test_the_class_minimum_is_declared_and_positive() -> None:
    with pytest.raises(ValueError, match="minimum_class_count"):
        SemanticScoringEvaluationPolicy(policy_id="p", minimum_class_count=0)


# ------------------------------------------------------------------------ envelope


def test_the_registry_has_a_semantic_scoring_stage_apart_from_the_interpreter() -> None:
    registry = default_metric_registry()

    names = {
        definition.name for definition in registry.definitions_for(EvaluationStage.SEMANTIC_SCORING)
    }

    assert names == {
        "scoring.ranking.top1_accuracy",
        "scoring.ranking.pairwise_accuracy",
        "scoring.auroc",
        "scoring.auprc",
    }
    for name in names:
        definition = registry.get(name, "1")
        assert definition.kind is MetricKind.QUALITY
        assert definition.required_annotations == (SEMANTICS,)
        assert (definition.minimum, definition.maximum) == (0.0, 1.0)
        assert definition.unit == "ratio"


def test_the_envelope_reports_not_applicable_without_semantics_never_zero() -> None:
    registry = default_metric_registry()
    report = _evaluate(
        _region((("chair", 0.6), ("table", 0.2)), annotation=None),
        context=_context(schemas=(REGIONS,), join=None),
    )

    envelope = semantic_scoring_evaluation_report(
        report, registry=registry, reference_set=make_valid_manifest().identity()
    )

    assert envelope.stage is EvaluationStage.SEMANTIC_SCORING
    assert {(item.kind, item.artifact_id) for item in envelope.reproducibility.input_artifacts} >= {
        ("region_perception_run", "regions-run-1"),
        ("visual_evidence_run", "visual-run-1"),
        ("semantic_interpretation_run", "perception-run-1"),
    }
    assert {result.metric for result in envelope.quality_metrics} == {
        "scoring.ranking.top1_accuracy",
        "scoring.ranking.pairwise_accuracy",
        "scoring.auroc",
        "scoring.auprc",
    }
    for result in envelope.quality_metrics:
        assert result.status is MetricStatus.NOT_APPLICABLE
        assert result.value is None
    assert envelope.stage_report == encode_semantic_scoring_report(report)


def test_the_envelope_stratifies_values_by_scorer_and_embedding_space() -> None:
    registry = default_metric_registry()
    report = _evaluate(
        _region((("chair", 0.9), ("table", 0.6)), region="region-1"),
        _region((("chair", 0.4), ("table", 0.1)), region="region-2"),
    )

    envelope = semantic_scoring_evaluation_report(
        report, registry=registry, reference_set=make_valid_manifest().identity()
    )

    auroc = next(item for item in envelope.quality_metrics if item.metric == "scoring.auroc")
    assert auroc.status is MetricStatus.VALUE
    assert auroc.value == pytest.approx(0.75)
    assert auroc.sample_count == 4
    assert dict(auroc.strata)["embedding_space_id"] == CLIP_SPACE
    assert CLIP.backend_id in dict(auroc.strata)["scorer"]


# ---------------------------------------------------------------------- region join


def test_joins_of_another_policy_than_the_context_are_refused() -> None:
    from contextmap.evaluation.annotations import Coverage, FrameRegionAnnotation
    from contextmap.evaluation.reference_set import ReferenceSampleId
    from contextmap.evaluation.region_join import RegionJoinPolicy, join_frame_regions

    join = join_frame_regions(
        RegionJoinPolicy(policy_id="reference-region-join/other", min_iou=0.5),
        FrameRegionAnnotation(
            sample_id=ReferenceSampleId("sample-1"),
            observation_id=OBSERVATION,
            image_width=4,
            image_height=4,
            coverage=Coverage.PARTIAL,
            regions=(),
        ),
        (),
    )

    assert _evaluate(_region((("chair", 0.6),))).region_join is None
    with pytest.raises(SemanticScoringEvaluationError, match="join policy"):
        evaluate_semantic_scoring(
            context=_context(),
            policy=POLICY,
            regions=(_region((("chair", 0.6),)),),
            region_joins=(join,),
        )
