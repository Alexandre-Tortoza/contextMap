"""Scored regions built from persisted perception and scoring runs through the join (#658).

Both runs are written and reopened for real; only the scores are hand-built, with the exact
contract a scorer returns, so no model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from contextmap.evaluation.annotations import (
    CASEFOLD_EXACT_POLICY,
    AnnotationFamily,
    Coverage,
    FrameRegionAnnotation,
    LabelNormalization,
    RegionAnnotation,
    RegionAnnotationSet,
    SemanticAnnotationRecord,
    SemanticAnnotationSet,
    SemanticStatus,
)
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.region_join import RegionJoinPolicy
from contextmap.evaluation.semantic_interpretation import SemanticAnnotation
from contextmap.evaluation.semantic_scoring import (
    SemanticScoringEvaluationError,
    SemanticScoringEvaluationPolicy,
    evaluate_semantic_scoring,
    join_scored_regions,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox,
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionId,
    RegionPerceptionRunManifest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
)
from contextmap.semantic_scoring import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
    SemanticScoringLineage,
    SemanticScoringRunId,
    SemanticScoringRunReader,
    SemanticScoringRunWriter,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    EmbeddingSpace,
    FeatureId,
    FeatureScope,
    PerceptionResult,
    PreparedImage,
    VisualFeature,
    embedding_space_fingerprint,
)
from semantic_artifact_builders import write_semantic_views

SAMPLE = ReferenceSampleId("sample-0001")
FRAME = SourceObservationId("frame-0001")
RUN = SemanticInterpretationRunId("perception-run-1")
WIDTH, HEIGHT = 6, 4
SPACE = EmbeddingSpace(
    family="alphaclip",
    model="ViT-L/14",
    version="1",
    checkpoint="sha256:" + "a" * 64,
    layer="alpha_conditioned_image_projection",
    dimension=2,
    normalization="l2",
)
SPACE_ID = embedding_space_fingerprint(SPACE)
SCORER = BackendProvenance(
    backend_id="alphaclip_region_scorer",
    capability="semantic_scorer",
    provider="SunzeY/AlphaCLIP",
    model="ViT-L/14",
    version="sha256:" + "a" * 64,
    configuration_fingerprint="sha256:scorer",
)
JOIN = RegionJoinPolicy(policy_id="reference-region-join/test", min_iou=0.5)
SCORES = {"chair": 0.31, "table": 0.12, "door": 0.25, "wall": 0.30}


def _provenance(capability: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id="fake", capability=capability, provider="f", model="f", version="1"
    )


def _mask(pixels: set[tuple[int, int]]) -> InlineMask:
    array = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x, y in pixels:
        array[y, x] = True
    return InlineMask(array)


def _row(y: int, *columns: int) -> set[tuple[int, int]]:
    return {(x, y) for x in columns}


def _claim(
    result_id: SemanticInterpretationRunId, name: str, region: str | None, role: HypothesisRole
) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(f"claim-{name}"),
        source_observation_id=FRAME,
        semantic_interpretation_run_id=result_id,
        hypothesis=name,
        role=role,
        provenance=SemanticInferenceProvenance(
            backend=_provenance("semantic_interpreter"),
            task_identity="region",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=None if region is None else RegionId(region),
    )


def _region(region_id: str, box: BoundingBox2D, mask: InlineMask | None) -> Region2D:
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=box,
        provenance=_provenance("region_discovery"),
        source_observation_id=FRAME,
        image_width=WIDTH,
        image_height=HEIGHT,
        mask=mask,
    )


def _feature(region: str) -> VisualFeature:
    return VisualFeature(
        source_observation_id=FRAME,
        feature_id=FeatureId(f"feature-{region}"),
        scope=FeatureScope.REGION,
        embedding_space_id=SPACE_ID,
        shape=(2,),
        dtype="float32",
        payload_reference=f"region/feature-{region}.npy",
        provenance=_provenance("feature_extractor"),
        region_id=RegionId(region),
        normalization="l2",
    )


def _result(run_id: SemanticInterpretationRunId) -> PerceptionResult:
    """Two masked regions and one box-only region; region-0002 has no feature to score."""
    result_id = SemanticInterpretationRunId(run_id)
    scene = _claim(result_id, "corridor", None, HypothesisRole.PRIMARY)
    return PerceptionResult(
        source_observation_id=FRAME,
        region_artifact_id="regional",
        semantic_interpretation_artifact_id=str(run_id),
        visual_evidence_artifact_id="visual",
        sequence_artifact_id="sequence-0001",
        regions=(
            _region(
                "region-0001",
                BoundingBox2D(x=0, y=0, width=4, height=1),
                _mask(_row(0, 0, 1, 2, 3)),
            ),
            _region(
                "region-0002",
                BoundingBox2D(x=0, y=2, width=3, height=1),
                _mask(_row(2, 0, 1, 2)),
            ),
            _region("region-0003", BoundingBox2D(x=4, y=2, width=2, height=2), None),
        ),
        features=(_feature("region-0001"), _feature("region-0003")),
        claims=(
            _claim(result_id, "chair", "region-0001", HypothesisRole.PRIMARY),
            _claim(result_id, "table", "region-0001", HypothesisRole.ALTERNATIVE),
            _claim(result_id, "lamp", "region-0002", HypothesisRole.PRIMARY),
            _claim(result_id, "door", "region-0003", HypothesisRole.PRIMARY),
            _claim(result_id, "wall", "region-0003", HypothesisRole.ALTERNATIVE),
        ),
        scene_context=SceneContext(
            context_id="scene",
            source_observation_id=FRAME,
            semantic_interpretation_run_id=result_id,
            provenance=scene.provenance,
            claims=(scene,),
        ),
    )


@dataclass(frozen=True)
class EvidenceRuns:
    regional: RegionPerceptionRunReader
    semantic: SemanticInterpretationRunReader
    view: PerceptionResult


def _write_perception(output: Path, run_id: SemanticInterpretationRunId = RUN) -> EvidenceRuns:
    result = _result(run_id)
    regional = RegionPerceptionRunWriter(output_dir=output / "regions")
    regional.write_manifest(
        RegionPerceptionRunManifest(
            run_id="regional",
            run_index=1,
            sequence_name="S1",
            selection_id="selection",
            sequence_artifact_id="sequence-0001",
            source_observation_ids=(str(FRAME),),
            created_at="fixture",
            prepared_image_id="prepared",
            discovery_backend_id="fake",
            discovery_model="fake",
            discovery_version="1",
            discovery_configuration_fingerprint="cfg",
            normalization_policy_fingerprint="norm",
        )
    )
    regional.write_frame(
        PreparedImage(
            source_observation_id=FRAME, payload_reference="source.png", width=WIDTH, height=HEIGHT
        )
    )
    for region in result.regions:
        regional.write_region(regional.externalize_region_mask(FRAME, region))
    regional.finalize()
    writer = SemanticInterpretationRunWriter(
        output_dir=output / "semantic",
        sequence_name="S1",
        run_id=run_id,
        run_index=1,
        sequence_artifact_id="sequence-0001",
        selection_id="selection",
        region_artifact_id="regional",
        configuration_digest="fixture",
    )
    write_semantic_views(writer, (result,), str(run_id))
    writer.finalize()
    semantic = SemanticInterpretationRunReader(output / "semantic")
    regional_reader = RegionPerceptionRunReader(output / "regions")
    (observation,) = semantic.list_observations()
    view = PerceptionResult(
        source_observation_id=FRAME,
        sequence_artifact_id="sequence-0001",
        region_artifact_id="regional",
        semantic_interpretation_artifact_id=str(run_id),
        visual_evidence_artifact_id="visual",
        claims=observation.claims,
        scene_context=observation.scene_context,
        regions=regional_reader.read_regions(),
        features=result.features,
    )
    return EvidenceRuns(regional_reader, semantic, view)


def _write_scoring(output: Path, perception: EvidenceRuns) -> SemanticScoringRunReader:
    writer = SemanticScoringRunWriter(
        output_dir=output,
        sequence_name="S1",
        run_id=SemanticScoringRunId("scoring-run-1"),
        run_index=2,
        lineage=SemanticScoringLineage(
            sequence_artifact_id="sequence-0001",
            selection_id="selection",
            region_artifact_id="regional",
            visual_evidence_artifact_id="visual",
            semantic_interpretation_artifact_id=str(perception.semantic.manifest.run_id),
        ),
        scorer=SCORER,
        embedding_space=SPACE,
        text_template="{hypothesis}",
    )
    result = perception.view
    writer.add_result(
        result,
        tuple(
            SemanticScore(
                score_id=ScoreId(f"score-{claim.claim_id}"),
                claim_id=claim.claim_id,
                feature_id=FeatureId(f"feature-{claim.region_id}"),
                score_type=SemanticScoreType.COSINE_SIMILARITY,
                value=SCORES[claim.hypothesis],
                calibrated_probability=None,
                embedding_space_id=SPACE_ID,
                source_observation_id=FRAME,
                region_id=claim.region_id,
                provenance=SCORER,
            )
            for claim in result.claims
            if claim.hypothesis in SCORES
        ),
    )
    writer.finalize()
    return SemanticScoringRunReader(output)


def _reference_regions() -> RegionAnnotationSet:
    return RegionAnnotationSet(
        frames=(
            FrameRegionAnnotation(
                sample_id=SAMPLE,
                observation_id=FRAME,
                image_width=WIDTH,
                image_height=HEIGHT,
                coverage=Coverage.PARTIAL,
                regions=(
                    RegionAnnotation(region_id="chair-ref", mask=_mask(_row(0, 0, 1, 2, 3))),
                    RegionAnnotation(
                        region_id="door-ref",
                        box=BoundingBox(x_min=4.0, y_min=2.0, x_max=6.0, y_max=4.0),
                    ),
                    RegionAnnotation(region_id="shelf-ref", mask=_mask(_row(3, 0, 1))),
                ),
            ),
        )
    )


def _semantics() -> SemanticAnnotationSet:
    def record(
        region: str, status: SemanticStatus, *concepts: str, rejected: str = ""
    ) -> SemanticAnnotationRecord:
        return SemanticAnnotationRecord(
            sample_id=SAMPLE,
            observation_id=FRAME,
            region_id=region,
            status=status,
            concepts=concepts,
            rejected_concepts=(rejected,) if rejected else (),
        )

    return SemanticAnnotationSet(
        normalization=LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY),
        records=(
            record("chair-ref", SemanticStatus.LABELED, "chair", rejected="table"),
            record("door-ref", SemanticStatus.LABELED, "door", rejected="wall"),
            record("shelf-ref", SemanticStatus.UNKNOWN),
        ),
    )


@pytest.fixture
def runs(tmp_path: Path) -> tuple[EvidenceRuns, SemanticScoringRunReader]:
    perception = _write_perception(tmp_path / "visual_perception")
    return perception, _write_scoring(tmp_path / "semantic_scoring", perception)


def test_each_run_region_receives_the_annotation_of_the_reference_region_joined_to_it(
    runs: tuple[EvidenceRuns, SemanticScoringRunReader],
) -> None:
    perception, scoring = runs

    joined = join_scored_regions(
        regional=perception.regional,
        semantic=perception.semantic,
        scoring=(scoring,),
        regions=_reference_regions(),
        semantics=_semantics(),
        policy=JOIN,
    )

    by_region = {region.region[2]: region for region in joined.regions}
    assert set(by_region) == {"region-0001", "region-0002", "region-0003"}
    assert by_region["region-0001"].annotation == SemanticAnnotation(
        acceptable_hypotheses=("chair",), rejected_hypotheses=("table",)
    )
    # A região só com caixa se junta pela caixa rasterizada nos centros dos pixels.
    assert by_region["region-0003"].annotation == SemanticAnnotation(
        acceptable_hypotheses=("door",), rejected_hypotheses=("wall",)
    )
    # Sem região de referência, nenhuma expectativa; a claim sem feature fica sem score.
    assert by_region["region-0002"].annotation is None
    assert by_region["region-0002"].scores == ()
    assert [claim.hypothesis for claim in by_region["region-0001"].claims] == ["chair", "table"]
    assert {score.value for score in by_region["region-0001"].scores} == {0.31, 0.12}
    # A claim de cena não é pontuada e não vira região.
    assert all(claim.region_id is not None for item in joined.regions for claim in item.claims)
    (join,) = joined.joins
    assert [(match.reference_region_id, match.run_region_id) for match in join.matches] == [
        ("chair-ref", "region-0001"),
        ("door-ref", "region-0003"),
    ]
    assert join.unmatched_reference_region_ids == ("shelf-ref",)
    assert join.unmatched_run_region_ids == ("region-0002",)
    assert joined.annotation_schemas == (
        AnnotationFamily.REGIONS.schema,
        AnnotationFamily.SEMANTICS.schema,
    )
    assert (joined.semantic_interpretation_artifact_id, joined.scoring_run_ids) == (
        RUN,
        ("scoring-run-1",),
    )


def test_the_joined_regions_evaluate_with_the_join_counted_in_the_report(
    runs: tuple[EvidenceRuns, SemanticScoringRunReader],
) -> None:
    perception, scoring = runs
    joined = join_scored_regions(
        regional=perception.regional,
        semantic=perception.semantic,
        scoring=(scoring,),
        regions=_reference_regions(),
        semantics=_semantics(),
        policy=JOIN,
    )

    report = evaluate_semantic_scoring(
        context=joined.context(
            evaluation_id="eval-1", reference_set_version="2.0.0", selection_id="test"
        ),
        policy=SemanticScoringEvaluationPolicy(
            policy_id="scoring-support/test", minimum_class_count=2
        ),
        regions=joined.regions,
        region_joins=joined.joins,
    )

    (quality,) = report.quality
    assert report.context.region_join_policy_id == JOIN.policy_id
    assert (quality.positive_count, quality.negative_count, quality.unscored_claim_count) == (
        2,
        2,
        1,
    )
    # region-0001: chair 0.31 > table 0.12; region-0003: door 0.25 < wall 0.30.
    assert quality.pairwise_accuracy.value == pytest.approx(0.5)
    assert quality.top1_accuracy.value == pytest.approx(0.5)
    assert quality.auroc.value == pytest.approx(0.75)
    assert report.region_join is not None
    assert report.region_join.policy_id == JOIN.policy_id
    assert (report.region_join.matched_count, report.region_join.unmatched_reference_count) == (
        2,
        1,
    )
    assert report.region_join.unmatched_run_count == 1


def test_without_a_semantics_family_the_join_still_counts_but_nothing_is_annotated(
    runs: tuple[EvidenceRuns, SemanticScoringRunReader],
) -> None:
    perception, scoring = runs

    joined = join_scored_regions(
        regional=perception.regional,
        semantic=perception.semantic,
        scoring=(scoring,),
        regions=_reference_regions(),
        semantics=None,
        policy=JOIN,
    )

    assert all(region.annotation is None for region in joined.regions)
    assert joined.annotation_schemas == (AnnotationFamily.REGIONS.schema,)
    assert len(joined.joins[0].matches) == 2


def test_a_scoring_run_of_another_perception_run_is_refused(tmp_path: Path) -> None:
    perception = _write_perception(tmp_path / "visual_perception")
    another = _write_perception(
        tmp_path / "another", SemanticInterpretationRunId("perception-run-2")
    )
    other = _write_scoring(tmp_path / "semantic_scoring", another)

    with pytest.raises(SemanticScoringEvaluationError, match="semantic producer"):
        join_scored_regions(
            regional=perception.regional,
            semantic=perception.semantic,
            scoring=(other,),
            regions=_reference_regions(),
            semantics=_semantics(),
            policy=JOIN,
        )
