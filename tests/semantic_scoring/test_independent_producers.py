from dataclasses import fields

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import ClaimId
from contextmap.semantic_scoring import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
    SemanticScoringLineage,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import FeatureId


def test_scoring_lineage_names_both_independent_producer_artifacts() -> None:
    lineage = SemanticScoringLineage(
        sequence_artifact_id="sequence",
        selection_id="selection",
        region_artifact_id="regions",
        visual_evidence_artifact_id="visual",
        semantic_interpretation_artifact_id="semantic",
    )

    assert lineage.visual_evidence_artifact_id != lineage.semantic_interpretation_artifact_id
    assert "perception_run_id" not in {field.name for field in fields(lineage)}


def test_a_score_anchors_to_the_observation_and_region_without_a_fused_result() -> None:
    score = SemanticScore(
        score_id=ScoreId("score"),
        claim_id=ClaimId("claim"),
        feature_id=FeatureId("feature"),
        source_observation_id=SourceObservationId("frame"),
        region_id=RegionId("region"),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=-0.2,
        calibrated_probability=None,
        embedding_space_id="space",
        provenance=BackendProvenance(
            backend_id="scorer",
            capability="semantic_scorer",
            provider="fake",
            model="clip",
            version="1",
        ),
    )

    assert score.value == -0.2
    assert "perception_result_id" not in {field.name for field in fields(score)}
