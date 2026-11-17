"""The semantic scoring run: scores of one pinned perception run, kept apart from its claims.

The artifact is written and reopened for real; the scores are hand-built with the exact
contract a scorer returns, so these tests pin what the artifact accepts, records and refuses.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_scoring import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
    SemanticScoringLineage,
    SemanticScoringRunError,
    SemanticScoringRunId,
    SemanticScoringRunReader,
    SemanticScoringRunWriter,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PerceptionResult
from contextmap.visual_perception.visual_features import (
    EmbeddingSpace,
    FeatureId,
    FeatureScope,
    VisualFeature,
    embedding_space_fingerprint,
)

OBSERVATION = SourceObservationId("frame-0001")
RUN = "visual-run-1"
SEMANTIC = SemanticInterpretationRunId("semantic-run-1")
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
LINEAGE = SemanticScoringLineage(
    sequence_artifact_id="sequence-0001",
    selection_id="sha256:selection",
    region_artifact_id="regions-run-1",
    visual_evidence_artifact_id=RUN,
    semantic_interpretation_artifact_id=str(SEMANTIC),
)
_ANY = BackendProvenance(backend_id="f", capability="c", provider="f", model="f", version="1")


def _claim(claim_id: str, region_id: str | None, role: HypothesisRole) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(claim_id),
        source_observation_id=OBSERVATION,
        semantic_interpretation_run_id=SEMANTIC,
        hypothesis=claim_id,
        role=role,
        provenance=SemanticInferenceProvenance(
            backend=replace(_ANY, capability="semantic_interpreter"),
            task_identity="region",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=None if region_id is None else RegionId(region_id),
    )


def _feature(feature_id: str, region_id: str) -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        source_observation_id=OBSERVATION,
        scope=FeatureScope.REGION,
        embedding_space_id=SPACE_ID,
        shape=(2,),
        dtype="float32",
        payload_reference=f"features/{feature_id}.npy",
        provenance=replace(_ANY, capability="feature_extractor"),
        region_id=RegionId(region_id),
        normalization="l2",
    )


def _result() -> PerceptionResult:
    regions = tuple(
        Region2D(
            region_id=RegionId(region_id),
            source_observation_id=OBSERVATION,
            bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
            provenance=_ANY,
        )
        for region_id in ("region-0001", "region-0002", "region-0003")
    )
    scene = _claim("scene-claim", None, HypothesisRole.PRIMARY)
    return PerceptionResult(
        source_observation_id=OBSERVATION,
        region_artifact_id="regions-run-1",
        visual_evidence_artifact_id=RUN,
        semantic_interpretation_artifact_id=str(SEMANTIC),
        sequence_artifact_id="sequence-0001",
        regions=regions,
        features=(_feature("feature-a", "region-0001"), _feature("feature-b", "region-0002")),
        claims=(
            _claim("claim-1", "region-0001", HypothesisRole.PRIMARY),
            _claim("claim-2", "region-0001", HypothesisRole.ALTERNATIVE),
            _claim("claim-3", "region-0003", HypothesisRole.PRIMARY),
        ),
        scene_context=SceneContext(
            context_id="scene-context-1",
            source_observation_id=OBSERVATION,
            semantic_interpretation_run_id=SEMANTIC,
            provenance=scene.provenance,
            claims=(scene,),
        ),
    )


def _score(claim_id: str, feature_id: str, value: float, **changes: object) -> SemanticScore:
    score = SemanticScore(
        score_id=ScoreId(f"score-{claim_id}-{feature_id}"),
        claim_id=ClaimId(claim_id),
        feature_id=FeatureId(feature_id),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=value,
        calibrated_probability=None,
        embedding_space_id=SPACE_ID,
        source_observation_id=OBSERVATION,
        region_id=RegionId("region-0003" if claim_id == "claim-3" else "region-0001"),
        provenance=SCORER,
    )
    return replace(score, **changes)  # type: ignore[arg-type]


def _writer(output_dir: Path) -> SemanticScoringRunWriter:
    return SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="corridor-02",
        run_id=SemanticScoringRunId("scoring-run-1"),
        run_index=3,
        lineage=LINEAGE,
        scorer=SCORER,
        embedding_space=SPACE,
        text_template="a photo of a {hypothesis}",
        code_version="abc123",
    )


def test_the_scores_are_persisted_apart_with_the_provenance_that_makes_them_auditable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "semantic_scoring"
    scores = (_score("claim-1", "feature-a", 0.31), _score("claim-2", "feature-a", -0.05))
    writer = _writer(output)
    writer.add_result(_result(), scores)

    manifest = writer.finalize()

    reader = SemanticScoringRunReader(output)
    assert reader.verify_integrity() == []
    assert reader.list_scores() == [
        replace(score, score_id=ScoreId(f"scoring-run-1--{score.score_id}")) for score in scores
    ]
    assert reader.manifest == manifest
    assert manifest.lineage == LINEAGE
    assert (manifest.scorer, manifest.embedding_space) == (SCORER, SPACE)
    assert manifest.embedding_space_id == SPACE_ID
    assert manifest.text_template == "a photo of a {hypothesis}"
    assert manifest.scope is FeatureScope.REGION
    assert manifest.score_type is SemanticScoreType.COSINE_SIMILARITY
    # Uma claim sem feature compatível fica sem score; as claims de cena nunca são pontuadas
    # no escopo de região. Nenhum zero é fabricado para elas.
    counts = manifest.counts
    assert counts is not None
    assert (counts.result_count, counts.region_claim_count, counts.scene_claim_count) == (1, 3, 1)
    assert (counts.scored_claim_count, counts.unscored_region_claim_count) == (2, 1)
    assert counts.score_count == 2
    assert {entry.path for entry in manifest.file_inventory} == {"outputs/scores.jsonl"}
    record = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert record["score_semantics"]["calibrated_probability"] is None
    assert "not a probability" in record["score_semantics"]["statement"]
    assert record["schema_version"] == "0.2.0"


def test_a_result_without_scores_is_recorded_as_unscored_not_as_zero(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "semantic_scoring")
    writer.add_result(_result(), ())

    manifest = writer.finalize()

    counts = manifest.counts
    assert counts is not None
    assert counts.score_count == 0
    assert counts.unscored_region_claim_count == 3
    assert SemanticScoringRunReader(tmp_path / "semantic_scoring").list_scores() == []


@pytest.mark.parametrize(
    ("score", "reason"),
    [
        (_score("claim-1", "feature-a", 0.3, embedding_space_id="sha256:other"), "space"),
        (
            _score("claim-1", "feature-a", 0.3, provenance=replace(SCORER, backend_id="x")),
            "scorer",
        ),
        (_score("claim-1", "feature-b", 0.3), "region"),
        (_score("claim-1", "feature-a", 0.3, calibrated_probability=0.5), "calibrated"),
        (_score("claim-9", "feature-a", 0.3), "unknown claim_id"),
    ],
)
def test_a_score_the_run_cannot_vouch_for_is_refused(
    tmp_path: Path, score: SemanticScore, reason: str
) -> None:
    writer = _writer(tmp_path / "semantic_scoring")

    with pytest.raises(SemanticScoringRunError, match=reason):
        writer.add_result(_result(), (score,))


def test_a_result_of_another_perception_run_or_a_repeated_one_is_refused(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "semantic_scoring")
    other = replace(_result(), visual_evidence_artifact_id="another-run")

    with pytest.raises(SemanticScoringRunError, match="producer artifact"):
        writer.add_result(other, ())
    writer.add_result(_result(), ())
    with pytest.raises(SemanticScoringRunError, match="twice"):
        writer.add_result(_result(), ())


def test_a_finished_run_is_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "semantic_scoring"
    output.mkdir()

    writer = _writer(output)
    writer.add_result(_result(), ())
    with pytest.raises(SemanticScoringRunError, match="already exists"):
        writer.finalize()


def test_a_run_of_an_unknown_schema_is_not_read(tmp_path: Path) -> None:
    output = tmp_path / "semantic_scoring"
    writer = _writer(output)
    writer.add_result(_result(), ())
    writer.finalize()
    manifest = output / "manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**record, "schema_version": "9.9.9"}), encoding="utf-8")

    with pytest.raises(SemanticScoringRunError, match="schema_version"):
        SemanticScoringRunReader(output)
