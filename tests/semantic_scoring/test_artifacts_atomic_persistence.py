"""Semantic scoring artifact persistence with atomic publication and typed serialization.

Tests verify that:
1. Artifact writes are atomic (all-or-nothing, never partial)
2. SemanticScore serialization is typed and validated
3. Integrity is verified with SHA-256 hashes
4. Manifest captures schema version, file inventory, and metadata
5. Reader rejects corrupted or incomplete artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import ClaimId
from contextmap.semantic_scoring import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
    SemanticScoringLineage,
    SemanticScoringRunError,
    SemanticScoringRunReader,
    SemanticScoringRunWriter,
    decode_semantic_score,
    encode_semantic_score,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.visual_features import EmbeddingSpace, FeatureId


@pytest.fixture
def scorer_provenance() -> BackendProvenance:
    """Standard scorer provenance for tests."""
    return BackendProvenance(
        backend_id="test-scorer",
        capability="semantic_scorer",
        provider="test-provider",
        model="test-model-v1",
        version="1.0.0",
        configuration_fingerprint="sha256:config123",
    )


@pytest.fixture
def embedding_space() -> EmbeddingSpace:
    """Standard embedding space for tests, as the capability's own contract."""
    return EmbeddingSpace(
        family="clip",
        model="test-embedding",
        version="1.0.0",
        dimension=512,
        checkpoint="openai/clip-vit-base-patch32",
        layer="pooled",
        normalization="l2",
    )


@pytest.fixture
def lineage() -> SemanticScoringLineage:
    """Standard lineage for tests."""
    return SemanticScoringLineage(
        sequence_artifact_id="seq-test-001",
        selection_id="sha256:selection-001",
        region_artifact_id="regional-run-001",
        visual_evidence_artifact_id="visual-run-001",
        semantic_interpretation_artifact_id="semantic-run-001",
    )


def test_typed_encode_semantic_score(scorer_provenance: BackendProvenance) -> None:
    """Encode produces JSON-serializable dict with all typed fields."""
    score = SemanticScore(
        score_id=ScoreId("score-001"),
        claim_id=ClaimId("claim-001"),
        feature_id=FeatureId("feature-001"),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=0.75,
        calibrated_probability=None,
        embedding_space_id="sha256:embedding-001",
        source_observation_id=SourceObservationId("obs-001"),
        region_id=None,
        provenance=scorer_provenance,
    )

    encoded = encode_semantic_score(score)

    assert isinstance(encoded, dict)
    assert encoded["score_id"] == "score-001"
    assert encoded["claim_id"] == "claim-001"
    assert encoded["feature_id"] == "feature-001"
    assert encoded["score_type"] == "cosine_similarity"
    assert encoded["value"] == 0.75
    assert encoded["embedding_space_id"] == "sha256:embedding-001"
    assert encoded["provenance"]["backend_id"] == "test-scorer"


def test_typed_decode_semantic_score(scorer_provenance: BackendProvenance) -> None:
    """Decode produces validated SemanticScore from dict."""
    encoded = {
        "score_id": "score-001",
        "claim_id": "claim-001",
        "feature_id": "feature-001",
        "score_type": "cosine_similarity",
        "value": 0.5,
        "calibrated_probability": None,
        "embedding_space_id": "sha256:embedding-001",
        "source_observation_id": "obs-001",
        "region_id": None,
        "provenance": {
            "backend_id": "test-scorer",
            "capability": "semantic_scorer",
            "provider": "test",
            "model": "test-v1",
            "version": "1.0",
            "configuration_fingerprint": None,
        },
    }

    decoded = decode_semantic_score(encoded)

    assert isinstance(decoded, SemanticScore)
    assert decoded.score_id == "score-001"
    assert decoded.value == 0.5
    assert decoded.score_type == SemanticScoreType.COSINE_SIMILARITY


def test_round_trip_serialization(scorer_provenance: BackendProvenance) -> None:
    """Encode and decode round-trip preserves all data."""
    original = SemanticScore(
        score_id=ScoreId("score-rt-001"),
        claim_id=ClaimId("claim-rt-001"),
        feature_id=FeatureId("feature-rt-001"),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=-0.25,
        calibrated_probability=None,
        embedding_space_id="sha256:rt-001",
        source_observation_id=SourceObservationId("obs-rt-001"),
        region_id=None,
        provenance=scorer_provenance,
    )

    encoded = encode_semantic_score(original)
    decoded = decode_semantic_score(encoded)

    assert decoded == original


def test_decode_rejects_missing_fields() -> None:
    """Decode validates required fields are present."""
    incomplete = {
        "score_id": "score-001",
        "claim_id": "claim-001",
        # Missing feature_id, score_type, value, etc.
    }

    with pytest.raises(SemanticScoringRunError, match="missing required fields"):
        decode_semantic_score(incomplete)


def test_decode_rejects_invalid_score_type() -> None:
    """Decode validates score_type enum value."""
    invalid = {
        "score_id": "score-001",
        "claim_id": "claim-001",
        "feature_id": "feature-001",
        "score_type": "invalid_type",  # Not a valid SemanticScoreType
        "value": 0.5,
        "embedding_space_id": "sha256:001",
        "source_observation_id": "obs-001",
        "region_id": None,
        "provenance": {
            "backend_id": "scorer",
            "capability": "semantic_scorer",
            "provider": "test",
            "model": "test-v1",
            "version": "1.0",
            "configuration_fingerprint": None,
        },
    }

    with pytest.raises(SemanticScoringRunError, match="invalid score_type"):
        decode_semantic_score(invalid)


def test_atomic_publication_creates_artifact(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Atomic publication creates complete artifact in final location."""
    output_dir = tmp_path / "test_artifact"

    writer = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="test-run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="test-v1",
    )

    writer.finalize()

    # Verify artifact structure
    assert output_dir.exists(), "Output directory should exist"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "outputs" / "scores.jsonl").exists()
    assert (output_dir / "README.md").exists()

    # Verify manifest contains schema version and file inventory
    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["schema_version"] == "0.2.0"
    assert "file_inventory" in manifest_data
    assert len(manifest_data["file_inventory"]) > 0


def test_atomic_publication_never_replaces(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Atomic publication refuses to replace existing artifact."""
    output_dir = tmp_path / "test_noreplace"

    # Create first artifact
    writer1 = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="v1",
    )
    writer1.finalize()

    # Try to create second artifact to same location
    writer2 = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="run-002",
        run_index=2,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="v1",
    )

    with pytest.raises(SemanticScoringRunError, match="already exists"):
        writer2.finalize()


def test_integrity_verification_passes_for_valid_artifact(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Reader validates artifact integrity successfully for uncorrupted files."""
    output_dir = tmp_path / "test_integrity_valid"

    writer = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="test-run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="test-v1",
    )
    writer.finalize()

    reader = SemanticScoringRunReader(output_dir)
    issues = reader.verify_integrity()

    assert issues == [], f"Unexpected integrity issues: {issues}"


def test_integrity_verification_detects_size_mismatch(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Reader detects when file size doesn't match manifest."""
    output_dir = tmp_path / "test_integrity_size"

    writer = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="test-run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="test-v1",
    )
    writer.finalize()

    # Corrupt file: change content
    scores_file = output_dir / "outputs" / "scores.jsonl"
    scores_file.write_text("corrupted content\n")

    reader = SemanticScoringRunReader(output_dir)
    issues = reader.verify_integrity()

    assert len(issues) > 0, "Should detect corruption"
    assert any("size mismatch" in issue for issue in issues)


def test_integrity_verification_detects_hash_mismatch(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Reader detects when file content hash doesn't match manifest."""
    output_dir = tmp_path / "test_integrity_hash"

    writer = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="test-run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="test-v1",
    )

    # Add a score to make the file non-empty
    from contextmap.ingestion import SourceObservationId
    from contextmap.semantic_interpretation import ClaimId
    from contextmap.semantic_scoring import ScoreId, SemanticScore
    from contextmap.visual_perception.visual_features import FeatureId

    score = SemanticScore(
        score_id=ScoreId("score-001"),
        claim_id=ClaimId("claim-001"),
        feature_id=FeatureId("feature-001"),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=0.5,
        calibrated_probability=None,
        embedding_space_id="sha256:emb-001",
        source_observation_id=SourceObservationId("obs-001"),
        region_id=None,
        provenance=scorer_provenance,
    )
    writer.scores.append(score)
    writer.results_processed.add("result-001")

    writer.finalize()

    # Corrupt file: change a byte in the middle to alter hash
    scores_file = output_dir / "outputs" / "scores.jsonl"
    current = scores_file.read_bytes()
    if len(current) > 5:
        # Change byte in the middle but keep size the same
        corrupted = bytearray(current)
        corrupted[len(current) // 2] = (corrupted[len(current) // 2] + 1) % 256
        scores_file.write_bytes(bytes(corrupted))

    reader = SemanticScoringRunReader(output_dir)
    issues = reader.verify_integrity()

    # Should detect hash mismatch (size is same, but hash differs)
    assert len(issues) > 0, "Should detect corruption"
    assert any("hash mismatch" in issue for issue in issues)


def test_reader_rejects_missing_manifest(tmp_path: Path) -> None:
    """Reader rejects artifact without manifest.json."""
    output_dir = tmp_path / "no_manifest"
    output_dir.mkdir()

    with pytest.raises(SemanticScoringRunError, match="manifest not found"):
        SemanticScoringRunReader(output_dir)


def test_reader_rejects_invalid_manifest_json(tmp_path: Path) -> None:
    """Reader rejects artifact with invalid manifest JSON."""
    output_dir = tmp_path / "bad_json"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{invalid json")

    with pytest.raises(SemanticScoringRunError, match="cannot read manifest"):
        SemanticScoringRunReader(output_dir)


def test_reader_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    """Reader rejects artifact with unsupported schema version."""
    output_dir = tmp_path / "bad_schema"
    output_dir.mkdir()

    manifest = {
        "run_id": "test-001",
        "run_index": 1,
        "lineage": {
            "sequence_artifact_id": "seq-001",
            "selection_id": "sha256:sel",
            "region_artifact_id": "regional-001",
            "visual_evidence_artifact_id": "visual-001",
            "semantic_interpretation_artifact_id": "semantic-001",
        },
        "scorer": {
            "backend_id": "scorer",
            "capability": "semantic_scorer",
            "provider": "test",
            "model": "test-v1",
            "version": "1.0",
            "configuration_fingerprint": None,
        },
        "embedding_space": {},
        "embedding_space_id": "sha256:emb",
        "text_template": "test",
        "scope": "region",
        "score_type": "cosine_similarity",
        "created_at": "2026-01-01T00:00:00Z",
        "schema_version": "9.9.9",  # Unsupported version
        "file_inventory": [],
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SemanticScoringRunError, match="schema_version"):
        SemanticScoringRunReader(output_dir)


def test_manifest_contains_schema_version_and_lineage(
    tmp_path: Path,
    lineage: SemanticScoringLineage,
    scorer_provenance: BackendProvenance,
    embedding_space: EmbeddingSpace,
) -> None:
    """Manifest includes schema version and upstream lineage."""
    output_dir = tmp_path / "test_manifest"

    writer = SemanticScoringRunWriter(
        output_dir=output_dir,
        sequence_name="test-seq",
        run_id="test-run-001",
        run_index=1,
        lineage=lineage,
        scorer=scorer_provenance,
        embedding_space=embedding_space,
        text_template="a photo of a {hypothesis}",
        code_version="test-v1",
    )
    writer.finalize()

    reader = SemanticScoringRunReader(output_dir)
    manifest = reader.read_manifest()

    assert manifest.schema_version == "0.2.0"
    assert manifest.lineage.sequence_artifact_id == "seq-test-001"
    assert manifest.lineage.semantic_interpretation_artifact_id == "semantic-run-001"
    assert manifest.scorer.backend_id == "test-scorer"
    assert manifest.text_template == "a photo of a {hypothesis}"
