"""Contract tests for independently persisted visual evidence."""

from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_evidence_artifact import (
    VisualEvidenceResult,
    VisualEvidenceRunError,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
)
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature


def _feature(obs: str = "frame-1") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId("visual-run::frame-1::global::0"),
        source_observation_id=SourceObservationId(obs),
        scope=FeatureScope.GLOBAL,
        embedding_space_id="sha256:space",
        shape=(2,),
        dtype="float32",
        payload_reference="feature.npy",
        provenance=BackendProvenance(
            backend_id="fake",
            capability="feature_extractor",
            provider="test",
            model="fake",
            version="1",
        ),
    )


def _writer(path: Path) -> VisualEvidenceRunWriter:
    return VisualEvidenceRunWriter(
        output_dir=path,
        run_id="visual-run",
        sequence_artifact_id="sequence-1",
        region_artifact_id="region-run",
        selection_id="selection-1",
        configuration_digest="sha256:config",
        code_version="commit",
    )


def test_visual_artifact_round_trip_is_independent_and_lazy(tmp_path: Path) -> None:
    feature = _feature()
    array = np.asarray([0.25, 0.75], dtype="float32")
    writer = _writer(tmp_path / "visual")
    writer.add_result(
        VisualEvidenceResult(
            source_observation_id=SourceObservationId("frame-1"),
            prepared_image=PreparedImage(
                source_observation_id=SourceObservationId("frame-1"),
                payload_reference="/tmp/private/frame.png",
                width=8,
                height=6,
            ),
            features=(feature,),
        )
    )
    writer.add_feature_payload(feature, SourceObservationId("frame-1"), array)
    manifest = writer.finalize()
    assert manifest.contract == "VisualEvidenceArtifact"
    assert manifest.schema_version == "0.1.0"
    assert manifest.region_artifact_id == "region-run"
    reader = VisualEvidenceRunReader(tmp_path / "visual")
    assert reader.manifest == manifest
    result = reader.result(SourceObservationId("frame-1"))
    assert result.features == (feature,)
    assert result.prepared_image is not None
    assert result.prepared_image.width == 8
    assert result.prepared_image.payload_reference == "source-observation:frame-1"
    assert reader.verify_integrity() == []
    assert not (tmp_path / "visual" / "outputs" / "masks").exists()
    assert "claims" not in (tmp_path / "visual" / "outputs" / "results.jsonl").read_text()
    entry = reader.feature_store().entry(feature.source_observation_id, feature.feature_id)
    np.testing.assert_array_equal(
        reader.feature_store().load(entry.source_observation_id, entry.feature_id), array
    )


def test_visual_result_rejects_feature_from_another_observation() -> None:
    with pytest.raises(ValueError, match="source_observation_id"):
        VisualEvidenceResult(
            source_observation_id=SourceObservationId("frame-2"), features=(_feature(),)
        )


def test_visual_finalize_rejects_unowned_payload_and_leaves_no_artifact(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "visual")
    writer.add_feature_payload(
        _feature(), SourceObservationId("frame-1"), np.ones(2, dtype="float32")
    )
    with pytest.raises(VisualEvidenceRunError, match="exactly one"):
        writer.finalize()
    assert not (tmp_path / "visual").exists()
    assert list(tmp_path.iterdir()) == []


def test_visual_empty_observation_is_persisted_without_features(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "visual")
    writer.add_result(VisualEvidenceResult(source_observation_id=SourceObservationId("frame-1")))
    writer.finalize()
    reader = VisualEvidenceRunReader(tmp_path / "visual")
    assert reader.result(SourceObservationId("frame-1")).features == ()
    assert reader.manifest.feature_count == 0
    assert reader.feature_store().feature_keys() == ()


def test_visual_artifact_is_immutable(tmp_path: Path) -> None:
    _writer(tmp_path / "visual").finalize()
    with pytest.raises(VisualEvidenceRunError, match="already exists"):
        _writer(tmp_path / "visual").finalize()


def test_visual_artifact_persists_feature_ineligibility(tmp_path: Path) -> None:
    from contextmap.region_perception import IneligibleRegion, IneligibleRegionReason, RegionId

    writer = _writer(tmp_path / "visual")
    record = IneligibleRegion(
        source_observation_id=SourceObservationId("frame-1"),
        region_id=RegionId("region-1"),
        stage_id="region_feature_extraction",
        reason=IneligibleRegionReason.NO_MASK,
        detail="AlphaCLIP requires an observed mask",
    )
    writer.add_result(VisualEvidenceResult(source_observation_id=record.source_observation_id))
    writer.add_ineligible_region(record)
    writer.finalize()
    assert list(VisualEvidenceRunReader(tmp_path / "visual").iter_ineligible_regions()) == [record]


def test_feature_payload_rejects_foreign_observation_at_owned_boundary(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "visual")
    with pytest.raises(VisualEvidenceRunError, match="source_observation_id"):
        writer.add_feature_payload(
            _feature(), SourceObservationId("frame-2"), np.ones(2, dtype="float32")
        )


def test_staged_feature_payload_can_be_read_before_publication(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "visual")
    feature = _feature()
    array = np.asarray([0.6, 0.8], dtype=np.float32)
    writer.add_feature_payload(feature, feature.source_observation_id, array)
    np.testing.assert_array_equal(
        writer.staged_feature_payload(feature.source_observation_id, feature.feature_id), array
    )
    assert not (tmp_path / "visual").exists()
