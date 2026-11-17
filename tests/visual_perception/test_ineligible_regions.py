"""Regions a region-scoped stage did not process, recorded in the perception run (#650).

A box-only region under a mask-conditioned stage (AlphaCLIP features, the ``masked_subject``
view) is neither embedded nor requested, and never silently dropped: the run keeps one record
per region and stage, with the reason, in ``outputs/ineligible-regions.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import IneligibleRegion, IneligibleRegionReason, RegionId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    VisualEvidenceResult,
    VisualEvidenceRunError,
    VisualEvidenceRunId,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
)
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

RUN_ID = VisualEvidenceRunId("run-0001")
SOURCE_ID = SourceObservationId("frame-0124")
WIDTH, HEIGHT = 4, 3
BOX_ONLY = RegionId("grounded-box")
MASKED = RegionId("region-0000")
STREAM = "outputs/ineligible-regions.jsonl"


def _provenance(capability: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=f"fake_{capability}",
        capability=capability,
        provider="fake",
        model="fake",
        version="1",
    )


def _result() -> VisualEvidenceResult:
    return VisualEvidenceResult(source_observation_id=SOURCE_ID)


def _record(
    region_id: RegionId = BOX_ONLY, stage_id: str = "region_feature_extraction"
) -> IneligibleRegion:
    return IneligibleRegion(
        source_observation_id=SOURCE_ID,
        region_id=region_id,
        stage_id=stage_id,
        reason=IneligibleRegionReason.NO_MASK,
        detail="the stage is conditioned on the region mask, and the region is box-only",
    )


def _writer(tmp_path: Path, *, features: tuple[VisualFeature, ...] = ()) -> VisualEvidenceRunWriter:
    writer = VisualEvidenceRunWriter(
        output_dir=tmp_path / "run",
        sequence_name="corridor-02",
        run_id=RUN_ID,
        run_index=1,
        sequence_artifact_id="sequence-0001",
        region_artifact_id="regions",
        selection_id="sha256:aaaa",
        configuration_digest="sha256:test",
    )
    writer.add_result(VisualEvidenceResult(source_observation_id=SOURCE_ID, features=features))
    return writer


def test_each_ineligible_region_is_persisted_and_read_back(tmp_path: Path) -> None:
    records = (_record(), _record(stage_id="region_interpretation"))
    writer = _writer(tmp_path)
    for record in records:
        writer.add_ineligible_region(record)

    manifest = writer.finalize()

    reader = VisualEvidenceRunReader(tmp_path / "run")
    assert reader.verify_integrity() == []
    assert tuple(reader.list_ineligible_regions()) == records
    assert STREAM in {entry.path for entry in manifest.file_inventory}


def test_a_run_where_every_region_was_eligible_has_no_stream(tmp_path: Path) -> None:
    _writer(tmp_path).finalize()

    reader = VisualEvidenceRunReader(tmp_path / "run")
    assert reader.list_ineligible_regions() == []
    assert not (tmp_path / "run" / STREAM).exists()


def test_a_region_is_recorded_ineligible_once_per_stage(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.add_ineligible_region(_record())

    with pytest.raises(VisualEvidenceRunError, match="duplicate ineligible region"):
        writer.add_ineligible_region(_record())


def _pooling_record(source_feature_id: str) -> IneligibleRegion:
    return IneligibleRegion(
        source_observation_id=SOURCE_ID,
        region_id=MASKED,
        stage_id="dense_region_pooling",
        reason=IneligibleRegionReason.POOLING_UNAVAILABLE,
        detail=f"dense feature {source_feature_id!r} cannot support this region",
        source_feature_id=FeatureId(source_feature_id),
    )


def test_pooling_ineligibility_is_recorded_once_per_dense_source(tmp_path: Path) -> None:
    # Uma região pode falhar o pooling em mais de uma dense feature do mesmo estágio; cada falha
    # é registrada com a sua fonte, sem colidir. Repetir a mesma fonte continua sendo duplicata.
    source_features = tuple(
        VisualFeature(
            feature_id=FeatureId(identity),
            source_observation_id=SOURCE_ID,
            scope=FeatureScope.DENSE,
            embedding_space_id="space",
            shape=(1, 1, 2),
            dtype="float32",
            payload_reference=f"{identity}.npy",
            provenance=_provenance("feature_extractor"),
        )
        for identity in ("dense-0001", "dense-0002")
    )
    writer = _writer(tmp_path, features=source_features)
    records = (_pooling_record("dense-0001"), _pooling_record("dense-0002"))
    for record in records:
        writer.add_ineligible_region(record)

    with pytest.raises(VisualEvidenceRunError, match="duplicate ineligible region"):
        writer.add_ineligible_region(_pooling_record("dense-0001"))

    manifest = writer.finalize()
    reader = VisualEvidenceRunReader(tmp_path / "run")
    assert reader.verify_integrity() == []
    assert tuple(reader.list_ineligible_regions()) == records
    assert STREAM in {entry.path for entry in manifest.file_inventory}


def test_the_source_feature_belongs_only_to_a_pooling_ineligibility() -> None:
    with pytest.raises(ValueError, match="source_feature_id"):
        IneligibleRegion(
            source_observation_id=SOURCE_ID,
            region_id=MASKED,
            stage_id="dense_region_pooling",
            reason=IneligibleRegionReason.POOLING_UNAVAILABLE,
            detail="no source named",
        )
    with pytest.raises(ValueError, match="source_feature_id"):
        IneligibleRegion(
            source_observation_id=SOURCE_ID,
            region_id=BOX_ONLY,
            stage_id="region_feature_extraction",
            reason=IneligibleRegionReason.NO_MASK,
            detail="box-only",
            source_feature_id=FeatureId("dense-0001"),
        )


@pytest.mark.parametrize(
    "line", ["not json", "[]", "{}"], ids=["not-json", "not-an-object", "empty"]
)
def test_a_malformed_record_is_reported_with_its_file_and_line(tmp_path: Path, line: str) -> None:
    writer = _writer(tmp_path)
    writer.add_ineligible_region(_record())
    writer.finalize()
    stream = tmp_path / "run" / STREAM
    stream.write_text(f"{stream.read_text(encoding='utf-8')}{line}\n", encoding="utf-8")

    with pytest.raises(VisualEvidenceRunError, match=r"outputs/ineligible-regions\.jsonl:2"):
        VisualEvidenceRunReader(tmp_path / "run").list_ineligible_regions()


def test_a_record_needs_its_identity_and_a_reason_detail() -> None:
    with pytest.raises(ValueError, match="stage_id"):
        IneligibleRegion(
            source_observation_id=SOURCE_ID,
            region_id=BOX_ONLY,
            stage_id=" ",
            reason=IneligibleRegionReason.NO_MASK,
            detail="box-only",
        )
    with pytest.raises(ValueError, match="detail"):
        IneligibleRegion(
            source_observation_id=SOURCE_ID,
            region_id=BOX_ONLY,
            stage_id="region_interpretation",
            reason=IneligibleRegionReason.NO_MASK,
            detail="",
        )
