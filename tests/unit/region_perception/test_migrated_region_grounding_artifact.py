"""Grounding executions persisted with, and linked to, the perception run artifact (#566)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingRejectionReason,
    GroundingTask,
    Region2D,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    RejectedGroundingOutput,
    assemble_regions,
)
from contextmap.region_perception import (
    RegionPerceptionRunError as RunArtifactError,
)
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage

from .test_artifacts import _manifest

RUN_ID = "run-0001"
SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = str(RUN_ID)
FINGERPRINT = "sha256:grounding-config"
RAW = "<ref>chair</ref><box><100><200><300><400></box><box><500><500></box><box><9></box>"


def _request(text: str = "the red chair") -> RegionGroundingRequest:
    return RegionGroundingRequest(
        region_perception_run_id=RESULT_ID,
        image=PreparedImage(
            source_observation_id=SOURCE_ID,
            payload_reference="frame-0124.png",
            payload_artifact=ArtifactReference(
                uri="frame-0124.png", sha256="c" * 64, media_type="image/png"
            ),
            width=640,
            height=480,
        ),
        query=GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id="fake.phrase-grounding/1",
            geometry=GroundingGeometry.BOX,
            text=text,
        ),
        configuration_fingerprint=FINGERPRINT,
    )


def _execution(text: str = "the red chair", *, raw: str = RAW) -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=_request(text),
        provenance=BackendProvenance(
            backend_id="fake_region_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake-grounder",
            version="1",
            configuration_fingerprint=FINGERPRINT,
        ),
        rendered_prompt=f"Locate all the instances that match the following description: {text}.",
        raw_response=raw,
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box><100><200><300><400></box>",
                label="chair",
                box=BoundingBox2D(x=64.0, y=96.0, width=128.0, height=96.0),
                native_diagnostics=(("decode_mode", "mtp"),),
            ),
            GroundingOutput(
                output_index=1,
                native_text="<box><500><500></box>",
                label="chair",
                point=GroundingPoint(x=320.0, y=240.0),
            ),
        ),
        rejected_outputs=(
            RejectedGroundingOutput(
                output_index=2,
                native_text="<box><9></box>",
                reason=GroundingRejectionReason.MALFORMED_GEOMETRY,
                detail="a box needs four coordinates and a point two",
                label="chair",
            ),
        ),
        diagnostics=GroundingDiagnostics(
            latency_ms=41.0, native=(("stats.switch_to_ar", 1), ("stats.tps", 88.25))
        ),
        effective_configuration=MappingProxyType({"generation_mode": "hybrid"}),
        runtime_identity=MappingProxyType({"transformers": "4.57.1"}),
    )


def _regions(*executions: RegionGroundingExecution) -> tuple[Region2D, ...]:
    return assemble_regions(discovered=(), groundings=executions)


def _writer(tmp_path: Path) -> RegionPerceptionRunWriter:
    writer = RegionPerceptionRunWriter(tmp_path / "run-0001")
    writer.write_manifest(_manifest(run_id=str(RUN_ID), source_observation_ids=(str(SOURCE_ID),)))
    return writer


def _persist_regions(writer: RegionPerceptionRunWriter, regions: tuple[Region2D, ...]) -> None:
    writer.write_frame(_request().image)
    for region in regions:
        writer.write_region(writer.externalize_region_mask(SOURCE_ID, region))


def test_raw_response_and_parsed_evidence_survive_the_run_artifact(tmp_path: Path) -> None:
    execution = _execution()
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions(execution))
    writer.write_grounding(execution)
    manifest = writer.finalize()

    reader = RegionPerceptionRunReader(tmp_path / "run-0001")
    (reopened,) = reader.read_grounding()

    assert reopened == execution
    assert reopened.request.query.text == "the red chair"
    assert tuple(reader.read_regions()) == execution.regions
    assert reader.verify_integrity() == []
    paths = {entry.path for entry in manifest.file_inventory}
    raw_path = "outputs/grounding-responses/response-000000.txt"
    assert {"outputs/grounding.jsonl", raw_path} <= paths
    assert (tmp_path / "run-0001" / raw_path).read_bytes() == RAW.encode("utf-8")
    record = json.loads((tmp_path / "run-0001" / "outputs/grounding.jsonl").read_text())
    assert "raw_response" not in record


def test_a_run_without_grounding_has_no_grounding_stream(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions())
    manifest = writer.finalize()

    reader = RegionPerceptionRunReader(tmp_path / "run-0001")

    assert reader.read_grounding() == ()
    assert any(entry.path == "outputs/grounding.jsonl" for entry in manifest.file_inventory)


def test_finalize_rejects_grounding_without_its_owning_result(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_grounding(_execution())

    with pytest.raises(RunArtifactError, match="frame coverage"):
        writer.finalize()
    assert not (tmp_path / "run-0001").exists()


def test_finalize_rejects_grounded_regions_missing_from_the_result(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions())
    writer.write_grounding(_execution())

    with pytest.raises(RunArtifactError, match="materialized"):
        writer.finalize()


def test_writer_rejects_the_same_grounding_request_twice(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_grounding(_execution())

    with pytest.raises(RunArtifactError, match="duplicate grounding"):
        writer.write_grounding(_execution())


def test_distinct_queries_on_one_observation_are_separate_records(tmp_path: Path) -> None:
    first, second = _execution("the red chair"), _execution("the table")
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions(first, second))
    writer.write_grounding(first)
    writer.write_grounding(second)
    writer.finalize()

    reopened = RegionPerceptionRunReader(tmp_path / "run-0001").read_grounding()

    assert [item.request.query.text for item in reopened] == ["the red chair", "the table"]


def test_reader_refuses_a_raw_response_changed_after_publication(tmp_path: Path) -> None:
    execution = _execution()
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions(execution))
    writer.write_grounding(execution)
    writer.finalize()
    raw_path = tmp_path / "run-0001" / "outputs/grounding-responses/response-000000.txt"
    raw_path.write_text("<box>none</box>", encoding="utf-8")

    with pytest.raises(RunArtifactError, match="grounding"):
        RegionPerceptionRunReader(tmp_path / "run-0001").read_grounding()


@pytest.mark.parametrize(
    "line",
    ["not json", "[]", '{"raw_response_reference": "outputs/region-grounding-raw/x.txt"}'],
    ids=["not-json", "not-an-object", "missing-fields"],
)
def test_a_malformed_grounding_line_is_reported_with_its_file_and_line(
    tmp_path: Path, line: str
) -> None:
    """#644: every stream the reader decodes names the file and line of a malformed record."""
    execution = _execution()
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions(execution))
    writer.write_grounding(execution)
    writer.finalize()
    stream = tmp_path / "run-0001" / "outputs/grounding.jsonl"
    stream.write_text(f"{stream.read_text(encoding='utf-8')}{line}\n", encoding="utf-8")

    with pytest.raises(RunArtifactError, match=r"outputs/grounding\.jsonl:2"):
        RegionPerceptionRunReader(tmp_path / "run-0001").read_grounding()


def test_writer_refuses_grounding_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions())
    writer.finalize()

    with pytest.raises(RunArtifactError, match="finalize"):
        writer.write_grounding(replace(_execution()))
