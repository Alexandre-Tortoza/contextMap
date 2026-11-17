"""Independent semantic artifacts preserve attempts, exact views and immutable lineage."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_producer_identity import provenance

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
    SemanticObservation,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.semantic_interpretation.prompt_policy import (
    SemanticConfidencePolicy,
    parse_semantic_response,
    render_semantic_prompt,
    semantic_prompt_template,
)

PAYLOAD = b"exact image sent to backend"


def request() -> SemanticInterpretationRequest:
    """Describe one exact scene input without visual-feature dependency."""
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-request"),
        source_observation_id=SourceObservationId("obs"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("sem-a"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(
            SemanticVisualView(
                view_id="view",
                kind=VisualViewKind.FULL_FRAME,
                payload_reference="outputs/semantic-views/scene.png",
                source_observation_id=SourceObservationId("obs"),
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="config",
    )


def writer(path: Path) -> SemanticInterpretationRunWriter:
    """Create a producer run explicitly pinning the regional artifact."""
    return SemanticInterpretationRunWriter(
        output_dir=path,
        sequence_name="seq",
        run_id=SemanticInterpretationRunId("sem-a"),
        run_index=0,
        sequence_artifact_id="sequence",
        selection_id="selection",
        region_artifact_id="regions",
        configuration_digest="config",
    )


def execution() -> SemanticInterpretationExecution:
    """Build one parsed scene result with its raw response."""
    req = request()
    prov = replace(
        provenance(), prompt_template_id="scene/v1", output_schema_version="semantic-response/1"
    )
    raw = '{"abstained":false,"claims":[],"scene_context":{"scene_type":"corridor"}}'
    return SemanticInterpretationExecution(
        request=req,
        rendered_prompt=render_semantic_prompt(
            req,
            semantic_prompt_template("scene/v1"),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        raw_response=raw,
        parsed=parse_semantic_response(
            raw,
            request=req,
            provenance=prov,
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1),
        effective_configuration={"api_key": "secret"},
    )


def test_round_trip_exact_views_raw_response_and_region_lineage(tmp_path: Path) -> None:
    result = execution()
    with writer(tmp_path / "sem") as output:
        output.add_semantic_view_payload(request().visual_views[0], PAYLOAD)
        output.add_semantic_execution(result)
        output.add_observation(
            SemanticObservation(
                source_observation_id=request().source_observation_id,
                semantic_interpretation_run_id=request().semantic_interpretation_run_id,
                scene_context=result.parsed.scene_context,
            )
        )
        manifest = output.finalize()
    reader = SemanticInterpretationRunReader(tmp_path / "sem")
    assert reader.manifest == manifest
    assert manifest.region_artifact_id == "regions"
    assert manifest.schema_version == "0.1.0"
    assert reader.verify_integrity() == []
    assert next(reader.iter_semantic_executions()).raw_response == result.raw_response
    context = reader.list_observations()[0].scene_context
    assert context is not None
    assert context.provenance.raw_response_reference == "outputs/semantic-interpretations.jsonl"
    assert (tmp_path / "sem" / "outputs/semantic-views/scene.png").read_bytes() == PAYLOAD
    assert "secret" not in (tmp_path / "sem" / "outputs/semantic-interpretations.jsonl").read_text()


def test_missing_view_prevents_atomic_publication(tmp_path: Path) -> None:
    result = execution()
    with writer(tmp_path / "sem") as output:
        output.add_semantic_execution(result)
        output.add_observation(
            SemanticObservation(
                source_observation_id=request().source_observation_id,
                semantic_interpretation_run_id=request().semantic_interpretation_run_id,
                scene_context=result.parsed.scene_context,
            )
        )
        with pytest.raises(ValueError, match="view payload"):
            output.finalize()
    assert not (tmp_path / "sem").exists()


def test_corrupt_or_missing_contractual_stream_is_not_empty(tmp_path: Path) -> None:
    with writer(tmp_path / "sem") as output:
        output.add_observation(
            SemanticObservation(
                source_observation_id=request().source_observation_id,
                semantic_interpretation_run_id=request().semantic_interpretation_run_id,
            )
        )
        output.finalize()
    (tmp_path / "sem" / "outputs/semantic-observations.jsonl").unlink()
    with pytest.raises(ValueError, match="missing"):
        SemanticInterpretationRunReader(tmp_path / "sem").list_observations()


def test_parse_and_backend_failures_remain_distinct_auditable_attempts(tmp_path: Path) -> None:
    from contextmap.semantic_interpretation import (
        semantic_failure_from_backend_error,
        semantic_failure_from_parse_error,
    )

    result = execution()
    failed_parse = semantic_failure_from_parse_error(
        ValueError("invalid answer"),
        request=result.request,
        rendered_prompt=result.rendered_prompt,
        raw_response="malformed JSON",
        provenance=replace(
            provenance(), prompt_template_id="scene/v1", output_schema_version="semantic-response/1"
        ),
        diagnostics=result.diagnostics,
        effective_configuration={"api_key": "secret"},
    ).failure
    req_b = replace(result.request, request_id=SemanticRequestId("backend-failed"))
    prompt_b = render_semantic_prompt(
        req_b,
        semantic_prompt_template("scene/v1"),
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    failed_backend = semantic_failure_from_backend_error(
        TimeoutError(),
        request=req_b,
        rendered_prompt=prompt_b,
        provenance=failed_parse.provenance,
        diagnostics=result.diagnostics,
        effective_configuration={},
    ).failure
    with writer(tmp_path / "sem") as output:
        output.add_observation(
            SemanticObservation(
                source_observation_id=result.request.source_observation_id,
                semantic_interpretation_run_id=result.request.semantic_interpretation_run_id,
            )
        )
        output.add_semantic_view_payload(result.request.visual_views[0], PAYLOAD)
        output.add_failed_semantic_interpretation(failed_parse)
        output.add_failed_semantic_backend_call(failed_backend)
        output.finalize()
    reader = SemanticInterpretationRunReader(tmp_path / "sem")
    assert reader.manifest.failure_count == 2
    assert reader.tracks_semantic_failures()
    assert reader.records_semantic_backend_call_failures()
    assert next(reader.iter_failed_semantic_interpretations()).raw_response == "malformed JSON"
    assert (
        next(reader.iter_failed_semantic_backend_calls()).provenance.raw_response_reference is None
    )
    assert reader.list_semantic_executions() == []
    assert reader.verify_integrity() == []


def test_observed_outputs_cannot_be_published_without_their_execution(tmp_path: Path) -> None:
    result = execution()
    with writer(tmp_path / "sem") as output:
        output.add_observation(
            SemanticObservation(
                source_observation_id=result.request.source_observation_id,
                semantic_interpretation_run_id=result.request.semantic_interpretation_run_id,
                scene_context=result.parsed.scene_context,
            )
        )
        with pytest.raises(ValueError, match="no producing execution"):
            output.finalize()
    assert not (tmp_path / "sem").exists()


def test_finished_semantic_artifact_is_never_overwritten(tmp_path: Path) -> None:
    with writer(tmp_path / "sem") as output:
        output.finalize()
    before = (tmp_path / "sem" / "manifest.json").read_bytes()
    with writer(tmp_path / "sem") as output, pytest.raises(ValueError, match="already exists"):
        output.finalize()
    assert (tmp_path / "sem" / "manifest.json").read_bytes() == before
