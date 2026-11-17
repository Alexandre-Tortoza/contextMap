"""Attempt accounting for Semantic Interpretation: attempted, parsed, parse_failed, call_failed.

Counting only materialized executions made a stage with many parse failures look better than
it is. The numbers are derived from the three contractual outcome streams by request_id, never
from StageOutcome and never inferred.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from contextmap.evaluation import (
    SemanticAttemptAccounting,
    SemanticMeasurementStatus,
    account_semantic_attempts,
    aggregate_semantic_attempts,
)
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    RenderedSemanticPrompt,
    SemanticInferenceProvenance,
    SemanticInterpretationRunError,
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
    SemanticObservation,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    FailedSemanticBackendCall,
    FailedSemanticInterpretation,
    SemanticBackendDiagnostics,
    SemanticInterpretationExecution,
    SemanticParseFailure,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
)
from contextmap.semantic_interpretation.prompt_policy import (
    SEMANTIC_PROMPT_TEMPLATES,
    SemanticConfidencePolicy,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.shared import BackendProvenance

_PAYLOAD = b"exact semantic view pixels"
_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0.1"
)


def _run_dir(tmp_path: Path, index: int = 1) -> Path:
    return tmp_path / f"run-{index:04d}"


def _writer(tmp_path: Path, index: int = 1) -> SemanticInterpretationRunWriter:
    return SemanticInterpretationRunWriter(
        output_dir=_run_dir(tmp_path, index),
        sequence_name="corridor-02",
        run_id=SemanticInterpretationRunId(f"run-{index:04d}"),
        run_index=index,
        sequence_artifact_id="corridor-02-a1b2c3",
        selection_id="sha256:aaaa",
        enabled_capabilities=frozenset({"region_discovery"}),
        region_artifact_id="regional",
        configuration_digest="sha256:test",
    )


def _result(run_id: str) -> SemanticObservation:
    return SemanticObservation(
        source_observation_id=SourceObservationId("frame-0001"),
        semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
        created_at="2026-01-01T00:00:00+00:00",
    )


def _request(run_id: str, request_id: str, view_name: str) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId(request_id),
        source_observation_id=SourceObservationId("frame-0001"),
        semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(
            SemanticVisualView(
                view_id=f"v-{request_id}",
                kind=VisualViewKind.FULL_FRAME,
                payload_reference=f"outputs/semantic-views/{view_name}.png",
                source_observation_id=SourceObservationId("frame-0001"),
                sha256=hashlib.sha256(_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )


def _rendered(request: SemanticInterpretationRequest) -> RenderedSemanticPrompt:
    return render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )


def _inference_provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="qwen",
            capability="semantic_interpreter",
            provider="alibaba",
            model="Qwen3-VL-4B",
            version="1",
        ),
        task_identity="scene",
        prompt_template_id="scene/v1",
        output_schema_version="semantic-response/1",
    )


def _execution(run_id: str, request_id: str, view_name: str) -> SemanticInterpretationExecution:
    request = _request(run_id, request_id, view_name)
    raw = json.dumps({"abstained": True, "claims": [], "scene_context": None})
    return SemanticInterpretationExecution(
        request=request,
        rendered_prompt=_rendered(request),
        raw_response=raw,
        parsed=parse_semantic_response(
            raw,
            request,
            _inference_provenance(),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        ),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        effective_configuration={"backend": "qwen"},
    )


def _failure(run_id: str, request_id: str, view_name: str) -> FailedSemanticInterpretation:
    request = _request(run_id, request_id, view_name)
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a door"}]})
    return FailedSemanticInterpretation(
        request=request,
        rendered_prompt=_rendered(request),
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=_inference_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        failure=SemanticParseFailure(
            kind="SemanticResponseParseError",
            message="claim[0] is missing required fields: ['confidence']",
        ),
        effective_configuration={"backend": "qwen"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def _finalize(
    writer: SemanticInterpretationRunWriter,
    *,
    executions: Sequence[SemanticInterpretationExecution] = (),
    failures: Sequence[FailedSemanticInterpretation] = (),
) -> None:
    writer.add_observation(_result(str(writer._manifest.run_id)))
    for execution in executions:
        writer.add_semantic_view_payload(execution.request.visual_views[0], _PAYLOAD)
        writer.add_semantic_execution(execution)
    for failed in failures:
        writer.add_semantic_view_payload(failed.request.visual_views[0], _PAYLOAD)
        writer.add_failed_semantic_interpretation(failed)
    writer.finalize()


def test_a_run_with_only_successes_reports_no_parse_failures(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])

    accounting = account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=1, parsed=1, parse_failed=0)
    assert accounting.parse_failure_rate == 0.0


def test_a_mixed_run_counts_both_streams(tmp_path: Path) -> None:
    """The number that matters: 1 of 3 attempts never became a claim."""
    writer = _writer(tmp_path)
    _finalize(
        writer,
        executions=[_execution("run-0001", "req-a", "a"), _execution("run-0001", "req-b", "b")],
        failures=[_failure("run-0001", "req-c", "c")],
    )

    accounting = account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=3, parsed=2, parse_failed=1)
    assert accounting.parse_failure_rate == pytest.approx(1 / 3)


def test_an_artifact_without_required_failure_stream_is_refused(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])
    _make_legacy(_run_dir(tmp_path))
    with pytest.raises(SemanticInterpretationRunError, match="contractual stream"):
        account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))


def _make_legacy(run_dir: Path) -> None:
    """Turn a freshly written run into a valid artifact from before the failures stream.

    Deleting the file alone would leave manifest.json citing it, which is a corrupt artifact
    that verify_integrity() rejects -- not the contract an older run actually had.
    """
    failures = "outputs/semantic-interpretation-failures.jsonl"
    (run_dir / failures).unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_inventory"] = [
        entry for entry in manifest["file_inventory"] if entry["path"] != failures
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_a_complete_run_declares_its_measurement_complete(tmp_path: Path) -> None:
    """A run that tracked failures can assert a rate; one that never tracked them cannot."""
    writer = _writer(tmp_path)
    _finalize(writer, executions=[_execution("run-0001", "req-a", "a")])

    accounting = account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))

    assert accounting.measurement_status is SemanticMeasurementStatus.COMPLETE
    assert accounting.parse_failure_rate == 0.0


def test_partial_report_accountings_do_not_become_complete_in_aggregation() -> None:
    total = aggregate_semantic_attempts(
        [
            SemanticAttemptAccounting(
                attempted=1,
                parsed=1,
                parse_failed=0,
                measurement_status=SemanticMeasurementStatus.LEGACY_LOWER_BOUND,
            ),
            SemanticAttemptAccounting(attempted=2, parsed=1, parse_failed=1),
        ]
    )
    assert total.attempted == 3
    assert total.parse_failed == 1
    assert total.measurement_status is SemanticMeasurementStatus.LEGACY_LOWER_BOUND
    assert total.parse_failure_rate is None


def test_a_run_with_no_semantic_attempt_has_an_undefined_failure_rate(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _finalize(writer)

    accounting = account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(attempted=0, parsed=0, parse_failed=0)
    assert accounting.parse_failure_rate is None, "0/0 is undefined, not zero"


def test_the_same_request_id_in_both_streams_is_refused() -> None:
    """attempted = |parsed_ids union failed_ids| only means something if they are disjoint."""
    with pytest.raises(ValueError, match="more than one outcome stream"):
        SemanticAttemptAccounting.from_request_ids(
            parsed_ids=frozenset({"req-a", "req-b"}),
            failed_ids=frozenset({"req-b"}),
            call_failed_ids=frozenset(),
        )


def test_aggregating_runs_reconciles_each_one_before_summing(tmp_path: Path) -> None:
    """Two runs may legitimately reuse a request_id; that must not read as a collision."""
    _finalize(_writer(tmp_path, 1), executions=[_execution("run-0001", "req-a", "a")])
    _finalize(
        _writer(tmp_path, 2),
        executions=[_execution("run-0002", "req-a", "a")],
        failures=[_failure("run-0002", "req-b", "b")],
    )

    total = aggregate_semantic_attempts(
        [
            account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path, 1))),
            account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path, 2))),
        ]
    )

    assert total == SemanticAttemptAccounting(attempted=3, parsed=2, parse_failed=1)
    assert total.parse_failure_rate == pytest.approx(1 / 3)


def _backend_call_failure(
    run_id: str, request_id: str, view_name: str
) -> FailedSemanticBackendCall:
    request = _request(run_id, request_id, view_name)
    return FailedSemanticBackendCall(
        request=request,
        rendered_prompt=_rendered(request),
        provenance=_inference_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0, retries=2),
        error_type="QwenInferenceError",
        message="CUDA out of memory",
        effective_configuration={"backend": "qwen"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def _add_backend_call_failure(
    writer: SemanticInterpretationRunWriter, failed: FailedSemanticBackendCall
) -> None:
    writer.add_semantic_view_payload(failed.request.visual_views[0], _PAYLOAD)
    writer.add_failed_semantic_backend_call(failed)


def test_a_call_that_never_answered_is_attempted_but_not_answered(tmp_path: Path) -> None:
    """#654: attempted counts every issued request; the parse rate keeps the answered ones.

    A backend call that failed before answering used to be absent from every contractual
    stream, so it vanished from ``attempted``. The parse failure rate is about answers the
    parser rejected, so its denominator stays the answered requests; the calls that never
    answered get their own rate over every issued request.
    """
    writer = _writer(tmp_path)
    _add_backend_call_failure(writer, _backend_call_failure("run-0001", "req-c", "c"))
    _finalize(
        writer,
        executions=[_execution("run-0001", "req-a", "a")],
        failures=[_failure("run-0001", "req-b", "b")],
    )

    accounting = account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))

    assert accounting == SemanticAttemptAccounting(
        attempted=3, parsed=1, parse_failed=1, call_failed=1
    )
    assert accounting.answered == 2
    assert accounting.parse_failure_rate == pytest.approx(1 / 2)
    assert accounting.call_failure_rate == pytest.approx(1 / 3)


def test_an_artifact_missing_backend_call_stream_is_refused(tmp_path: Path) -> None:
    _finalize(_writer(tmp_path), executions=[_execution("run-0001", "req-a", "a")])
    (_run_dir(tmp_path) / "outputs/semantic-backend-call-failures.jsonl").unlink()
    with pytest.raises(SemanticInterpretationRunError):
        account_semantic_attempts(SemanticInterpretationRunReader(_run_dir(tmp_path)))


def test_partial_answer_measurement_keeps_call_failure_rate_undefined() -> None:
    total = aggregate_semantic_attempts(
        [
            SemanticAttemptAccounting(attempted=2, parsed=1, parse_failed=0, call_failed=1),
            SemanticAttemptAccounting(
                attempted=1,
                parsed=1,
                parse_failed=0,
                measurement_status=SemanticMeasurementStatus.ANSWERED_ONLY,
            ),
        ]
    )
    assert total.attempted == 3
    assert total.parse_failure_rate == 0.0
    assert total.call_failure_rate is None


def test_one_request_id_in_two_outcome_streams_is_refused() -> None:
    """Each issued request has exactly one outcome among the three streams."""
    with pytest.raises(ValueError, match="more than one outcome stream"):
        SemanticAttemptAccounting.from_request_ids(
            parsed_ids=frozenset({"req-a"}),
            failed_ids=frozenset(),
            call_failed_ids=frozenset({"req-a"}),
        )


def test_attempted_must_reconcile_with_all_three_outcomes() -> None:
    with pytest.raises(ValueError, match="do not reconcile"):
        SemanticAttemptAccounting(attempted=2, parsed=1, parse_failed=0, call_failed=0)
