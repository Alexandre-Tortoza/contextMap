"""Contract tests for the Gemini semantic interpreter adapter (fake client, no provider)."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import (
    SEMANTIC_PROMPT_TEMPLATES,
    SemanticBackendCallFailedError,
    SemanticConfidencePolicy,
    SemanticInterpretationFailedError,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpretationRunId,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    render_semantic_prompt,
)
from contextmap.semantic_interpretation.adapters import gemini
from contextmap.semantic_interpretation.adapters.gemini import (
    GeminiProviderResponse,
    GeminiSemanticConfig,
    GeminiSemanticError,
    GeminiSemanticInterpreter,
    GeminiTransientError,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    encode_semantic_execution,
)


def _response(confidence: float | None = None) -> GeminiProviderResponse:
    return GeminiProviderResponse(
        text=json.dumps(
            {
                "abstained": False,
                "claims": [
                    {
                        "hypothesis": "warehouse",
                        "role": "primary",
                        "category": None,
                        "region_kind": None,
                        "attributes": {},
                        "confidence": confidence,
                    }
                ],
                "scene_context": {"scene_type": "warehouse"},
            }
        ),
        input_tokens=10,
        output_tokens=8,
    )


class _Client:
    def __init__(
        self,
        failures: int = 0,
        confidence: float | None = None,
        terminal: Exception | None = None,
        text: str | None = None,
    ) -> None:
        self.failures = failures
        self.confidence = confidence
        self.terminal = terminal
        self.text = text
        self.calls = 0
        self.received_views: list[object] = []
        self.prompts: list[object] = []

    def generate(self, **kwargs: object) -> GeminiProviderResponse:
        self.calls += 1
        self.received_views.append(kwargs["visual_views"])
        self.prompts.append(kwargs["prompt"])
        if self.terminal is not None:
            raise self.terminal
        if self.calls <= self.failures:
            raise GeminiTransientError("rate limited")
        if self.text is not None:
            return GeminiProviderResponse(text=self.text)
        return _response(self.confidence)


def _config(**overrides: object) -> GeminiSemanticConfig:
    values: dict[str, object] = {
        "model": "gemini-2.5-pro",
        "timeout_s": 30,
        "max_retries": 2,
        "temperature": 0,
        "thinking_budget": 128,
    }
    values.update(overrides)
    return GeminiSemanticConfig(**values)  # type: ignore[arg-type]


_SCENE_REFERENCE = "outputs/semantic-views/full.jpg"
_SCENE_PAYLOAD = b"gemini full frame"
_REGION_REFERENCE = "outputs/semantic-views/tight.jpg"
_REGION_PAYLOAD = b"gemini tight crop"


@pytest.fixture
def view_root(tmp_path: Path) -> Path:
    """The run directory holding the views the requests here name, with their exact bytes."""
    for reference, payload in (
        (_SCENE_REFERENCE, _SCENE_PAYLOAD),
        (_REGION_REFERENCE, _REGION_PAYLOAD),
    ):
        path = tmp_path / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return tmp_path


def _adapter(
    client: _Client, *, retries: int = 2, waits: list[int] | None = None, view_root: Path
) -> GeminiSemanticInterpreter:
    recorded = [] if waits is None else waits
    return GeminiSemanticInterpreter(
        config=_config(max_retries=retries),
        client=client,
        view_root=view_root,
        retry_wait=recorded.append,
    )


def _request(adapter: GeminiSemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("run-0001--frame-0001"),
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(
            SemanticVisualView(
                view_id="full",
                kind=VisualViewKind.FULL_FRAME,
                payload_reference=_SCENE_REFERENCE,
                source_observation_id=SourceObservationId("frame-0001"),
                sha256=hashlib.sha256(_SCENE_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def _region_request(adapter: GeminiSemanticInterpreter) -> SemanticInterpretationRequest:
    region = RegionId("region-0003")
    return replace(
        _request(adapter),
        request_id=SemanticRequestId("region-0003"),
        mode=SemanticInterpretationMode.REGION,
        region_id=region,
        visual_views=(
            SemanticVisualView(
                view_id="tight",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference=_REGION_REFERENCE,
                source_observation_id=SourceObservationId("frame-0001"),
                region_id=region,
                sha256=hashlib.sha256(_REGION_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="region-abstention/v1",
    )


def test_gemini_consumes_exactly_the_prompt_policy_the_request_selects(view_root: Path) -> None:
    """#542: the same backend-neutral policy Qwen renders, never an internal default."""
    client = _Client(text=json.dumps({"abstained": True, "claims": [], "scene_context": None}))
    adapter = _adapter(client, view_root=view_root)
    request = _region_request(adapter)

    execution = adapter.interpret(request)

    expected = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    assert client.prompts == [expected.text]
    assert execution.rendered_prompt == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt_template_id": "scene/v9"}, "unknown semantic prompt template 'scene/v9'"),
        ({"prompt_template_id": "region/v1"}, "mode must match"),
        ({"requested_output_schema": "semantic-response/2"}, "schema must match"),
    ],
)
def test_gemini_refuses_a_prompt_policy_it_cannot_render_before_calling_the_provider(
    changes: dict[str, str], message: str, view_root: Path
) -> None:
    client = _Client()
    adapter = _adapter(client, view_root=view_root)

    with pytest.raises(ValueError, match=message):
        adapter.interpret(replace(_request(adapter), **changes))  # type: ignore[arg-type]

    assert client.calls == 0


def test_gemini_retries_transient_failure_and_preserves_usage(view_root: Path) -> None:
    waits: list[int] = []
    client = _Client(failures=1)
    adapter = _adapter(client, waits=waits, view_root=view_root)

    execution = adapter.interpret(_request(adapter))

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.scene_context is not None
    assert execution.diagnostics.retries == 1
    assert execution.diagnostics.input_tokens == 10
    assert waits == [1]
    assert set(execution.effective_configuration) == {
        "model",
        "timeout_s",
        "max_retries",
        "temperature",
        "thinking_budget",
        "structured_output",
        "retry_backoff_s",
    }


def test_gemini_hands_the_full_view_identity_to_the_client_on_every_attempt(
    view_root: Path,
) -> None:
    client = _Client(failures=1)
    adapter = _adapter(client, view_root=view_root)
    request = _request(adapter)

    adapter.interpret(request)

    # O adapter verifica o sha256 uma vez, antes da primeira tentativa, e toda tentativa envia as
    # views inteiras com os mesmos bytes verificados (#661).
    first, retry = client.received_views
    assert first is retry
    assert tuple(item.view for item in first) == request.visual_views  # type: ignore[attr-defined]
    assert [item.payload for item in first] == [_SCENE_PAYLOAD]  # type: ignore[attr-defined]


def test_a_view_whose_bytes_changed_is_never_sent_to_the_provider(view_root: Path) -> None:
    """Bytes that reach a remote provider cannot be taken back: a diverging view stops the call."""
    (view_root / _SCENE_REFERENCE).write_bytes(b"bytes written after the request was built")
    client = _Client()
    adapter = _adapter(client, view_root=view_root)

    with pytest.raises(SemanticBackendCallFailedError) as raised:
        adapter.interpret(_request(adapter))

    assert client.calls == 0
    assert raised.value.failure.error_type == "ValueError"
    assert "does not match" in raised.value.failure.message


def test_gemini_exhausted_retries_are_explicit_without_fallback(view_root: Path) -> None:
    client = _Client(failures=3)
    adapter = _adapter(client, retries=1, view_root=view_root)
    request = _request(adapter)

    with pytest.raises(SemanticBackendCallFailedError, match="exhausted") as raised:
        adapter.interpret(request)

    assert client.calls == 2
    assert isinstance(raised.value.__cause__, GeminiSemanticError)
    # #558: the request, the prompt it was sent with and the attempts survive the failure.
    failed = raised.value.failure
    assert failed.request == request
    assert failed.rendered_prompt.template_id == "scene/v1"
    assert failed.error_type == "GeminiSemanticError"
    assert failed.diagnostics.retries == 1
    assert failed.provenance.backend == adapter.backend_provenance()


def test_gemini_rejects_model_reported_confidence(view_root: Path) -> None:
    adapter = _adapter(_Client(confidence=0.93), view_root=view_root)

    # Still rejected, but the drifting response is preserved as evidence now.
    with pytest.raises(SemanticInterpretationFailedError, match="confidence must be null"):
        adapter.interpret(_request(adapter))


def test_a_terminal_provider_error_is_not_retried_and_not_replaced(view_root: Path) -> None:
    client = _Client(terminal=GeminiSemanticError("Gemini rejected the request: 403"))
    adapter = _adapter(client, view_root=view_root)

    with pytest.raises(SemanticBackendCallFailedError, match="403") as raised:
        adapter.interpret(_request(adapter))

    assert client.calls == 1
    assert raised.value.__cause__ is client.terminal
    assert raised.value.failure.diagnostics.retries == 0


def test_malformed_structured_output_is_a_parser_failure_and_is_not_retried(
    view_root: Path,
) -> None:
    client = _Client(text='{"abstained": false, "claims": [')
    adapter = _adapter(client, view_root=view_root)

    with pytest.raises(SemanticInterpretationFailedError, match="malformed") as raised:
        adapter.interpret(_request(adapter))

    assert client.calls == 1
    # A truncated response is exactly what you need preserved to diagnose the truncation.
    assert raised.value.failure.raw_response == '{"abstained": false, "claims": ['
    assert raised.value.failure.failure.kind == "SemanticResponseParseError"


def test_an_empty_response_is_an_explicit_provider_failure_not_an_abstention(
    view_root: Path,
) -> None:
    adapter = _adapter(_Client(text="   "), view_root=view_root)

    with pytest.raises(SemanticBackendCallFailedError, match="empty or blocked") as raised:
        adapter.interpret(_request(adapter))

    assert isinstance(raised.value.__cause__, GeminiSemanticError)
    assert raised.value.failure.error_type == "GeminiSemanticError"


def test_the_default_wait_between_attempts_backs_off_exponentially(
    monkeypatch: pytest.MonkeyPatch,
    view_root: Path,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", slept.append)
    client = _Client(failures=3)
    adapter = GeminiSemanticInterpreter(
        config=_config(max_retries=3, retry_backoff_s=0.5), client=client, view_root=view_root
    )

    adapter.interpret(_request(adapter))

    assert slept == [0.5, 1.0, 2.0]


def test_the_default_wait_is_capped() -> None:
    config = _config(max_retries=10, retry_backoff_s=10)

    assert gemini.backoff_seconds(config, attempt=1) == 10
    assert gemini.backoff_seconds(config, attempt=10) == gemini.MAX_BACKOFF_SECONDS


def test_structured_output_and_backoff_change_the_fingerprint_and_are_secret_free(
    view_root: Path,
) -> None:
    plain = GeminiSemanticInterpreter(
        config=_config(structured_output=False), client=_Client(), view_root=view_root
    )
    structured = GeminiSemanticInterpreter(
        config=_config(structured_output=True), client=_Client(), view_root=view_root
    )

    assert plain.configuration_fingerprint != structured.configuration_fingerprint
    serialized = json.dumps(_config().to_dict()).casefold()
    assert "key" not in serialized
    assert "secret" not in serialized


def test_configuration_validates_the_new_settings() -> None:
    with pytest.raises(ValueError, match="retry_backoff_s"):
        _config(retry_backoff_s=-1)


def test_gemini_leaves_the_raw_response_reference_to_the_run_writer(view_root: Path) -> None:
    """#619 (VP-06): the adapter named a debug/ file; where a response is persisted is not its call.

    It cannot know it either: the same provenance is built before parsing, so it ends up in a
    successful execution or in a rejected response alike.
    """
    adapter = _adapter(_Client(), view_root=view_root)
    execution = adapter.interpret(_request(adapter))
    context = execution.parsed.scene_context
    assert context is not None
    provenances = [context.provenance]
    provenances.extend(claim.provenance for claim in (*execution.parsed.claims, *context.claims))
    assert {provenance.raw_response_reference for provenance in provenances} == {None}

    malformed = _adapter(_Client(text='{"abstained": false, "claims": ['), view_root=view_root)
    with pytest.raises(SemanticInterpretationFailedError) as raised:
        malformed.interpret(_request(malformed))
    assert raised.value.failure.provenance.raw_response_reference is None


def test_the_persisted_execution_record_carries_no_credential_field(view_root: Path) -> None:
    adapter = _adapter(_Client(), view_root=view_root)
    execution = adapter.interpret(_request(adapter))

    encoded = json.dumps(encode_semantic_execution(execution, raw_response_reference="raw.txt"))

    assert "api_key" not in encoded.casefold()
    assert "[REDACTED]" not in encoded
