"""Contract tests for the Qwen semantic interpreter adapter."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import (
    SEMANTIC_PROMPT_TEMPLATES,
    MaterializedSemanticView,
    SemanticBackendCallFailedError,
    SemanticConfidencePolicy,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpretationRunId,
    SemanticInterpreter,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    render_semantic_prompt,
)
from contextmap.semantic_interpretation.adapters.qwen import (
    QwenGenerationResponse,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    SemanticVisualInputMeasurement,
    decode_semantic_execution,
    encode_semantic_execution,
)


class _FakeQwenRuntime:
    def __init__(
        self,
        confidence: float | None = None,
        visual_inputs: tuple[SemanticVisualInputMeasurement, ...] | None = None,
    ) -> None:
        self.calls: list[tuple[tuple[MaterializedSemanticView, ...], str, QwenSemanticConfig]] = []
        self.confidence = confidence
        self.visual_inputs = visual_inputs

    def generate(
        self,
        *,
        visual_views: tuple[MaterializedSemanticView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        self.calls.append((visual_views, prompt, config))
        return QwenGenerationResponse(
            text=json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "hypothesis": "wooden pallet",
                            "role": "primary",
                            "category": None,
                            "region_kind": "thing",
                            "attributes": {},
                            "confidence": self.confidence,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=24,
            peak_memory_bytes=1024,
            warnings=("deterministic fake",),
            visual_inputs=self.visual_inputs,
        )


_VIEW_PAYLOAD = b"qwen semantic view pixels"
_VIEW_REFERENCE = "outputs/semantic-views/region-0007.jpg"


@pytest.fixture
def view_root(tmp_path: Path) -> Path:
    """The run directory holding the one view every request here names, with its bytes."""
    path = tmp_path / _VIEW_REFERENCE
    path.parent.mkdir(parents=True)
    path.write_bytes(_VIEW_PAYLOAD)
    return tmp_path


def _request(adapter: QwenSemanticInterpreter) -> SemanticInterpretationRequest:
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("request-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=RegionId("region-0007"),
        visual_views=(
            SemanticVisualView(
                view_id="tight-crop",
                kind=VisualViewKind.TIGHT_CROP,
                payload_reference=_VIEW_REFERENCE,
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=RegionId("region-0007"),
                sha256=hashlib.sha256(_VIEW_PAYLOAD).hexdigest(),
            ),
        ),
        prompt_template_id="region/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint=adapter.configuration_fingerprint,
    )


def test_qwen_maps_request_and_returns_canonical_unscored_claim(view_root: Path) -> None:
    runtime = _FakeQwenRuntime()
    config = QwenSemanticConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda:0",
        precision="bfloat16",
        quantization="4bit",
        max_new_tokens=128,
        temperature=0.0,
    )
    adapter = QwenSemanticInterpreter(config=config, runtime=runtime, view_root=view_root)
    request = _request(adapter)

    execution = adapter.interpret(request)

    assert isinstance(adapter, SemanticInterpreter)
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"
    assert execution.parsed.claims[0].confidence is None
    assert execution.diagnostics.input_tokens == 120
    assert execution.effective_configuration["quantization"] == "4bit"
    # O runtime recebe a identidade completa da view e os bytes que o sha256 dela identifica,
    # lidos e verificados pelo adapter; nunca só a referência (#661).
    assert tuple(item.view for item in runtime.calls[0][0]) == request.visual_views
    assert [item.payload for item in runtime.calls[0][0]] == [_VIEW_PAYLOAD]
    assert "region/v1" in runtime.calls[0][1]
    assert (
        adapter.backend_provenance().configuration_fingerprint == adapter.configuration_fingerprint
    )


def _cpu_adapter(runtime: _FakeQwenRuntime, view_root: Path) -> QwenSemanticInterpreter:
    return QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=runtime,
        view_root=view_root,
    )


@pytest.mark.parametrize("tampering", ["changed", "missing"])
def test_a_view_the_adapter_cannot_verify_never_reaches_the_runtime(
    view_root: Path, tampering: str
) -> None:
    """#661: the runtime decodes only bytes whose SHA-256 is the one the request names.

    The call fails before inference, recorded with the request and the prompt it was built with.
    """
    path = view_root / _VIEW_REFERENCE
    if tampering == "changed":
        path.write_bytes(b"bytes written after the request was built")
    else:
        path.unlink()
    runtime = _FakeQwenRuntime()
    adapter = _cpu_adapter(runtime, view_root)

    with pytest.raises(SemanticBackendCallFailedError) as raised:
        adapter.interpret(_request(adapter))

    assert runtime.calls == []
    expected = {"changed": ("ValueError", "does not match"), "missing": ("FileNotFoundError", "")}
    error_type, fragment = expected[tampering]
    assert raised.value.failure.error_type == error_type
    assert fragment in raised.value.failure.message


def test_qwen_consumes_exactly_the_prompt_policy_the_request_selects(view_root: Path) -> None:
    """#542: a non-canonical policy reaches the model; nothing falls back to ``region/v1``."""
    runtime = _FakeQwenRuntime()
    adapter = _cpu_adapter(runtime, view_root)
    request = replace(_request(adapter), prompt_template_id="region-abstention/v1")

    execution = adapter.interpret(request)

    expected = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES["region-abstention/v1"],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    assert runtime.calls[0][1] == expected.text
    assert execution.rendered_prompt == expected
    assert execution.parsed.claims[0].provenance.prompt_template_id == "region-abstention/v1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt_template_id": "region/v9"}, "unknown semantic prompt template 'region/v9'"),
        ({"prompt_template_id": "scene/v1"}, "mode must match"),
        ({"requested_output_schema": "semantic-response/2"}, "schema must match"),
    ],
)
def test_qwen_refuses_a_prompt_policy_it_cannot_render_before_inference(
    changes: dict[str, str], message: str, view_root: Path
) -> None:
    runtime = _FakeQwenRuntime()
    adapter = _cpu_adapter(runtime, view_root)

    with pytest.raises(ValueError, match=message):
        adapter.interpret(replace(_request(adapter), **changes))  # type: ignore[arg-type]

    assert runtime.calls == []


def test_qwen_configuration_rejects_ambiguous_generation_settings() -> None:
    with pytest.raises(ValueError, match="temperature"):
        QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            precision="bfloat16",
            max_new_tokens=128,
            temperature=-0.1,
        )

    with pytest.raises(ValueError, match="quantization"):
        QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            precision="bfloat16",
            quantization="unknown",
            max_new_tokens=128,
            temperature=0.0,
        )


def test_qwen_rejects_request_for_another_effective_configuration(view_root: Path) -> None:
    """#558: a request refused before invocation never reached the backend.

    It is a plain contract error, not a failed backend call: no prompt was sent, so there is no
    backend evidence to preserve and the runtime is never called.
    """
    runtime = _FakeQwenRuntime()
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=runtime,
        view_root=view_root,
    )
    request = _request(adapter)
    mismatched = SemanticInterpretationRequest(
        **{**request.__dict__, "configuration_fingerprint": "sha256:other"}
    )

    with pytest.raises(ValueError, match="configuration fingerprint"):
        adapter.interpret(mismatched)
    assert runtime.calls == []


class _FailingQwenRuntime:
    """A runtime that raises before producing any response, as a CUDA OOM does."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        visual_views: tuple[MaterializedSemanticView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        self.prompts.append(prompt)
        raise self.error


def test_a_qwen_runtime_failure_keeps_the_request_and_the_prompt_it_was_sent(
    view_root: Path,
) -> None:
    """#558: a provider failure before any response must not lose what was sent."""
    out_of_memory = RuntimeError("CUDA out of memory")
    runtime = _FailingQwenRuntime(out_of_memory)
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda:0",
            precision="bfloat16",
            max_new_tokens=128,
            temperature=0.0,
        ),
        runtime=runtime,
        view_root=view_root,
    )
    request = _request(adapter)

    with pytest.raises(SemanticBackendCallFailedError, match="CUDA out of memory") as raised:
        adapter.interpret(request)

    assert raised.value.__cause__ is out_of_memory, "the provider error is kept, not replaced"
    failed = raised.value.failure
    assert failed.request == request
    assert failed.rendered_prompt.text == runtime.prompts[0]
    assert failed.error_type == "RuntimeError"
    assert failed.message == "CUDA out of memory"
    assert failed.provenance.backend == adapter.backend_provenance()
    assert failed.provenance.raw_response_reference is None
    assert failed.diagnostics.retries == 0
    assert dict(failed.effective_configuration)["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_qwen_rejects_model_reported_confidence(view_root: Path) -> None:
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=_FakeQwenRuntime(confidence=0.93),
        view_root=view_root,
    )

    from contextmap.semantic_interpretation import SemanticInterpretationFailedError

    # Still a rejection, but the drifting response is now preserved instead of discarded:
    # a model reporting its own confidence is exactly the drift you need the raw text for.
    with pytest.raises(SemanticInterpretationFailedError, match="confidence must be null"):
        adapter.interpret(_request(adapter))


def test_qwen_stage_materializes_and_persists_canonical_result(
    tmp_path: Path, view_root: Path
) -> None:
    from contextmap.semantic_interpretation import (
        SemanticInterpretationRunReader,
        SemanticInterpretationRunWriter,
        SemanticObservation,
    )

    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        runtime=_FakeQwenRuntime(),
        view_root=view_root,
    )
    execution = adapter.interpret(_request(adapter))
    assert execution.parsed.claims[0].hypothesis == "wooden pallet"
    with SemanticInterpretationRunWriter(
        output_dir=tmp_path / "semantic_interpretation",
        sequence_name="sequence",
        run_id=execution.request.semantic_interpretation_run_id,
        run_index=1,
        sequence_artifact_id="sequence-artifact-0001",
        selection_id="sha256:selection",
        region_artifact_id="region-artifact-0001",
        configuration_digest="config",
    ) as writer:
        writer.add_observation(
            SemanticObservation(
                source_observation_id=execution.request.source_observation_id,
                semantic_interpretation_run_id=execution.request.semantic_interpretation_run_id,
                claims=execution.parsed.claims,
            )
        )
        writer.add_semantic_execution(execution)
        writer.add_semantic_view_payload(execution.request.visual_views[0], _VIEW_PAYLOAD)
        writer.finalize()
    reader = SemanticInterpretationRunReader(tmp_path / "semantic_interpretation")
    assert reader.list_observations()[0].claims[0].hypothesis == "wooden pallet"
    assert reader.list_semantic_executions()[0].request == _request(adapter)
    assert reader.verify_integrity() == []


def _cpu_config(**changes: Any) -> QwenSemanticConfig:
    return replace(
        QwenSemanticConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        ),
        **changes,
    )


_BUDGET = {"min_pixels": 256 * 32 * 32, "max_pixels": 1280 * 32 * 32}
"""A visual input budget of 256 to 1280 merged 32 px patches per image."""


def test_an_unset_visual_budget_keeps_the_configuration_identity_it_had_before_526(
    view_root: Path,
) -> None:
    """Existing configurations keep their fingerprint; no budget key is invented for them."""
    before_526 = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "device": "cpu",
        "precision": "float32",
        "max_new_tokens": 32,
        "temperature": 0.0,
        "quantization": None,
        "revision": None,
    }
    encoded = json.dumps(before_526, sort_keys=True, separators=(",", ":")).encode("utf-8")

    adapter = QwenSemanticInterpreter(
        config=_cpu_config(), runtime=_FakeQwenRuntime(), view_root=view_root
    )

    assert _cpu_config().to_dict() == before_526
    assert adapter.configuration_fingerprint == "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_the_visual_input_budget_is_part_of_the_configuration_identity(view_root: Path) -> None:
    """#526: the budget is a scientific input, so it is fingerprinted and recorded."""
    unset = _cpu_config()
    budget = _cpu_config(**_BUDGET)
    tighter = _cpu_config(**{**_BUDGET, "max_pixels": 640 * 32 * 32})

    fingerprints = {
        QwenSemanticInterpreter(
            config=config, runtime=_FakeQwenRuntime(), view_root=view_root
        ).configuration_fingerprint
        for config in (unset, budget, tighter)
    }
    adapter = QwenSemanticInterpreter(
        config=budget, runtime=_FakeQwenRuntime(), view_root=view_root
    )
    execution = adapter.interpret(_request(adapter))

    assert len(fingerprints) == 3
    assert execution.effective_configuration["min_pixels"] == 262_144
    assert execution.effective_configuration["max_pixels"] == 1_310_720


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"max_pixels": 1_310_720}, "set together"),
        ({"min_pixels": 262_144}, "set together"),
        ({"min_pixels": 0, "max_pixels": 1_310_720}, "min_pixels must be positive"),
        ({"min_pixels": 262_144, "max_pixels": -1}, "max_pixels must be positive"),
        ({"min_pixels": 1_310_720, "max_pixels": 262_144}, "must not exceed max_pixels"),
    ],
)
def test_an_incomplete_or_impossible_visual_budget_is_rejected(
    budget: dict[str, int], message: str
) -> None:
    """A partial budget would leave the other bound at a checkpoint default nobody recorded."""
    with pytest.raises(ValueError, match=message):
        _cpu_config(**budget)


def test_measured_visual_inputs_are_execution_diagnostics_not_identity(view_root: Path) -> None:
    measured = (
        SemanticVisualInputMeasurement(
            view_id="tight-crop", height_px=448, width_px=320, visual_tokens=140
        ),
    )
    adapter = QwenSemanticInterpreter(
        config=_cpu_config(**_BUDGET),
        runtime=_FakeQwenRuntime(visual_inputs=measured),
        view_root=view_root,
    )
    fingerprint = adapter.configuration_fingerprint

    execution = adapter.interpret(_request(adapter))
    # Como o PerceptionRunWriter persiste (#619): as claims nomeiam o registro que guarda a
    # resposta bruta.
    reference = "outputs/semantic-interpretations.jsonl"
    persisted = replace(
        execution,
        parsed=replace(
            execution.parsed,
            claims=tuple(
                replace(
                    claim, provenance=replace(claim.provenance, raw_response_reference=reference)
                )
                for claim in execution.parsed.claims
            ),
        ),
    )
    decoded = decode_semantic_execution(
        encode_semantic_execution(persisted, raw_response_reference=reference)
    )

    assert execution.diagnostics.visual_inputs == measured
    assert decoded.diagnostics.visual_inputs == measured
    assert "visual_inputs" not in execution.effective_configuration
    assert adapter.configuration_fingerprint == fingerprint


class _NonScalarAttributeQwenRuntime:
    """Reproduces a real failure family that survives the confidence fix.

    The model returns a list where `attributes` values must be scalars: 163 of the 457
    rejected responses in the real 360-frame corridor-02 run failed exactly this way. (The
    once-dominant "missing confidence" family is gone: omitting the key is now legal under
    UNSCORED_ONLY, since null was its only permitted value.)
    """

    def generate(
        self,
        *,
        visual_views: tuple[MaterializedSemanticView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        return QwenGenerationResponse(
            text=json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "hypothesis": "wooden pallet",
                            "role": "primary",
                            "category": None,
                            "region_kind": "thing",
                            "attributes": {"features": ["slatted", "wooden"]},
                            "confidence": None,
                        }
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=24,
            peak_memory_bytes=1024,
            warnings=(),
            visual_inputs=(
                SemanticVisualInputMeasurement(
                    view_id="tight-crop", height_px=448, width_px=320, visual_tokens=140
                ),
            ),
        )


def test_a_rejected_qwen_response_is_preserved_as_evidence_not_reduced_to_a_string(
    view_root: Path,
) -> None:
    """PR #438 review: the real raw_response must survive a parse failure."""
    from contextmap.semantic_interpretation import SemanticInterpretationFailedError

    config = QwenSemanticConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda:0",
        precision="bfloat16",
        quantization="4bit",
        max_new_tokens=128,
        temperature=0.0,
    )
    adapter = QwenSemanticInterpreter(
        config=config, runtime=_NonScalarAttributeQwenRuntime(), view_root=view_root
    )
    request = _request(adapter)

    with pytest.raises(SemanticInterpretationFailedError) as raised:
        adapter.interpret(request)

    failed = raised.value.failure
    assert "wooden pallet" in failed.raw_response, "the observed response must not be lost"
    assert failed.raw_response_sha256
    assert failed.failure.kind == "SemanticResponseParseError"
    assert "scalar value" in failed.failure.message
    assert failed.request.request_id == request.request_id, "shares the attempt identity"
    assert failed.provenance.backend.model == config.model
    assert failed.diagnostics.input_tokens == 120
    # #526: a rejected response still records what the model actually consumed.
    assert failed.diagnostics.visual_inputs is not None
    assert failed.diagnostics.visual_inputs[0].visual_tokens == 140
    assert failed.rendered_prompt.text, "the exact prompt sent must be preserved"


class _OmitsConfidenceQwenRuntime:
    """Replays the real failure pattern of #557: the model obeys "never invent confidence".

    Every claim leaves the `confidence` key out, as the prompt allows. Before the prompt, the
    schema and the parser agreed, this was rejected as "claim[0] is missing required fields:
    ['confidence']": 232 of the 284 failed requests of a real 150-frame corridor-02 run.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        visual_views: tuple[MaterializedSemanticView, ...],
        prompt: str,
        config: QwenSemanticConfig,
    ) -> QwenGenerationResponse:
        self.prompts.append(prompt)
        return QwenGenerationResponse(
            text=json.dumps(
                {
                    "abstained": False,
                    "claims": [
                        {
                            "hypothesis": "wooden pallet",
                            "role": "primary",
                            "category": "load carrier",
                            "region_kind": "thing",
                            "attributes": {"material": "wood"},
                        },
                        {
                            "hypothesis": "wooden crate",
                            "role": "alternative",
                            "category": None,
                            "region_kind": "thing",
                            "attributes": {},
                        },
                    ],
                    "scene_context": None,
                }
            ),
            input_tokens=120,
            output_tokens=48,
            peak_memory_bytes=1024,
            warnings=(),
        )


def test_qwen_materializes_claims_that_omit_confidence_as_its_prompt_instructs(
    view_root: Path,
) -> None:
    """Regression for #557: an unscored claim is valid evidence, not a failed request."""
    runtime = _OmitsConfidenceQwenRuntime()
    adapter = QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=128,
            temperature=0.0,
        ),
        runtime=runtime,
        view_root=view_root,
    )

    execution = adapter.interpret(_request(adapter))

    assert "omit it or set it to null" in runtime.prompts[0]
    assert [claim.hypothesis for claim in execution.parsed.claims] == [
        "wooden pallet",
        "wooden crate",
    ]
    assert [claim.confidence for claim in execution.parsed.claims] == [None, None]


def test_qwen_leaves_the_raw_response_reference_to_the_run_writer(view_root: Path) -> None:
    """#619 (VP-06): the adapter named a debug/ file; where a response lands is not its call."""
    from contextmap.semantic_interpretation import SemanticInterpretationFailedError

    config = QwenSemanticConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cpu",
        precision="float32",
        max_new_tokens=32,
        temperature=0.0,
    )
    adapter = QwenSemanticInterpreter(
        config=config, runtime=_FakeQwenRuntime(), view_root=view_root
    )
    execution = adapter.interpret(_request(adapter))
    assert [claim.provenance.raw_response_reference for claim in execution.parsed.claims] == [None]

    rejecting = QwenSemanticInterpreter(
        config=config, runtime=_NonScalarAttributeQwenRuntime(), view_root=view_root
    )
    with pytest.raises(SemanticInterpretationFailedError) as raised:
        rejecting.interpret(_request(rejecting))
    assert raised.value.failure.provenance.raw_response_reference is None
