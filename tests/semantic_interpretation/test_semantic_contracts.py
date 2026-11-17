"""Contract tests for canonical Semantic Interpretation evidence."""

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import (
    SEMANTIC_PROMPT_TEMPLATES,
    ClaimId,
    FailedSemanticInterpretation,
    HypothesisRole,
    SceneContext,
    SemanticAttribute,
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticRegionKind,
)
from contextmap.shared import BackendProvenance

SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = SemanticInterpretationRunId("run-0001--frame-0124")


def _provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake-v1",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="region-labeling",
        prompt_template_id="region/v1",
        output_schema_version="semantic-response/1",
        raw_response_reference="__DEBUG__/40-semantic-interpretation/raw-response.txt",
    )


def _claim(**overrides: object) -> SemanticClaim:
    values: dict[str, object] = {
        "claim_id": ClaimId("claim-0001"),
        "source_observation_id": SOURCE_ID,
        "semantic_interpretation_run_id": RESULT_ID,
        "hypothesis": "wooden pallet",
        "role": HypothesisRole.PRIMARY,
        "provenance": _provenance(),
        "region_id": RegionId("region-0007"),
        "region_kind": SemanticRegionKind.THING,
        "attributes": (SemanticAttribute(name="material", value="wood"),),
        "evidence_references": (
            SemanticEvidenceReference(evidence_type="region", evidence_id="region-0007"),
        ),
    }
    values.update(overrides)
    return SemanticClaim(**values)  # type: ignore[arg-type]


def test_claim_preserves_scored_and_unscored_alternative_hypotheses() -> None:
    primary = _claim(confidence=None)
    alternative = _claim(
        claim_id=ClaimId("claim-0002"),
        hypothesis="wooden crate",
        role=HypothesisRole.ALTERNATIVE,
        confidence=0.42,
    )

    assert primary.confidence is None
    assert alternative.confidence == 0.42
    assert primary.role is HypothesisRole.PRIMARY
    assert alternative.role is HypothesisRole.ALTERNATIVE
    assert primary.attributes[0].value == "wood"


def test_claim_rejects_empty_hypothesis_and_duplicate_evidence() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _claim(hypothesis="  ")

    reference = SemanticEvidenceReference(evidence_type="region", evidence_id="region-0007")
    with pytest.raises(ValueError, match="evidence"):
        _claim(evidence_references=(reference, reference))


def test_scene_context_is_distinct_from_region_claims_and_keeps_context_fields() -> None:
    scene_claim = _claim(
        claim_id=ClaimId("claim-scene-0001"),
        hypothesis="industrial storage area",
        region_id=None,
        region_kind=None,
    )
    context = SceneContext(
        context_id=str(RESULT_ID),
        source_observation_id=SOURCE_ID,
        semantic_interpretation_run_id=RESULT_ID,
        provenance=_provenance(),
        scene_type="warehouse",
        environment="indoor",
        layout="aisles",
        lighting="artificial",
        visibility="clear",
        navigability="partially obstructed",
        claims=(scene_claim,),
        evidence_references=(
            SemanticEvidenceReference(evidence_type="prepared_image", evidence_id="frame-0124"),
        ),
    )

    assert context.scene_type == "warehouse"
    assert context.claims == (scene_claim,)

    with pytest.raises(ValueError, match="region_id"):
        SceneContext(
            context_id=str(RESULT_ID),
            source_observation_id=SOURCE_ID,
            semantic_interpretation_run_id=RESULT_ID,
            provenance=_provenance(),
            claims=(_claim(),),
        )


def test_scene_context_rejects_claim_from_another_inference_result() -> None:
    claim = _claim(
        region_id=None, semantic_interpretation_run_id=SemanticInterpretationRunId("another-result")
    )

    with pytest.raises(ValueError, match="semantic_interpretation_run_id"):
        SceneContext(
            context_id=str(RESULT_ID),
            source_observation_id=SOURCE_ID,
            semantic_interpretation_run_id=RESULT_ID,
            provenance=_provenance(),
            claims=(claim,),
        )


def _scene_provenance() -> SemanticInferenceProvenance:
    """``_provenance()`` describes a region call; a SCENE request needs a matching template."""
    from dataclasses import replace

    return replace(_provenance(), prompt_template_id="scene/v1", task_identity="scene-labeling")


def _failed_interpretation() -> FailedSemanticInterpretation:
    """A real backend call whose response was observed but could not be materialized."""
    import hashlib
    import json

    from contextmap.semantic_interpretation import (
        FailedSemanticInterpretation,
        SemanticBackendDiagnostics,
        SemanticConfidencePolicy,
        SemanticInterpretationMode,
        SemanticInterpretationRequest,
        SemanticParseFailure,
        SemanticRequestId,
        SemanticVisualView,
        VisualViewKind,
        render_semantic_prompt,
    )

    payload = b"pixels"
    view = SemanticVisualView(
        view_id="v-frame-0124-scene",
        kind=VisualViewKind.FULL_FRAME,
        payload_reference="outputs/semantic-views/frame-0124__scene.png",
        source_observation_id=SOURCE_ID,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    request = SemanticInterpretationRequest(
        request_id=SemanticRequestId("scene-frame-0124"),
        source_observation_id=SOURCE_ID,
        semantic_interpretation_run_id=RESULT_ID,
        mode=SemanticInterpretationMode.SCENE,
        visual_views=(view,),
        prompt_template_id="scene/v1",
        requested_output_schema="semantic-response/1",
        configuration_fingerprint="sha256:config",
    )
    rendered = render_semantic_prompt(
        request,
        SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )
    # A real Qwen failure from the campaign: the model omitted the required key.
    raw = json.dumps({"abstained": False, "claims": [{"hypothesis": "a door"}]})
    return FailedSemanticInterpretation(
        request=request,
        rendered_prompt=rendered,
        raw_response=raw,
        raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        provenance=_scene_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=12.5),
        failure=SemanticParseFailure(
            kind="SemanticResponseParseError",
            message="claim[0] is missing required fields: ['confidence']",
        ),
        effective_configuration={"backend": "qwen"},
        occurred_at="2026-01-01T00:00:00+00:00",
    )


def test_failed_interpretation_preserves_the_observed_response_as_evidence() -> None:
    """An invalid response is still evidence; only its materialization failed (PR #438 review)."""
    failed = _failed_interpretation()

    assert failed.raw_response
    assert failed.failure.kind == "SemanticResponseParseError"
    assert "confidence" in failed.failure.message
    # Shares the attempt identity with the success stream.
    assert str(failed.request.request_id) == "scene-frame-0124"
    assert failed.provenance.backend.model == "fake-v1"


def test_failed_interpretation_hashes_the_response_exactly_like_the_success_path() -> None:
    """Evidence identity must not depend on whether the parser succeeded."""
    import hashlib

    from contextmap.semantic_interpretation import (
        SemanticConfidencePolicy,
        parse_semantic_response,
    )

    failed = _failed_interpretation()
    raw = failed.raw_response
    parsed_ok = parse_semantic_response(
        '{"abstained": true, "claims": [], "scene_context": null}',
        failed.request,
        _scene_provenance(),
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )

    assert (
        parsed_ok.raw_response_sha256
        == hashlib.sha256(b'{"abstained": true, "claims": [], "scene_context": null}').hexdigest()
    )
    assert failed.raw_response_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_failed_interpretation_round_trips_through_its_encoding() -> None:
    """The failures stream is a first-class artifact, not a throwaway log."""
    from contextmap.semantic_interpretation import (
        decode_failed_semantic_interpretation,
        encode_failed_semantic_interpretation,
    )

    failed = _failed_interpretation()
    encoded = encode_failed_semantic_interpretation(failed)

    assert encoded["raw_response_sha256"] == failed.raw_response_sha256
    assert encoded["parse_failure"]["kind"] == "SemanticResponseParseError"
    assert decode_failed_semantic_interpretation(encoded) == failed


def test_a_rejected_response_raises_with_the_evidence_attached() -> None:
    """The backend must not reduce a real response to an error string (PR #438 review)."""
    import hashlib
    import json

    from contextmap.semantic_interpretation import (
        SemanticBackendDiagnostics,
        SemanticConfidencePolicy,
        SemanticInterpretationFailedError,
        parse_semantic_response,
        semantic_failure_from_parse_error,
    )
    from contextmap.semantic_interpretation.prompt_policy import (
        SemanticResponseParseError,
    )

    failed = _failed_interpretation()
    raw = failed.raw_response

    with pytest.raises(SemanticResponseParseError) as rejected:
        parse_semantic_response(
            raw,
            failed.request,
            _scene_provenance(),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )

    error = semantic_failure_from_parse_error(
        rejected.value,
        request=failed.request,
        rendered_prompt=failed.rendered_prompt,
        raw_response=raw,
        provenance=_scene_provenance(),
        diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
        effective_configuration={"backend": "qwen"},
    )

    assert isinstance(error, SemanticInterpretationFailedError)
    assert error.failure.raw_response == raw
    assert error.failure.raw_response_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert error.failure.failure.kind == "SemanticResponseParseError"
    # The parser's own message is kept verbatim; no failure taxonomy is invented here.
    assert error.failure.failure.message == str(rejected.value)
    assert json.loads(error.failure.raw_response)  # the observed response is still readable


@pytest.mark.parametrize(
    ("field", "value"),
    [("template_id", "scene/other"), ("output_schema_version", "semantic-response/2")],
)
def test_attempt_evidence_records_only_the_prompt_policy_its_request_selected(
    field: str, value: str
) -> None:
    """#542: no backend can record a prompt other than the one the request names.

    Every outcome of a backend call is held to it, so an adapter that rendered a hidden
    default (or anything else) cannot publish evidence claiming the requested policy.
    """
    from dataclasses import replace

    from contextmap.semantic_interpretation.interpretation_execution import (
        SemanticInterpretationExecution,
    )
    from contextmap.semantic_interpretation.prompt_policy import ParsedSemanticResponse

    failed = _failed_interpretation()
    substituted = replace(failed.rendered_prompt, **{field: value})
    with pytest.raises(ValueError, match="prompt"):
        replace(failed, rendered_prompt=substituted)  # type: ignore[type-var]

    execution = SemanticInterpretationExecution(
        request=failed.request,
        rendered_prompt=failed.rendered_prompt,
        raw_response=failed.raw_response,
        parsed=ParsedSemanticResponse(
            raw_response_sha256=failed.raw_response_sha256,
            claims=(),
            scene_context=None,
            abstained=True,
            diagnostics=(),
        ),
        diagnostics=failed.diagnostics,
        effective_configuration=failed.effective_configuration,
    )
    with pytest.raises(ValueError, match="prompt"):
        replace(execution, rendered_prompt=replace(execution.rendered_prompt, **{field: value}))

    # #558 acrescentou o terceiro desfecho, a chamada que falhou antes de responder: o prompt
    # que ela registra também é o que o request selecionou.
    from contextmap.semantic_interpretation import (
        SemanticBackendDiagnostics,
        semantic_failure_from_backend_error,
    )

    with pytest.raises(ValueError, match="prompt"):
        semantic_failure_from_backend_error(
            RuntimeError("CUDA out of memory"),
            request=failed.request,
            rendered_prompt=substituted,
            provenance=replace(_scene_provenance(), raw_response_reference=None),
            diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
            effective_configuration={"backend": "qwen"},
        )


def test_a_failed_backend_call_keeps_the_provider_error_verbatim() -> None:
    """#558: no response exists, so no response reference; the error text is never replaced."""
    from dataclasses import replace

    from contextmap.semantic_interpretation import (
        SemanticBackendCallFailedError,
        SemanticBackendDiagnostics,
        semantic_failure_from_backend_error,
    )

    failed = _failed_interpretation()
    unreferenced = replace(_scene_provenance(), raw_response_reference=None)

    def wrap(
        error: Exception, provenance: SemanticInferenceProvenance
    ) -> SemanticBackendCallFailedError:
        return semantic_failure_from_backend_error(
            error,
            request=failed.request,
            rendered_prompt=failed.rendered_prompt,
            provenance=provenance,
            diagnostics=SemanticBackendDiagnostics(latency_ms=1.0),
            effective_configuration={"backend": "qwen"},
        )

    # Um TimeoutError() sem mensagem continua sendo o erro do provider, nao um erro nosso.
    bare = wrap(TimeoutError(), unreferenced)
    assert bare.failure.error_type == "TimeoutError"
    assert bare.failure.message == ""
    with pytest.raises(ValueError, match="no raw response"):
        wrap(RuntimeError("CUDA out of memory"), _scene_provenance())
