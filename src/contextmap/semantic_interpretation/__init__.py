"""Independent semantic interpretation capability and producer artifact."""

from typing import TYPE_CHECKING

from contextmap.semantic_interpretation.scene_context import SceneContext
from contextmap.semantic_interpretation.semantic_claims import (
    ClaimId,
    HypothesisRole,
    SemanticAttribute,
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticRegionKind,
)

if TYPE_CHECKING:
    # Em execução estes nomes chegam por ``__getattr__`` (ver ``_LAZY_EXPORTS``). O bloco
    # abaixo existe só para o type checker: sem ele, cada nome desta API pública chega ao
    # consumidor como variável ``Any``, e não como o contrato que ele declara.
    from contextmap.semantic_interpretation.evidence_views import (
        SEMANTIC_VIEW_POLICY_VERSION,
        SemanticContextBoundary,
    )
    from contextmap.semantic_interpretation.interpretation_audit import (
        redact_semantic_secrets,
    )
    from contextmap.semantic_interpretation.interpretation_execution import (
        semantic_failure_from_backend_error,
        semantic_failure_from_parse_error,
    )
    from contextmap.semantic_interpretation.interpretation_policy import (
        RegionRequestBudget,
        check_request_policy_supported,
    )
    from contextmap.semantic_interpretation.interpretation_request import (
        SemanticFeatureReference,
        SemanticRequestMetadata,
        SemanticViewConstruction,
        decode_semantic_request,
        encode_semantic_request,
        validate_semantic_request,
    )
    from contextmap.semantic_interpretation.ports import SemanticInterpreter
    from contextmap.semantic_interpretation.prompt_policy import (
        SCENE_CONTEXT_RENDERING,
        ParsedSemanticResponse,
        RenderedSemanticPrompt,
        SemanticPromptTemplate,
    )

if TYPE_CHECKING:
    from contextmap.semantic_interpretation.run_artifact import (
        SEMANTIC_INTERPRETATION_SCHEMA_VERSION,
        SemanticInterpretationRunError,
        SemanticInterpretationRunManifest,
        SemanticInterpretationRunReader,
        SemanticInterpretationRunWriter,
        SemanticObservation,
        SemanticObservationStatus,
    )

if TYPE_CHECKING:
    from contextmap.semantic_interpretation.evidence_views import (
        MaterializedSemanticView,
        SemanticMaskOutline,
        SemanticViewPolicy,
        check_view_policy_supported,
        materialize_region_views,
        materialize_scene_view,
    )
    from contextmap.semantic_interpretation.interpretation_audit import (
        SemanticDebugLevel,
        write_semantic_audit,
    )
    from contextmap.semantic_interpretation.interpretation_execution import (
        FailedSemanticBackendCall,
        FailedSemanticInterpretation,
        SemanticBackendCallFailedError,
        SemanticBackendDiagnostics,
        SemanticInterpretationExecution,
        SemanticInterpretationFailedError,
        SemanticParseFailure,
        decode_failed_semantic_interpretation,
        encode_failed_semantic_interpretation,
    )
    from contextmap.semantic_interpretation.interpretation_policy import (
        SEMANTIC_REQUEST_POLICY_VERSION,
        SemanticModePrompt,
        SemanticRequestPolicy,
    )
    from contextmap.semantic_interpretation.interpretation_request import (
        SemanticInterpretationMode,
        SemanticInterpretationRequest,
        SemanticInterpreterCapabilities,
        SemanticRequestId,
        SemanticVisualView,
        VisualViewKind,
    )
    from contextmap.semantic_interpretation.prompt_policy import (
        SEMANTIC_PROMPT_TEMPLATES,
        SemanticConfidencePolicy,
        SemanticPromptPolicy,
        SemanticResponseParseError,
        parse_semantic_response,
        render_semantic_prompt,
    )

if TYPE_CHECKING:
    from contextmap.semantic_interpretation.service import SemanticInterpretationService

_LAZY_EXPORTS = {
    "SemanticInterpretationService": "service",
    "MaterializedSemanticView": "evidence_views",
    "SemanticBackendCallFailedError": "interpretation_execution",
    "SemanticDebugLevel": "interpretation_audit",
    "SemanticInterpretationFailedError": "interpretation_execution",
    "SemanticModePrompt": "interpretation_policy",
    "SemanticPromptPolicy": "prompt_policy",
    "SemanticRequestPolicy": "interpretation_policy",
    "SemanticViewPolicy": "evidence_views",
    "FailedSemanticInterpretation": "interpretation_execution",
    "SemanticBackendDiagnostics": "interpretation_execution",
    "SemanticConfidencePolicy": "prompt_policy",
    "SemanticInterpretationExecution": "interpretation_execution",
    "SemanticInterpretationMode": "interpretation_request",
    "SemanticInterpretationRequest": "interpretation_request",
    "SemanticInterpreterCapabilities": "interpretation_request",
    "SemanticRequestId": "interpretation_request",
    "SemanticVisualView": "interpretation_request",
    "VisualViewKind": "interpretation_request",
    "SEMANTIC_PROMPT_TEMPLATES": "prompt_policy",
    "SemanticResponseParseError": "prompt_policy",
    "materialize_region_views": "evidence_views",
    "materialize_scene_view": "evidence_views",
    "parse_semantic_response": "prompt_policy",
    "render_semantic_prompt": "prompt_policy",
    "FailedSemanticBackendCall": "interpretation_execution",
    "SemanticMaskOutline": "evidence_views",
    "SemanticParseFailure": "interpretation_execution",
    "SEMANTIC_REQUEST_POLICY_VERSION": "interpretation_policy",
    "check_view_policy_supported": "evidence_views",
    "decode_failed_semantic_interpretation": "interpretation_execution",
    "encode_failed_semantic_interpretation": "interpretation_execution",
    "write_semantic_audit": "interpretation_audit",
    "SemanticObservation": "run_artifact",
    "SemanticObservationStatus": "run_artifact",
    "SemanticInterpretationRunReader": "run_artifact",
    "SemanticInterpretationRunWriter": "run_artifact",
    "SemanticInterpretationRunManifest": "run_artifact",
    "SemanticInterpretationRunError": "run_artifact",
    "SEMANTIC_INTERPRETATION_SCHEMA_VERSION": "run_artifact",
    # prompt_policy
    "ParsedSemanticResponse": "prompt_policy",
    "RenderedSemanticPrompt": "prompt_policy",
    "SCENE_CONTEXT_RENDERING": "prompt_policy",
    "SemanticPromptTemplate": "prompt_policy",
    # interpretation_request
    "SemanticFeatureReference": "interpretation_request",
    "SemanticRequestMetadata": "interpretation_request",
    "SemanticViewConstruction": "interpretation_request",
    "decode_semantic_request": "interpretation_request",
    "encode_semantic_request": "interpretation_request",
    "validate_semantic_request": "interpretation_request",
    # interpretation_policy
    "RegionRequestBudget": "interpretation_policy",
    "check_request_policy_supported": "interpretation_policy",
    # evidence_views
    "SEMANTIC_VIEW_POLICY_VERSION": "evidence_views",
    "SemanticContextBoundary": "evidence_views",
    # interpretation_execution
    "semantic_failure_from_backend_error": "interpretation_execution",
    "semantic_failure_from_parse_error": "interpretation_execution",
    # interpretation_audit
    "redact_semantic_secrets": "interpretation_audit",
    # ports
    "SemanticInterpreter": "ports",
}
"""Public symbols resolved on first access, mapped to the module that defines each.

Loading is deferred so reading semantic contracts does not import image-processing or
model runtimes. Concrete adapters are deliberately absent from the public surface.
"""


def __getattr__(name: str) -> object:
    """Resolve a public symbol declared in :data:`_LAZY_EXPORTS`.

    Args:
        name: Symbol named in this package's ``__all__``.

    Returns:
        The symbol, loaded from the module that defines it.

    Raises:
        AttributeError: If the name is not part of this capability's public API.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


__all__ = [
    "SCENE_CONTEXT_RENDERING",
    "SEMANTIC_INTERPRETATION_SCHEMA_VERSION",
    "SEMANTIC_PROMPT_TEMPLATES",
    "SEMANTIC_REQUEST_POLICY_VERSION",
    "SEMANTIC_VIEW_POLICY_VERSION",
    "ClaimId",
    "FailedSemanticBackendCall",
    "FailedSemanticInterpretation",
    "HypothesisRole",
    "MaterializedSemanticView",
    "ParsedSemanticResponse",
    "RegionRequestBudget",
    "RenderedSemanticPrompt",
    "SceneContext",
    "SemanticAttribute",
    "SemanticBackendCallFailedError",
    "SemanticBackendDiagnostics",
    "SemanticClaim",
    "SemanticConfidencePolicy",
    "SemanticContextBoundary",
    "SemanticDebugLevel",
    "SemanticEvidenceReference",
    "SemanticFeatureReference",
    "SemanticInferenceProvenance",
    "SemanticInterpretationExecution",
    "SemanticInterpretationFailedError",
    "SemanticInterpretationMode",
    "SemanticInterpretationRequest",
    "SemanticInterpretationRunError",
    "SemanticInterpretationRunId",
    "SemanticInterpretationRunManifest",
    "SemanticInterpretationRunReader",
    "SemanticInterpretationRunWriter",
    "SemanticInterpretationService",
    "SemanticInterpreter",
    "SemanticInterpreterCapabilities",
    "SemanticMaskOutline",
    "SemanticModePrompt",
    "SemanticObservation",
    "SemanticObservationStatus",
    "SemanticParseFailure",
    "SemanticPromptPolicy",
    "SemanticPromptTemplate",
    "SemanticRegionKind",
    "SemanticRequestId",
    "SemanticRequestMetadata",
    "SemanticRequestPolicy",
    "SemanticResponseParseError",
    "SemanticViewConstruction",
    "SemanticViewPolicy",
    "SemanticVisualView",
    "VisualViewKind",
    "check_request_policy_supported",
    "check_view_policy_supported",
    "decode_failed_semantic_interpretation",
    "decode_semantic_request",
    "encode_failed_semantic_interpretation",
    "encode_semantic_request",
    "materialize_region_views",
    "materialize_scene_view",
    "parse_semantic_response",
    "redact_semantic_secrets",
    "render_semantic_prompt",
    "semantic_failure_from_backend_error",
    "semantic_failure_from_parse_error",
    "validate_semantic_request",
    "write_semantic_audit",
]
