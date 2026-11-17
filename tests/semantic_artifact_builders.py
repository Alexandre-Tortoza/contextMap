"""Create real, independently auditable semantic artifacts for evaluation fixtures."""

import json
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256

from contextmap.semantic_interpretation import (
    SEMANTIC_PROMPT_TEMPLATES,
    SemanticBackendDiagnostics,
    SemanticClaim,
    SemanticConfidencePolicy,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpretationRunId,
    SemanticInterpretationRunWriter,
    SemanticObservation,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
)
from contextmap.visual_perception import PerceptionResult


def write_semantic_views(
    writer: SemanticInterpretationRunWriter,
    views: Sequence[PerceptionResult],
    run_id: str,
    *,
    created_at: str = "",
) -> None:
    """Persist exact requests, raw responses and parsed claims for every fixture view."""
    payload = b"fixture semantic pixels"
    for view in views:
        materialized: list[SemanticClaim] = []
        for region_id in sorted(
            {claim.region_id for claim in view.claims}, key=lambda item: item or ""
        ):
            claims = tuple(claim for claim in view.claims if claim.region_id == region_id)
            mode = (
                SemanticInterpretationMode.REGION
                if region_id is not None
                else SemanticInterpretationMode.SCENE
            )
            request_id = SemanticRequestId(
                f"{run_id}-{view.source_observation_id}-{region_id or 'scene'}"
            )
            request = SemanticInterpretationRequest(
                request_id=request_id,
                semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
                source_observation_id=view.source_observation_id,
                mode=mode,
                region_id=region_id,
                configuration_fingerprint="fixture",
                prompt_template_id=f"{mode.value}/v1",
                requested_output_schema="semantic-response/1",
                visual_views=(
                    SemanticVisualView(
                        view_id=str(request_id),
                        source_observation_id=view.source_observation_id,
                        kind=VisualViewKind.TIGHT_CROP if region_id else VisualViewKind.FULL_FRAME,
                        region_id=region_id,
                        payload_reference=f"outputs/semantic-views/{request_id}.png",
                        sha256=sha256(payload).hexdigest(),
                    ),
                ),
            )
            raw = json.dumps(
                {
                    "abstained": False,
                    "scene_context": {} if region_id is None else None,
                    "claims": [
                        {
                            "hypothesis": claim.hypothesis,
                            "role": claim.role.value,
                            "category": claim.category,
                            "region_kind": claim.region_kind,
                            "attributes": {attr.name: attr.value for attr in claim.attributes},
                            "confidence": claim.confidence,
                        }
                        for claim in claims
                    ],
                }
            )
            provenance = replace(
                claims[0].provenance,
                prompt_template_id=request.prompt_template_id,
                output_schema_version=request.requested_output_schema,
            )
            parsed = parse_semantic_response(
                raw, request, provenance, confidence_policy=SemanticConfidencePolicy.MEASURED
            )
            materialized.extend(parsed.claims)
            writer.add_semantic_view_payload(request.visual_views[0], payload)
            writer.add_semantic_execution(
                SemanticInterpretationExecution(
                    request=request,
                    rendered_prompt=render_semantic_prompt(
                        request,
                        SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id],
                        confidence_policy=SemanticConfidencePolicy.MEASURED,
                    ),
                    raw_response=raw,
                    parsed=parsed,
                    diagnostics=SemanticBackendDiagnostics(latency_ms=1),
                    effective_configuration={},
                )
            )
        context = None
        if view.scene_context is not None:
            source = view.scene_context
            rid = SemanticRequestId(f"{run_id}-{view.source_observation_id}-scene")
            request = SemanticInterpretationRequest(
                request_id=rid,
                semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
                source_observation_id=view.source_observation_id,
                mode=SemanticInterpretationMode.SCENE,
                prompt_template_id="scene/v1",
                requested_output_schema="semantic-response/1",
                configuration_fingerprint="fixture",
                visual_views=(
                    SemanticVisualView(
                        view_id=str(rid),
                        kind=VisualViewKind.FULL_FRAME,
                        source_observation_id=view.source_observation_id,
                        payload_reference=f"outputs/semantic-views/{rid}.png",
                        sha256=sha256(payload).hexdigest(),
                    ),
                ),
            )
            raw = json.dumps(
                {
                    "abstained": False,
                    "scene_context": {
                        name: getattr(source, name)
                        for name in (
                            "scene_type",
                            "environment",
                            "layout",
                            "lighting",
                            "visibility",
                            "navigability",
                        )
                    },
                    "claims": [
                        {
                            "hypothesis": claim.hypothesis,
                            "role": claim.role.value,
                            "category": claim.category,
                            "region_kind": claim.region_kind,
                            "attributes": {},
                            "confidence": claim.confidence,
                        }
                        for claim in source.claims
                    ],
                }
            )
            parsed = parse_semantic_response(
                raw,
                request,
                replace(
                    source.provenance,
                    prompt_template_id="scene/v1",
                    output_schema_version=request.requested_output_schema,
                ),
                confidence_policy=SemanticConfidencePolicy.MEASURED,
            )
            context = parsed.scene_context
            writer.add_semantic_view_payload(request.visual_views[0], payload)
            writer.add_semantic_execution(
                SemanticInterpretationExecution(
                    request=request,
                    rendered_prompt=render_semantic_prompt(
                        request,
                        SEMANTIC_PROMPT_TEMPLATES["scene/v1"],
                        confidence_policy=SemanticConfidencePolicy.MEASURED,
                    ),
                    raw_response=raw,
                    parsed=parsed,
                    diagnostics=SemanticBackendDiagnostics(latency_ms=1),
                    effective_configuration={},
                )
            )
        writer.add_observation(
            SemanticObservation(
                source_observation_id=view.source_observation_id,
                semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
                claims=tuple(materialized),
                scene_context=context,
                created_at=created_at,
            )
        )
