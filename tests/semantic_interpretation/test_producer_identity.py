"""Producer identity remains independent from visual extraction runs."""

from dataclasses import replace

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.shared import BackendProvenance


def provenance() -> SemanticInferenceProvenance:
    """Build deterministic semantic provenance."""
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake",
            capability="semantic_interpreter",
            provider="test",
            model="fake",
            version="1",
            configuration_fingerprint="config",
        ),
        task_identity="scene-v1",
        prompt_template_id="scene-v1",
        output_schema_version="1",
    )


def test_scene_context_rejects_claim_from_other_semantic_run() -> None:
    claim = SemanticClaim(
        claim_id=ClaimId("claim"),
        source_observation_id=SourceObservationId("obs"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("sem-a"),
        hypothesis="corridor",
        role=HypothesisRole.PRIMARY,
        provenance=provenance(),
    )
    context = SceneContext(
        context_id="ctx",
        source_observation_id=claim.source_observation_id,
        semantic_interpretation_run_id=claim.semantic_interpretation_run_id,
        provenance=provenance(),
        claims=(claim,),
    )
    with pytest.raises(ValueError, match="semantic_interpretation_run_id"):
        replace(
            context,
            claims=(
                replace(claim, semantic_interpretation_run_id=SemanticInterpretationRunId("sem-b")),
            ),
        )


def test_service_is_owned_by_semantic_interpretation() -> None:
    from contextmap.semantic_interpretation import SemanticInterpretationService

    assert SemanticInterpretationService.__module__ == "contextmap.semantic_interpretation.service"


def test_scene_context_freezes_claims_and_rejects_empty_identity() -> None:
    claims: list[SemanticClaim] = []
    context = SceneContext(
        context_id="context",
        source_observation_id=SourceObservationId("obs"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("sem"),
        provenance=provenance(),
        claims=claims,
    )
    claims.clear()
    assert context.claims == ()
    with pytest.raises(ValueError, match="context_id"):
        replace(context, context_id="")


def test_request_scene_conditioning_cannot_cross_semantic_producers() -> None:
    from test_run_artifact import request

    from contextmap.region_perception import RegionId
    from contextmap.semantic_interpretation import (
        SemanticEvidenceReference,
        SemanticInterpretationMode,
    )

    context = SceneContext(
        context_id="context",
        source_observation_id=SourceObservationId("obs"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("other"),
        provenance=provenance(),
    )
    with pytest.raises(ValueError, match="semantic producer"):
        replace(
            request(),
            mode=SemanticInterpretationMode.REGION,
            region_id=RegionId("region"),
            scene_context=context,
            scene_context_reference=SemanticEvidenceReference(
                evidence_type="scene_context", evidence_id="context"
            ),
        )


def test_execution_cannot_materialize_context_from_another_semantic_producer() -> None:
    from test_run_artifact import execution

    observed = execution()
    assert observed.parsed.scene_context is not None
    foreign = replace(
        observed.parsed.scene_context,
        semantic_interpretation_run_id=SemanticInterpretationRunId("other"),
    )
    with pytest.raises(ValueError, match="parsed semantic evidence"):
        replace(observed, parsed=replace(observed.parsed, scene_context=foreign))
