"""Pure JSON codecs for semantic producer evidence."""

from __future__ import annotations

from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
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
from contextmap.shared import BackendProvenance


def encode_provenance(provenance: BackendProvenance) -> dict[str, Any]:
    """Encode backend provenance."""
    from dataclasses import asdict

    return asdict(provenance)


def decode_provenance(record: dict[str, Any]) -> BackendProvenance:
    """Decode backend provenance."""
    return BackendProvenance(**record)


def encode_claim(claim: SemanticClaim) -> dict[str, Any]:
    """Encode a :class:`SemanticClaim` into a JSON-serializable dict."""
    return {
        "claim_id": str(claim.claim_id),
        "source_observation_id": str(claim.source_observation_id),
        "semantic_interpretation_run_id": str(claim.semantic_interpretation_run_id),
        "hypothesis": claim.hypothesis,
        "role": claim.role.value,
        "provenance": encode_semantic_provenance(claim.provenance),
        "category": claim.category,
        "region_kind": None if claim.region_kind is None else claim.region_kind.value,
        "attributes": [
            {"name": attribute.name, "value": attribute.value} for attribute in claim.attributes
        ],
        "confidence": claim.confidence,
        "region_id": str(claim.region_id) if claim.region_id is not None else None,
        "evidence_references": [
            {
                "evidence_type": reference.evidence_type,
                "evidence_id": reference.evidence_id,
            }
            for reference in claim.evidence_references
        ],
    }


def decode_claim(record: dict[str, Any]) -> SemanticClaim:
    """Decode a :class:`SemanticClaim` from :func:`encode_claim`'s output."""
    return SemanticClaim(
        claim_id=ClaimId(record["claim_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        semantic_interpretation_run_id=SemanticInterpretationRunId(
            record["semantic_interpretation_run_id"]
        ),
        hypothesis=record["hypothesis"],
        role=HypothesisRole(record["role"]),
        provenance=decode_semantic_provenance(record["provenance"]),
        category=record["category"],
        region_kind=(
            None if record["region_kind"] is None else SemanticRegionKind(record["region_kind"])
        ),
        attributes=tuple(
            SemanticAttribute(name=item["name"], value=item["value"])
            for item in record["attributes"]
        ),
        confidence=record["confidence"],
        region_id=RegionId(record["region_id"]) if record["region_id"] is not None else None,
        evidence_references=tuple(
            SemanticEvidenceReference(
                evidence_type=item["evidence_type"], evidence_id=item["evidence_id"]
            )
            for item in record["evidence_references"]
        ),
    )


def encode_semantic_provenance(
    provenance: SemanticInferenceProvenance,
) -> dict[str, Any]:
    """Encode semantic inference provenance into a JSON-serializable dict."""
    return {
        "backend": encode_provenance(provenance.backend),
        "task_identity": provenance.task_identity,
        "prompt_template_id": provenance.prompt_template_id,
        "output_schema_version": provenance.output_schema_version,
        "raw_response_reference": provenance.raw_response_reference,
    }


def decode_semantic_provenance(record: dict[str, Any]) -> SemanticInferenceProvenance:
    """Decode semantic inference provenance from its canonical representation."""
    return SemanticInferenceProvenance(
        backend=decode_provenance(record["backend"]),
        task_identity=record["task_identity"],
        prompt_template_id=record["prompt_template_id"],
        output_schema_version=record["output_schema_version"],
        raw_response_reference=record["raw_response_reference"],
    )


def encode_scene_context(scene_context: SceneContext) -> dict[str, Any]:
    """Encode a :class:`SceneContext` into a JSON-serializable dict."""
    return {
        "context_id": scene_context.context_id,
        "source_observation_id": str(scene_context.source_observation_id),
        "semantic_interpretation_run_id": str(scene_context.semantic_interpretation_run_id),
        "claims": [encode_claim(claim) for claim in scene_context.claims],
        "provenance": encode_semantic_provenance(scene_context.provenance),
        "scene_type": scene_context.scene_type,
        "environment": scene_context.environment,
        "layout": scene_context.layout,
        "lighting": scene_context.lighting,
        "visibility": scene_context.visibility,
        "navigability": scene_context.navigability,
        "evidence_references": [
            {
                "evidence_type": reference.evidence_type,
                "evidence_id": reference.evidence_id,
            }
            for reference in scene_context.evidence_references
        ],
    }


def decode_scene_context(record: dict[str, Any]) -> SceneContext:
    """Decode a :class:`SceneContext` from :func:`encode_scene_context`'s output."""
    return SceneContext(
        context_id=record["context_id"],
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        semantic_interpretation_run_id=SemanticInterpretationRunId(
            record["semantic_interpretation_run_id"]
        ),
        claims=tuple(decode_claim(item) for item in record["claims"]),
        provenance=decode_semantic_provenance(record["provenance"]),
        scene_type=record["scene_type"],
        environment=record["environment"],
        layout=record["layout"],
        lighting=record["lighting"],
        visibility=record["visibility"],
        navigability=record["navigability"],
        evidence_references=tuple(
            SemanticEvidenceReference(
                evidence_type=item["evidence_type"], evidence_id=item["evidence_id"]
            )
            for item in record["evidence_references"]
        ),
    )
