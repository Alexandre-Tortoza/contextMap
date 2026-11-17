"""Scene-level semantic evidence owned by the semantic producer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation.semantic_claims import (
    SemanticClaim,
    SemanticEvidenceReference,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)


@dataclass(frozen=True, kw_only=True)
class SceneContext:
    """Scene-level semantic evidence.

    Scene-level evidence must never silently override region-level
    evidence; a consumer decides how to reconcile the two explicitly.

    Attributes:
        source_observation_id: Physical observation interpreted by the backend.
        semantic_interpretation_run_id: Inference result that owns this context.
        claims: Scene-level semantic claims. Every claim must have
            ``region_id=None``.
        provenance: Backend, task, prompt, schema, and raw-response traceability.
    """

    context_id: str
    source_observation_id: SourceObservationId
    semantic_interpretation_run_id: SemanticInterpretationRunId
    provenance: SemanticInferenceProvenance
    scene_type: str | None = None
    environment: str | None = None
    layout: str | None = None
    lighting: str | None = None
    visibility: str | None = None
    navigability: str | None = None
    claims: Sequence[SemanticClaim] = ()
    evidence_references: tuple[SemanticEvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate no scene-level claim references a region.

        Raises:
            ValueError: If any claim in ``claims`` has a non-``None``
                ``region_id``.
        """
        object.__setattr__(self, "claims", tuple(self.claims))
        for name in ("context_id", "source_observation_id", "semantic_interpretation_run_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"SceneContext {name} must not be empty")
        if any(claim.region_id is not None for claim in self.claims):
            raise ValueError("SceneContext claims must not reference a region_id")
        if any(claim.source_observation_id != self.source_observation_id for claim in self.claims):
            raise ValueError("SceneContext claim source_observation_id must match its context")
        if any(
            claim.semantic_interpretation_run_id != self.semantic_interpretation_run_id
            for claim in self.claims
        ):
            raise ValueError(
                "SceneContext claim semantic_interpretation_run_id must match its context"
            )
        evidence_keys = [
            (reference.evidence_type, reference.evidence_id)
            for reference in self.evidence_references
        ]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("scene evidence references must be unique")
