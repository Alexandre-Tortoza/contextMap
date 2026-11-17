"""Semantic claims: evidence and interpretation from VLM/semantic inference.

Uma `SemanticClaim` é uma hipótese semântica sobre a evidência visual, produzida por um backend
de interpretação semântica (LLM, VLM, etc.). Claims carregam provenance completa: qual
observação/feature foi processada, qual modelo/prompt foi usado, qual foi a resposta bruta.

Claims são evidência, não crença: elas representam um ponto de vista de um modelo em um momento,
nunca a verdade semântica da cena.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, NewType

from contextmap.ingestion import SourceObservationId
from contextmap.shared import BackendProvenance, JsonScalar

if TYPE_CHECKING:
    from contextmap.region_perception import RegionId

SemanticInterpretationRunId = NewType("SemanticInterpretationRunId", str)
"""Identity of one independent semantic interpretation producer execution."""

ClaimId = NewType("ClaimId", str)
"""Identity of one SemanticClaim, qualified by its semantic producer run."""


class HypothesisRole(Enum):
    """Whether a SemanticClaim is the leading or a secondary hypothesis."""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"


class SemanticRegionKind(Enum):
    """Whether a semantic hypothesis describes a countable thing or amorphous stuff."""

    THING = "thing"
    STUFF = "stuff"


@dataclass(frozen=True, kw_only=True)
class SemanticAttribute:
    """One backend-provided attribute attached to a semantic hypothesis."""

    name: str
    value: JsonScalar
    """Backend-defined attribute value (scalar JSON type, ``null`` included).

    ``null`` is part of the contract: a backend may report an attribute whose value it
    could not determine, and the parser already accepts it (``_is_json_scalar``). The
    annotation says ``JsonScalar`` so the type checker sees what the parser builds.
    """

    def __post_init__(self) -> None:
        """Reject unnamed attributes."""
        if not self.name.strip():
            raise ValueError("semantic attribute name must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticEvidenceReference:
    """Identify one immutable input or intermediate supporting semantic evidence."""

    evidence_type: str
    evidence_id: str

    def __post_init__(self) -> None:
        """Require both evidence type and identity for auditability."""
        if not self.evidence_type.strip():
            raise ValueError("semantic evidence type must not be empty")
        if not self.evidence_id.strip():
            raise ValueError("semantic evidence id must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticInferenceProvenance:
    """Trace a semantic output through backend, task, prompt, schema, and raw response.

    Attributes:
        backend: Backend producing this inference.
        task_identity: Versioned task/policy that determined semantic processing.
        prompt_template_id: Identifier of the prompt template used.
        output_schema_version: Version of the output schema (for parsing).
        raw_response_reference: Reference to persisted raw response (e.g.,
            ``outputs/semantic-interpretations.jsonl``). ``None`` when backend
            has no persistent output location; the run writer materializes
            this reference when persisting the evidence.
    """

    backend: BackendProvenance
    task_identity: str
    prompt_template_id: str
    output_schema_version: str
    raw_response_reference: str | None = None

    def __post_init__(self) -> None:
        """Require versioned semantic policy identities."""
        if self.backend.capability != "semantic_interpreter":
            raise ValueError("semantic inference backend capability must be semantic_interpreter")
        for field_name, value in (
            ("task_identity", self.task_identity),
            ("prompt_template_id", self.prompt_template_id),
            ("output_schema_version", self.output_schema_version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.raw_response_reference is not None and not self.raw_response_reference.strip():
            raise ValueError("raw_response_reference must not be empty when provided")


@dataclass(frozen=True, kw_only=True)
class SemanticClaim:
    """A candidate semantic interpretation of visual evidence.

    A claim preserves uncertainty rather than collapsing directly to a
    hard label: ``confidence=None`` explicitly means "unscored", never
    equivalent to a score of ``0.0`` or ``1.0``.

    Attributes:
        claim_id: Identity qualified by semantic producer, observation and request.
        source_observation_id: Physical observation interpreted by the backend.
        semantic_interpretation_run_id: Semantic execution artifact that owns this claim.
        hypothesis: The backend-proposed label or semantic hypothesis.
        role: Whether this is the ``PRIMARY`` interpretation or an
            ``ALTERNATIVE`` hypothesis.
        provenance: Backend, task, prompt, schema, and raw-response traceability.
        category: Optional coarse category for the claim.
        confidence: Optional score in ``[0, 1]``. ``None`` means unscored.
        region_id: The frozen ``Region2D`` this claim describes, when
            region-scoped. ``None`` for a scene-level claim.
        region_kind: Whether region describes a THING or STUFF (when region-scoped).
        attributes: Structured backend-provided attributes (key-value pairs).
        evidence_references: Identities of supporting observations/intermediates.
    """

    claim_id: ClaimId
    source_observation_id: SourceObservationId
    semantic_interpretation_run_id: SemanticInterpretationRunId
    hypothesis: str
    role: HypothesisRole
    provenance: SemanticInferenceProvenance
    category: str | None = None
    region_kind: SemanticRegionKind | None = None
    attributes: tuple[SemanticAttribute, ...] = ()
    confidence: float | None = None
    region_id: RegionId | None = None
    evidence_references: tuple[SemanticEvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate ``confidence`` is a valid score when present.

        Raises:
            ValueError: If ``confidence`` is set and outside ``[0, 1]``.
        """
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be empty")
        if self.confidence is not None and (
            not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        attribute_names = [attribute.name for attribute in self.attributes]
        if len(set(attribute_names)) != len(attribute_names):
            raise ValueError("semantic attribute names must be unique")
        evidence_keys = [
            (reference.evidence_type, reference.evidence_id)
            for reference in self.evidence_references
        ]
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("semantic evidence references must be unique")
