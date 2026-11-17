"""Public visual evidence model for the Visual Perception capability.

``PerceptionResult`` is a read-only join over explicitly selected region,
feature, and semantic artifacts for one physical observation. Producer
identities and provenance remain on their respective immutable artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextmap.ingestion import SourceObservationId

# NOTE: BoundingBox2D, InlineMask e RegionId são de region_perception, dona do conceito de
# região; aqui BoundingBox2D e RegionId só continuam alcançáveis pelos nomes antigos, como os
# mesmos objetos. InlineMask é consumida de fato: é a máscara de ValidRegion e de
# ExclusionRegion, que declaram que pixels da imagem preparada são elegíveis.
from contextmap.region_perception import (  # noqa: F401
    BoundingBox2D,
    InlineMask,
    RegionId,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,  # noqa: F401
    SceneContext,
    SemanticAttribute,  # noqa: F401
    SemanticClaim,
)
from contextmap.shared import ArtifactReference, BackendProvenance, JsonScalar
from contextmap.visual_perception.visual_features import (  # noqa: F401
    FeatureId,
    FeatureScope,
    VisualFeature,
)

if TYPE_CHECKING:
    from contextmap.region_perception import Region2D
    from contextmap.semantic_scoring import SemanticScore


@dataclass(frozen=True, kw_only=True)
class PerceptionResult:
    """Read-only join of explicitly selected producer evidence for one observation.

    This view has no producer or execution identity. Each evidence item remains owned
    by the immutable artifact named here, and composing a view never persists new truth.
    """

    source_observation_id: SourceObservationId
    sequence_artifact_id: str
    region_artifact_id: str
    visual_evidence_artifact_id: str | None = None
    semantic_interpretation_artifact_id: str | None = None
    semantic_scoring_artifact_ids: tuple[str, ...] = ()
    regions: Sequence[Region2D] = ()
    features: Sequence[VisualFeature] = ()
    claims: Sequence[SemanticClaim] = ()
    scene_context: SceneContext | None = None
    semantic_scores: Sequence[SemanticScore] = ()

    def __post_init__(self) -> None:
        """Validate local identities are unique and every reference resolves.

        Raises:
            ValueError: If regions, features, or claims contain a
                duplicate local identity, or a feature/claim references
                a ``region_id`` absent from ``regions``.
        """
        for name in ("source_observation_id", "sequence_artifact_id", "region_artifact_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in ("regions", "features", "claims", "semantic_scores"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.features and self.visual_evidence_artifact_id is None:
            raise ValueError("features require an explicitly selected visual producer")
        for region in self.regions:
            if (
                region.source_observation_id is not None
                and region.source_observation_id != self.source_observation_id
            ):
                raise ValueError("region source_observation_id must match PerceptionResult")
        for feature in self.features:
            if feature.source_observation_id != self.source_observation_id:
                raise ValueError("feature source_observation_id must match PerceptionResult")
        region_ids = {region.region_id for region in self.regions}
        if len(region_ids) != len(self.regions):
            raise ValueError("duplicate region_id in PerceptionResult")
        feature_ids = {feature.feature_id for feature in self.features}
        if len(feature_ids) != len(self.features):
            raise ValueError("duplicate feature_id in PerceptionResult")
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("duplicate claim_id in PerceptionResult")
        for feature in self.features:
            if feature.region_id is not None and feature.region_id not in region_ids:
                raise ValueError(f"feature references unknown region_id: {feature.region_id!r}")
        for claim in self.claims:
            if claim.region_id is not None and claim.region_id not in region_ids:
                raise ValueError(f"claim references unknown region_id: {claim.region_id!r}")
            if claim.source_observation_id != self.source_observation_id:
                raise ValueError("claim source_observation_id must match PerceptionResult")
            if claim.semantic_interpretation_run_id != self.semantic_interpretation_artifact_id:
                raise ValueError("claim semantic producer must match the selected artifact")
        if self.scene_context is not None:
            if self.scene_context.source_observation_id != self.source_observation_id:
                raise ValueError("scene context source_observation_id must match PerceptionResult")
            if (
                self.scene_context.semantic_interpretation_run_id
                != self.semantic_interpretation_artifact_id
            ):
                raise ValueError("scene context semantic producer must match the selected artifact")
        all_claim_ids = set(claim_ids)
        if self.scene_context is not None:
            all_claim_ids.update(claim.claim_id for claim in self.scene_context.claims)
        score_ids = {score.score_id for score in self.semantic_scores}
        if len(score_ids) != len(self.semantic_scores):
            raise ValueError("duplicate score_id in PerceptionResult")
        features_by_id = {feature.feature_id: feature for feature in self.features}
        for score in self.semantic_scores:
            if score.claim_id not in all_claim_ids:
                raise ValueError(f"semantic score references unknown claim_id: {score.claim_id!r}")
            scored_feature = features_by_id.get(score.feature_id)
            if scored_feature is None:
                raise ValueError(
                    f"semantic score references unknown feature_id: {score.feature_id!r}"
                )
            if score.embedding_space_id != scored_feature.embedding_space_id:
                raise ValueError("semantic score embedding_space_id must match its feature")
            if score.source_observation_id != self.source_observation_id:
                raise ValueError("semantic score source_observation_id must match PerceptionResult")


@dataclass(frozen=True, slots=True)
class ValidRegion:
    """Explicitly identify pixels eligible for visual processing."""

    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require auditable constraint motivation and source."""
        if not self.reason or not self.source:
            raise ValueError("valid region reason and source must not be empty")


@dataclass(frozen=True, slots=True)
class ExclusionRegion:
    """Explicitly identify pixels excluded from visual processing."""

    name: str
    mask: InlineMask
    reason: str
    source: str

    def __post_init__(self) -> None:
        """Require a stable diagnostic name, motivation, and source."""
        if not self.name or not self.reason or not self.source:
            raise ValueError("exclusion name, reason, and source must not be empty")


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """Describe one applied image transformation in execution order."""

    operation: str
    provenance_source: str
    parameters: tuple[tuple[str, JsonScalar], ...]
    input_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    output_image: ArtifactReference
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the structured transformation record."""
        if not self.operation:
            raise ValueError("transformation operation must not be empty")
        if not self.provenance_source:
            raise ValueError("transformation provenance source must not be empty")
        for width, height in (self.input_dimensions, self.output_dimensions):
            if width <= 0 or height <= 0:
                raise ValueError("transformation dimensions must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit record."""
        return {
            "operation": self.operation,
            "provenance_source": self.provenance_source,
            "parameters": [{"name": name, "value": value} for name, value in self.parameters],
            "input_dimensions": list(self.input_dimensions),
            "output_dimensions": list(self.output_dimensions),
            "output_image": self.output_image.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, kw_only=True)
class PreparedImage:
    """The image-preparation output consumed by discovery/extraction/interpretation ports.

    This is Visual Perception Core's canonical contract for "an image
    ready for capability backends to consume" — any image-preparation
    implementation (resize, rectification, cropping, valid-area masking,
    ...) must produce this shape. The transformations actually applied
    are recorded for audit, but their specific parameters/implementation
    are not part of this core contract.

    Attributes:
        source_observation_id: The physical observation this was
            prepared from.
        payload_reference: Reference to the prepared image payload
            (e.g. an ``__DEBUG__/`` or output file path) — never raw pixel data
            inline.
        width: Prepared image width in pixels.
        height: Prepared image height in pixels.
        transformations: Structured, ordered record of preparation steps.
        payload_artifact: Optional content-addressed metadata for the payload.
        valid_region: Optional explicit eligibility mask.
        exclusion_regions: Optional named exclusion masks.
    """

    source_observation_id: SourceObservationId
    payload_reference: str
    width: int
    height: int
    transformations: Sequence[TransformationRecord] = ()
    payload_artifact: ArtifactReference | None = None
    valid_region: ValidRegion | None = None
    exclusion_regions: Sequence[ExclusionRegion] = ()

    def __post_init__(self) -> None:
        """Validate the prepared image has positive extent.

        Raises:
            ValueError: If ``width`` or ``height`` is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not self.payload_reference:
            raise ValueError("payload_reference must not be empty")
        if (
            self.payload_artifact is not None
            and self.payload_artifact.uri != self.payload_reference
        ):
            raise ValueError("payload artifact uri must match payload_reference")
        constraints = (() if self.valid_region is None else (self.valid_region.mask,)) + tuple(
            region.mask for region in self.exclusion_regions
        )
        if any(mask.width != self.width or mask.height != self.height for mask in constraints):
            raise ValueError("spatial constraints must match final prepared image dimensions")
        names = [region.name for region in self.exclusion_regions]
        if len(set(names)) != len(names):
            raise ValueError("exclusion regions must have unique names")

    def to_dict(self) -> dict[str, object]:
        """Return an inspectable representation for manifests and diagnostics."""
        return {
            "source_observation_id": str(self.source_observation_id),
            "payload_reference": self.payload_reference,
            "payload_artifact": (
                None if self.payload_artifact is None else self.payload_artifact.to_dict()
            ),
            "width": self.width,
            "height": self.height,
            "transformations": [record.to_dict() for record in self.transformations],
            "valid_region": (
                None
                if self.valid_region is None
                else {
                    "mask": self.valid_region.mask.to_dict(),
                    "reason": self.valid_region.reason,
                    "source": self.valid_region.source,
                }
            ),
            "exclusion_regions": [
                {
                    "name": region.name,
                    "mask": region.mask.to_dict(),
                    "reason": region.reason,
                    "source": region.source,
                }
                for region in self.exclusion_regions
            ],
        }


@dataclass(frozen=True, kw_only=True)
class SemanticSupport:
    """Legacy normalized scorer evidence retained for downstream compatibility.

    New semantic scorer adapters emit :class:`SemanticScore`, whose raw value keeps
    its declared score semantics. This contract remains separate for existing
    downstream evidence explicitly produced as normalized support in [0, 1];
    raw cosine values must never be coerced into it implicitly.
    """

    claim_id: ClaimId
    support_score: float
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate the normalized support value and scorer provenance."""
        if not 0.0 <= self.support_score <= 1.0:
            raise ValueError(f"support_score must be in [0, 1], got {self.support_score}")
        if self.provenance.capability != "semantic_scorer":
            raise ValueError("semantic support provenance capability must be semantic_scorer")
