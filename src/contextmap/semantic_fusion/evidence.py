"""Contratos de **evidência**: o que uma vista observou, referenciado por identidade.

Evidência é o que uma execução específica produziu sobre um suporte. Uma
:class:`EvidenceContribution` é *uma vista*: uma região de um frame físico como
interpretada por uma execução de inferência. Várias execuções sobre o mesmo frame são
inferência correlacionada, então a contribuição carrega a identidade do frame físico e
um :class:`PhysicalObservationGroup` conta frames separadamente das inferências.

Os sinais ficam tipados e sem mistura: a confiança da própria claim, o suporte de um
scorer e a qualidade mensurável da observação são três quantidades diferentes. A
qualidade é apenas *referenciada*, nunca vira :class:`SupportSignal`, e um sinal com
``value`` ``None`` é não pontuado -- nunca zero.

Nada aqui copia XYZ, embeddings, claims ou componentes de qualidade. Este módulo não
possui crença acumulada (ver ``fused_evidence``) nem peso derivado (ver
``quality_weighting``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from contextmap.geometric_mapping import GeometryReference
from contextmap.ingestion import SourceObservationId
from contextmap.point_representation import PointRepresentationId, PointRepresentationRunId
from contextmap.region_perception import RegionId
from contextmap.semantic_fusion.identity import EvidenceContributionId
from contextmap.semantic_fusion.invariants import (
    producer_key,
    require_canonical,
    require_geometry_support,
    require_identified,
    require_present,
)
from contextmap.semantic_interpretation import ClaimId
from contextmap.semantic_scoring import SemanticScoreType
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import BackendProvenance, SourceTimestamp
from contextmap.visual_perception import FeatureId, FeatureScope


@dataclass(frozen=True, kw_only=True)
class SemanticClaimRef:
    """Reference to a producer-qualified claim carried by one evidence contribution."""

    claim_id: ClaimId

    def __post_init__(self) -> None:
        """Require the actual semantic evidence identity."""
        if not str(self.claim_id).strip():
            raise ValueError("claim_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class VisualFeatureRef:
    """Reference visual evidence by its artifact, physical observation and feature."""

    visual_artifact_id: str
    source_observation_id: SourceObservationId
    feature_id: FeatureId
    embedding_space_id: str
    scope: FeatureScope
    region_id: RegionId | None

    def __post_init__(self) -> None:
        """Require exact producer/observation identity and coherent region scope."""
        for name in (
            "visual_artifact_id",
            "source_observation_id",
            "feature_id",
            "embedding_space_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if (self.scope is FeatureScope.REGION) != (self.region_id is not None):
            raise ValueError("region_id must be present exactly for region-scoped features")


class EvidenceChannel(Enum):
    """The typed evidence channels a fusion policy can declare.

    Each channel keeps its own semantics, scale and provenance. None is renamed, converted
    into or averaged with another, and a channel takes part only when the policy declares
    it: data being available never activates it.

    Attributes:
        SEMANTIC_CLAIMS: Hypotheses, alternatives and abstentions from an interpreter.
        SEMANTIC_SCORES: Scorer outputs, each specific to its scorer unless calibrated.
        VISUAL_FEATURES: References to visual features, with their embedding space.
        OBSERVATION_QUALITY: References to the measurable view quality.
        GEOMETRY_SUPPORT: The geometry an observation sees. It is intrinsic to every
            contribution, so it is always active and cannot be ablated.
        POINT_REPRESENTATION: References to static 3D structure, attached to the support.
    """

    SEMANTIC_CLAIMS = "semantic_claims"
    SEMANTIC_SCORES = "semantic_scores"
    VISUAL_FEATURES = "visual_features"
    OBSERVATION_QUALITY = "observation_quality"
    GEOMETRY_SUPPORT = "geometry_support"
    POINT_REPRESENTATION = "point_representation"


class SupportSignalKind(Enum):
    """The typed origin of a :class:`SupportSignal`.

    The two kinds are different quantities that are never combined here. Observation
    quality is deliberately **not** a kind: it is measured upstream and only referenced.

    Attributes:
        CLAIM_CONFIDENCE: The confidence an interpreter reported for its own claim.
        SCORER_SUPPORT: How well visual evidence supports a claim according to a scorer.
    """

    CLAIM_CONFIDENCE = "claim_confidence"
    SCORER_SUPPORT = "scorer_support"


@dataclass(frozen=True, kw_only=True)
class ScoreReference:
    """Reference to the support a scorer assigned to one claim.

    Attributes:
        claim_id: The scored claim, local to the contribution's perception result.
        scorer: The scorer that produced the score.
    """

    claim_id: ClaimId
    scorer: BackendProvenance

    def __post_init__(self) -> None:
        """Require the claim and the scorer identity.

        Raises:
            ValueError: If the claim is empty or the scorer is not identified.
        """
        require_present(self, "claim_id")
        require_identified("scorer", self.scorer)


@dataclass(frozen=True, kw_only=True)
class RawSemanticScoreReference:
    """Raw scorer evidence attached to a claim without calibration or reinterpretation.

    Cosine similarities retain their native ``[-1, 1]`` range.  They deliberately do
    not become :class:`SupportSignal`, whose contract is normalized support only.
    """

    semantic_scoring_run_id: str
    score_id: str
    claim_id: ClaimId
    feature_id: FeatureId
    score_type: SemanticScoreType
    value: float
    embedding_space_id: str
    scorer: BackendProvenance

    def __post_init__(self) -> None:
        """Validate identity, native numerical range, and scorer provenance."""
        require_present(
            self,
            "semantic_scoring_run_id",
            "score_id",
            "claim_id",
            "feature_id",
            "embedding_space_id",
        )
        require_identified("scorer", self.scorer)
        if not math.isfinite(self.value):
            raise ValueError("raw semantic score value must be finite")
        if self.score_type is SemanticScoreType.COSINE_SIMILARITY and not -1.0 <= self.value <= 1.0:
            raise ValueError("raw cosine similarity must be within [-1, 1]")


@dataclass(frozen=True, kw_only=True)
class ObservationQualityRef:
    """Reference to the measurable quality of one spatial observation.

    The quality itself stays with Sensor Association as separate typed components;
    a reference is all a contribution carries, so quality cannot be mistaken for a
    semantic score.

    Attributes:
        spatial_observation_id: The observation the quality describes.
        definitions_version: Version of the quality component definitions.
    """

    spatial_observation_id: SpatialObservationId
    definitions_version: str

    def __post_init__(self) -> None:
        """Require the observation and the definitions version.

        Raises:
            ValueError: If either is empty.
        """
        require_present(self, "spatial_observation_id", "definitions_version")


@dataclass(frozen=True, kw_only=True)
class EvidenceContribution:
    """One view's evidence about a support: one region of one frame, one inference run.

    ``physical_observation_id`` is the correlation key: contributions that share it
    are correlated inference over the same sensor data, never independent views.
    A claim appears once however many geometry elements the region sees.

    Attributes:
        contribution_id: Identity of this contribution.
        physical_observation_id: The physical frame that was interpreted.
        semantic_interpretation_artifact_id: The inference result that produced the claims.
        region_artifact_id: The run that owns that result.
        spatial_observation_id: The spatial observation this view comes from.
        region_id: The observed region.
        geometry_support: The geometry the region sees, sorted by ``geometry_id`` and
            unique, all from one map.
        claim_refs: Claims about the region, sorted by claim and unique; empty when the
            inference produced none.
        score_refs: Scorer outputs for those claims, sorted and unique.
        visual_feature_refs: Features describing the region, sorted by feature and unique.
        observation_quality: Reference to the measurable view quality, when available.
    """

    contribution_id: EvidenceContributionId
    physical_observation_id: SourceObservationId
    semantic_interpretation_artifact_id: str | None
    region_artifact_id: str
    spatial_observation_id: SpatialObservationId
    region_id: RegionId
    geometry_support: tuple[GeometryReference, ...]
    claim_refs: tuple[SemanticClaimRef, ...]
    score_refs: tuple[ScoreReference, ...]
    visual_feature_refs: tuple[VisualFeatureRef, ...]
    observation_quality: ObservationQualityRef | None
    raw_score_refs: tuple[RawSemanticScoreReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate identities, geometry and the internal consistency of the references.

        Raises:
            ValueError: If an identity or the geometry is empty, the geometry spans more than
                one map, a collection is not sorted and unique, a score names a claim the
                contribution does not carry, a region-scoped feature belongs to another
                region, or the quality reference describes another observation.
        """
        require_present(
            self,
            "contribution_id",
            "physical_observation_id",
            "region_artifact_id",
            "spatial_observation_id",
            "region_id",
        )
        require_geometry_support("geometry_support", self.geometry_support)
        if len({reference.map_id for reference in self.geometry_support}) > 1:
            raise ValueError("geometry_support must belong to one map")
        require_canonical("claim_refs", self.claim_refs, lambda ref: (ref.claim_id,))
        if self.claim_refs and self.semantic_interpretation_artifact_id is None:
            raise ValueError("claims require a semantic producer")
        if (
            self.semantic_interpretation_artifact_id is not None
            and not self.semantic_interpretation_artifact_id.strip()
        ):
            raise ValueError("semantic_interpretation_artifact_id must not be empty")
        claim_ids = {ref.claim_id for ref in self.claim_refs}
        for score in self.score_refs:
            if score.claim_id not in claim_ids:
                raise ValueError(
                    f"score reference names claim {score.claim_id!r}, which the "
                    f"contribution does not carry"
                )
        for raw_score in self.raw_score_refs:
            if raw_score.claim_id not in claim_ids:
                raise ValueError(
                    f"raw score reference names claim {raw_score.claim_id!r}, which the "
                    "contribution does not carry"
                )
        require_canonical(
            "score_refs", self.score_refs, lambda ref: (ref.claim_id, *producer_key(ref.scorer))
        )
        require_canonical(
            "raw_score_refs",
            self.raw_score_refs,
            lambda ref: (ref.claim_id, ref.semantic_scoring_run_id, ref.score_id),
        )
        require_canonical(
            "visual_feature_refs",
            self.visual_feature_refs,
            lambda ref: (ref.visual_artifact_id, ref.source_observation_id, ref.feature_id),
        )
        for feature in self.visual_feature_refs:
            if feature.source_observation_id != self.physical_observation_id:
                raise ValueError("feature source observation must match contribution")
            if feature.scope is FeatureScope.REGION and feature.region_id != self.region_id:
                raise ValueError(
                    f"region-scoped feature {feature.feature_id!r} belongs to region "
                    f"{feature.region_id!r}, not the contribution region {self.region_id!r}"
                )
        quality = self.observation_quality
        if quality is not None and quality.spatial_observation_id != self.spatial_observation_id:
            raise ValueError(
                f"observation_quality describes {quality.spatial_observation_id!r}, not the "
                f"contribution observation {self.spatial_observation_id!r}"
            )


@dataclass(frozen=True, kw_only=True)
class PhysicalObservationGroup:
    """Everything inferred from one physical frame, kept apart from the frame itself.

    Attributes:
        physical_observation_id: The physical frame.
        acquisition_timestamp: When the frame was acquired.
        spatial_observation_ids: Spatial observations of this frame, sorted and unique.
        semantic_interpretation_artifact_ids: The correlated inference results over this frame.
        region_artifact_ids: The runs that produced those results.
    """

    physical_observation_id: SourceObservationId
    acquisition_timestamp: SourceTimestamp
    spatial_observation_ids: tuple[SpatialObservationId, ...]
    semantic_interpretation_artifact_ids: tuple[str, ...]
    region_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the identity and that each collection is present, sorted and unique.

        Raises:
            ValueError: If the identity or a collection is empty or not sorted and unique.
        """
        require_present(self, "physical_observation_id")
        for name in ("spatial_observation_ids", "region_artifact_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if not items:
                raise ValueError(f"{name} must not be empty")
            require_canonical(name, items, lambda item: (item,))
        require_canonical(
            "semantic_interpretation_artifact_ids",
            self.semantic_interpretation_artifact_ids,
            lambda item: (item,),
        )

    @property
    def inference_result_count(self) -> int:
        """Number of inference results over this one physical observation."""
        return len(self.semantic_interpretation_artifact_ids) or len(self.region_artifact_ids)


@dataclass(frozen=True, kw_only=True)
class SupportSignal:
    """A typed score attached to one claim, with the model whose semantics it has.

    ``value`` is ``None`` when the producer emitted no score. That is *unscored*, not
    zero: a low score and a missing score are different evidence.

    Attributes:
        kind: Which quantity this is.
        producer: The interpreter or scorer that produced the value.
        value: A score in ``[0, 1]`` under the producer's own semantics, or ``None``.
    """

    kind: SupportSignalKind
    producer: BackendProvenance
    value: float | None

    def __post_init__(self) -> None:
        """Validate the producer and the value.

        Raises:
            ValueError: If the producer is not identified or the value is neither ``None``
                nor a finite number in ``[0, 1]``.
        """
        require_identified("producer", self.producer)
        if self.value is not None and not (math.isfinite(self.value) and 0.0 <= self.value <= 1.0):
            raise ValueError(f"signal value must be None or within [0, 1], got {self.value!r}")


@dataclass(frozen=True, kw_only=True)
class EvidenceReference:
    """Reference to a contribution, and to one of its claims when the evidence is a claim.

    Attributes:
        contribution_id: The contribution.
        claim_id: The claim, or ``None`` when the evidence is the contribution itself, for
            example a view that produced no claim.
    """

    contribution_id: EvidenceContributionId
    claim_id: ClaimId | None = None

    def __post_init__(self) -> None:
        """Require the contribution and, when present, the claim.

        Raises:
            ValueError: If the contribution is empty or a given claim is empty.
        """
        require_present(self, "contribution_id")
        if self.claim_id is not None:
            require_present(self, "claim_id")


@dataclass(frozen=True, kw_only=True)
class PointRepresentationRef:
    """Reference to a 3D representation anchored to the support's geometry.

    Static structure of the map, not a view: it is listed once on the
    :class:`FusedEvidence`, so it is never counted once per camera observation.

    Attributes:
        representation_id: The representation, local to its run.
        run_id: The Point Representation run that owns it.
        representation_space_id: Fingerprint of its representation space; representations
            are comparable only within one space.
        geometry_reference: The geometry element the representation is centered on.
    """

    representation_id: PointRepresentationId
    run_id: PointRepresentationRunId
    representation_space_id: str
    geometry_reference: GeometryReference

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        require_present(self, "representation_id", "run_id", "representation_space_id")


@dataclass(frozen=True, kw_only=True)
class ChannelProvenance:
    """An active evidence channel and the identities of what fed it.

    A channel can be active and empty: a declared channel that found no data for a support
    has no identities. Identities are strings whose form depends on the channel: an
    interpreter or scorer as ``backend_id/model/version``, an embedding space, a quality
    definitions version, a representation space, or the map and support policy.

    Attributes:
        channel: The active channel.
        identities: What fed it, sorted and unique.
    """

    channel: EvidenceChannel
    identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the identities.

        Raises:
            ValueError: If an identity is empty or the identities are not sorted and unique.
        """
        if any(not identity.strip() for identity in self.identities):
            raise ValueError("a channel identity must not be empty")
        require_canonical("identities", self.identities, lambda identity: (identity,))
