"""Public contract for the semantic fusion capability.

Semantic Fusion accumulates evidence from several views over a shared spatial
support and keeps it as :class:`FusedEvidence`: the competing hypotheses, the
exact claims and typed signals behind each, the conflicts between physical
observations and what is simply unknown. It creates no entity identity, elects
no winner and never merges heterogeneous scores into one number. See
``src/contextmap/semantic_fusion/docs/README.md`` for the full capability
documentation.

The surface is grouped by what it is: the contracts of each semantic level
(``supports``, ``evidence``, ``fused_evidence``, ``quality_weighting``), the
policies that produce them (``grouping``, ``support_construction``,
``accumulation``, ``quality_aware``) and the persisted run artifact. The
validation vocabulary, the overlap engine and the JSON codecs stay internal.
"""

from contextmap.semantic_fusion.accumulation import (
    BASELINE_ACCUMULATION_POLICY_ID,
    BaselineAccumulationPolicy,
    accumulate_baseline_evidence,
)
from contextmap.semantic_fusion.evidence import (
    ChannelProvenance,
    EvidenceChannel,
    EvidenceContribution,
    EvidenceReference,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    RawSemanticScoreReference,
    ScoreReference,
    SemanticClaimRef,
    SupportSignal,
    SupportSignalKind,
    VisualFeatureRef,
)
from contextmap.semantic_fusion.fused_evidence import (
    INTERACTION_FACTOR_AGGREGATION_POLICY_ID,
    EvidenceStance,
    FusedAttribute,
    FusedEvidence,
    FusedEvidenceProvenance,
    FusedHypothesis,
    FusedInteractionFactor,
    FusionOutcome,
    HypothesisEvidence,
    UncertaintyKind,
    UncertaintyRecord,
)
from contextmap.semantic_fusion.grouping import (
    PHYSICAL_OBSERVATION_GROUPING_POLICY_ID,
    PhysicalObservationGrouping,
    group_by_physical_observation,
)
from contextmap.semantic_fusion.hypotheses import label_key
from contextmap.semantic_fusion.identity import (
    EvidenceContributionId,
    FusedEvidenceId,
    FusedHypothesisId,
    FusionSupportId,
    evidence_contribution_id_for,
    fused_evidence_id_for,
)
from contextmap.semantic_fusion.quality_aware import (
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    accumulate_quality_aware_evidence,
)
from contextmap.semantic_fusion.quality_weighting import (
    ComponentFactor,
    ComponentTreatment,
    ContributionWeight,
    HypothesisSupport,
    ObservationFactor,
    QualityWeighting,
)
from contextmap.semantic_fusion.run_artifact import (
    FusionRunArtifactError,
    FusionRunLineage,
    IncompleteFusionRunArtifactError,
    SemanticFusionDebugLevel,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
)
from contextmap.semantic_fusion.support_construction import (
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    FusionSupportBuild,
    GeometryOverlapSupportPolicy,
    build_fusion_supports,
)
from contextmap.semantic_fusion.supports import (
    ExcludedObservation,
    FusionSupport,
    FusionSupportProvenance,
)

__all__ = [
    "BASELINE_ACCUMULATION_POLICY_ID",
    "GEOMETRY_OVERLAP_SUPPORT_POLICY_ID",
    "INTERACTION_FACTOR_AGGREGATION_POLICY_ID",
    "PHYSICAL_OBSERVATION_GROUPING_POLICY_ID",
    "QUALITY_AWARE_ACCUMULATION_POLICY_ID",
    "BaselineAccumulationPolicy",
    "ChannelProvenance",
    "ComponentFactor",
    "ComponentTreatment",
    "ContributionWeight",
    "EvidenceChannel",
    "EvidenceContribution",
    "EvidenceContributionId",
    "EvidenceReference",
    "EvidenceStance",
    "ExcludedObservation",
    "FusedAttribute",
    "FusedEvidence",
    "FusedEvidenceId",
    "FusedEvidenceProvenance",
    "FusedHypothesis",
    "FusedHypothesisId",
    "FusedInteractionFactor",
    "FusionOutcome",
    "FusionRunArtifactError",
    "FusionRunLineage",
    "FusionSupport",
    "FusionSupportBuild",
    "FusionSupportId",
    "FusionSupportProvenance",
    "GeometryOverlapSupportPolicy",
    "HypothesisEvidence",
    "HypothesisSupport",
    "IncompleteFusionRunArtifactError",
    "ObservationFactor",
    "ObservationQualityRef",
    "PhysicalObservationGroup",
    "PhysicalObservationGrouping",
    "PointRepresentationRef",
    "QualityAwareAccumulationPolicy",
    "QualityInput",
    "QualityRamp",
    "QualityWeighting",
    "RawSemanticScoreReference",
    "ScoreReference",
    "SemanticClaimRef",
    "SemanticFusionDebugLevel",
    "SemanticFusionRunId",
    "SemanticFusionRunManifest",
    "SemanticFusionRunReader",
    "SemanticFusionRunWriter",
    "SupportSignal",
    "SupportSignalKind",
    "UncertaintyKind",
    "UncertaintyRecord",
    "VisualFeatureRef",
    "accumulate_baseline_evidence",
    "accumulate_quality_aware_evidence",
    "build_fusion_supports",
    "evidence_contribution_id_for",
    "fused_evidence_id_for",
    "group_by_physical_observation",
    "label_key",
]
