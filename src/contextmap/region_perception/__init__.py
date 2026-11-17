"""Region perception: discovery, grounding, refinement, and normalization of candidate regions.

Regions are 2D spatial candidates in images. A region is discovered by a backend,
optionally normalized (merged, filtered), optionally refined, and optionally grounded
to semantic text queries.

Region perception owns all regional evidence production. Downstream capabilities
consume regions through the public RegionPerceptionRun artifact and Region2D contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextmap.region_perception.artifacts.frame import RegionPerceptionFrame
from contextmap.region_perception.artifacts.region_serialization import (
    decode_region,
    encode_region,
)
from contextmap.region_perception.artifacts.run_manifest import (
    REGION_PERCEPTION_SCHEMA_VERSION,
    RegionPerceptionRunError,
    RegionPerceptionRunId,
    RegionPerceptionRunManifest,
)
from contextmap.region_perception.artifacts.run_reader import RegionPerceptionRunReader
from contextmap.region_perception.artifacts.run_writer import RegionPerceptionRunWriter
from contextmap.region_perception.evidence_identity import (
    NamespacedRegions,
    grounding_point_id_for,
    grounding_region_id_for,
    namespace_regions,
    refined_region_id_for,
    region_id_for,
)
from contextmap.region_perception.region_assembly import (
    REGION_SET_ASSEMBLY_POLICY,
    assemble_regions,
)
from contextmap.region_perception.region_evidence import (
    Region2D,
    RegionId,
)

# NOTE: RegionCandidate vem de region_geometry, não de region_evidence. Há duas definições
# com esse nome, e é a de region_geometry que a pipeline desta capability consome
# (discovery/region_normalization.py). Exportar a outra entregava ao consumidor um
# contrato sem source_observation_id, incompatível com normalize_regions.
from contextmap.region_perception.region_geometry import (
    BackendScore,
    BoundingBox,
    BoundingBox2D,
    CoordinateConvention,
    InlineMask,
    NativeRegionText,
    RegionCandidate,
    RegionProvenance,
    RejectedRegionCandidate,
    RejectionReason,
    mask_bounding_box,
)

if TYPE_CHECKING:
    # Em execução estes nomes chegam por ``__getattr__``, para não fechar o ciclo de
    # import entre discovery/grounding/refinement e esta API pública. O bloco abaixo
    # existe só para o type checker: sem ele, cada nome desta API pública chega ao
    # consumidor como variável ``Any``, e não como o contrato que ele declara.
    from contextmap.region_perception.artifacts.mask_store import (
        MaskStoreError,
        RegionMaskResolver,
        region_mask_pixels,
    )
    from contextmap.region_perception.discovery.region_discovery import (
        AuditedRegions,
        BackendDiagnostics,
        BorderPolicy,
        DiscoveryInput,
        DiscoveryOutput,
        DiscoveryPass,
        DiscoveryPassConfig,
        DiscoveryRunResult,
        PassKind,
        RegionDiscoveryAudit,
        RegionDiscoveryIdentity,
        TilingConfig,
        build_discovery_passes,
        run_discovery_passes,
    )
    from contextmap.region_perception.discovery.region_normalization import (
        MergeDecision,
        MergeKind,
        MergeRepresentativePolicy,
        NormalizationConfig,
        NormalizationResult,
        normalize_regions,
    )
    from contextmap.region_perception.grounding.region_grounding import (
        GroundingDiagnostics,
        GroundingGeometry,
        GroundingOutput,
        GroundingPoint,
        GroundingQuery,
        GroundingQueryPolicy,
        GroundingQuerySet,
        GroundingRejectionReason,
        GroundingRequestError,
        GroundingRequestId,
        GroundingTask,
        RegionGroundingCapabilities,
        RegionGroundingExecution,
        RegionGroundingRequest,
        RejectedGroundingOutput,
        decode_prepared_image_reference,
        decode_region_grounding_execution,
        decode_region_grounding_request,
        encode_prepared_image_reference,
        encode_region_grounding_execution,
        encode_region_grounding_request,
        validate_grounding_query,
    )
    from contextmap.region_perception.ports import (
        AuditedRegionDiscovery,
        RegionDiscovery,
        RegionGrounding,
        RegionRefinement,
    )
    from contextmap.region_perception.refinement.region_refinement import (
        REFINEMENT_ACCEPTANCE_POLICY,
        RefinementDiagnostics,
        RefinementOutcome,
        RefinementPrompt,
        RefinementRejectionReason,
        RefinementRequestError,
        RefinementRequestId,
        RegionRefinementCapabilities,
        RegionRefinementExecution,
        RegionRefinementRequest,
        decode_region_refinement_execution,
        encode_region_refinement_execution,
        refinement_outcome,
        refinement_prompts_from,
        validate_refinement_request,
    )
    from contextmap.region_perception.region_eligibility import (
        IneligibleRegion,
        IneligibleRegionReason,
        decode_ineligible_region,
        encode_ineligible_region,
    )
    from contextmap.region_perception.region_semantic_hint import (
        HintContribution,
        RegionSemanticHint,
    )


def __getattr__(name: str) -> object:
    """Lazy imports for discovery, grounding, refinement to avoid circular dependencies."""
    import importlib

    _mappings = {
        "encode_ineligible_region": (
            "contextmap.region_perception.region_eligibility",
            "encode_ineligible_region",
        ),
        "decode_ineligible_region": (
            "contextmap.region_perception.region_eligibility",
            "decode_ineligible_region",
        ),
        "MaskStoreError": (
            "contextmap.region_perception.artifacts.mask_store",
            "MaskStoreError",
        ),
        "RegionMaskResolver": (
            "contextmap.region_perception.artifacts.mask_store",
            "RegionMaskResolver",
        ),
        "REFINEMENT_ACCEPTANCE_POLICY": (
            "contextmap.region_perception.refinement.region_refinement",
            "REFINEMENT_ACCEPTANCE_POLICY",
        ),
        "decode_prepared_image_reference": (
            "contextmap.region_perception.grounding.region_grounding",
            "decode_prepared_image_reference",
        ),
        "encode_prepared_image_reference": (
            "contextmap.region_perception.grounding.region_grounding",
            "encode_prepared_image_reference",
        ),
        # Discovery
        "AuditedRegions": (
            "contextmap.region_perception.discovery.region_discovery",
            "AuditedRegions",
        ),
        "BackendDiagnostics": (
            "contextmap.region_perception.discovery.region_discovery",
            "BackendDiagnostics",
        ),
        "BorderPolicy": ("contextmap.region_perception.discovery.region_discovery", "BorderPolicy"),
        "DiscoveryInput": (
            "contextmap.region_perception.discovery.region_discovery",
            "DiscoveryInput",
        ),
        "DiscoveryOutput": (
            "contextmap.region_perception.discovery.region_discovery",
            "DiscoveryOutput",
        ),
        "DiscoveryPass": (
            "contextmap.region_perception.discovery.region_discovery",
            "DiscoveryPass",
        ),
        "DiscoveryPassConfig": (
            "contextmap.region_perception.discovery.region_discovery",
            "DiscoveryPassConfig",
        ),
        "DiscoveryRunResult": (
            "contextmap.region_perception.discovery.region_discovery",
            "DiscoveryRunResult",
        ),
        "PassKind": ("contextmap.region_perception.discovery.region_discovery", "PassKind"),
        "RegionDiscoveryAudit": (
            "contextmap.region_perception.discovery.region_discovery",
            "RegionDiscoveryAudit",
        ),
        "RegionDiscoveryIdentity": (
            "contextmap.region_perception.discovery.region_discovery",
            "RegionDiscoveryIdentity",
        ),
        "TilingConfig": ("contextmap.region_perception.discovery.region_discovery", "TilingConfig"),
        "build_discovery_passes": (
            "contextmap.region_perception.discovery.region_discovery",
            "build_discovery_passes",
        ),
        "run_discovery_passes": (
            "contextmap.region_perception.discovery.region_discovery",
            "run_discovery_passes",
        ),
        "RegionSemanticHint": (
            "contextmap.region_perception.region_semantic_hint",
            "RegionSemanticHint",
        ),
        "HintContribution": (
            "contextmap.region_perception.region_semantic_hint",
            "HintContribution",
        ),
        # Normalization
        "MergeDecision": (
            "contextmap.region_perception.discovery.region_normalization",
            "MergeDecision",
        ),
        "MergeKind": ("contextmap.region_perception.discovery.region_normalization", "MergeKind"),
        "MergeRepresentativePolicy": (
            "contextmap.region_perception.discovery.region_normalization",
            "MergeRepresentativePolicy",
        ),
        "NormalizationConfig": (
            "contextmap.region_perception.discovery.region_normalization",
            "NormalizationConfig",
        ),
        "NormalizationResult": (
            "contextmap.region_perception.discovery.region_normalization",
            "NormalizationResult",
        ),
        "normalize_regions": (
            "contextmap.region_perception.discovery.region_normalization",
            "normalize_regions",
        ),
        # Ports
        "AuditedRegionDiscovery": ("contextmap.region_perception.ports", "AuditedRegionDiscovery"),
        "RegionDiscovery": ("contextmap.region_perception.ports", "RegionDiscovery"),
        "RegionGrounding": ("contextmap.region_perception.ports", "RegionGrounding"),
        "RegionRefinement": ("contextmap.region_perception.ports", "RegionRefinement"),
        # Grounding
        "GroundingDiagnostics": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingDiagnostics",
        ),
        "GroundingGeometry": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingGeometry",
        ),
        "GroundingOutput": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingOutput",
        ),
        "GroundingPoint": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingPoint",
        ),
        "GroundingQuery": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingQuery",
        ),
        "GroundingQueryPolicy": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingQueryPolicy",
        ),
        "GroundingQuerySet": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingQuerySet",
        ),
        "GroundingRejectionReason": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingRejectionReason",
        ),
        "GroundingRequestError": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingRequestError",
        ),
        "GroundingRequestId": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingRequestId",
        ),
        "GroundingTask": (
            "contextmap.region_perception.grounding.region_grounding",
            "GroundingTask",
        ),
        "RegionGroundingCapabilities": (
            "contextmap.region_perception.grounding.region_grounding",
            "RegionGroundingCapabilities",
        ),
        "RegionGroundingExecution": (
            "contextmap.region_perception.grounding.region_grounding",
            "RegionGroundingExecution",
        ),
        "RegionGroundingRequest": (
            "contextmap.region_perception.grounding.region_grounding",
            "RegionGroundingRequest",
        ),
        "RejectedGroundingOutput": (
            "contextmap.region_perception.grounding.region_grounding",
            "RejectedGroundingOutput",
        ),
        "decode_region_grounding_execution": (
            "contextmap.region_perception.grounding.region_grounding",
            "decode_region_grounding_execution",
        ),
        "decode_region_grounding_request": (
            "contextmap.region_perception.grounding.region_grounding",
            "decode_region_grounding_request",
        ),
        "encode_region_grounding_execution": (
            "contextmap.region_perception.grounding.region_grounding",
            "encode_region_grounding_execution",
        ),
        "encode_region_grounding_request": (
            "contextmap.region_perception.grounding.region_grounding",
            "encode_region_grounding_request",
        ),
        "validate_grounding_query": (
            "contextmap.region_perception.grounding.region_grounding",
            "validate_grounding_query",
        ),
        "RefinementDiagnostics": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementDiagnostics",
        ),
        "RefinementOutcome": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementOutcome",
        ),
        "RefinementPrompt": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementPrompt",
        ),
        "RefinementRejectionReason": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementRejectionReason",
        ),
        "RefinementRequestError": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementRequestError",
        ),
        "RefinementRequestId": (
            "contextmap.region_perception.refinement.region_refinement",
            "RefinementRequestId",
        ),
        "RegionRefinementCapabilities": (
            "contextmap.region_perception.refinement.region_refinement",
            "RegionRefinementCapabilities",
        ),
        "RegionRefinementExecution": (
            "contextmap.region_perception.refinement.region_refinement",
            "RegionRefinementExecution",
        ),
        "RegionRefinementRequest": (
            "contextmap.region_perception.refinement.region_refinement",
            "RegionRefinementRequest",
        ),
        "decode_region_refinement_execution": (
            "contextmap.region_perception.refinement.region_refinement",
            "decode_region_refinement_execution",
        ),
        "encode_region_refinement_execution": (
            "contextmap.region_perception.refinement.region_refinement",
            "encode_region_refinement_execution",
        ),
        "refinement_outcome": (
            "contextmap.region_perception.refinement.region_refinement",
            "refinement_outcome",
        ),
        "refinement_prompts_from": (
            "contextmap.region_perception.refinement.region_refinement",
            "refinement_prompts_from",
        ),
        "validate_refinement_request": (
            "contextmap.region_perception.refinement.region_refinement",
            "validate_refinement_request",
        ),
        "region_mask_pixels": (
            "contextmap.region_perception.artifacts.mask_store",
            "region_mask_pixels",
        ),
        "IneligibleRegion": (
            "contextmap.region_perception.region_eligibility",
            "IneligibleRegion",
        ),
        "IneligibleRegionReason": (
            "contextmap.region_perception.region_eligibility",
            "IneligibleRegionReason",
        ),
    }

    if name in _mappings:
        module_name, attr_name = _mappings[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# All imports are lazy-loaded via __getattr__ above to avoid circular dependencies

__all__ = [
    "REFINEMENT_ACCEPTANCE_POLICY",
    "REGION_PERCEPTION_SCHEMA_VERSION",
    "REGION_SET_ASSEMBLY_POLICY",
    "AuditedRegionDiscovery",
    "AuditedRegions",
    "BackendDiagnostics",
    "BackendScore",
    "BorderPolicy",
    "BoundingBox",
    "BoundingBox2D",
    "CoordinateConvention",
    "DiscoveryInput",
    "DiscoveryOutput",
    "DiscoveryPass",
    "DiscoveryPassConfig",
    "DiscoveryRunResult",
    "GroundingDiagnostics",
    "GroundingGeometry",
    "GroundingOutput",
    "GroundingPoint",
    "GroundingQuery",
    "GroundingQueryPolicy",
    "GroundingQuerySet",
    "GroundingRejectionReason",
    "GroundingRequestError",
    "GroundingRequestId",
    "GroundingTask",
    "HintContribution",
    "IneligibleRegion",
    "IneligibleRegionReason",
    "InlineMask",
    "MaskStoreError",
    "MergeDecision",
    "MergeKind",
    "MergeRepresentativePolicy",
    "NamespacedRegions",
    "NativeRegionText",
    "NormalizationConfig",
    "NormalizationResult",
    "PassKind",
    "RefinementDiagnostics",
    "RefinementOutcome",
    "RefinementPrompt",
    "RefinementRejectionReason",
    "RefinementRequestError",
    "RefinementRequestId",
    "Region2D",
    "RegionCandidate",
    "RegionDiscovery",
    "RegionDiscoveryAudit",
    "RegionDiscoveryIdentity",
    "RegionGrounding",
    "RegionGroundingCapabilities",
    "RegionGroundingExecution",
    "RegionGroundingRequest",
    "RegionId",
    "RegionMaskResolver",
    "RegionPerceptionFrame",
    "RegionPerceptionRunError",
    "RegionPerceptionRunId",
    "RegionPerceptionRunManifest",
    "RegionPerceptionRunReader",
    "RegionPerceptionRunWriter",
    "RegionProvenance",
    "RegionRefinement",
    "RegionRefinementCapabilities",
    "RegionRefinementExecution",
    "RegionRefinementRequest",
    "RegionSemanticHint",
    "RejectedGroundingOutput",
    "RejectedRegionCandidate",
    "RejectionReason",
    "TilingConfig",
    "assemble_regions",
    "build_discovery_passes",
    "decode_ineligible_region",
    "decode_prepared_image_reference",
    "decode_region",
    "decode_region_grounding_execution",
    "decode_region_grounding_request",
    "decode_region_refinement_execution",
    "encode_ineligible_region",
    "encode_prepared_image_reference",
    "encode_region",
    "encode_region_grounding_execution",
    "encode_region_grounding_request",
    "encode_region_refinement_execution",
    "grounding_point_id_for",
    "grounding_region_id_for",
    "mask_bounding_box",
    "namespace_regions",
    "normalize_regions",
    "refined_region_id_for",
    "refinement_outcome",
    "refinement_prompts_from",
    "region_id_for",
    "region_mask_pixels",
    "run_discovery_passes",
    "validate_grounding_query",
    "validate_refinement_request",
]
