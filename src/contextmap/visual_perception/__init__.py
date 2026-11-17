"""Public visual evidence API and derived read-only perception views.

Region contracts live in ``contextmap.region_perception``; semantic inference lives in
``contextmap.semantic_interpretation``. Concrete backends remain internal to runtime.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextmap.shared import (
        ArtifactReference as ArtifactReference,
    )
    from contextmap.shared import (
        BackendProvenance as BackendProvenance,
    )
    from contextmap.visual_perception.evidence_identity import feature_id_for as feature_id_for
    from contextmap.visual_perception.execution.pipeline_presets import (
        CANONICAL_PRESET_V1 as CANONICAL_PRESET_V1,
    )
    from contextmap.visual_perception.execution.pipeline_presets import (
        with_dense_region_pooling as with_dense_region_pooling,
    )
    from contextmap.visual_perception.execution.pipeline_presets import (
        without_scene_interpretation as without_scene_interpretation,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        KNOWN_CAPABILITIES as KNOWN_CAPABILITIES,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        PipelineConfigError as PipelineConfigError,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        PipelinePreset as PipelinePreset,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        ResolvedPipeline as ResolvedPipeline,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        StageBackendFactory as StageBackendFactory,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        StageSpec as StageSpec,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        decode_pipeline_preset as decode_pipeline_preset,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        encode_pipeline_preset as encode_pipeline_preset,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        resolve_pipeline as resolve_pipeline,
    )
    from contextmap.visual_perception.execution.pipeline_resolution import (
        validate_pipeline_preset as validate_pipeline_preset,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        StageDefinition as StageDefinition,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        StageGraphError as StageGraphError,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        StageOutcome as StageOutcome,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        StageRunner as StageRunner,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        StageStatus as StageStatus,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        execute_stage_graph as execute_stage_graph,
    )
    from contextmap.visual_perception.execution.stage_execution import (
        execute_stage_graphs_stage_major as execute_stage_graphs_stage_major,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        FEATURE_METRICS_PATH as FEATURE_METRICS_PATH,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        DenseFeatureDiagnostic as DenseFeatureDiagnostic,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        FeatureDebugLevel as FeatureDebugLevel,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        FeatureDiagnosticPreview as FeatureDiagnosticPreview,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        FeatureEventStatus as FeatureEventStatus,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        FeatureExtractionDiagnostic as FeatureExtractionDiagnostic,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        RegionFeatureDiagnostic as RegionFeatureDiagnostic,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        dense_extraction_diagnostic as dense_extraction_diagnostic,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        dense_feature_map_from_record as dense_feature_map_from_record,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        dense_feature_map_of as dense_feature_map_of,
    )
    from contextmap.visual_perception.feature_diagnostics import (
        write_feature_diagnostics as write_feature_diagnostics,
    )
    from contextmap.visual_perception.image_preparation import (
        CropOperation as CropOperation,
    )
    from contextmap.visual_perception.image_preparation import (
        NormalizeOperation as NormalizeOperation,
    )
    from contextmap.visual_perception.image_preparation import (
        RectifyOperation as RectifyOperation,
    )
    from contextmap.visual_perception.image_preparation import (
        ResizeOperation as ResizeOperation,
    )
    from contextmap.visual_perception.image_preparation import (
        SourceImage as SourceImage,
    )
    from contextmap.visual_perception.image_preparation import (
        prepare_image as prepare_image,
    )
    from contextmap.visual_perception.models import (
        ExclusionRegion as ExclusionRegion,
    )
    from contextmap.visual_perception.models import (
        PerceptionResult as PerceptionResult,
    )
    from contextmap.visual_perception.models import (
        PreparedImage as PreparedImage,
    )
    from contextmap.visual_perception.models import (
        SemanticSupport as SemanticSupport,
    )
    from contextmap.visual_perception.models import (
        TransformationRecord as TransformationRecord,
    )
    from contextmap.visual_perception.models import (
        ValidRegion as ValidRegion,
    )
    from contextmap.visual_perception.perception_evidence_set import (
        EvidenceSetError as EvidenceSetError,
    )
    from contextmap.visual_perception.perception_evidence_set import (
        ObservationEvidence as ObservationEvidence,
    )
    from contextmap.visual_perception.perception_evidence_set import (
        PerceptionEvidenceSelection as PerceptionEvidenceSelection,
    )
    from contextmap.visual_perception.perception_evidence_set import (
        PerceptionEvidenceSet as PerceptionEvidenceSet,
    )
    from contextmap.visual_perception.ports import (
        FeatureExtractor as FeatureExtractor,
    )
    from contextmap.visual_perception.ports import (
        FeatureResolutionEnhancement as FeatureResolutionEnhancement,
    )
    from contextmap.visual_perception.ports import (
        ReleasableBackend as ReleasableBackend,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VISUAL_EVIDENCE_SCHEMA_VERSION as VISUAL_EVIDENCE_SCHEMA_VERSION,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        RecordedDenseFeatureMap as RecordedDenseFeatureMap,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceResult as VisualEvidenceResult,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceRunError as VisualEvidenceRunError,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceRunId as VisualEvidenceRunId,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceRunManifest as VisualEvidenceRunManifest,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceRunReader as VisualEvidenceRunReader,
    )
    from contextmap.visual_perception.visual_evidence_artifact import (
        VisualEvidenceRunWriter as VisualEvidenceRunWriter,
    )
    from contextmap.visual_perception.visual_features import (
        FEATURE_INDEX_FILENAME as FEATURE_INDEX_FILENAME,
    )
    from contextmap.visual_perception.visual_features import (
        FEATURE_INDEX_SCHEMA_VERSION as FEATURE_INDEX_SCHEMA_VERSION,
    )
    from contextmap.visual_perception.visual_features import (
        POOLING_BACKEND_ID as POOLING_BACKEND_ID,
    )
    from contextmap.visual_perception.visual_features import (
        POOLING_BACKEND_VERSION as POOLING_BACKEND_VERSION,
    )
    from contextmap.visual_perception.visual_features import (
        POOLING_POLICY as POOLING_POLICY,
    )
    from contextmap.visual_perception.visual_features import (
        DenseFeatureMap as DenseFeatureMap,
    )
    from contextmap.visual_perception.visual_features import (
        DenseFeatureSampling as DenseFeatureSampling,
    )
    from contextmap.visual_perception.visual_features import (
        EmbeddingSpace as EmbeddingSpace,
    )
    from contextmap.visual_perception.visual_features import (
        EmbeddingSpaceMismatchError as EmbeddingSpaceMismatchError,
    )
    from contextmap.visual_perception.visual_features import (
        EmptyRegionSupportError as EmptyRegionSupportError,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureId as FeatureId,
    )
    from contextmap.visual_perception.visual_features import (
        FeaturePayloadEntry as FeaturePayloadEntry,
    )
    from contextmap.visual_perception.visual_features import (
        FeaturePayloadIntegrityError as FeaturePayloadIntegrityError,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureResolutionEnhancementError as FeatureResolutionEnhancementError,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureResolutionEnhancementProvenance as FeatureResolutionEnhancementProvenance,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureScope as FeatureScope,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureStoreError as FeatureStoreError,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureStoreReader as FeatureStoreReader,
    )
    from contextmap.visual_perception.visual_features import (
        FeatureStoreWriter as FeatureStoreWriter,
    )
    from contextmap.visual_perception.visual_features import (
        RegionAssociationError as RegionAssociationError,
    )
    from contextmap.visual_perception.visual_features import (
        RegionPoolingDiagnostics as RegionPoolingDiagnostics,
    )
    from contextmap.visual_perception.visual_features import (
        RegionPoolingProvenance as RegionPoolingProvenance,
    )
    from contextmap.visual_perception.visual_features import (
        RegionPoolingResult as RegionPoolingResult,
    )
    from contextmap.visual_perception.visual_features import (
        VisualFeature as VisualFeature,
    )
    from contextmap.visual_perception.visual_features import (
        alphaclip_embedding_space as alphaclip_embedding_space,
    )
    from contextmap.visual_perception.visual_features import (
        box_mask_shape as box_mask_shape,
    )
    from contextmap.visual_perception.visual_features import (
        clip_embedding_space as clip_embedding_space,
    )
    from contextmap.visual_perception.visual_features import (
        decode_embedding_space as decode_embedding_space,
    )
    from contextmap.visual_perception.visual_features import (
        decode_feature_payload_entry as decode_feature_payload_entry,
    )
    from contextmap.visual_perception.visual_features import (
        embedding_space_fingerprint as embedding_space_fingerprint,
    )
    from contextmap.visual_perception.visual_features import (
        encode_embedding_space as encode_embedding_space,
    )
    from contextmap.visual_perception.visual_features import (
        encode_feature_payload_entry as encode_feature_payload_entry,
    )
    from contextmap.visual_perception.visual_features import (
        ensure_compatible_embedding_spaces as ensure_compatible_embedding_spaces,
    )
    from contextmap.visual_perception.visual_features import (
        ensure_compatible_features as ensure_compatible_features,
    )
    from contextmap.visual_perception.visual_features import (
        map_box_to_grid_cells as map_box_to_grid_cells,
    )
    from contextmap.visual_perception.visual_features import (
        pool_region_feature as pool_region_feature,
    )
    from contextmap.visual_perception.visual_features import (
        pooling_backend_provenance as pooling_backend_provenance,
    )
    from contextmap.visual_perception.visual_features import (
        write_feature_index as write_feature_index,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactReference": ("contextmap.shared", "ArtifactReference"),
    "BackendProvenance": ("contextmap.shared", "BackendProvenance"),
    "CANONICAL_PRESET_V1": (
        "contextmap.visual_perception.execution.pipeline_presets",
        "CANONICAL_PRESET_V1",
    ),
    "CropOperation": ("contextmap.visual_perception.image_preparation", "CropOperation"),
    "DenseFeatureDiagnostic": (
        "contextmap.visual_perception.feature_diagnostics",
        "DenseFeatureDiagnostic",
    ),
    "DenseFeatureMap": ("contextmap.visual_perception.visual_features", "DenseFeatureMap"),
    "DenseFeatureSampling": (
        "contextmap.visual_perception.visual_features",
        "DenseFeatureSampling",
    ),
    "EmbeddingSpace": ("contextmap.visual_perception.visual_features", "EmbeddingSpace"),
    "EmbeddingSpaceMismatchError": (
        "contextmap.visual_perception.visual_features",
        "EmbeddingSpaceMismatchError",
    ),
    "EmptyRegionSupportError": (
        "contextmap.visual_perception.visual_features",
        "EmptyRegionSupportError",
    ),
    "EvidenceSetError": (
        "contextmap.visual_perception.perception_evidence_set",
        "EvidenceSetError",
    ),
    "ExclusionRegion": ("contextmap.visual_perception.models", "ExclusionRegion"),
    "FEATURE_INDEX_FILENAME": (
        "contextmap.visual_perception.visual_features",
        "FEATURE_INDEX_FILENAME",
    ),
    "FEATURE_INDEX_SCHEMA_VERSION": (
        "contextmap.visual_perception.visual_features",
        "FEATURE_INDEX_SCHEMA_VERSION",
    ),
    "FEATURE_METRICS_PATH": (
        "contextmap.visual_perception.feature_diagnostics",
        "FEATURE_METRICS_PATH",
    ),
    "FeatureDebugLevel": ("contextmap.visual_perception.feature_diagnostics", "FeatureDebugLevel"),
    "FeatureDiagnosticPreview": (
        "contextmap.visual_perception.feature_diagnostics",
        "FeatureDiagnosticPreview",
    ),
    "FeatureEventStatus": (
        "contextmap.visual_perception.feature_diagnostics",
        "FeatureEventStatus",
    ),
    "FeatureExtractionDiagnostic": (
        "contextmap.visual_perception.feature_diagnostics",
        "FeatureExtractionDiagnostic",
    ),
    "FeatureExtractor": ("contextmap.visual_perception.ports", "FeatureExtractor"),
    "FeaturePayloadEntry": ("contextmap.visual_perception.visual_features", "FeaturePayloadEntry"),
    "FeaturePayloadIntegrityError": (
        "contextmap.visual_perception.visual_features",
        "FeaturePayloadIntegrityError",
    ),
    "FeatureResolutionEnhancement": (
        "contextmap.visual_perception.ports",
        "FeatureResolutionEnhancement",
    ),
    "FeatureResolutionEnhancementError": (
        "contextmap.visual_perception.visual_features",
        "FeatureResolutionEnhancementError",
    ),
    "FeatureResolutionEnhancementProvenance": (
        "contextmap.visual_perception.visual_features",
        "FeatureResolutionEnhancementProvenance",
    ),
    "FeatureId": ("contextmap.visual_perception.visual_features", "FeatureId"),
    "FeatureScope": ("contextmap.visual_perception.visual_features", "FeatureScope"),
    "FeatureStoreError": ("contextmap.visual_perception.visual_features", "FeatureStoreError"),
    "FeatureStoreReader": ("contextmap.visual_perception.visual_features", "FeatureStoreReader"),
    "FeatureStoreWriter": ("contextmap.visual_perception.visual_features", "FeatureStoreWriter"),
    "KNOWN_CAPABILITIES": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "KNOWN_CAPABILITIES",
    ),
    "NormalizeOperation": ("contextmap.visual_perception.image_preparation", "NormalizeOperation"),
    "ObservationEvidence": (
        "contextmap.visual_perception.perception_evidence_set",
        "ObservationEvidence",
    ),
    "POOLING_BACKEND_ID": ("contextmap.visual_perception.visual_features", "POOLING_BACKEND_ID"),
    "POOLING_BACKEND_VERSION": (
        "contextmap.visual_perception.visual_features",
        "POOLING_BACKEND_VERSION",
    ),
    "POOLING_POLICY": ("contextmap.visual_perception.visual_features", "POOLING_POLICY"),
    "PerceptionEvidenceSelection": (
        "contextmap.visual_perception.perception_evidence_set",
        "PerceptionEvidenceSelection",
    ),
    "PerceptionEvidenceSet": (
        "contextmap.visual_perception.perception_evidence_set",
        "PerceptionEvidenceSet",
    ),
    "PerceptionResult": ("contextmap.visual_perception.models", "PerceptionResult"),
    "PipelineConfigError": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "PipelineConfigError",
    ),
    "PipelinePreset": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "PipelinePreset",
    ),
    "PreparedImage": ("contextmap.visual_perception.models", "PreparedImage"),
    "RecordedDenseFeatureMap": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "RecordedDenseFeatureMap",
    ),
    "RectifyOperation": ("contextmap.visual_perception.image_preparation", "RectifyOperation"),
    "RegionAssociationError": (
        "contextmap.visual_perception.visual_features",
        "RegionAssociationError",
    ),
    "RegionFeatureDiagnostic": (
        "contextmap.visual_perception.feature_diagnostics",
        "RegionFeatureDiagnostic",
    ),
    "RegionPoolingDiagnostics": (
        "contextmap.visual_perception.visual_features",
        "RegionPoolingDiagnostics",
    ),
    "RegionPoolingProvenance": (
        "contextmap.visual_perception.visual_features",
        "RegionPoolingProvenance",
    ),
    "RegionPoolingResult": ("contextmap.visual_perception.visual_features", "RegionPoolingResult"),
    "ReleasableBackend": ("contextmap.visual_perception.ports", "ReleasableBackend"),
    "ResizeOperation": ("contextmap.visual_perception.image_preparation", "ResizeOperation"),
    "ResolvedPipeline": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "ResolvedPipeline",
    ),
    "SemanticSupport": ("contextmap.visual_perception.models", "SemanticSupport"),
    "SourceImage": ("contextmap.visual_perception.image_preparation", "SourceImage"),
    "StageBackendFactory": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "StageBackendFactory",
    ),
    "StageDefinition": (
        "contextmap.visual_perception.execution.stage_execution",
        "StageDefinition",
    ),
    "StageGraphError": (
        "contextmap.visual_perception.execution.stage_execution",
        "StageGraphError",
    ),
    "StageOutcome": ("contextmap.visual_perception.execution.stage_execution", "StageOutcome"),
    "StageRunner": ("contextmap.visual_perception.execution.stage_execution", "StageRunner"),
    "StageSpec": ("contextmap.visual_perception.execution.pipeline_resolution", "StageSpec"),
    "StageStatus": ("contextmap.visual_perception.execution.stage_execution", "StageStatus"),
    "TransformationRecord": ("contextmap.visual_perception.models", "TransformationRecord"),
    "VISUAL_EVIDENCE_SCHEMA_VERSION": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VISUAL_EVIDENCE_SCHEMA_VERSION",
    ),
    "ValidRegion": ("contextmap.visual_perception.models", "ValidRegion"),
    "VisualEvidenceResult": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceResult",
    ),
    "VisualEvidenceRunError": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceRunError",
    ),
    "VisualEvidenceRunId": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceRunId",
    ),
    "VisualEvidenceRunManifest": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceRunManifest",
    ),
    "VisualEvidenceRunReader": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceRunReader",
    ),
    "VisualEvidenceRunWriter": (
        "contextmap.visual_perception.visual_evidence_artifact",
        "VisualEvidenceRunWriter",
    ),
    "VisualFeature": ("contextmap.visual_perception.visual_features", "VisualFeature"),
    "alphaclip_embedding_space": (
        "contextmap.visual_perception.visual_features",
        "alphaclip_embedding_space",
    ),
    "box_mask_shape": ("contextmap.visual_perception.visual_features", "box_mask_shape"),
    "clip_embedding_space": (
        "contextmap.visual_perception.visual_features",
        "clip_embedding_space",
    ),
    "decode_embedding_space": (
        "contextmap.visual_perception.visual_features",
        "decode_embedding_space",
    ),
    "decode_feature_payload_entry": (
        "contextmap.visual_perception.visual_features",
        "decode_feature_payload_entry",
    ),
    "decode_pipeline_preset": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "decode_pipeline_preset",
    ),
    "dense_extraction_diagnostic": (
        "contextmap.visual_perception.feature_diagnostics",
        "dense_extraction_diagnostic",
    ),
    "dense_feature_map_from_record": (
        "contextmap.visual_perception.feature_diagnostics",
        "dense_feature_map_from_record",
    ),
    "dense_feature_map_of": (
        "contextmap.visual_perception.feature_diagnostics",
        "dense_feature_map_of",
    ),
    "embedding_space_fingerprint": (
        "contextmap.visual_perception.visual_features",
        "embedding_space_fingerprint",
    ),
    "encode_embedding_space": (
        "contextmap.visual_perception.visual_features",
        "encode_embedding_space",
    ),
    "encode_feature_payload_entry": (
        "contextmap.visual_perception.visual_features",
        "encode_feature_payload_entry",
    ),
    "encode_pipeline_preset": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "encode_pipeline_preset",
    ),
    "ensure_compatible_embedding_spaces": (
        "contextmap.visual_perception.visual_features",
        "ensure_compatible_embedding_spaces",
    ),
    "ensure_compatible_features": (
        "contextmap.visual_perception.visual_features",
        "ensure_compatible_features",
    ),
    "execute_stage_graph": (
        "contextmap.visual_perception.execution.stage_execution",
        "execute_stage_graph",
    ),
    "execute_stage_graphs_stage_major": (
        "contextmap.visual_perception.execution.stage_execution",
        "execute_stage_graphs_stage_major",
    ),
    "feature_id_for": ("contextmap.visual_perception.evidence_identity", "feature_id_for"),
    "map_box_to_grid_cells": (
        "contextmap.visual_perception.visual_features",
        "map_box_to_grid_cells",
    ),
    "pool_region_feature": ("contextmap.visual_perception.visual_features", "pool_region_feature"),
    "pooling_backend_provenance": (
        "contextmap.visual_perception.visual_features",
        "pooling_backend_provenance",
    ),
    "prepare_image": ("contextmap.visual_perception.image_preparation", "prepare_image"),
    "resolve_pipeline": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "resolve_pipeline",
    ),
    "validate_pipeline_preset": (
        "contextmap.visual_perception.execution.pipeline_resolution",
        "validate_pipeline_preset",
    ),
    "with_dense_region_pooling": (
        "contextmap.visual_perception.execution.pipeline_presets",
        "with_dense_region_pooling",
    ),
    "without_scene_interpretation": (
        "contextmap.visual_perception.execution.pipeline_presets",
        "without_scene_interpretation",
    ),
    "write_feature_diagnostics": (
        "contextmap.visual_perception.feature_diagnostics",
        "write_feature_diagnostics",
    ),
    "write_feature_index": ("contextmap.visual_perception.visual_features", "write_feature_index"),
}


def __getattr__(name: str) -> Any:
    """Load a public visual contract only when a consumer needs it."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    return getattr(importlib.import_module(module_name), attribute)


__all__ = [
    "CANONICAL_PRESET_V1",
    "FEATURE_INDEX_FILENAME",
    "FEATURE_INDEX_SCHEMA_VERSION",
    "FEATURE_METRICS_PATH",
    "KNOWN_CAPABILITIES",
    "POOLING_BACKEND_ID",
    "POOLING_BACKEND_VERSION",
    "POOLING_POLICY",
    "VISUAL_EVIDENCE_SCHEMA_VERSION",
    "ArtifactReference",
    "BackendProvenance",
    "CropOperation",
    "DenseFeatureDiagnostic",
    "DenseFeatureMap",
    "DenseFeatureSampling",
    "EmbeddingSpace",
    "EmbeddingSpaceMismatchError",
    "EmptyRegionSupportError",
    "EvidenceSetError",
    "ExclusionRegion",
    "FeatureDebugLevel",
    "FeatureDiagnosticPreview",
    "FeatureEventStatus",
    "FeatureExtractionDiagnostic",
    "FeatureExtractor",
    "FeatureId",
    "FeaturePayloadEntry",
    "FeaturePayloadIntegrityError",
    "FeatureResolutionEnhancement",
    "FeatureResolutionEnhancementError",
    "FeatureResolutionEnhancementProvenance",
    "FeatureScope",
    "FeatureStoreError",
    "FeatureStoreReader",
    "FeatureStoreWriter",
    "NormalizeOperation",
    "ObservationEvidence",
    "PerceptionEvidenceSelection",
    "PerceptionEvidenceSet",
    "PerceptionResult",
    "PipelineConfigError",
    "PipelinePreset",
    "PreparedImage",
    "RecordedDenseFeatureMap",
    "RectifyOperation",
    "RegionAssociationError",
    "RegionFeatureDiagnostic",
    "RegionPoolingDiagnostics",
    "RegionPoolingProvenance",
    "RegionPoolingResult",
    "ReleasableBackend",
    "ResizeOperation",
    "ResolvedPipeline",
    "SemanticSupport",
    "SourceImage",
    "StageBackendFactory",
    "StageDefinition",
    "StageGraphError",
    "StageOutcome",
    "StageRunner",
    "StageSpec",
    "StageStatus",
    "TransformationRecord",
    "ValidRegion",
    "VisualEvidenceResult",
    "VisualEvidenceRunError",
    "VisualEvidenceRunId",
    "VisualEvidenceRunManifest",
    "VisualEvidenceRunReader",
    "VisualEvidenceRunWriter",
    "VisualFeature",
    "alphaclip_embedding_space",
    "box_mask_shape",
    "clip_embedding_space",
    "decode_embedding_space",
    "decode_feature_payload_entry",
    "decode_pipeline_preset",
    "dense_extraction_diagnostic",
    "dense_feature_map_from_record",
    "dense_feature_map_of",
    "embedding_space_fingerprint",
    "encode_embedding_space",
    "encode_feature_payload_entry",
    "encode_pipeline_preset",
    "ensure_compatible_embedding_spaces",
    "ensure_compatible_features",
    "execute_stage_graph",
    "execute_stage_graphs_stage_major",
    "feature_id_for",
    "map_box_to_grid_cells",
    "pool_region_feature",
    "pooling_backend_provenance",
    "prepare_image",
    "resolve_pipeline",
    "validate_pipeline_preset",
    "with_dense_region_pooling",
    "without_scene_interpretation",
    "write_feature_diagnostics",
    "write_feature_index",
]
