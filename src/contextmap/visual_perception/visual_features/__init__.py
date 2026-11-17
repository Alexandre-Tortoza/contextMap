"""Visual features: dense and region-scoped visual embeddings.

Visual features are evidence of visual structure (embeddings, dense maps).
They carry no semantic identity; semantics come from later comparison with claims.

Não há writer nem reader de run aqui: a extração de features é um componente do stage
``visual_perception``, e as features são publicadas no artifact daquele stage — os
metadados em ``outputs/`` e os tensores no store de payloads (``FeatureStoreWriter``).
Este subpacote devolve contratos, não artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextmap.visual_perception.visual_features.feature_payload_store import (
    FEATURE_INDEX_FILENAME,
    FEATURE_INDEX_SCHEMA_VERSION,
    FeaturePayloadEntry,
    FeaturePayloadIntegrityError,
    FeatureStoreError,
    FeatureStoreReader,
    FeatureStoreWriter,
    decode_feature_payload_entry,
    encode_feature_payload_entry,
    write_feature_index,
)
from contextmap.visual_perception.visual_features.visual_evidence import (
    EmbeddingSpace,
    EmbeddingSpaceMismatchError,
    FeatureId,
    FeatureScope,
    PoolingMethod,
    VisualFeature,
    alphaclip_embedding_space,
    clip_embedding_space,
    decode_embedding_space,
    embedding_space_fingerprint,
    encode_embedding_space,
    ensure_compatible_embedding_spaces,
    ensure_compatible_features,
)

if TYPE_CHECKING:
    # Em execução estes nomes chegam por ``__getattr__``, para não fechar o ciclo de
    # import com o resto de visual_perception. O bloco abaixo existe só para o type
    # checker: sem ele, cada nome desta API pública chega ao consumidor como variável
    # ``Any``, e não como o contrato que ele declara.
    from contextmap.visual_perception.visual_features.feature_enhancement import (
        FeatureResolutionEnhancementError,
        enhance_feature_resolution,
    )
    from contextmap.visual_perception.visual_features.ports import (
        FeatureExtractor,
        FeatureResolutionEnhancement,
    )
    from contextmap.visual_perception.visual_features.region_feature_pooling import (
        POOLING_BACKEND_ID,
        POOLING_BACKEND_VERSION,
        POOLING_POLICY,
        DenseFeatureMap,
        DenseFeatureSampling,
        EmptyRegionSupportError,
        FeatureResolutionEnhancementProvenance,
        RegionAssociationError,
        RegionPoolingDiagnostics,
        RegionPoolingProvenance,
        RegionPoolingResult,
        box_mask_shape,
        map_box_to_grid_cells,
        pool_region_feature,
        pooling_backend_provenance,
    )


def __getattr__(name: str) -> object:
    """Lazy import for region_feature_pooling to avoid circular dependencies."""
    import importlib

    _mappings = {
        "POOLING_BACKEND_ID": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "POOLING_BACKEND_ID",
        ),
        "POOLING_BACKEND_VERSION": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "POOLING_BACKEND_VERSION",
        ),
        "POOLING_POLICY": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "POOLING_POLICY",
        ),
        "DenseFeatureMap": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "DenseFeatureMap",
        ),
        "DenseFeatureSampling": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "DenseFeatureSampling",
        ),
        "EmptyRegionSupportError": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "EmptyRegionSupportError",
        ),
        "FeatureResolutionEnhancementProvenance": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "FeatureResolutionEnhancementProvenance",
        ),
        "FeatureExtractor": (
            "contextmap.visual_perception.visual_features.ports",
            "FeatureExtractor",
        ),
        "FeatureResolutionEnhancement": (
            "contextmap.visual_perception.visual_features.ports",
            "FeatureResolutionEnhancement",
        ),
        "FeatureResolutionEnhancementError": (
            "contextmap.visual_perception.visual_features.feature_enhancement",
            "FeatureResolutionEnhancementError",
        ),
        "enhance_feature_resolution": (
            "contextmap.visual_perception.visual_features.feature_enhancement",
            "enhance_feature_resolution",
        ),
        "RegionAssociationError": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "RegionAssociationError",
        ),
        "RegionPoolingDiagnostics": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "RegionPoolingDiagnostics",
        ),
        "RegionPoolingProvenance": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "RegionPoolingProvenance",
        ),
        "RegionPoolingResult": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "RegionPoolingResult",
        ),
        "box_mask_shape": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "box_mask_shape",
        ),
        "map_box_to_grid_cells": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "map_box_to_grid_cells",
        ),
        "pool_region_feature": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "pool_region_feature",
        ),
        "pooling_backend_provenance": (
            "contextmap.visual_perception.visual_features.region_feature_pooling",
            "pooling_backend_provenance",
        ),
    }

    if name in _mappings:
        module_name, attr_name = _mappings[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "FEATURE_INDEX_FILENAME",
    "FEATURE_INDEX_SCHEMA_VERSION",
    "POOLING_BACKEND_ID",
    "POOLING_BACKEND_VERSION",
    "POOLING_POLICY",
    "DenseFeatureMap",
    "DenseFeatureSampling",
    "EmbeddingSpace",
    "EmbeddingSpaceMismatchError",
    "EmptyRegionSupportError",
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
    "PoolingMethod",
    "RegionAssociationError",
    "RegionPoolingDiagnostics",
    "RegionPoolingProvenance",
    "RegionPoolingResult",
    "VisualFeature",
    "alphaclip_embedding_space",
    "box_mask_shape",
    "clip_embedding_space",
    "decode_embedding_space",
    "decode_feature_payload_entry",
    "embedding_space_fingerprint",
    "encode_embedding_space",
    "encode_feature_payload_entry",
    "enhance_feature_resolution",
    "ensure_compatible_embedding_spaces",
    "ensure_compatible_features",
    "map_box_to_grid_cells",
    "pool_region_feature",
    "pooling_backend_provenance",
    "write_feature_index",
]
