"""Contrato público da capability spatial_support.

A capability deriva um suporte espacial semântico persistente de um ``ContextMap`` finalizado
e de uma agregação voxel (geometry-only) do seu mapa: uma :class:`SpatialCell` por voxel
ocupado em que alguma entidade cai, carregando quanto da geometria de cada entidade o voxel
guarda. A identidade de uma célula é função pura da grade, **nunca** de um índice de ponto
bruto, e é isso que faz os atributos semânticos sobreviverem à simplificação da geometria
para centroides de voxel na apresentação.

Ela é uma consumidora derivada do ``ContextMap`` público e não possui lógica científica
nenhuma do pipeline que produz o mapa. O subpacote ``run_artifact`` é interno; esta é a
superfície cross-capability. Ver ``src/contextmap/spatial_support/docs/code-map.md`` para
qual módulo possui cada responsabilidade.
"""

from contextmap.spatial_support.derivation import derive_semantic_spatial_support_artifact
from contextmap.spatial_support.models import (
    EntityCellInput,
    EntitySupport,
    SemanticSpatialSupport,
    SpatialCell,
    SpatialCellId,
    spatial_cell_id_for,
)
from contextmap.spatial_support.policy import EntityVoxelSupportPolicy
from contextmap.spatial_support.run_artifact import (
    SCHEMA_VERSION,
    SpatialSupportArtifactError,
    SpatialSupportArtifactReader,
    SpatialSupportArtifactWriter,
    SpatialSupportManifest,
    SpatialSupportRunId,
)
from contextmap.spatial_support.service import (
    SemanticSpatialSupportError,
    build_semantic_spatial_support,
)

__all__ = [
    "SCHEMA_VERSION",
    "EntityCellInput",
    "EntitySupport",
    "EntityVoxelSupportPolicy",
    "SemanticSpatialSupport",
    "SemanticSpatialSupportError",
    "SpatialCell",
    "SpatialCellId",
    "SpatialSupportArtifactError",
    "SpatialSupportArtifactReader",
    "SpatialSupportArtifactWriter",
    "SpatialSupportManifest",
    "SpatialSupportRunId",
    "build_semantic_spatial_support",
    "derive_semantic_spatial_support_artifact",
    "spatial_cell_id_for",
]
