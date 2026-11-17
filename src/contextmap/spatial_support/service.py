"""A junção: atribuir o suporte geométrico das entidades às células de uma agregação voxel.

A junção reutiliza o ``membership`` da própria agregação -- qual ponto bruto cada voxel
ocupado resume -- então o voxel a que um ponto pertence é decidido pela geometria, nunca por
um re-binning próprio. Cada ponto de uma entidade é atribuído ao voxel em que cai, e as
contagens por entidade viram :class:`~contextmap.spatial_support.EntitySupport` naquela
célula.

Este módulo declara, em :data:`IMPLEMENTED_POLICY`, qual versão da política ele implementa;
ele não persiste nada e não abre artifact a montante.
"""

from __future__ import annotations

from collections.abc import Iterable

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryBlockSource,
    VoxelAggregation,
    VoxelGridSpec,
    geometry_index_of,
)
from contextmap.spatial_support.models import (
    EntityCellInput,
    EntitySupport,
    SemanticSpatialSupport,
    SpatialCell,
    spatial_cell_id_for,
)
from contextmap.spatial_support.policy import EntityVoxelSupportPolicy

IMPLEMENTED_POLICY = EntityVoxelSupportPolicy()
"""The policy version this join implements; a run records it as its own identity."""


class SemanticSpatialSupportError(ValueError):
    """Raised when entity support cannot be joined onto an aggregation without inconsistency."""


def build_semantic_spatial_support(
    *,
    entities: Iterable[EntityCellInput],
    aggregation: VoxelAggregation,
    raw_geometry: GeometryBlockSource,
) -> SemanticSpatialSupport:
    """Attribute each entity's geometry support to the voxels its points fall in.

    Args:
        entities: The entities to place, each with its geometry references.
        aggregation: The voxel aggregation of the raw map the entities reference.
        raw_geometry: The raw map the aggregation was derived from; only read, to recover
            which voxel summarizes each raw point.

    Returns:
        The occupied cells that some entity supports, with the per-entity point counts as
        :class:`EntitySupport`. The cells come out in the order the aggregation stores its
        aggregates, which it guarantees to be voxel-key order.

    Raises:
        SemanticSpatialSupportError: If an entity references a map other than the raw map,
            or a referenced point falls in no voxel of the aggregation.
    """
    source_map_id = raw_geometry.geometric_map.map_id
    membership = aggregation.membership(raw_geometry)
    grid = aggregation.policy.grid
    # NOTE: a identidade da célula é um endereço espacial, então só pode depender da grade.
    # O fingerprint da política de agregação cobre também representante, estatísticas,
    # lineage e a versão dela: usá-lo renomearia toda célula a cada mudança de semântica da
    # agregação, sobre uma grade byte-idêntica.
    grid_fingerprint = grid.fingerprint()

    weights: dict[int, dict[ContextEntityId, int]] = {}
    for entity in entities:
        for reference in entity.geometry_refs:
            if reference.map_id != source_map_id:
                raise SemanticSpatialSupportError(
                    f"entity {entity.entity_id!r} references map {reference.map_id!r}, not the "
                    f"aggregation's source map {source_map_id!r}"
                )
            index = geometry_index_of(map_id=reference.map_id, geometry_id=reference.geometry_id)
            aggregate_index = int(membership[index])
            if aggregate_index < 0:
                raise SemanticSpatialSupportError(
                    f"point {index} of entity {entity.entity_id!r} falls in no voxel of the "
                    "aggregation"
                )
            per_entity = weights.setdefault(aggregate_index, {})
            per_entity[entity.entity_id] = per_entity.get(entity.entity_id, 0) + 1

    cells = tuple(
        _cell(aggregation, aggregate_index, weights[aggregate_index], grid_fingerprint, grid)
        for aggregate_index in sorted(weights)
    )
    return SemanticSpatialSupport(
        source_map_id=source_map_id,
        frame_id=grid.frame_id,
        grid_fingerprint=grid_fingerprint,
        aggregation_rule=aggregation.policy.rule,
        cells=cells,
    )


def _cell(
    aggregation: VoxelAggregation,
    aggregate_index: int,
    per_entity: dict[ContextEntityId, int],
    grid_fingerprint: str,
    grid: VoxelGridSpec,
) -> SpatialCell:
    aggregate = aggregation.aggregate(aggregate_index)
    key = (int(aggregate.key[0]), int(aggregate.key[1]), int(aggregate.key[2]))
    entity_support = tuple(
        EntitySupport(entity_id=entity_id, support_weight=per_entity[entity_id])
        for entity_id in sorted(per_entity)
    )
    return SpatialCell(
        cell_id=spatial_cell_id_for(grid_fingerprint=grid_fingerprint, key=key),
        aggregate_index=aggregate_index,
        key=key,
        centroid_m=aggregate.centroid_m,
        bounds=Bounds3D(
            frame_id=grid.frame_id,
            minimum_m=aggregate.minimum_m,
            maximum_m=aggregate.maximum_m,
        ),
        point_count=aggregate.point_count,
        entity_support=entity_support,
    )
