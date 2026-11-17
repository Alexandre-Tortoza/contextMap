"""O vocabulário do suporte espacial semântico e os seus invariantes de construção.

Uma :class:`SpatialCell` é um *endereço espacial*, não uma entidade: a identidade dela é
função pura da grade e da chave inteira do voxel, então nunca depende do índice de um ponto
de uma nuvem concreta. Entidades ficam presas a uma célula como *suporte* -- quantos pontos
da entidade caem nela -- e é assim que os atributos semânticos continuam recuperáveis quando
a geometria é simplificada para centroides de voxel na apresentação.

O suporte é evidência de nível de crença *sobre onde* as entidades estão, agregada por
célula; ele nunca transforma uma célula em entidade e nunca substitui as entidades que
referencia.

Este módulo é folha: ele não importa nada de dentro da capability, não faz I/O e não possui
a regra de junção.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId
from contextmap.ingestion import FrameId
from contextmap.shared import Vector3

SpatialCellId = NewType("SpatialCellId", str)
"""Identity of one spatial cell, stable for a grid and independent of raw point indices."""


def spatial_cell_id_for(*, grid_fingerprint: str, key: tuple[int, int, int]) -> SpatialCellId:
    """Compute the deterministic identity of a cell from its grid and integer key.

    Args:
        grid_fingerprint: The voxel grid's own fingerprint -- frame, origin, resolution and
            indexing, and nothing else. Not the aggregation policy's fingerprint: a cell is
            a spatial address, so its identity must not move when what the aggregation
            computes per voxel changes.
        key: The cell's integer voxel key ``(kx, ky, kz)``.

    Returns:
        A pure function of the inputs, so a cell keeps its identity across derivations of
        the same grid without an identity registry, and never encodes a raw point index.
    """
    return SpatialCellId(f"spatial-cell:{grid_fingerprint}:{key[0]}_{key[1]}_{key[2]}")


@dataclass(frozen=True, kw_only=True)
class EntityCellInput:
    """One entity's authoritative geometry support, as the join consumes it.

    Attributes:
        entity_id: Identity of the entity in its map.
        geometry_refs: The entity's geometry references; at least one, all from one map.
    """

    entity_id: ContextEntityId
    geometry_refs: tuple[GeometryReference, ...]

    def __post_init__(self) -> None:
        """Require a non-empty support drawn from a single map.

        Raises:
            ValueError: If the entity has no geometry reference or mixes maps.
        """
        if not self.geometry_refs:
            raise ValueError(f"entity {self.entity_id!r} has no geometry reference")
        maps = {reference.map_id for reference in self.geometry_refs}
        if len(maps) != 1:
            raise ValueError(
                f"entity {self.entity_id!r} references more than one map: {sorted(maps)}"
            )


@dataclass(frozen=True, kw_only=True)
class EntitySupport:
    """How much of one entity's geometry a cell holds.

    Attributes:
        entity_id: The supported entity.
        support_weight: Raw points of the entity that fall in the cell; at least one.
            A count of measurements, never a probability, and never mixed with a score.
    """

    entity_id: ContextEntityId
    support_weight: int

    def __post_init__(self) -> None:
        """Require a positive weight.

        Raises:
            ValueError: If ``support_weight`` is not at least one.
        """
        if self.support_weight < 1:
            raise ValueError(
                f"support_weight of {self.entity_id!r} must be at least one, got "
                f"{self.support_weight}"
            )


@dataclass(frozen=True, kw_only=True)
class SpatialCell:
    """One occupied voxel of the support, with the entities that fall in it.

    Attributes:
        cell_id: Space-derived identity; see :func:`spatial_cell_id_for`.
        aggregate_index: The cell's position in the source voxel aggregation, which is the
            transport index the presentation geometry serves its centroid under. A consumer
            aligns a support row to a rendered voxel point by this index, never by a raw
            point index. At least zero.
        key: The cell's integer voxel key ``(kx, ky, kz)``.
        centroid_m: The voxel's centroid, in the grid frame; the presentation point.
        bounds: The axis-aligned box of the raw points the voxel summarizes.
        point_count: Raw measurements the voxel summarizes.
        entity_support: The entities that fall in the cell, sorted by entity id and unique;
            at least one, since a cell is emitted only when an entity supports it.
    """

    cell_id: SpatialCellId
    aggregate_index: int
    key: tuple[int, int, int]
    centroid_m: Vector3
    bounds: Bounds3D
    point_count: int
    entity_support: tuple[EntitySupport, ...]

    def __post_init__(self) -> None:
        """Validate the support list and the aggregate index.

        Raises:
            ValueError: If the aggregate index is negative, the cell has no entity support,
                or the support is not sorted by entity id and unique.
        """
        if self.aggregate_index < 0:
            raise ValueError(
                f"cell {self.cell_id!r} aggregate_index must be at least zero, got "
                f"{self.aggregate_index}"
            )
        if not self.entity_support:
            raise ValueError(f"cell {self.cell_id!r} is emitted only when an entity supports it")
        ids = [item.entity_id for item in self.entity_support]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ValueError(
                f"cell {self.cell_id!r} support must be sorted by entity id and unique"
            )


@dataclass(frozen=True, kw_only=True)
class SemanticSpatialSupport:
    """Every occupied cell of one grid that some entity supports.

    Attributes:
        source_map_id: The raw map whose entities and geometry the support was derived from.
        frame_id: The grid frame every cell's centroid and bounds are expressed in.
        grid_fingerprint: The voxel grid's own fingerprint; part of every cell identity.
        aggregation_rule: The rule the underlying voxel aggregation declares.
        cells: The supported cells, ordered by voxel key. The order is the one
            ``VoxelAggregation`` guarantees for its aggregates, borrowed rather than
            imposed here; a test pins it.
    """

    source_map_id: MapId
    frame_id: FrameId
    grid_fingerprint: str
    aggregation_rule: str
    cells: tuple[SpatialCell, ...]

    def __post_init__(self) -> None:
        """Require every cell to be expressed in the support's frame.

        Raises:
            ValueError: If a cell's bounds declare a frame other than ``frame_id``.
        """
        for cell in self.cells:
            if cell.bounds.frame_id != self.frame_id:
                raise ValueError(
                    f"cell {cell.cell_id!r} is in frame {cell.bounds.frame_id!r}, not the "
                    f"support frame {self.frame_id!r}"
                )
