"""Invariants of the spatial_support domain types.

These protect the contract that keeps a cell a space-derived address with well-formed
support, independent of any raw point index.
"""

from __future__ import annotations

import pytest

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import FrameId
from contextmap.spatial_support import (
    EntityCellInput,
    EntitySupport,
    SemanticSpatialSupport,
    SpatialCell,
    spatial_cell_id_for,
)

MAP = MapId("map-0001")


def _ref(index: int) -> GeometryReference:
    return GeometryReference(map_id=MAP, geometry_id=geometry_id_for(map_id=MAP, index=index))


def _bounds() -> Bounds3D:
    return Bounds3D(frame_id=FrameId("map"), minimum_m=(0.0, 0.0, 0.0), maximum_m=(1.0, 1.0, 1.0))


def test_cell_identity_is_a_pure_function_of_grid_and_key_not_of_a_point_index() -> None:
    key = (2, -3, 5)
    first = spatial_cell_id_for(grid_fingerprint="sha256:abc", key=key)
    again = spatial_cell_id_for(grid_fingerprint="sha256:abc", key=key)
    other_grid = spatial_cell_id_for(grid_fingerprint="sha256:def", key=key)

    assert first == again
    assert first == "spatial-cell:sha256:abc:2_-3_5"
    assert first != other_grid


def test_an_entity_input_must_have_support_from_a_single_map() -> None:
    other = MapId("map-0002")
    with pytest.raises(ValueError, match="more than one map"):
        EntityCellInput(
            entity_id=ContextEntityId("entity--x"),
            geometry_refs=(
                _ref(0),
                GeometryReference(map_id=other, geometry_id=geometry_id_for(map_id=other, index=1)),
            ),
        )


def test_an_entity_input_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="no geometry reference"):
        EntityCellInput(entity_id=ContextEntityId("entity--x"), geometry_refs=())


def test_support_weight_counts_measurements_and_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EntitySupport(entity_id=ContextEntityId("entity--x"), support_weight=0)


def test_a_cell_without_entity_support_is_never_emitted() -> None:
    with pytest.raises(ValueError, match="only when an entity supports it"):
        SpatialCell(
            cell_id=spatial_cell_id_for(grid_fingerprint="sha256:abc", key=(0, 0, 0)),
            aggregate_index=0,
            key=(0, 0, 0),
            centroid_m=(0.5, 0.5, 0.5),
            bounds=_bounds(),
            point_count=3,
            entity_support=(),
        )


def test_cell_support_must_be_sorted_by_entity_id_and_unique() -> None:
    beta = EntitySupport(entity_id=ContextEntityId("entity--beta"), support_weight=1)
    alpha = EntitySupport(entity_id=ContextEntityId("entity--alpha"), support_weight=2)
    with pytest.raises(ValueError, match="sorted by entity id and unique"):
        SpatialCell(
            cell_id=spatial_cell_id_for(grid_fingerprint="sha256:abc", key=(0, 0, 0)),
            aggregate_index=0,
            key=(0, 0, 0),
            centroid_m=(0.5, 0.5, 0.5),
            bounds=_bounds(),
            point_count=3,
            entity_support=(beta, alpha),
        )


def test_a_cell_aggregate_index_must_not_be_negative() -> None:
    support = EntitySupport(entity_id=ContextEntityId("entity--x"), support_weight=1)
    with pytest.raises(ValueError, match="aggregate_index must be at least zero"):
        SpatialCell(
            cell_id=spatial_cell_id_for(grid_fingerprint="sha256:abc", key=(0, 0, 0)),
            aggregate_index=-1,
            key=(0, 0, 0),
            centroid_m=(0.5, 0.5, 0.5),
            bounds=_bounds(),
            point_count=3,
            entity_support=(support,),
        )


def test_every_cell_must_be_expressed_in_the_support_frame() -> None:
    support = EntitySupport(entity_id=ContextEntityId("entity--x"), support_weight=1)
    cell = SpatialCell(
        cell_id=spatial_cell_id_for(grid_fingerprint="sha256:abc", key=(0, 0, 0)),
        aggregate_index=0,
        key=(0, 0, 0),
        centroid_m=(0.5, 0.5, 0.5),
        bounds=_bounds(),
        point_count=3,
        entity_support=(support,),
    )

    # A célula declara frame "map"; um suporte em outro frame misturaria sistemas de
    # coordenadas, que é exatamente a regressão de frame que este invariante intercepta.
    with pytest.raises(ValueError, match="not the support frame"):
        SemanticSpatialSupport(
            source_map_id=MAP,
            frame_id=FrameId("odom"),
            grid_fingerprint="sha256:abc",
            aggregation_rule="inter-scan-voxel-centroid-v1",
            cells=(cell,),
        )
