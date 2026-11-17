"""The join that anchors entity support to persistent spatial cells.

Contract evidence only: a raw map whose scans revisit the same landmarks is aggregated
into voxels, and entities defined over disjoint sets of raw points must have their support
attributed to the voxels their points fall in, with a stable, space-derived cell identity
that does not depend on any raw point index. This is the property that lets semantic
attributes survive geometry simplification.
"""

from __future__ import annotations

import numpy as np
import pytest
from map_builders import MAP_ID
from voxel_builders import revisit_map, unit_policy

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import (
    GeometryReference,
    MapId,
    aggregate_geometry,
    geometry_id_for,
)
from contextmap.spatial_support import (
    EntityCellInput,
    SemanticSpatialSupportError,
    build_semantic_spatial_support,
    spatial_cell_id_for,
)

# Marcos que caem em voxels distintos de um grid de 1 m a partir da origem: alguns colapsam
# no voxel (0, 0, 0), outro fica isolado em (1, 0, 0), garantindo mais de uma célula ocupada.
REVISIT = (
    ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
    ((0.40, 0.40, 0.40),),
    ((2.60, 0.20, 0.80),),
)


def _references(map_id: MapId, indices: list[int]) -> tuple[GeometryReference, ...]:
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))
        for index in indices
    )


def test_entity_support_is_attributed_to_the_voxel_its_points_fall_in() -> None:
    raw = revisit_map(REVISIT)
    policy = unit_policy(cell_m=1.0, frame="map")
    aggregation = aggregate_geometry(raw, policy)

    membership = aggregation.membership(raw)
    count = raw.geometric_map.point_count
    left = list(range(count // 2))
    right = list(range(count // 2, count))
    alpha = ContextEntityId("entity--alpha")
    beta = ContextEntityId("entity--beta")

    support = build_semantic_spatial_support(
        entities=[
            EntityCellInput(entity_id=alpha, geometry_refs=_references(MAP_ID, left)),
            EntityCellInput(entity_id=beta, geometry_refs=_references(MAP_ID, right)),
        ],
        aggregation=aggregation,
        raw_geometry=raw,
    )

    # Uma célula por agregado tocado por alguma entidade, com identidade derivada do espaço.
    touched = sorted({int(index) for index in membership})
    assert [cell.key for cell in support.cells] == [
        tuple(int(component) for component in aggregation.aggregate(index).key) for index in touched
    ]
    for cell in support.cells:
        assert cell.cell_id == spatial_cell_id_for(
            grid_fingerprint=policy.grid.fingerprint(), key=cell.key
        )

    # O índice do agregado é o índice de transporte que a geometria de apresentação serve:
    # ele identifica o voxel renderizado, e a chave da célula é a do agregado nessa posição.
    for cell in support.cells:
        assert cell.key == tuple(
            int(component) for component in aggregation.aggregate(cell.aggregate_index).key
        )

    # O peso por entidade em cada célula é a contagem de pontos daquela entidade no voxel.
    weight_of = {
        (cell.cell_id, item.entity_id): item.support_weight
        for cell in support.cells
        for item in cell.entity_support
    }
    for aggregate_index in touched:
        cell_id = spatial_cell_id_for(
            grid_fingerprint=policy.grid.fingerprint(),
            key=aggregation.aggregate(aggregate_index).key,
        )
        for entity_id, indices in ((alpha, left), (beta, right)):
            expected = int(np.count_nonzero(membership[indices] == aggregate_index))
            assert weight_of.get((cell_id, entity_id), 0) == expected

    # Nenhuma informação de suporte depende do índice do ponto bruto: some das células.
    total_weight = sum(
        item.support_weight for cell in support.cells for item in cell.entity_support
    )
    assert total_weight == count


def test_the_cells_are_ordered_by_voxel_key() -> None:
    raw = revisit_map(REVISIT)
    aggregation = aggregate_geometry(raw, unit_policy(cell_m=1.0, frame="map"))
    count = raw.geometric_map.point_count

    support = build_semantic_spatial_support(
        entities=[
            EntityCellInput(
                entity_id=ContextEntityId("entity--alpha"),
                geometry_refs=_references(MAP_ID, list(range(count))),
            )
        ],
        aggregation=aggregation,
        raw_geometry=raw,
    )

    # NOTE: a junção itera os agregados tocados por índice crescente, e isso só é ordem de
    # chave porque VoxelAggregation garante agregados ordenados por (x, y, z). A garantia é
    # emprestada de geometric_mapping; este teste falha alto se ela mudar lá.
    keys = [cell.key for cell in support.cells]
    assert keys == sorted(keys)


def test_an_entity_from_another_map_is_refused() -> None:
    raw = revisit_map(REVISIT)
    aggregation = aggregate_geometry(raw, unit_policy(cell_m=1.0, frame="map"))
    other = MapId("map-9999")

    with pytest.raises(SemanticSpatialSupportError, match="not the aggregation's source map"):
        build_semantic_spatial_support(
            entities=[
                EntityCellInput(
                    entity_id=ContextEntityId("entity--alien"),
                    geometry_refs=_references(other, [0]),
                )
            ],
            aggregation=aggregation,
            raw_geometry=raw,
        )


def test_a_point_that_falls_in_no_voxel_of_the_aggregation_is_refused() -> None:
    # A agregação cobre só os dois primeiros scans; o mapa bruto completo tem um marco
    # distante, cuja chave fica fora da caixa de agregados, então membership devolve -1.
    narrow = revisit_map(REVISIT[:2])
    aggregation = aggregate_geometry(narrow, unit_policy(cell_m=1.0, frame="map"))
    full = revisit_map(REVISIT)
    outside = full.geometric_map.point_count - 1

    with pytest.raises(SemanticSpatialSupportError, match="falls in no voxel"):
        build_semantic_spatial_support(
            entities=[
                EntityCellInput(
                    entity_id=ContextEntityId("entity--far"),
                    geometry_refs=_references(MAP_ID, [outside]),
                )
            ],
            aggregation=aggregation,
            raw_geometry=full,
        )


def test_the_cell_identity_depends_on_the_grid_and_not_on_the_aggregation_policy() -> None:
    raw = revisit_map(REVISIT)
    policy = unit_policy(cell_m=1.0, frame="map")
    aggregation = aggregate_geometry(raw, policy)
    count = raw.geometric_map.point_count

    support = build_semantic_spatial_support(
        entities=[
            EntityCellInput(
                entity_id=ContextEntityId("entity--alpha"),
                geometry_refs=_references(MAP_ID, list(range(count))),
            )
        ],
        aggregation=aggregation,
        raw_geometry=raw,
    )

    # A célula é um endereço espacial: a identidade dela só pode depender de frame, origem,
    # resolução e indexação. O fingerprint da política de agregação também cobre o
    # representante, as estatísticas, o lineage e a versão da própria política, então usá-lo
    # renomearia toda célula a cada mudança de semântica da agregação.
    assert support.grid_fingerprint == policy.grid.fingerprint()
    assert support.grid_fingerprint != policy.fingerprint()
    for cell in support.cells:
        assert cell.cell_id == spatial_cell_id_for(
            grid_fingerprint=policy.grid.fingerprint(), key=cell.key
        )
