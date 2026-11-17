"""The optional ``geometric_mapping.aggregation`` component: an intra-scan voxel map by config.

Selected, the map executor merges the points of each scan that share a voxel into their centroid
(``ScanVoxelPolicy``), as the experiment drivers did by hand. Unselected, the map keeps every raw
point, exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.geometric_mapping import ScanVoxelPolicy
from contextmap.runtime import compose, compose_executors
from contextmap.runtime.composition import executor_identities
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.runtime.executors import GeometricMappingExecutor

_COMPONENT = "geometric_mapping.aggregation"


def _document(parameters: dict[str, Any]) -> dict[str, Any]:
    document = selected_document()
    document["components"]["geometric_mapping"]["aggregation"] = {
        "backend": "scan-voxel-centroid-v1",
        "scan-voxel-centroid-v1": parameters,
    }
    return document


def _compose(tmp_path: Path, document: dict[str, Any] | None = None) -> Any:
    return compose(
        effective_from(tmp_path, document),
        module_available=lambda _name: True,
        environ={},
        stages=["geometric_mapping"],
    )


def test_the_selected_voxel_size_is_composed(tmp_path: Path) -> None:
    composed = _compose(tmp_path, _document({"cell_m": 0.1}))

    assert composed.geometric_mapping_aggregation == ScanVoxelPolicy(cell_m=0.1)


def test_unselected_the_map_keeps_every_raw_point(tmp_path: Path) -> None:
    assert _compose(tmp_path).geometric_mapping_aggregation is None


@pytest.mark.parametrize("parameters", [{}, {"cell_m": 0.0}, {"cell_m": -1.0}])
def test_a_voxel_size_the_rule_does_not_define_is_refused(
    tmp_path: Path, parameters: dict[str, Any]
) -> None:
    with pytest.raises(BackendConfigurationError) as error:
        _compose(tmp_path, _document(parameters))

    assert error.value.component_id == _COMPONENT


def test_the_map_executor_receives_the_selected_aggregation(tmp_path: Path) -> None:
    effective = effective_from(tmp_path, _document({"cell_m": 0.1}))
    executors = compose_executors(
        effective,
        module_available=lambda _name: True,
        environ={},
        **executor_identities(effective, "test"),
    )

    executor = executors["geometric_mapping"]
    assert isinstance(executor, GeometricMappingExecutor)
    assert executor.aggregation == ScanVoxelPolicy(cell_m=0.1)
