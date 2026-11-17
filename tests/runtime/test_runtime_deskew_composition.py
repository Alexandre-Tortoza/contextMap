"""The optional ``geometric_mapping.deskew`` component (C3): selected, it reaches the map executor.

Unselected, geometric mapping keeps the previous behavior exactly: no deskew policy is composed and
the executor assembles its inputs without one. A deskew configuration the capability does not define
is a configuration error, never a silently different correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.geometric_mapping import (
    PointTimeReference,
    PointTimeUnit,
    PoseInterpolationDeskewPolicy,
)
from contextmap.runtime import compose, compose_executors
from contextmap.runtime.composition import executor_identities
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.runtime.executors import GeometricMappingExecutor

_COMPONENT = "geometric_mapping.deskew"
_PARAMETERS = {
    "time_field": "time",
    "time_unit": "seconds",
    "reference": "scan_timestamp",
    "time_bin_ns": 1_000_000,
}


def _document(parameters: dict[str, Any]) -> dict[str, Any]:
    document = selected_document()
    document["components"]["geometric_mapping"]["deskew"] = {
        "backend": "pose-interpolation-deskew-v1",
        "pose-interpolation-deskew-v1": parameters,
    }
    return document


def _compose(tmp_path: Path, document: dict[str, Any] | None = None) -> Any:
    return compose(
        effective_from(tmp_path, document),
        module_available=lambda _name: True,
        environ={},
        stages=["geometric_mapping"],
    )


def test_the_selected_deskew_policy_is_composed_with_every_declared_parameter(
    tmp_path: Path,
) -> None:
    composed = _compose(tmp_path, _document(dict(_PARAMETERS)))

    assert composed.geometric_mapping_deskew == PoseInterpolationDeskewPolicy(
        time_field="time",
        time_unit=PointTimeUnit.SECONDS,
        reference=PointTimeReference.SCAN_TIMESTAMP,
        time_bin_ns=1_000_000,
    )


def test_unselected_there_is_no_deskew_never_a_default_one(tmp_path: Path) -> None:
    assert _compose(tmp_path).geometric_mapping_deskew is None


@pytest.mark.parametrize(
    "parameters",
    [
        {key: value for key, value in _PARAMETERS.items() if key != "time_unit"},
        {**_PARAMETERS, "time_unit": "fortnights"},
        {**_PARAMETERS, "reference": "scan_end"},
    ],
)
def test_a_deskew_the_capability_does_not_define_is_refused(
    tmp_path: Path, parameters: dict[str, Any]
) -> None:
    with pytest.raises(BackendConfigurationError) as error:
        _compose(tmp_path, _document(parameters))

    assert error.value.component_id == _COMPONENT


def test_the_map_executor_receives_the_selected_deskew_policy(tmp_path: Path) -> None:
    effective = effective_from(tmp_path, _document(dict(_PARAMETERS)))
    executors = compose_executors(
        effective,
        module_available=lambda _name: True,
        environ={},
        **executor_identities(effective, "test"),
    )

    executor = executors["geometric_mapping"]
    assert isinstance(executor, GeometricMappingExecutor)
    assert (
        executor.deskew == _compose(tmp_path, _document(dict(_PARAMETERS))).geometric_mapping_deskew
    )
