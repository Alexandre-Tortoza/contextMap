"""The box rule of membership is an optional, explicit variation point of Sensor Association.

Selecting ``sensor_association.membership = box-depth-component-v1`` builds its policy from the
configuration and hands it to the association executor; without the component the association
keeps the mask rule, exactly as before.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.runtime.composition import compose, compose_executors
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.runtime.executors import SensorAssociationExecutor
from contextmap.sensor_association import BoxDepthComponentPolicy

_POLICY = BoxDepthComponentPolicy(
    cell_size_m=0.15, core_fraction=0.5, minimum_support_points=3, ambiguity_ratio=0.5
)
_PARAMETERS = dataclasses.asdict(_POLICY)


def _document(**parameters: Any) -> dict[str, Any]:
    document = copy.deepcopy(selected_document())
    document["components"].setdefault("sensor_association", {})["membership"] = {
        "backend": "box-depth-component-v1",
        "box-depth-component-v1": {**_PARAMETERS, **parameters},
    }
    return document


def _compose(tmp_path: Path, document: dict[str, Any]) -> Any:
    return compose(
        effective_from(tmp_path, document),
        stages=["sensor_association"],
        module_available=lambda _: True,
        environ={},
    )


def test_selecting_the_box_rule_builds_its_policy(tmp_path: Path) -> None:
    composed = _compose(tmp_path, _document())

    assert composed.association_box_policy == _POLICY


def test_without_the_component_the_association_keeps_the_mask_rule(tmp_path: Path) -> None:
    composed = _compose(tmp_path, selected_document())

    assert composed.association_box_policy is None


def test_a_meaningless_parameter_is_refused_with_the_component_named(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigurationError, match=r"sensor_association\.membership"):
        _compose(tmp_path, _document(core_fraction=0.0))


def test_the_composed_executor_carries_the_policy(tmp_path: Path) -> None:
    executors = compose_executors(
        effective_from(tmp_path, _document()), module_available=lambda _: True, environ={}
    )

    executor = executors["sensor_association"]
    assert isinstance(executor, SensorAssociationExecutor)
    assert executor.box_policy == _POLICY
