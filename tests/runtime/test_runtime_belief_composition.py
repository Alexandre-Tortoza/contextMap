"""The optional ``entity_resolution.belief`` component (C9): selected, it reaches the resolver.

Unselected, Entity Resolution derives no belief and its output is exactly the previous one. The
minimum support has no default: it is a scientific choice the configuration must state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document

from contextmap.entity_resolution import ResolvedBeliefPolicy
from contextmap.runtime import compose, compose_executors
from contextmap.runtime.composition import executor_identities
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.runtime.executors import EntityResolutionExecutor

_COMPONENT = "entity_resolution.belief"


def _document(parameters: dict[str, Any]) -> dict[str, Any]:
    document = selected_document()
    document["components"]["entity_resolution"]["belief"] = {
        "backend": "resolved-belief-min-support-v1",
        "resolved-belief-min-support-v1": parameters,
    }
    return document


def _compose(tmp_path: Path, document: dict[str, Any] | None = None) -> Any:
    return compose(
        effective_from(tmp_path, document),
        module_available=lambda _name: True,
        environ={},
        stages=["entity_resolution"],
    )


def test_the_selected_belief_policy_is_composed_with_its_declared_minimum(tmp_path: Path) -> None:
    composed = _compose(tmp_path, _document({"min_supporting_physical_observations": 2}))

    assert composed.entity_belief_policy == ResolvedBeliefPolicy(
        min_supporting_physical_observations=2
    )


def test_unselected_no_belief_is_derived_never_a_default_minimum(tmp_path: Path) -> None:
    assert _compose(tmp_path).entity_belief_policy is None


@pytest.mark.parametrize("parameters", [{}, {"min_supporting_physical_observations": 0}])
def test_a_minimum_the_rule_does_not_define_is_refused(
    tmp_path: Path, parameters: dict[str, Any]
) -> None:
    with pytest.raises(BackendConfigurationError) as error:
        _compose(tmp_path, _document(parameters))

    assert error.value.component_id == _COMPONENT


def test_the_resolution_executor_receives_the_selected_belief_policy(tmp_path: Path) -> None:
    effective = effective_from(tmp_path, _document({"min_supporting_physical_observations": 2}))
    executors = compose_executors(
        effective,
        module_available=lambda _name: True,
        environ={},
        **executor_identities(effective, "test"),
    )

    executor = executors["entity_resolution"]
    assert isinstance(executor, EntityResolutionExecutor)
    assert executor.belief_policy == ResolvedBeliefPolicy(min_supporting_physical_observations=2)
