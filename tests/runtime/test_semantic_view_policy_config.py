"""C6: a run configures the mask-outline views (arms B and D) from TOML, like any other view.

Only reads the runtime: the nested ``mask_outline`` group reaches
:class:`~contextmap.semantic_interpretation.SemanticViewPolicy` through the generic configuration
coercion, with no runtime code that knows about it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from runtime_documents import selected_document

from contextmap.runtime import resolve_effective_config
from contextmap.runtime.composition import resolve_semantic_request_policy
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.semantic_interpretation import (
    SemanticMaskOutline,
    SemanticViewPolicy,
    VisualViewKind,
)


def _resolve(tmp_path: Path, view_policy: str) -> SemanticViewPolicy:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(selected_document()), encoding="utf-8")
    arm = tmp_path / "arm.toml"
    arm.write_text(
        f"[components.semantic_interpretation.interpreter.qwen]\nview_policy = {view_policy}\n",
        encoding="utf-8",
    )
    effective = resolve_effective_config(files=[base, arm])
    return resolve_semantic_request_policy(effective).views


@pytest.mark.parametrize(
    ("kind", "arm"),
    [(VisualViewKind.MASK_OUTLINE_CROP, "b"), (VisualViewKind.MASK_OUTLINE_FULL_FRAME, "d")],
)
def test_an_outline_arm_resolves_from_toml_with_its_outline(
    tmp_path: Path, kind: VisualViewKind, arm: str
) -> None:
    views = _resolve(
        tmp_path,
        f'{{ region_views = ["{kind.value}"], mask_outline = {{ rgb = [255, 0, 255], '
        "width_px = 2 } }",
    )

    assert views == SemanticViewPolicy(
        region_views=(kind,), mask_outline=SemanticMaskOutline(rgb=(255, 0, 255), width_px=2)
    ), arm


def test_an_outline_arm_without_its_outline_is_refused_before_anything_runs(
    tmp_path: Path,
) -> None:
    with pytest.raises(BackendConfigurationError, match="mask_outline"):
        _resolve(tmp_path, '{ region_views = ["mask_outline_crop"] }')


def test_a_region_budget_resolves_from_toml_into_the_request_policy(tmp_path: Path) -> None:
    from contextmap.semantic_interpretation import RegionRequestBudget

    base = tmp_path / "base.json"
    base.write_text(json.dumps(selected_document()), encoding="utf-8")
    arm = tmp_path / "arm.toml"
    arm.write_text(
        "[components.semantic_interpretation.interpreter.qwen]\n"
        "region_budget = { max_regions = 8 }\n",
        encoding="utf-8",
    )

    policy = resolve_semantic_request_policy(resolve_effective_config(files=[base, arm]))

    assert policy.region_budget == RegionRequestBudget(max_regions=8)


def test_without_a_region_budget_the_request_policy_has_none(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(selected_document()), encoding="utf-8")

    policy = resolve_semantic_request_policy(resolve_effective_config(files=[base]))

    assert policy.region_budget is None
