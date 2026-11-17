"""Scene interpretation is optional: a run can ask the interpreter for region labels only.

Without a scene prompt the policy interprets regions alone, the preset drops the
``scene_interpretation`` stage (region interpretation never depends on it), and conditioning region
requests on a scene context that is never produced is refused.
"""

from __future__ import annotations

import pytest

from contextmap.semantic_interpretation import (
    SemanticInterpretationMode,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    with_dense_region_pooling,
    without_scene_interpretation,
)

_VIEWS = SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,))


def test_a_prompt_policy_without_a_scene_interprets_regions_only() -> None:
    policy = SemanticPromptPolicy(region="region-interaction/v1")

    request = SemanticRequestPolicy.from_prompt_policy(policy, views=_VIEWS)

    assert policy.scene is None
    assert request.prompt_for(SemanticInterpretationMode.SCENE) is None
    assert request.prompt_for(SemanticInterpretationMode.REGION) is not None
    assert request.to_document()["prompts"]["scene"] is None


def test_scene_context_without_a_scene_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="scene"):
        SemanticPromptPolicy(region="region-scene-context/v1", region_scene_context=True)


def test_a_policy_with_a_scene_keeps_its_identity() -> None:
    before = SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(scene="scene/v1", region="region/v1"), views=_VIEWS
    )

    assert before.prompt_for(SemanticInterpretationMode.SCENE) is not None


def test_the_preset_without_scene_drops_only_that_stage() -> None:
    preset = without_scene_interpretation(CANONICAL_PRESET_V1)

    ids = [stage.stage_id for stage in preset.stages]
    assert "scene_interpretation" not in ids
    assert ids == [
        stage.stage_id
        for stage in CANONICAL_PRESET_V1.stages
        if stage.stage_id != "scene_interpretation"
    ]
    assert preset.preset_id != CANONICAL_PRESET_V1.preset_id


def test_dropping_the_scene_composes_with_dense_region_pooling() -> None:
    preset = without_scene_interpretation(with_dense_region_pooling(CANONICAL_PRESET_V1))

    ids = [stage.stage_id for stage in preset.stages]
    assert "dense_region_pooling" in ids and "scene_interpretation" not in ids


def test_a_preset_without_a_scene_stage_is_refused() -> None:
    once = without_scene_interpretation(CANONICAL_PRESET_V1)

    with pytest.raises(ValueError, match="scene_interpretation"):
        without_scene_interpretation(once)
