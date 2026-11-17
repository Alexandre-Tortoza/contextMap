"""Presets versionados da topologia de Percepção Visual.

Este módulo declara somente topologias selecionadas e suas derivações explícitas.
Ele não resolve backends, não executa estágios nem altera regras científicas dos
estágios que conecta.
"""

from __future__ import annotations

from dataclasses import replace

from contextmap.visual_perception.execution.pipeline_resolution import (
    REGION_SET_ASSEMBLY_POLICY,
    PipelinePreset,
    StageSpec,
)
from contextmap.visual_perception.visual_features import FeatureScope

CANONICAL_PRESET_V1 = PipelinePreset(
    preset_id="canonical/1",
    description=(
        "Currently validated default Visual Perception topology: region "
        "discovery over the prepared image, dense and region-scoped "
        "feature extraction, and scene- and region-level semantic "
        "interpretation."
    ),
    stages=(
        StageSpec(stage_id="image_preparation", capability="image_preparation"),
        StageSpec(
            stage_id="region_discovery",
            capability="region_discovery",
            inputs={"image": "image_preparation"},
            backend_id="region_discovery/canonical",
        ),
        StageSpec(
            stage_id="dense_feature_extraction",
            capability="feature_extractor",
            inputs={"image": "image_preparation"},
            backend_id="dense_feature_extractor/canonical",
            feature_scope=FeatureScope.DENSE,
        ),
        StageSpec(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            inputs={"image": "image_preparation", "regions": "region_discovery"},
            backend_id="region_feature_extractor/canonical",
            feature_scope=FeatureScope.REGION,
        ),
        StageSpec(
            stage_id="scene_interpretation",
            capability="scene_interpretation",
            inputs={"image": "image_preparation"},
            backend_id="semantic_interpreter/canonical",
        ),
        StageSpec(
            stage_id="region_interpretation",
            capability="region_interpretation",
            inputs={"image": "image_preparation", "regions": "region_discovery"},
            backend_id="semantic_interpreter/canonical",
        ),
    ),
)
"""The versioned canonical/default Visual Perception preset."""


def with_dense_region_pooling(preset: PipelinePreset) -> PipelinePreset:
    """Return a preset with explicit mask-weighted dense region pooling.

    Args:
        preset: Preset to derive without mutating it.

    Returns:
        A separately identified derived preset.
    """
    pooled = StageSpec(
        stage_id="dense_region_pooling",
        capability="dense_region_pooling",
        inputs={
            "image": "image_preparation",
            "regions": "region_discovery",
            "dense_features": "dense_feature_extraction",
        },
        backend_id="dense_region_pooling/canonical2",
        parameters={"policy_id": "mask_weighted_mean_preserve_l2_v2"},
    )
    stages = list(preset.stages)
    region_source = (
        "region_set"
        if any(stage.stage_id == "region_set" for stage in stages)
        else "region_discovery"
    )
    pooled = replace(pooled, inputs={**pooled.inputs, "regions": region_source})
    scene_index = next(
        (index for index, stage in enumerate(stages) if stage.stage_id == "scene_interpretation"),
        len(stages),
    )
    stages.insert(scene_index, pooled)
    return PipelinePreset(
        preset_id=f"{preset.preset_id}+dense-region-pooling/1",
        description=f"{preset.description} Includes explicit mask-weighted dense region pooling.",
        stages=tuple(stages),
    )


def without_scene_interpretation(preset: PipelinePreset) -> PipelinePreset:
    """Return a derived preset without the independent scene stage.

    Args:
        preset: Preset containing ``scene_interpretation``.

    Returns:
        A separately identified region-only preset.

    Raises:
        ValueError: If ``preset`` has no scene stage.
    """
    stages = tuple(stage for stage in preset.stages if stage.stage_id != "scene_interpretation")
    if len(stages) == len(preset.stages):
        raise ValueError(f"preset {preset.preset_id!r} has no scene_interpretation stage to drop")
    return PipelinePreset(
        preset_id=f"{preset.preset_id}+without-scene/1",
        description=f"{preset.description} Interprets regions only, without the scene.",
        stages=stages,
    )


def _consuming_the_region_set(stage: StageSpec) -> StageSpec:
    """Redirect canonical region consumers to the assembled region evidence."""
    if stage.inputs.get("regions") != "region_discovery":
        return stage
    return replace(stage, inputs={**stage.inputs, "regions": "region_set"})


def _grounded_preset(preset_id: str, description: str, *, refined: bool) -> PipelinePreset:
    """Build the grounded topology from canonical V1 without changing its stages."""
    sources = {"discovered": "region_discovery", "grounded": "region_grounding"}
    added = [
        StageSpec(
            stage_id="region_grounding",
            capability="region_grounding",
            inputs={"image": "image_preparation"},
            backend_id="region_grounding/composed",
            optional=True,
        )
    ]
    if refined:
        added.append(
            StageSpec(
                stage_id="region_refinement",
                capability="region_refinement",
                inputs={"image": "image_preparation", "groundings": "region_grounding"},
                backend_id="region_refinement/composed",
                optional=True,
            )
        )
        sources["refined"] = "region_refinement"
    added.append(
        StageSpec(
            stage_id="region_set",
            capability="region_set_assembly",
            inputs=sources,
            optional=True,
            parameters={"policy_id": REGION_SET_ASSEMBLY_POLICY},
        )
    )
    stages: list[StageSpec] = []
    for stage in CANONICAL_PRESET_V1.stages:
        stages.append(_consuming_the_region_set(stage))
        if stage.stage_id == "region_discovery":
            stages.extend(added)
    return PipelinePreset(preset_id=preset_id, description=description, stages=tuple(stages))


GROUNDED_PRESET_V1 = _grounded_preset(
    "grounded/1",
    "canonical/1 plus prompt-conditioned region grounding: region features and region "
    "interpretation run over the discovered and the grounded regions of each frame "
    f"({REGION_SET_ASSEMBLY_POLICY}).",
    refined=False,
)
"""Canonical V1 plus prompt-conditioned grounding evidence."""

GROUNDED_REFINED_PRESET_V1 = _grounded_preset(
    "grounded-refined/1",
    "canonical/1 plus prompt-conditioned region grounding and grounding-to-mask refinement: "
    "region features and region interpretation run over the discovered, grounded and refined "
    f"regions of each frame ({REGION_SET_ASSEMBLY_POLICY}).",
    refined=True,
)
"""Grounded V1 plus explicit prompted-mask refinement evidence."""
