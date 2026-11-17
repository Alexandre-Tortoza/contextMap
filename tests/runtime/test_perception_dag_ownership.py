"""Regression contracts for independently selectable perception producers."""

from pathlib import Path

from contextmap.runtime.components import CANONICAL_PRESET


def test_regions_own_discovery_grounding_and_refinement() -> None:
    stage = CANONICAL_PRESET.stage("region_perception")
    assert stage.components == (
        "region_perception.discovery",
        "region_perception.grounding",
        "region_perception.refinement",
    )


def test_visual_and_semantic_evidence_have_independent_producers() -> None:
    visual = CANONICAL_PRESET.stage("visual_perception")
    semantic = CANONICAL_PRESET.stage("semantic_interpretation")
    assert visual.components == ("visual_features.dense", "visual_features.region")
    assert visual.output == "VisualEvidenceArtifact"
    assert semantic.components == ("semantic_interpretation.interpreter",)
    assert semantic.output == "SemanticInterpretationArtifact"
    assert {item.source for item in semantic.inputs} == {"ingestion", "region_perception"}
    scoring = CANONICAL_PRESET.stage("semantic_scoring")
    assert {item.source for item in scoring.inputs} == {
        "region_perception",
        "visual_perception",
        "semantic_interpretation",
    }


def test_geometric_association_requires_regions_without_semantic_evidence() -> None:
    association = CANONICAL_PRESET.stage("sensor_association")
    assert "semantic_interpretation" not in {item.source for item in association.inputs}
    assert next(item for item in association.inputs if item.name == "regions").contract == (
        "RegionPerceptionRunArtifact"
    )


def test_association_only_selects_visual_input_for_dense_sampling() -> None:
    from contextmap.runtime import resolve_effective_config, resolve_plan

    geometry_only = resolve_plan(resolve_effective_config())
    association = geometry_only.stage("sensor_association")
    assert "visual_evidence" not in {item.name for item in association.inputs}

    dense = resolve_plan(
        resolve_effective_config(
            overrides=[
                "components.sensor_association.dense_sampling.backend=dense-feature-sampling-v1"
            ]
        )
    )
    association = dense.stage("sensor_association")
    assert "visual_evidence" in {item.name for item in association.inputs}


def test_context_selection_requires_explicit_regional_producer(tmp_path: Path) -> None:
    import pytest
    from runtime_context import ContextWorkspace, foundation

    from contextmap.runtime import ArtifactRef, ContextRunError, context_scope

    _, plan = ContextWorkspace(tmp_path).plan()
    visual = ArtifactRef(
        stage_id="visual_perception",
        contract="VisualEvidenceArtifact",
        artifact_id="visual-A",
        content_hash="sha256:visual-A",
        location="S1/run-0001/visual_perception",
    )
    with pytest.raises(ContextRunError, match="region_perception"):
        context_scope(plan, foundation(), selected={"visual_perception": (visual,)})
