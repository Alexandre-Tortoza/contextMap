"""Factorized experiment phases expanded into resolved runtime configurations (#527).

A phase names what it varies; expansion resolves every arm through the runtime's own
configuration and topology resolution (never composing a backend or loading a model), checks
its wiring against the capability matrix, records blocked and duplicate arms with their
reasons and builds the controlled comparison of the runnable ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from experiment_builders import REGISTRY, make_executor, pinned, validated_reference
from technique_builders import (
    BASELINE_ACCUMULATION_CAPABILITY,
    QUALITY_AWARE_ACCUMULATION_CAPABILITY,
    quality_fusion_phase,
)

from contextmap.evaluation.capability_matrix import (
    CAPABILITY_MATRIX,
    CompositionStatus,
    ImplementationStatus,
)
from contextmap.evaluation.experiment_phases import (
    ArmState,
    Factor,
    FactorLevel,
    MatchedParameters,
    PhaseArm,
    PhaseExpansionError,
    PhaseKind,
    PhaseManifest,
    PhaseSpec,
    expand_phase,
    record_phase_outcomes,
    write_phase_manifest,
)
from contextmap.evaluation.experiment_runner import (
    ArmUnavailableError,
    read_verified_document,
    run_experiment,
)
from contextmap.evaluation.experiments import (
    AblationMode,
    DifferenceKind,
    ExperimentPurpose,
    MetricRef,
    SelectionBinding,
    VariationKind,
    arm_differences,
)
from contextmap.evaluation.metrics import EvaluationStage
from contextmap.evaluation.reference_integrity import ValidatedReferenceSet
from contextmap.runtime import EffectiveConfig, resolve_effective_config
from contextmap.runtime.composition import compose_executors

REGION = "components.region_perception"
FEATURES = "components.visual_features"
SEMANTIC = "components.semantic_interpretation.interpreter"
REVISION = "1" * 40
PROMPTS = {"scene": "scene/v1", "region": "region/v1"}
VIEWS = {"region_views": ["tight_crop"]}
QWEN = {
    f"{SEMANTIC}.backend": "qwen",
    f"{SEMANTIC}.qwen.model": "Qwen/Qwen3-VL-4B-Instruct",
    f"{SEMANTIC}.qwen.revision": REVISION,
    f"{SEMANTIC}.qwen.precision": "bfloat16",
    f"{SEMANTIC}.qwen.max_new_tokens": 256,
    f"{SEMANTIC}.qwen.temperature": 0.0,
    f"{SEMANTIC}.qwen.prompt_policy": PROMPTS,
    f"{SEMANTIC}.qwen.view_policy": VIEWS,
}
EAGLE = {
    f"{SEMANTIC}.backend": "eagle2_5",
    f"{SEMANTIC}.eagle2_5.model": "nvidia/Eagle2.5-8B",
    f"{SEMANTIC}.eagle2_5.revision": "2" * 40,
    f"{SEMANTIC}.eagle2_5.max_new_tokens": 256,
    f"{SEMANTIC}.eagle2_5.temperature": 0.0,
    f"{SEMANTIC}.eagle2_5.max_dynamic_tiles": 6,
    f"{SEMANTIC}.eagle2_5.prompt_policy": PROMPTS,
    f"{SEMANTIC}.eagle2_5.view_policy": VIEWS,
}
DINOV3 = {
    f"{FEATURES}.dense.backend": "dinov3",
    f"{FEATURES}.dense.dinov3.checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    f"{FEATURES}.dense.dinov3.revision": REVISION,
}
SIGLIP2 = {
    f"{FEATURES}.dense.backend": "siglip2",
    f"{FEATURES}.dense.siglip2.checkpoint": "google/siglip2-so400m-patch16-512",
    f"{FEATURES}.dense.siglip2.revision": REVISION,
    f"{FEATURES}.dense.siglip2.input_size": 512,
}
BASE = {
    "components.ingestion.source_adapter.backend": "ros1_bag",
    f"{REGION}.discovery.backend": "sam2",
    f"{REGION}.discovery.sam2.checkpoint": "sam2.1_hiera_tiny.pt",
    **DINOV3,
    f"{FEATURES}.region.backend": "alphaclip",
    f"{FEATURES}.region.alphaclip.model_name": "ViT-L/14",
    **QWEN,
}
BASE_EDGES = (
    ("sam2.automatic_mask_generation", "contextmap2.semantic_view_assembly"),
    ("contextmap2.semantic_view_assembly", "qwen.region_interpretation"),
    ("sam2.automatic_mask_generation", "alphaclip.region_embedding"),
)
LOCATE = {
    f"{REGION}.grounding.backend": "locateanything",
    f"{REGION}.grounding.locateanything.model": "nvidia/LocateAnything-3B",
    f"{REGION}.grounding.locateanything.revision": REVISION,
    f"{REGION}.grounding.locateanything.query_set": {
        "queries": [
            {
                "task": "category_detection",
                "policy_id": "locateanything.category-detection/1",
                "geometry": "box",
                "categories": ["chair", "door"],
            }
        ]
    },
}
REFINE = {
    f"{REGION}.refinement.backend": "sam2",
    f"{REGION}.refinement.sam2.checkpoint": "sam2.1_hiera_tiny.pt",
}
SEQUENCE = pinned("sequence", "sequence-0001")
REGION_RUN = pinned("regions", "region-perception-0001")
"""O run de descoberta de regiões que toda fase da percepção reaproveita."""


def level(value: str, overrides: dict[str, Any] | None = None, **kwargs: Any) -> FactorLevel:
    return FactorLevel(value=value, overrides=tuple((overrides or {}).items()), **kwargs)


def factor(name: str, kind: VariationKind, *levels: FactorLevel) -> Factor:
    return Factor(name=name, kind=kind, levels=levels, baseline=levels[0].value)


PROMPT_FACTOR = factor(
    "prompt_policy",
    VariationKind.POLICY,
    level("region/v1", {f"{SEMANTIC}.qwen.prompt_policy": PROMPTS}),
    level(
        "region-abstention/v1",
        {f"{SEMANTIC}.qwen.prompt_policy": {**PROMPTS, "region": "region-abstention/v1"}},
    ),
)
DENSE_FACTOR = factor(
    "dense_features", VariationKind.BACKEND, level("dinov3", DINOV3), level("siglip2", SIGLIP2)
)
INTERPRETER_FACTOR = factor(
    "semantic_backend",
    VariationKind.BACKEND,
    level(
        "qwen", QWEN, edges=(("contextmap2.semantic_view_assembly", "qwen.region_interpretation"),)
    ),
    level(
        "eagle2_5",
        EAGLE,
        edges=(("contextmap2.semantic_view_assembly", "eagle2_5.region_interpretation"),),
    ),
)
REFINEMENT_FACTOR = factor(
    "refinement",
    VariationKind.TOPOLOGY,
    level("none"),
    level(
        "sam2",
        REFINE,
        edges=(("locateanything.category_detection", "sam2.box_prompt_refinement"),),
    ),
)


@pytest.fixture
def validated(tmp_path: Path) -> ValidatedReferenceSet:
    return validated_reference(tmp_path / "reference")


def phase(validated: ValidatedReferenceSet, **changes: Any) -> PhaseSpec:
    fields: dict[str, Any] = {
        "experiment_id": "perception-ablation",
        "phase_id": "semantic-prompt",
        "kind": PhaseKind.SEMANTIC_PROMPT,
        "version": "1.0.0",
        "description": "controlled phase",
        "purpose": ExperimentPurpose.EVALUATION,
        "evaluated_stage": EvaluationStage.SEMANTIC_INTERPRETATION,
        "selection": SelectionBinding.for_split(validated.manifest, "regions-by-sequence", "test"),
        "base_overrides": tuple(BASE.items()),
        "base_edges": BASE_EDGES,
        "factors": (PROMPT_FACTOR,),
        "targets": ("visual_perception", "semantic_interpretation"),
        # region_perception é um estágio executado a montante de visual_perception, e ele roda
        # só a descoberta: nenhuma fase da percepção a varia, então toda fase pina o run de
        # descoberta. Sem isso cada arm descobriria as suas próprias regiões e a comparação
        # ficaria confundida (#545); pinar é condição de validade, não otimização.
        "pinned": (("ingestion", SEQUENCE), ("region_perception", REGION_RUN)),
        "quality_metrics": (MetricRef(name="semantic.acceptable_claim_rate", version="1"),),
        "resource_capture": (MetricRef(name="runtime.wall_time", version="1"),),
        "registry": REGISTRY.identity(),
    }
    fields.update(changes)
    return PhaseSpec(**fields)


def _arm(manifest: PhaseManifest, value: str) -> Any:
    return next(item for item in manifest.arms if value in dict(item.assignments).values())


# --------------------------------------------------------------------- one-factor phases


def test_a_prompt_phase_resolves_every_arm_and_declares_only_the_prompt_field(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(phase(validated))

    assert [item.arm_id for item in manifest.arms] == ["baseline", "arm-01"]
    assert all(item.state is ArmState.PLANNED for item in manifest.arms)
    assert all(str(item.effective_config_digest).startswith("sha256:") for item in manifest.arms)
    assert len({item.effective_config_digest for item in manifest.arms}) == 2
    experiment = manifest.experiment
    assert experiment is not None
    assert experiment.mode is AblationMode.SELECTED
    (difference,) = arm_differences(experiment.variables, *experiment.arms)
    assert difference.stage_id == "semantic_interpretation.interpreter"
    assert difference.field == "qwen.prompt_policy.region"
    assert difference.declared_by == ("prompt_policy",)
    assert {item.arm_id: item.topology_digest for item in manifest.arms} == {
        arm.arm_id: arm.topology.digest() for arm in experiment.arms
    }


def test_a_backend_substitution_keeps_the_topology_and_changes_only_the_backend(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(
        phase(
            validated,
            phase_id="dense-features",
            kind=PhaseKind.FEATURE_BACKEND,
            factors=(DENSE_FACTOR,),
            evaluated_stage=EvaluationStage.FEATURE_EXTRACTION,
        )
    )

    experiment = manifest.experiment
    assert experiment is not None
    baseline, siglip = experiment.arms
    assert [item.stage_id for item in baseline.topology.stages] == [
        item.stage_id for item in siglip.topology.stages
    ]
    differences = arm_differences(experiment.variables, baseline, siglip)
    assert {item.stage_id for item in differences} == {"visual_features.dense"}
    assert DifferenceKind.BACKEND in {item.kind for item in differences}
    assert "siglip2.dense_patch_features" in _arm(manifest, "siglip2").capabilities


def test_qwen_and_eagle_are_compared_under_a_matched_prompt_and_view_policy(
    validated: ValidatedReferenceSet,
) -> None:
    matched = MatchedParameters(
        component_id="semantic_interpretation.interpreter",
        names=("prompt_policy", "view_policy", "max_new_tokens", "temperature"),
    )
    spec = phase(
        validated,
        phase_id="semantic-interpreter",
        kind=PhaseKind.SEMANTIC_INTERPRETER,
        factors=(INTERPRETER_FACTOR,),
        base_edges=BASE_EDGES[::2],
        matched=(matched,),
    )

    manifest = expand_phase(spec)

    assert _arm(manifest, "eagle2_5").state is ArmState.PLANNED
    assert "eagle2_5.region_interpretation" in _arm(manifest, "eagle2_5").capabilities
    other_views = factor(
        "semantic_backend",
        VariationKind.BACKEND,
        INTERPRETER_FACTOR.levels[0],
        level(
            "eagle2_5",
            {
                **EAGLE,
                f"{SEMANTIC}.eagle2_5.view_policy": {
                    "region_views": ["masked_subject", "tight_crop"],
                    "mask_fill_rgb": [0, 0, 0],
                },
            },
            edges=INTERPRETER_FACTOR.levels[1].edges,
        ),
    )
    with pytest.raises(PhaseExpansionError, match="view_policy") as error:
        expand_phase(replace(spec, factors=(other_views,)))
    assert "masked_subject" in str(error.value)


def test_each_arm_names_its_resolved_semantic_request_policy(
    validated: ValidatedReferenceSet,
) -> None:
    conditioned = {**PROMPTS, "region": "region-scene-context/v1", "region_scene_context": True}
    context = factor(
        "scene_context",
        VariationKind.EVIDENCE_CHANNELS,
        level("none", {f"{SEMANTIC}.qwen.prompt_policy": PROMPTS}),
        level("scene", {f"{SEMANTIC}.qwen.prompt_policy": conditioned}),
    )

    manifest = expand_phase(phase(validated, kind=PhaseKind.SCENE_CONTEXT, factors=(context,)))

    fingerprints = [item.request_policy_fingerprint for item in manifest.arms]
    assert all(str(item).startswith("sha256:") for item in fingerprints)
    assert len(set(fingerprints)) == 2
    experiment = manifest.experiment
    assert experiment is not None
    fields = {item.field for item in arm_differences(experiment.variables, *experiment.arms)}
    assert fields == {"qwen.prompt_policy.region", "qwen.prompt_policy.region_scene_context"}
    # Um template de região que não renderiza contexto de cena é recusado antes de qualquer modelo.
    broken = replace(
        context,
        levels=(
            context.levels[0],
            level(
                "scene", {f"{SEMANTIC}.qwen.prompt_policy": {**conditioned, "region": "region/v1"}}
            ),
        ),
    )
    with pytest.raises(PhaseExpansionError, match="scene context"):
        expand_phase(phase(validated, kind=PhaseKind.SCENE_CONTEXT, factors=(broken,)))


def test_a_backend_substitution_that_changes_the_task_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    florence = level(
        "florence2",
        {
            f"{SEMANTIC}.backend": "florence2",
            f"{SEMANTIC}.florence2.checkpoint": "florence-community/Florence-2-large",
            f"{SEMANTIC}.florence2.revision": REVISION,
            f"{SEMANTIC}.florence2.task": "<REGION_TO_CATEGORY>",
            f"{SEMANTIC}.florence2.supported_modes": ["region"],
            f"{SEMANTIC}.florence2.device": "cuda",
            f"{SEMANTIC}.florence2.precision": "float16",
            f"{SEMANTIC}.florence2.max_new_tokens": 32,
            f"{SEMANTIC}.florence2.temperature": 0.0,
            f"{SEMANTIC}.florence2.view_policy": VIEWS,
        },
        edges=(("contextmap2.semantic_view_assembly", "florence2.region_category"),),
    )

    with pytest.raises(PhaseExpansionError, match="no comparable counterpart"):
        expand_phase(
            phase(
                validated,
                kind=PhaseKind.SEMANTIC_INTERPRETER,
                base_edges=BASE_EDGES[::2],
                factors=(
                    factor(
                        "semantic_backend", VariationKind.BACKEND, level("qwen", QWEN), florence
                    ),
                ),
            )
        )


# ------------------------------------------------------------------ topology and wiring


def test_box_only_grounding_and_grounding_with_refinement_are_a_topology_ablation(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(
        phase(
            validated,
            phase_id="grounding-refinement",
            kind=PhaseKind.GROUNDING_REFINEMENT,
            base_overrides=tuple({**BASE, **LOCATE}.items()),
            factors=(REFINEMENT_FACTOR,),
            pinned=(("ingestion", SEQUENCE),),
            evaluated_stage=EvaluationStage.REGION_DISCOVERY,
        )
    )

    experiment = manifest.experiment
    assert experiment is not None
    box_only, refined = experiment.arms
    assert "region_perception.refinement" not in {
        item.stage_id for item in box_only.topology.stages
    }
    # Os slots regionais são computados no mesmo artifact; o DAG externo depende da sequência.
    assert refined.topology.stage("region_perception.refinement").depends_on == (
        "ingestion.source_adapter",
    )
    # A fiação produtor -> consumidor fica registrada no arm e verificada contra a matriz.
    assert (
        "locateanything.category_detection",
        "sam2.box_prompt_refinement",
    ) in _arm(manifest, "sam2").edges
    differences = arm_differences(experiment.variables, box_only, refined)
    assert {item.stage_id for item in differences} == {
        "region_perception.refinement",
        "visual_features.dense",
        "visual_features.region",
        "semantic_interpretation.interpreter",
    }
    assert sum(item.kind is DifferenceKind.PRESENCE for item in differences) == 1
    assert all(item.declared_by == ("refinement",) for item in differences)
    assert "sam2.box_prompt_refinement" in _arm(manifest, "sam2").capabilities
    assert "locateanything.category_detection" in _arm(manifest, "none").capabilities


def test_a_topology_phase_cannot_hide_a_backend_change(validated: ValidatedReferenceSet) -> None:
    sneaky = replace(
        REFINEMENT_FACTOR,
        levels=(
            REFINEMENT_FACTOR.levels[0],
            replace(
                REFINEMENT_FACTOR.levels[1],
                overrides=(*REFINEMENT_FACTOR.levels[1].overrides, *SIGLIP2.items()),
            ),
        ),
    )

    with pytest.raises(PhaseExpansionError, match="undeclared change"):
        expand_phase(
            phase(
                validated,
                kind=PhaseKind.GROUNDING_REFINEMENT,
                base_overrides=tuple({**BASE, **LOCATE}.items()),
                factors=(sneaky,),
            )
        )


@pytest.mark.parametrize(
    ("edge", "reason"),
    [
        (
            ("locateanything.category_detection", "contextmap2.mask_membership_association"),
            "not in the arm",
        ),
        (("locateanything.category_detection", "alphaclip.region_embedding"), "incompatible"),
        (("dinov3.dense_patch_features", "qwen.region_interpretation"), "undeclared composition"),
        (("florence2.object_detection", "clip.region_embedding"), "not in the arm"),
    ],
)
def test_invalid_wiring_fails_preflight(
    validated: ValidatedReferenceSet, edge: tuple[str, str], reason: str
) -> None:
    with pytest.raises(PhaseExpansionError, match=reason):
        expand_phase(
            phase(
                validated,
                base_overrides=tuple({**BASE, **LOCATE}.items()),
                base_edges=(*BASE_EDGES, edge),
            )
        )


# ------------------------------------------------- grounding and 2D->3D association (#649)

MEMBERSHIP = "contextmap2.mask_membership_association"
GROUNDING = f"{REGION}.grounding.locateanything"
OCCLUSION = "components.sensor_association.occlusion.conservative-depth-support-v1"
ASSOCIATION = {
    "components.state_estimation.estimator.backend": "external_pose",
    "components.geometric_mapping.pose_lookup.backend": "lookup-policy-v1",
    "components.geometric_mapping.motion_correction.backend": "motion-correction-v1",
    "components.sensor_association.occlusion.backend": "conservative-depth-support-v1",
    "components.sensor_association.tolerances.backend": "diagnostic-tolerances-v1",
    "components.sensor_association.pose_policy.backend": "lookup-policy-v1",
}
GROUNDING_QUERIES = {
    "locateanything.category_detection": LOCATE[f"{GROUNDING}.query_set"],
    "locateanything.phrase_grounding": {
        "queries": [
            {
                "task": "phrase_grounding",
                "policy_id": "locateanything.phrase-grounding/1",
                "geometry": "box",
                "text": "the red chair",
            }
        ]
    },
    "locateanything.pointing": {
        "queries": [
            {
                "task": "phrase_grounding",
                "policy_id": "locateanything.pointing/1",
                "geometry": "point",
                "text": "the red chair",
            }
        ]
    },
}
DEPTH_MARGIN = factor(
    "depth_margin",
    VariationKind.CONFIGURATION,
    level("narrow", {f"{OCCLUSION}.depth_margin_m": 0.1}),
    level("wide", {f"{OCCLUSION}.depth_margin_m": 0.2}),
)
UPSTREAM = tuple(
    (stage, pinned(stage, f"{stage}-0001"))
    for stage in (
        "ingestion",
        "region_perception",
        "visual_perception",
        "semantic_interpretation",
        "state_estimation",
        "geometric_mapping",
    )
)


def association_phase(
    validated: ValidatedReferenceSet,
    region_source: dict[str, Any],
    edges: tuple[tuple[str, str], ...] = (),
) -> PhaseSpec:
    """A downstream confirmation measured at 2D->3D association over one pinned perception run."""
    return phase(
        validated,
        phase_id="association-confirmation",
        kind=PhaseKind.DOWNSTREAM_CONFIRMATION,
        evaluated_stage=EvaluationStage.SENSOR_ASSOCIATION,
        base_overrides=tuple({**BASE, **ASSOCIATION, **region_source}.items()),
        base_edges=(*BASE_EDGES, *edges),
        factors=(DEPTH_MARGIN,),
        compositions=((("depth_margin", "wide"),),),
        targets=("sensor_association", "visual_perception", "semantic_interpretation"),
        pinned=UPSTREAM,
        quality_metrics=(MetricRef(name="association.visible_support.ratio", version="1"),),
    )


def test_a_discovery_only_phase_reaches_association(validated: ValidatedReferenceSet) -> None:
    manifest = expand_phase(association_phase(validated, {}))

    assert [item.state for item in manifest.arms] == [ArmState.PLANNED, ArmState.PLANNED]
    assert MEMBERSHIP in manifest.arms[0].capabilities


@pytest.mark.parametrize("grounding", GROUNDING_QUERIES)
def test_unrefined_grounding_never_reaches_association_as_a_disguised_reference(
    validated: ValidatedReferenceSet, grounding: str
) -> None:
    # Sem refino, as caixas (ou pontos) do grounding não sustentam nenhum ponto 3D: a associação
    # do arm teria só as regiões da descoberta (P0) sob outro digest, que a deduplicação não vê.
    region_source = {**LOCATE, f"{GROUNDING}.query_set": GROUNDING_QUERIES[grounding]}

    with pytest.raises(PhaseExpansionError, match="#649") as refused:
        expand_phase(association_phase(validated, region_source))
    assert f"{grounding}->{MEMBERSHIP}" in str(refused.value)


def test_declaring_direct_grounding_association_fails_the_phase_with_the_decision(
    validated: ValidatedReferenceSet,
) -> None:
    # #577 P1 (composição B do #528): o arm declara a aresta grounding -> associação direta.
    p1 = association_phase(
        validated, LOCATE, edges=(("locateanything.category_detection", MEMBERSHIP),)
    )

    with pytest.raises(PhaseExpansionError, match="incompatible") as refused:
        expand_phase(p1)
    assert "#649" in str(refused.value)
    assert "#528 P1" not in str(refused.value)


def test_grounding_refined_into_masks_reaches_association(
    validated: ValidatedReferenceSet,
) -> None:
    # #577 P2: o único caminho do grounding até a associação passa pelo refino em máscara.
    p2 = association_phase(
        validated,
        {**LOCATE, **REFINE},
        edges=(
            ("locateanything.category_detection", "sam2.box_prompt_refinement"),
            ("sam2.box_prompt_refinement", MEMBERSHIP),
        ),
    )

    manifest = expand_phase(p2)

    assert [item.state for item in manifest.arms] == [ArmState.PLANNED, ArmState.PLANNED]
    assert {"locateanything.category_detection", "sam2.box_prompt_refinement", MEMBERSHIP} <= set(
        manifest.arms[0].capabilities
    )


# ------------------------------- a variation point varied inside the perception run (#660)

DISCOVERY = "region_perception.discovery"
P2_LEVEL = level(
    "grounding+refinement",
    {**LOCATE, **REFINE},
    edges=(
        ("locateanything.category_detection", "sam2.box_prompt_refinement"),
        ("sam2.box_prompt_refinement", MEMBERSHIP),
    ),
)


def region_source_phase(
    validated: ValidatedReferenceSet,
    treatment: FactorLevel = P2_LEVEL,
    *,
    pinned_stages: tuple[str, ...] = (
        "ingestion",
        "state_estimation",
        "geometric_mapping",
    ),
) -> PhaseSpec:
    """P0 (discovery only) x a grounding topology, measured at 2D->3D association."""
    return phase(
        validated,
        phase_id="region-source-association",
        kind=PhaseKind.GROUNDING_REFINEMENT,
        evaluated_stage=EvaluationStage.SENSOR_ASSOCIATION,
        base_overrides=tuple({**BASE, **ASSOCIATION}.items()),
        factors=(factor("region_source", VariationKind.TOPOLOGY, level("discovery"), treatment),),
        targets=("sensor_association", "visual_perception", "semantic_interpretation"),
        pinned=tuple((stage, pinned(stage, f"{stage}-0001")) for stage in pinned_stages),
        quality_metrics=(MetricRef(name="association.visible_support.ratio", version="1"),),
    )


def test_p0_and_p2_measured_at_association_pin_discovery_not_the_perception_run(
    validated: ValidatedReferenceSet,
) -> None:
    manifest = expand_phase(region_source_phase(validated))

    assert [item.state for item in manifest.arms] == [ArmState.PLANNED, ArmState.PLANNED]
    experiment = manifest.experiment
    assert experiment is not None
    p0, p2 = (arm.topology.stage(DISCOVERY) for arm in experiment.arms)
    # Discovery tem a mesma configuração; grounding/refinement variam dentro do produtor
    # regional. Cada braço publica outro artifact regional, então discovery é fixado por
    # identidade de implementação e nunca por um artifact regional compartilhado.
    assert p0 == p2
    assert p0.artifact is None
    assert p0.execution == "region_perception"
    assert experiment.arms[1].topology.stage("region_perception.refinement").execution == (
        "region_perception"
    )


def test_a_perception_run_varied_inside_cannot_be_a_reused_artifact(
    validated: ValidatedReferenceSet,
) -> None:
    # O grounding e o refino rodam no run de percepção: fixar o estágio inteiro faria o arm P2
    # reaproveitar o run de percepção do P0, sem grounding.
    pinned_run = region_source_phase(
        validated,
        pinned_stages=(
            "ingestion",
            "region_perception",
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
        ),
    )

    with pytest.raises(PhaseExpansionError, match="one computation"):
        expand_phase(pinned_run)


def test_varying_discovery_inside_a_grounding_topology_is_still_undeclared(
    validated: ValidatedReferenceSet,
) -> None:
    sneaky = replace(
        P2_LEVEL,
        overrides=(*P2_LEVEL.overrides, (f"{REGION}.discovery.sam2.checkpoint", "sam2.1_l.pt")),
    )

    with pytest.raises(PhaseExpansionError, match=f"undeclared change.*{DISCOVERY}"):
        expand_phase(region_source_phase(validated, sneaky))


def test_an_unvaried_perception_run_upstream_of_association_must_still_be_pinned(
    validated: ValidatedReferenceSet,
) -> None:
    unpinned = replace(
        association_phase(validated, {}),
        pinned=tuple(item for item in UPSTREAM if item[0] != "region_perception"),
    )

    with pytest.raises(PhaseExpansionError, match="pinned immutable artifact"):
        expand_phase(unpinned)


def test_p0_and_p1_measured_at_association_stay_refused_by_the_decision(
    validated: ValidatedReferenceSet,
) -> None:
    with pytest.raises(PhaseExpansionError, match="#649"):
        expand_phase(region_source_phase(validated, level("grounding", LOCATE)))


def test_a_selection_the_matrix_does_not_know_fails_preflight(
    validated: ValidatedReferenceSet,
) -> None:
    automatic_sam3 = {
        f"{REGION}.discovery.backend": "sam3",
        f"{REGION}.discovery.sam3.checkpoint": "facebook/sam3",
        f"{REGION}.discovery.sam3.strategy": "automatic",
    }

    with pytest.raises(PhaseExpansionError, match="no capability of the matrix"):
        expand_phase(phase(validated, base_overrides=tuple({**BASE, **automatic_sam3}.items())))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({f"{REGION}.discovery.backend": "sam9"}, "sam9"),
        ({f"{REGION}.discovery.backend": None}, "no backend selected"),
    ],
)
def test_an_arm_that_does_not_resolve_fails_preflight(
    validated: ValidatedReferenceSet, overrides: dict[str, Any], reason: str
) -> None:
    with pytest.raises(PhaseExpansionError, match=reason):
        expand_phase(phase(validated, base_overrides=tuple({**BASE, **overrides}.items())))


# ------------------------------------ declared out of the campaign by decision (#656, #657)

FEATURE_RESOLUTION = "contextmap2.feature_resolution_enhancement"
POINT_REPRESENTATION = "contextmap2.point_representation"
ENCODER = "components.point_representation.encoder"
FUSION_SUPPORT = "components.semantic_fusion.support"
FUSION = {
    f"{FUSION_SUPPORT}.backend": "geometry-jaccard-support-v1",
    "components.semantic_fusion.accumulation.backend": "baseline-evidence-accumulation-v1",
}
ENCODERS = ("geometric_descriptor", "ptv3")
REPRESENTATION_ON = {
    "pipeline.stages.point_representation": True,
    "components.point_representation.center_selection.backend": "observation-support-anchor-v1",
    f"{ENCODER}.backend": ENCODERS[0],
}
REPRESENTATION_FACTOR = factor(
    "point_representation",
    VariationKind.TOPOLOGY,
    level("off"),
    *(level(item, {**REPRESENTATION_ON, f"{ENCODER}.backend": item}) for item in ENCODERS),
)
ENCODER_FACTOR = factor(
    "point_encoder",
    VariationKind.BACKEND,
    *(level(item, {f"{ENCODER}.backend": item}) for item in ENCODERS),
)
MIN_OVERLAP = factor(
    "min_overlap",
    VariationKind.CONFIGURATION,
    *(
        level(str(value), {f"{FUSION_SUPPORT}.geometry-jaccard-support-v1.min_overlap": value})
        for value in (0.7, 0.8)
    ),
)
ENTITY_RESOLUTION = "components.entity_resolution"
ENTITY_REPRESENTATION = "contextmap2.entity_representation_comparison"
ENTITY_BASE = {
    f"{ENTITY_RESOLUTION}.retrieval.backend": "entity-candidate-retrieval-v1",
    f"{ENTITY_RESOLUTION}.resolution.backend": "conservative-staged-resolution-v1",
    f"{ENTITY_RESOLUTION}.geometry_comparison.backend": "entity-geometry-comparison-v1",
    "components.semantic_mapping.geometry_summary.backend": "entity-geometry-summary-v1",
}


def entity_evidence(channel: str, backend: str) -> Factor:
    """Entity Resolution on geometry only x with one more evidence channel (#180)."""
    return factor(
        "entity_evidence",
        VariationKind.TOPOLOGY,
        level("geometry-only"),
        level(f"with-{channel}", {f"{ENTITY_RESOLUTION}.{channel}.backend": backend}),
    )


# Estágio avaliado, métrica e estágios a montante pinados, por alvo.
TARGETS: dict[str, tuple[EvaluationStage, str, tuple[str, ...]]] = {
    "sensor_association": (
        EvaluationStage.SENSOR_ASSOCIATION,
        "association.visible_support.ratio",
        (),
    ),
    "point_representation": (
        EvaluationStage.POINT_REPRESENTATION,
        "pointrep.repeatability.cosine",
        ("sensor_association",),
    ),
    "semantic_fusion": (
        EvaluationStage.SEMANTIC_FUSION,
        "fusion.reference_recovery.rate",
        ("sensor_association",),
    ),
    "entity_resolution": (
        EvaluationStage.ENTITY_RESOLUTION,
        "entity.false_merge.rate",
        ("sensor_association", "semantic_fusion", "semantic_mapping"),
    ),
}


def downstream_phase(
    validated: ValidatedReferenceSet,
    treatment: Factor,
    target: str = "sensor_association",
    base: dict[str, Any] | None = None,
) -> PhaseSpec:
    """A #180 downstream confirmation of an optional technique, with its upstream pinned."""
    stage, metric, pinned_below = TARGETS[target]
    return phase(
        validated,
        phase_id="optional-technique-confirmation",
        kind=PhaseKind.DOWNSTREAM_CONFIRMATION,
        evaluated_stage=stage,
        base_overrides=tuple({**BASE, **ASSOCIATION, **FUSION, **(base or {})}.items()),
        factors=(treatment,),
        compositions=tuple(
            ((treatment.name, item.value),)
            for item in treatment.levels
            if item.value != treatment.baseline
        ),
        targets=(target, "visual_perception", "semantic_interpretation"),
        pinned=(*UPSTREAM, *((item, pinned(item, f"{item}-0001")) for item in pinned_below)),
        quality_metrics=(MetricRef(name=metric, version="1"),),
    )


def test_native_versus_enhanced_dense_features_is_out_of_the_campaign(
    validated: ValidatedReferenceSet,
) -> None:
    # #197 A2 / #180: o mesmo artifact DINO passado por FeatureResolutionEnhancement. Sem backend
    # concreto nem estágio de runtime, a #656 tirou o braço da campanha: a fase é recusada.
    enhancement = factor(
        "feature_resolution",
        VariationKind.TOPOLOGY,
        level("native"),
        level("enhanced", capabilities=(FEATURE_RESOLUTION,)),
    )

    with pytest.raises(PhaseExpansionError, match="#656") as refused:
        expand_phase(downstream_phase(validated, enhancement))
    assert FEATURE_RESOLUTION in str(refused.value)
    assert "out of scope" in str(refused.value)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ("pipeline.stages.feature_resolution_enhancement", "unknown stage"),
        (f"{FEATURES}.feature_resolution_enhancement.backend", "unknown slot"),
    ],
)
def test_the_runtime_has_no_feature_resolution_enhancement_to_select(
    validated: ValidatedReferenceSet, override: str, reason: str
) -> None:
    # Guarda: o runtime não tem estágio nem slot de enhancement, então um braço que tente
    # selecioná-lo pela configuração não resolve; nunca sai `planned`.
    enhancement = factor(
        "feature_resolution", VariationKind.TOPOLOGY, level("native"), level("on", {override: True})
    )

    with pytest.raises(PhaseExpansionError, match=reason):
        expand_phase(downstream_phase(validated, enhancement))


def test_point_representation_through_the_final_map_is_out_of_the_campaign(
    validated: ValidatedReferenceSet,
) -> None:
    # #180 "Point Representation off vs deterministic vs PTv3": o estágio roda (#663), mas a
    # fusão não tem entrada `representation` (#657), então nenhum alvo a jusante depende dele. O
    # fator não toca estágio algum da fase e ela é recusada (#545): nenhum braço sai `planned`
    # como se a representação chegasse ao mapa.
    with pytest.raises(PhaseExpansionError, match="must touch at least one stage"):
        expand_phase(downstream_phase(validated, REPRESENTATION_FACTOR, "semantic_fusion"))


def test_point_representation_measured_at_its_own_stage_is_out_of_the_campaign(
    validated: ValidatedReferenceSet,
) -> None:
    # Nem a comparação só do estágio (descritor determinístico x PTv3) entra: o runtime já
    # produz o run (#663), mas no piloto o estágio é um controle fixo em todo braço, não uma
    # variável, e a #657 mantém a comparação fora da campanha.
    encoders = downstream_phase(
        validated, ENCODER_FACTOR, "point_representation", base=REPRESENTATION_ON
    )

    with pytest.raises(PhaseExpansionError, match="#657"):
        expand_phase(encoders)


def test_entity_resolution_with_3d_representation_evidence_is_out_of_the_campaign(
    validated: ValidatedReferenceSet,
) -> None:
    # #180 "Entity Resolution geometry-only vs 3D evidence": o canal lê vetores de um run de
    # Point Representation que o runtime não produz e que o estágio não recebe como entrada; só
    # um provider manual, fora do DAG, os entregaria. Sem a recusa, os dois braços saíam
    # `planned` sem motivo; a decisão da #657 cobre também esta comparação.
    channel = entity_evidence("representation", "entity-representation-comparison-v1")

    with pytest.raises(PhaseExpansionError, match="#657") as refused:
        expand_phase(downstream_phase(validated, channel, "entity_resolution", base=ENTITY_BASE))
    assert ENTITY_REPRESENTATION in str(refused.value)
    assert "out of scope" in str(refused.value)


def test_entity_resolution_with_semantic_evidence_stays_in_the_campaign(
    validated: ValidatedReferenceSet,
) -> None:
    # Só a evidência 3D sai da campanha: os outros canais da Entity Resolution seguem.
    channel = entity_evidence("semantic_compatibility", "entity-semantic-compatibility-v1")

    manifest = expand_phase(
        downstream_phase(validated, channel, "entity_resolution", base=ENTITY_BASE)
    )

    assert [item.state for item in manifest.arms] == [ArmState.PLANNED, ArmState.PLANNED]
    assert ENTITY_REPRESENTATION not in manifest.arms[1].capabilities


def test_enabling_point_representation_that_no_target_consumes_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    # Guarda: com alvo na associação, o estágio ligado fica fora do arm; o fator não toca
    # nenhum estágio e a fase é recusada (#545), então nenhum braço sai `planned` com ele ligado.
    with pytest.raises(PhaseExpansionError, match="must touch at least one stage"):
        expand_phase(downstream_phase(validated, REPRESENTATION_FACTOR))


def test_a_downstream_phase_without_point_representation_is_untouched(
    validated: ValidatedReferenceSet,
) -> None:
    # O estágio fica desligado no canônico: sem ligá-lo, a fase a jusante segue normalmente.
    manifest = expand_phase(downstream_phase(validated, MIN_OVERLAP, "semantic_fusion"))

    assert [item.state for item in manifest.arms] == [ArmState.PLANNED, ArmState.PLANNED]
    assert "contextmap2.semantic_fusion" in manifest.arms[0].capabilities
    assert POINT_REPRESENTATION not in manifest.arms[0].capabilities


# ------------------------------------------------------- blocked, duplicate and selected


SCORING = "components.semantic_scoring.scorer"
SEMANTIC_RUN = pinned("semantic_interpretation_run", "semantic-run-0001")
PERCEPTION = pinned("visual_evidence_run", "visual-run-0001")
ALPHACLIP_SCORER = {
    "pipeline.stages.semantic_scoring": True,
    f"{SCORING}.backend": "alphaclip",
    f"{SCORING}.alphaclip.model_name": "ViT-L/14",
    f"{SCORING}.alphaclip.base_checkpoint_path": "base.pt",
    f"{SCORING}.alphaclip.alpha_checkpoint_path": "alpha.pth",
    f"{SCORING}.alphaclip.checkpoint_fingerprint": "sha256:" + "a" * 64,
    f"{SCORING}.alphaclip.text_template": "{hypothesis}",
}


def test_a_scoring_off_versus_alphaclip_phase_expands_both_arms_over_one_perception_run(
    validated: ValidatedReferenceSet,
) -> None:
    """#652: the scorer arm is no longer blocked, and it never re-runs the perception stage.

    Scoring is a stage of its own, so switching it on inserts a stage: a topology factor, which
    only a composition phase varies. Every stage upstream of it must be pinned, so both arms
    provably share one perception run and its VLM claims (#580).
    """
    scorer = factor(
        "semantic_scoring",
        VariationKind.TOPOLOGY,
        level("disabled"),
        level(
            "alphaclip",
            ALPHACLIP_SCORER,
            edges=(
                ("qwen.region_interpretation", "alphaclip.semantic_scoring"),
                ("alphaclip.region_embedding", "alphaclip.semantic_scoring"),
            ),
        ),
    )

    manifest = expand_phase(
        phase(
            validated,
            phase_id="semantic-scoring",
            kind=PhaseKind.PIPELINE_COMPOSITION,
            factors=(scorer,),
            compositions=((("semantic_scoring", "alphaclip"),),),
            targets=("visual_perception", "semantic_interpretation", "semantic_scoring"),
            pinned=(
                ("ingestion", SEQUENCE),
                ("region_perception", REGION_RUN),
                ("visual_perception", PERCEPTION),
                ("semantic_interpretation", SEMANTIC_RUN),
            ),
        )
    )

    enabled = _arm(manifest, "alphaclip")
    assert enabled.state is ArmState.PLANNED
    assert "alphaclip.semantic_scoring" in enabled.capabilities
    assert _arm(manifest, "disabled").state is ArmState.PLANNED
    experiment = manifest.experiment
    assert experiment is not None
    baseline, scored = experiment.arms
    scorer_node = "semantic_scoring.scorer"
    # A percepção é o mesmo artifact pinado nos dois braços: o scorer é um nó a mais, nunca uma
    # mudança nela.
    assert [node for node in scored.topology.stages if node.stage_id != scorer_node] == list(
        baseline.topology.stages
    )
    assert all(
        node.artifact == PERCEPTION
        for node in baseline.topology.stages
        if node.execution == "visual_perception"
    )
    assert scorer_node not in {node.stage_id for node in baseline.topology.stages}
    assert scorer_node in {node.stage_id for node in scored.topology.stages}


def test_a_target_no_arm_produces_is_refused(validated: ValidatedReferenceSet) -> None:
    """A disabled optional target is legal in one arm, never in every arm of the phase."""
    spec = phase(
        validated, targets=("visual_perception", "semantic_interpretation", "semantic_scoring")
    )

    with pytest.raises(PhaseExpansionError, match="no arm produces target stages"):
        expand_phase(spec)


def test_an_arm_that_disables_every_target_produces_nothing_and_is_refused(
    validated: ValidatedReferenceSet,
) -> None:
    scorer = factor(
        "semantic_scoring",
        VariationKind.TOPOLOGY,
        level("disabled"),
        level("alphaclip", ALPHACLIP_SCORER),
    )
    spec = phase(
        validated,
        phase_id="semantic-scoring",
        kind=PhaseKind.PIPELINE_COMPOSITION,
        factors=(scorer,),
        compositions=((("semantic_scoring", "alphaclip"),),),
        targets=("semantic_scoring",),
        pinned=(
            ("ingestion", SEQUENCE),
            ("region_perception", REGION_RUN),
            ("visual_perception", PERCEPTION),
            ("semantic_interpretation", SEMANTIC_RUN),
        ),
    )

    with pytest.raises(PhaseExpansionError, match="disables every target stage"):
        expand_phase(spec)


@pytest.mark.parametrize(
    ("capability", "issue"),
    [
        ("locateanything.visual_prompt_grounding", 574),
        ("locateanything3d.open_vocabulary_3d_proposal", 575),
    ],
)
def test_arms_that_need_unpublished_assets_stay_blocked(
    validated: ValidatedReferenceSet, capability: str, issue: int
) -> None:
    blocked = factor(
        "region_source",
        VariationKind.BACKEND,
        level("discovery"),
        level("unpublished", capabilities=(capability,)),
    )

    manifest = expand_phase(phase(validated, kind=PhaseKind.BACKEND_GRID, factors=(blocked,)))

    arm = _arm(manifest, "unpublished")
    assert arm.state is ArmState.BLOCKED
    assert arm.issues == (issue,)
    assert f"#{issue}" in str(arm.reason)


def test_equivalent_effective_configurations_deduplicate_deterministically(
    validated: ValidatedReferenceSet,
) -> None:
    budget = factor(
        "visual_budget",
        VariationKind.CONFIGURATION,
        level("unbounded"),
        level(
            "small", {f"{SEMANTIC}.qwen.min_pixels": 200704, f"{SEMANTIC}.qwen.max_pixels": 401408}
        ),
        level(
            "small-again",
            {f"{SEMANTIC}.qwen.max_pixels": 401408, f"{SEMANTIC}.qwen.min_pixels": 200704},
        ),
    )
    spec = phase(validated, kind=PhaseKind.VISUAL_BUDGET, factors=(budget,))

    first = expand_phase(spec)
    second = expand_phase(spec)

    duplicate = _arm(first, "small-again")
    assert duplicate.state is ArmState.SKIPPED
    assert duplicate.duplicate_of == _arm(first, "small").arm_id
    assert duplicate.effective_config_digest == _arm(first, "small").effective_config_digest
    assert first.digest() == second.digest()
    assert first.experiment is not None
    assert [item.arm_id for item in first.experiment.arms] == ["baseline", "arm-01"]


def test_a_composition_phase_expands_only_the_selected_compositions(
    validated: ValidatedReferenceSet,
) -> None:
    grounding = factor(
        "grounding",
        VariationKind.TOPOLOGY,
        level("none"),
        level("locateanything", LOCATE),
    )
    spec = phase(
        validated,
        phase_id="compositions",
        kind=PhaseKind.PIPELINE_COMPOSITION,
        pinned=(("ingestion", SEQUENCE),),
        factors=(grounding, REFINEMENT_FACTOR, DENSE_FACTOR, INTERPRETER_FACTOR),
        base_edges=BASE_EDGES[::2],
        compositions=(
            (
                ("grounding", "locateanything"),
                ("refinement", "sam2"),
                ("dense_features", "dinov3"),
                ("semantic_backend", "qwen"),
            ),
            (
                ("grounding", "locateanything"),
                ("refinement", "sam2"),
                ("dense_features", "siglip2"),
                ("semantic_backend", "eagle2_5"),
            ),
        ),
    )

    manifest = expand_phase(spec)

    assert len(manifest.arms) == 3  # baseline + 2 selected, not the 16 of the product
    assert all(item.state is ArmState.PLANNED for item in manifest.arms)
    with pytest.raises(PhaseExpansionError, match="assign"):
        expand_phase(replace(spec, compositions=((("grounding", "locateanything"),),)))
    with pytest.raises(PhaseExpansionError, match="selected compositions"):
        expand_phase(replace(spec, compositions=()))


def test_a_phase_varies_only_the_kind_of_factor_it_is_named_for(
    validated: ValidatedReferenceSet,
) -> None:
    with pytest.raises(PhaseExpansionError, match="topology"):
        expand_phase(phase(validated, factors=(REFINEMENT_FACTOR,)))
    with pytest.raises(PhaseExpansionError, match="exactly one factor"):
        expand_phase(phase(validated, factors=(PROMPT_FACTOR, replace(PROMPT_FACTOR, name="x"))))
    assert len(PhaseKind) == 11


def test_expansion_imports_no_backend_and_no_model_runtime() -> None:
    script = (
        "import sys, json\n"
        "sys.path.insert(0, 'tests/evaluation')\n"
        "from contextmap.evaluation.experiment_phases import expand_phase\n"
        "import test_experiment_phases as t\n"
        "from experiment_builders import reference_with_semantics\n"
        "ref = reference_with_semantics()\n"
        "class V: manifest = ref\n"
        "expand_phase(t.phase(V))\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in {'torch', 'transformers'}"
        " or m.startswith('contextmap.visual_perception.adapters.')]\n"
        "print(json.dumps(heavy))\n"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=True
    )

    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# --------------------------------------------- Semantic Fusion accumulation policy (#655)


def _arm_effective(spec: PhaseSpec, arm: PhaseArm) -> EffectiveConfig:
    """Resolve an arm's configuration again, exactly as expansion resolved it."""
    overrides = (
        *spec.base_overrides,
        *(
            item
            for name, value in arm.assignments
            for item in spec.factor(name).level(value).overrides
        ),
    )
    effective = resolve_effective_config(
        overrides=[f"{path}={json.dumps(value)}" for path, value in overrides]
    )
    assert effective.digest == arm.effective_config_digest
    return effective


def test_a_quality_aware_fusion_arm_is_planned_because_the_runtime_runs_it(
    validated: ValidatedReferenceSet,
) -> None:
    spec = quality_fusion_phase(validated.manifest)

    manifest = expand_phase(spec)

    uniform, aware = manifest.arms
    assert [uniform.state, aware.state] == [ArmState.PLANNED, ArmState.PLANNED]
    assert BASELINE_ACCUMULATION_CAPABILITY in uniform.capabilities
    assert QUALITY_AWARE_ACCUMULATION_CAPABILITY in aware.capabilities
    assert (MEMBERSHIP, QUALITY_AWARE_ACCUMULATION_CAPABILITY) in aware.edges
    assert manifest.experiment is not None
    # Planejado quer dizer executável: a composition root monta o estágio de fusão dos dois arms,
    # com a identidade do código que o run de fusão grava (#659).
    for arm in (uniform, aware):
        executors = compose_executors(
            _arm_effective(spec, arm),
            module_available=lambda _name: True,
            environ={},
            code_version="commit-1",
        )
        assert "semantic_fusion" in executors, arm.arm_id


def test_an_accumulation_the_runtime_cannot_run_blocks_its_arm_with_the_reason(
    validated: ValidatedReferenceSet,
) -> None:
    unavailable = replace(
        CAPABILITY_MATRIX.capability(QUALITY_AWARE_ACCUMULATION_CAPABILITY),
        status=ImplementationStatus.PLANNED,
        status_reason="no executor runs this accumulation",
        issue=655,
    )
    matrix = replace(
        CAPABILITY_MATRIX,
        capabilities=tuple(
            unavailable if item.capability_id == unavailable.capability_id else item
            for item in CAPABILITY_MATRIX.capabilities
        ),
        compositions=tuple(
            replace(item, status=CompositionStatus.PLANNED, issue=655)
            if item.consumer == unavailable.capability_id
            else item
            for item in CAPABILITY_MATRIX.compositions
        ),
    )

    manifest = expand_phase(quality_fusion_phase(validated.manifest), matrix=matrix)

    aware = _arm(manifest, "quality_aware")
    assert aware.state is ArmState.BLOCKED
    assert 655 in aware.issues
    assert "no executor runs this accumulation" in str(aware.reason)


# ------------------------------------------------------------------ outcomes, persistence


def test_failed_oom_and_skipped_arms_are_preserved_with_blocked_ones(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    budget = factor(
        "visual_budget",
        VariationKind.CONFIGURATION,
        level("unbounded"),
        level("small", {f"{SEMANTIC}.qwen.max_pixels": 401408, f"{SEMANTIC}.qwen.min_pixels": 1}),
        level("large", {f"{SEMANTIC}.qwen.max_pixels": 4014080, f"{SEMANTIC}.qwen.min_pixels": 1}),
        level("unpublished", capabilities=("locateanything.visual_prompt_grounding",)),
    )
    manifest = expand_phase(phase(validated, kind=PhaseKind.VISUAL_BUDGET, factors=(budget,)))
    experiment = manifest.experiment
    assert experiment is not None
    values = {arm.arm_id: {"semantic.acceptable_claim_rate": 0.5} for arm in experiment.arms}
    executor, _ = make_executor(
        values,
        failures={
            _arm(manifest, "small").arm_id: ArmUnavailableError("no GPU was free"),
            _arm(manifest, "large").arm_id: RuntimeError("CUDA out of memory"),
        },
    )
    run = run_experiment(
        experiment,
        executor=executor,
        registry=REGISTRY,
        reference_set=validated,
        root=tmp_path / "run",
    )

    recorded = record_phase_outcomes(manifest, run)

    states = {dict(item.assignments)["visual_budget"]: item for item in recorded.arms}
    assert states["unbounded"].state is ArmState.EXECUTED
    assert states["small"].state is ArmState.SKIPPED
    assert "no GPU was free" in str(states["small"].reason)
    assert states["large"].state is ArmState.FAILED
    assert "CUDA out of memory" in str(states["large"].reason)
    assert states["unpublished"].state is ArmState.BLOCKED
    assert recorded.digest() != manifest.digest()
    with pytest.raises(PhaseExpansionError, match="another experiment"):
        record_phase_outcomes(replace(manifest, experiment=None), run)


def test_the_phase_manifest_persists_what_reproduces_every_arm(
    validated: ValidatedReferenceSet, tmp_path: Path
) -> None:
    manifest = expand_phase(phase(validated))

    write_phase_manifest(tmp_path, manifest)

    document = read_verified_document(tmp_path / "phase.json")
    assert document["schema"] == "contextmap.experiment-phase/v1"
    assert (document["experiment_id"], document["phase_id"], document["kind"]) == (
        "perception-ablation",
        "semantic-prompt",
        "semantic_prompt",
    )
    assert document["base_configuration_digest"].startswith("sha256:")
    assert document["selection"]["reference_set"] == validated.manifest.identity().to_record()
    assert document["matrix_version"] == CAPABILITY_MATRIX.version
    assert (
        document["factors"][0]["levels"][1]["overrides"][0][0] == f"{SEMANTIC}.qwen.prompt_policy"
    )
    # O run de regiões entra no registro congelado: sem ele a fase não fixaria as regiões que os
    # dois braços descrevem, e o documento tem de dizer qual artifact imutável é reaproveitado.
    assert document["pinned"] == [
        ["ingestion", SEQUENCE.to_record()],
        ["region_perception", REGION_RUN.to_record()],
    ]
    assert document["targets"] == ["visual_perception", "semantic_interpretation"]
    assert {item["state"] for item in document["arms"]} == {"planned"}
    assert all(item["effective_config_digest"] for item in document["arms"])
    assert document["experiment"]["digest"] == manifest.experiment.digest()  # type: ignore[union-attr]
    with pytest.raises(FileExistsError):
        write_phase_manifest(tmp_path, manifest)
