"""Guard the static runtime catalog against drifting from the capabilities it names."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from contextmap.runtime import (
    CANONICAL_PROFILE_ID,
    check_selection,
    resolve_effective_config,
)
from contextmap.runtime.components import (
    CANONICAL2_PRESET,
    CANONICAL2_PROFILE_ID,
    CANONICAL_PRESET,
    COMPONENTS,
    PRESETS,
    RuntimePreset,
    StageDeclaration,
    StageInput,
)
from contextmap.semantic_fusion import (
    BASELINE_ACCUMULATION_POLICY_ID,
    GEOMETRY_OVERLAP_SUPPORT_POLICY_ID,
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
    EvidenceChannel,
)


def _package_exists(capability: str) -> bool:
    return importlib.util.find_spec(f"contextmap.{capability}") is not None


class TestCatalogAgainstCapabilities:
    def test_canonical_2_is_a_distinct_sam2_first_profile(self) -> None:
        """The published baseline must not be silently repurposed for canonical/2."""
        assert CANONICAL2_PROFILE_ID == "canonical/2"
        assert PRESETS[CANONICAL2_PROFILE_ID] is CANONICAL2_PRESET

        scoring = CANONICAL2_PRESET.stage("semantic_scoring")
        assert scoring.default_enabled

        fusion = CANONICAL2_PRESET.stage("semantic_fusion")
        assert [(item.name, item.contract, item.optional) for item in fusion.inputs][-1] == (
            "semantic_scores",
            "SemanticScoringRunArtifact",
            True,
        )

    def test_fusion_policy_identities_are_the_ones_the_capability_versions(self) -> None:
        assert set(COMPONENTS["semantic_fusion.support"].backends) == {
            GEOMETRY_OVERLAP_SUPPORT_POLICY_ID
        }
        assert set(COMPONENTS["semantic_fusion.accumulation"].backends) == {
            BASELINE_ACCUMULATION_POLICY_ID,
            QUALITY_AWARE_ACCUMULATION_POLICY_ID,
        }

    def test_the_dense_sampling_policy_identity_is_the_one_the_capability_versions(self) -> None:
        from contextmap.sensor_association.dense_sampling import SAMPLING_POLICY_ID

        assert set(COMPONENTS["sensor_association.dense_sampling"].backends) == {SAMPLING_POLICY_ID}

    def test_the_center_selection_policy_identity_is_the_one_the_capability_versions(
        self,
    ) -> None:
        from contextmap.point_representation import OBSERVATION_SUPPORT_ANCHOR_POLICY_ID

        assert set(COMPONENTS["point_representation.center_selection"].backends) == {
            OBSERVATION_SUPPORT_ANCHOR_POLICY_ID
        }

    def test_the_channel_the_configuration_refuses_by_name_still_exists(self) -> None:
        # A configuração recusa o canal pelo nome: o nome tem de continuar o da fusão.
        assert EvidenceChannel.POINT_REPRESENTATION.value == "point_representation"

    def test_entity_resolution_policy_identities_are_the_ones_the_capability_versions(
        self,
    ) -> None:
        from contextmap.entity_resolution import (
            APPEARANCE_COMPARISON_POLICY_ID,
            CANDIDATE_RETRIEVAL_POLICY_ID,
            CONSERVATIVE_RESOLUTION_POLICY_ID,
            GEOMETRY_COMPARISON_POLICY_ID,
            REPRESENTATION_COMPARISON_POLICY_ID,
            SEMANTIC_COMPATIBILITY_POLICY_ID,
            TEMPORAL_COMPATIBILITY_POLICY_ID,
        )

        assert set(COMPONENTS["entity_resolution.retrieval"].backends) == {
            CANDIDATE_RETRIEVAL_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.resolution"].backends) == {
            CONSERVATIVE_RESOLUTION_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.geometry_comparison"].backends) == {
            GEOMETRY_COMPARISON_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.semantic_compatibility"].backends) == {
            SEMANTIC_COMPATIBILITY_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.temporal_compatibility"].backends) == {
            TEMPORAL_COMPATIBILITY_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.appearance"].backends) == {
            APPEARANCE_COMPARISON_POLICY_ID
        }
        assert set(COMPONENTS["entity_resolution.representation"].backends) == {
            REPRESENTATION_COMPARISON_POLICY_ID
        }

    def test_spatial_relations_policy_identities_are_the_ones_the_capability_versions(
        self,
    ) -> None:
        from contextmap.semantic_mapping import GEOMETRY_SUMMARY_ALGORITHM_ID
        from contextmap.spatial_relations import (
            CANDIDATE_POLICY_ID,
            CONTACT_POLICY_ID,
            FRAME_CONVENTIONS_POLICY_ID,
            GEOMETRIC_POLICY_ID,
            SEMANTIC_POLICY_ID,
        )

        assert set(COMPONENTS["spatial_relations.frame_conventions"].backends) == {
            FRAME_CONVENTIONS_POLICY_ID
        }
        assert set(COMPONENTS["spatial_relations.candidate"].backends) == {CANDIDATE_POLICY_ID}
        assert set(COMPONENTS["spatial_relations.geometry_summary"].backends) == {
            GEOMETRY_SUMMARY_ALGORITHM_ID
        }
        assert set(COMPONENTS["spatial_relations.geometric_predicate"].backends) == {
            GEOMETRIC_POLICY_ID
        }
        assert set(COMPONENTS["spatial_relations.contact_predicate"].backends) == {
            CONTACT_POLICY_ID
        }
        assert set(COMPONENTS["spatial_relations.semantic_predicate"].backends) == {
            SEMANTIC_POLICY_ID
        }

    def test_only_the_genuinely_optional_evidence_channels_are_marked_optional(self) -> None:
        optional = {component_id for component_id, spec in COMPONENTS.items() if spec.optional}

        assert optional == {
            # Grounding por prompt é evidência adicional de Visual Perception: sem ele,
            # o estágio roda exatamente como antes.
            "region_perception.grounding",
            # O refinamento por máscara é um estágio de evidência separado e ablável.
            "region_perception.refinement",
            # A amostragem densa na associação é um canal de evidência opcional: sem ela,
            # a associação é a geometry-only de sempre.
            "sensor_association.dense_sampling",
            # A regra por caixa (box-depth-component-v1) só existe para regiões sem máscara: sem
            # ela, a associação mantém a regra de máscara e a caixa é pulada.
            "sensor_association.membership",
            # O deskew corrige o movimento da plataforma dentro do scan: sem ele, cada scan é
            # mapeado com a pose do seu timestamp, exatamente como antes.
            "geometric_mapping.deskew",
            # Sem agregação, o mapa guarda todo ponto bruto, como antes.
            "geometric_mapping.aggregation",
            "entity_resolution.semantic_compatibility",
            "entity_resolution.temporal_compatibility",
            "entity_resolution.appearance",
            "entity_resolution.representation",
            # A crença pós-resolução é estado derivado opcional: sem ela, a resolução é a de
            # sempre, sem status de suporte nem primária.
            "entity_resolution.belief",
            "spatial_relations.geometric_predicate",
            "spatial_relations.contact_predicate",
            # O canal semântico (within_interaction_range) é evidência opcional derivada do raio
            # de interação por entidade: sem ele, as relações são as geométricas de sempre.
            "spatial_relations.semantic_predicate",
        }

    def test_every_available_stage_names_an_implemented_capability(self) -> None:
        for stage in CANONICAL_PRESET.stages:
            if stage.available:
                assert _package_exists(stage.capability), stage.stage_id

    def test_every_unavailable_stage_names_a_capability_that_is_still_missing(self) -> None:
        # Quando a capability passar a existir, o catálogo precisa ser atualizado junto.
        for stage in CANONICAL_PRESET.stages:
            if not stage.available:
                assert not _package_exists(stage.capability), (
                    f"{stage.capability} now exists: mark stage {stage.stage_id!r} available"
                )
                assert stage.unavailable_reason

    def test_every_component_belongs_to_exactly_one_stage_named_by_its_own_capability(
        self,
    ) -> None:
        """A component runs in exactly one stage, which is not necessarily its capability's.

        A component's capability names **who owns the concept**; the stage names **where it
        executes**, and the two are deliberately distinct
        (``docs/perception-dag-decomposition.md``). The catalog has declared components of
        another capability ever since ``visual_features`` and ``semantic_interpretation`` began
        executing in the ``visual_perception`` stage, so requiring the two to be equal asserted
        the very conflation that decision rejects.

        What still holds: a component's id is prefixed by the capability that owns it, no
        component executes in two stages, and every declared component executes in one.
        """
        owners: dict[str, list[str]] = {}
        for stage in CANONICAL_PRESET.stages:
            for component_id in stage.components:
                assert component_id in COMPONENTS
                # O prefixo do id é o dono do conceito; o stage em que ele entra é outra coisa.
                assert component_id.startswith(f"{COMPONENTS[component_id].capability}.")
                owners.setdefault(component_id, []).append(stage.stage_id)
        assert set(owners) == set(COMPONENTS)
        assert all(len(stages) == 1 for stages in owners.values())
        # O estágio regional publica a geometria final; visual e semântico são produtores
        # independentes sobre esse mesmo artifact selecionado.
        assert CANONICAL_PRESET.stage("region_perception").components == (
            "region_perception.discovery",
            "region_perception.grounding",
            "region_perception.refinement",
        )
        assert CANONICAL_PRESET.stage("visual_perception").components == (
            "visual_features.dense",
            "visual_features.region",
        )
        assert CANONICAL_PRESET.stage("semantic_interpretation").components == (
            "semantic_interpretation.interpreter",
        )

    def test_only_the_optional_stages_are_optional_and_every_one_is_off_by_default(
        self,
    ) -> None:
        optional = [stage.stage_id for stage in CANONICAL_PRESET.stages if stage.optional]

        assert optional == ["pose_ingestion", "semantic_scoring", "point_representation"]
        assert not CANONICAL_PRESET.stage("point_representation").default_enabled
        assert not CANONICAL_PRESET.stage("pose_ingestion").default_enabled
        assert not CANONICAL_PRESET.stage("semantic_scoring").default_enabled

    def test_semantic_scoring_reads_pinned_producers_and_feeds_nothing_yet(
        self,
    ) -> None:
        """Scorer arms reuse the same regional, visual, and semantic producer artifacts."""
        stage = CANONICAL_PRESET.stage("semantic_scoring")

        # A pontuação semântica é capability própria desde a extração: o estágio a nomeia.
        assert stage.capability == "semantic_scoring"
        assert stage.components == ("semantic_scoring.scorer",)
        assert [
            (item.name, item.contract, item.source, item.multiple) for item in stage.inputs
        ] == [
            ("regions", "RegionPerceptionRunArtifact", "region_perception", False),
            ("visual_evidence", "VisualEvidenceArtifact", "visual_perception", False),
            (
                "semantic_interpretation",
                "SemanticInterpretationArtifact",
                "semantic_interpretation",
                False,
            ),
        ]
        assert stage.output == "SemanticScoringRunArtifact"
        # O escopo de observações vem do run de percepção lido, nunca da configuração.
        assert not stage.observation_scoped
        assert set(COMPONENTS["semantic_scoring.scorer"].backends) == {
            "clip",
            "alphaclip",
        }
        assert not COMPONENTS["semantic_scoring.scorer"].optional
        # Nenhuma política de fusão consome SemanticScore ainda.
        consumers = [
            item.source
            for other in CANONICAL_PRESET.stages
            for item in other.inputs
            if item.contract == "SemanticScoringRunArtifact"
        ]
        assert consumers == []

    def test_every_input_is_wired_to_a_stage_that_produces_its_contract(self) -> None:
        stages = {stage.stage_id: stage for stage in CANONICAL_PRESET.stages}

        for stage in CANONICAL_PRESET.stages:
            for item in stage.inputs:
                assert item.source in stages, f"{stage.stage_id}.{item.name}"
                assert stages[item.source].output == item.contract, f"{stage.stage_id}.{item.name}"

    def test_every_stage_declares_what_it_produces(self) -> None:
        assert all(stage.output for stage in CANONICAL_PRESET.stages)

    def test_the_canonical_topology_resolves_without_structural_problems(self) -> None:
        from contextmap.runtime import resolve_effective_config, resolve_plan

        plan = resolve_plan(resolve_effective_config())

        assert plan.problems == ()
        assert plan.order is not None
        assert plan.order[-1] == "context_map"

    def test_the_canonical_profile_is_a_known_preset(self) -> None:
        assert PRESETS[CANONICAL_PROFILE_ID] is CANONICAL_PRESET

    def test_an_unknown_stage_lookup_fails_loudly(self) -> None:
        with pytest.raises(KeyError):
            CANONICAL_PRESET.stage("nope")

    def test_the_canonical_topology_is_the_whole_solution_1_pipeline(self) -> None:
        """Pre-v0.1.0 there is one topology, end to end (see ``CANONICAL_PROFILE_ID``'s own
        docstring for why this identity is free to keep evolving until the release)."""
        assert CANONICAL_PROFILE_ID == "canonical/1"
        assert [stage.stage_id for stage in CANONICAL_PRESET.stages] == [
            "ingestion",
            "pose_ingestion",
            # `region_perception` é obrigatório e produz as regiões que a percepção consome.
            "region_perception",
            "visual_perception",
            "semantic_interpretation",
            "semantic_scoring",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "point_representation",
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        ]
        assert CANONICAL_PRESET.stages[-1].output == "ContextMapArtifact"


_SEQ = "SequenceArtifact"
_REG = "RegionPerceptionRunArtifact"
_VIS = "VisualEvidenceArtifact"
_INT = "SemanticInterpretationArtifact"
_TRJ = "StateEstimationRunArtifact"
_GEO = "GeometricMapArtifact"
_ASC = "SensorAssociationRunArtifact"
_FUS = "SemanticFusionRunArtifact"
_RES = "EntityResolutionRunArtifact"
_REL = "SpatialRelationsRunArtifact"

# A fiação inteira de canonical/1, escrita à mão com strings literais: qualquer mudança na
# topologia (estágio, componente, entrada, contrato, opcionalidade) precisa editar este teste.
_CANONICAL1_WIRING: dict[str, dict[str, Any]] = {
    "ingestion": {
        "capability": "ingestion",
        "optional": False,
        "default_enabled": True,
        "components": ("ingestion.source_adapter",),
        "inputs": (),
        "output": _SEQ,
        "observation_scoped": False,
    },
    "pose_ingestion": {
        "capability": "ingestion",
        "optional": True,
        "default_enabled": False,
        "components": (),
        "inputs": (),
        "output": _SEQ,
        "observation_scoped": False,
    },
    "region_perception": {
        "capability": "region_perception",
        "optional": False,
        "default_enabled": True,
        "components": (
            "region_perception.discovery",
            "region_perception.grounding",
            "region_perception.refinement",
        ),
        "inputs": (("sequence", _SEQ, "ingestion", False, False),),
        "output": _REG,
        "observation_scoped": True,
    },
    "visual_perception": {
        "capability": "visual_perception",
        "optional": False,
        "default_enabled": True,
        "components": ("visual_features.dense", "visual_features.region"),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("regions", _REG, "region_perception", False, False),
        ),
        "output": _VIS,
        "observation_scoped": True,
    },
    "semantic_interpretation": {
        "capability": "semantic_interpretation",
        "optional": False,
        "default_enabled": True,
        "components": ("semantic_interpretation.interpreter",),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("regions", _REG, "region_perception", False, False),
        ),
        "output": _INT,
        "observation_scoped": True,
    },
    "semantic_scoring": {
        "capability": "semantic_scoring",
        "optional": True,
        "default_enabled": False,
        "components": ("semantic_scoring.scorer",),
        "inputs": (
            ("regions", _REG, "region_perception", False, False),
            ("visual_evidence", _VIS, "visual_perception", False, False),
            ("semantic_interpretation", _INT, "semantic_interpretation", False, False),
        ),
        "output": "SemanticScoringRunArtifact",
        "observation_scoped": False,
    },
    "state_estimation": {
        "capability": "state_estimation",
        "optional": False,
        "default_enabled": True,
        "components": ("state_estimation.estimator",),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("pose_sequence", _SEQ, "pose_ingestion", True, False),
        ),
        "output": _TRJ,
        "observation_scoped": False,
    },
    "geometric_mapping": {
        "capability": "geometric_mapping",
        "optional": False,
        "default_enabled": True,
        "components": (
            "geometric_mapping.pose_lookup",
            "geometric_mapping.motion_correction",
            "geometric_mapping.deskew",
            "geometric_mapping.aggregation",
        ),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("trajectory", _TRJ, "state_estimation", False, False),
        ),
        "output": _GEO,
        "observation_scoped": False,
    },
    "sensor_association": {
        "capability": "sensor_association",
        "optional": False,
        "default_enabled": True,
        "components": (
            "sensor_association.occlusion",
            "sensor_association.tolerances",
            "sensor_association.pose_policy",
            "sensor_association.dense_sampling",
            "sensor_association.membership",
        ),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("regions", _REG, "region_perception", False, True),
            ("visual_evidence", _VIS, "visual_perception", True, True),
            ("trajectory", _TRJ, "state_estimation", False, False),
            ("geometry", _GEO, "geometric_mapping", False, False),
        ),
        "output": _ASC,
        "observation_scoped": False,
    },
    "point_representation": {
        "capability": "point_representation",
        "optional": True,
        "default_enabled": False,
        "components": ("point_representation.center_selection", "point_representation.encoder"),
        "inputs": (
            ("geometry", _GEO, "geometric_mapping", False, False),
            ("association", _ASC, "sensor_association", False, False),
        ),
        "output": "PointRepresentationRunArtifact",
        "observation_scoped": False,
    },
    "semantic_fusion": {
        "capability": "semantic_fusion",
        "optional": False,
        "default_enabled": True,
        "components": ("semantic_fusion.support", "semantic_fusion.accumulation"),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("association", _ASC, "sensor_association", False, True),
            ("semantic_interpretation", _INT, "semantic_interpretation", False, True),
            ("regions", _REG, "region_perception", False, True),
            ("visual_evidence", _VIS, "visual_perception", True, True),
            ("geometry", _GEO, "geometric_mapping", False, False),
        ),
        "output": _FUS,
        "observation_scoped": False,
    },
    "semantic_mapping": {
        "capability": "semantic_mapping",
        "optional": False,
        "default_enabled": True,
        "components": ("semantic_mapping.geometry_summary",),
        "inputs": (
            ("fusion", _FUS, "semantic_fusion", False, False),
            ("geometry", _GEO, "geometric_mapping", False, False),
        ),
        "output": "SemanticEntityArtifact",
        "observation_scoped": False,
    },
    "entity_resolution": {
        "capability": "entity_resolution",
        "optional": False,
        "default_enabled": True,
        "components": (
            "entity_resolution.retrieval",
            "entity_resolution.resolution",
            "entity_resolution.geometry_comparison",
            "entity_resolution.semantic_compatibility",
            "entity_resolution.temporal_compatibility",
            "entity_resolution.appearance",
            "entity_resolution.representation",
            "entity_resolution.belief",
        ),
        "inputs": (("entities", "SemanticEntityArtifact", "semantic_mapping", False, False),),
        "output": _RES,
        "observation_scoped": False,
    },
    "spatial_relations": {
        "capability": "spatial_relations",
        "optional": False,
        "default_enabled": True,
        "components": (
            "spatial_relations.frame_conventions",
            "spatial_relations.candidate",
            "spatial_relations.geometry_summary",
            "spatial_relations.geometric_predicate",
            "spatial_relations.contact_predicate",
            "spatial_relations.semantic_predicate",
        ),
        "inputs": (
            ("entities", _RES, "entity_resolution", False, False),
            ("geometry", _GEO, "geometric_mapping", False, False),
        ),
        "output": _REL,
        "observation_scoped": False,
    },
    "context_map": {
        "capability": "artifact",
        "optional": False,
        "default_enabled": True,
        "components": (),
        "inputs": (
            ("sequence", _SEQ, "ingestion", False, False),
            ("geometry", _GEO, "geometric_mapping", False, False),
            ("entities", _RES, "entity_resolution", False, False),
            ("relations", _REL, "spatial_relations", False, False),
        ),
        "output": "ContextMapArtifact",
        "observation_scoped": False,
    },
}


def _wiring(stage: StageDeclaration) -> dict[str, Any]:
    """Describe every field of a stage declaration with plain values, for exact comparison."""
    return {
        "capability": stage.capability,
        "optional": stage.optional,
        "default_enabled": stage.default_enabled,
        "components": stage.components,
        "inputs": tuple(
            (item.name, item.contract, item.source, item.optional, item.multiple)
            for item in stage.inputs
        ),
        "output": stage.output,
        "observation_scoped": stage.observation_scoped,
    }


def _differing_stages(left: RuntimePreset, right: RuntimePreset) -> set[str]:
    return {stage.stage_id for stage in left.stages if stage != right.stage(stage.stage_id)}


class TestPresetTopologyIsPinned:
    """canonical/2 is declared as a diff over canonical/1; these pins keep that diff explicit.

    A change to canonical/1 would otherwise flow silently into canonical/2.
    """

    def test_canonical_1_declares_exactly_this_topology(self) -> None:
        stages = CANONICAL_PRESET.stages

        assert [stage.stage_id for stage in stages] == list(_CANONICAL1_WIRING)
        assert {stage.stage_id: _wiring(stage) for stage in stages} == _CANONICAL1_WIRING
        assert all(stage.available and stage.intercepts is None for stage in stages)

    def test_canonical_2_differs_from_canonical_1_in_exactly_two_stages(self) -> None:
        baseline, context_rich = CANONICAL_PRESET, CANONICAL2_PRESET

        assert [stage.stage_id for stage in context_rich.stages] == [
            stage.stage_id for stage in baseline.stages
        ]
        assert _differing_stages(context_rich, baseline) == {
            "semantic_scoring",
            "semantic_fusion",
        }
        # semantic_scoring: só passa a participar por padrão.
        scoring = context_rich.stage("semantic_scoring")
        assert scoring.default_enabled
        assert dataclasses.replace(scoring, default_enabled=False) == baseline.stage(
            "semantic_scoring"
        )
        # semantic_fusion: só ganha, ao fim, a entrada opcional de scores semânticos.
        fusion = context_rich.stage("semantic_fusion")
        assert fusion.inputs[-1] == StageInput(
            name="semantic_scores",
            contract="SemanticScoringRunArtifact",
            source="semantic_scoring",
            optional=True,
        )
        assert dataclasses.replace(fusion, inputs=fusion.inputs[:-1]) == baseline.stage(
            "semantic_fusion"
        )


class TestRealisticConfigurationFile:
    def test_a_toml_configuration_selects_backends_and_enables_the_optional_stage(
        self, tmp_path: Path
    ) -> None:
        file = tmp_path / "experiment.toml"
        file.write_text(
            """
[pipeline.stages]
point_representation = true

[resources]
device = "cuda"
workspace = "workspace/run-a"

[inputs]
sequence = "seq-01"
[inputs.selections]
state_estimation = "seq-01--run-0003"

[policies]
debug_level = "standard"

[components.region_perception.discovery]
backend = "sam3"
[components.region_perception.discovery.sam3]
checkpoint = "sam3-x"
score_threshold = 0.25

[components.point_representation.encoder]
backend = "geometric_descriptor"
""",
            encoding="utf-8",
        )

        config = resolve_effective_config(files=[file]).config

        assert config.pipeline.stages["point_representation"] is True
        assert config.components["point_representation.encoder"].backend == "geometric_descriptor"
        assert config.components["region_perception.discovery"].parameters == {
            "checkpoint": "sam3-x",
            "score_threshold": 0.25,
            "device": "cuda",
        }
        assert config.inputs.sequence == "seq-01"
        assert config.inputs.selections == {"state_estimation": ("seq-01--run-0003",)}
        assert config.resources.workspace == "workspace/run-a"
        assert config.policies.debug_level == "standard"

    def test_enabling_the_optional_stage_makes_its_backend_a_required_choice(self) -> None:
        disabled = check_selection(resolve_effective_config().config)
        enabled = check_selection(
            resolve_effective_config(overrides=["pipeline.stages.point_representation=true"]).config
        )

        assert not any("point_representation" in problem.path for problem in disabled)
        assert any("point_representation.encoder" in problem.path for problem in enabled)

    def test_enabling_semantic_scoring_makes_its_scorer_a_required_choice(self) -> None:
        disabled = check_selection(resolve_effective_config().config)
        enabled = check_selection(
            resolve_effective_config(overrides=["pipeline.stages.semantic_scoring=true"]).config
        )

        assert not any("semantic_scoring" in problem.path for problem in disabled)
        assert any("semantic_scoring.scorer" in problem.path for problem in enabled)
