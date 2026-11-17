"""Topologias versionadas do runtime: estágios, sua fiação e os presets que os selecionam.

Um *estágio* é um nó do DAG; declara a capability dona, os componentes (de ``components``)
para os quais precisa de backend, as entradas tipadas -- cada uma ligada ao estágio que a
produz, pelo nome de contrato de ``contracts`` -- e o contrato que produz. Um *preset* é uma
seleção versionada de estágios; um perfil de mesma identidade parte dele.

Mudar a topologia é uma edição deste módulo e só dele; adicionar um backend é uma edição de
``components``. As duas mudam por razões independentes.

``canonical/2`` é declarado como uma diferença sobre ``canonical/1`` (``_canonical2_stage``),
não como uma cópia literal: a diferença *é* a afirmação científica "canonical/2 difere de
canonical/1 exatamente nestes estágios", e uma declaração expandida a perderia. O preço é um
acoplamento -- mudar ``canonical/1`` muda ``canonical/2`` -- que contradiria a promessa de
nunca mutar silenciosamente uma topologia versionada. Por isso as duas pontas estão fixadas em
``tests/runtime/test_runtime_catalog.py``: um teste fixa a fiação exata de ``canonical/1`` e
outro o conjunto exato de estágios em que ``canonical/2`` difere dele. Qualquer mudança em uma
das duas falha um teste em vez de se propagar em silêncio.

``optional`` tem três sentidos distintos no runtime, e nenhum implica os outros:

- ``StageDeclaration.optional``: a configuração pode desligar o estágio
  (``pipeline.stages.<id> = false``);
- ``StageInput.optional``: o estágio roda sem aquela entrada, que só é ligada quando o
  estágio-fonte participa;
- ``ComponentSpec.optional`` (em ``components``): o estágio compõe sem backend selecionado
  para aquele componente.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from contextmap.runtime.components.contracts import (
    ASSOCIATION,
    CONTEXT_MAP,
    ENTITIES,
    FUSION,
    GEOMETRY,
    REGIONS,
    RELATIONS,
    REPRESENTATION,
    RESOLUTION,
    SEMANTIC_INTERPRETATION,
    SEMANTIC_SCORES,
    SEQUENCE,
    TRAJECTORY,
    VISUAL_EVIDENCE,
)

CANONICAL_PROFILE_ID = "canonical/1"
"""Versioned identity of the canonical Solution 1 topology and profile.

Before v0.1.0 ships, this is the repository's one and only topology, end to end: ingestion
through ``context_map``. There is no released consumer yet to protect from a topology change, so
this identity is free to keep evolving with the pipeline itself until the release. Once v0.1.0
ships, a topology change starts a new versioned identity instead (``canonical/2`` and so on,
never a silent mutation of a published one) -- that discipline begins at the release, not before
it.
"""

CANONICAL2_PROFILE_ID = "canonical/2"
"""Identity of the SAM2-first, context-rich canonical topology.

``canonical/1`` remains the historical baseline.  The second profile makes the
semantic-scoring branch part of its declared topology so that a selected CLIP
or AlphaCLIP scorer is explicit in the run plan; the semantic-score input of
fusion remains optional to preserve a controlled no-score comparison.
"""


@dataclass(frozen=True, kw_only=True)
class StageInput:
    """One typed input of a stage, wired to the stage that produces it.

    Attributes:
        name: Input name, unique inside the stage.
        contract: Identity of the artifact kind the input consumes.
        source: Stage that produces it in the base topology.
        optional: Whether the stage runs without it. An optional input is dropped when
            its source stage does not take part, and wired when it does. This concerns the
            *input's presence* only: it neither lets configuration disable the stage
            (``StageDeclaration.optional``) nor lets a component go without a backend
            (``ComponentSpec.optional``).
        multiple: Whether the stage accepts several runs of its source stage at once, kept
            as distinct evidence (for example repeated perception runs over the same
            observations). Every other input accepts exactly one run.
    """

    name: str
    contract: str
    source: str
    optional: bool = False
    multiple: bool = False


@dataclass(frozen=True, kw_only=True)
class Interception:
    """Where an optional stage inserts itself between a producer and a consumer.

    The intercepting stage consumes what the consumer's input used to read and produces
    the same contract, so the consumer keeps reading its declared contract and never
    learns which stage produced it.

    Attributes:
        consumer: Stage whose input is redirected.
        input_name: The redirected input of ``consumer``.
    """

    consumer: str
    input_name: str


@dataclass(frozen=True, kw_only=True)
class StageDeclaration:
    """One node of a runtime topology.

    Attributes:
        stage_id: Identity of the stage inside its preset.
        capability: Owner capability package name.
        optional: Whether a configuration may disable the stage. This concerns the
            *stage's participation* only: whether one of its inputs may be absent is
            ``StageInput.optional``, and whether one of its components may go without a
            backend is ``ComponentSpec.optional``.
        default_enabled: Whether the stage takes part when configuration is silent.
        available: Whether the capability that owns the stage is implemented.
        unavailable_reason: Why the stage cannot run, when ``available`` is false.
        components: Component identities this stage needs a backend for.
        inputs: Typed inputs, each wired to its producing stage.
        output: Identity of the artifact kind the stage produces.
        intercepts: For an optional stage, the input it inserts itself into.
        observation_scoped: Whether the stage processes only the observations of
            ``inputs.observation_selection``, which then takes part in its configuration
            identity. A stage downstream of it inherits the selection through the
            content of its input, never through configuration.
    """

    stage_id: str
    capability: str
    optional: bool = False
    default_enabled: bool = True
    available: bool = True
    unavailable_reason: str = ""
    components: tuple[str, ...] = ()
    inputs: tuple[StageInput, ...] = ()
    output: str | None = None
    intercepts: Interception | None = None
    observation_scoped: bool = False


@dataclass(frozen=True, kw_only=True)
class RuntimePreset:
    """A versioned, named selection of runtime stages.

    Two presets that select different stages use different ``preset_id``s: a version
    is never reused for a different topology. A preset derived from another (canonical/2
    from canonical/1) is pinned by tests on both ends, so a change to the base cannot
    silently change the derived one.

    Attributes:
        preset_id: Versioned identity, for example ``"canonical/1"``.
        stages: The stages, in canonical documentation order.
        description: What the preset is for.
    """

    preset_id: str
    stages: tuple[StageDeclaration, ...]
    description: str = ""

    def __post_init__(self) -> None:
        """Reject a preset that declares one stage identity twice."""
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError(f"preset {self.preset_id!r} declares a stage twice")

    def stage(self, stage_id: str) -> StageDeclaration:
        """Return one stage of the preset.

        Args:
            stage_id: Identity of the stage.

        Returns:
            The declaration.

        Raises:
            KeyError: If the preset has no such stage.
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


CANONICAL_PRESET = RuntimePreset(
    preset_id=CANONICAL_PROFILE_ID,
    description=(
        "The Solution 1 topology, end to end: from a recorded source to the ContextMapArtifact. "
        "Pre-v0.1.0, this is the repository's only topology and it is free to grow with the "
        "pipeline; a topology change starts a new versioned identity only once the release ships."
    ),
    stages=(
        StageDeclaration(
            stage_id="ingestion",
            capability="ingestion",
            components=("ingestion.source_adapter",),
            output=SEQUENCE,
        ),
        # An opt-in second ingestion stage, publishing an auxiliary, pose-only
        # SequenceArtifact for state_estimation to merge with the main one. No component: its
        # executor is always injected by the caller (see IngestionStageExecutor), same as
        # "ingestion" itself -- an incremental bridge, not general multi-source ingestion.
        StageDeclaration(
            stage_id="pose_ingestion",
            capability="ingestion",
            optional=True,
            default_enabled=False,
            output=SEQUENCE,
        ),
        # A região final é publicada antes das inferências visuais e semânticas.
        StageDeclaration(
            stage_id="region_perception",
            capability="region_perception",
            components=(
                "region_perception.discovery",
                "region_perception.grounding",
                "region_perception.refinement",
            ),
            inputs=(StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),),
            output=REGIONS,
            observation_scoped=True,
        ),
        StageDeclaration(
            stage_id="visual_perception",
            capability="visual_perception",
            components=(
                "visual_features.dense",
                "visual_features.region",
            ),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="regions", contract=REGIONS, source="region_perception"),
            ),
            output=VISUAL_EVIDENCE,
            observation_scoped=True,
        ),
        StageDeclaration(
            stage_id="semantic_interpretation",
            capability="semantic_interpretation",
            components=("semantic_interpretation.interpreter",),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="regions", contract=REGIONS, source="region_perception"),
            ),
            output=SEMANTIC_INTERPRETATION,
            observation_scoped=True,
        ),
        # Pontuação semântica opcional a jusante dos três produtores pinados: cada braço de
        # scorer lê as mesmas regiões, features e claims sem reexecutá-los.
        StageDeclaration(
            stage_id="semantic_scoring",
            capability="semantic_scoring",
            optional=True,
            default_enabled=False,
            components=("semantic_scoring.scorer",),
            inputs=(
                StageInput(name="regions", contract=REGIONS, source="region_perception"),
                StageInput(
                    name="visual_evidence", contract=VISUAL_EVIDENCE, source="visual_perception"
                ),
                StageInput(
                    name="semantic_interpretation",
                    contract=SEMANTIC_INTERPRETATION,
                    source="semantic_interpretation",
                ),
            ),
            output=SEMANTIC_SCORES,
        ),
        StageDeclaration(
            stage_id="state_estimation",
            capability="state_estimation",
            components=("state_estimation.estimator",),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="pose_sequence",
                    contract=SEQUENCE,
                    source="pose_ingestion",
                    optional=True,
                ),
            ),
            output=TRAJECTORY,
        ),
        StageDeclaration(
            stage_id="geometric_mapping",
            capability="geometric_mapping",
            components=(
                "geometric_mapping.pose_lookup",
                "geometric_mapping.motion_correction",
                "geometric_mapping.deskew",
                "geometric_mapping.aggregation",
            ),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="trajectory", contract=TRAJECTORY, source="state_estimation"),
            ),
            output=GEOMETRY,
        ),
        StageDeclaration(
            stage_id="sensor_association",
            capability="sensor_association",
            components=(
                "sensor_association.occlusion",
                "sensor_association.tolerances",
                "sensor_association.pose_policy",
                "sensor_association.dense_sampling",
                "sensor_association.membership",
            ),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="regions",
                    contract=REGIONS,
                    source="region_perception",
                    multiple=True,
                ),
                StageInput(
                    name="visual_evidence",
                    contract=VISUAL_EVIDENCE,
                    source="visual_perception",
                    optional=True,
                    multiple=True,
                ),
                StageInput(name="trajectory", contract=TRAJECTORY, source="state_estimation"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=ASSOCIATION,
        ),
        # Saída derivada e opcional. Os centros vêm das observações da associação, que por isso
        # é uma entrada obrigatória; nenhum estágio consome o run.
        StageDeclaration(
            stage_id="point_representation",
            capability="point_representation",
            optional=True,
            default_enabled=False,
            components=("point_representation.center_selection", "point_representation.encoder"),
            inputs=(
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(name="association", contract=ASSOCIATION, source="sensor_association"),
            ),
            output=REPRESENTATION,
        ),
        # Sem entrada `representation`: consumir o run de Point Representation exige um contrato
        # novo de fusão, ainda inexistente; ligar aquele estágio não muda este.
        StageDeclaration(
            stage_id="semantic_fusion",
            capability="semantic_fusion",
            components=("semantic_fusion.support", "semantic_fusion.accumulation"),
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(
                    name="association",
                    contract=ASSOCIATION,
                    source="sensor_association",
                    multiple=True,
                ),
                StageInput(
                    name="semantic_interpretation",
                    contract=SEMANTIC_INTERPRETATION,
                    source="semantic_interpretation",
                    multiple=True,
                ),
                StageInput(
                    name="regions", contract=REGIONS, source="region_perception", multiple=True
                ),
                StageInput(
                    name="visual_evidence",
                    contract=VISUAL_EVIDENCE,
                    source="visual_perception",
                    optional=True,
                    multiple=True,
                ),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=FUSION,
        ),
        StageDeclaration(
            stage_id="semantic_mapping",
            capability="semantic_mapping",
            components=("semantic_mapping.geometry_summary",),
            inputs=(
                StageInput(name="fusion", contract=FUSION, source="semantic_fusion"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=ENTITIES,
        ),
        StageDeclaration(
            stage_id="entity_resolution",
            capability="entity_resolution",
            components=(
                "entity_resolution.retrieval",
                "entity_resolution.resolution",
                "entity_resolution.geometry_comparison",
                "entity_resolution.semantic_compatibility",
                "entity_resolution.temporal_compatibility",
                "entity_resolution.appearance",
                "entity_resolution.representation",
                "entity_resolution.belief",
            ),
            inputs=(StageInput(name="entities", contract=ENTITIES, source="semantic_mapping"),),
            output=RESOLUTION,
        ),
        StageDeclaration(
            stage_id="spatial_relations",
            capability="spatial_relations",
            components=(
                "spatial_relations.frame_conventions",
                "spatial_relations.candidate",
                "spatial_relations.geometry_summary",
                "spatial_relations.geometric_predicate",
                "spatial_relations.contact_predicate",
                "spatial_relations.semantic_predicate",
            ),
            inputs=(
                StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
            ),
            output=RELATIONS,
        ),
        StageDeclaration(
            stage_id="context_map",
            capability="artifact",
            inputs=(
                StageInput(name="sequence", contract=SEQUENCE, source="ingestion"),
                StageInput(name="geometry", contract=GEOMETRY, source="geometric_mapping"),
                StageInput(name="entities", contract=RESOLUTION, source="entity_resolution"),
                StageInput(name="relations", contract=RELATIONS, source="spatial_relations"),
            ),
            output=CONTEXT_MAP,
        ),
    ),
)


def _canonical2_stage(stage: StageDeclaration) -> StageDeclaration:
    """Return the canonical/2 declaration derived from one baseline stage.

    Every stage not named here is the canonical/1 declaration itself. The pinning tests in
    ``tests/runtime/test_runtime_catalog.py`` fix the set of stages this function changes.
    """
    if stage.stage_id == "semantic_scoring":
        return replace(stage, default_enabled=True)
    if stage.stage_id == "semantic_fusion":
        return replace(
            stage,
            inputs=(
                *stage.inputs,
                StageInput(
                    name="semantic_scores",
                    contract=SEMANTIC_SCORES,
                    source="semantic_scoring",
                    optional=True,
                ),
            ),
        )
    return stage


CANONICAL2_PRESET = RuntimePreset(
    preset_id=CANONICAL2_PROFILE_ID,
    description=(
        "SAM2-first context-rich spatial-memory topology. It retains canonical/1 as an "
        "independent historical baseline and makes semantic compatibility scoring explicit."
    ),
    stages=tuple(_canonical2_stage(stage) for stage in CANONICAL_PRESET.stages),
)

PRESETS: Mapping[str, RuntimePreset] = {
    CANONICAL_PRESET.preset_id: CANONICAL_PRESET,
    CANONICAL2_PRESET.preset_id: CANONICAL2_PRESET,
}
"""Known presets. A profile of the same identity starts from each of them."""
