"""Registro estático dos pontos de variação e dos backends que o runtime pode selecionar.

Este módulo é dado puro: nomeia cada *componente* -- um ponto de variação de uma capability,
identificado como ``"<capability>.<slot>"`` (por exemplo
``"region_perception.discovery"``) -- e os backends que podem preenchê-lo, com os fatos
de implantação de cada um: módulos opcionais que o código empacotado importa, a dica de
instalação, os segredos de ambiente e o parâmetro que recebe o dispositivo. Não importa
backend, não carrega modelo e não decide valor científico: parâmetro, limiar e checkpoint
pertencem à capability dona e à configuração do usuário, nunca a esta tabela.

Adicionar um backend é uma edição deste módulo e só dele; mudar a topologia é uma edição de
``presets``. Os dois mudam por razões independentes.

Dívida de propriedade: muitos ``backend_id`` aqui são identidades de política que a capability
dona já versiona, duplicadas como literais. O certo a longo prazo é cada capability declarar
os próprios pontos de variação e o runtime agregá-los, mas uma capability não pode importar o
runtime (``tests/architecture/test_boundaries.py``) e este módulo não importa capability
nenhuma (``tests/architecture/test_runtime_boundaries.py``). Enquanto isso,
``tests/runtime/test_runtime_catalog.py`` compara cada literal com a constante da dona, e cada
grupo duplicado carrega um ``FIXME`` nomeando a dona.

``optional`` em :class:`ComponentSpec` significa "o estágio compõe sem backend selecionado
para este componente". Não confundir com ``StageDeclaration.optional`` (a configuração pode
desligar o estágio) nem com ``StageInput.optional`` (a entrada pode estar ausente), ambos em
``presets``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_ROSBAGS_HINT = "pip install 'contextmap[ros1]'"


@dataclass(frozen=True, kw_only=True)
class BackendSpec:
    """What one selectable backend or policy needs in order to run.

    Attributes:
        backend_id: Identity used in configuration. For a policy it is the policy
            identity the owning capability already versions.
        requires: Top-level optional modules the backend's *bundled* code imports when
            it is constructed. A backend whose model runtime the caller supplies lists
            none: that runtime's modules are the caller's concern. They are only looked
            up, never imported, by discovery.
        secrets: Names of the environment variables that carry credentials this
            backend needs. Values are never part of configuration.
        device_parameter: Name of the backend parameter that receives
            ``resources.device`` when the backend does not set it explicitly, or
            ``None`` when the backend has no device notion.
        install_hint: How to obtain ``requires``, shown when a module is missing.
    """

    backend_id: str
    requires: tuple[str, ...] = ()
    secrets: tuple[str, ...] = ()
    device_parameter: str | None = None
    install_hint: str = ""


@dataclass(frozen=True, kw_only=True)
class ComponentSpec:
    """One variation point of a capability and the backends that can fill it.

    Attributes:
        capability: Owner capability package name.
        slot: Name of the variation point inside the capability.
        backends: Selectable backends keyed by ``backend_id``.
        optional: Whether a stage that owns this component still composes with no backend
            selected for it. It marks a genuinely optional variation point (for example one
            evidence channel of a decision that evaluates several independently): its absence
            is a valid, explicit configuration, never a default standing in for the missing
            choice. ``False`` for every variation point a stage always needs to run. This is
            about the *backend selection* only: it says nothing about whether the stage itself
            may be disabled (``StageDeclaration.optional``) or whether one of its inputs may be
            absent (``StageInput.optional``).
    """

    capability: str
    slot: str
    backends: Mapping[str, BackendSpec]
    optional: bool = False

    @property
    def component_id(self) -> str:
        """Return the ``"<capability>.<slot>"`` identity used across the runtime."""
        return f"{self.capability}.{self.slot}"


def _component(
    capability: str, slot: str, *backends: BackendSpec, optional: bool = False
) -> ComponentSpec:
    """Build one component, keying its backends by ``backend_id``."""
    return ComponentSpec(
        capability=capability,
        slot=slot,
        backends={backend.backend_id: backend for backend in backends},
        optional=optional,
    )


_COMPONENT_LIST: tuple[ComponentSpec, ...] = (
    _component(
        "ingestion",
        "source_adapter",
        BackendSpec(backend_id="ros1_bag", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
        BackendSpec(backend_id="ros2_bag", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
        BackendSpec(backend_id="pose_file"),
    ),
    _component(
        "region_perception",
        "discovery",
        BackendSpec(backend_id="sam2", device_parameter="device"),
        BackendSpec(backend_id="sam3", device_parameter="device"),
        BackendSpec(backend_id="florence2", device_parameter="device"),
    ),
    _component(
        "region_perception",
        "grounding",
        BackendSpec(
            backend_id="locateanything",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        optional=True,
    ),
    _component(
        "region_perception",
        "refinement",
        BackendSpec(backend_id="sam2", device_parameter="device"),
        optional=True,
    ),
    _component(
        "visual_features",
        "dense",
        BackendSpec(
            backend_id="dinov2",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        BackendSpec(
            backend_id="dinov3",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        BackendSpec(
            backend_id="siglip2",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
    ),
    _component(
        "visual_features",
        "region",
        BackendSpec(
            backend_id="clip",
            requires=("torch", "transformers", "PIL"),
            device_parameter="device",
        ),
        BackendSpec(
            backend_id="alphaclip",
            requires=("torch", "alpha_clip", "PIL"),
            device_parameter="device",
        ),
    ),
    _component(
        "semantic_interpretation",
        "interpreter",
        BackendSpec(backend_id="qwen", device_parameter="device"),
        BackendSpec(backend_id="gemini", secrets=("GEMINI_API_KEY",)),
        BackendSpec(backend_id="florence2", device_parameter="device"),
        BackendSpec(backend_id="eagle2_5", device_parameter="device"),
    ),
    # O scorer lê claims e features dos produtores pinados; o texto do CLIP tem loader
    # empacotado, o do AlphaCLIP vem do chamador (pesos locais).
    _component(
        "semantic_scoring",
        "scorer",
        BackendSpec(
            backend_id="clip",
            requires=("torch", "transformers"),
            device_parameter="device",
        ),
        BackendSpec(backend_id="alphaclip", device_parameter="device"),
    ),
    _component(
        "state_estimation",
        "estimator",
        BackendSpec(backend_id="external_pose"),
        BackendSpec(backend_id="fast_lio", requires=("rosbags",), install_hint=_ROSBAGS_HINT),
    ),
    # FIXME: `lookup-policy-v1` é a LookupPolicy versionada por state_estimation; as demais
    # identidades deste bloco são de geometric_mapping. Duplicadas aqui até cada capability
    # declarar os próprios pontos de variação: hoje nenhuma delas pode importar o runtime.
    _component(
        "geometric_mapping",
        "pose_lookup",
        BackendSpec(backend_id="lookup-policy-v1"),
    ),
    _component(
        "geometric_mapping",
        "motion_correction",
        BackendSpec(backend_id="motion-correction-v1"),
    ),
    # Opcional: sem deskew, cada scan é mapeado com a pose do seu timestamp, como antes.
    _component(
        "geometric_mapping",
        "deskew",
        BackendSpec(backend_id="pose-interpolation-deskew-v1"),
        optional=True,
    ),
    # Opcional: sem agregação, o mapa guarda todo ponto bruto, como antes.
    _component(
        "geometric_mapping",
        "aggregation",
        BackendSpec(backend_id="scan-voxel-centroid-v1"),
        optional=True,
    ),
    # FIXME: identidades versionadas por sensor_association (e `lookup-policy-v1`, por
    # state_estimation), duplicadas até a capability declarar os próprios pontos de variação;
    # adiado porque sensor_association não pode importar o runtime.
    _component(
        "sensor_association",
        "occlusion",
        BackendSpec(backend_id="conservative-depth-support-v1"),
    ),
    _component(
        "sensor_association",
        "tolerances",
        BackendSpec(backend_id="diagnostic-tolerances-v1"),
    ),
    _component(
        "sensor_association",
        "pose_policy",
        BackendSpec(backend_id="lookup-policy-v1"),
    ),
    # Opcional: a amostragem dos mapas densos da run de percepção é um canal de
    # evidência a mais; sem ela, a associação é a geometry-only.
    _component(
        "sensor_association",
        "dense_sampling",
        BackendSpec(backend_id="dense-feature-sampling-v1"),
        optional=True,
    ),
    # Opcional: a regra de pertencimento por caixa para regiões sem máscara (aceita só o
    # componente 3D que o núcleo da caixa vê, não a caixa inteira). Sem ela, a associação
    # mantém a regra de máscara e uma região só com caixa é pulada.
    _component(
        "sensor_association",
        "membership",
        BackendSpec(backend_id="box-depth-component-v1"),
        optional=True,
    ),
    # FIXME: `observation-support-anchor-v1` é versionada por point_representation, duplicada
    # até a capability declarar os próprios pontos de variação (ela não pode importar o runtime).
    # É a política que escolhe os centros de um run a partir das observações espaciais da
    # associação; sem parâmetros, um backend só.
    _component(
        "point_representation",
        "center_selection",
        BackendSpec(backend_id="observation-support-anchor-v1"),
    ),
    _component(
        "point_representation",
        "encoder",
        BackendSpec(backend_id="geometric_descriptor"),
        BackendSpec(backend_id="ptv3", device_parameter="device"),
    ),
    # FIXME: identidades versionadas por semantic_fusion, duplicadas até a capability declarar
    # os próprios pontos de variação; adiado porque semantic_fusion não pode importar o runtime.
    _component(
        "semantic_fusion",
        "support",
        BackendSpec(backend_id="geometry-jaccard-support-v1"),
    ),
    _component(
        "semantic_fusion",
        "accumulation",
        BackendSpec(backend_id="baseline-evidence-accumulation-v1"),
        BackendSpec(backend_id="quality-aware-evidence-accumulation-v1"),
    ),
    # FIXME: identidades versionadas por entity_resolution, duplicadas até a capability declarar
    # os próprios pontos de variação; adiado porque entity_resolution não pode importar o runtime.
    _component(
        "entity_resolution",
        "retrieval",
        BackendSpec(backend_id="entity-candidate-retrieval-v1"),
    ),
    _component(
        "entity_resolution",
        "resolution",
        BackendSpec(backend_id="conservative-staged-resolution-v1"),
    ),
    _component(
        "entity_resolution",
        "geometry_comparison",
        BackendSpec(backend_id="entity-geometry-comparison-v1"),
    ),
    _component(
        "entity_resolution",
        "semantic_compatibility",
        BackendSpec(backend_id="entity-semantic-compatibility-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "temporal_compatibility",
        BackendSpec(backend_id="entity-temporal-compatibility-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "appearance",
        BackendSpec(backend_id="entity-appearance-comparison-v1"),
        optional=True,
    ),
    _component(
        "entity_resolution",
        "representation",
        BackendSpec(backend_id="entity-representation-comparison-v1"),
        optional=True,
    ),
    # Opcional: sem ela, nenhuma crença é derivada e a resolução é exatamente a anterior.
    _component(
        "entity_resolution",
        "belief",
        BackendSpec(backend_id="resolved-belief-min-support-v1"),
        optional=True,
    ),
    # FIXME: `entity-geometry-summary-v1` é versionada por semantic_mapping (e reaparece em
    # spatial_relations.geometry_summary), duplicada até a capability declarar os próprios pontos
    # de variação; adiado porque semantic_mapping não pode importar o runtime.
    _component(
        "semantic_mapping",
        "geometry_summary",
        BackendSpec(backend_id="entity-geometry-summary-v1"),
    ),
    # FIXME: identidades versionadas por spatial_relations (e `entity-geometry-summary-v1`, por
    # semantic_mapping), duplicadas até a capability declarar os próprios pontos de variação;
    # adiado porque spatial_relations não pode importar o runtime.
    _component(
        "spatial_relations",
        "frame_conventions",
        BackendSpec(backend_id="map-frame-conventions-v1"),
    ),
    _component(
        "spatial_relations",
        "candidate",
        BackendSpec(backend_id="bounds-neighborhood-candidates-v1"),
    ),
    _component(
        "spatial_relations",
        "geometry_summary",
        BackendSpec(backend_id="entity-geometry-summary-v1"),
    ),
    _component(
        "spatial_relations",
        "geometric_predicate",
        BackendSpec(backend_id="bounds-geometric-predicates-v1"),
        optional=True,
    ),
    _component(
        "spatial_relations",
        "contact_predicate",
        BackendSpec(backend_id="point-contact-predicates-v1"),
        optional=True,
    ),
    _component(
        "spatial_relations",
        "semantic_predicate",
        BackendSpec(backend_id="interaction-range-predicate-v1"),
        optional=True,
    ),
)

COMPONENTS: Mapping[str, ComponentSpec] = {
    component.component_id: component for component in _COMPONENT_LIST
}
"""Every selectable variation point, keyed by ``"<capability>.<slot>"``."""
