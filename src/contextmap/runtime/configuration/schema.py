"""Schema versionado da configuração de runtime.

Uma configuração de runtime diz *o que executar e com quais implementações*; nunca diz
como uma capability funciona. Ela se divide nas preocupações que a arquitetura mantém
separadas:

- ``pipeline``: o preset de topologia e quais estágios opcionais participam;
- ``components``: o backend selecionado para cada ponto de variação, só com os parâmetros
  desse backend;
- ``inputs``: de qual sequência e de quais runs upstream uma execução parte;
- ``resources``: dispositivo e workspace;
- ``policies``: quanta evidência de debug persistir e as políticas transversais do run;
- ``image_preparation``: as máscaras estáticas de preparação de imagem por câmera.

Este módulo é a forma tipada e já resolvida desse documento e a sua serialização
(``to_document``), que é exatamente o que o digest cobre e o que um run persiste: mudar
essa serialização muda a identidade de toda configuração já gravada. Os defaults das
políticas são os defaults dos campos de :class:`PoliciesConfig` e de nenhum outro lugar.

A versão do schema de configuração é independente da versão do schema do ContextMap: a
primeira descreve como um run é pedido, a segunda o que um run produz.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

CONFIG_SCHEMA_VERSION = "0.1.0"
"""Version of the runtime configuration schema (not of the ContextMap schema)."""

DEBUG_LEVELS = ("none", "standard", "full")
"""Debug levels every capability run artifact understands."""

ConfigValue: TypeAlias = (
    "bool | int | float | str | tuple[ConfigValue, ...] | Mapping[str, ConfigValue] | None"
)
"""A JSON-compatible value, frozen: sequences are tuples and mappings are read-only."""


@dataclass(frozen=True, kw_only=True)
class PipelineConfig:
    """Topology selection.

    Attributes:
        preset: Identity of the topology preset, for example ``"canonical/1"``.
        stages: Whether each stage of the preset takes part, fully resolved: a
            stage the configuration was silent about carries its preset default.
    """

    preset: str
    stages: Mapping[str, bool]


@dataclass(frozen=True, kw_only=True)
class ComponentConfig:
    """The backend chosen for one variation point, with its own parameters.

    Attributes:
        backend: Identity of the selected backend, or ``None`` while unselected.
        parameters: Parameters of the selected backend only. The capability that owns
            the backend validates and interprets them; the runtime only carries them.
    """

    backend: str | None
    parameters: Mapping[str, ConfigValue]


@dataclass(frozen=True, kw_only=True)
class InputsConfig:
    """What an execution starts from.

    Attributes:
        sequence: Identity of the physical sequence the execution is about, when already
            chosen. It is an explicit expectation checked against the lineage of the
            selected artifacts, never inferred from names.
        selections: For each stage, which upstream runs to start from: an exact artifact
            id, several ids (distinct runs kept as separate evidence), ``"latest"``
            (explicit opt-in to the newest compatible run) or ``"named:<name>"`` (a named
            selection defined in ``named``). A stage that is not listed is never
            selected implicitly.
        named: Named selections, each mapping stages to exact artifact ids.
        observation_selection: Which observations of the sequence the observation-scoped
            stages process, as a selection encoded by
            ``contextmap.ingestion.encode_selection``; ``None`` for the whole
            sequence. Only its shape is checked here: the stage that uses it decodes it
            through the Ingestion, which owns the selection kinds.
    """

    sequence: str | None
    selections: Mapping[str, tuple[str, ...]]
    named: Mapping[str, Mapping[str, tuple[str, ...]]]
    observation_selection: Mapping[str, ConfigValue] | None = None


@dataclass(frozen=True, kw_only=True)
class ResourcesConfig:
    """Where and on what an execution runs.

    Attributes:
        device: Device handed to every backend that declares a device parameter and
            does not set one itself, or ``None`` to leave each backend's own choice.
        workspace: Directory that holds this execution's artifacts (the run store: datasets,
            ``run-NNNN`` runs and their stages, and branches), when chosen.
        maps_root: Directory that holds **only** the final :class:`ContextMapArtifact` of each
            map, one immutable directory per build under ``<maps_root>/<map_id>/<NNNNNN>``. It is
            deliberately separate from ``workspace`` so a viewer can scan it alone: nothing but
            published maps is ever written there. A ``context build`` publishes its terminal
            ``context_map`` artifact here; every other artifact stays under ``workspace``. Left
            ``None`` in the config, a ``context build`` still publishes by ``map_id``, into the
            default clean sibling of the run store ``<workspace>/../maps`` (resolved by the CLI);
            set it explicitly to publish maps somewhere else.
        providers: Declared :data:`~contextmap.runtime.composition.RuntimeProvider` targets
            for backends without a bundled model loader, keyed by component identity
            (``"<capability>.<slot>"``). Each value is a ``"module:attribute"`` string,
            resolved lazily -- only for a component actually being composed -- by
            :func:`~contextmap.runtime.composition.resolve_provider`. This is a deployment
            decision (which process supplies which model runtime), never a backend's own
            scientific parameter, so it lives here and not under a backend's own
            ``components.<capability>.<slot>`` parameters. An explicit ``providers=``
            argument to :func:`~contextmap.runtime.composition.compose` still wins over a
            declared target for the same component. See ``docs/composition.md`` for the
            security posture of resolving a configured target.
    """

    device: str | None
    workspace: str | None
    providers: Mapping[str, str]
    maps_root: str | None = None


# FIXME: o vocabulário de `trajectory_mode` pertence a state_estimation, que é quem decide
# se uma pose auxiliar de ground truth pode virar a trajetória operacional. Ele fica aqui
# enquanto state_estimation é refatorada em paralelo e enquanto uma capability não pode
# importar o runtime (tests/architecture/test_boundaries.py); a mudança é adiada até lá.
TRAJECTORY_MODES = ("operational_only", "allow_ground_truth")
PERCEPTION_SCHEDULES = ("frame_major", "stage_major")


@dataclass(frozen=True, kw_only=True)
class PoliciesConfig:
    """How much diagnostic evidence an execution persists, and cross-cutting run policies.

    Attributes:
        debug_level: ``"none"`` (default), ``"standard"`` or ``"full"``. Debug output is never a
            contractual dependency of a downstream stage.
        trajectory_mode: ``"operational_only"`` (default) or ``"allow_ground_truth"``.
            Governs whether ``state_estimation``'s optional auxiliary pose input, when
            declared ``pose_role="ground_truth"``, may become the operational trajectory; the
            default keeps a ground-truth-tagged auxiliary pose from affecting the run at all,
            exactly as if it were absent.
        association_max_range_m: Largest distance, in meters, from the camera optical centre
            at which ``sensor_association`` still evaluates a map element.
            ``None`` (default) evaluates the whole map, exactly as if the policy were absent,
            so enabling culling is always a deliberate, recorded decision. A value bounds
            per-frame work and memory by local geometry instead of map size, and narrows the
            evaluated population: it is a scientific choice, which is why it has no default.
        perception_schedule: ``"frame_major"`` (default) runs every stage of a frame before
            the next frame; ``"stage_major"`` runs one stage over every frame of the run and
            releases each model after its last stage, so models that do not fit a GPU together
            run one at a time. It changes order and GPU residency, never the results.
    """

    debug_level: str = "none"
    trajectory_mode: str = "operational_only"
    association_max_range_m: float | None = None
    perception_schedule: str = "frame_major"


@dataclass(frozen=True, kw_only=True)
class CameraPreparationConfig:
    """Static per-camera image-preparation constraints, in the prepared-image space.

    The recording rig and a fisheye's black border occupy the same pixels every frame, so
    they are authored once per camera and referenced here. Both constraints are honored by
    the perception and the sensor-association stages; a camera the section is silent about
    carries neither.

    Attributes:
        valid_region: Reference to the eligibility mask asset (foreground = usable pixels),
            or ``None`` when the whole image is eligible.
        exclusions: Named exclusion mask assets (foreground = excluded pixels), keyed by a
            stable diagnostic name such as ``"ego"``.
    """

    valid_region: str | None
    exclusions: Mapping[str, str]


def _no_image_preparation() -> Mapping[str, CameraPreparationConfig]:
    """Return the empty, immutable per-camera preparation mapping.

    A ``MappingProxyType`` cannot be a field default: ``dataclasses`` refuses an
    unhashable default on the Python versions this project supports, so the empty
    mapping is built by a factory instead.
    """
    return MappingProxyType({})


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    """A validated, fully resolved runtime configuration.

    Attributes:
        schema_version: Version of the configuration schema this document follows.
        pipeline: Topology selection.
        components: Selected backend per variation point of every enabled stage.
        inputs: Sequence and upstream selections.
        resources: Device and workspace.
        policies: Debug policy.
        image_preparation: Static per-camera image-preparation masks, keyed by camera
            calibration id. Empty when no camera declares any.
    """

    schema_version: str
    pipeline: PipelineConfig
    components: Mapping[str, ComponentConfig]
    inputs: InputsConfig
    resources: ResourcesConfig
    policies: PoliciesConfig
    image_preparation: Mapping[str, CameraPreparationConfig] = field(
        default_factory=_no_image_preparation
    )

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible document this configuration is persisted as.

        Returns:
            A new nested dictionary; feeding it back through the resolver yields an
            equal configuration.
        """
        nested: dict[str, dict[str, Any]] = {}
        for component_id, component in self.components.items():
            capability, slot = component_id.split(".", 1)
            entry: dict[str, Any] = {"backend": component.backend}
            if component.backend is not None:
                entry[component.backend] = _thaw(component.parameters)
            nested.setdefault(capability, {})[slot] = entry
        return {
            "schema_version": self.schema_version,
            "pipeline": {
                "preset": self.pipeline.preset,
                "stages": dict(self.pipeline.stages),
            },
            "components": nested,
            "inputs": {
                "sequence": self.inputs.sequence,
                "selections": {
                    stage: list(references) for stage, references in self.inputs.selections.items()
                },
                "named": {
                    name: {stage: list(ids) for stage, ids in stages.items()}
                    for name, stages in self.inputs.named.items()
                },
                **(
                    {}
                    if self.inputs.observation_selection is None
                    else {"observation_selection": _thaw(self.inputs.observation_selection)}
                ),
            },
            "resources": {
                "device": self.resources.device,
                "workspace": self.resources.workspace,
                "maps_root": self.resources.maps_root,
                "providers": dict(self.resources.providers),
            },
            "policies": {
                "debug_level": self.policies.debug_level,
                "trajectory_mode": self.policies.trajectory_mode,
                "association_max_range_m": self.policies.association_max_range_m,
                "perception_schedule": self.policies.perception_schedule,
            },
            **(
                {}
                if not self.image_preparation
                else {
                    "image_preparation": {
                        camera: {
                            **(
                                {}
                                if prep.valid_region is None
                                else {"valid_region": prep.valid_region}
                            ),
                            **(
                                {} if not prep.exclusions else {"exclusions": dict(prep.exclusions)}
                            ),
                        }
                        for camera, prep in self.image_preparation.items()
                    }
                }
            ),
        }


@dataclass(frozen=True, kw_only=True)
class ConfigurationSource:
    """One layer that contributed to an effective configuration.

    Attributes:
        kind: ``"profile"``, ``"file"`` or ``"override"``.
        identity: Profile identity, file path as given, or the dotted key an override
            set. An override's value is never recorded here.
        content_hash: SHA-256 of the file bytes for a file layer, otherwise ``None``.
    """

    kind: Literal["profile", "file", "override"]
    identity: str
    content_hash: str | None


@dataclass(frozen=True, kw_only=True)
class EffectiveConfig:
    """The configuration an execution actually uses, with its identity.

    Attributes:
        config: The fully resolved configuration.
        digest: ``"sha256:<hex>"`` of the canonical form of ``config``. Two
            configurations that resolve to the same values have the same digest,
            whatever files or overrides produced them.
        sources: The layers applied, lowest precedence first, for auditing.
    """

    config: RuntimeConfig
    digest: str
    sources: tuple[ConfigurationSource, ...]

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible document persisted as ``effective_config.json``.

        Returns:
            The schema version, the digest, the resolved configuration and the layers
            that produced it. It contains no secret and no override value.
        """
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "digest": self.digest,
            "config": self.config.to_document(),
            "sources": [
                {
                    "kind": source.kind,
                    "identity": source.identity,
                    "content_hash": source.content_hash,
                }
                for source in self.sources
            ],
        }


def _thaw(value: ConfigValue) -> Any:
    """Turn a frozen value back into plain JSON containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
