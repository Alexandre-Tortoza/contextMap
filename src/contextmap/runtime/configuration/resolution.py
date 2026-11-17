"""Resolução determinística da configuração efetiva.

Uma :class:`~contextmap.runtime.configuration.schema.EffectiveConfig` é o que sobra depois
de aplicar as camadas nesta precedência (a posterior vence): os defaults do perfil, os
arquivos de configuração na ordem dada e os overrides de linha de comando. Variáveis de
ambiente não participam dessa mescla: elas só carregam segredos, que nunca são gravados,
então nem a configuração efetiva nem o seu digest podem conter um.

Este módulo valida a estrutura do documento mesclado, devolvendo **todos** os problemas de
uma vez, e o normaliza num :class:`~contextmap.runtime.configuration.schema.RuntimeConfig`.
É o único lugar onde defaults se aplicam: uma entrada que o documento omite fica de fora da
construção, e o default vem do campo correspondente do schema. A gramática de seleção de
upstream (:data:`LATEST`, :data:`NAMED_PREFIX`) mora aqui porque é validada aqui;
``contextmap.runtime.selection`` a importa para resolvê-la contra o run store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from contextmap.runtime.components import (
    CANONICAL_PROFILE_ID,
    COMPONENTS,
    PRESETS,
    ComponentSpec,
    RuntimePreset,
)
from contextmap.runtime.configuration.documents import (
    deep_merge,
    load_file,
    parse_override,
    profile_document,
    with_value,
)
from contextmap.runtime.configuration.problems import ConfigProblem, ConfigurationError
from contextmap.runtime.configuration.schema import (
    CONFIG_SCHEMA_VERSION,
    DEBUG_LEVELS,
    PERCEPTION_SCHEDULES,
    TRAJECTORY_MODES,
    CameraPreparationConfig,
    ComponentConfig,
    ConfigurationSource,
    ConfigValue,
    EffectiveConfig,
    InputsConfig,
    PipelineConfig,
    PoliciesConfig,
    ResourcesConfig,
    RuntimeConfig,
)
from contextmap.runtime.configuration.secrets import looks_like_secret
from contextmap.runtime.configuration.validation import association_range, incompatibilities

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "pipeline",
        "components",
        "inputs",
        "resources",
        "policies",
        "image_preparation",
    }
)


LATEST = "latest"
"""Selection value that explicitly opts a stage into the newest compatible run."""

NAMED_PREFIX = "named:"
"""Prefix of a selection value that refers to a named selection."""


def resolve_effective_config(
    *,
    profile: str = CANONICAL_PROFILE_ID,
    files: Sequence[str | os.PathLike[str]] = (),
    overrides: Sequence[str] = (),
) -> EffectiveConfig:
    """Resolve the effective configuration from its layers.

    Precedence, lowest to highest: the profile defaults, each file in the order
    given, then each override in the order given. Nested mappings merge key by key;
    every other value replaces. Nothing is read from the environment.

    Args:
        profile: Identity of the base profile and topology preset.
        files: JSON (``.json``) or TOML (``.toml``) configuration files.
        overrides: ``"dotted.path=value"`` overrides; the value is read as a JSON
            literal when it is one and as plain text otherwise.

    Returns:
        The validated effective configuration, its digest and its sources.

    Raises:
        ConfigurationError: If the profile is unknown, a file cannot be read, or the
            merged document is invalid or selects an incompatible combination.
    """
    document = profile_document(profile)
    sources = [ConfigurationSource(kind="profile", identity=profile, content_hash=None)]
    for file in files:
        layer, content_hash = load_file(Path(file))
        document = deep_merge(document, layer)
        sources.append(
            ConfigurationSource(kind="file", identity=str(file), content_hash=content_hash)
        )
    for text in overrides:
        path, value = parse_override(text)
        document = with_value(document, path, value)
        sources.append(
            ConfigurationSource(kind="override", identity=".".join(path), content_hash=None)
        )
    config = build_config(document)
    return EffectiveConfig(
        config=config, digest=config_digest(config.to_document()), sources=tuple(sources)
    )


def build_config(document: object) -> RuntimeConfig:
    """Validate a merged document and normalize it into a :class:`RuntimeConfig`."""
    problems: list[ConfigProblem] = []
    config = _parse_document(document, problems)
    if config is not None:
        problems.extend(incompatibilities(config))
    if problems or config is None:
        raise ConfigurationError(problems)
    return config


def _parse_document(document: object, problems: list[ConfigProblem]) -> RuntimeConfig | None:
    if not isinstance(document, Mapping):
        problems.append(ConfigProblem(path="", message="the configuration must be a mapping"))
        return None
    for key in document:
        if key not in _TOP_LEVEL_KEYS:
            problems.append(
                ConfigProblem(
                    path=str(key),
                    message=f"unknown section; expected one of {sorted(_TOP_LEVEL_KEYS)}",
                )
            )
    version = document.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        problems.append(
            ConfigProblem(
                path="schema_version",
                message=f"is {version!r}, this runtime reads {CONFIG_SCHEMA_VERSION!r}",
            )
        )
    preset, toggles = _parse_pipeline(document.get("pipeline", {}), problems)
    raw_components = _parse_components(document.get("components", {}), problems)
    inputs = _parse_inputs(
        document.get("inputs", {}),
        problems,
        None if preset is None else {stage.stage_id for stage in preset.stages},
    )
    resources = _parse_resources(document.get("resources", {}), problems)
    policies = _parse_policies(document.get("policies", {}), problems)
    image_preparation = _parse_image_preparation(document.get("image_preparation", {}), problems)
    if preset is None or problems:
        return None

    stages = {
        stage.stage_id: toggles.get(stage.stage_id, stage.default_enabled)
        for stage in preset.stages
    }
    components: dict[str, ComponentConfig] = {}
    for stage in preset.stages:
        if not stages[stage.stage_id]:
            continue  # estágio desligado não é construído: sua configuração não conta.
        for component_id in stage.components:
            components[component_id] = _resolve_component(
                COMPONENTS[component_id], raw_components.get(component_id), resources.device
            )
    return RuntimeConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        pipeline=PipelineConfig(preset=preset.preset_id, stages=MappingProxyType(stages)),
        components=MappingProxyType(components),
        inputs=inputs,
        resources=resources,
        policies=policies,
        image_preparation=image_preparation,
    )


def _parse_pipeline(
    value: object, problems: list[ConfigProblem]
) -> tuple[RuntimePreset | None, dict[str, bool]]:
    section = _mapping(value, "pipeline", problems)
    _reject_unknown(section, {"preset", "stages"}, "pipeline", problems)
    preset_id = section.get("preset", CANONICAL_PROFILE_ID)
    preset = PRESETS.get(preset_id) if isinstance(preset_id, str) else None
    if preset is None:
        problems.append(
            ConfigProblem(
                path="pipeline.preset",
                message=(
                    f"unknown preset {preset_id!r}; known presets: {', '.join(sorted(PRESETS))}"
                ),
            )
        )
    toggles: dict[str, bool] = {}
    for stage_id, enabled in _mapping(
        section.get("stages", {}), "pipeline.stages", problems
    ).items():
        path = f"pipeline.stages.{stage_id}"
        if not isinstance(enabled, bool):
            problems.append(ConfigProblem(path=path, message="must be true or false"))
            continue
        if preset is not None:
            try:
                declaration = preset.stage(stage_id)
            except KeyError:
                known = ", ".join(stage.stage_id for stage in preset.stages)
                problems.append(
                    ConfigProblem(path=path, message=f"unknown stage; known stages: {known}")
                )
                continue
            if not declaration.optional and not enabled:
                problems.append(
                    ConfigProblem(
                        path=path,
                        message=f"stage {stage_id!r} is not optional and cannot be disabled",
                    )
                )
                continue
        toggles[stage_id] = enabled
    return preset, toggles


@dataclass(frozen=True, kw_only=True)
class _RawComponent:
    backend: str | None
    blocks: Mapping[str, Mapping[str, ConfigValue]]


def _parse_components(value: object, problems: list[ConfigProblem]) -> dict[str, _RawComponent]:
    raw: dict[str, _RawComponent] = {}
    for capability, slots in _mapping(value, "components", problems).items():
        capability_path = f"components.{capability}"
        known_slots = {spec.slot for spec in COMPONENTS.values() if spec.capability == capability}
        if not known_slots:
            known = sorted({spec.capability for spec in COMPONENTS.values()})
            problems.append(
                ConfigProblem(path=capability_path, message=f"unknown capability; known: {known}")
            )
            continue
        for slot, entry in _mapping(slots, capability_path, problems).items():
            path = f"{capability_path}.{slot}"
            if slot not in known_slots:
                problems.append(
                    ConfigProblem(path=path, message=f"unknown slot; known: {sorted(known_slots)}")
                )
                continue
            parsed = _parse_component(COMPONENTS[f"{capability}.{slot}"], entry, path, problems)
            if parsed is not None:
                raw[f"{capability}.{slot}"] = parsed
    return raw


def _parse_component(
    spec: ComponentSpec, entry: object, path: str, problems: list[ConfigProblem]
) -> _RawComponent | None:
    section = _mapping(entry, path, problems)
    backend = section.get("backend")
    supported = sorted(spec.backends)
    if backend is not None and backend not in spec.backends:
        problems.append(
            ConfigProblem(
                path=f"{path}.backend",
                message=f"unknown backend {backend!r}; supported: {', '.join(supported)}",
            )
        )
    blocks: dict[str, Mapping[str, ConfigValue]] = {}
    for key, block in section.items():
        if key == "backend":
            continue
        if key not in spec.backends:
            problems.append(
                ConfigProblem(
                    path=f"{path}.{key}",
                    message=f"is neither 'backend' nor a supported backend; supported: "
                    f"{', '.join(supported)}",
                )
            )
            continue
        if not isinstance(block, Mapping):
            problems.append(ConfigProblem(path=f"{path}.{key}", message="must be a mapping"))
            continue
        frozen = _freeze(block, f"{path}.{key}", problems)
        if isinstance(frozen, Mapping):
            blocks[key] = frozen
    return _RawComponent(backend=backend if isinstance(backend, str) else None, blocks=blocks)


def _resolve_component(
    spec: ComponentSpec, raw: _RawComponent | None, device: str | None
) -> ComponentConfig:
    """Keep only the selected backend's parameters and hand it the resource device."""
    if raw is None or raw.backend is None:
        return ComponentConfig(backend=None, parameters=MappingProxyType({}))
    parameters = dict(raw.blocks.get(raw.backend, {}))
    device_parameter = spec.backends[raw.backend].device_parameter
    if device_parameter is not None and device is not None and device_parameter not in parameters:
        parameters[device_parameter] = device
    return ComponentConfig(backend=raw.backend, parameters=MappingProxyType(parameters))


def _parse_inputs(
    value: object, problems: list[ConfigProblem], known_stages: set[str] | None
) -> InputsConfig:
    section = _mapping(value, "inputs", problems)
    _reject_unknown(
        section, {"sequence", "selections", "named", "observation_selection"}, "inputs", problems
    )
    named = _parse_named(section.get("named", {}), problems, known_stages)
    selections: dict[str, tuple[str, ...]] = {}
    for stage_id, raw in _mapping(
        section.get("selections", {}), "inputs.selections", problems
    ).items():
        path = f"inputs.selections.{stage_id}"
        if known_stages is not None and stage_id not in known_stages:
            problems.append(ConfigProblem(path=path, message="is not a stage of the preset"))
            continue
        references = _references(raw, path, problems)
        if references is None:
            continue
        for reference in references:
            if reference.startswith(NAMED_PREFIX):
                name = reference.removeprefix(NAMED_PREFIX)
                if stage_id not in named.get(name, {}):
                    problems.append(
                        ConfigProblem(
                            path=path,
                            message=f"refers to the named selection {name!r}, which does not "
                            f"define stage {stage_id!r} in inputs.named",
                        )
                    )
        exclusive = [r for r in references if r == LATEST or r.startswith(NAMED_PREFIX)]
        if exclusive and len(references) > 1:
            problems.append(
                ConfigProblem(
                    path=path,
                    message=f"{exclusive[0]!r} must be the only reference of the stage",
                )
            )
        selections[stage_id] = references
    return InputsConfig(
        sequence=_optional_text(section.get("sequence"), "inputs.sequence", problems),
        selections=MappingProxyType(selections),
        named=MappingProxyType(named),
        observation_selection=_parse_observation_selection(
            section.get("observation_selection"), problems
        ),
    )


def _parse_observation_selection(
    value: object, problems: list[ConfigProblem]
) -> Mapping[str, ConfigValue] | None:
    """Read an encoded sequence selection; ``None`` selects the whole sequence."""
    if value is None:
        return None
    path = "inputs.observation_selection"
    kind = value.get("kind") if isinstance(value, Mapping) else None
    if not isinstance(kind, str) or not kind:
        problems.append(
            ConfigProblem(
                path=path,
                message="must be an encoded sequence selection: a mapping with a non-empty 'kind'",
            )
        )
        return None
    frozen = _freeze(value, path, problems)
    return frozen if isinstance(frozen, Mapping) else None


def _parse_named(
    value: object, problems: list[ConfigProblem], known_stages: set[str] | None
) -> dict[str, Mapping[str, tuple[str, ...]]]:
    named: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for name, stages in _mapping(value, "inputs.named", problems).items():
        path = f"inputs.named.{name}"
        if not name:
            problems.append(ConfigProblem(path="inputs.named", message="a name must not be empty"))
            continue
        resolved: dict[str, tuple[str, ...]] = {}
        for stage_id, raw in _mapping(stages, path, problems).items():
            stage_path = f"{path}.{stage_id}"
            if known_stages is not None and stage_id not in known_stages:
                problems.append(
                    ConfigProblem(path=stage_path, message="is not a stage of the preset")
                )
                continue
            references = _references(raw, stage_path, problems)
            if references is None:
                continue
            if any(r == LATEST or r.startswith(NAMED_PREFIX) for r in references):
                problems.append(
                    ConfigProblem(
                        path=stage_path,
                        message="a named selection lists exact artifact ids only",
                    )
                )
                continue
            resolved[stage_id] = references
        named[name] = MappingProxyType(resolved)
    return named


def _references(raw: object, path: str, problems: list[ConfigProblem]) -> tuple[str, ...] | None:
    """Read one reference or a list of them; every reference is a non-empty text."""
    items = [raw] if isinstance(raw, str) else raw
    if (
        not isinstance(items, list | tuple)
        or not items
        or not all(isinstance(item, str) and item for item in items)
    ):
        problems.append(
            ConfigProblem(
                path=path, message="must be a non-empty reference or a non-empty list of them"
            )
        )
        return None
    if len(set(items)) != len(items):
        problems.append(ConfigProblem(path=path, message="lists the same reference twice"))
        return None
    return tuple(items)


def _parse_resources(value: object, problems: list[ConfigProblem]) -> ResourcesConfig:
    section = _mapping(value, "resources", problems)
    _reject_unknown(
        section, {"device", "workspace", "maps_root", "providers"}, "resources", problems
    )
    return ResourcesConfig(
        device=_optional_text(section.get("device"), "resources.device", problems),
        workspace=_optional_text(section.get("workspace"), "resources.workspace", problems),
        maps_root=_optional_text(section.get("maps_root"), "resources.maps_root", problems),
        providers=_parse_providers(section.get("providers", {}), problems),
    )


def _parse_providers(value: object, problems: list[ConfigProblem]) -> Mapping[str, str]:
    """Read the declared ``component_id -> "module:attribute"`` provider targets.

    Validation here is only structural (each target is a non-empty string): resolving a
    target into a callable is deferred to
    :func:`~contextmap.runtime.composition.resolve_provider`, lazily, only for a component
    actually being composed. A target need not name a known component: an entry for a
    component that is never composed simply never resolves, exactly like an unused
    ``components.<capability>.<slot>`` selection.
    """
    providers: dict[str, str] = {}
    for component_id, target in _mapping(value, "resources.providers", problems).items():
        path = f"resources.providers.{component_id}"
        if not isinstance(target, str) or not target:
            problems.append(ConfigProblem(path=path, message="must be a non-empty string"))
            continue
        providers[component_id] = target
    return MappingProxyType(providers)


def _parse_image_preparation(
    value: object, problems: list[ConfigProblem]
) -> Mapping[str, CameraPreparationConfig]:
    """Read the per-camera image-preparation masks.

    Validation here is only structural (each mask reference is a non-empty string): decoding
    a reference into a mask and checking its dimensions against the prepared image is deferred
    to the composition root, only for a camera whose frames are actually prepared.
    """
    cameras: dict[str, CameraPreparationConfig] = {}
    for camera, entry in _mapping(value, "image_preparation", problems).items():
        base = f"image_preparation.{camera}"
        section = _mapping(entry, base, problems)
        _reject_unknown(section, {"valid_region", "exclusions"}, base, problems)
        valid_region = _optional_text(section.get("valid_region"), f"{base}.valid_region", problems)
        exclusions: dict[str, str] = {}
        for name, reference in _mapping(
            section.get("exclusions", {}), f"{base}.exclusions", problems
        ).items():
            path = f"{base}.exclusions.{name}"
            if not isinstance(reference, str) or not reference:
                problems.append(ConfigProblem(path=path, message="must be a non-empty string"))
                continue
            exclusions[name] = reference
        cameras[camera] = CameraPreparationConfig(
            valid_region=valid_region, exclusions=MappingProxyType(exclusions)
        )
    return MappingProxyType(cameras)


def _parse_policies(value: object, problems: list[ConfigProblem]) -> PoliciesConfig:
    """Read the policies section, leaving every omitted policy to its field default.

    Only what the document declares reaches :class:`PoliciesConfig`, so its field defaults
    are the single source of every policy default. A declared value outside its vocabulary
    is reported and left out: the configuration then never resolves, so nothing stands in
    for it.
    """
    section = _mapping(value, "policies", problems)
    _reject_unknown(
        section,
        {"debug_level", "trajectory_mode", "association_max_range_m", "perception_schedule"},
        "policies",
        problems,
    )
    declared: dict[str, str] = {}
    # A ordem da tupla é a ordem em que os problemas são reportados.
    for name, choices in (
        ("debug_level", DEBUG_LEVELS),
        ("trajectory_mode", TRAJECTORY_MODES),
        ("perception_schedule", PERCEPTION_SCHEDULES),
    ):
        if name not in section:
            continue
        choice = section[name]
        if choice in choices:
            declared[name] = choice
            continue
        problems.append(
            ConfigProblem(
                path=f"policies.{name}",
                message=f"is {choice!r}; expected one of {', '.join(choices)}",
            )
        )
    return PoliciesConfig(**declared, association_max_range_m=association_range(section, problems))


def _mapping(value: object, path: str, problems: list[ConfigProblem]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    problems.append(ConfigProblem(path=path, message="must be a mapping"))
    return {}


def _reject_unknown(
    section: Mapping[str, Any], allowed: set[str], path: str, problems: list[ConfigProblem]
) -> None:
    for key in section:
        if key not in allowed:
            problems.append(
                ConfigProblem(
                    path=f"{path}.{key}",
                    message=f"unknown entry; expected one of {sorted(allowed)}",
                )
            )


def _optional_text(value: object, path: str, problems: list[ConfigProblem]) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    problems.append(ConfigProblem(path=path, message="must be a non-empty string or null"))
    return None


def _freeze(value: object, path: str, problems: list[ConfigProblem]) -> ConfigValue:
    """Deep-freeze a JSON-compatible value, reporting anything else."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            problems.append(ConfigProblem(path=path, message="must be a finite number"))
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, ConfigValue] = {}
        for key, item in value.items():
            child = f"{path}.{key}"
            if not isinstance(key, str):
                problems.append(ConfigProblem(path=path, message=f"key {key!r} must be text"))
                continue
            if looks_like_secret(key):
                problems.append(
                    ConfigProblem(
                        path=child,
                        message=(
                            "looks like a secret; secrets come from the environment and are "
                            "never stored in configuration"
                        ),
                    )
                )
                continue
            frozen[key] = _freeze(item, child, problems)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze(item, f"{path}[{index}]", problems) for index, item in enumerate(value)
        )
    problems.append(
        ConfigProblem(path=path, message=f"{type(value).__name__} is not a JSON-compatible value")
    )
    return None


def config_digest(document: object) -> str:
    """Hash the canonical JSON form of a document."""
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
