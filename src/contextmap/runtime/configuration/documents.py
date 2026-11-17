"""Camada não tipada da configuração: documentos JSON/TOML e a mescla das camadas.

Antes de virar um :class:`~contextmap.runtime.configuration.schema.RuntimeConfig`, uma
configuração é um documento aninhado de dicionários. Este módulo produz e combina esses
documentos, sem validá-los: o perfil dá a topologia do preset com nenhum backend
selecionado, cada arquivo ``.json`` ou ``.toml`` se mescla chave a chave por cima (não há
YAML: seria uma dependência nova sem consumidor), e cada override ``dotted.path=valor``
substitui o valor no seu caminho. A validação estrutural e os defaults ficam na resolução;
aqui só se garante que cada camada é legível e que a precedência perfil < arquivos <
overrides é aplicada.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextmap.runtime.components import PRESETS
from contextmap.runtime.configuration.problems import ConfigurationError
from contextmap.runtime.configuration.schema import CONFIG_SCHEMA_VERSION


def parse_override(text: str) -> tuple[tuple[str, ...], Any]:
    """Parse one ``"dotted.path=value"`` override.

    Args:
        text: The override as typed by a user.

    Returns:
        The path segments and the value: a JSON literal (``true``, ``12``, ``[1]``,
        ``"x"``, ``null``) when the text is one, otherwise the text itself.

    Raises:
        ConfigurationError: If there is no ``=``, or a path segment is empty.
    """
    key, separator, raw = text.partition("=")
    segments = tuple(key.split("."))
    if not separator or not key or any(not segment for segment in segments):
        raise ConfigurationError.single(
            f"override {text!r} must look like 'dotted.path=value' with no empty path segment"
        )
    try:
        value: Any = json.loads(raw)
    except ValueError:
        value = raw
    return segments, value


def profile_document(profile: str) -> dict[str, Any]:
    """Return the defaults of a profile: the preset's topology with nothing selected.

    The document carries no ``policies`` section: every policy default is the field default
    of :class:`~contextmap.runtime.configuration.schema.PoliciesConfig`, applied when the
    merged document is resolved, so the profile never restates one.
    """
    preset = PRESETS.get(profile)
    if preset is None:
        known = ", ".join(sorted(PRESETS))
        raise ConfigurationError.single(f"unknown profile {profile!r}; known profiles: {known}")
    components: dict[str, dict[str, Any]] = {}
    for stage in preset.stages:
        for component_id in stage.components:
            capability, slot = component_id.split(".", 1)
            components.setdefault(capability, {})[slot] = {"backend": None}
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "pipeline": {
            "preset": preset.preset_id,
            "stages": {stage.stage_id: stage.default_enabled for stage in preset.stages},
        },
        "components": components,
        "inputs": {"sequence": None, "selections": {}, "named": {}},
        "resources": {"device": None, "workspace": None, "maps_root": None, "providers": {}},
    }


def load_file(path: Path) -> tuple[dict[str, Any], str]:
    """Read one JSON or TOML layer and hash its bytes."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".toml"}:
        raise ConfigurationError.single(
            f"{path}: unsupported configuration file format {suffix!r}; use .json or .toml"
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigurationError.single(
            f"{path}: cannot read configuration file: {error}"
        ) from error
    try:
        text = raw.decode("utf-8")
        loaded = (
            tomllib.loads(text)
            if suffix == ".toml"
            else json.loads(text, parse_constant=_reject_constant)
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ConfigurationError.single(f"{path}: invalid {suffix[1:]} syntax: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError.single(f"{path}: the document root must be a mapping")
    return loaded, f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _reject_constant(name: str) -> Any:
    """Refuse ``NaN``/``Infinity``: a configuration value is always a finite JSON number."""
    raise ValueError(f"{name} is not a valid JSON number")


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings key by key; any other value in the overlay replaces the base."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def with_value(document: Mapping[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Return a copy of ``document`` with ``value`` set at ``path``."""
    result = copy.deepcopy(dict(document))
    node = result
    for depth, segment in enumerate(path[:-1]):
        child = node.setdefault(segment, {})
        if not isinstance(child, dict):
            here = ".".join(path[: depth + 1])
            raise ConfigurationError.single(
                f"cannot set {'.'.join(path)!r}: {here!r} is not a mapping"
            )
        node = child
    node[path[-1]] = value
    return result
