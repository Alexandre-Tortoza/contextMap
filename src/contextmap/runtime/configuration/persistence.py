"""Persistência da configuração efetiva de um run.

Um run grava a configuração que de fato usou em ``effective_config.json``: a versão do
schema, o digest, a configuração resolvida e as camadas que a produziram (nunca um segredo
nem o valor de um override). A escrita é atômica e um arquivo publicado é imutável: gravar
a mesma configuração de novo é um no-op e gravar outra por cima é recusado.

A leitura confere o digest e exige que o documento já esteja na forma resolvida (ida e
volta pela resolução produz o mesmo documento). Por isso a serialização de
:meth:`~contextmap.runtime.configuration.schema.RuntimeConfig.to_document` e a forma
canônica do digest não podem mudar sem invalidar todo registro de run já gravado.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from contextmap.runtime._files import publish_text
from contextmap.runtime.configuration.problems import ConfigurationError
from contextmap.runtime.configuration.resolution import build_config, config_digest
from contextmap.runtime.configuration.schema import (
    CONFIG_SCHEMA_VERSION,
    ConfigurationSource,
    EffectiveConfig,
)

EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
"""Name under which a run persists its effective configuration."""


def write_effective_config(effective: EffectiveConfig, directory: str | os.PathLike[str]) -> Path:
    """Persist the effective configuration, its digest and its sources.

    The write is atomic (a reader sees the whole file or none of it) and never
    replaces an existing file: writing the same configuration again is a no-op, and
    writing a different one over it is refused, because a published run is immutable.

    Args:
        effective: The effective configuration to persist.
        directory: Directory that receives ``effective_config.json``; created if
            missing.

    Returns:
        Path of the persisted file.

    Raises:
        ConfigurationError: If the directory already holds a different effective
            configuration.
    """
    target = Path(directory)
    final = target / EFFECTIVE_CONFIG_FILENAME
    text = json.dumps(effective.to_document(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        return publish_text(target, EFFECTIVE_CONFIG_FILENAME, text)
    except FileExistsError:
        existing = read_effective_config(final)
        if existing.digest != effective.digest:
            raise ConfigurationError.single(
                f"{final} already holds a different effective configuration "
                f"({existing.digest}); a published run is never rewritten"
            ) from None
    return final


def read_effective_config(path: str | os.PathLike[str]) -> EffectiveConfig:
    """Read a persisted effective configuration and verify it.

    Args:
        path: Path of an ``effective_config.json`` file.

    Returns:
        The effective configuration, exactly as persisted.

    Raises:
        ConfigurationError: If the file is unreadable, follows another schema
            version, does not match its digest, or is not in normalized form.
    """
    source = Path(path)
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigurationError.single(f"cannot read {source}: {error}") from error
    if not isinstance(stored, dict) or not {"schema_version", "digest", "config", "sources"} <= (
        stored.keys()
    ):
        raise ConfigurationError.single(f"{source} is not an effective configuration document")
    if stored["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError.single(
            f"{source} follows schema_version {stored['schema_version']!r}, this runtime "
            f"reads {CONFIG_SCHEMA_VERSION!r}"
        )
    document = stored["config"]
    if config_digest(document) != stored["digest"]:
        raise ConfigurationError.single(f"{source} does not match its digest; it was altered")
    config = build_config(document)
    if config.to_document() != document:
        raise ConfigurationError.single(f"{source} holds a configuration that is not resolved")
    return EffectiveConfig(
        config=config,
        digest=stored["digest"],
        sources=tuple(
            ConfigurationSource(
                kind=entry["kind"],
                identity=entry["identity"],
                content_hash=entry["content_hash"],
            )
            for entry in stored["sources"]
        ),
    )
