"""Verificação de integridade estrutural dos artifacts de sequência.

Este módulo confere o inventário versionado e as referências entre index.jsonl e payloads,
inclusive garantindo que caminhos persistidos permaneçam dentro do artifact. Ele detecta
corrupção/inconsistência; não tenta reparar um artifact publicado.
"""

from __future__ import annotations

import json
from pathlib import Path

from contextmap.ingestion.sequence_artifact.sequence_artifact_manifest import (
    _INDEX_FILENAME,
    SequenceArtifactManifest,
)
from contextmap.shared import FileEntry, check_file_inventory, is_run_relative_path


def _check_file_inventory(root: Path, manifest: SequenceArtifactManifest) -> list[str]:
    """Check the inventory with the shared rule: hashed in chunks, never outside ``root``."""
    return check_file_inventory(
        root,
        (
            FileEntry(path=entry.path, size_bytes=entry.size_bytes, content_hash=entry.content_hash)
            for entry in manifest.file_inventory
        ),
    )


def _check_index_cross_references(root: Path, manifest: SequenceArtifactManifest) -> list[str]:
    """Validate every payload path referenced by the canonical index.

    Args:
        root: Artifact root containing the index and payload files.
        manifest: Manifest whose inventory defines the allowed persisted paths.

    Returns:
        Human-readable cross-reference problems; empty means every reference is valid.
    """
    problems: list[str] = []
    known_paths = {entry.path for entry in manifest.file_inventory}
    index_path = root / _INDEX_FILENAME
    if not index_path.is_file():
        # NOTE: A ausência do índice já é coberta pelo inventário; repetir o problema aqui
        # geraria dois diagnósticos para uma única causa.
        return problems

    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            payload_path = record.get("payload_path")
            if payload_path is not None and not (
                isinstance(payload_path, str) and is_run_relative_path(payload_path)
            ):
                problems.append(
                    f"{_INDEX_FILENAME} line {line_number} references payload_path "
                    f"{payload_path!r}, which is not a relative path inside the artifact"
                )
            elif payload_path is not None and payload_path not in known_paths:
                problems.append(
                    f"{_INDEX_FILENAME} line {line_number} references payload_path "
                    f"{payload_path!r} not present in manifest file_inventory"
                )
    return problems
