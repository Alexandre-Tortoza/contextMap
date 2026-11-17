"""Contrato de metadata de um run persistido, e a recusa de um schema desconhecido.

O manifest é o que um consumidor lê sobre uma derivação sem abrir payload nenhum: as três
identidades a montante, a grade, o frame, as três figuras do run e o inventário de arquivos
contratuais. O writer o lê de volta para devolver o que publicou, e o reader o lê ao abrir,
então ambos dependem daqui.

Este módulo não escreve nada e não conhece ``AtomicRunDirectory``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import FrameId
from contextmap.shared import FileEntry
from contextmap.spatial_support.run_artifact.layout import (
    MANIFEST,
    SCHEMA_VERSION,
    SpatialSupportArtifactError,
    read_json_document,
)

SpatialSupportRunId = NewType("SpatialSupportRunId", str)
"""Identity of one derivation, supplied by the caller."""


@dataclass(frozen=True, kw_only=True)
class SpatialSupportManifest:
    """Authoritative metadata of a persisted semantic spatial support.

    Attributes:
        run_id: Identity of the derivation.
        source_context_map_id: The ContextMap whose entities were placed.
        source_map_id: The raw map the entities and the voxel grid share.
        grid_fingerprint: The voxel grid's policy fingerprint.
        policy_fingerprint: Identity of the join rule this run applied.
        aggregation_rule: The rule the underlying voxel aggregation declares.
        frame_id: The frame every cell is expressed in.
        cell_count: Supported occupied voxels.
        entity_count: Distinct entities that support at least one cell.
        total_support_weight: Entity-point attributions summed over every cell.
        code_version: Code revision that derived the support, when known.
        schema_version: Artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash.
    """

    run_id: SpatialSupportRunId
    source_context_map_id: str
    source_map_id: MapId
    grid_fingerprint: str
    policy_fingerprint: str
    aggregation_rule: str
    frame_id: FrameId
    cell_count: int
    entity_count: int
    total_support_weight: int
    code_version: str | None
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def environment_record(code_version: str | None) -> dict[str, Any]:
    """Describe the interpreter and platform the derivation ran on.

    Args:
        code_version: Code revision that derived the support, when known.

    Returns:
        The ``environment.json`` document.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_version": code_version,
    }


def load_manifest(run_dir: Path) -> SpatialSupportManifest:
    """Read and decode the manifest of a finalized run.

    Args:
        run_dir: The run directory.

    Returns:
        The manifest.

    Raises:
        SpatialSupportArtifactError: If the manifest is missing or declares a
            schema version this package does not understand.
    """
    path = run_dir / MANIFEST
    if not path.is_file():
        raise SpatialSupportArtifactError(f"missing {MANIFEST} in {run_dir}")
    raw = read_json_document(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SpatialSupportArtifactError(
            f"unsupported semantic spatial support schema_version: {raw.get('schema_version')!r}"
        )
    return SpatialSupportManifest(
        run_id=SpatialSupportRunId(raw["run_id"]),
        source_context_map_id=raw["source_context_map_id"],
        source_map_id=MapId(raw["source_map_id"]),
        grid_fingerprint=raw["grid_fingerprint"],
        policy_fingerprint=raw["policy_fingerprint"],
        aggregation_rule=raw["aggregation_rule"],
        frame_id=FrameId(raw["frame_id"]),
        cell_count=raw["cell_count"],
        entity_count=raw["entity_count"],
        total_support_weight=raw["total_support_weight"],
        code_version=raw["code_version"],
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        file_inventory=tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )
