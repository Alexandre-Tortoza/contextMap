"""Contrato do manifest e nomes de layout de um SequenceArtifact.

O manifest é a metadata autoritativa do artifact: identidade, schema, sequência, contagens e
inventário de arquivos. Este módulo define a forma persistida e rejeita versões incompatíveis sem
carregar observações ou aplicar regras de domínio downstream.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType

SCHEMA_VERSION = "0.2.0"
"""Sequence artifact schema version written and understood by ingestion."""

SequenceArtifactId = NewType("SequenceArtifactId", str)
"""Identity of a persisted canonical sequence artifact."""

_MANIFEST_FILENAME = "manifest.json"
_INDEX_FILENAME = "index.jsonl"
_CALIBRATION_FILENAME = "calibration/calibration.json"
_PROVENANCE_FILENAME = "provenance/provenance.json"
_DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics/summary.json"
_DIAGNOSTICS_WARNINGS_FILENAME = "diagnostics/warnings.jsonl"
_DIAGNOSTICS_SYNCHRONIZATION_FILENAME = "diagnostics/synchronization.jsonl"
_DIAGNOSTICS_DROPPED_EVENTS_FILENAME = "diagnostics/dropped-events.jsonl"
_DIAGNOSTICS_FRAME_GRAPH_FILENAME = "diagnostics/frame-graph.json"


class SequenceArtifactError(Exception):
    """Base class for sequence artifact read/write failures."""


class IncompleteSequenceArtifactError(SequenceArtifactError):
    """Raised when a directory does not contain a complete, valid artifact."""


@dataclass(frozen=True)
class SequenceArtifactFileEntry:
    """One entry in a sequence artifact's file inventory.

    Attributes:
        path: Path relative to the artifact root, using ``/`` separators.
        size_bytes: Size of the file in bytes.
        content_hash: Content hash as ``"sha256:<hex digest>"``.
    """

    path: str
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class SequenceArtifactManifest:
    """Authoritative metadata for a canonical sequence artifact.

    Full provenance/content-identity semantics (detecting equivalent
    re-ingestion, source content hashing, configuration hashing) are added
    by the sequence provenance and integrity contracts; this manifest only
    carries the structural identity of the artifact itself.

    Attributes:
        artifact_id: Identity of this artifact.
        sequence_name: Name of the sequence this artifact belongs to.
        schema_version: Sequence artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        observation_counts: Number of observations per modality
            ("image", "lidar", "imu", "external_pose").
        file_inventory: Every payload/index file this artifact contains,
            excluding ``manifest.json`` itself.
    """

    artifact_id: SequenceArtifactId
    sequence_name: str
    schema_version: str
    created_at: str
    observation_counts: Mapping[str, int]
    file_inventory: Sequence[SequenceArtifactFileEntry]


def _manifest_fields(manifest: SequenceArtifactManifest) -> dict[str, Any]:
    """Return the manifest fields this module owns; the publication adds ``file_inventory``."""
    return {
        "artifact_id": str(manifest.artifact_id),
        "sequence_name": manifest.sequence_name,
        "schema_version": manifest.schema_version,
        "created_at": manifest.created_at,
        "observation_counts": dict(manifest.observation_counts),
    }


def _load_manifest(artifact_dir: Path) -> SequenceArtifactManifest:
    """Load and validate the authoritative manifest of one sequence artifact.

    Args:
        artifact_dir: Directory expected to contain a finalized SequenceArtifact.

    Returns:
        Decoded manifest for the supported schema version.

    Raises:
        IncompleteSequenceArtifactError: If manifest.json is absent.
        SequenceArtifactError: If the artifact schema version is unsupported.
    """
    manifest_path = artifact_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise IncompleteSequenceArtifactError(f"missing {_MANIFEST_FILENAME} in {artifact_dir}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SequenceArtifactError(
            f"unsupported sequence artifact schema_version: {schema_version!r}"
        )

    return SequenceArtifactManifest(
        artifact_id=SequenceArtifactId(raw["artifact_id"]),
        sequence_name=raw["sequence_name"],
        schema_version=schema_version,
        created_at=raw["created_at"],
        observation_counts=dict(raw["observation_counts"]),
        file_inventory=tuple(
            SequenceArtifactFileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )
