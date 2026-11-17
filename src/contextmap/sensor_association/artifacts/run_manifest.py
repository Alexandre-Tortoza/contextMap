"""Formato persistido de uma execução de Sensor Association.

Define schema, layout, manifest e codecs compartilhados pela escrita e pela leitura.
Não executa associação nem publica diretórios; a leitura do manifest valida a versão do schema.
"""

from __future__ import annotations

import json
import sys
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SequenceArtifactId
from contextmap.shared import FileEntry

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCHEMA_VERSION = "0.3.0"
"""Sensor Association run artifact schema version written and understood by this module.

``0.2.0`` is the candidate-selection schema: the manifest gained the required
``candidate_policy``, and every projection record gained ``candidates`` and ``stage_counts``.
The single ``point_count`` is **gone**: it meant the map's size in ``0.1.0`` and would mean the
evaluated population here, which a reader could not tell apart, so both concepts are named
explicitly as ``candidates.map_point_count`` and ``candidates.candidate_count``.
``geometry-support.u32`` and the dense ``eligible_indices`` now hold global
geometry indices explicitly rather than candidate rows that happened to coincide with them.
A ``0.1.0`` artifact is refused with a clear error instead of being migrated: during ``v0.x``
a run is re-executed, never rewritten.
``0.3.0`` replaces monolithic ``perception_run_ids`` with independent selected
``region_artifact_ids`` and ``visual_artifact_ids``.
"""

SensorAssociationRunId = NewType("SensorAssociationRunId", str)
"""Identity of one Sensor Association run, local to its capability and sequence."""

MANIFEST = "manifest.json"
OBSERVATIONS = "outputs/spatial-observations.jsonl"
OBSERVATION_INDEX = "outputs/observation-index.jsonl"
SUPPORT = "outputs/geometry-support.u32"
QUALITY = "outputs/observation-quality.jsonl"
PROJECTION = "outputs/projection-records.jsonl"
VISIBILITY = "outputs/visibility-records.jsonl"
DENSE_INDEX = "outputs/dense-feature-associations.jsonl"
DENSE_CELLS = "outputs/dense-feature-cells.bin"
DIAGNOSTICS = "metrics/frame-diagnostics.jsonl"
SUMMARY = "metrics/summary.json"
RUNTIME = "metrics/runtime.json"
GEOMETRY_INFIX = "--geom-"
U32_INDEX_LIMIT = 2**32
"""Global geometry indices must stay below this to fit the ``uint32`` support and dense tables."""


class RunArtifactError(Exception):
    """Base class for Sensor Association run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class SensorAssociationDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and metrics only.
        STANDARD: Adds per-point samples and the distributions of the associated support.
        FULL: Adds the state-colored projection overlays, the dense sampling coordinates and
            the exact feature sources.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class SensorAssociationRunManifest:
    """Authoritative metadata of a persisted Sensor Association run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        sequence_artifact_id: Canonical sequence artifact consumed.
        selection_id: Deterministic identity of the sequence selection.
        geometric_map_id: The geometric map the observations reference.
        trajectory_id: The trajectory that placed the camera.
        state_estimation_run_id: The state-estimation run it came from, when there is one.
        region_artifact_ids: Selected regional evidence artifacts.
        visual_artifact_ids: Optional selected dense visual artifacts.
        calibration_identity: Hash of the calibration used.
        candidate_policy: The candidate policy record, with its fingerprint: which map
            geometry each frame evaluated before projection.
        visibility_policy: The occlusion policy record, with its fingerprint.
        membership_policy_id: The configured membership policy (``mask-membership-v1``, or
            ``mask-or-box-depth-component-v1`` when the box rule was enabled); the rule applied
            to each observation is in its provenance.
        definitions: Versions of the coverage, quality, diagnostics and sampling definitions.
        pose_policy: The pose lookup policy applied.
        tolerances: The diagnostic tolerances applied.
        dense_channels: Each declared dense channel with the exact feature sources it consumed.
        configuration_fingerprint: Hash of the effective configuration.
        code_version: Code revision that produced the run, when known.
        frame_count: Frames that were associated.
        rejected_frame_count: Frames whose pose the lookup policy rejected.
        failed_frame_count: Associated frames whose diagnostics hold a failure.
        observation_count: Spatial observations persisted.
        finding_counts: Diagnostic findings by code.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest and
            ``__DEBUG__/``.
    """

    run_id: SensorAssociationRunId
    run_index: int
    sequence_name: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    geometric_map_id: MapId
    trajectory_id: str
    state_estimation_run_id: str | None
    region_artifact_ids: tuple[str, ...]
    visual_artifact_ids: tuple[str, ...]
    calibration_identity: str
    candidate_policy: Mapping[str, Any]
    visibility_policy: Mapping[str, Any]
    membership_policy_id: str
    definitions: Mapping[str, str]
    pose_policy: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    dense_channels: tuple[Mapping[str, Any], ...]
    configuration_fingerprint: str
    code_version: str | None
    frame_count: int
    rejected_frame_count: int
    failed_frame_count: int
    observation_count: int
    finding_counts: Mapping[str, int]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


@dataclass(frozen=True, kw_only=True)
class DenseAssociationRecord:
    """The persisted association of one frame's points to one dense feature channel.

    Attributes:
        source_observation_id: The camera frame.
        channel_id: The evidence channel.
        provenance: The full lineage of the association, as JSON primitives.
        terms: Cells read per point: ``1`` for nearest and ``4`` for bilinear.
        eligible_indices: Global geometry indices of the eligible points, the identity
            :func:`~contextmap.geometric_mapping.geometry_id_for` uses.
        sampled: Whether the grid served each eligible point.
        cell_rows: Row of each cell read, ``terms`` per point, ``-1`` where out of support.
        cell_cols: Column of each cell read.
        weights: Weight of each cell read.
    """

    source_observation_id: str
    channel_id: str
    provenance: Mapping[str, Any]
    terms: int
    eligible_indices: array[int]
    sampled: tuple[bool, ...]
    cell_rows: array[int]
    cell_cols: array[int]
    weights: array[float]


def load_manifest(run_dir: Path) -> SensorAssociationRunManifest:
    """Read the manifest and reject missing metadata or an unsupported schema.

    Raises:
        IncompleteRunArtifactError: If the manifest is missing.
        RunArtifactError: If the schema version is unsupported.
    """
    manifest_path = run_dir / MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    return SensorAssociationRunManifest(
        run_id=SensorAssociationRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        sequence_artifact_id=SequenceArtifactId(raw["sequence_artifact_id"]),
        selection_id=raw["selection_id"],
        geometric_map_id=MapId(raw["geometric_map_id"]),
        trajectory_id=raw["trajectory_id"],
        state_estimation_run_id=raw["state_estimation_run_id"],
        region_artifact_ids=tuple(raw["region_artifact_ids"]),
        visual_artifact_ids=tuple(raw["visual_artifact_ids"]),
        calibration_identity=raw["calibration_identity"],
        candidate_policy=dict(raw["candidate_policy"]),
        visibility_policy=dict(raw["visibility_policy"]),
        membership_policy_id=raw["membership_policy_id"],
        definitions=dict(raw["definitions"]),
        pose_policy=dict(raw["pose_policy"]),
        tolerances=dict(raw["tolerances"]),
        dense_channels=tuple(raw["dense_channels"]),
        configuration_fingerprint=raw["configuration_fingerprint"],
        code_version=raw["code_version"],
        frame_count=raw["frame_count"],
        rejected_frame_count=raw["rejected_frame_count"],
        failed_frame_count=raw["failed_frame_count"],
        observation_count=raw["observation_count"],
        finding_counts=dict(raw["finding_counts"]),
        debug_level=raw["debug_level"],
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


def require_contractual(relative_path: str, suffix: str) -> None:
    """Require a record path in outputs or metrics with the declared suffix.

    Raises:
        RunArtifactError: If the path is not a contractual record.
    """
    if not relative_path.endswith(suffix) or not relative_path.startswith(("outputs/", "metrics/")):
        raise RunArtifactError(f"not a contractual {suffix} record: {relative_path!r}")


def as_u32_indices(global_indices: NDArray[Any], *, table: str) -> NDArray[Any]:
    """Return global geometry indices as the little-endian ``uint32`` a table stores.

    Raises:
        RunArtifactError: If an index does not fit in 32 bits. The cast would wrap it onto
            another element's identity, so the persisted support would silently name the wrong
            geometry; the schema has no wider index, so such a map is refused instead.
    """
    import numpy as np

    if global_indices.size and int(global_indices.max()) >= U32_INDEX_LIMIT:
        raise RunArtifactError(
            f"global geometry index {int(global_indices.max())} does not fit the 32-bit {table} "
            f"table of schema {SCHEMA_VERSION}: casting it would wrap it onto another element"
        )
    return np.asarray(global_indices, dtype="<u4")


def decode_array(typecode: str, data: bytes) -> array[Any]:
    """Decode little-endian 4-byte items with the standard library alone."""
    values = array(typecode)
    if values.itemsize != 4:
        raise RunArtifactError(f"the platform's {typecode!r} items are not 4 bytes wide")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def encode_json(record: Mapping[str, Any]) -> str:
    """Encode a sorted, indented JSON document with its final newline."""
    return json.dumps(record, indent=2, sort_keys=True) + "\n"
