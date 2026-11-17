"""Schema, configuration, and manifest types for Geometric Mapping run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

from contextmap.geometric_mapping.models import MapId
from contextmap.ingestion import FrameId, SequenceArtifactId
from contextmap.shared import FileEntry
from contextmap.state_estimation import StateEstimationRunId, TrajectoryId

SCHEMA_VERSION = "0.1.0"
"""Geometric Mapping run artifact schema version written and understood by this module."""

GeometricMapRunId = NewType("GeometricMapRunId", str)
"""Identity of one Geometric Mapping run, local to its capability and sequence."""

# Artifact layout constants
_MANIFEST = "manifest.json"
_LINEAGE = "lineage.json"
_CONFIG = "config.json"
_ENVIRONMENT = "environment.json"
_GEOMETRY = "outputs/geometry.bin"
_SOURCE_INDEX = "outputs/source-index.jsonl"
_MAP_METADATA = "outputs/map-metadata.json"
_INPUT_PLAN = "metrics/input-plan.json"
_MAPPING = "metrics/mapping.json"
_RUNTIME = "metrics/runtime.json"

_CONTRACTUAL_RECORDS = frozenset({_LINEAGE, _CONFIG, _ENVIRONMENT})
_CONTRACTUAL_DIRECTORIES = ("outputs/", "metrics/")

# Amostras de traces de transformação: quantos scans e quantos pontos por scan.
_STANDARD_TRACE_SCANS, _STANDARD_TRACE_POINTS = 5, 2
_FULL_TRACE_SCANS, _FULL_TRACE_POINTS = 50, 3


class MapArtifactError(Exception):
    """Base class for Geometric Mapping run artifact read/write failures."""


class IncompleteMapArtifactError(MapArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class MapDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds scan and bounds summaries, the trajectory over the map and
            a small sample of transform traces.
        FULL: Adds a denser sample of traces and their reconstruction residuals.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


@dataclass(frozen=True, kw_only=True)
class GeometricMapArtifactManifest:
    """Authoritative metadata of a persisted Geometric Mapping run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the mapped sequence.
        map_id: Identity of the map; part of every geometry reference.
        map_frame: The global frame every coordinate is expressed in.
        sequence_artifact_id: Canonical sequence artifact consumed.
        selection_id: Deterministic identity of the sequence selection.
        trajectory_id: Trajectory whose poses placed the geometry.
        state_estimation_run_id: The State Estimation run behind the trajectory, if known.
        calibration_identity: Hash of the calibration the extrinsics came from.
        configuration_fingerprint: Hash of the effective mapping configuration.
        code_version: Code revision that produced the run, when known.
        point_count: Geometry elements persisted.
        source_point_count: Finite points measured before any aggregation.
        scan_count: Scans accumulated into the map.
        rejected_scan_count: Selected scans left out, with reasons in the metrics.
        aggregation_rule: The explicit aggregation rule; ``None`` when every point
            is a raw measurement.
        spatial_index_kind: Identity of the derived spatial index.
        clock_id: Clock domain of the time range.
        start_time_ns: Acquisition time of the earliest contributing scan.
        end_time_ns: Acquisition time of the latest contributing scan.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the
            manifest, the README and ``__DEBUG__/``.
    """

    run_id: GeometricMapRunId
    run_index: int
    sequence_name: str
    map_id: MapId
    map_frame: FrameId
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str | None
    configuration_fingerprint: str
    code_version: str | None
    point_count: int
    source_point_count: int
    scan_count: int
    rejected_scan_count: int
    aggregation_rule: str | None
    spatial_index_kind: str
    clock_id: str
    start_time_ns: int
    end_time_ns: int
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def decode_manifest(raw: dict[str, Any]) -> GeometricMapArtifactManifest:
    """Build a manifest from its parsed ``manifest.json`` record.

    Args:
        raw: Parsed ``manifest.json`` content.

    Returns:
        The typed manifest.

    Raises:
        MapArtifactError: If the schema version is not understood.
    """
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MapArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    run_id = raw["state_estimation_run_id"]
    return GeometricMapArtifactManifest(
        run_id=GeometricMapRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        map_id=MapId(raw["map_id"]),
        map_frame=FrameId(raw["map_frame"]),
        sequence_artifact_id=SequenceArtifactId(raw["sequence_artifact_id"]),
        selection_id=raw["selection_id"],
        trajectory_id=TrajectoryId(raw["trajectory_id"]),
        state_estimation_run_id=None if run_id is None else StateEstimationRunId(run_id),
        calibration_identity=raw["calibration_identity"],
        configuration_fingerprint=raw["configuration_fingerprint"],
        code_version=raw["code_version"],
        point_count=raw["point_count"],
        source_point_count=raw["source_point_count"],
        scan_count=raw["scan_count"],
        rejected_scan_count=raw["rejected_scan_count"],
        aggregation_rule=raw["aggregation_rule"],
        spatial_index_kind=raw["spatial_index_kind"],
        clock_id=raw["clock_id"],
        start_time_ns=raw["start_time_ns"],
        end_time_ns=raw["end_time_ns"],
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
