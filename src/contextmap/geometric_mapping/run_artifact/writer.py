"""Escrita e publicação atômica do artifact de Geometric Mapping.

Persiste geometria, lineage e métricas; diagnósticos humanos ficam em __DEBUG__.
Não altera o formato contratual nem as regras científicas de construção do mapa.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import platform
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping.accumulation import (
    SCAN_BOUNDS_INDEX,
    GeometryBuildSummary,
    ScanVoxelPolicy,
    accumulate_plan,
)
from contextmap.geometric_mapping.deskew import encode_deskew_policy
from contextmap.geometric_mapping.geometry_storage import PACKED_POINT, PackedGeometry, ScanRecord
from contextmap.geometric_mapping.inputs import GeometryMappingPlan
from contextmap.geometric_mapping.models import (
    GeometricMap,
    GeometryReference,
    MapId,
    geometry_id_for,
)
from contextmap.geometric_mapping.serialization import (
    encode_bounds,
    encode_geometric_map,
    encode_lookup_policy,
    encode_motion_correction_policy,
    encode_scan_record,
    encode_transform_trace,
)
from contextmap.geometric_mapping.transformation import (
    TransformTrace,
    transform_trace_residual_m,
    verify_transform_trace,
)
from contextmap.ingestion import (
    encode_selection,
)
from contextmap.shared import DEBUG_DIRECTORY, AtomicRunDirectory, RunDirectoryError
from contextmap.state_estimation import (
    summarize_lookups,
)

from .manifest import (
    _CONFIG,
    _ENVIRONMENT,
    _FULL_TRACE_POINTS,
    _FULL_TRACE_SCANS,
    _GEOMETRY,
    _INPUT_PLAN,
    _LINEAGE,
    _MANIFEST,
    _MAP_METADATA,
    _MAPPING,
    _RUNTIME,
    _SOURCE_INDEX,
    _STANDARD_TRACE_POINTS,
    _STANDARD_TRACE_SCANS,
    SCHEMA_VERSION,
    GeometricMapArtifactManifest,
    GeometricMapRunId,
    MapArtifactError,
    MapDebugLevel,
    decode_manifest,
)


def encode_mapping_configuration(
    plan: GeometryMappingPlan, aggregation: ScanVoxelPolicy | None
) -> dict[str, Any]:
    """Describe the effective mapping configuration of a run.

    Everything that changes the persisted geometry appears here; the debug level,
    which does not, does not.

    Args:
        plan: The assembled geometry inputs.
        aggregation: The explicit aggregation rule, or ``None`` for raw points.
    """
    configuration: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "map_frame": str(plan.sources.map_frame),
        "body_frame": str(plan.sources.body_frame),
        "pose_lookup": encode_lookup_policy(plan.policy.pose_lookup),
        "motion_correction_policy": encode_motion_correction_policy(
            plan.policy.motion_correction_policy
        ),
        "aggregation": None
        if aggregation is None
        else {
            "kind": "scan_voxel_centroid",
            "cell_m": aggregation.cell_m,
            "rule": aggregation.rule,
        },
        "packed_point": {"format": PACKED_POINT.format, "size_bytes": PACKED_POINT.size},
        "spatial_index": {
            "kind": SCAN_BOUNDS_INDEX.kind,
            "is_derived": SCAN_BOUNDS_INDEX.is_derived,
        },
    }
    if plan.policy.deskew is not None:
        configuration["deskew"] = encode_deskew_policy(plan.policy.deskew)
    return configuration


def mapping_configuration_fingerprint(
    plan: GeometryMappingPlan, aggregation: ScanVoxelPolicy | None
) -> str:
    """Hash the effective mapping configuration.

    Args:
        plan: The assembled geometry inputs.
        aggregation: The explicit aggregation rule, or ``None`` for raw points.

    Returns:
        ``"sha256:<hex>"``: equal for equal configuration, different when the pose
        lookup, the correction policy, the aggregation or the storage format changes.
    """
    payload = json.dumps(encode_mapping_configuration(plan, aggregation), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class GeometricMapArtifactWriter:
    """Builds an immutable Geometric Mapping run artifact on the local filesystem."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: GeometricMapRunId,
        run_index: int,
        debug_level: MapDebugLevel = MapDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/geometric_mapping``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run maps.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run among the caller's runs of the same sequence,
                supplied by the caller and recorded as given.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._debug_level = debug_level
        self._final_dir = output_dir
        self._finalized = False

    def finalize(
        self,
        *,
        plan: GeometryMappingPlan,
        aggregation: ScanVoxelPolicy | None,
        code_version: str | None,
        runtime_s: float | None = None,
        peak_memory_bytes: int | None = None,
    ) -> GeometricMapArtifactManifest:
        """Transform, accumulate and persist a run atomically.

        The geometry is streamed to disk one scan at a time, so a map larger
        than memory can be written.

        Args:
            plan: The assembled geometry inputs.
            aggregation: The explicit aggregation rule, or ``None`` to persist
                every point as a raw measurement.
            code_version: Code revision that builds the map, when known.
            runtime_s: Wall-clock time of the run, when measured. Recorded as a
                metric apart from every quality measure.
            peak_memory_bytes: Peak memory of the run, when measured.

        Returns:
            The manifest of the finalized run.

        Raises:
            MapArtifactError: If already finalized or a run already exists at the
                target path.
            GeometryTransformError: If a scan cannot be transformed.
            AccumulationError: If the scans cannot form one map (for example,
                none produced geometry). Nothing is left on disk.
        """
        if self._finalized:
            raise MapArtifactError("writer already finalized")
        map_id = MapId(f"{self._sequence_name}--{self._run_id}")
        fingerprint = mapping_configuration_fingerprint(plan, aggregation)

        try:
            with AtomicRunDirectory(self._final_dir) as run:
                with run.open_binary(_GEOMETRY) as sink:
                    accumulated = accumulate_plan(
                        plan,
                        map_id=map_id,
                        sink=sink,
                        aggregation=aggregation,
                        configuration_fingerprint=fingerprint,
                        code_version=code_version,
                    )
                self._write_outputs(run, accumulated)
                self._write_lineage(run, plan, code_version)
                self._write_configuration(run, plan, aggregation, fingerprint)
                self._write_environment(run, code_version)
                self._write_metrics(run, plan, accumulated, runtime_s, peak_memory_bytes)
                self._write_debug(run, plan, accumulated)
                run.publish(
                    manifest=self._manifest_record(plan, accumulated, fingerprint, code_version),
                    readme=_render_readme(self._run_id, self._run_index, accumulated),
                )
        except RunDirectoryError as error:
            raise MapArtifactError(str(error)) from error

        self._finalized = True
        manifest_path = self._final_dir / _MANIFEST
        if not manifest_path.is_file():
            raise MapArtifactError(f"missing {_MANIFEST} in {self._final_dir}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return decode_manifest(raw)

    def _write_outputs(self, run: AtomicRunDirectory, accumulated: GeometryBuildSummary) -> None:
        run.write_text(
            _SOURCE_INDEX, _lines(encode_scan_record(scan) for scan in accumulated.scans)
        )
        run.write_text(_MAP_METADATA, _json(encode_geometric_map(accumulated.geometric_map)))

    def _write_lineage(
        self, run: AtomicRunDirectory, plan: GeometryMappingPlan, code_version: str | None
    ) -> None:
        run.write_text(
            _LINEAGE,
            _json(
                {
                    "sequence_artifact_id": str(plan.sources.sequence_artifact_id),
                    "selection": encode_selection(plan.sources.selection),
                    "selection_id": plan.sources.selection_id,
                    "state_estimation_run_id": None
                    if plan.sources.state_estimation_run_id is None
                    else str(plan.sources.state_estimation_run_id),
                    "trajectory_id": str(plan.sources.trajectory_id),
                    "calibration_identity": plan.sources.calibration_identity,
                    "map_frame": str(plan.sources.map_frame),
                    "body_frame": str(plan.sources.body_frame),
                    "code_version": code_version,
                    "source_observation_ids": [str(item.observation_id) for item in plan.inputs],
                    "rejected_observation_ids": [
                        str(item.observation_id) for item in plan.rejections
                    ],
                }
            ),
        )

    def _write_configuration(
        self,
        run: AtomicRunDirectory,
        plan: GeometryMappingPlan,
        aggregation: ScanVoxelPolicy | None,
        fingerprint: str,
    ) -> None:
        run.write_text(
            _CONFIG,
            _json(
                {
                    **encode_mapping_configuration(plan, aggregation),
                    "debug_level": self._debug_level.value,
                    "configuration_fingerprint": fingerprint,
                }
            ),
        )

    def _write_environment(self, run: AtomicRunDirectory, code_version: str | None) -> None:
        run.write_text(
            _ENVIRONMENT,
            _json(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": _installed_version("numpy"),
                    "code_version": code_version,
                }
            ),
        )

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        plan: GeometryMappingPlan,
        accumulated: GeometryBuildSummary,
        runtime_s: float | None,
        peak_memory_bytes: int | None,
    ) -> None:
        run.write_text(_INPUT_PLAN, _json(_encode_input_plan_metrics(plan)))
        run.write_text(_MAPPING, _json(_encode_mapping_metrics(accumulated)))
        if runtime_s is not None or peak_memory_bytes is not None:
            measured: dict[str, Any] = {}
            if runtime_s is not None:
                measured["runtime_s"] = runtime_s
            if peak_memory_bytes is not None:
                measured["peak_memory_bytes"] = peak_memory_bytes
            run.write_text(_RUNTIME, _json(measured))

    def _write_debug(
        self, run: AtomicRunDirectory, plan: GeometryMappingPlan, accumulated: GeometryBuildSummary
    ) -> None:
        if self._debug_level is MapDebugLevel.NONE:
            return
        scans = accumulated.scans
        run.write_text(
            f"{DEBUG_DIRECTORY}/scans-by-time.jsonl",
            _lines(
                {
                    "ordinal": scan.ordinal,
                    "observation_id": str(scan.observation_id),
                    "timestamp_ns": scan.acquisition_timestamp.total_nanoseconds(),
                    "geometry_count": scan.geometry_count,
                    "dropped_non_finite_count": scan.dropped_non_finite_count,
                    "bounds": None if scan.bounds is None else encode_bounds(scan.bounds),
                }
                for scan in scans
            ),
            contractual=False,
        )
        run.write_text(
            f"{DEBUG_DIRECTORY}/trajectory-over-map.csv",
            "ordinal,timestamp_ns,x_m,y_m,z_m\n"
            + "".join(
                f"{ordinal},{item.timestamp.total_nanoseconds()},"
                f"{item.pose.pose.translation_m[0]!r},{item.pose.pose.translation_m[1]!r},"
                f"{item.pose.pose.translation_m[2]!r}\n"
                for ordinal, item in enumerate(plan.inputs)
            ),
            contractual=False,
        )
        run.write_text(
            f"{DEBUG_DIRECTORY}/bounds-summary.json",
            _json(_encode_bounds_summary(accumulated)),
            contractual=False,
        )
        run.write_text(
            f"{DEBUG_DIRECTORY}/warnings.jsonl", _lines(_warnings(plan)), contractual=False
        )

        full = self._debug_level is MapDebugLevel.FULL
        max_scans, per_scan = (
            (_FULL_TRACE_SCANS, _FULL_TRACE_POINTS)
            if full
            else (_STANDARD_TRACE_SCANS, _STANDARD_TRACE_POINTS)
        )
        traces = self._sample_traces(run, accumulated, max_scans=max_scans, per_scan=per_scan)
        run.write_text(
            f"{DEBUG_DIRECTORY}/selected-transform-traces.jsonl",
            _lines(encode_transform_trace(trace) for trace in traces),
            contractual=False,
        )
        if full:
            run.write_text(
                f"{DEBUG_DIRECTORY}/trace-residuals.jsonl",
                _lines(
                    {
                        "source_observation_id": str(trace.source_observation_id),
                        "source_point_index": trace.source_point_index,
                        "residual_m": transform_trace_residual_m(trace),
                        "problems": verify_transform_trace(trace),
                    }
                    for trace in traces
                ),
                contractual=False,
            )

    def _sample_traces(
        self,
        run: AtomicRunDirectory,
        accumulated: GeometryBuildSummary,
        *,
        max_scans: int,
        per_scan: int,
    ) -> list[TransformTrace]:
        """Trace a deterministic sample of persisted points, read back from the payload."""
        indices = _sample_geometry_indices(
            accumulated.scans, max_scans=max_scans, per_scan=per_scan
        )
        with run.written_path(_GEOMETRY).open("rb") as handle:
            payload = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            geometry = PackedGeometry(
                geometric_map=accumulated.geometric_map, scans=accumulated.scans, records=payload
            )
            try:
                map_id = accumulated.geometric_map.map_id
                return [
                    geometry.trace(
                        GeometryReference(
                            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
                        )
                    )
                    for index in indices
                ]
            finally:
                geometry.close()
                payload.close()

    def _manifest_record(
        self,
        plan: GeometryMappingPlan,
        accumulated: GeometryBuildSummary,
        fingerprint: str,
        code_version: str | None,
    ) -> dict[str, Any]:
        geometric_map = accumulated.geometric_map
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "map_id": str(geometric_map.map_id),
            "map_frame": str(geometric_map.frame_id),
            "sequence_artifact_id": str(plan.sources.sequence_artifact_id),
            "selection_id": plan.sources.selection_id,
            "trajectory_id": str(plan.sources.trajectory_id),
            "state_estimation_run_id": None
            if plan.sources.state_estimation_run_id is None
            else str(plan.sources.state_estimation_run_id),
            "calibration_identity": plan.sources.calibration_identity,
            "configuration_fingerprint": fingerprint,
            "code_version": code_version,
            "point_count": geometric_map.point_count,
            "source_point_count": accumulated.source_point_count,
            "scan_count": len(accumulated.scans),
            "rejected_scan_count": len(plan.rejections),
            "aggregation_rule": geometric_map.aggregation_rule,
            "spatial_index_kind": SCAN_BOUNDS_INDEX.kind,
            "clock_id": geometric_map.time_bounds.start.clock_id,
            "start_time_ns": geometric_map.time_bounds.start.total_nanoseconds(),
            "end_time_ns": geometric_map.time_bounds.end.total_nanoseconds(),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }


def _json(record: Mapping[str, Any]) -> str:
    """Serialize a record to pretty-printed JSON with sorted keys."""
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def _lines(records: Iterable[Any]) -> str:
    """Serialize an iterable of records as JSONL."""
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _installed_version(package: str) -> str | None:
    """Get the installed version of a package, or None if not found."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _encode_input_plan_metrics(plan: GeometryMappingPlan) -> dict[str, Any]:
    """Encode metrics about the input plan and selection."""
    lookups = summarize_lookups(item.pose for item in plan.inputs)
    states = Counter(item.motion_correction.record.state.value for item in plan.inputs)
    dispositions = Counter(item.motion_correction.disposition.value for item in plan.inputs)
    pose_rejections = sum(
        1 for item in plan.rejections if item.reason.value == "pose_lookup_rejected"
    )
    return {
        "selected_scan_count": len(plan.inputs) + len(plan.rejections),
        "accepted_scan_count": len(plan.inputs),
        "rejected_scan_count": len(plan.rejections),
        "rejections": [
            {
                "observation_id": str(item.observation_id),
                "reason": item.reason.value,
                "detail": item.detail,
            }
            for item in plan.rejections
        ],
        "ignored_observation_counts": dict(plan.ignored_observation_counts),
        "pose_lookup": {
            "lookup_count": lookups.lookup_count,
            "exact_count": lookups.exact_count,
            "nearest_count": lookups.nearest_count,
            "interpolated_count": lookups.interpolated_count,
            "min_time_delta_ns": lookups.min_time_delta_ns,
            "median_time_delta_ns": lookups.median_time_delta_ns,
            "max_time_delta_ns": lookups.max_time_delta_ns,
        },
        "pose_lookup_rejected_count": pose_rejections,
        "motion_correction": {
            "policy": encode_motion_correction_policy(plan.policy.motion_correction_policy),
            "state_counts": dict(states),
            "disposition_counts": dict(dispositions),
            "warnings": [
                item.motion_correction.message
                for item in plan.inputs
                if item.motion_correction.message is not None
            ],
        },
    }


def _encode_mapping_metrics(accumulated: GeometryBuildSummary) -> dict[str, Any]:
    """Encode metrics about the accumulated map."""
    geometric_map = accumulated.geometric_map
    return {
        "map_id": str(geometric_map.map_id),
        "scan_count": len(accumulated.scans),
        "scans_without_geometry": sum(1 for scan in accumulated.scans if scan.geometry_count == 0),
        "source_point_count": accumulated.source_point_count,
        "dropped_non_finite_count": sum(
            scan.dropped_non_finite_count for scan in accumulated.scans
        ),
        "point_count": geometric_map.point_count,
        "aggregation_rule": geometric_map.aggregation_rule,
        "reduction_ratio": geometric_map.point_count / accumulated.source_point_count,
        "bounds": encode_bounds(geometric_map.bounds),
        "time_bounds": {
            "start": geometric_map.time_bounds.start.to_record(),
            "end": geometric_map.time_bounds.end.to_record(),
        },
        "geometry_size_bytes": geometric_map.point_count * PACKED_POINT.size,
    }


def _encode_bounds_summary(accumulated: GeometryBuildSummary) -> dict[str, Any]:
    """Encode a summary of the map bounds and point distribution."""
    geometric_map = accumulated.geometric_map
    low, high = geometric_map.bounds.minimum_m, geometric_map.bounds.maximum_m
    counts = sorted(scan.geometry_count for scan in accumulated.scans)
    return {
        "map_bounds": encode_bounds(geometric_map.bounds),
        "extent_m": [high[axis] - low[axis] for axis in range(3)],
        "point_count": geometric_map.point_count,
        "points_per_scan": {
            "minimum": counts[0],
            "median": counts[len(counts) // 2],
            "maximum": counts[-1],
        },
    }


def _warnings(plan: GeometryMappingPlan) -> Iterator[dict[str, Any]]:
    """Yield warning records for rejections and motion correction messages."""
    for rejection in plan.rejections:
        yield {
            "kind": "rejected_scan",
            "observation_id": str(rejection.observation_id),
            "reason": rejection.reason.value,
            "detail": rejection.detail,
        }
    for item in plan.inputs:
        if item.motion_correction.message is not None:
            yield {
                "kind": "motion_correction",
                "observation_id": str(item.observation_id),
                "message": item.motion_correction.message,
            }


def _sample_geometry_indices(
    scans: Sequence[ScanRecord], *, max_scans: int, per_scan: int
) -> list[int]:
    """Pick geometry indices evenly across the scans that contributed geometry."""
    contributing = [scan for scan in scans if scan.geometry_count > 0]
    count = min(len(contributing), max_scans)
    if count == 0:
        return []
    positions = sorted(
        {round(i * (len(contributing) - 1) / max(count - 1, 1)) for i in range(count)}
    )
    indices: list[int] = []
    for position in positions:
        scan = contributing[position]
        last = scan.geometry_count - 1
        offsets = {0, last // 2, last} if per_scan >= 3 else {0, last} if per_scan == 2 else {0}
        indices.extend(scan.first_geometry_index + offset for offset in sorted(offsets))
    return indices


def _render_readme(
    run_id: GeometricMapRunId, run_index: int, accumulated: GeometryBuildSummary
) -> str:
    """Render a human-readable README for the artifact."""
    geometric_map: GeometricMap = accumulated.geometric_map
    provenance = geometric_map.provenance
    return (
        f"# Geometric mapping run {run_index:04d}\n"
        "\n"
        f"- Run ID: `{run_id}`\n"
        f"- Map: `{geometric_map.map_id}` in frame `{geometric_map.frame_id}`\n"
        f"- Sequence artifact: `{provenance.sequence_artifact_id}`\n"
        f"- Selection: `{provenance.selection_id}`\n"
        f"- Trajectory: `{provenance.trajectory_id}`\n"
        f"- Points: {geometric_map.point_count} from {len(accumulated.scans)} scans\n"
        "\n"
        "Contractual data is in `outputs/`, `metrics/`, `lineage.json`, `config.json` and\n"
        "`environment.json`; `__DEBUG__/` is human evidence and no downstream stage may\n"
        "depend on it.\n"
    )
