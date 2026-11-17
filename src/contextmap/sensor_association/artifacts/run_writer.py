"""Escrita streaming e publicação atômica do artifact de Sensor Association.

A transação recebe um frame por vez, grava referências e medidas e libera o frame.
O formato pertence a run_manifest; evidência de debug não é entrada de consumidores.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from contextmap.sensor_association.artifacts.debug_evidence import (
    write_full_debug,
    write_standard_debug,
)
from contextmap.sensor_association.artifacts.run_manifest import (
    DENSE_CELLS,
    DENSE_INDEX,
    DIAGNOSTICS,
    OBSERVATION_INDEX,
    OBSERVATIONS,
    PROJECTION,
    QUALITY,
    RUNTIME,
    SCHEMA_VERSION,
    SUMMARY,
    SUPPORT,
    VISIBILITY,
    RunArtifactError,
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunManifest,
    as_u32_indices,
    encode_json,
    load_manifest,
)
from contextmap.sensor_association.run_definitions import DEFINITION_VERSIONS
from contextmap.sensor_association.serialization import (
    encode_calibration_ref,
    encode_observation_quality,
    encode_pose_ref,
    encode_spatial_observation,
)
from contextmap.sensor_association.service import FrameAssociation, SensorAssociationOutcome
from contextmap.shared import DEBUG_DIRECTORY, AtomicRunDirectory, RunDirectoryError


class SensorAssociationRunWriter:
    """Builds an immutable Sensor Association run artifact on the local filesystem.

    The writer holds the run's identity and opens a :meth:`transaction`; the transaction is
    what touches the filesystem, and it is a
    :class:`~contextmap.sensor_association.service.FrameSink`, so a run is persisted frame by
    frame instead of from a fully materialized result::

        with writer.transaction() as run:
            outcome = SensorAssociationService().run(request, sink=run)
            manifest = run.finalize(outcome)
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: SensorAssociationRunId,
        run_index: int,
        debug_level: SensorAssociationDebugLevel = SensorAssociationDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/artifacts/sensor_association/<run>``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run in the caller's run namespace,
                supplied by the caller and recorded as given.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._debug_level = debug_level
        self._final_dir = output_dir
        self._opened = False

    @property
    def run_id(self) -> SensorAssociationRunId:
        """Identity of the run this writer persists."""
        return self._run_id

    @property
    def run_index(self) -> int:
        """Ordinal of the run in the caller's run namespace."""
        return self._run_index

    @property
    def sequence_name(self) -> str:
        """Name of the sequence the run processed."""
        return self._sequence_name

    @property
    def debug_level(self) -> SensorAssociationDebugLevel:
        """How much non-contractual debug evidence the run persists."""
        return self._debug_level

    @property
    def final_dir(self) -> Path:
        """Where the finished artifact will live."""
        return self._final_dir

    @contextmanager
    def transaction(self) -> Iterator[SensorAssociationRunTransaction]:
        """Open the run, yielding the sink that persists each frame as it arrives.

        Leaving the block without a successful :meth:`SensorAssociationRunTransaction.finalize`
        -- normally or through an exception -- discards everything written: an interrupted run
        never leaves a publishable artifact.

        Raises:
            RunArtifactError: If this writer already opened a transaction, or a run already
                exists at the target path.
        """
        if self._opened:
            raise RunArtifactError("writer already finalized")
        self._opened = True
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                transaction = SensorAssociationRunTransaction(run=run, writer=self)
                try:
                    yield transaction
                finally:
                    transaction.close()
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error


class SensorAssociationRunTransaction:
    """One run being written: a frame sink that persists and then forgets each frame.

    Per-frame payloads are appended to open streams and nothing about a frame is retained:
    no array, no observation, no projection. Nothing is visible until :meth:`finalize`
    publishes the run atomically.

    What the transaction does keep is bounded and scalar. Counts and byte offsets are
    constant, and one timing record per frame accumulates to preserve per-frame
    candidate-query and projection times and ``metrics/runtime.json`` is a single document
    written only when the caller measured. That record is four primitives, about 567 B, so
    3096 frames cost under 1.7 MB -- a linear term worth naming rather than hiding, and four
    orders of magnitude below the 4.3 GB peak of the real 350-frame run. The rule it must keep
    obeying is that the linear term holds **scalars**, never a frame or an array.
    """

    def __init__(self, *, run: AtomicRunDirectory, writer: SensorAssociationRunWriter) -> None:
        """Open the run's contractual streams.

        Args:
            run: The atomic directory the run is built in.
            writer: The writer that owns the run's identity and debug level.
        """
        self._run = run
        self._writer = writer
        self._streams = ExitStack()
        self._closed = False
        self._finalized = False
        self._observations = self._open(OBSERVATIONS)
        self._index = self._open(OBSERVATION_INDEX)
        self._support = self._open(SUPPORT)
        self._quality = self._open(QUALITY)
        self._projection = self._open(PROJECTION)
        self._visibility = self._open(VISIBILITY)
        self._diagnostics = self._open(DIAGNOSTICS)
        self._dense_index: BinaryIO | None = None
        self._dense_cells: BinaryIO | None = None
        # Agregados de run: contagens e bytes, nunca arrays de frame.
        self._support_offset = 0
        self._observation_offset = 0
        self._quality_offset = 0
        self._dense_offset = 0
        self._frame_count = 0
        self._observation_count = 0
        self._observations_without_support = 0
        self._failed_frame_count = 0
        self._range_limit_candidate_support_count = 0
        self._min_range_slack_m: float | None = None
        self._finding_counts: dict[str, int] = {}
        self._state_counts: dict[str, int] = {}
        self._dense_sources: dict[str, dict[str, dict[str, Any]]] = {}
        self._debug_feature_sources: list[dict[str, Any]] = []
        self._timings: list[dict[str, Any]] = []

    def _open(self, relative_path: str) -> BinaryIO:
        """Open a staged binary stream and register its closure with the transaction."""
        return self._streams.enter_context(self._run.open_binary(relative_path))

    def accept(self, frame: FrameAssociation) -> None:
        """Persist one completed frame and update the run's aggregates.

        Implements :class:`~contextmap.sensor_association.service.FrameSink`. Nothing about the
        frame is retained: only counts, byte offsets and the frame's own timings survive the
        call.

        Raises:
            RunArtifactError: If the run is already finalized or closed, an observation's
                geometry support disagrees with its region membership, or a persisted global
                geometry index does not fit the 32-bit support or dense tables.
        """
        if self._finalized or self._closed:
            raise RunArtifactError("the run is no longer open for frames")
        self._write_frame_observations(frame)
        self._projection.write(_line(_projection_record(frame)))
        self._visibility.write(_line(_visibility_record(frame)))
        self._diagnostics.write(_line(frame.diagnostics.to_record()))
        if frame.dense_samples:
            self._write_frame_dense(frame)
        self._accumulate(frame)
        self._write_frame_debug(frame)

    def finalize(
        self, outcome: SensorAssociationOutcome, *, runtime_s: float | None = None
    ) -> SensorAssociationRunManifest:
        """Close the streams, write the run-level files and publish the artifact atomically.

        Args:
            outcome: The run's identity, policies and aggregates, as the service returned it.
            runtime_s: Wall-clock time the run took, when measured. It is a metric apart from
                every quality measure, and it never enters ``outputs/``.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, the outcome counts other frames than this
                transaction persisted, or the run cannot be published.
        """
        if self._finalized:
            raise RunArtifactError("writer already finalized")
        if outcome.association_run_id != self._writer.run_id:
            raise RunArtifactError("outcome belongs to another association run")
        if outcome.frame_count != self._frame_count:
            raise RunArtifactError(
                f"the outcome counts {outcome.frame_count} frames but {self._frame_count} "
                "reached the run"
            )
        self.close()
        try:
            self._run.write_text(SUMMARY, encode_json(self._summary(outcome)))
            if runtime_s is not None:
                # Tempo de parede nunca entra em outputs/: um rerun não o reproduz. Aqui ele
                # só existe quando o chamador realmente mediu, o que mantém o artefato
                # reprodutível byte a byte.
                self._run.write_text(
                    RUNTIME,
                    encode_json(
                        {
                            "runtime_s": runtime_s,
                            "candidate_query_seconds": sum(
                                float(t["candidate_query_seconds"]) for t in self._timings
                            ),
                            "projection_seconds": sum(
                                float(t["projection_seconds"]) for t in self._timings
                            ),
                            "frames": self._timings,
                        }
                    ),
                )
            if self._debug_feature_sources:
                self._run.write_text(
                    f"{DEBUG_DIRECTORY}/feature-sources.json",
                    json.dumps(self._debug_feature_sources, indent=2, sort_keys=True) + "\n",
                    contractual=False,
                )
            self._run.publish(manifest=self._manifest_record(outcome), readme=None)
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error
        self._finalized = True
        return load_manifest(self._writer.final_dir)

    def close(self) -> None:
        """Close every open stream; idempotent, and required before publishing."""
        if not self._closed:
            self._closed = True
            self._streams.close()

    # Escrita por frame.

    def _write_frame_observations(self, frame: FrameAssociation) -> None:
        """Append observations, quality and indexed global geometry support."""
        import numpy as np

        projection = frame.resolution.frame
        for region, observation, quality in zip(
            frame.membership.regions, frame.observations, frame.qualities, strict=True
        ):
            # As posições são linhas de candidatos; o que se persiste é a identidade global
            # de cada uma, que é o que o leitor reconstrói.
            rows = np.asarray(region.associated_indices)
            positions = as_u32_indices(projection.global_indices[rows], table=SUPPORT)
            if projection.map_references(rows) != observation.geometry_support:
                raise RunArtifactError(
                    f"the geometry support of {observation.spatial_observation_id!r} "
                    "does not match its region membership"
                )
            record = encode_spatial_observation(observation)
            del record["geometry_support"]
            record["support"] = {
                "offset": self._support_offset,
                "count": int(positions.shape[0]),
            }
            observation_line = json.dumps(record, sort_keys=True).encode("utf-8")
            quality_line = json.dumps(encode_observation_quality(quality), sort_keys=True).encode(
                "utf-8"
            )
            self._index.write(
                _line(
                    {
                        "spatial_observation_id": str(observation.spatial_observation_id),
                        "source_observation_id": str(observation.source_observation_id),
                        "region_id": str(observation.region_id),
                        "observation": {
                            "byte_offset": self._observation_offset,
                            "byte_length": len(observation_line),
                        },
                        "quality": {
                            "byte_offset": self._quality_offset,
                            "byte_length": len(quality_line),
                        },
                    }
                )
            )
            self._support.write(positions.tobytes())
            self._observations.write(observation_line + b"\n")
            self._quality.write(quality_line + b"\n")
            self._support_offset += int(positions.shape[0])
            self._observation_offset += len(observation_line) + 1
            self._quality_offset += len(quality_line) + 1

    def _write_frame_dense(self, frame: FrameAssociation) -> None:
        """Append dense sampling indices and weights without copying embeddings."""
        import numpy as np

        if self._dense_index is None or self._dense_cells is None:
            self._dense_index = self._open(DENSE_INDEX)
            self._dense_cells = self._open(DENSE_CELLS)
        global_indices = frame.resolution.frame.global_indices
        for channel_id in sorted(frame.dense_samples):
            samples = frame.dense_samples[channel_id]
            sections = (
                as_u32_indices(
                    global_indices[samples.eligible_indices], table=DENSE_CELLS
                ).tobytes(),
                np.asarray(samples.sampled, dtype="u1").tobytes(),
                np.asarray(samples.cell_rows, dtype="<i4").tobytes(),
                np.asarray(samples.cell_cols, dtype="<i4").tobytes(),
                np.asarray(samples.weights, dtype="<f4").tobytes(),
            )
            self._dense_index.write(
                _line(
                    {
                        "source_observation_id": str(frame.source_observation_id),
                        "channel_id": channel_id,
                        "provenance": samples.provenance.to_record(),
                        "terms": int(samples.cell_rows.shape[1]),
                        "eligible_count": int(samples.eligible_indices.shape[0]),
                        "byte_offset": self._dense_offset,
                    }
                )
            )
            for section in sections:
                self._dense_cells.write(section)
                self._dense_offset += len(section)

    def _write_frame_debug(self, frame: FrameAssociation) -> None:
        """Write only the non-contractual evidence selected by the debug level."""
        level = self._writer.debug_level
        if level is SensorAssociationDebugLevel.NONE:
            return
        write_standard_debug(self._run, frame)
        if level is SensorAssociationDebugLevel.FULL:
            self._debug_feature_sources.extend(write_full_debug(self._run, frame))

    def _accumulate(self, frame: FrameAssociation) -> None:
        """Accumulate scalar run metrics and feature lineage from one completed frame."""
        self._frame_count += 1
        self._observation_count += len(frame.observations)
        self._observations_without_support += sum(
            1 for observation in frame.observations if not observation.geometry_support
        )
        self._failed_frame_count += int(frame.diagnostics.failed)
        count = frame.diagnostics.range_limit_candidate_support_count
        if count is not None:
            self._range_limit_candidate_support_count += count
        slack = frame.diagnostics.min_range_slack_m
        if slack is not None:
            self._min_range_slack_m = (
                slack if self._min_range_slack_m is None else min(self._min_range_slack_m, slack)
            )
        for finding in frame.diagnostics.findings:
            code = finding.code.value
            self._finding_counts[code] = self._finding_counts.get(code, 0) + 1
        resolution = frame.resolution
        for state, count in resolution.state_counts().items():
            self._state_counts[state.value] = self._state_counts.get(state.value, 0) + count
        self._state_counts["visible"] = (
            self._state_counts.get("visible", 0) + resolution.visible_count
        )
        projection = resolution.frame
        self._timings.append(
            {
                "source_observation_id": str(frame.source_observation_id),
                "candidate_query_seconds": projection.candidates.query_seconds,
                "projection_seconds": projection.projection_seconds,
                "candidate_count": projection.candidate_count,
            }
        )
        for channel_id, samples in frame.dense_samples.items():
            source = _dense_feature_source(samples.provenance)
            channel = self._dense_sources.setdefault(channel_id, {})
            key = json.dumps(source, sort_keys=True)
            entry = channel.setdefault(key, {**source, "frame_count": 0})
            entry["frame_count"] += 1

    # Registros agregados da execução.

    def _summary(self, outcome: SensorAssociationOutcome) -> dict[str, Any]:
        """Encode aggregate counts and explicit pose rejections for the run."""
        return {
            "frame_count": self._frame_count,
            "observation_count": self._observation_count,
            "observations_without_support": self._observations_without_support,
            "state_counts": dict(self._state_counts),
            # Sem limite de alcance não se aplica: nulo, nunca zero.
            "range_limit_candidate_support_count": None
            if outcome.candidate_policy.max_range_m is None
            else self._range_limit_candidate_support_count,
            "min_range_slack_m": self._min_range_slack_m,
            "rejected_frames": [
                {
                    "source_observation_id": str(rejected.source_observation_id),
                    "rejection": rejected.rejection.value,
                }
                for rejected in outcome.rejected
            ],
        }

    def _manifest_record(self, outcome: SensorAssociationOutcome) -> dict[str, Any]:
        """Encode the run identity, policies and canonical definition versions."""
        pose = outcome.pose_policy
        tolerances = outcome.tolerances
        return {
            "run_id": str(self._writer.run_id),
            "run_index": self._writer.run_index,
            "sequence_name": self._writer.sequence_name,
            "sequence_artifact_id": str(outcome.sequence_artifact_id),
            "selection_id": outcome.selection_id,
            "geometric_map_id": str(outcome.geometric_map.map_id),
            "trajectory_id": str(outcome.trajectory_id),
            "state_estimation_run_id": None
            if outcome.state_estimation_run_id is None
            else str(outcome.state_estimation_run_id),
            "region_artifact_ids": list(outcome.region_artifact_ids),
            "visual_artifact_ids": list(outcome.visual_artifact_ids),
            "calibration_identity": outcome.calibration_identity,
            "candidate_policy": {
                **outcome.candidate_policy.to_record(),
                "fingerprint": outcome.candidate_policy.fingerprint(),
            },
            "visibility_policy": {
                **outcome.occlusion_policy.to_record(),
                "fingerprint": outcome.occlusion_policy.fingerprint(),
            },
            "membership_policy_id": outcome.membership_policy_id,
            "membership_policy": None
            if outcome.box_policy is None
            else outcome.box_policy.to_record(),
            "definitions": dict(DEFINITION_VERSIONS),
            "pose_policy": {
                "mode": pose.mode.value,
                "max_time_delta_ns": pose.max_time_delta_ns,
                "max_interpolation_gap_ns": pose.max_interpolation_gap_ns,
            },
            "tolerances": {
                "max_pose_time_delta_ns": tolerances.max_pose_time_delta_ns,
                "max_map_window_offset_ns": tolerances.max_map_window_offset_ns,
                "max_reprojection_p95_px": tolerances.max_reprojection_p95_px,
                "max_reprojection_invalid_rate": tolerances.max_reprojection_invalid_rate,
            },
            "dense_channels": [
                {
                    "channel_id": channel.channel_id,
                    "interpolation": channel.interpolation.value,
                    "feature_sources": [
                        self._dense_sources[channel.channel_id][key]
                        for key in sorted(self._dense_sources.get(channel.channel_id, {}))
                    ],
                }
                for channel in outcome.dense_channels
            ],
            "configuration_fingerprint": outcome.configuration_fingerprint,
            "code_version": outcome.code_version,
            "frame_count": self._frame_count,
            "rejected_frame_count": len(outcome.rejected),
            "failed_frame_count": self._failed_frame_count,
            "observation_count": self._observation_count,
            "finding_counts": dict(self._finding_counts),
            "debug_level": self._writer.debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }


def _line(record: Mapping[str, Any]) -> bytes:
    """Encode one JSONL record, with its newline, exactly as a batch write would."""
    return json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"


def _dense_feature_source(provenance: Any) -> dict[str, Any]:
    """The feature source one dense channel sampled in a frame, as JSON primitives."""
    enhancement = provenance.enhancement
    return {
        "source_artifact_id": provenance.source_artifact_id,
        "embedding_space_id": provenance.embedding_space_id,
        "coordinate_transform_id": provenance.coordinate_transform_id,
        "sampling_fingerprint": provenance.sampling_fingerprint,
        "extractor": {
            "backend_id": provenance.extractor.backend_id,
            "provider": provenance.extractor.provider,
            "model": provenance.extractor.model,
            "version": provenance.extractor.version,
            "configuration_fingerprint": provenance.extractor.configuration_fingerprint,
        },
        "enhancement": None
        if enhancement is None
        else {
            "backend_id": enhancement.backend.backend_id,
            "model": enhancement.backend.model,
            "version": enhancement.backend.version,
            "configuration_fingerprint": enhancement.backend.configuration_fingerprint,
            "source_embedding_space_id": enhancement.source_embedding_space_id,
            "output_embedding_space_id": enhancement.output_embedding_space_id,
            "input_grid_size": list(enhancement.input_grid_size),
            "output_grid_size": list(enhancement.output_grid_size),
        },
    }


def _projection_record(frame: FrameAssociation) -> dict[str, Any]:
    """Encode projection stages, candidate counts and the transforms used."""
    projection = frame.resolution.frame
    transform = projection.image_transform
    return {
        "source_observation_id": str(projection.source_observation_id),
        "image_timestamp": projection.image_timestamp.to_record(),
        "map_id": str(projection.map_id),
        # Sem um `point_count` só: ele significaria o tamanho do mapa no schema 0.1.0 e a
        # população avaliada no 0.2.0, e um leitor não teria como saber qual. Os dois conceitos
        # ficam nomeados dentro de `candidates`.
        "candidates": projection.candidates.to_record(),
        "stage_counts": {stage.value: count for stage, count in projection.stage_counts().items()},
        "calibration_ref": encode_calibration_ref(projection.calibration_ref),
        "camera": {
            "calibration_id": str(projection.camera.calibration_id),
            "content_hash": projection.camera.content_hash,
            "camera_frame": str(projection.camera.camera_frame),
            "camera_model_kind": projection.camera.camera_model_kind,
            "image_size": list(projection.camera.image_size),
        },
        "pose_ref": encode_pose_ref(projection.pose_ref),
        "extrinsic": {
            "calibration_identity": projection.extrinsic.calibration_identity,
            "parent_frame": str(projection.extrinsic.parent_frame),
            "child_frame": str(projection.extrinsic.child_frame),
        },
        "image_transform": {
            "transform_id": transform.transform_id,
            "raw_size": list(transform.raw_size),
            "prepared_size": list(transform.prepared_size),
            "steps": [
                {
                    "operation": step.operation,
                    "input_size": list(step.input_size),
                    "output_size": list(step.output_size),
                    "offset_px": list(step.offset_px),
                    "scale": list(step.scale),
                }
                for step in transform.steps
            ],
        },
    }


def _visibility_record(frame: FrameAssociation) -> dict[str, Any]:
    """Encode the visibility policy, state counts and depth support of a frame."""
    resolution = frame.resolution
    membership = frame.membership
    statistics = membership.statistics()
    return {
        "source_observation_id": str(frame.source_observation_id),
        "policy_id": resolution.policy.policy_id,
        "policy_fingerprint": resolution.policy.fingerprint(),
        "depth_metric": resolution.depth_metric.value,
        "state_counts": {
            **{state.value: count for state, count in resolution.state_counts().items()},
        },
        "visible": resolution.visible_count,
        "neighborhood_only_occlusions": resolution.neighborhood_only_occlusions,
        "membership": {
            "definitions_version": statistics.definitions_version,
            "associated_count": statistics.associated_count,
            "visible_unassigned_count": statistics.visible_unassigned_count,
            "overlap_point_count": statistics.overlap_point_count,
            "membership_pair_count": statistics.membership_pair_count,
            "points_by_region_count": {
                str(k): v for k, v in statistics.points_by_region_count.items()
            },
        },
        "regions": [
            {
                "region_id": str(region.region_id),
                "footprint_kind": region.footprint_kind.value,
                "footprint_area_px": region.footprint_area_px,
                "associated_count": region.associated_count,
                "excluded_visible_count": region.excluded_visible_count,
                "occluded_count": region.occluded_count,
                "outside_valid_support_count": region.outside_valid_support_count,
                "covered_pixel_count": region.covered_pixel_count,
            }
            for region in membership.regions
        ],
        "skipped_regions": [
            {"region_id": str(s.region_id), "reason": s.reason.value} for s in membership.skipped
        ],
    }
