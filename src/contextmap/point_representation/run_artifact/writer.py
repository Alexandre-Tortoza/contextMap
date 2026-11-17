"""Construction and atomic publication of a Point Representation run artifact.

Outcomes are added one at a time, as the execution service streams them, and the run is
written atomically on finalization. The payload row, the JSONL byte offset of each
representation and the manifest counts are one invariant, maintained together in
``PointRepresentationRunWriter._add_represented``; that is why the writer is a single
module.

The writer records exactly in the ``output_dir`` the caller hands over. It computes no
path and allocates no identity: the run's identity and index come from its caller.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import GeometricMap, GeometryId, GeometryReference
from contextmap.point_representation.comparability import representation_space_fingerprint
from contextmap.point_representation.models import (
    EncoderIdentity,
    FailedSupport,
    FailureReason,
    PointRepresentation,
    PointRepresentationId,
    PointRepresentationRunId,
    RepresentationSpace,
)
from contextmap.point_representation.run_artifact.layout import (
    COUNTS,
    DTYPES,
    FAILED_SUPPORTS,
    GEOMETRY_INDEX,
    NORMS,
    REPRESENTATION_INDEX,
    REPRESENTATIONS,
    RUNTIME,
    SCHEMA_VERSION,
    SPACES,
    SUPPORT_SIZE,
    TRACE_LIMIT,
    PointRepresentationDebugLevel,
    RunArtifactError,
    encode_document,
    encode_lines,
)
from contextmap.point_representation.run_artifact.manifest import (
    PointRepresentationRunManifest,
    center_selection_id,
    load_manifest,
)
from contextmap.point_representation.run_artifact.metrics import (
    ComponentStatistics,
    distribution,
    failure_reason_counts,
)
from contextmap.point_representation.serialization import (
    encode_failed_support,
    encode_point_representation,
    encode_representation_space,
    encode_support_policy,
)
from contextmap.point_representation.service import EncodedRepresentation, RepresentationMetrics
from contextmap.shared import DEBUG_DIRECTORY, AtomicRunDirectory, RunDirectoryError


class PointRepresentationRunWriter:
    """Builds an immutable Point Representation run artifact on the local filesystem.

    Outcomes are added one at a time, as the execution service streams them, and
    the run is written atomically by :meth:`finalize`. The writer holds the
    encoded records in memory until then, so a run's size is bounded by memory:
    ``shared.AtomicRunDirectory`` writes whole files.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: PointRepresentationRunId,
        run_index: int,
        geometric_map: GeometricMap,
        space: RepresentationSpace,
        encoder_identity: EncoderIdentity,
        code_version: str,
        association_context_id: str | None = None,
        center_selection_policy_id: str | None = None,
        debug_level: PointRepresentationDebugLevel = PointRepresentationDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/point_representation``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run among the caller's runs of the same sequence,
                supplied by the caller and recorded as given.
            geometric_map: The map the geometry came from; only its identity
                and lineage are recorded, never its points.
            space: The run's representation space.
            encoder_identity: The encoder that produced the run.
            code_version: Code revision that produced the run.
            association_context_id: Identity of an association run explicitly
                used as selection/context input, if any.
            center_selection_policy_id: Versioned policy that selected the centers, or
                ``None`` when the caller chose them itself.
            debug_level: Amount of non-contractual debug evidence to persist.

        Raises:
            RunArtifactError: If ``association_context_id`` or
                ``center_selection_policy_id`` is given but empty.
        """
        if association_context_id is not None and not association_context_id:
            raise RunArtifactError("association_context_id must be None or a non-empty identity")
        if center_selection_policy_id is not None and not center_selection_policy_id:
            raise RunArtifactError(
                "center_selection_policy_id must be None or a non-empty policy identity"
            )
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._map = geometric_map
        self._space = space
        self._space_id = representation_space_fingerprint(space)
        self._encoder = encoder_identity
        self._code_version = code_version
        self._association_context_id = association_context_id
        self._center_selection_policy_id = center_selection_policy_id
        self._debug_level = debug_level
        self._final_dir = output_dir
        code, self._item_size, suffix = DTYPES[space.dtype]
        self._row = struct.Struct(f"<{space.dimension}{code}")
        self._payload_path = f"outputs/payloads/vectors.{suffix}"
        self._payload = bytearray()
        self._records: list[bytes] = []
        self._index_lines: list[str] = []
        self._geometry_lines: list[str] = []
        self._failed_lines: list[str] = []
        self._offset = 0
        self._centers: set[GeometryId] = set()
        self._representation_ids: set[PointRepresentationId] = set()
        self._partial = 0
        self._failed_by_reason: dict[FailureReason, int] = {}
        self._support_sizes: list[int] = []
        self._norms: list[tuple[PointRepresentationId, float]] = []
        self._traces: list[dict[str, Any]] = []
        self._component_stats = ComponentStatistics(space.dimension)
        self._finalized = False

    def add(self, outcome: EncodedRepresentation | FailedSupport) -> None:
        """Add one outcome of the execution service.

        Args:
            outcome: A representation with its vector, or a failed support.

        Raises:
            RunArtifactError: If the writer is finalized, the outcome belongs to
                another map, space or encoder, its center already has an outcome,
                or its vector breaks the space (dimension, non-finite values, a
                value that does not fit the payload dtype).
        """
        self._require_open()
        if isinstance(outcome, FailedSupport):
            self._add_failed(outcome)
        else:
            self._add_represented(outcome)

    def finalize(
        self,
        *,
        metrics: RepresentationMetrics,
        backend_diagnostics: Mapping[str, str | int | float | None] | None = None,
    ) -> PointRepresentationRunManifest:
        """Persist a completed run atomically.

        Args:
            metrics: The execution service's metrics for the run; they must
                match the outcomes added, so a stream that stopped early is
                never persisted as a complete run.
            backend_diagnostics: Backend-specific runtime figures, for example
                peak device memory, recorded apart from every quality measure.

        Returns:
            The manifest of the finalized run.

        Raises:
            RunArtifactError: If already finalized, if the outcomes do not match
                the metrics, or if a run already exists at the target path.
        """
        self._require_open()
        self._require_complete(metrics)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                self._write_outputs(run)
                self._write_metrics(run, metrics, backend_diagnostics)
                self._write_debug(run)
                run.publish(manifest=self._manifest_record(), readme=self._render_readme())
        except RunDirectoryError as error:
            raise RunArtifactError(str(error)) from error
        self._finalized = True
        return load_manifest(self._final_dir)

    def _require_open(self) -> None:
        if self._finalized:
            raise RunArtifactError("writer already finalized")

    def _require_complete(self, metrics: RepresentationMetrics) -> None:
        failed = sum(self._failed_by_reason.values())
        represented = len(self._representation_ids)
        observed = (represented + failed, represented, self._partial, self._failed_by_reason)
        reported = (
            metrics.requested,
            metrics.represented,
            metrics.partial,
            dict(metrics.failed_by_reason),
        )
        if observed != reported:
            raise RunArtifactError(
                "incomplete run: the outcomes added (requested/represented/partial/failed by "
                f"reason: {observed}) do not match the run metrics ({reported})"
            )

    def _require_unclaimed_center(self, center: GeometryReference) -> None:
        if center.map_id != self._map.map_id:
            raise RunArtifactError(
                f"center {center.geometry_id!r} belongs to map {center.map_id!r}, but this run "
                f"is over map {self._map.map_id!r}"
            )
        if center.geometry_id in self._centers:
            raise RunArtifactError(f"center {center.geometry_id!r} already has an outcome")

    def _add_failed(self, failed: FailedSupport) -> None:
        self._require_unclaimed_center(failed.support.center)
        self._centers.add(failed.support.center.geometry_id)
        self._failed_lines.append(json.dumps(encode_failed_support(failed), sort_keys=True))
        self._failed_by_reason[failed.reason] = self._failed_by_reason.get(failed.reason, 0) + 1
        self._support_sizes.append(failed.support.statistics.count)
        if self._debug_level is not PointRepresentationDebugLevel.NONE:
            self._traces.append(
                {
                    "outcome": "failed",
                    "reason": failed.reason.value,
                    "center": str(failed.support.center.geometry_id),
                    "member_count": failed.support.statistics.count,
                }
            )

    def _add_represented(self, outcome: EncodedRepresentation) -> None:
        representation = outcome.representation
        self._require_matches_run(representation)
        values = self._require_storable_vector(outcome.values)
        self._require_unclaimed_center(representation.geometry_reference)
        if representation.representation_id in self._representation_ids:
            raise RunArtifactError(
                f"representation {representation.representation_id!r} already has an outcome"
            )
        try:
            packed = self._row.pack(*values)
        except (struct.error, OverflowError) as error:
            raise RunArtifactError(
                f"a vector value does not fit {self._space.dtype} and would be stored as "
                f"infinity: {error}"
            ) from error
        # Tudo foi validado: só agora o estado do writer muda, então um add recusado é repetível.
        self._centers.add(representation.geometry_reference.geometry_id)
        row = len(self._representation_ids)
        stored = replace(representation, payload_reference=f"{self._payload_path}#{row}")
        line = json.dumps(encode_point_representation(stored), sort_keys=True).encode("utf-8")
        self._payload += packed
        self._records.append(line)
        self._index_lines.append(
            json.dumps(
                {
                    "representation_id": str(stored.representation_id),
                    "byte_offset": self._offset,
                    "byte_length": len(line),
                },
                sort_keys=True,
            )
        )
        self._offset += len(line) + 1
        self._geometry_lines.append(
            json.dumps(
                {
                    "map_id": str(stored.geometry_reference.map_id),
                    "geometry_id": str(stored.geometry_reference.geometry_id),
                    "representation_id": str(stored.representation_id),
                },
                sort_keys=True,
            )
        )
        self._representation_ids.add(stored.representation_id)
        self._observe(stored, values)

    def _observe(self, representation: PointRepresentation, values: tuple[float, ...]) -> None:
        undefined = set(representation.undefined_components)
        defined = [value for index, value in enumerate(values) if index not in undefined]
        self._partial += representation.is_partial
        self._support_sizes.append(representation.support.statistics.count)
        if defined:
            self._norms.append((representation.representation_id, math.hypot(*defined)))
        if self._debug_level is PointRepresentationDebugLevel.FULL:
            self._component_stats.observe(values, undefined)
        if self._debug_level is not PointRepresentationDebugLevel.NONE:
            support_stats = representation.support.statistics
            self._traces.append(
                {
                    "outcome": "represented",
                    "representation_id": str(representation.representation_id),
                    "center": str(representation.geometry_reference.geometry_id),
                    "member_count": support_stats.count,
                    "max_distance_m": support_stats.max_distance_m,
                    "near_map_bounds": support_stats.near_map_bounds,
                    "undefined_components": list(representation.undefined_components),
                }
            )

    def _require_matches_run(self, representation: PointRepresentation) -> None:
        if representation.representation_space_id != self._space_id:
            raise RunArtifactError(
                "the representation space of this representation is not the run's: "
                f"{representation.representation_space_id} != {self._space_id}"
            )
        if representation.encoder_identity != self._encoder:
            raise RunArtifactError("this representation was produced by another encoder")
        if (
            representation.shape != (self._space.dimension,)
            or representation.dtype != self._space.dtype
            or representation.normalization != self._space.normalization
        ):
            raise RunArtifactError(
                "this representation's shape, dtype or normalization does not match its space"
            )
        if representation.payload_reference is not None:
            raise RunArtifactError(
                "the payload_reference is assigned by the writer and must not be set beforehand"
            )

    def _require_storable_vector(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) != self._space.dimension:
            raise RunArtifactError(
                f"the vector has {len(values)} values but the space is "
                f"{self._space.dimension}-dimensional"
            )
        if not all(math.isfinite(value) for value in values):
            raise RunArtifactError("the vector holds a non-finite value, which is never stored")
        return values

    def _write_outputs(self, run: AtomicRunDirectory) -> None:
        run.write_bytes(
            REPRESENTATIONS, b"\n".join(self._records) + (b"\n" if self._records else b"")
        )
        run.write_text(REPRESENTATION_INDEX, encode_lines(self._index_lines))
        run.write_text(
            SPACES,
            encode_document(
                {
                    "spaces": [
                        {
                            "representation_space_id": self._space_id,
                            **encode_representation_space(self._space),
                        }
                    ]
                }
            ),
        )
        run.write_text(GEOMETRY_INDEX, encode_lines(self._geometry_lines))
        run.write_text(FAILED_SUPPORTS, encode_lines(self._failed_lines))
        run.write_bytes(self._payload_path, bytes(self._payload))

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        metrics: RepresentationMetrics,
        backend_diagnostics: Mapping[str, str | int | float | None] | None,
    ) -> None:
        run.write_text(
            COUNTS,
            encode_document(
                {
                    "requested": metrics.requested,
                    "represented": metrics.represented,
                    "partial": metrics.partial,
                    "failed": metrics.failed,
                    "failed_by_reason": failure_reason_counts(self._failed_by_reason),
                }
            ),
        )
        run.write_text(SUPPORT_SIZE, encode_document(distribution(self._support_sizes)))
        run.write_text(
            NORMS,
            encode_document(
                {**distribution([norm for _, norm in self._norms]), "non_finite_values": 0}
            ),
        )
        run.write_text(
            RUNTIME,
            encode_document(
                {
                    "support_extraction_seconds": metrics.support_extraction_seconds,
                    "encoding_seconds": metrics.encoding_seconds,
                    "backend": dict(backend_diagnostics or {}),
                }
            ),
        )

    def _write_debug(self, run: AtomicRunDirectory) -> None:
        if self._debug_level is PointRepresentationDebugLevel.NONE:
            return
        step = max(1, len(self._traces) // TRACE_LIMIT)
        run.write_text(
            f"{DEBUG_DIRECTORY}/support-traces.jsonl",
            encode_lines(
                json.dumps(trace, sort_keys=True) for trace in self._traces[::step][:TRACE_LIMIT]
            ),
            contractual=False,
        )
        run.write_text(
            f"{DEBUG_DIRECTORY}/representation-norms.csv",
            "".join(
                ["representation_id,norm\n"]
                + [f"{identity},{norm!r}\n" for identity, norm in self._norms]
            ),
            contractual=False,
        )
        if self._debug_level is PointRepresentationDebugLevel.FULL:
            run.write_text(
                f"{DEBUG_DIRECTORY}/component-statistics.json",
                encode_document(
                    {"components": self._component_stats.summary(self._space.feature_names)}
                ),
                contractual=False,
            )

    def _manifest_record(self) -> dict[str, Any]:
        provenance = self._map.provenance
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "geometric_map": {
                "map_id": str(self._map.map_id),
                "frame_id": str(self._map.frame_id),
                "point_count": self._map.point_count,
                "sequence_artifact_id": str(provenance.sequence_artifact_id),
                "selection_id": provenance.selection_id,
                "trajectory_id": str(provenance.trajectory_id),
            },
            "association_context_id": self._association_context_id,
            "center_selection_id": center_selection_id(
                map_id=self._map.map_id,
                centers=(
                    GeometryReference(map_id=self._map.map_id, geometry_id=geometry_id)
                    for geometry_id in self._centers
                ),
            ),
            "center_selection_policy_id": self._center_selection_policy_id,
            "support_policy": encode_support_policy(self._space.support_semantics),
            "representation_space_id": self._space_id,
            "dimension": self._space.dimension,
            "dtype": self._space.dtype,
            "payload_path": self._payload_path,
            "encoder": {
                "backend_id": self._encoder.backend_id,
                "backend_version": self._encoder.backend_version,
                "configuration_fingerprint": self._encoder.configuration_fingerprint,
                "checkpoint_hash": self._encoder.checkpoint_hash,
            },
            "code_version": self._code_version,
            "representation_count": len(self._representation_ids),
            "partial_count": self._partial,
            "failed_count": sum(self._failed_by_reason.values()),
            "failed_by_reason": failure_reason_counts(self._failed_by_reason),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self) -> str:
        return (
            f"# Point representation run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Geometric map: `{self._map.map_id}`\n"
            f"- Encoder: `{self._encoder.backend_id}` (version `{self._encoder.backend_version}`)\n"
            f"- Representation space: `{self._space_id}`\n"
            f"- Representations: {len(self._representation_ids)} "
            f"({self._partial} partial), failed supports: {sum(self._failed_by_reason.values())}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `__DEBUG__/` is human "
            "evidence and no "
            "downstream stage may depend on it. A representation is 3D evidence, not semantic "
            "truth.\n"
        )
