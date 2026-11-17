"""Persistência streaming, atômica e imutável do resultado de ingestion.

O writer grava payloads e índice em diretório temporário, anexa calibração, provenance e
diagnostics quando fornecidos, verifica consistência e só então publica o artifact final. Uma
falha nunca deve deixar um diretório parcial parecendo um SequenceArtifact válido.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from contextmap.ingestion.calibration import (
    CalibrationSet,
    encode_calibration_set,
    ensure_valid_calibration_set,
)
from contextmap.ingestion.canonical_observations import MODALITY_NAMES, SourceObservation
from contextmap.ingestion.observation_synchronization import SynchronizationDiagnostics
from contextmap.ingestion.sequence_artifact.sequence_artifact_integrity import (
    _check_index_cross_references,
)
from contextmap.ingestion.sequence_artifact.sequence_artifact_manifest import (
    _CALIBRATION_FILENAME,
    _DIAGNOSTICS_DROPPED_EVENTS_FILENAME,
    _DIAGNOSTICS_FRAME_GRAPH_FILENAME,
    _DIAGNOSTICS_SUMMARY_FILENAME,
    _DIAGNOSTICS_SYNCHRONIZATION_FILENAME,
    _DIAGNOSTICS_WARNINGS_FILENAME,
    _INDEX_FILENAME,
    _PROVENANCE_FILENAME,
    SCHEMA_VERSION,
    SequenceArtifactError,
    SequenceArtifactFileEntry,
    SequenceArtifactId,
    SequenceArtifactManifest,
    _manifest_fields,
)
from contextmap.ingestion.sequence_artifact.source_observation_serialization import (
    _encode_observation,
    _without_payload,
)
from contextmap.ingestion.sequence_observation_diagnostics import (
    SequenceDiagnostics,
    build_frame_graph_diagnostics,
    encode_diagnostics,
    encode_dropped_event,
    encode_frame_graph,
    encode_synchronization_decision,
    summarize_observations,
)
from contextmap.ingestion.sequence_provenance import SequenceProvenance, encode_provenance
from contextmap.shared import AtomicRunDirectory, RunDirectoryError, is_run_relative_path

_MODALITY_COUNTS_TEMPLATE: dict[str, int] = dict.fromkeys(MODALITY_NAMES, 0)


class SequenceArtifactWriter:
    """Builds an immutable canonical sequence artifact on the local filesystem.

    Payloads and index lines are streamed to a temporary directory by
    :meth:`add_observation`; only observation metadata (payload bytes
    excluded) is kept in memory, for the diagnostics summary written by
    :meth:`finalize`. A writer that is not finalized leaves its temporary
    directory behind until :meth:`abort` is called, so prefer the context
    manager form, which aborts on any exit path that did not finalize.

    Example:
        with SequenceArtifactWriter(
            output_dir=Path("workspace/corridor-02/run-0001/ingestion"),
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            writer.add_observation(image_observation)
            writer.add_observation(lidar_observation)
            manifest = writer.finalize()
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        artifact_id: SequenceArtifactId,
    ) -> None:
        """Create a writer for a new sequence artifact.

        No filesystem access happens until the first observation is added
        or :meth:`finalize` is called.

        Args:
            output_dir: The final directory of the artifact. The caller chooses
                it (in the runtime, ``<workspace>/<dataset>/<run>/ingestion``);
                the writer computes no path, builds the artifact in a temporary
                sibling of ``output_dir`` and refuses to replace a directory
                that already exists.
            sequence_name: Name of the sequence this artifact belongs to.
            artifact_id: Identity of the artifact, supplied by the caller and
                recorded as given; the writer never allocates one.
        """
        self._sequence_name = sequence_name
        self._artifact_id = artifact_id
        self._final_dir = output_dir
        # O run e o fluxo do índice atravessam chamadas: cada um vive no seu escopo, aberto no
        # primeiro uso e fechado por finalize() ou abort().
        self._run_scope = contextlib.ExitStack()
        self._run: AtomicRunDirectory | None = None
        self._index_scope: contextlib.ExitStack | None = None
        self._index: BinaryIO | None = None
        self._counts = dict(_MODALITY_COUNTS_TEMPLATE)
        self._metadata: list[SourceObservation] = []
        self._seen_observation_ids: set[str] = set()
        self._calibration: CalibrationSet | None = None
        self._provenance: SequenceProvenance | None = None
        self._diagnostic_warnings: tuple[str, ...] | None = None
        self._synchronization_diagnostics: SynchronizationDiagnostics | None = None
        self._closed_by: str | None = None

    def __enter__(self) -> SequenceArtifactWriter:
        """Return the writer for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Abort the writer unless it was already finalized.

        When the block is already failing, a cleanup error is attached to that
        exception as a note instead of replacing it.
        """
        if exc_value is None:
            self.abort()
        else:
            self._abort_after_failure(exc_value)

    def set_diagnostics(
        self,
        *,
        warnings: Sequence[str] = (),
        synchronization: SynchronizationDiagnostics | None = None,
    ) -> None:
        """Enable writing diagnostics for this sequence.

        The summary itself (per-modality counts/time-range, image
        resolutions, point-cloud field layouts) is computed automatically
        from the observations added to this writer; only warning text is
        supplied by the caller.

        Args:
            warnings: Adapter/validation/synchronization warnings to
                record alongside the summary, as human-readable text.
            synchronization: Structured synchronization decisions and dropped
                events to persist alongside the summary.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
        """
        self._require_open("set diagnostics")
        self._diagnostic_warnings = tuple(warnings)
        self._synchronization_diagnostics = synchronization

    def set_provenance(self, provenance: SequenceProvenance) -> None:
        """Attach the sequence's provenance/content-identity metadata.

        Args:
            provenance: Provenance to persist alongside the artifact.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
        """
        self._require_open("set provenance")
        self._provenance = provenance

    def set_calibration(self, calibration_set: CalibrationSet) -> None:
        """Attach the sequence's calibration set, written by :meth:`finalize`.

        Args:
            calibration_set: Calibration and coordinate-frame inventory for
                this sequence.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or :meth:`abort`.
            CalibrationError: If ``calibration_set`` is invalid; see
                :func:`~contextmap.ingestion.calibration.validate_calibration_set`.
        """
        self._require_open("set calibration")
        ensure_valid_calibration_set(calibration_set)
        self._calibration = calibration_set

    def add_observation(self, observation: SourceObservation) -> None:
        """Write an observation's payload and index line to the temporary artifact.

        The payload is persisted immediately and not retained, so the
        caller may release ``observation`` as soon as this returns. An I/O
        failure while writing aborts the writer, since the partial
        temporary artifact can no longer be trusted.

        Args:
            observation: Any canonical source observation.

        Raises:
            SequenceArtifactError: If called after :meth:`finalize` or
                :meth:`abort`, if ``observation.observation_id`` was
                already added, or if it would place the payload outside the
                artifact (for example an id with ``../``); nothing is written.
        """
        self._require_open("add observations")
        observation_id = str(observation.observation_id)
        if observation_id in self._seen_observation_ids:
            raise SequenceArtifactError(f"duplicate observation_id: {observation_id!r}")
        record, payload = _encode_observation(observation)
        if payload is not None and not is_run_relative_path(payload[0]):
            raise SequenceArtifactError(
                f"observation_id {observation_id!r} would place its payload at {payload[0]!r}, "
                "outside the artifact"
            )
        modality = record["modality"]
        assert isinstance(modality, str)

        try:
            run, index = self._open_temporary_artifact()
            if payload is not None:
                run.write_bytes(*payload)
            index.write(f"{json.dumps(record, sort_keys=True)}\n".encode())
            # NOTE: A linha chega ao arquivo na hora: uma falha de escrita aparece na observação,
            # não num flush posterior do buffer do fluxo.
            index.flush()
        except BaseException as error:
            self._abort_after_failure(error)
            raise

        self._counts[modality] += 1
        self._seen_observation_ids.add(observation_id)
        self._metadata.append(_without_payload(observation))

    def abort(self) -> None:
        """Discard the temporary artifact and close the writer.

        The writer is closed immediately, even if cleanup then fails. Errors
        from flushing the index are ignored, since that data is being
        discarded. A failure to remove the temporary directory is raised so
        it is never silent; calling :meth:`abort` again retries the removal.
        It has no effect on a finalized artifact and is a no-op once the
        temporary directory is gone.

        Raises:
            OSError: If the temporary directory could not be removed.
        """
        if self._closed_by == "finalize()":
            return
        self._closed_by = "abort()"
        index_scope, self._index_scope, self._index = self._index_scope, None, None
        if index_scope is not None:
            with contextlib.suppress(OSError):
                index_scope.close()
        if self._run is not None:
            self._run.discard()
            self._run_scope.close()

    def finalize(self) -> SequenceArtifactManifest:
        """Complete the temporary artifact and publish it atomically.

        The artifact is built under a temporary sibling directory and only
        moved to its final path after an internal integrity check succeeds,
        so a process interrupted mid-write never leaves a directory that
        looks like a complete artifact at the final path. The publication is
        :meth:`~contextmap.shared.AtomicRunDirectory.publish`: every file and
        directory reaches stable storage before the rename, and the rename
        never replaces anything at ``output_dir``, even a directory created
        there while this call runs. Any failure aborts the writer and removes
        the temporary directory.

        Returns:
            The manifest of the finalized artifact.

        Raises:
            SequenceArtifactError: If already finalized or aborted, if
                anything exists at the target path or appears there before the
                artifact is published, if the platform cannot publish without
                replacing, or if writing fails.
            OSError: If a file or directory cannot be forced to stable storage.
        """
        if self._closed_by == "finalize()":
            raise SequenceArtifactError("writer already finalized")
        if self._closed_by is not None:
            raise SequenceArtifactError(f"writer closed by {self._closed_by}")

        try:
            if self._final_dir.exists():
                raise SequenceArtifactError(f"sequence artifact already exists: {self._final_dir}")
            run, _ = self._open_temporary_artifact()
            manifest = self._write_contents(run)
            # O inventário é conferido pela publicação; o cruzamento índice/payload é do artifact.
            problems = _check_index_cross_references(
                run.written_path(_INDEX_FILENAME).parent, manifest
            )
            if problems:
                raise SequenceArtifactError(
                    f"internal consistency check failed before finalize: {problems}"
                )
            try:
                run.publish(manifest=_manifest_fields(manifest), readme=None)
            except RunDirectoryError as error:
                raise SequenceArtifactError(f"cannot publish sequence artifact: {error}") from error
            self._run_scope.close()
        except BaseException as error:
            self._abort_after_failure(error)
            raise

        self._closed_by = "finalize()"
        return manifest

    def _abort_after_failure(self, error: BaseException) -> None:
        """Abort without letting a cleanup failure replace ``error``, the real cause."""
        try:
            self.abort()
        except OSError as cleanup_error:
            error.add_note(
                f"could not remove the temporary artifact of {self._final_dir}: {cleanup_error}; "
                "call abort() to retry"
            )

    def _require_open(self, action: str) -> None:
        """Refuse mutations after the writer has been finalized or aborted.

        Args:
            action: Human-readable action used in the resulting error message.

        Raises:
            SequenceArtifactError: If the writer lifecycle is already closed.
        """
        if self._closed_by is not None:
            raise SequenceArtifactError(f"cannot {action} after {self._closed_by}")

    def _open_temporary_artifact(self) -> tuple[AtomicRunDirectory, BinaryIO]:
        """Create the temporary artifact and its index stream on first use and return them.

        The existing target is refused by :meth:`finalize`, not here, so an
        artifact already at ``output_dir`` is detected once everything is written.
        """
        if self._run is None:
            self._run = self._run_scope.enter_context(
                AtomicRunDirectory(self._final_dir, refuse_existing_early=False)
            )
            self._index_scope = contextlib.ExitStack()
            self._index = self._index_scope.enter_context(self._run.open_binary(_INDEX_FILENAME))
        assert self._index is not None
        return self._run, self._index

    def _write_contents(self, run: AtomicRunDirectory) -> SequenceArtifactManifest:
        """Close the index, write the sequence-level files and return the manifest to publish."""
        index_scope, self._index_scope, self._index = self._index_scope, None, None
        assert index_scope is not None
        index_scope.close()

        if self._calibration is not None:
            run.write_text(
                _CALIBRATION_FILENAME,
                json.dumps(encode_calibration_set(self._calibration), indent=2, sort_keys=True),
            )

        if self._provenance is not None:
            run.write_text(
                _PROVENANCE_FILENAME,
                json.dumps(encode_provenance(self._provenance), indent=2, sort_keys=True),
            )

        if self._diagnostic_warnings is not None:
            summary = summarize_observations(
                self._metadata,
                warning_count=len(self._diagnostic_warnings),
                synchronization=self._synchronization_diagnostics,
                calibration=self._calibration,
            )
            run.write_text(
                _DIAGNOSTICS_SUMMARY_FILENAME,
                json.dumps(
                    encode_diagnostics(SequenceDiagnostics(summary=summary, warnings=())),
                    indent=2,
                    sort_keys=True,
                ),
            )
            run.write_text(
                _DIAGNOSTICS_WARNINGS_FILENAME,
                "".join(
                    f"{json.dumps({'warning': warning})}\n" for warning in self._diagnostic_warnings
                ),
            )

            if self._synchronization_diagnostics is not None:
                run.write_text(
                    _DIAGNOSTICS_SYNCHRONIZATION_FILENAME,
                    "".join(
                        f"{json.dumps(encode_synchronization_decision(decision), sort_keys=True)}\n"
                        for decision in self._synchronization_diagnostics.decisions
                    ),
                )
                run.write_text(
                    _DIAGNOSTICS_DROPPED_EVENTS_FILENAME,
                    "".join(
                        f"{json.dumps(encode_dropped_event(event), sort_keys=True)}\n"
                        for event in self._synchronization_diagnostics.dropped_events
                    ),
                )

            if self._calibration is not None:
                run.write_text(
                    _DIAGNOSTICS_FRAME_GRAPH_FILENAME,
                    json.dumps(
                        encode_frame_graph(build_frame_graph_diagnostics(self._calibration)),
                        indent=2,
                        sort_keys=True,
                    ),
                )

        return SequenceArtifactManifest(
            artifact_id=self._artifact_id,
            sequence_name=self._sequence_name,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            observation_counts=dict(self._counts),
            file_inventory=tuple(
                SequenceArtifactFileEntry(
                    path=entry.path, size_bytes=entry.size_bytes, content_hash=entry.content_hash
                )
                for entry in run.inventory()
            ),
        )
