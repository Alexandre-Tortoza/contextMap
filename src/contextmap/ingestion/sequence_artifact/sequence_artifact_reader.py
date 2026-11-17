"""Leitura seletiva de SequenceArtifact finalizado sem depender da fonte original.

O reader abre manifest e índice, oferece integridade, calibração, provenance, diagnostics e replay
por offset ou identidade. Payloads de imagem/LiDAR só são carregados quando a operação realmente
precisa deles, preservando custo proporcional ao subconjunto consultado.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from contextmap.ingestion.calibration import CalibrationSet, decode_calibration_set
from contextmap.ingestion.canonical_observations import SourceObservation, SourceObservationId
from contextmap.ingestion.observation_synchronization import (
    DroppedEvent,
    SynchronizationDiagnostics,
)
from contextmap.ingestion.sequence_artifact.sequence_artifact_integrity import (
    _check_file_inventory,
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
    SequenceArtifactError,
    SequenceArtifactManifest,
    _load_manifest,
)
from contextmap.ingestion.sequence_artifact.source_observation_serialization import (
    _decode_observation,
)
from contextmap.ingestion.sequence_observation_diagnostics import (
    SequenceDiagnostics,
    decode_diagnostics_summary,
    decode_dropped_event,
    decode_frame_graph,
    decode_synchronization_decision,
)
from contextmap.ingestion.sequence_provenance import SequenceProvenance, decode_provenance
from contextmap.shared import SourceTimestamp


class IndexEntry(NamedTuple):
    """One observation's identity, timestamp, and position, without its payload.

    Returned by :meth:`SequenceArtifactReader.iter_index`, which streams
    ``index.jsonl`` without ever reading a payload file.

    Attributes:
        offset: Byte offset of this entry's line within ``index.jsonl``.
            An implementation detail, kept only so a caller can later
            decode this one observation directly through
            :meth:`SequenceArtifactReader.observation_at` — never
            persisted or meaningful across a different artifact.
        observation_id: Identity of the observation this entry describes.
        timestamp: Capture time in the source's own clock domain.
        modality: The observation's modality (``"image"``, ``"lidar"``,
            ``"imu"``, ``"external_pose"``), exactly as recorded in the index. It is
            here so a consumer that only wants one modality can select it
            while paying the index cost, never the payload-decoding cost of
            :meth:`SequenceArtifactReader.list_observations`.
    """

    offset: int
    observation_id: SourceObservationId
    timestamp: SourceTimestamp
    modality: str


class SequenceArtifactReader:
    """Reads a finalized canonical sequence artifact from the local filesystem."""

    def __init__(self, artifact_dir: Path) -> None:
        """Open a sequence artifact for reading.

        Args:
            artifact_dir: Path to the artifact's directory.

        Raises:
            IncompleteSequenceArtifactError: If ``manifest.json`` is missing.
            SequenceArtifactError: If the manifest's schema version is not
                understood by this module.
        """
        self._root = artifact_dir
        self._manifest = _load_manifest(artifact_dir)
        self._offsets_by_id_cache: dict[SourceObservationId, int] | None = None

    @property
    def manifest(self) -> SequenceArtifactManifest:
        """The artifact's manifest."""
        return self._manifest

    def verify_integrity(self) -> list[str]:
        """Check the artifact's file inventory and internal cross-references.

        Detects: missing files referenced by the manifest, size/content
        hash mismatches, and index.jsonl records whose ``payload_path``
        does not resolve to a file in the manifest's file inventory
        (an invalid index-to-payload cross-reference). An incompatible
        schema version is instead raised by :meth:`__init__`/:func:`_load_manifest`,
        since a reader that does not understand the schema cannot safely
        interpret anything else about the artifact.

        Returns:
            A list of human-readable problems; empty means no problem found.
        """
        return _check_file_inventory(self._root, self._manifest) + _check_index_cross_references(
            self._root, self._manifest
        )

    def read_provenance(self) -> SequenceProvenance | None:
        """Read this sequence's provenance/content-identity metadata, when written.

        Returns:
            The provenance, or ``None`` if the artifact was finalized
            without one (e.g. by a writer predating this contract).
        """
        provenance_path = self._root / _PROVENANCE_FILENAME
        if not provenance_path.is_file():
            return None
        return decode_provenance(json.loads(provenance_path.read_text(encoding="utf-8")))

    def read_diagnostics(self) -> SequenceDiagnostics | None:
        """Read this sequence's diagnostics (summary + warnings), when written.

        Returns:
            The diagnostics, or ``None`` if the artifact was finalized
            without calling
            :meth:`SequenceArtifactWriter.set_diagnostics`.
        """
        summary_path = self._root / _DIAGNOSTICS_SUMMARY_FILENAME
        if not summary_path.is_file():
            return None
        summary = decode_diagnostics_summary(json.loads(summary_path.read_text(encoding="utf-8")))

        warnings_path = self._root / _DIAGNOSTICS_WARNINGS_FILENAME
        warnings: list[str] = []
        if warnings_path.is_file():
            for line in warnings_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    warnings.append(json.loads(stripped)["warning"])

        synchronization_path = self._root / _DIAGNOSTICS_SYNCHRONIZATION_FILENAME
        decisions = (
            tuple(
                decode_synchronization_decision(json.loads(line))
                for line in synchronization_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if synchronization_path.is_file()
            else ()
        )

        dropped_path = self._root / _DIAGNOSTICS_DROPPED_EVENTS_FILENAME
        dropped_events: tuple[DroppedEvent, ...] = ()
        if dropped_path.is_file():
            dropped_records = [
                json.loads(line)
                for line in dropped_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if dropped_records:
                needed_ids = {
                    SourceObservationId(record["observation_id"]) for record in dropped_records
                }
                offsets_by_id = self._offsets_by_id()
                # NOTE: Diagnostics de dropped_events precisam só de identidade/modalidade
                # (ver decode_dropped_event); carregar o payload binário aqui
                # violaria a garantia de custo de ler somente metadados.
                observations_by_id = {
                    str(observation_id): self.observation_at(
                        offsets_by_id[observation_id], load_payload=False
                    )
                    for observation_id in needed_ids
                    if observation_id in offsets_by_id
                }
                dropped_events = tuple(
                    decode_dropped_event(record, observations_by_id) for record in dropped_records
                )
        synchronization = (
            SynchronizationDiagnostics(
                dropped_events=dropped_events,
                decisions=decisions,
            )
            if synchronization_path.is_file() or dropped_path.is_file()
            else None
        )

        frame_graph_path = self._root / _DIAGNOSTICS_FRAME_GRAPH_FILENAME
        frame_graph = (
            decode_frame_graph(json.loads(frame_graph_path.read_text(encoding="utf-8")))
            if frame_graph_path.is_file()
            else None
        )

        return SequenceDiagnostics(
            summary=summary,
            warnings=tuple(warnings),
            synchronization=synchronization,
            frame_graph=frame_graph,
        )

    def read_calibration(self) -> CalibrationSet | None:
        """Read this sequence's calibration set, when one was written.

        Returns:
            The calibration set, or ``None`` if the artifact was finalized
            without one (e.g. by a writer predating the calibration
            contract, or a source with no calibration available).
        """
        calibration_path = self._root / _CALIBRATION_FILENAME
        if not calibration_path.is_file():
            return None
        return decode_calibration_set(json.loads(calibration_path.read_text(encoding="utf-8")))

    def list_observations(self) -> list[SourceObservation]:
        """Return every observation in the artifact, in index order.

        This necessarily decodes every payload (image/LiDAR data included),
        so its cost is proportional to the artifact's total payload size —
        use it only when the whole sequence is genuinely needed. Looking up
        one observation or a subset should use :meth:`get_observation` or
        :func:`~contextmap.ingestion.sequence_observation_selection.resolve_selection`
        instead, which cost memory proportional to the index, not to the
        payload.

        Returns:
            All observations, decoded from the index and their payload
            files.
        """
        return list(self._iter_observations())

    def get_observation(self, observation_id: SourceObservationId) -> SourceObservation:
        """Return a single observation by identity.

        Resolves ``observation_id`` through an id-to-index-offset lookup
        (built once per reader, from the index alone, and cached) instead
        of scanning and decoding every preceding observation's payload.

        Args:
            observation_id: Identity of the observation to look up.

        Returns:
            The matching observation.

        Raises:
            SequenceArtifactError: If no observation has this identity.
        """
        offset = self._offsets_by_id().get(observation_id)
        if offset is None:
            raise SequenceArtifactError(f"observation not found: {observation_id!r}")
        return self.observation_at(offset)

    def iter_index(self) -> Iterator[IndexEntry]:
        """Stream this sequence's index in canonical order, without reading any payload.

        Cost is proportional to the index size (``index.jsonl``), never to
        the total payload size: no image/LiDAR payload file is opened.
        Used by :func:`~contextmap.ingestion.sequence_observation_selection.resolve_selection`
        to decide which observations match a selection before paying the
        cost of decoding any of them.

        Yields:
            One :class:`IndexEntry` per observation, in canonical index order.
        """
        index_path = self._root / _INDEX_FILENAME
        with index_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                yield IndexEntry(
                    offset=offset,
                    observation_id=SourceObservationId(record["observation_id"]),
                    timestamp=SourceTimestamp(**record["timestamp"]),
                    modality=record["modality"],
                )

    def observation_at(self, offset: int, *, load_payload: bool = True) -> SourceObservation:
        """Decode a single observation given a byte offset from :meth:`iter_index`.

        Reads exactly one index line and, when its modality has one and
        ``load_payload`` is true, its one payload file — never any other
        observation's data.

        Args:
            offset: An :attr:`IndexEntry.offset` returned by this same reader's
                :meth:`iter_index`.
            load_payload: When ``False``, decode only the metadata carried by
                the index line: an ``ImageObservation``/``LidarObservation``
                is returned with ``data=b""`` and no payload file is opened.
                Used where only identity/provenance/shape metadata is
                needed, e.g. reconstructing dropped-event diagnostics.

        Returns:
            The decoded observation.
        """
        index_path = self._root / _INDEX_FILENAME
        with index_path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        return _decode_observation(
            json.loads(line.decode("utf-8")), self._root, load_payload=load_payload
        )

    def _offsets_by_id(self) -> dict[SourceObservationId, int]:
        """Build (once) and cache the id-to-index-offset lookup used by :meth:`get_observation`."""
        if self._offsets_by_id_cache is None:
            self._offsets_by_id_cache = {
                entry.observation_id: entry.offset for entry in self.iter_index()
            }
        return self._offsets_by_id_cache

    def _iter_observations(self) -> Iterator[SourceObservation]:
        """Decode all observations sequentially from the canonical index.

        Yields:
            Observations in index order, including payload bytes for modalities that store them
            out of line.
        """
        index_path = self._root / _INDEX_FILENAME
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                yield _decode_observation(json.loads(stripped), self._root)
