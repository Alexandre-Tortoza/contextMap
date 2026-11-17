"""Leitura por identidade e integridade do artifact de Sensor Association.

Reconstrói contratos a partir das tabelas existentes e valida o inventário do run.
Não escreve, não consulta debug e não carrega NumPy para abrir ou ler observações.
"""

from __future__ import annotations

import json
from array import array
from bisect import bisect_left
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.region_perception import RegionId
from contextmap.sensor_association.artifacts.run_manifest import (
    DENSE_CELLS,
    DENSE_INDEX,
    GEOMETRY_INFIX,
    OBSERVATION_INDEX,
    OBSERVATIONS,
    QUALITY,
    SUPPORT,
    DenseAssociationRecord,
    RunArtifactError,
    SensorAssociationRunManifest,
    decode_array,
    load_manifest,
    require_contractual,
)
from contextmap.sensor_association.models import SpatialObservation
from contextmap.sensor_association.quality import ObservationQuality
from contextmap.sensor_association.serialization import (
    decode_observation_quality,
    decode_spatial_observation,
)
from contextmap.shared import check_file_inventory


class SensorAssociationRunReader:
    """Reads a finalized Sensor Association run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is
                needed.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing.
            RunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = load_manifest(run_dir)
        self._index: dict[str, dict[str, Any]] | None = None
        self._by_frame: dict[str, list[str]] | None = None
        self._dense_index: dict[tuple[str, str], dict[str, Any]] | None = None

    @property
    def manifest(self) -> SensorAssociationRunManifest:
        """The run's manifest."""
        return self._manifest

    def observations(self) -> Iterator[SpatialObservation]:
        """Iterate every spatial observation, in run order, reading the files sequentially."""
        for identity in self._load_index():
            yield self.observation(identity)

    def observation(self, spatial_observation_id: str) -> SpatialObservation:
        """Read one observation by identity without loading the others.

        Args:
            spatial_observation_id: Identity of the observation.

        Returns:
            The observation, revalidated against the contract, with its geometry support
            resolved from the columnar table.

        Raises:
            RunArtifactError: If the run has no such observation.
        """
        entry = self._entry(spatial_observation_id)
        record = json.loads(
            _read_slice(self._root / OBSERVATIONS, **entry["observation"]).decode("utf-8")
        )
        support = record.pop("support")
        map_id = MapId(record["provenance"]["geometric_map_id"])
        record["geometry_support"] = [
            {"map_id": str(map_id), "geometry_id": str(geometry_id_for(map_id=map_id, index=i))}
            for i in self._read_support(support["offset"], support["count"])
        ]
        return decode_spatial_observation(record)

    def observations_of_frame(self, source_observation_id: str) -> tuple[SpatialObservation, ...]:
        """Read every observation of one camera frame, in region order."""
        by_frame = self._load_by_frame()
        return tuple(self.observation(i) for i in by_frame.get(str(source_observation_id), []))

    def quality(self, spatial_observation_id: str) -> ObservationQuality:
        """Read the quality of one observation.

        Raises:
            RunArtifactError: If the run has no such observation.
        """
        entry = self._entry(spatial_observation_id)
        return decode_observation_quality(
            json.loads(_read_slice(self._root / QUALITY, **entry["quality"]).decode("utf-8"))
        )

    def geometry_support(self, spatial_observation_id: str) -> tuple[GeometryReference, ...]:
        """Read the geometry references one observation's region supports."""
        entry = self._entry(spatial_observation_id)
        record = json.loads(
            _read_slice(self._root / OBSERVATIONS, **entry["observation"]).decode("utf-8")
        )
        support = record["support"]
        map_id = self._manifest.geometric_map_id
        return tuple(
            GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
            for i in self._read_support(support["offset"], support["count"])
        )

    def regions_of(
        self, source_observation_id: str, reference: GeometryReference
    ) -> tuple[RegionId, ...]:
        """Tell which regions of a frame support a geometry element, in region order.

        An empty tuple means no region of that frame contains it; overlaps keep every region.
        """
        if reference.map_id != self._manifest.geometric_map_id:
            return ()
        position = int(str(reference.geometry_id).rsplit(GEOMETRY_INFIX, 1)[1])
        regions: list[RegionId] = []
        for identity in self._load_by_frame().get(str(source_observation_id), []):
            entry = self._entry(identity)
            record = json.loads(
                _read_slice(self._root / OBSERVATIONS, **entry["observation"]).decode("utf-8")
            )
            support = self._read_support(record["support"]["offset"], record["support"]["count"])
            at = bisect_left(support, position)
            if at < len(support) and support[at] == position:
                regions.append(RegionId(entry["region_id"]))
        return tuple(regions)

    def dense_association(
        self, source_observation_id: str, channel_id: str
    ) -> DenseAssociationRecord:
        """Read the dense feature association of one frame and channel.

        Raises:
            RunArtifactError: If the run has no such association.
        """
        record = self._load_dense_index().get((str(source_observation_id), channel_id))
        if record is None:
            raise RunArtifactError(
                f"no dense association for frame {source_observation_id!r} in channel "
                f"{channel_id!r}"
            )
        eligible, terms = record["eligible_count"], record["terms"]
        # Ordem das seções: pontos elegíveis, amostrados, linhas, colunas e pesos das células.
        sizes = (
            eligible * 4,
            eligible,
            eligible * terms * 4,
            eligible * terms * 4,
            eligible * terms * 4,
        )
        with (self._root / DENSE_CELLS).open("rb") as handle:
            handle.seek(record["byte_offset"])
            sections = [handle.read(size) for size in sizes]
        return DenseAssociationRecord(
            source_observation_id=record["source_observation_id"],
            channel_id=channel_id,
            provenance=record["provenance"],
            terms=terms,
            eligible_indices=decode_array("I", sections[0]),
            sampled=tuple(bool(b) for b in sections[1]),
            cell_rows=decode_array("i", sections[2]),
            cell_cols=decode_array("i", sections[3]),
            weights=decode_array("f", sections[4]),
        )

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSON record of a contractual directory;
                ``__DEBUG__/`` is never a valid source.
        """
        require_contractual(relative_path, ".json")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_records(self, relative_path: str) -> list[dict[str, Any]]:
        """Read the lines of a JSONL file from ``outputs/`` or ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSONL file of a contractual directory.
        """
        require_contractual(relative_path, ".jsonl")
        text = (self._root / relative_path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _entry(self, spatial_observation_id: str) -> dict[str, Any]:
        """Resolve an observation identity or raise RunArtifactError if it is absent."""
        entry = self._load_index().get(str(spatial_observation_id))
        if entry is None:
            raise RunArtifactError(
                f"unknown spatial observation in this run: {spatial_observation_id!r}"
            )
        return entry

    def _read_support(self, offset: int, count: int) -> array[int]:
        """Read a support slice as little-endian global geometry indices."""
        with (self._root / SUPPORT).open("rb") as handle:
            handle.seek(offset * 4)
            return decode_array("I", handle.read(count * 4))

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Lazily index observation records by their persistent identities."""
        if self._index is None:
            index: dict[str, dict[str, Any]] = {}
            for record in self.read_records(OBSERVATION_INDEX):
                index[record["spatial_observation_id"]] = record
            self._index = index
        return self._index

    def _load_by_frame(self) -> dict[str, list[str]]:
        """Group observation identities by frame while preserving record order."""
        if self._by_frame is None:
            by_frame: dict[str, list[str]] = {}
            for identity, record in self._load_index().items():
                by_frame.setdefault(record["source_observation_id"], []).append(identity)
            self._by_frame = by_frame
        return self._by_frame

    def _load_dense_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Lazily index declared dense associations by frame and channel."""
        if self._dense_index is None:
            index: dict[tuple[str, str], dict[str, Any]] = {}
            if (self._root / DENSE_INDEX).is_file():
                for record in self.read_records(DENSE_INDEX):
                    index[(record["source_observation_id"], record["channel_id"])] = record
            self._dense_index = index
        return self._dense_index


def _read_slice(path: Path, *, byte_offset: int, byte_length: int) -> bytes:
    """Read one byte range without loading the complete record file."""
    with path.open("rb") as handle:
        handle.seek(byte_offset)
        return handle.read(byte_length)
