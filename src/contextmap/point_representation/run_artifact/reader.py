"""Reading of a finalized Point Representation run from its own directory.

The reader opens a run without a registry, NumPy, a model library or the geometric map
itself. A representation, or only its vector, is read by identity or by the geometry it is
anchored to, through the byte offsets the indexes record, without loading the others.

Every record is decoded by the contract constructors, so a tampered record is refused
rather than trusted. ``__DEBUG__/`` is never a valid source.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import GeometryId, GeometryReference, MapId
from contextmap.point_representation.comparability import representation_space_fingerprint
from contextmap.point_representation.models import (
    FailedSupport,
    PointRepresentation,
    PointRepresentationId,
    RepresentationSpace,
)
from contextmap.point_representation.run_artifact.layout import (
    DTYPES,
    FAILED_SUPPORTS,
    GEOMETRY_INDEX,
    REPRESENTATION_INDEX,
    REPRESENTATIONS,
    SPACES,
    RunArtifactError,
)
from contextmap.point_representation.run_artifact.manifest import (
    PointRepresentationRunManifest,
    load_manifest,
)
from contextmap.point_representation.serialization import (
    decode_failed_support,
    decode_point_representation,
    decode_representation_space,
)
from contextmap.shared import check_file_inventory


class PointRepresentationRunReader:
    """Reads a finalized Point Representation run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file
                outside it is needed.

        Raises:
            IncompleteRunArtifactError: If ``manifest.json`` is missing.
            RunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = load_manifest(run_dir)
        self._offsets: dict[PointRepresentationId, tuple[int, int]] | None = None
        self._by_geometry: dict[GeometryReference, PointRepresentationId] | None = None

    @property
    def manifest(self) -> PointRepresentationRunManifest:
        """The run's manifest."""
        return self._manifest

    def space(self) -> RepresentationSpace:
        """Load the run's representation space, checked against its fingerprint.

        Raises:
            RunArtifactError: If the persisted space does not match the manifest.
        """
        record = json.loads((self._root / SPACES).read_text(encoding="utf-8"))
        space = decode_representation_space(record["spaces"][0])
        if representation_space_fingerprint(space) != self._manifest.representation_space_id:
            raise RunArtifactError("the persisted representation space does not match the manifest")
        return space

    def representation(self, representation_id: PointRepresentationId) -> PointRepresentation:
        """Read one representation by identity without loading the others.

        Raises:
            RunArtifactError: If the run has no such representation.
        """
        entry = self._load_offsets().get(representation_id)
        if entry is None:
            raise RunArtifactError(f"unknown representation_id in this run: {representation_id!r}")
        offset, length = entry
        with (self._root / REPRESENTATIONS).open("rb") as handle:
            handle.seek(offset)
            return self._decode(handle.read(length))

    def representation_for(self, geometry: GeometryReference) -> PointRepresentation | None:
        """Read the representation anchored to a geometry element, if the run has one."""
        representation_id = self._load_geometry_index().get(geometry)
        return None if representation_id is None else self.representation(representation_id)

    def iter_representations(self) -> Iterator[PointRepresentation]:
        """Iterate every representation in the order they were persisted."""
        with (self._root / REPRESENTATIONS).open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield self._decode(line)

    def vector(self, representation_id: PointRepresentationId) -> tuple[float, ...]:
        """Read one representation's vector without reading the rest of the payload.

        Components the representation lists as undefined hold zero placeholders
        and must not be interpreted.

        Raises:
            RunArtifactError: If the run has no such representation, its
                ``payload_reference`` does not point into this run's payload, or
                the payload is truncated.
        """
        representation = self.representation(representation_id)
        path, _, row_text = (representation.payload_reference or "").partition("#")
        if path != self._manifest.payload_path or not row_text.isdigit():
            raise RunArtifactError(
                f"payload_reference {representation.payload_reference!r} does not point into "
                f"this run's payload {self._manifest.payload_path!r}"
            )
        code, item_size, _ = DTYPES[self._manifest.dtype]
        size = item_size * self._manifest.dimension
        with (self._root / self._manifest.payload_path).open("rb") as handle:
            handle.seek(int(row_text) * size)
            data = handle.read(size)
        if len(data) != size:
            raise RunArtifactError(
                f"payload truncated: row {row_text} of {self._manifest.payload_path} needs "
                f"{size} bytes, found {len(data)}"
            )
        return tuple(
            float(value) for value in struct.unpack(f"<{self._manifest.dimension}{code}", data)
        )

    def failed_supports(self) -> list[FailedSupport]:
        """Read every support that produced no representation, with its reason."""
        text = (self._root / FAILED_SUPPORTS).read_text(encoding="utf-8")
        return [decode_failed_support(json.loads(line)) for line in text.splitlines() if line]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Raises:
            RunArtifactError: If the path is not a JSON record of a contractual
                directory; ``__DEBUG__/`` is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith(
            ("outputs/", "metrics/")
        ):
            raise RunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _decode(self, line: bytes) -> PointRepresentation:
        representation = decode_point_representation(json.loads(line))
        if representation.representation_space_id != self._manifest.representation_space_id:
            raise RunArtifactError(
                f"representation {representation.representation_id!r} is not in the run's space"
            )
        return representation

    def _load_offsets(self) -> dict[PointRepresentationId, tuple[int, int]]:
        if self._offsets is None:
            offsets: dict[PointRepresentationId, tuple[int, int]] = {}
            text = (self._root / REPRESENTATION_INDEX).read_text(encoding="utf-8")
            for line in text.splitlines():
                record = json.loads(line)
                offsets[PointRepresentationId(record["representation_id"])] = (
                    record["byte_offset"],
                    record["byte_length"],
                )
            self._offsets = offsets
        return self._offsets

    def _load_geometry_index(self) -> dict[GeometryReference, PointRepresentationId]:
        if self._by_geometry is None:
            index: dict[GeometryReference, PointRepresentationId] = {}
            text = (self._root / GEOMETRY_INDEX).read_text(encoding="utf-8")
            for line in text.splitlines():
                record = json.loads(line)
                reference = GeometryReference(
                    map_id=MapId(record["map_id"]), geometry_id=GeometryId(record["geometry_id"])
                )
                index[reference] = PointRepresentationId(record["representation_id"])
            self._by_geometry = index
        return self._by_geometry
