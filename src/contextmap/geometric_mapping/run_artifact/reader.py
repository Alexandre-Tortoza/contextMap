"""Read and verify persisted Geometric Mapping run artifacts from disk."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

from contextmap.geometric_mapping.geometry_storage import PackedGeometry
from contextmap.geometric_mapping.serialization import (
    decode_geometric_map,
    decode_scan_record,
)
from contextmap.shared import check_file_inventory

from .manifest import (
    _CONTRACTUAL_DIRECTORIES,
    _CONTRACTUAL_RECORDS,
    _GEOMETRY,
    _MANIFEST,
    _MAP_METADATA,
    _SOURCE_INDEX,
    GeometricMapArtifactManifest,
    IncompleteMapArtifactError,
    MapArtifactError,
    decode_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON record.
    """
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _load_manifest(run_dir: Path) -> GeometricMapArtifactManifest:
    """Load and deserialize a run manifest from disk.

    Args:
        run_dir: Path to the run's artifact directory.

    Returns:
        The parsed manifest.

    Raises:
        IncompleteMapArtifactError: If ``manifest.json`` is missing.
        MapArtifactError: If the schema version is not understood.
    """
    manifest_path = run_dir / _MANIFEST
    if not manifest_path.is_file():
        raise IncompleteMapArtifactError(f"missing {_MANIFEST} in {run_dir}")
    return decode_manifest(_read_json(manifest_path))


class GeometricMapArtifactReader:
    """Reads a finalized Geometric Mapping run from its own directory.

    Use it as a context manager (or call :meth:`close`) so the memory map of the
    geometry payload is released.
    """

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file
                outside it is needed.

        Raises:
            IncompleteMapArtifactError: If ``manifest.json`` is missing.
            MapArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = _load_manifest(run_dir)
        self._geometry: PackedGeometry | None = None
        self._payload: BinaryIO | None = None
        self._map: mmap.mmap | None = None

    def __enter__(self) -> GeometricMapArtifactReader:
        """Return the reader for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the geometry payload."""
        self.close()

    @property
    def manifest(self) -> GeometricMapArtifactManifest:
        """The run's manifest."""
        return self._manifest

    def geometry(self) -> PackedGeometry:
        """Open the map's geometry as a :class:`~contextmap.geometric_mapping.GeometrySource`.

        The payload is memory-mapped, not read.

        Returns:
            The geometry, backed by the persisted payload and source index.

        Raises:
            MapArtifactError: If a contractual file is missing or the payload does
                not match the map's metadata (for example, it is truncated).
        """
        if self._geometry is not None:
            return self._geometry
        for relative in (_GEOMETRY, _SOURCE_INDEX, _MAP_METADATA):
            if not (self._root / relative).is_file():
                raise MapArtifactError(f"missing {relative} in {self._root}")
        geometric_map = decode_geometric_map(_read_json(self._root / _MAP_METADATA))
        scans = [
            decode_scan_record(json.loads(line))
            for line in (self._root / _SOURCE_INDEX).read_text(encoding="utf-8").splitlines()
            if line
        ]
        handle = (self._root / _GEOMETRY).open("rb")
        try:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as error:
            handle.close()
            raise MapArtifactError(f"{_GEOMETRY} cannot be mapped: {error}") from error
        try:
            geometry = PackedGeometry(geometric_map=geometric_map, scans=scans, records=mapped)
        except ValueError as error:
            mapped.close()
            handle.close()
            raise MapArtifactError(str(error)) from error
        self._payload, self._map, self._geometry = handle, mapped, geometry
        return geometry

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a contractual JSON record.

        Args:
            relative_path: For example ``"metrics/mapping.json"`` or ``"lineage.json"``.

        Returns:
            The parsed record.

        Raises:
            MapArtifactError: If the path is not a contractual JSON record;
                ``__DEBUG__/`` and the geometry payload are never valid sources.
        """
        contractual = relative_path in _CONTRACTUAL_RECORDS or (
            relative_path.endswith(".json") and relative_path.startswith(_CONTRACTUAL_DIRECTORIES)
        )
        if not contractual or ".." in relative_path:
            raise MapArtifactError(f"not a contractual JSON record: {relative_path!r}")
        return _read_json(self._root / relative_path)

    def verify_integrity(self, *, check_index: bool = True) -> list[str]:
        """Check the file inventory and the derived index against what is on disk.

        Args:
            check_index: Also recompute the derived bounds from the geometry and
                compare them with the recorded ones; this reads the whole payload.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        problems = check_file_inventory(self._root, self._manifest.file_inventory)
        if check_index:
            try:
                problems.extend(self.geometry().verify_index())
            except MapArtifactError as error:
                problems.append(str(error))
        return problems

    def close(self) -> None:
        """Release the geometry payload; the geometry must not be read afterwards."""
        if self._geometry is not None:
            self._geometry.close()
        if self._map is not None:
            self._map.close()
        if self._payload is not None:
            self._payload.close()
        self._geometry = self._map = self._payload = None
