"""Persistent region mask storage with index and integrity verification.

Masks are persisted with bit-packing (one byte per pixel, canonical 0 or 1),
keyed by (source_observation_id, region_id) pair, with SHA-256 integrity
verification and a JSONL index for rapid lookup.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextmap.ingestion import SourceObservationId
    from contextmap.region_perception import Region2D, RegionId
    from contextmap.region_perception.region_geometry import InlineMask

MASK_INDEX_FILENAME = "mask-index.jsonl"
"""Canonical name of the mask index file."""

MASK_INDEX_SCHEMA_VERSION = "0.1.0"

_MASK_INDEX_RECORD_TYPE = "mask_index"
"""Record type of the mask index header line."""
"""Schema version for mask index."""


@dataclass(frozen=True)
class MaskPayloadEntry:
    """Metadata for one persisted mask payload."""

    region_id: str
    source_observation_id: str
    payload_reference: str
    size_bytes: int
    content_hash: str
    width: int
    height: int

    def to_dict(self) -> dict:
        """Convert to JSON-compatible dict."""
        return {
            "region_id": self.region_id,
            "source_observation_id": self.source_observation_id,
            "payload_reference": self.payload_reference,
            "size_bytes": self.size_bytes,
            "content_hash": self.content_hash,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_dict(data: dict) -> MaskPayloadEntry:
        """Restore from JSON dict."""
        return MaskPayloadEntry(
            region_id=data["region_id"],
            source_observation_id=data["source_observation_id"],
            payload_reference=data["payload_reference"],
            size_bytes=data["size_bytes"],
            content_hash=data["content_hash"],
            width=data["width"],
            height=data["height"],
        )


class MaskStoreError(Exception):
    """Base error for mask store operations."""

    pass


class MaskPayloadIntegrityError(MaskStoreError):
    """Mask payload failed integrity check."""

    pass


def _encode_mask_payload(mask: InlineMask) -> bytes:
    """Bit-pack a mask and wrap it in the ``.npy`` envelope the payload reference promises.

    Args:
        mask: The mask to persist.

    Returns:
        The bytes to write: a real ``.npy`` file whose header carries dtype and shape, so the
        payload stays self-describing and ``numpy.load`` can open it outside this store.
    """
    import io

    import numpy as np

    packed = np.packbits(mask.as_array().reshape(-1))
    buffer = io.BytesIO()
    np.save(buffer, packed, allow_pickle=False)
    return buffer.getvalue()


def _decode_mask_payload(data: bytes, width: int, height: int) -> InlineMask:
    """Rebuild a mask from a persisted ``.npy`` payload.

    Args:
        data: The payload bytes as written, envelope included.
        width: Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        The reconstructed InlineMask.
    """
    import io

    import numpy as np

    from contextmap.region_perception.region_geometry import InlineMask

    packed = np.load(io.BytesIO(data), allow_pickle=False)
    unpacked = np.unpackbits(packed)
    pixels = unpacked[: width * height].reshape((height, width)).astype(np.bool_)
    return InlineMask(pixels)


class MaskStoreWriter:
    """Writes mask payloads atomically to disk."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize writer for mask payloads.

        Args:
            output_dir: Directory to write masks to.
        """
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[MaskPayloadEntry] = []
        self._staged: dict[str, tuple[str, bytes, int, int]] = {}
        self._keys: set[tuple[str, str]] = set()

    def write(
        self,
        region_id: RegionId,
        source_observation_id: SourceObservationId,
        mask: InlineMask,
    ) -> MaskPayloadEntry:
        """Write a mask payload.

        Args:
            region_id: The region's ID (local to source_observation_id).
            source_observation_id: The source observation this region came from.
            mask: The InlineMask to persist.

        Returns:
            The entry describing the persisted payload.

        Raises:
            MaskStoreError: If this (source_observation_id, region_id) was already staged,
                or if the payload_reference would escape the store root.
        """
        region_id_str = str(region_id)
        source_obs_str = str(source_observation_id)
        key = (source_obs_str, region_id_str)

        if key in self._keys:
            raise MaskStoreError(
                f"duplicate mask payload key: {source_obs_str!r}, {region_id_str!r}"
            )

        # Generate payload reference: source_observation_id/region_id.npy
        payload_reference = f"{source_obs_str}/{region_id_str}.npy"

        # Validate path containment: ensure reference doesn't escape root
        reference_path = (self._output_dir / payload_reference).resolve()
        output_dir_resolved = self._output_dir.resolve()
        try:
            reference_path.relative_to(output_dir_resolved)
        except ValueError as err:
            raise MaskStoreError(
                f"payload_reference resolves outside the store root: {payload_reference!r}"
            ) from err

        # O hash cobre o payload como ele vai para o disco, envelope incluso.
        packed = _encode_mask_payload(mask)
        sha256 = hashlib.sha256(packed).hexdigest()

        # Write the payload
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(packed)

        # Create entry
        entry = MaskPayloadEntry(
            region_id=region_id_str,
            source_observation_id=source_obs_str,
            payload_reference=payload_reference,
            size_bytes=len(packed),
            content_hash=f"sha256:{sha256}",
            width=mask.width,
            height=mask.height,
        )

        self._entries.append(entry)
        self._staged[payload_reference] = (sha256, packed, mask.width, mask.height)
        self._keys.add(key)

        return entry

    def entries(self) -> Sequence[MaskPayloadEntry]:
        """Get all entries written so far."""
        return list(self._entries)

    def entry_for(self, payload_reference: str) -> MaskPayloadEntry:
        """Retrieve the entry for a specific payload reference.

        Args:
            payload_reference: The reference string (e.g., "frame-001/region-0.npy").

        Returns:
            The MaskPayloadEntry for this reference.

        Raises:
            MaskStoreError: If the reference was not staged by this writer.
        """
        for entry in self._entries:
            if entry.payload_reference == payload_reference:
                return entry
        raise MaskStoreError(f"payload reference {payload_reference!r} not staged by this writer")

    def load(self, payload_reference: str) -> InlineMask:
        """Load a mask from a staged payload (before index is written).

        Args:
            payload_reference: The reference string (e.g., "frame-001/region-0.npy").

        Returns:
            The reconstructed InlineMask.

        Raises:
            MaskStoreError: If the reference was not staged, or the file is missing.
            MaskPayloadIntegrityError: If the hash verification fails.
        """
        if payload_reference not in self._staged:
            raise MaskStoreError(f"payload reference {payload_reference!r} not staged")

        sha256_expected, _, width, height = self._staged[payload_reference]

        filepath = self._output_dir / payload_reference
        if not filepath.exists():
            raise MaskStoreError(f"missing payload file: {payload_reference!r}")

        packed = filepath.read_bytes()
        sha256_computed = hashlib.sha256(packed).hexdigest()

        if sha256_computed != sha256_expected:
            raise MaskPayloadIntegrityError(
                f"content hash mismatch for {payload_reference!r}: "
                f"{sha256_computed} != {sha256_expected}"
            )

        return _decode_mask_payload(packed, width, height)


class MaskStoreReader:
    """Reads mask payloads from disk with integrity verification."""

    def __init__(self, root_dir: Path, entries: dict[tuple[str, str], MaskPayloadEntry]) -> None:
        """Initialize reader.

        Args:
            root_dir: Directory containing mask files.
            entries: Map of (source_observation_id, region_id) to MaskPayloadEntry.
        """
        self._root = root_dir
        self._entries = entries

    @staticmethod
    def open(root_dir: Path) -> MaskStoreReader:
        """Open an existing mask store.

        Args:
            root_dir: Directory containing the mask store.

        Returns:
            Reader initialized from the index, or empty reader if no index exists.
        """
        index_path = root_dir / MASK_INDEX_FILENAME
        entries: dict[tuple[str, str], MaskPayloadEntry] = {}
        header_found = False

        if index_path.exists():
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                # O header é obrigatório e é a primeira linha: um índice sem ele não declara
                # qual esquema está lendo, e seguir adiante seria adivinhar.
                if not header_found:
                    if data.get("record_type") != _MASK_INDEX_RECORD_TYPE:
                        raise MaskStoreError("mask index is missing its schema header")
                    schema_version = data.get("schema_version")
                    if schema_version != MASK_INDEX_SCHEMA_VERSION:
                        raise MaskStoreError(
                            f"unsupported mask index schema_version: {schema_version!r}"
                        )
                    header_found = True
                    continue
                entry = decode_mask_payload_entry(data)
                key = (entry.source_observation_id, entry.region_id)
                entries[key] = entry

        return MaskStoreReader(root_dir, entries)

    def entry(
        self, source_observation_id: SourceObservationId, region_id: RegionId
    ) -> MaskPayloadEntry:
        """Return one mask's indexed metadata without loading its payload.

        A consumer that only needs the shape, the hash or the payload reference of a
        persisted mask reads it here, instead of paying a full decode to find out.

        Args:
            source_observation_id: The physical observation owning the region.
            region_id: The observation-local region whose entry to return.

        Returns:
            The indexed metadata of that region's mask.

        Raises:
            MaskStoreError: If the index holds no entry for that key.
        """
        key = (str(source_observation_id), str(region_id))
        try:
            return self._entries[key]
        except KeyError:
            raise MaskStoreError(
                "no persisted mask for "
                f"source_observation_id={str(source_observation_id)!r}, "
                f"region_id={str(region_id)!r}"
            ) from None

    def load(self, source_observation_id: SourceObservationId, region_id: RegionId) -> InlineMask:
        """Load a mask with integrity verification.

        Args:
            source_observation_id: The source observation ID.
            region_id: The region ID (local to source_observation_id).

        Returns:
            The reconstructed InlineMask.

        Raises:
            MaskStoreError: If no persisted mask exists for this key.
            MaskPayloadIntegrityError: If hash verification fails.
        """
        source_obs_str = str(source_observation_id)
        region_id_str = str(region_id)
        key = (source_obs_str, region_id_str)

        if key not in self._entries:
            raise MaskStoreError(f"no persisted mask for {source_obs_str!r}, {region_id_str!r}")

        entry = self._entries[key]
        filepath = self._root / entry.payload_reference
        packed = filepath.read_bytes()

        # Verify integrity
        sha256_computed = hashlib.sha256(packed).hexdigest()
        expected_hash = entry.content_hash.removeprefix("sha256:")
        if sha256_computed != expected_hash:
            raise MaskPayloadIntegrityError(
                f"content hash mismatch for {source_obs_str!r}, {region_id_str!r}: "
                f"{sha256_computed} != {expected_hash}"
            )

        return _decode_mask_payload(packed, entry.width, entry.height)

    def region_keys(self) -> tuple[tuple[SourceObservationId, RegionId], ...]:
        """Get all region keys in the store.

        Returns:
            Tuple of (source_observation_id, region_id) pairs.
        """
        from contextmap.ingestion import SourceObservationId
        from contextmap.region_perception import RegionId

        return tuple(
            (SourceObservationId(k[0]), RegionId(k[1])) for k in sorted(self._entries.keys())
        )


@runtime_checkable
class RegionMaskResolver(Protocol):
    """Where a region's externalized mask reference is turned back into pixels."""

    def resolve_mask(self, reference: str) -> InlineMask:
        """Load the mask a reference points at.

        Args:
            reference: The mask reference a region carries, as the store wrote it.

        Returns:
            The mask the reference names.

        Raises:
            MaskStoreError: If this resolver cannot serve the reference.
            MaskPayloadIntegrityError: If the payload fails its integrity check.
        """
        ...


def region_mask_pixels(region: Region2D, resolver: RegionMaskResolver | None) -> InlineMask | None:
    """Return a region's full-image mask, wherever its pixels live.

    Args:
        region: The region whose mask is wanted.
        resolver: Where a ``mask_reference`` is resolved; may be ``None`` only for regions
            known to carry their mask inline, or to carry no mask at all.

    Returns:
        The inline mask when the region carries one, the resolved mask when it only carries
        a reference, and ``None`` for a box-only region (neither inline mask nor reference).

    Raises:
        MaskStoreError: If the region only has a reference and no ``resolver`` was given, or
            the resolver cannot serve the reference.
        MaskPayloadIntegrityError: If the referenced payload fails its integrity check.
    """
    if region.mask is not None:
        return region.mask
    if region.mask_reference is None:
        return None
    # NOTE: uma referência sem resolver é falha de acesso a uma máscara que existe, não
    # ausência de máscara. Devolver None aqui apagaria essa distinção, e o chamador seguiria
    # como se a região fosse box-only.
    if resolver is None:
        raise MaskStoreError(
            f"region {region.region_id!r} only has mask_reference {region.mask_reference!r} and "
            "no resolver was given to load its pixels"
        )
    return resolver.resolve_mask(region.mask_reference)


def write_mask_index(root_dir: Path, entries: Sequence[MaskPayloadEntry]) -> None:
    """Write the mask index to disk as JSONL with schema header.

    Args:
        root_dir: Directory to write index to.
        entries: Mask entries to index.
    """
    index_path = root_dir / MASK_INDEX_FILENAME
    root_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: o header declara record_type além da versão, e tudo é gravado com sort_keys. O
    # índice entra no inventário do run com hash próprio, então a ordem das chaves é parte do
    # byte publicado: sem sort_keys, reordenar um campo do dataclass mudaria o artifact.
    header = {
        "record_type": _MASK_INDEX_RECORD_TYPE,
        "schema_version": MASK_INDEX_SCHEMA_VERSION,
    }
    content = f"{json.dumps(header, sort_keys=True)}\n" + "".join(
        f"{json.dumps(encode_mask_payload_entry(entry), sort_keys=True)}\n" for entry in entries
    )
    index_path.write_text(content, encoding="utf-8")


def encode_mask_payload_entry(entry: MaskPayloadEntry) -> dict[str, Any]:
    """Encode a :class:`MaskPayloadEntry` into a JSON-serializable dict."""
    return {
        "region_id": str(entry.region_id),
        "source_observation_id": str(entry.source_observation_id),
        "width": entry.width,
        "height": entry.height,
        "payload_reference": entry.payload_reference,
        "content_hash": entry.content_hash,
        "size_bytes": entry.size_bytes,
    }


def decode_mask_payload_entry(record: dict[str, Any]) -> MaskPayloadEntry:
    """Decode a :class:`MaskPayloadEntry` from :func:`encode_mask_payload_entry`'s output."""
    return MaskPayloadEntry(
        region_id=record["region_id"],
        source_observation_id=record["source_observation_id"],
        width=record["width"],
        height=record["height"],
        payload_reference=record["payload_reference"],
        content_hash=record["content_hash"],
        size_bytes=record["size_bytes"],
    )
