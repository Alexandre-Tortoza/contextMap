"""Deterministic digests and immutable, durable JSON persistence shared by evaluation manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

_CHUNK_BYTES = 1 << 20


def canonical_json(value: object) -> str:
    """Return the canonical JSON text digests are computed over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: object) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON form of a JSON-compatible value."""
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def file_digest(path: Path) -> str:
    """Return ``sha256:<hex>`` of a file, streamed so large payloads stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_immutable_json(path: Path, value: object, artifact_name: str) -> None:
    """Publish a JSON document atomically and durably, refusing to replace an existing one.

    The document is written to a temporary sibling, forced to stable storage (``fsync``)
    and only then hard-linked to ``path``, which fails if the name is taken; the directory
    is forced right after, so after a power loss ``path`` either does not exist or holds
    the whole document. It is the discipline of the runtime's JSON documents (#616), at
    the cost of two ``fsync`` calls per document.

    Args:
        path: Where the document is published; its directory is created when missing.
        value: JSON-compatible content, written indented, key-sorted and UTF-8.
        artifact_name: What the document is, for the error message.

    Raises:
        FileExistsError: If ``path`` already exists; it is left untouched.
        OSError: If the document or its directory cannot be forced to stable storage;
            the document is not published when its content could not be synced.
    """
    if path.exists():
        raise FileExistsError(f"{artifact_name} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"{artifact_name} already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    # O nome novo (e a remoção do temporário) também precisa chegar ao disco.
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Force a directory's entries to stable storage; opening it read-only is POSIX."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
