"""Integrity and containment checks for local model checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path


def combined_checkpoint_fingerprint(base_path: Path, alpha_path: Path) -> str:
    """Hash base and alpha checkpoint bytes in an unambiguous order."""
    digest = hashlib.sha256()
    for label, path in ((b"base\0", base_path), (b"alpha\0", alpha_path)):
        digest.update(label)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_checkpoint_file(root: Path, reference: str, *, error_type: type[Exception]) -> Path:
    """Resolve a required local file without allowing path escape."""
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise error_type(f"reference escapes configured root: {reference!r}")
    if not candidate.is_file():
        raise error_type(f"required local file does not exist: {reference!r}")
    return candidate
