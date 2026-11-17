"""Spy on ``os.fsync`` to check the order of a durable publication (#616, #640).

Every writer of a contractual artifact must force its data to stable storage before the name
that publishes it, and the directory holding that name right after. The tests of each writer
share this recorder instead of each keeping a copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

Identity = tuple[int, int]


def identity(path: Path) -> Identity | None:
    """Return ``(device, inode)`` of ``path``, or ``None`` when it does not exist."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


class SyncRecorder:
    """Records every ``os.fsync`` together with what the published path was at that moment.

    Identities are ``(device, inode)``: a rename or a hard link keeps them, so a file synced in a
    temporary location is recognized under its published path.
    """

    def __init__(self, published: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._published = published
        self._calls: list[tuple[Identity, Identity | None]] = []
        real_fsync = os.fsync

        def recording_fsync(descriptor: int) -> None:
            status = os.fstat(descriptor)
            self._calls.append(((status.st_dev, status.st_ino), identity(published)))
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", recording_fsync)

    def synced_before_publication(self, path: Path) -> bool:
        """Tell whether ``path`` was synced while the published path was not yet there."""
        target, published = identity(path), identity(self._published)
        return any(synced == target and seen != published for synced, seen in self._calls)

    def synced_after_publication(self, path: Path) -> bool:
        """Tell whether ``path`` was synced once the published path was there."""
        target, published = identity(path), identity(self._published)
        return any(synced == target and seen == published for synced, seen in self._calls)
