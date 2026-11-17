"""The viewer's index of one campaign run (#670), published as each arm finishes.

The external viewer reads ``maps.json`` while the campaign runs, so the index is never written in
place: each version is written to a temporary file in the same directory, forced to stable
storage and renamed over the previous one (``os.replace``), and a reader sees either the previous
index or the new one, whole. Each finished arm also gets its own entry, ``maps/<arm>.json``,
published once: an entry that exists is never replaced. The index lists the arms in the order
they finished and the voxel layers of the foundation, the same for every arm.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ViewerIndex:
    """The ``maps.json`` of one campaign run and the entry of each finished arm."""

    def __init__(self, harness: Path, *, schema: str) -> None:
        """Hold an empty index of a run's harness directory; nothing is written yet.

        Args:
            harness: The run's harness directory, where ``maps.json`` and ``maps/`` live.
            schema: The index's schema identity.
        """
        self._harness = harness
        self._document: dict[str, Any] = {
            "schema": schema,
            "paths_relative_to": "workspace",
            "voxel_layers": None,
            "arms": {},
        }

    def start(self) -> None:
        """Publish the empty index, so a reader finds a valid one from the start of the run."""
        _publish(self._harness / "maps.json", self._document, replace=True)

    def publish_voxel_layers(self, layers: Mapping[str, str]) -> None:
        """Add the voxel layers, by cell size, relative to the harness directory; republish."""
        self._document["voxel_layers"] = {"paths_relative_to": "harness", "layers": dict(layers)}
        _publish(self._harness / "maps.json", self._document, replace=True)

    def publish_arm(self, arm: str, entry: Mapping[str, Any]) -> None:
        """Publish a finished arm's entry once, then the index that lists it.

        Raises:
            FileExistsError: If the arm's entry was already published; nothing changes.
            OSError: If a file cannot be written or renamed; the previous index stays whole.
        """
        _publish(self._harness / "maps" / f"{arm}.json", entry, replace=False)
        arms = {**self._document["arms"], arm: dict(entry)}
        _publish(self._harness / "maps.json", {**self._document, "arms": arms}, replace=True)
        self._document["arms"] = arms


def _publish(path: Path, document: Mapping[str, Any], *, replace: bool) -> None:
    """Write ``document`` as ``path`` atomically: a reader sees the old file or the new one.

    Args:
        path: The file to publish.
        document: The JSON document; its key order is kept, so arms stay in execution order.
        replace: Rename over an existing file; otherwise refuse one (a hard link fails when the
            name exists).

    Raises:
        FileExistsError: If ``replace`` is false and ``path`` exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    # O nome novo só é durável depois do fsync do diretório.
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
