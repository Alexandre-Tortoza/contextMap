"""The frozen observation selection of the perception pilot (#665) and its foundation check.

The pilot perceives the images a frozen scenario snapshot selected: by default the 20-image
sample of corridor-02 in scenario 1.0.5 (``solution-1-canonical-1.0.5-corridor-02-sample.json``
under ``src/contextmap/evaluation/docs/scenarios/``). Every arm receives exactly those images as
an ``explicit_ids`` observation selection, the form the runtime's ``inputs.observation_selection``
accepts. A selection the foundation's sequence does not hold is refused before anything runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.evaluation import SCENARIO_SCHEMA


@dataclass(frozen=True, kw_only=True)
class FrozenSelection:
    """The images a scenario snapshot selected, with the identity of the snapshot.

    Attributes:
        scenario_id: The scenario the snapshot freezes.
        scenario_version: Its version.
        scenario_digest: The digest the snapshot records for itself.
        subject_id: The subject whose selection this is.
        sequence_artifact_id: The sequence the selection was made over, when the snapshot pins
            one; ``None`` for a subject that pins none (the CI subset).
        observation_ids: The selected image observations, in snapshot order.
        sha256: SHA-256 of the snapshot file's bytes.
    """

    scenario_id: str
    scenario_version: str
    scenario_digest: str | None
    subject_id: str
    sequence_artifact_id: str | None
    observation_ids: tuple[str, ...]
    sha256: str

    def to_document(self, file: str) -> dict[str, Any]:
        """Return the protocol record of the selection; ``file`` names the snapshot portably."""
        return {
            "source": {
                "file": file,
                "sha256": self.sha256,
                "scenario_id": self.scenario_id,
                "scenario_version": self.scenario_version,
                "scenario_digest": self.scenario_digest,
                "subject_id": self.subject_id,
                "sequence_artifact_id": self.sequence_artifact_id,
            },
            "observation_ids": list(self.observation_ids),
            "observation_selection": self.observation_selection(),
        }

    def observation_selection(self) -> dict[str, Any]:
        """Return the canonical ``explicit_ids`` selection document of the images."""
        return {"kind": "explicit_ids", "observation_ids": sorted(self.observation_ids)}


def read_frozen_selection(path: Path) -> FrozenSelection:
    """Read the selected images of a scenario snapshot (``contextmap.e2e.scenario/v1``).

    Raises:
        ValueError: If the file cannot be read, is not a scenario snapshot, or selects no image
            or the same image twice.
    """
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the selection snapshot {path.name}: {error}") from None
    if not isinstance(document, dict) or document.get("schema") != SCENARIO_SCHEMA:
        raise ValueError(f"{path.name} is not a scenario snapshot ({SCENARIO_SCHEMA})")
    try:
        subject = document["subject"]
        images = [str(item["observation_id"]) for item in subject["selection"]["selected_images"]]
        selection = FrozenSelection(
            scenario_id=str(document["scenario_id"]),
            scenario_version=str(document["version"]),
            scenario_digest=document.get("digest"),
            subject_id=str(subject["subject_id"]),
            sequence_artifact_id=subject.get("sequence_artifact_id"),
            observation_ids=tuple(images),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path.name} has no subject selection to read: {error!r}") from None
    if not images or len(set(images)) != len(images):
        raise ValueError(f"{path.name} must select at least one image, each once, got {images}")
    return selection


def check_against_foundation(
    selection: FrozenSelection, *, sequence_artifact_id: str, image_ids: Collection[str]
) -> None:
    """Refuse a selection the foundation's sequence cannot serve.

    Args:
        selection: The frozen selection.
        sequence_artifact_id: The foundation's sequence.
        image_ids: Every image observation of that sequence.

    Raises:
        ValueError: If the snapshot pins another sequence, or selects an image the sequence does
            not hold.
    """
    pinned = selection.sequence_artifact_id
    if pinned is not None and pinned != sequence_artifact_id:
        raise ValueError(
            f"the selection {selection.subject_id!r} was frozen over sequence {pinned!r}, but the "
            f"foundation's sequence is {sequence_artifact_id!r}"
        )
    known = set(image_ids)
    missing = [image for image in selection.observation_ids if image not in known]
    if missing:
        raise ValueError(
            f"the foundation's sequence {sequence_artifact_id!r} holds no image {missing} of the "
            f"selection {selection.subject_id!r}"
        )
