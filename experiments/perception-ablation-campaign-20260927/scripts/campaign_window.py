"""The window of the capability ablation campaign (#670) and the images it selects.

The ingestion lane (#669) cut corridor-02 from 2:30 to 2:35 and recorded that cut in a
``window.json`` (``contextmap.corridor-window/v1``). The campaign reads only the fields it uses:
the schema, the sequence the window was ingested into, the clock and the ``[start_ns, end_ns)``
interval on it, and whether the pose and LiDAR gaps stayed within their thresholds. Every other
field, including the ones #669 added after the contract (``start_recording_ns``,
``canonical_sequence_artifact_id``, ...), is kept as given and recorded, never rejected.

From the foundation's sequence, every ``stride``-th camera image whose timestamp lies in the
interval is selected, the first one included, in the order of the canonical index. A window over
another sequence or another clock, a window without a camera image, and a window outside the gap
thresholds (unless explicitly accepted) are refused before anything exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.ingestion import SequenceArtifactReader

WINDOW_SCHEMA = "contextmap.corridor-window/v1"
"""The contract of #669's window record."""


@dataclass(frozen=True, kw_only=True)
class CorridorWindow:
    """The window #669 ingested, as its record states it.

    Attributes:
        document: The record, every field as given.
        sha256: SHA-256 of the record file's bytes.
        file: The record's file name.
        sequence_artifact_id: The sequence the window was ingested into.
        clock_id: The clock ``start_ns`` and ``end_ns`` are on: the clock of the ingested
            observation timestamps, after #669's offset correction.
        start_ns: First instant of the window, included.
        end_ns: End of the window, excluded.
        within_thresholds: Whether the largest pose and LiDAR gaps stayed within their
            thresholds; false also when a gap could not be measured.
    """

    document: Mapping[str, Any]
    sha256: str
    file: str
    sequence_artifact_id: str
    clock_id: str
    start_ns: int
    end_ns: int
    within_thresholds: bool


@dataclass(frozen=True, kw_only=True)
class WindowSelection:
    """The camera images the campaign perceives.

    Attributes:
        window_images: Camera images of the sequence inside the window.
        stride: One image in every ``stride`` is selected, from the first.
        observation_ids: The selected images, in the order of the canonical index.
    """

    window_images: int
    stride: int
    observation_ids: tuple[str, ...]

    def observation_selection(self) -> dict[str, Any]:
        """Return the canonical ``explicit_ids`` selection document of the images."""
        return {"kind": "explicit_ids", "observation_ids": sorted(self.observation_ids)}

    def to_document(self) -> dict[str, Any]:
        """Return the protocol record of the selection."""
        return {
            "image_stride": self.stride,
            "window_images": self.window_images,
            "observation_ids": list(self.observation_ids),
            "observation_selection": self.observation_selection(),
        }


def read_window(path: Path) -> CorridorWindow:
    """Read #669's window record, checking only the fields the campaign uses.

    Raises:
        ValueError: If the file cannot be read, follows another schema, or a used field is
            missing or of the wrong type.
    """
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the window record {path.name}: {error}") from None
    if not isinstance(document, dict) or document.get("schema") != WINDOW_SCHEMA:
        found = document.get("schema") if isinstance(document, dict) else None
        raise ValueError(f"{path.name} is not a window record {WINDOW_SCHEMA!r} (found {found!r})")
    quality = document.get("quality")
    fields = {
        "sequence_artifact_id": (document.get("sequence_artifact_id"), str),
        "clock_id": (document.get("clock_id"), str),
        "start_ns": (document.get("start_ns"), int),
        "end_ns": (document.get("end_ns"), int),
        "quality.within_thresholds": (
            quality.get("within_thresholds") if isinstance(quality, dict) else None,
            bool,
        ),
    }
    # bool é subclasse de int: um instante não pode ser `true`.
    problems = [
        f"{name} must be a {kind.__name__}, got {value!r}"
        for name, (value, kind) in fields.items()
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool))
    ]
    if problems:
        raise ValueError(f"{path.name} cannot be used: " + "; ".join(problems))
    if document["end_ns"] <= document["start_ns"]:
        raise ValueError(
            f"{path.name} ends ({document['end_ns']}) before it starts ({document['start_ns']})"
        )
    return CorridorWindow(
        document=document,
        sha256=hashlib.sha256(data).hexdigest(),
        file=path.name,
        sequence_artifact_id=document["sequence_artifact_id"],
        clock_id=document["clock_id"],
        start_ns=document["start_ns"],
        end_ns=document["end_ns"],
        within_thresholds=document["quality"]["within_thresholds"],
    )


def select_images(
    window: CorridorWindow,
    sequence: Path,
    *,
    sequence_artifact_id: str,
    stride: int,
    accept_gaps: bool,
) -> WindowSelection:
    """Select the window's images from the foundation's sequence, or refuse the window.

    Args:
        window: The window record.
        sequence: The foundation's ``SequenceArtifact`` directory.
        sequence_artifact_id: The foundation's sequence.
        stride: One image in every ``stride`` is selected.
        accept_gaps: Whether a window outside the gap thresholds is explicitly accepted.

    Raises:
        ValueError: If the window was ingested into another sequence, is outside the gap
            thresholds without being accepted, is on a clock the sequence's images are not on,
            or holds no camera image.
    """
    if window.sequence_artifact_id != sequence_artifact_id:
        raise ValueError(
            f"the window {window.file} was ingested into sequence "
            f"{window.sequence_artifact_id!r}, but the foundation's sequence is "
            f"{sequence_artifact_id!r}"
        )
    if not window.within_thresholds and not accept_gaps:
        raise ValueError(
            f"the window {window.file} is outside its gap thresholds "
            f"({dict(window.document['quality'])}); pass --accept-window-gaps to run it anyway, "
            "which the protocol records"
        )
    images: list[tuple[str, int]] = []
    clocks: set[str] = set()
    for entry in SequenceArtifactReader(sequence).iter_index():
        if entry.modality == "image":
            clocks.add(entry.timestamp.clock_id)
            images.append((str(entry.observation_id), entry.timestamp.total_nanoseconds()))
    # Dois timestamps só se comparam no mesmo clock: a janela precisa estar no das imagens.
    if clocks - {window.clock_id}:
        raise ValueError(
            f"the window {window.file} is on clock {window.clock_id!r}, but the sequence's "
            f"images are on {sorted(clocks)}"
        )
    return every_nth_in_window(
        images, start_ns=window.start_ns, end_ns=window.end_ns, stride=stride
    )


def every_nth_in_window(
    images: Sequence[tuple[str, int]], *, start_ns: int, end_ns: int, stride: int
) -> WindowSelection:
    """Select every ``stride``-th image whose timestamp lies in ``[start_ns, end_ns)``.

    Args:
        images: ``(observation id, timestamp in ns)`` of every camera image, in index order.
        start_ns: First instant of the window, included.
        end_ns: End of the window, excluded.
        stride: One image in every ``stride`` is selected, from the first in the window.

    Raises:
        ValueError: If the stride is not positive or no image lies in the window.
    """
    if stride < 1:
        raise ValueError(f"the image stride must be positive, got {stride}")
    inside = [observation for observation, stamp in images if start_ns <= stamp < end_ns]
    if not inside:
        raise ValueError(f"the window [{start_ns}, {end_ns}) ns holds no camera image")
    return WindowSelection(
        window_images=len(inside), stride=stride, observation_ids=tuple(inside[::stride])
    )
