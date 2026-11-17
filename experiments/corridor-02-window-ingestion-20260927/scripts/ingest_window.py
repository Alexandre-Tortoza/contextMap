"""Ingest a time window of corridor-02 reproducibly, for the ablation campaign (issue #669).

The canonical corridor-02 ingestion (``experiments/e2e-real-ingestion-corridor-02-20260923/``)
ran ``IngestionService`` from an uncommitted driver, with a recording-time ``SourceWindow``, a
``ConstantOffsetCorrection`` derived from two anchors, a calibration built from the dataset's
YAML files and ``on_problems="warn"``. ``contextmap ingest`` cannot repeat it: it passes no
window nor timestamp policy and loads no external calibration. This command repeats that
ingestion for another window, and publishes, under one outputs root:

- the ``SequenceArtifact`` of the window of ``corridor-02.bag``;
- the auxiliary pose ``SequenceArtifact`` of ``corridor-02-gt.txt`` (``pose_file`` adapter), the
  one state estimation's pose bridge merges;
- ``window.json``: the resolved instants, both artifacts, the correction and the window quality.

What makes the new ingestion the same as the canonical one is read from the canonical
``SequenceArtifact`` itself, through the public ingestion API, and never retyped: its calibration
(exactly the one built from the YAML files, with the same content hashes), its topics,
synchronization, validation policy and header clock, and the two anchors its correction was
derived from, from which the correction is derived again by the same rule
(``ConstantOffsetCorrection.from_anchors``). Only the window changes.

The window is given in seconds from the first message of the bag (any topic), on the bag's
recording clock, plus a declared padding on both sides that is read too. The quality of the
window (the largest pose and LiDAR gaps inside it) is measured against the thresholds of the
canonical selection and reported; it never changes the window.

Usage:
    python ingest_window.py --bag PATH --pose-file PATH --canonical-sequence DIR
        --outputs-root DIR [--start-offset-s 150] [--end-offset-s 155] [--padding-s 5]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

from rosbags.rosbag1 import Reader

from contextmap.ingestion import (
    CalibrationSet,
    ConstantOffsetCorrection,
    SequenceArtifactReader,
    SourceAdapter,
    SourceAdapterConfig,
    SourceTopicMapping,
    SourceWindow,
    SynchronizationConfig,
    TimestampPolicy,
)
from contextmap.ingestion.adapters.pose_file_source_adapter import PoseFileSourceAdapter
from contextmap.ingestion.adapters.ros.ros1_bag_source_adapter import Ros1BagSourceAdapter
from contextmap.runtime import (
    ExecutionEvent,
    IngestionRequest,
    IngestionService,
    ValidationPolicy,
)
from contextmap.shared import SourceTimestamp

WINDOW_SCHEMA = "contextmap.corridor-window/v1"
"""Schema of ``window.json``, the contract with the campaign harness (#670)."""

WINDOW_RECORD = "window.json"

DEFAULT_START_OFFSET_S = 150.0
DEFAULT_END_OFFSET_S = 155.0
"""The campaign window, 2:30 to 2:35 after the first message of the bag."""

DEFAULT_PADDING_S = 5.0
"""Read on both sides of the window, as the canonical ingestion did (+/-5 s around its 90 s
window): a frame at the edge of the window keeps synchronization and pose context on both sides.
It must also exceed how far a sensor is recorded from its corrected time (on corridor-02 the IMU
is recorded about 0.14 s before it), or an observation of the window recorded outside the window
would not be read."""

POSE_GAP_THRESHOLD_MS = 605
LIDAR_GAP_THRESHOLD_MS = 250
"""The canonical selection's thresholds (``experiments/semantic-fusion-corridor-02-20260921/
selection.json``, ``rule``): max pose gap <= 605 ms and max LiDAR gap <= 250 ms."""

SEQUENCE_NAME = "corridor-02"
POSE_SEQUENCE_NAME = "corridor-02-pose"

POSE_FILE_SETTINGS: Mapping[str, object] = {
    "format": "tum",
    "parent_frame": "map",
    "body_frame": "epson",
    "pose_role": "ground_truth",
}
"""What ``corridor-02-gt.txt`` is, declared: the file states no frame and no role
(``ingestion/docs/adapters.md``, "Adapter pose_file")."""

EXIT_OK = 0
EXIT_FAILED = 1
"""An ingestion failed; a sequence already published stays, immutable, and no record is written."""
EXIT_REFUSED = 2
"""Refused before anything is read or written: invalid window, existing record, unusable
canonical sequence or empty bag."""

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000


class RequestRefused(ValueError):
    """The command's inputs cannot produce a trustworthy window; nothing was written."""


@dataclass(frozen=True, kw_only=True)
class WindowSpec:
    """The requested window, in seconds from the first message of the bag.

    Attributes:
        start_offset_s: Inclusive start.
        end_offset_s: Exclusive end.
        padding_s: Seconds read before the start and after the end.

    Raises:
        RequestRefused: If a value is not finite, the start is before the first message, the
            window is empty or inverted, or the padding is negative.
    """

    start_offset_s: float
    end_offset_s: float
    padding_s: float

    def __post_init__(self) -> None:
        """Refuse a window that selects nothing or that the bag cannot hold."""
        for name in ("start_offset_s", "end_offset_s", "padding_s"):
            if not math.isfinite(getattr(self, name)):
                raise RequestRefused(f"{name} must be finite, got {getattr(self, name)!r}")
        if self.start_offset_s < 0:
            raise RequestRefused(
                f"start_offset_s must not be before the first message, got {self.start_offset_s}"
            )
        if self.end_offset_s <= self.start_offset_s:
            raise RequestRefused(
                f"end_offset_s ({self.end_offset_s}) must come after start_offset_s "
                f"({self.start_offset_s}): the window must not be empty or inverted"
            )
        if self.padding_s < 0:
            raise RequestRefused(f"padding_s must not be negative, got {self.padding_s}")


@dataclass(frozen=True, kw_only=True)
class CanonicalIngestion:
    """What the canonical ingestion used, read back from its ``SequenceArtifact``.

    Attributes:
        artifact_id: The canonical sequence.
        topics: Its topic mapping.
        required_topics: Its required topics.
        timestamp_clock_id: Its declared header clock, which the pose sequence shares.
        synchronization: Its synchronization policy.
        validation: Its validation policy.
        hash_source: Whether it hashed what it read.
        timestamp_policy: Its policy, with the correction derived again from its anchors.
        calibration: Its calibration, exactly as persisted.
    """

    artifact_id: str
    topics: SourceTopicMapping
    required_topics: frozenset[str]
    timestamp_clock_id: str
    synchronization: SynchronizationConfig
    validation: ValidationPolicy
    hash_source: bool
    timestamp_policy: TimestampPolicy
    calibration: CalibrationSet

    @property
    def correction(self) -> ConstantOffsetCorrection:
        """The constant offset correction (always present: :meth:`read` refuses otherwise)."""
        assert self.timestamp_policy.correction is not None
        return self.timestamp_policy.correction

    @classmethod
    def read(cls, directory: Path) -> CanonicalIngestion:
        """Read the canonical ingestion's settings from its sequence artifact.

        Only the anchors of the correction are read, never its recorded offset: the provenance
        of the real canonical sequence predates #618 and records the offset in float seconds,
        while the anchors are exact, and the issue asks for the offset derived from them.

        Args:
            directory: The canonical ``SequenceArtifact``.

        Returns:
            Its settings.

        Raises:
            RequestRefused: If it is not a ROS 1 bag ingestion, or it has no calibration, no
                declared header clock or no anchored constant offset correction.
        """
        reader = SequenceArtifactReader(directory)
        artifact_id = str(reader.manifest.artifact_id)
        calibration = reader.read_calibration()
        if calibration is None:
            raise RequestRefused(f"canonical sequence {artifact_id} has no calibration")
        provenance = reader.read_provenance()
        if provenance is None or provenance.source_type != "ros1_bag":
            raise RequestRefused(
                f"canonical sequence {artifact_id} does not record a ros1_bag ingestion"
            )
        config: Mapping[str, Any] = provenance.ingestion_config
        try:
            clock = config["timestamp_clock_id"]
            if not clock:
                # Sem um relógio declarado, a sequência de pose não teria como compartilhar o
                # relógio do bag, e a ponte de pose recusaria a mesclagem.
                raise RequestRefused(
                    f"canonical sequence {artifact_id} declares no timestamp_clock_id"
                )
            return cls(
                artifact_id=artifact_id,
                topics=SourceTopicMapping(**config["topics"]),
                required_topics=frozenset(config["required_topics"]),
                timestamp_clock_id=str(clock),
                synchronization=SynchronizationConfig(**config["synchronization"]),
                validation=ValidationPolicy(**config["validation"]),
                hash_source=bool(config["hash_source"]),
                timestamp_policy=_policy_from_anchors(artifact_id, config["timestamp_policy"]),
                calibration=calibration,
            )
        except (KeyError, TypeError) as error:
            raise RequestRefused(
                f"canonical sequence {artifact_id} records an ingestion configuration of an "
                f"unexpected shape: {error!r}"
            ) from error


def _policy_from_anchors(artifact_id: str, document: Mapping[str, Any]) -> TimestampPolicy:
    """Rebuild the canonical timestamp policy, deriving the offset from its two anchors."""
    correction = document["correction"]
    source = correction.get("anchor_source_time")
    reference = correction.get("anchor_reference_time")
    if correction["type"] != "constant_offset" or source is None or reference is None:
        raise RequestRefused(
            f"canonical sequence {artifact_id} has no constant offset correction derived from "
            "anchors: there is nothing to derive the same correction from"
        )
    return TimestampPolicy(
        event_clock=document["event_clock"],
        window_clock=document["window_clock"],
        correction=ConstantOffsetCorrection.from_anchors(
            source_time=SourceTimestamp.from_record(source),
            reference_time=SourceTimestamp.from_record(reference),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class ResolvedWindow:
    """The window as absolute instants.

    Attributes:
        bag_first_message_recording_ns: Recording time of the bag's first message, any topic.
        start_recording_ns: Window start on the recording clock.
        end_recording_ns: Window end (exclusive) on the recording clock.
        start_raw_header_ns: Window start on the raw header clock, the clock of
            ``corridor-02-gt.txt`` and of the canonical selection.
        end_raw_header_ns: Window end (exclusive) on the raw header clock.
        start_ns: Window start on the corrected header clock, the observations' clock.
        end_ns: Window end (exclusive) on the corrected header clock.
        read_window: What is read from the bag: the window plus the padding, on the recording
            clock.
    """

    bag_first_message_recording_ns: int
    start_recording_ns: int
    end_recording_ns: int
    start_raw_header_ns: int
    end_raw_header_ns: int
    start_ns: int
    end_ns: int
    read_window: SourceWindow


def bag_first_message_recording_ns(bag: Path) -> int:
    """Return the recording time of the bag's first message, whatever its topic.

    Read from the bag's chunk index, without decoding any message. The ingestion API has no
    such query: its adapters only bound a window by the configured topics.

    Args:
        bag: A ROS 1 bag.

    Returns:
        Nanoseconds on the bag's recording clock.

    Raises:
        RequestRefused: If the bag holds no message.
    """
    with Reader(bag) as reader:
        if reader.message_count == 0:
            raise RequestRefused(f"{bag.name} holds no message")
        return int(reader.start_time)


def resolve_window(
    spec: WindowSpec,
    *,
    first_message_ns: int,
    correction: ConstantOffsetCorrection,
    window_clock_id: str,
) -> ResolvedWindow:
    """Turn offsets from the first message into instants on both clocks.

    The correction is ``event = header + offset`` with ``offset = recording - header`` at the
    anchor: the corrected header clock is the recording clock as the header clock tells it.
    The recording instant ``R`` is the raw header instant ``R - offset`` (the clock of the pose
    file and of the canonical selection), which the correction takes back to ``R``, so the window
    has the same numbers on the recording clock and on the clock of the ingested observations.
    Each sensor departs from that model by its own latency
    (``experiments/e2e-real-ingestion-corridor-02-20260923/``, Finding 1), which the padding
    covers.

    Args:
        spec: The requested window.
        first_message_ns: Recording time of the bag's first message.
        correction: The canonical correction.
        window_clock_id: The bag's own recording clock identity.

    Returns:
        The resolved window.
    """
    start_recording_ns = first_message_ns + round(spec.start_offset_s * _NS_PER_S)
    end_recording_ns = first_message_ns + round(spec.end_offset_s * _NS_PER_S)
    padding_ns = round(spec.padding_s * _NS_PER_S)
    start_raw_header_ns = start_recording_ns - correction.offset_nanoseconds
    end_raw_header_ns = end_recording_ns - correction.offset_nanoseconds
    return ResolvedWindow(
        bag_first_message_recording_ns=first_message_ns,
        start_recording_ns=start_recording_ns,
        end_recording_ns=end_recording_ns,
        start_raw_header_ns=start_raw_header_ns,
        end_raw_header_ns=end_raw_header_ns,
        start_ns=start_raw_header_ns + correction.offset_nanoseconds,
        end_ns=end_raw_header_ns + correction.offset_nanoseconds,
        read_window=SourceWindow(
            clock_id=window_clock_id,
            start_seconds=(start_recording_ns - padding_ns) / _NS_PER_S,
            end_seconds=(end_recording_ns + padding_ns) / _NS_PER_S,
        ),
    )


def bag_request(
    bag: Path,
    canonical: CanonicalIngestion,
    window: SourceWindow,
    *,
    output_dir: Path,
    artifact_id: str,
) -> IngestionRequest:
    """The canonical bag request, with this window."""
    return IngestionRequest(
        source_type="ros1_bag",
        source_path=str(bag),
        sequence_name=SEQUENCE_NAME,
        output_dir=str(output_dir),
        topics=canonical.topics,
        synchronization=canonical.synchronization,
        required_topics=canonical.required_topics,
        timestamp_clock_id=canonical.timestamp_clock_id,
        timestamp_policy=canonical.timestamp_policy,
        window=window,
        calibration=canonical.calibration,
        validation=canonical.validation,
        hash_source=canonical.hash_source,
        artifact_id=artifact_id,
    )


def pose_request(
    pose_file: Path, canonical: CanonicalIngestion, *, output_dir: Path, artifact_id: str
) -> IngestionRequest:
    """The auxiliary pose request: the whole file, on the bag's clock and correction.

    The ``pose_file`` adapter refuses a window (it always reads the whole file), so the pose
    sequence is not windowed; state estimation merges it whole anyway. It carries the canonical
    calibration, so its body frame is checked against the same frames.
    """
    return IngestionRequest(
        source_type="pose_file",
        source_path=str(pose_file),
        sequence_name=POSE_SEQUENCE_NAME,
        output_dir=str(output_dir),
        # O adapter de arquivo não lê tópicos; o serviço exige um tópico para a modalidade.
        topics=SourceTopicMapping(pose=pose_file.name),
        synchronization=SynchronizationConfig(
            reference_modality="external_pose",
            tolerance_nanoseconds=canonical.synchronization.tolerance_nanoseconds,
        ),
        required_topics=frozenset({"pose"}),
        timestamp_clock_id=canonical.timestamp_clock_id,
        timestamp_policy=canonical.timestamp_policy,
        calibration=canonical.calibration,
        validation=canonical.validation,
        hash_source=canonical.hash_source,
        artifact_id=artifact_id,
        extra=POSE_FILE_SETTINGS,
    )


def build_adapter(config: SourceAdapterConfig) -> SourceAdapter:
    """Build the adapter of a request: this command is the composition root of its two sources.

    Raises:
        ValueError: For any other adapter family; none is ever substituted.
    """
    if config.source_type == "ros1_bag":
        return Ros1BagSourceAdapter(config)
    if config.source_type == "pose_file":
        return PoseFileSourceAdapter(config)
    raise ValueError(f"this command ingests ros1_bag and pose_file, not {config.source_type!r}")


def max_gap_ms(timestamps_ns: Iterable[int], *, start_ns: int, end_ns: int) -> float | None:
    """The largest interval between consecutive samples inside ``[start_ns, end_ns)``.

    The canonical selection's definition: only samples inside the window count, so the
    intervals to the window's edges do not.

    Args:
        timestamps_ns: Sample timestamps, in any order.
        start_ns: Inclusive window start.
        end_ns: Exclusive window end.

    Returns:
        Milliseconds, or ``None`` when fewer than two samples fall inside the window.
    """
    inside = sorted(stamp for stamp in timestamps_ns if start_ns <= stamp < end_ns)
    if len(inside) < 2:
        return None
    return max(later - earlier for earlier, later in pairwise(inside)) / _NS_PER_MS


def window_quality(
    sequence: Path, pose_sequence: Path, *, start_ns: int, end_ns: int
) -> dict[str, object]:
    """Measure the published window against the canonical selection's thresholds.

    Args:
        sequence: The published bag sequence (its LiDAR scans).
        pose_sequence: The published pose sequence.
        start_ns: Window start on the observations' clock.
        end_ns: Window end (exclusive) on the observations' clock.

    Returns:
        The ``quality`` block of ``window.json``. A gap without two samples is ``None`` and the
        window is then not within the thresholds.
    """
    pose_gap = max_gap_ms(
        _index_timestamps(pose_sequence, "external_pose"), start_ns=start_ns, end_ns=end_ns
    )
    lidar_gap = max_gap_ms(_index_timestamps(sequence, "lidar"), start_ns=start_ns, end_ns=end_ns)
    within = (
        pose_gap is not None
        and lidar_gap is not None
        and pose_gap <= POSE_GAP_THRESHOLD_MS
        and lidar_gap <= LIDAR_GAP_THRESHOLD_MS
    )
    return {
        "max_pose_gap_ms": pose_gap,
        "max_lidar_gap_ms": lidar_gap,
        "pose_gap_threshold_ms": POSE_GAP_THRESHOLD_MS,
        "lidar_gap_threshold_ms": LIDAR_GAP_THRESHOLD_MS,
        "within_thresholds": within,
    }


def _index_timestamps(sequence: Path, modality: str) -> list[int]:
    """The timestamps of one modality, from the index alone (no payload is read)."""
    return [
        entry.timestamp.total_nanoseconds()
        for entry in SequenceArtifactReader(sequence).iter_index()
        if entry.modality == modality
    ]


def write_record(path: Path, record: Mapping[str, object]) -> None:
    """Publish ``window.json`` whole or not at all, and never over an existing one.

    The document reaches the disk under a temporary name first; the hard link then publishes it
    and fails if the name already exists.
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class _PhasePrinter:
    """Prints each phase of one ingestion on stderr, as ``contextmap ingest`` does."""

    def __init__(self, sequence_name: str) -> None:
        """Print under this sequence name."""
        self._sequence_name = sequence_name

    def emit(self, event: ExecutionEvent) -> None:
        """Print the event kind, except the periodic progress."""
        if event.kind != "ingestion.progress":
            print(f"{self._sequence_name}: {event.kind}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bag", type=Path, required=True, help="corridor-02.bag")
    parser.add_argument("--pose-file", type=Path, required=True, help="corridor-02-gt.txt")
    parser.add_argument(
        "--canonical-sequence",
        type=Path,
        required=True,
        help="the canonical SequenceArtifact whose calibration, topics and anchors are reused",
    )
    parser.add_argument(
        "--outputs-root", type=Path, required=True, help="where the sequences and window.json go"
    )
    parser.add_argument("--start-offset-s", type=float, default=DEFAULT_START_OFFSET_S)
    parser.add_argument("--end-offset-s", type=float, default=DEFAULT_END_OFFSET_S)
    parser.add_argument("--padding-s", type=float, default=DEFAULT_PADDING_S)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_FAILED` or :data:`EXIT_REFUSED`.
    """
    arguments = build_parser().parse_args(argv)
    outputs: Path = arguments.outputs_root
    record_path = outputs / WINDOW_RECORD
    try:
        spec = WindowSpec(
            start_offset_s=arguments.start_offset_s,
            end_offset_s=arguments.end_offset_s,
            padding_s=arguments.padding_s,
        )
        if record_path.exists():
            raise RequestRefused(f"{record_path} already exists: a window is never re-recorded")
        canonical = CanonicalIngestion.read(arguments.canonical_sequence)
        window_clock_id = SourceAdapterConfig(
            source_type="ros1_bag", path=str(arguments.bag), topics=canonical.topics
        ).resolved_window_clock_id()
        window = resolve_window(
            spec,
            first_message_ns=bag_first_message_recording_ns(arguments.bag),
            correction=canonical.correction,
            window_clock_id=window_clock_id,
        )
    except RequestRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED

    sequence_id, pose_sequence_id = uuid4().hex, uuid4().hex
    sequence_location = Path("sequences", SEQUENCE_NAME, sequence_id)
    pose_sequence_location = Path("sequences", POSE_SEQUENCE_NAME, pose_sequence_id)
    requests = (
        bag_request(
            arguments.bag,
            canonical,
            window.read_window,
            output_dir=outputs / sequence_location,
            artifact_id=sequence_id,
        ),
        pose_request(
            arguments.pose_file,
            canonical,
            output_dir=outputs / pose_sequence_location,
            artifact_id=pose_sequence_id,
        ),
    )
    service = IngestionService(build_adapter)
    # As duas pré-checagens antes de qualquer leitura: um pedido inviável não publica nada.
    for request in requests:
        preflight = service.preflight(request)
        if not preflight.ok:
            for problem in preflight.problems:
                print(f"{request.sequence_name}: {problem}", file=sys.stderr)
            return EXIT_FAILED
    for request in requests:
        result = service.run(request, event_sink=_PhasePrinter(request.sequence_name))
        if result.failure is not None or result.status != "completed":
            print(f"{request.sequence_name} {result.status}: {result.failure}", file=sys.stderr)
            return EXIT_FAILED

    quality = window_quality(
        outputs / sequence_location,
        outputs / pose_sequence_location,
        start_ns=window.start_ns,
        end_ns=window.end_ns,
    )
    correction = canonical.correction
    assert correction.anchor_source_time is not None
    assert correction.anchor_reference_time is not None
    record: dict[str, object] = {
        "schema": WINDOW_SCHEMA,
        "bag": arguments.bag.name,
        "bag_first_message_recording_ns": window.bag_first_message_recording_ns,
        "start_offset_s": spec.start_offset_s,
        "end_offset_s": spec.end_offset_s,
        "padding_s": spec.padding_s,
        "clock_id": canonical.timestamp_clock_id,
        "start_ns": window.start_ns,
        "end_ns": window.end_ns,
        "start_recording_ns": window.start_recording_ns,
        "end_recording_ns": window.end_recording_ns,
        "start_raw_header_ns": window.start_raw_header_ns,
        "end_raw_header_ns": window.end_raw_header_ns,
        "sequence_artifact_id": sequence_id,
        "sequence_location": sequence_location.as_posix(),
        "pose_sequence_artifact_id": pose_sequence_id,
        "pose_sequence_location": pose_sequence_location.as_posix(),
        "canonical_sequence_artifact_id": canonical.artifact_id,
        "offset_correction": {
            "offset_ns": correction.offset_nanoseconds,
            "anchors": {
                "source_time": correction.anchor_source_time.to_record(),
                "reference_time": correction.anchor_reference_time.to_record(),
            },
        },
        "quality": quality,
    }
    write_record(record_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not quality["within_thresholds"]:
        print(
            "warning: the window is outside the canonical selection thresholds "
            f"(pose <= {POSE_GAP_THRESHOLD_MS} ms, LiDAR <= {LIDAR_GAP_THRESHOLD_MS} ms); "
            "it was kept as requested",
            file=sys.stderr,
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
