"""Resolução de SourceWindow sobre o clock de gravação de bags ROS.

O módulo valida domínio de clock, sobreposição com a gravação e converte limites temporais para os
parâmetros usados pelo reader. O clock de gravação é tratado separadamente do header.stamp dos
sensores para impedir comparações implícitas entre domínios diferentes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contextmap.ingestion.recorded_source_adapter import (
    InvalidSourceWindowError,
    SourceAdapterConfig,
)


def resolve_window_bounds(
    config: SourceAdapterConfig,
    *,
    reader: Any,
    topic_kinds: Mapping[str, str],
) -> tuple[int | None, int | None]:
    """Resolve a configured window into nanosecond bounds for the bag reader's own time filter.

    Both ``rosbags.rosbag1.Reader.messages`` and ``rosbags.rosbag2.Reader.messages``
    accept ``start``/``stop`` (nanoseconds, the bag's own per-message
    recording time) and skip the chunk data of any message outside that
    range without decompressing or decoding it — the mechanism that makes
    windowed ingestion cost proportional to the window, not the source.

    Args:
        config: The adapter configuration; ``config.window`` may be ``None``.
        reader: An open ``rosbags`` reader of the configured source. Only
            its connections and index are read, never a message chunk, so
            the caller can reuse the same reader for its other checks.
        topic_kinds: Configured topic name -> modality kind, as built by
            each adapter's own ``_configured_topic_kinds()``.

    Returns:
        ``(start_nanoseconds, stop_nanoseconds)``, both ``None`` when no
        window is configured.

    Raises:
        InvalidSourceWindowError: If ``config.window.clock_id`` does not
            match ``config.resolved_window_clock_id()``, or if the window
            does not overlap the source's actual recording-time range at
            all (never silently truncated to an empty read).
    """
    window = config.window
    if window is None:
        return None, None

    expected_clock_id = config.resolved_window_clock_id()
    if window.clock_id != expected_clock_id:
        raise InvalidSourceWindowError(
            f"window.clock_id {window.clock_id!r} does not match this source's own "
            f"recording-time clock {expected_clock_id!r}; a window is always expressed "
            "in the source's own recording time, never the header clock"
        )

    start_ns = round(window.start_seconds * 1_000_000_000)
    stop_ns = round(window.end_seconds * 1_000_000_000)

    bounds = _recording_time_bounds(reader, topic_kinds)
    if bounds is None:
        return start_ns, stop_ns
    source_min_ns, source_max_ns = bounds
    if stop_ns <= source_min_ns or start_ns > source_max_ns:
        raise InvalidSourceWindowError(
            f"window [{window.start_seconds}, {window.end_seconds}) does not overlap "
            f"the source's recording-time range [{source_min_ns / 1e9}, {source_max_ns / 1e9}] "
            "seconds for the configured topics"
        )
    return start_ns, stop_ns


def _recording_time_bounds(reader: Any, topic_kinds: Mapping[str, str]) -> tuple[int, int] | None:
    """Return ``(min, max)`` recording-time nanoseconds for the configured topics, cheaply.

    Reading either bound never decompresses a message chunk: a ROS 1
    reader already carries a per-connection index (``entry.time``) built
    from the bag's own index records; a ROS 2 reader instead exposes only
    a bag-wide ``start_time``/``end_time`` (from the storage's metadata,
    not per-topic), which is used as a safe, if less precise, superset —
    a window that does not overlap the *whole* bag cannot overlap one of
    its topics either.

    Args:
        reader: An open reader of the configured source.
        topic_kinds: Configured topic name -> modality kind.

    Returns:
        The bounds, or ``None`` when the source has no messages on any
        configured topic (ROS 1) or no messages at all (ROS 2).
    """
    indexes = getattr(reader, "indexes", None)
    if indexes is not None:
        connections = [
            connection for connection in reader.connections if connection.topic in topic_kinds
        ]
        recording_times = [
            entry.time for connection in connections for entry in indexes[connection.id]
        ]
        return (min(recording_times), max(recording_times)) if recording_times else None

    if reader.message_count == 0:
        return None
    return reader.start_time, reader.end_time
