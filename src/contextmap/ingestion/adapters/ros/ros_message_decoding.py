"""Operações comuns de desserialização e timestamp para mensagens ROS.

Este módulo encapsula diferenças mecânicas da serialização, transforma payloads truncados em erros
de conteúdo legíveis e converte header.stamp para SourceTimestamp. Ele não decide modalidade,
sincronização nem política de correção temporal.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any

from contextmap.ingestion.canonical_observations import ImageEncoding
from contextmap.shared import SourceTimestamp

IMAGE_ENCODING_MAP: dict[str, ImageEncoding] = {
    "rgb8": ImageEncoding.RGB8,
    "bgr8": ImageEncoding.BGR8,
    "mono8": ImageEncoding.MONO8,
    "mono16": ImageEncoding.MONO16,
}
IMAGE_BYTES_PER_PIXEL: dict[ImageEncoding, int] = {
    ImageEncoding.RGB8: 3,
    ImageEncoding.BGR8: 3,
    ImageEncoding.MONO8: 1,
    ImageEncoding.MONO16: 2,
}


def sanitize_topic(topic: str) -> str:
    """Turn a ROS topic name into an identifier usable as a sensor/observation ID.

    Args:
        topic: A ROS topic name, e.g. ``"/camera/image_raw"``.

    Returns:
        The topic with leading/trailing slashes stripped and internal
        slashes replaced by underscores, e.g. ``"camera_image_raw"``.
    """
    return topic.strip("/").replace("/", "_")


def deserialize_payload(
    deserialize: Callable[[bytes, str], Any],
    rawdata: bytes,
    msgtype: str,
    *,
    bag_timestamp_nanoseconds: int,
    header_bytes: int = 0,
) -> Any:
    """Deserialize one raw bag message, refusing a truncated payload as malformed content.

    ``rosbags`` unpacks each field with ``struct.unpack_from`` and raises
    ``struct.error`` when the payload ends before the message layout does (a
    truncated recording or a corrupted length prefix). That is malformed
    input, the same category as an unsupported image encoding, so it is
    translated here into ``ValueError``, the adapters' contractual error for
    a message whose content cannot be decoded. Only the
    deserialization call is guarded: a ``struct.error`` raised later by the
    decoders of this module is a programming error and propagates unchanged.

    Args:
        deserialize: The typestore deserializer of the bag's format.
        rawdata: The message's raw, still-serialized bytes.
        msgtype: The ROS message type of the message's connection.
        bag_timestamp_nanoseconds: The message's recording timestamp, reported
            in the error so the message can be found in the bag.
        header_bytes: Bytes the serialization format puts before the message
            itself (the CDR encapsulation header of ROS 2), checked before
            deserializing because a shorter payload has no layout to read.

    Returns:
        The deserialized message.

    Raises:
        ValueError: If the payload is shorter than ``header_bytes`` or than
            the layout of ``msgtype``.
    """
    if len(rawdata) < header_bytes:
        raise ValueError(
            f"{_truncated_payload(msgtype, rawdata, bag_timestamp_nanoseconds)} is shorter than "
            f"the {header_bytes}-byte header of its serialization format"
        )
    try:
        return deserialize(rawdata, msgtype)
    except struct.error as error:
        raise ValueError(
            f"{_truncated_payload(msgtype, rawdata, bag_timestamp_nanoseconds)} is shorter than "
            f"its serialized layout: {error}"
        ) from error


def _truncated_payload(msgtype: str, rawdata: bytes, bag_timestamp_nanoseconds: int) -> str:
    """Describe a truncated serialized message with enough source context to locate it.

    Args:
        msgtype: ROS message type expected by the deserializer.
        rawdata: Serialized bytes that were shorter than the expected layout.
        bag_timestamp_nanoseconds: Recording timestamp of the affected message.

    Returns:
        Human-readable description used by the adapter warning.
    """
    return (
        f"truncated {msgtype} payload of {len(rawdata)} bytes "
        f"at bag timestamp {bag_timestamp_nanoseconds} ns"
    )


def stamp_to_timestamp(stamp: Any, *, clock_id: str) -> SourceTimestamp:
    """Convert a ROS ``builtin_interfaces/Time``-shaped stamp into a canonical timestamp.

    Args:
        stamp: A decoded message's ``header.stamp`` (has ``.sec``/``.nanosec``).
        clock_id: Clock domain identity to attach, e.g. ``"ros1_bag:/imu/data"``.

    Returns:
        The canonical timestamp.
    """
    return SourceTimestamp(seconds=stamp.sec, nanoseconds=stamp.nanosec, clock_id=clock_id)
