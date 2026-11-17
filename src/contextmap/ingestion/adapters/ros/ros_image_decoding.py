"""Decodificação de sensor_msgs/Image para ImageObservation canônica.

O módulo valida encoding e layout, remove padding de linha quando necessário, normaliza a ordem de
bytes suportada e registra essas conversões em provenance. Ele trata somente representação da
mensagem; sincronização, seleção e percepção visual permanecem fora desta borda.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion.adapters.ros.ros_message_decoding import (
    IMAGE_BYTES_PER_PIXEL,
    IMAGE_ENCODING_MAP,
)
from contextmap.ingestion.adapters.ros.ros_payload_normalization import (
    _remove_row_padding,
    _swap_fixed_width_values,
    _with_raw_metadata,
)
from contextmap.ingestion.canonical_observations import (
    CalibrationReferenceId,
    FrameId,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp


def decode_image(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
) -> ImageObservation:
    """Decode a ``sensor_msgs/Image`` into a canonical :class:`ImageObservation`.

    Args:
        message: A decoded ``sensor_msgs/Image`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical image observation.

    Raises:
        ValueError: If ``message.encoding`` is not a supported encoding.
    """
    encoding = IMAGE_ENCODING_MAP.get(message.encoding)
    if encoding is None:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")
    row_payload_size = int(message.width) * IMAGE_BYTES_PER_PIXEL[encoding]
    data = _remove_row_padding(
        message.data.tobytes(),
        height=int(message.height),
        row_step=int(message.step),
        row_payload_size=row_payload_size,
        payload_name="image",
    )
    source_is_bigendian = bool(message.is_bigendian)
    byte_order_normalized = source_is_bigendian and encoding is ImageEncoding.MONO16
    if byte_order_normalized:
        data = _swap_fixed_width_values(data, value_size=2)
    normalized_provenance = _with_raw_metadata(
        provenance,
        source_step_bytes=int(message.step),
        source_is_bigendian=source_is_bigendian,
        row_padding_removed=int(message.step) != row_payload_size,
        byte_order_normalized=byte_order_normalized,
        canonical_byte_order="little",
    )
    return ImageObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=normalized_provenance,
        calibration_id=calibration_id,
        width=message.width,
        height=message.height,
        encoding=encoding,
        data=data,
    )
