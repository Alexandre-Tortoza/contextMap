"""Decodificação de sensor_msgs/PointCloud2 para LidarObservation canônica.

O módulo preserva o layout dos PointField, valida strides e payload, remove padding de linha e
normaliza byte order quando necessário. Os bytes e descritores resultantes representam a medição
de fonte; voxelização, geometria e interpretação 3D pertencem a etapas downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contextmap.ingestion.adapters.ros.ros_payload_normalization import (
    _remove_row_padding,
    _with_raw_metadata,
)
from contextmap.ingestion.canonical_observations import (
    CalibrationReferenceId,
    FrameId,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp

POINTFIELD_TYPE_MAP: dict[int, PointFieldDataType] = {
    1: PointFieldDataType.INT8,
    2: PointFieldDataType.UINT8,
    3: PointFieldDataType.INT16,
    4: PointFieldDataType.UINT16,
    5: PointFieldDataType.INT32,
    6: PointFieldDataType.UINT32,
    7: PointFieldDataType.FLOAT32,
    8: PointFieldDataType.FLOAT64,
}
POINTFIELD_SIZE_BYTES: dict[PointFieldDataType, int] = {
    PointFieldDataType.INT8: 1,
    PointFieldDataType.UINT8: 1,
    PointFieldDataType.INT16: 2,
    PointFieldDataType.UINT16: 2,
    PointFieldDataType.INT32: 4,
    PointFieldDataType.UINT32: 4,
    PointFieldDataType.FLOAT32: 4,
    PointFieldDataType.FLOAT64: 8,
}


def decode_lidar(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
) -> LidarObservation:
    """Decode a ``sensor_msgs/PointCloud2`` into a canonical :class:`LidarObservation`.

    Args:
        message: A decoded ``sensor_msgs/PointCloud2`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical LiDAR observation.

    Raises:
        ValueError: If a ``PointField.datatype`` is outside the supported
            ``sensor_msgs/PointField`` datatypes.
    """
    fields = tuple(
        PointFieldDescriptor(
            name=field.name,
            offset_bytes=field.offset,
            data_type=_point_field_data_type(field),
            count=field.count,
        )
        for field in message.fields
    )
    row_payload_size = int(message.width) * int(message.point_step)
    data = _remove_row_padding(
        message.data.tobytes(),
        height=int(message.height),
        row_step=int(message.row_step),
        row_payload_size=row_payload_size,
        payload_name="point cloud",
    )
    source_is_bigendian = bool(message.is_bigendian)
    if source_is_bigendian:
        data = _normalize_point_field_byte_order(
            data,
            point_step=int(message.point_step),
            fields=fields,
        )
    normalized_provenance = _with_raw_metadata(
        provenance,
        source_row_step_bytes=int(message.row_step),
        source_is_bigendian=source_is_bigendian,
        row_padding_removed=int(message.row_step) != row_payload_size,
        byte_order_normalized=source_is_bigendian,
        canonical_byte_order="little",
    )
    return LidarObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=normalized_provenance,
        calibration_id=calibration_id,
        point_count=message.width * message.height,
        point_step_bytes=message.point_step,
        fields=fields,
        is_dense=bool(message.is_dense),
        data=data,
    )


def _point_field_data_type(field: Any) -> PointFieldDataType:
    """Map a ``sensor_msgs/PointField`` datatype code to its canonical data type.

    Args:
        field: A decoded ``sensor_msgs/PointField`` (has ``.name``/``.datatype``).

    Returns:
        The canonical point field data type.

    Raises:
        ValueError: If ``field.datatype`` is not one of the supported codes;
            the message names the field, the received code and the supported
            range, so the resulting adapter warning is actionable.
    """
    data_type = POINTFIELD_TYPE_MAP.get(field.datatype)
    if data_type is None:
        raise ValueError(
            f"unsupported PointField datatype {field.datatype!r} for field {field.name!r}; "
            f"supported datatypes are {min(POINTFIELD_TYPE_MAP)}-{max(POINTFIELD_TYPE_MAP)}"
        )
    return data_type


def _available_covariance(values: Any) -> tuple[float, ...] | None:
    """Return a ROS covariance unless its first value marks data unavailable."""
    if values[0] == -1.0:
        return None
    return tuple(float(value) for value in values)


def _normalize_point_field_byte_order(
    data: bytes,
    *,
    point_step: int,
    fields: tuple[PointFieldDescriptor, ...],
) -> bytes:
    """Convert declared multibyte point fields from big- to little-endian.

    ``data`` is tightly packed (a whole number of ``point_step`` records, as
    :func:`_remove_row_padding` guarantees). Each field is swapped for every
    point at once, as a ``(points, count, value_size)`` byte view reversed on
    its last axis, with the same bytes as a per-point, per-element loop.
    """
    points = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step).copy()
    for field in fields:
        value_size = POINTFIELD_SIZE_BYTES[field.data_type]
        field_end = field.offset_bytes + value_size * field.count
        if field.offset_bytes < 0 or field_end > point_step:
            raise ValueError(f"point field {field.name!r} exceeds point_step={point_step}")
        if value_size == 1:
            continue
        values = points[:, field.offset_bytes : field_end].reshape(-1, field.count, value_size)
        points[:, field.offset_bytes : field_end] = values[:, :, ::-1].reshape(
            -1, field.count * value_size
        )
    return points.tobytes()
