"""Decodificação de sensor_msgs/Imu para ImuObservation canônica.

O decoder preserva aceleração, velocidade angular, orientação e covariâncias quando disponíveis,
respeitando a convenção ROS de campo ausente. Nenhuma filtragem, integração ou estimativa de estado
é realizada durante ingestion.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion.canonical_observations import (
    CalibrationReferenceId,
    FrameId,
    ImuObservation,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp


def _available_covariance(values: Any) -> tuple[float, ...] | None:
    """Return a ROS covariance unless its first value marks data unavailable."""
    if values[0] == -1.0:
        return None
    return tuple(float(value) for value in values)


def decode_imu(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
) -> ImuObservation:
    """Decode a ``sensor_msgs/Imu`` into a canonical :class:`ImuObservation`.

    Each of ``linear_acceleration``/``angular_velocity``/``orientation`` is
    ``None`` when its own covariance's first element is ``-1.0`` — the
    ``sensor_msgs/Imu`` convention for "not provided" — checked
    independently per field.

    Args:
        message: A decoded ``sensor_msgs/Imu`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical IMU observation.
    """
    linear_acceleration = (
        None
        if message.linear_acceleration_covariance[0] == -1.0
        else (
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        )
    )
    angular_velocity = (
        None
        if message.angular_velocity_covariance[0] == -1.0
        else (
            float(message.angular_velocity.x),
            float(message.angular_velocity.y),
            float(message.angular_velocity.z),
        )
    )
    orientation = (
        None
        if message.orientation_covariance[0] == -1.0
        else (
            float(message.orientation.x),
            float(message.orientation.y),
            float(message.orientation.z),
            float(message.orientation.w),
        )
    )
    return ImuObservation(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.header.frame_id),
        timestamp=timestamp,
        provenance=provenance,
        calibration_id=calibration_id,
        linear_acceleration=linear_acceleration,
        angular_velocity=angular_velocity,
        orientation=orientation,
        linear_acceleration_covariance=_available_covariance(
            message.linear_acceleration_covariance
        ),
        angular_velocity_covariance=_available_covariance(message.angular_velocity_covariance),
        orientation_covariance=_available_covariance(message.orientation_covariance),
    )
