"""Decodificação de nav_msgs/Odometry para ExternalPoseMeasurement.

O módulo preserva frames, pose, velocidades, covariâncias, timestamp e provenance no contrato
canônico. A saída continua sendo uma medição externa de entrada; estimar PoseEstimate ou trajetória
é responsabilidade de state_estimation.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion.canonical_observations import (
    CalibrationReferenceId,
    ExternalPoseMeasurement,
    FrameId,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.shared import SourceTimestamp


def decode_pose(
    message: Any,
    *,
    observation_id: SourceObservationId,
    sensor_id: SensorId,
    timestamp: SourceTimestamp,
    provenance: SourceProvenance,
    calibration_id: CalibrationReferenceId | None = None,
) -> ExternalPoseMeasurement:
    """Decode a ``nav_msgs/Odometry`` into a canonical :class:`ExternalPoseMeasurement`.

    ``header.frame_id``/``child_frame_id`` map to
    ``parent_frame``/``frame_id``, matching the ownership rule that an
    externally supplied pose is an input measurement, never a State
    Estimation ``PoseEstimate``.

    Args:
        message: A decoded ``nav_msgs/Odometry`` message.
        observation_id: Identity to assign to the resulting observation.
        sensor_id: Sensor identity to assign.
        timestamp: Canonical timestamp to assign.
        provenance: Provenance to assign.
        calibration_id: Applicable canonical calibration reference, when available.

    Returns:
        The canonical external pose measurement.
    """
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    linear_velocity = message.twist.twist.linear
    angular_velocity = message.twist.twist.angular
    return ExternalPoseMeasurement(
        observation_id=observation_id,
        sensor_id=sensor_id,
        frame_id=FrameId(message.child_frame_id),
        timestamp=timestamp,
        provenance=provenance,
        calibration_id=calibration_id,
        parent_frame=FrameId(message.header.frame_id),
        translation=(float(position.x), float(position.y), float(position.z)),
        orientation=(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ),
        pose_covariance=tuple(float(value) for value in message.pose.covariance),
        linear_velocity=(
            float(linear_velocity.x),
            float(linear_velocity.y),
            float(linear_velocity.z),
        ),
        angular_velocity=(
            float(angular_velocity.x),
            float(angular_velocity.y),
            float(angular_velocity.z),
        ),
        twist_covariance=tuple(float(value) for value in message.twist.covariance),
    )
