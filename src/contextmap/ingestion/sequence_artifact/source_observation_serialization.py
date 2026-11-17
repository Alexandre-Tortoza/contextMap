"""Serialização das SourceObservation entre memória, índice e payloads do artifact.

O módulo preserva os campos canônicos de cada modalidade, mantendo imagem e LiDAR em payloads
binários e registros menores inline no índice. A desserialização valida caminhos antes de acessar
arquivos e não depende de mensagens ROS ou da fonte original.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from contextmap.ingestion.canonical_observations import (
    CalibrationReferenceId,
    ExternalPoseMeasurement,
    FrameId,
    ImageEncoding,
    ImageObservation,
    ImuObservation,
    LidarObservation,
    PointFieldDataType,
    PointFieldDescriptor,
    SensorId,
    SourceObservation,
    SourceObservationId,
    SourceProvenance,
    observation_modality,
)
from contextmap.ingestion.sequence_artifact.sequence_artifact_manifest import SequenceArtifactError
from contextmap.shared import SourceTimestamp, is_run_relative_path


def _encode_observation(
    observation: SourceObservation,
) -> tuple[dict[str, Any], tuple[str, bytes] | None]:
    """Encode one canonical observation into index metadata and an optional binary payload.

    Args:
        observation: Canonical source observation to persist.

    Returns:
        A JSON-compatible index record plus optional relative payload path and bytes.

    Raises:
        SequenceArtifactError: If the observation variant is not supported by the schema.
    """
    record: dict[str, Any] = {
        "observation_id": str(observation.observation_id),
        "sensor_id": str(observation.sensor_id),
        "frame_id": str(observation.frame_id),
        "timestamp": {
            "seconds": observation.timestamp.seconds,
            "nanoseconds": observation.timestamp.nanoseconds,
            "clock_id": observation.timestamp.clock_id,
        },
        "provenance": {
            "source_type": observation.provenance.source_type,
            "source_path": observation.provenance.source_path,
            "source_topic": observation.provenance.source_topic,
            "source_message_index": observation.provenance.source_message_index,
            "raw_metadata": dict(observation.provenance.raw_metadata),
        },
        "calibration_id": (
            str(observation.calibration_id) if observation.calibration_id is not None else None
        ),
        "modality": observation_modality(observation),
    }

    if isinstance(observation, ImageObservation):
        record["width"] = observation.width
        record["height"] = observation.height
        record["encoding"] = observation.encoding.value
        payload_path = f"rgb/{observation.observation_id}.bin"
        record["payload_path"] = payload_path
        return record, (payload_path, observation.data)

    if isinstance(observation, LidarObservation):
        record["point_count"] = observation.point_count
        record["point_step_bytes"] = observation.point_step_bytes
        record["is_dense"] = observation.is_dense
        record["fields"] = [
            {
                "name": item.name,
                "offset_bytes": item.offset_bytes,
                "data_type": item.data_type.value,
                "count": item.count,
            }
            for item in observation.fields
        ]
        payload_path = f"pointcloud/{observation.observation_id}.bin"
        record["payload_path"] = payload_path
        return record, (payload_path, observation.data)

    if isinstance(observation, ImuObservation):
        record["linear_acceleration"] = (
            list(observation.linear_acceleration)
            if observation.linear_acceleration is not None
            else None
        )
        record["angular_velocity"] = (
            list(observation.angular_velocity) if observation.angular_velocity is not None else None
        )
        record["orientation"] = (
            list(observation.orientation) if observation.orientation is not None else None
        )
        record["linear_acceleration_covariance"] = _encode_optional_tuple(
            observation.linear_acceleration_covariance
        )
        record["angular_velocity_covariance"] = _encode_optional_tuple(
            observation.angular_velocity_covariance
        )
        record["orientation_covariance"] = _encode_optional_tuple(
            observation.orientation_covariance
        )
        return record, None

    if isinstance(observation, ExternalPoseMeasurement):
        record["parent_frame"] = str(observation.parent_frame)
        record["translation"] = list(observation.translation)
        record["orientation"] = list(observation.orientation)
        record["pose_covariance"] = _encode_optional_tuple(observation.pose_covariance)
        record["linear_velocity"] = _encode_optional_tuple(observation.linear_velocity)
        record["angular_velocity"] = _encode_optional_tuple(observation.angular_velocity)
        record["twist_covariance"] = _encode_optional_tuple(observation.twist_covariance)
        return record, None

    raise SequenceArtifactError(f"unsupported observation type: {type(observation)!r}")


def _payload_file(root: Path, record: Mapping[str, Any]) -> Path:
    """Return the payload file an index record names, refusing one outside the artifact.

    Raises:
        SequenceArtifactError: If ``payload_path`` is absolute, has a ``..`` part or is empty.
    """
    payload_path = record["payload_path"]
    if not isinstance(payload_path, str) or not is_run_relative_path(payload_path):
        raise SequenceArtifactError(
            f"observation {record['observation_id']!r} has payload_path {payload_path!r}, "
            "which is not a relative path inside the artifact"
        )
    return root / payload_path


def _decode_observation(
    record: dict[str, Any], root: Path, *, load_payload: bool = True
) -> SourceObservation:
    """Decode one persisted index record into its canonical observation type.

    Args:
        record: Parsed index.jsonl record.
        root: Artifact root used to resolve validated payload paths.
        load_payload: Whether image/LiDAR payload bytes should be opened and loaded.

    Returns:
        Reconstructed canonical source observation.

    Raises:
        SequenceArtifactError: If the modality or payload path is invalid.
    """
    common: dict[str, Any] = {
        "observation_id": SourceObservationId(record["observation_id"]),
        "sensor_id": SensorId(record["sensor_id"]),
        "frame_id": FrameId(record["frame_id"]),
        "timestamp": SourceTimestamp(**record["timestamp"]),
        "provenance": SourceProvenance(**record["provenance"]),
        "calibration_id": (
            CalibrationReferenceId(record["calibration_id"])
            if record["calibration_id"] is not None
            else None
        ),
    }

    modality = record["modality"]

    if modality == "image":
        data = _payload_file(root, record).read_bytes() if load_payload else b""
        return ImageObservation(
            **common,
            width=record["width"],
            height=record["height"],
            encoding=ImageEncoding(record["encoding"]),
            data=data,
        )

    if modality == "lidar":
        data = _payload_file(root, record).read_bytes() if load_payload else b""
        fields = tuple(
            PointFieldDescriptor(
                name=item["name"],
                offset_bytes=item["offset_bytes"],
                data_type=PointFieldDataType(item["data_type"]),
                count=item["count"],
            )
            for item in record["fields"]
        )
        return LidarObservation(
            **common,
            point_count=record["point_count"],
            point_step_bytes=record["point_step_bytes"],
            fields=fields,
            is_dense=record["is_dense"],
            data=data,
        )

    if modality == "imu":
        return ImuObservation(
            **common,
            linear_acceleration=(
                tuple(record["linear_acceleration"])
                if record["linear_acceleration"] is not None
                else None
            ),
            angular_velocity=(
                tuple(record["angular_velocity"])
                if record["angular_velocity"] is not None
                else None
            ),
            orientation=(
                tuple(record["orientation"]) if record["orientation"] is not None else None
            ),
            linear_acceleration_covariance=_decode_optional_tuple(
                record["linear_acceleration_covariance"]
            ),
            angular_velocity_covariance=_decode_optional_tuple(
                record["angular_velocity_covariance"]
            ),
            orientation_covariance=_decode_optional_tuple(record["orientation_covariance"]),
        )

    if modality == "external_pose":
        return ExternalPoseMeasurement(
            **common,
            parent_frame=FrameId(record["parent_frame"]),
            translation=tuple(record["translation"]),
            orientation=tuple(record["orientation"]),
            pose_covariance=_decode_optional_tuple(record["pose_covariance"]),
            linear_velocity=_decode_optional_vector3(record["linear_velocity"]),
            angular_velocity=_decode_optional_vector3(record["angular_velocity"]),
            twist_covariance=_decode_optional_tuple(record["twist_covariance"]),
        )

    raise SequenceArtifactError(f"unknown modality in index record: {modality!r}")


def _encode_optional_tuple(values: tuple[float, ...] | None) -> list[float] | None:
    """Encode an optional numeric tuple for JSON persistence."""
    return list(values) if values is not None else None


def _decode_optional_tuple(values: list[float] | None) -> tuple[float, ...] | None:
    """Decode an optional numeric tuple from JSON persistence."""
    return tuple(values) if values is not None else None


def _decode_optional_vector3(
    values: list[float] | None,
) -> tuple[float, float, float] | None:
    """Decode an optional three-element vector from JSON persistence."""
    if values is None:
        return None
    if len(values) != 3:
        raise SequenceArtifactError(f"expected a three-element vector, found {len(values)}")
    return values[0], values[1], values[2]


def _without_payload(observation: SourceObservation) -> SourceObservation:
    """Return ``observation`` minus its binary payload, keeping every metadata field."""
    if isinstance(observation, ImageObservation | LidarObservation):
        return replace(observation, data=b"")
    return observation
