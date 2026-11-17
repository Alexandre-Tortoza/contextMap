"""Fachada interna das operações compartilhadas pelos adapters ROS 1 e ROS 2.

Este módulo reúne imports dos helpers ROS especializados para que os adapters componham um único
ponto interno sem recriar um arquivo genérico com múltiplas responsabilidades. Ele é um detalhe da
borda ROS e não faz parte da API pública de ingestion.
"""

from .ros_calibration_extraction import (
    build_camera_model,
    calibration_ids_by_sensor,
    merge_calibration,
)
from .ros_external_pose_decoding import decode_pose
from .ros_image_decoding import decode_image
from .ros_imu_decoding import decode_imu
from .ros_lidar_decoding import decode_lidar
from .ros_message_decoding import deserialize_payload, sanitize_topic, stamp_to_timestamp
from .ros_recording_window import resolve_window_bounds
from .ros_source_content_hash import StreamingContentHash

__all__ = [
    "StreamingContentHash",
    "build_camera_model",
    "calibration_ids_by_sensor",
    "decode_image",
    "decode_imu",
    "decode_lidar",
    "decode_pose",
    "deserialize_payload",
    "merge_calibration",
    "resolve_window_bounds",
    "sanitize_topic",
    "stamp_to_timestamp",
]
