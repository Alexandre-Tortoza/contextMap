"""Identidades determinísticas para conteúdo canônico de calibração.

Os hashes deste módulo cobrem valores que alteram a calibração efetiva, como intrínsecos e
transforms, sem misturar provenance descritiva. Eles permitem detectar mudanças e vincular
refinamentos à base correta sem declarar que duas calibrações são fisicamente equivalentes.
"""

from __future__ import annotations

import hashlib
import json

from contextmap.ingestion.calibration.calibration_serialization import (
    _encode_camera_model,
    _encode_transform,
)
from contextmap.ingestion.calibration.calibration_set import CameraModel, RigidTransform
from contextmap.ingestion.canonical_observations import FrameId, SensorId


def compute_content_hash(
    *,
    sensor_id: SensorId,
    frame_id: FrameId,
    camera_model: CameraModel | None,
) -> str:
    """Compute the deterministic content hash for a calibration entry.

    Only the canonical values a downstream consumer relies on are hashed
    (sensor/frame identity and camera model); provenance/free-text fields
    are intentionally excluded so hash changes track calibration content,
    not bookkeeping metadata.

    Args:
        sensor_id: Sensor the calibration applies to.
        frame_id: Coordinate frame of the camera model, if any.
        camera_model: Camera intrinsics/distortion, or ``None``.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    payload = json.dumps(
        {
            "sensor_id": str(sensor_id),
            "frame_id": str(frame_id),
            "camera_model": _encode_camera_model(camera_model),
        },
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def rigid_transform_digest(transform: RigidTransform) -> str:
    """Digest the canonical encoding of one static transform.

    The encoding is the transform's record in ``calibration.json``, with sorted keys and no
    whitespace, so the digest of a transform read back from a sequence artifact equals the
    digest of the one that was written.

    Args:
        transform: The transform to digest.

    Returns:
        ``"sha256:<hex digest>"``.
    """
    payload = json.dumps(_encode_transform(transform), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
