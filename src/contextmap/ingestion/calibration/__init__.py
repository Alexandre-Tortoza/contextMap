"""Fachada pública do subdomínio de calibração pertencente à ingestion.

Este pacote reúne modelos de câmera, transforms estáticos, CalibrationSet, validação, refinamento,
serialização e identidade de conteúdo. A calibration continua sendo evidência canônica de
ingestion; aplicar projeção 2D/3D ou estimar trajetória permanece responsabilidade downstream.
"""

from .calibration_content_identity import compute_content_hash, rigid_transform_digest
from .calibration_refinement import apply_calibration_refinement, camera_rotation_correction
from .calibration_serialization import (
    decode_calibration_refinement_document,
    decode_calibration_set,
    encode_calibration_refinement_document,
    encode_calibration_set,
    load_calibration_refinement,
)
from .calibration_set import (
    CAMERA_OPTICAL_ROTATION_CONVENTION,
    MAX_ROTATION_CORRECTION_DEG,
    REFINEMENT_DOCUMENT_SCHEMA,
    SCHEMA_VERSION,
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationRefinement,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    MeiCameraModel,
    PinholeCameraModel,
    RefinementValidation,
    RigidTransform,
    camera_model_kind,
)
from .calibration_set_validation import (
    ensure_valid_calibration_set,
    validate_calibration_refinement,
    validate_calibration_set,
)

__all__ = [
    "CAMERA_OPTICAL_ROTATION_CONVENTION",
    "MAX_ROTATION_CORRECTION_DEG",
    "REFINEMENT_DOCUMENT_SCHEMA",
    "SCHEMA_VERSION",
    "CalibrationEntry",
    "CalibrationError",
    "CalibrationProvenance",
    "CalibrationRefinement",
    "CalibrationSet",
    "CameraModel",
    "DistortionModel",
    "FisheyeCameraModel",
    "MeiCameraModel",
    "PinholeCameraModel",
    "RefinementValidation",
    "RigidTransform",
    "apply_calibration_refinement",
    "camera_model_kind",
    "camera_rotation_correction",
    "compute_content_hash",
    "decode_calibration_refinement_document",
    "decode_calibration_set",
    "encode_calibration_refinement_document",
    "encode_calibration_set",
    "ensure_valid_calibration_set",
    "load_calibration_refinement",
    "rigid_transform_digest",
    "validate_calibration_refinement",
    "validate_calibration_set",
]
