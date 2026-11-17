"""Validação das invariantes estruturais de CalibrationSet e refinamentos.

As regras verificam modelos de câmera, coeficientes, valores finitos, quaternions, transforms,
alvos e convenções de refinamento. A validação reporta problemas ou falha explicitamente; não
normaliza silenciosamente uma calibração inválida.
"""

from __future__ import annotations

import math

from contextmap.ingestion.calibration.calibration_set import (
    _PLUMB_BOB_COEFFICIENT_COUNTS,
    _QUATERNION_NORM_TOLERANCE,
    CAMERA_OPTICAL_ROTATION_CONVENTION,
    MAX_ROTATION_CORRECTION_DEG,
    CalibrationError,
    CalibrationRefinement,
    CalibrationSet,
    CameraModel,
    FisheyeCameraModel,
    MeiCameraModel,
    PinholeCameraModel,
)
from contextmap.ingestion.canonical_observations import CalibrationReferenceId


def validate_calibration_set(calibration_set: CalibrationSet) -> list[str]:
    """Check a calibration set for missing, malformed, or inconsistent data.

    Args:
        calibration_set: The calibration set to validate.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []

    for key, entry in calibration_set.entries.items():
        if entry.calibration_id != key:
            problems.append(
                f"entries key {key!r} does not match entry.calibration_id {entry.calibration_id!r}"
            )
        if entry.camera_model is not None:
            problems.extend(_validate_camera_model(entry.calibration_id, entry.camera_model))

    seen_pairs: set[tuple[str, str]] = set()
    for transform in calibration_set.static_transforms:
        pair = (str(transform.parent_frame), str(transform.child_frame))
        if transform.parent_frame == transform.child_frame:
            problems.append(f"transform has identical parent_frame and child_frame: {pair[0]!r}")
        if pair in seen_pairs:
            problems.append(f"duplicate static transform for parent={pair[0]!r} child={pair[1]!r}")
        seen_pairs.add(pair)

        norm = math.sqrt(sum(component**2 for component in transform.rotation))
        if not math.isclose(norm, 1.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
            problems.append(
                f"transform parent={pair[0]!r} child={pair[1]!r} rotation is not a unit "
                f"quaternion (norm={norm:.6f})"
            )

    seen_refinements: set[str] = set()
    for refinement in calibration_set.refinements:
        if refinement.refinement_id in seen_refinements:
            problems.append(f"duplicate refinement_id {refinement.refinement_id!r}")
        seen_refinements.add(refinement.refinement_id)
        problems.extend(validate_calibration_refinement(refinement))
        problems.extend(_target_problems(calibration_set, refinement))

    return problems


def ensure_valid_calibration_set(calibration_set: CalibrationSet) -> None:
    """Validate a calibration set and raise if it has any problem.

    Args:
        calibration_set: The calibration set to validate.

    Raises:
        CalibrationError: If :func:`validate_calibration_set` finds any
            problem.
    """
    problems = validate_calibration_set(calibration_set)
    if problems:
        raise CalibrationError(f"invalid calibration set: {problems}")


def validate_calibration_refinement(refinement: CalibrationRefinement) -> list[str]:
    """Check whether a calibration refinement is valid independently of its target.

    Args:
        refinement: The refinement to validate.

    Returns:
        A list of human-readable problems; empty means no problem found.
    """
    problems: list[str] = []
    name = refinement.refinement_id
    if not name:
        problems.append("refinement_id must not be empty")
    if refinement.convention != CAMERA_OPTICAL_ROTATION_CONVENTION:
        problems.append(
            f"{name}: unsupported rotation convention {refinement.convention!r}; "
            f"expected {CAMERA_OPTICAL_ROTATION_CONVENTION!r}"
        )
    for axis in ("yaw_deg", "pitch_deg", "roll_deg"):
        value = getattr(refinement, axis)
        if not math.isfinite(value):
            problems.append(f"{name}: {axis} must be finite, got {value!r}")
        elif abs(value) > MAX_ROTATION_CORRECTION_DEG:
            problems.append(
                f"{name}: {axis}={value} exceeds the {MAX_ROTATION_CORRECTION_DEG} degree limit"
            )
    return problems


def _target_index(calibration_set: CalibrationSet, refinement: CalibrationRefinement) -> int:
    """Return the target transform position, or ``-1`` when it is absent."""
    for position, transform in enumerate(calibration_set.static_transforms):
        if (transform.parent_frame, transform.child_frame) == (
            refinement.parent_frame,
            refinement.child_frame,
        ):
            return position
    return -1


def _target_problems(
    calibration_set: CalibrationSet, refinement: CalibrationRefinement
) -> list[str]:
    """Return problems with a refinement target in a calibration set."""
    name = refinement.refinement_id
    if _target_index(calibration_set, refinement) < 0:
        return [
            f"{name}: no static transform parent={refinement.parent_frame!r} "
            f"child={refinement.child_frame!r}"
        ]
    # NOTE: Yaw, pitch e roll são eixos ópticos; sem câmera calibrada no frame filho, não
    # significam nada.
    if not any(
        entry.camera_model is not None and entry.frame_id == refinement.child_frame
        for entry in calibration_set.entries.values()
    ):
        return [f"{name}: frame {refinement.child_frame!r} is not a calibrated camera frame"]
    return []


def _validate_camera_model(calibration_id: CalibrationReferenceId, model: CameraModel) -> list[str]:
    """Return structural problems for one canonical camera model.

    Args:
        calibration_id: Identity used to make each diagnostic actionable.
        model: Camera model whose numeric and shape invariants are checked.

    Returns:
        Human-readable validation problems; empty means the model is structurally valid.
    """
    problems: list[str] = []
    if model.width <= 0 or model.height <= 0:
        problems.append(f"{calibration_id}: width and height must be positive")
    for name in ("fx", "fy", "cx", "cy"):
        if not math.isfinite(getattr(model, name)):
            problems.append(f"{calibration_id}: {name} must be finite")
    if model.fx <= 0 or model.fy <= 0:
        problems.append(f"{calibration_id}: fx and fy must be positive")
    if not all(math.isfinite(value) for value in model.distortion_coefficients):
        problems.append(f"{calibration_id}: distortion coefficients must be finite")

    if isinstance(model, PinholeCameraModel):
        expected = _PLUMB_BOB_COEFFICIENT_COUNTS[model.distortion_model]
        actual = len(model.distortion_coefficients)
        if actual != expected:
            problems.append(
                f"{calibration_id}: {model.distortion_model.value} expects {expected} "
                f"distortion coefficients, got {actual}"
            )
    elif isinstance(model, FisheyeCameraModel) and len(model.distortion_coefficients) != 4:
        problems.append(
            f"{calibration_id}: fisheye model requires exactly 4 distortion coefficients, "
            f"got {len(model.distortion_coefficients)}"
        )
    elif isinstance(model, MeiCameraModel):
        if len(model.distortion_coefficients) != 4:
            problems.append(
                f"{calibration_id}: mei model requires exactly 4 distortion coefficients "
                f"(k1, k2, p1, p2), got {len(model.distortion_coefficients)}"
            )
        if not math.isfinite(model.xi) or model.xi < 0:
            problems.append(
                f"{calibration_id}: mei mirror parameter xi must be finite and not negative, "
                f"got {model.xi!r}"
            )

    return problems
