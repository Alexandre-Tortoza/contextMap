"""Aplicação controlada de refinamentos versionados à calibração de câmera.

O módulo compõe uma correção de rotação sobre um transform estático somente quando alvo,
convenção e digest da base são compatíveis. Refinement é evidência reproduzível de uma correção
já estimada; este código não executa auto-calibração nem procura parâmetros automaticamente.
"""

from __future__ import annotations

import dataclasses
import math

from contextmap.ingestion.calibration.calibration_content_identity import rigid_transform_digest
from contextmap.ingestion.calibration.calibration_set import (
    CalibrationError,
    CalibrationRefinement,
    CalibrationSet,
)
from contextmap.ingestion.calibration.calibration_set_validation import (
    _target_index,
    _target_problems,
    validate_calibration_refinement,
)
from contextmap.shared import (
    Quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
)


def camera_rotation_correction(*, yaw_deg: float, pitch_deg: float, roll_deg: float) -> Quaternion:
    """Return ``Rz(roll) @ Rx(pitch) @ Ry(yaw)`` as a unit quaternion ``(x, y, z, w)``.

    The rotation maps base camera optical coordinates to refined ones,
    ``p_camera = R @ p_camera_base`` (:data:`CAMERA_OPTICAL_ROTATION_CONVENTION`).

    Args:
        yaw_deg: Rotation about the optical ``y`` axis, in degrees.
        pitch_deg: Rotation about the optical ``x`` axis, in degrees.
        roll_deg: Rotation about the optical ``z`` axis, in degrees.

    Returns:
        The correction rotation.
    """

    def about(axis: int, angle_deg: float) -> Quaternion:
        """Build a unit quaternion for one optical-axis correction.

        Args:
            axis: Quaternion vector component index for the rotation axis.
            angle_deg: Rotation angle in degrees.

        Returns:
            Unit quaternion representing the requested axis rotation.
        """
        half = math.radians(angle_deg) / 2.0
        vector = [0.0, 0.0, 0.0]
        vector[axis] = math.sin(half)
        return (vector[0], vector[1], vector[2], math.cos(half))

    # NOTE: Em quatérnios, o fator da direita é aplicado primeiro: Ry(yaw), depois Rx, depois Rz.
    rotation = quaternion_multiply(
        about(2, roll_deg), quaternion_multiply(about(0, pitch_deg), about(1, yaw_deg))
    )
    return normalize_quaternion(rotation)


def apply_calibration_refinement(
    base: CalibrationSet, refinement: CalibrationRefinement
) -> CalibrationSet:
    """Apply a camera-body refinement once, returning a new calibration set.

    With ``T_parent_camera`` taking ``p_parent = R p_camera + t`` and the correction
    ``p_camera = Rcorr p_camera_base``, the refined transform is ``R' = R Rcorr^T`` and
    ``t' = t``: the optical centre stays where it was and only the camera axes turn. The
    refinement is appended to :attr:`CalibrationSet.refinements`; entries and every other
    transform are unchanged.

    Args:
        base: The calibration the refinement was estimated on.
        refinement: The refinement to apply.

    Returns:
        The refined calibration set; ``base`` is not modified.

    Raises:
        CalibrationError: If the refinement is malformed (unknown convention, a non-finite
            angle or one above :data:`MAX_ROTATION_CORRECTION_DEG`), was already applied to
            ``base``, targets a transform ``base`` does not have or a frame that is not a
            calibrated camera, or was estimated on another transform than the one ``base``
            holds (which also refuses applying it on top of itself).
    """
    problems = validate_calibration_refinement(refinement)
    if problems:
        raise CalibrationError(f"invalid calibration refinement: {problems}")
    if any(item.refinement_id == refinement.refinement_id for item in base.refinements):
        raise CalibrationError(
            f"refinement {refinement.refinement_id!r} was already applied to this calibration"
        )
    target_problems = _target_problems(base, refinement)
    if target_problems:
        raise CalibrationError(f"cannot apply refinement: {target_problems}")
    index = _target_index(base, refinement)
    target = base.static_transforms[index]
    digest = rigid_transform_digest(target)
    if digest != refinement.base_transform_digest:
        raise CalibrationError(
            f"refinement {refinement.refinement_id!r} was estimated on a transform with digest "
            f"{refinement.base_transform_digest}, but T_{target.parent_frame}_{target.child_frame} "
            f"has digest {digest}: it is another base or it was already refined"
        )
    correction = camera_rotation_correction(
        yaw_deg=refinement.yaw_deg, pitch_deg=refinement.pitch_deg, roll_deg=refinement.roll_deg
    )
    # NOTE: R' = R · Rcorrᵀ: p_parent = R' p_camera = R Rcorrᵀ (Rcorr p_camera_base).
    rotation = normalize_quaternion(
        quaternion_multiply(target.rotation, quaternion_conjugate(correction))
    )
    refined = dataclasses.replace(target, rotation=rotation)
    transforms = tuple(
        refined if position == index else transform
        for position, transform in enumerate(base.static_transforms)
    )
    return dataclasses.replace(
        base, static_transforms=transforms, refinements=(*base.refinements, refinement)
    )
