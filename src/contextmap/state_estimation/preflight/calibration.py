"""Avaliação da calibração estática: identidade, checagens numéricas e consistência de laços.

Este módulo responde "a calibração é válida e consistente?", sem saber o que cada capability
exige dela: quem cruza o resultado com os frames realmente usados é ``readiness``. Ele não
possui nem reescreve a calibração (isso é de Ingestion); só a avalia.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.ingestion import CalibrationSet, FrameId, RigidTransform
from contextmap.shared import (
    RotationMatrix,
    compose_rigid,
    invert_rigid,
    quaternion_angle_between,
    quaternion_norm,
    quaternion_to_rotation_matrix,
)
from contextmap.state_estimation.frame_graph import LoopInconsistency, StaticFrameGraph
from contextmap.state_estimation.preflight.models import (
    PreflightFinding,
    PreflightTolerances,
    TransformCheck,
)


def calibration_identity(calibration: CalibrationSet | None) -> str | None:
    """Compute a deterministic identity of the calibration a run used.

    Args:
        calibration: The canonical calibration, or ``None``.

    Returns:
        ``"sha256:<hex digest>"`` over the static transforms and the entry
        content hashes, independent of their order, and over the identity of each
        applied refinement when there is one (an unrefined calibration keeps the
        identity it had before refinements existed); ``None`` without calibration.
    """
    if calibration is None:
        return None
    payload: dict[str, object] = {
        "schema_version": calibration.schema_version,
        "static_transforms": sorted(
            [str(t.parent_frame), str(t.child_frame), list(t.translation), list(t.rotation)]
            for t in calibration.static_transforms
        ),
        "entries": sorted(
            [str(entry_id), entry.content_hash] for entry_id, entry in calibration.entries.items()
        ),
    }
    if calibration.refinements:
        payload["refinements"] = [
            refinement.identity_record() for refinement in calibration.refinements
        ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _transform_checks(
    transforms: Sequence[RigidTransform], tolerances: PreflightTolerances
) -> tuple[list[TransformCheck], set[int]]:
    """Check every static transform; return the checks and the indexes that failed."""
    checks: list[TransformCheck] = []
    failed: set[int] = set()
    for index, transform in enumerate(transforms):
        subject = f"T_{transform.parent_frame}_{transform.child_frame}"
        values = (*transform.translation, *transform.rotation)
        finite = all(math.isfinite(value) for value in values)
        checks.append(
            TransformCheck(
                subject=subject,
                check="finite",
                passed=finite,
                detail="all values finite" if finite else f"non-finite values in {values!r}",
            )
        )
        if not finite:
            failed.add(index)
            continue

        # A matriz sozinha não denuncia um quaternion sem norma unitária cuja parte vetorial é
        # nula (vira a identidade), então a norma entra na mesma medida de erro.
        error = max(
            abs(quaternion_norm(transform.rotation) - 1.0),
            _orthonormality_error(quaternion_to_rotation_matrix(transform.rotation)),
        )
        orthonormal = error <= tolerances.rotation_orthonormality
        checks.append(
            TransformCheck(
                subject=subject,
                check="rotation_orthonormal",
                passed=orthonormal,
                detail=(
                    f"max(||q|| - 1, |R^T R - I|, |det R - 1|) = {error:.3e}, "
                    f"tolerance {tolerances.rotation_orthonormality:.1e}"
                ),
            )
        )
        if not orthonormal:
            failed.add(index)
            continue

        inverse_translation, inverse_rotation = invert_rigid(
            translation=transform.translation, rotation=transform.rotation
        )
        round_trip_translation, round_trip_rotation = compose_rigid(
            outer_translation=transform.translation,
            outer_rotation=transform.rotation,
            inner_translation=inverse_translation,
            inner_rotation=inverse_rotation,
        )
        translation_left = math.hypot(*round_trip_translation)
        rotation_left = quaternion_angle_between((0.0, 0.0, 0.0, 1.0), round_trip_rotation)
        round_trip = (
            translation_left <= tolerances.inverse_round_trip_translation_m
            and rotation_left <= tolerances.inverse_round_trip_rotation_rad
        )
        checks.append(
            TransformCheck(
                subject=subject,
                check="inverse_round_trip",
                passed=round_trip,
                detail=f"T * T^-1 leaves {translation_left:.3e} m and {rotation_left:.3e} rad",
            )
        )
        if not round_trip:
            failed.add(index)
    return checks, failed


def _orthonormality_error(matrix: RotationMatrix) -> float:
    """Return the largest deviation of a matrix from a proper rotation."""
    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    row0, row1, row2 = matrix
    cross = (
        row1[1] * row2[2] - row1[2] * row2[1],
        row1[2] * row2[0] - row1[0] * row2[2],
        row1[0] * row2[1] - row1[1] * row2[0],
    )
    determinant = sum(a * b for a, b in zip(row0, cross, strict=True))
    return max(worst, abs(determinant - 1.0))


@dataclass(frozen=True)
class StaticCalibrationAssessment:
    """What the static calibration looks like, independent of which capability uses it.

    Attributes:
        transforms: Every static transform of the calibration, in input order.
        graph: The graph over all of them, used to answer which frames are connected.
        failed_indexes: Indexes of the transforms that failed a numeric check.
        loops: Redundant paths that disagree, computed only over transforms that passed.
        checks: Numeric and loop-consistency checks performed, in report order.
    """

    transforms: Sequence[RigidTransform]
    graph: StaticFrameGraph
    failed_indexes: frozenset[int]
    loops: tuple[LoopInconsistency, ...]
    checks: tuple[TransformCheck, ...]


def assess_static_calibration(
    calibration: CalibrationSet | None, limits: PreflightTolerances
) -> StaticCalibrationAssessment:
    """Check every static transform and the consistency of the graph they form."""
    transforms = calibration.static_transforms if calibration is not None else ()
    checks, failed_indexes = _transform_checks(transforms, limits)
    # NOTE: os laços só são avaliados sobre transforms que passaram nas checagens numéricas;
    # a extrínseca inválida já é reportada por `_transform_checks`.
    valid_graph = StaticFrameGraph(
        [t for index, t in enumerate(transforms) if index not in failed_indexes]
    )
    loops = valid_graph.loop_inconsistencies(
        translation_tolerance_m=limits.loop_translation_m,
        rotation_tolerance_rad=limits.loop_rotation_rad,
    )
    for loop in loops:
        checks.append(
            TransformCheck(
                subject=f"{loop.frames[0]}<->{loop.frames[1]}",
                check="loop_consistency",
                passed=False,
                detail=f"redundant paths disagree by {loop.translation_error_m:.3e} m "
                f"and {loop.rotation_error_rad:.3e} rad",
            )
        )
    if valid_graph.loop_count > 0 and not loops:
        checks.append(
            TransformCheck(
                subject="static frame graph",
                check="loop_consistency",
                passed=True,
                detail=f"{valid_graph.loop_count} redundant loop(s) agree within tolerance",
            )
        )
    return StaticCalibrationAssessment(
        transforms=transforms,
        graph=StaticFrameGraph(transforms),
        failed_indexes=frozenset(failed_indexes),
        loops=loops,
        checks=tuple(checks),
    )


def calibration_findings(
    assessment: StaticCalibrationAssessment, touched: set[FrameId]
) -> list[PreflightFinding]:
    """Report invalid or ambiguous calibration in the frames a capability actually uses."""
    found: list[PreflightFinding] = []
    for index in sorted(assessment.failed_indexes):
        transform = assessment.transforms[index]
        if transform.parent_frame in touched:
            found.append(
                PreflightFinding(
                    code="invalid_transform",
                    message=f"static transform T_{transform.parent_frame}_"
                    f"{transform.child_frame} failed validation",
                )
            )
    for loop in assessment.loops:
        if loop.frames[0] in touched:
            found.append(
                PreflightFinding(
                    code="ambiguous_frame_graph",
                    message=f"redundant static paths through {loop.frames[0]!r} and "
                    f"{loop.frames[1]!r} disagree by {loop.translation_error_m:.3e} m and "
                    f"{loop.rotation_error_rad:.3e} rad",
                )
            )
    return found


def unused_calibration_warnings(
    assessment: StaticCalibrationAssessment, touched: set[FrameId]
) -> list[PreflightFinding]:
    """Report invalid or ambiguous calibration the executing capability does not use."""
    warnings: list[PreflightFinding] = []
    for index in sorted(assessment.failed_indexes):
        transform = assessment.transforms[index]
        if transform.parent_frame not in touched:
            warnings.append(
                PreflightFinding(
                    code="unused_calibration_invalid",
                    message=f"static transform T_{transform.parent_frame}_{transform.child_frame} "
                    "failed validation, but this run does not use it",
                )
            )
    for loop in assessment.loops:
        if loop.frames[0] not in touched:
            warnings.append(
                PreflightFinding(
                    code="unused_calibration_inconsistent",
                    message=f"redundant static paths through {loop.frames[0]!r} and "
                    f"{loop.frames[1]!r} disagree, but this run does not use them",
                )
            )
    return warnings
