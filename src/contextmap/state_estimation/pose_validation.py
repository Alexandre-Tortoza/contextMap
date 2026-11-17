"""Regras de publicação de pose compartilhadas pelos backends.

Os backends leem poses de fontes diferentes (uma medição externa, o arquivo de saída de um
estimador), mas publicam por um único contrato, então aplicam as mesmas regras aos números:
translação finita, orientação unitária finita (renormalizada somente dentro de uma tolerância
declarada, e registrada quando ocorre), covariância finita, detecção de gap temporal e o
diagnóstico correspondente. Regras de frame, clock e ordenação dependem da fonte e ficam em
cada backend. Nada aqui repara um valor em silêncio. Este módulo não faz parte da API pública.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.ingestion import SourceObservationId
from contextmap.shared import (
    Quaternion,
    Vector3,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_norm,
)
from contextmap.state_estimation.models import PoseEstimate, TrajectoryGap
from contextmap.state_estimation.ports import DiagnosticSeverity, EstimationDiagnostic


@dataclass(frozen=True, kw_only=True)
class PoseSampleViolation:
    """One rule a pose sample breaks.

    Attributes:
        rule: ``"non_finite_translation"``, ``"invalid_orientation"`` or
            ``"invalid_covariance"``; backends prefix it with their identity.
        detail: What is wrong, with the numbers involved.
    """

    rule: str
    detail: str


def check_pose_values(
    *,
    translation: Vector3,
    orientation: Quaternion,
    covariance: Sequence[float] | None,
    orientation_norm_tolerance: float,
) -> PoseSampleViolation | None:
    """Check the numbers of a pose sample.

    Args:
        translation: ``(x, y, z)`` in meters.
        orientation: Quaternion ``(x, y, z, w)`` as the source reported it.
        covariance: Row-major 6x6 covariance, or ``None`` when absent.
        orientation_norm_tolerance: Largest ``|norm - 1|`` a source quaternion
            may have and still be renormalized instead of rejected.

    Returns:
        The first violated rule, or ``None`` when the values are publishable.
    """
    if not all(math.isfinite(value) for value in translation):
        return PoseSampleViolation(
            rule="non_finite_translation", detail=f"translation is not finite: {translation!r}"
        )
    try:
        # A regra de quem normaliza é a de ``shared``: o que ela recusa (componente não
        # finito, norma nula, subnormal ou que transborda) nunca vira pose, mesmo que a
        # tolerância da fonte aceitasse a norma.
        normalize_quaternion(orientation)
    except ValueError as error:
        return PoseSampleViolation(
            rule="invalid_orientation",
            detail=f"orientation cannot be normalized to a rotation: {error}",
        )
    norm = quaternion_norm(orientation)
    if abs(norm - 1.0) > orientation_norm_tolerance:
        return PoseSampleViolation(
            rule="invalid_orientation",
            detail=(
                f"orientation norm {norm:.6f} differs from 1 by more than "
                f"{orientation_norm_tolerance}"
            ),
        )
    if covariance is not None and not all(math.isfinite(value) for value in covariance):
        return PoseSampleViolation(
            rule="invalid_covariance", detail="pose covariance contains a non-finite value"
        )
    return None


def canonical_orientation(orientation: Quaternion) -> tuple[Quaternion, tuple[str, ...]]:
    """Return the canonical unit orientation and the conversion it needed.

    Args:
        orientation: A quaternion already accepted by :func:`check_pose_values`,
            so one that :func:`~contextmap.shared.normalize_quaternion` accepts.

    Returns:
        The orientation unchanged when it is already unit within the contract's
        tolerance, otherwise the renormalized one with a note for
        ``PoseProvenance.conversions_applied``.
    """
    if is_unit_quaternion(orientation):
        return orientation, ()
    note = f"renormalized orientation (norm={quaternion_norm(orientation):.9f})"
    return normalize_quaternion(orientation), (note,)


def gap_between(
    previous: PoseEstimate, current: PoseEstimate, *, max_gap_ns: int | None
) -> TrajectoryGap | None:
    """Return a gap record when two consecutive poses are too far apart in time.

    Args:
        previous: The earlier pose.
        current: The later pose.
        max_gap_ns: Interval beyond which interpolation is not trusted;
            ``None`` records no gaps.

    Returns:
        A :class:`TrajectoryGap`, or ``None`` when the interval is acceptable.
    """
    if max_gap_ns is None:
        return None
    interval_ns = current.timestamp.total_nanoseconds() - previous.timestamp.total_nanoseconds()
    if interval_ns <= max_gap_ns:
        return None
    return TrajectoryGap(
        previous_estimate_id=previous.estimate_id,
        next_estimate_id=current.estimate_id,
        duration_ns=interval_ns,
    )


def timestamp_gap_diagnostic(
    *,
    backend_id: str,
    gap: TrajectoryGap,
    max_gap_ns: int | None,
    observation_id: SourceObservationId | None,
) -> EstimationDiagnostic:
    """Describe a recorded gap as the diagnostic every backend reports for it.

    Args:
        backend_id: Identity of the backend that recorded the gap; prefixes the code.
        gap: The gap :func:`gap_between` returned.
        max_gap_ns: The tolerance the gap exceeded, quoted in the message.
        observation_id: Observation that ended the gap, when it is traceable to one.

    Returns:
        A warning coded ``"<backend_id>.timestamp_gap"``.
    """
    return EstimationDiagnostic(
        severity=DiagnosticSeverity.WARNING,
        code=f"{backend_id}.timestamp_gap",
        message=(
            f"{gap.duration_ns} ns between {gap.previous_estimate_id!r} and "
            f"{gap.next_estimate_id!r} exceeds max_gap_ns={max_gap_ns}"
        ),
        observation_id=observation_id,
    )
