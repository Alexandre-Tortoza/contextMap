"""Contratos e resultados do preflight de geometria.

Este módulo só descreve: tolerâncias, achados, checagens e o relatório. Nenhuma avaliação mora
aqui; quem calcula cada parte é ``calibration``, ``readiness`` e ``clocks``, e ``service`` as
orquestra.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import FrameId
from contextmap.state_estimation.frame_graph import STATIC_ROTATION_NORM_TOLERANCE

CONVENTIONS = {
    "transform": "T_parent_child",
    "rotation": "unit quaternion (x, y, z, w)",
    "translation_unit": "m",
}
"""Conventions the checks assume; stated so a report is self-describing."""


@dataclass(frozen=True, kw_only=True)
class PreflightTolerances:
    """Documented numeric tolerances of the transform checks.

    Attributes:
        rotation_orthonormality: Largest ``|norm(q) - 1|``, ``|R^T R - I|``
            entry and ``|det(R) - 1|`` accepted for a static rotation. The
            default tolerates text-precision quaternions while rejecting a
            rotation that is not a rotation.
        inverse_round_trip_translation_m: Largest translation left by
            ``T * T^-1``, in meters.
        inverse_round_trip_rotation_rad: Largest rotation left by ``T * T^-1``.
        loop_translation_m: Largest position disagreement between redundant
            paths, in meters (1 mm by default).
        loop_rotation_rad: Largest rotation disagreement between redundant paths.
    """

    rotation_orthonormality: float = STATIC_ROTATION_NORM_TOLERANCE
    inverse_round_trip_translation_m: float = 1e-9
    inverse_round_trip_rotation_rad: float = 1e-9
    loop_translation_m: float = 1e-3
    loop_rotation_rad: float = 1e-3


class PreflightStatus(Enum):
    """Whether the run may start."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, kw_only=True)
class PreflightFinding:
    """One reason a run is blocked, or one caveat worth recording.

    Attributes:
        code: Stable identifier, e.g. ``"missing_static_transform"``.
        message: Human-readable explanation naming the frames or inputs involved.
    """

    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class TransformCheck:
    """Result of one numeric or availability check on a static transform.

    Attributes:
        subject: What was checked, e.g. ``"T_imu_velodyne"`` or ``"imu<-velodyne"``.
        check: ``"finite"``, ``"rotation_orthonormal"``, ``"inverse_round_trip"``,
            ``"relation_available"`` or ``"loop_consistency"``.
        passed: Whether the check passed.
        detail: The numbers behind the result.
    """

    subject: str
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True, kw_only=True)
class ClockCheck:
    """Result of one clock/timestamp metadata check.

    Attributes:
        subject: What was checked, e.g. ``"modality:lidar"``.
        passed: Whether the check passed.
        detail: The clock identities involved.
    """

    subject: str
    passed: bool
    detail: str


@dataclass(frozen=True, kw_only=True)
class FrameGraphSummary:
    """The frames the run relies on.

    Attributes:
        reference_frame: Reference frame of the dynamic transform.
        body_frame: Body frame of the dynamic transform.
        static_frames: Frames connected by static calibration transforms, sorted.
        static_edges: ``(parent, child)`` of each static transform.
        component_count: Independent groups of statically connected frames.
    """

    reference_frame: FrameId
    body_frame: FrameId
    static_frames: tuple[FrameId, ...]
    static_edges: tuple[tuple[FrameId, FrameId], ...]
    component_count: int


@dataclass(frozen=True, kw_only=True)
class DownstreamReadiness:
    """Whether a later capability's prerequisites exist, without blocking this run.

    Attributes:
        capability: The later capability, e.g. ``"sensor_association"``.
        ready: ``True`` when every prerequisite is present and valid.
        missing: What is missing or invalid; empty when ``ready``.
    """

    capability: str
    ready: bool
    missing: tuple[PreflightFinding, ...]


@dataclass(frozen=True, kw_only=True)
class GeometryPreflightReport:
    """Structured readiness of a run's geometry prerequisites.

    Attributes:
        required_inputs: Modalities the executing capability needs.
        available_inputs: Modalities present in the observations.
        frame_graph: The frames the run relies on.
        calibration_identity: Hash of the calibration used, or ``None`` when
            there is none.
        transform_checks: Every transform and relation check performed.
        clock_checks: Every clock/timestamp metadata check performed.
        blockers: Reasons the run must not start.
        warnings: Caveats that do not block, including downstream gaps.
        downstream: Readiness of later capabilities.
        conventions: Conventions the checks assume.
    """

    required_inputs: frozenset[str]
    available_inputs: frozenset[str]
    frame_graph: FrameGraphSummary
    calibration_identity: str | None
    transform_checks: tuple[TransformCheck, ...]
    clock_checks: tuple[ClockCheck, ...]
    blockers: tuple[PreflightFinding, ...]
    warnings: tuple[PreflightFinding, ...]
    downstream: tuple[DownstreamReadiness, ...]
    conventions: dict[str, str]

    @property
    def status(self) -> PreflightStatus:
        """``BLOCKED`` if there is any blocker, otherwise ``READY``."""
        return PreflightStatus.BLOCKED if self.blockers else PreflightStatus.READY
