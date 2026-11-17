"""Diagnósticos das condições em que um frame produziu evidência.

Reúne provenance, contagens, qualidade de referência e achados sob tolerâncias explícitas.
Não otimiza calibração nem altera políticas, timestamps ou associações.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SourceObservationId
from contextmap.sensor_association.camera import CameraIdentity
from contextmap.sensor_association.dense_sampling import DenseFeatureSamples, InterpolationPolicy
from contextmap.sensor_association.diagnostics.reprojection import (
    TrustedCorrespondences,
    evaluated_correspondences,
    reprojection_statistics,
)
from contextmap.sensor_association.membership import FrameMembership, MembershipStatistics
from contextmap.sensor_association.models import (
    CalibrationRef,
    DepthMetric,
    PoseRef,
    VisibilityState,
)
from contextmap.sensor_association.projection_chain.frame_projection import (
    ExtrinsicRef,
    FrameProjection,
)
from contextmap.sensor_association.projection_chain.visibility import VisibilityResolution
from contextmap.sensor_association.quality import ReprojectionStatistics, ValueSummary
from contextmap.sensor_association.serialization import encode_calibration_ref, encode_pose_ref
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import TimeBounds

if TYPE_CHECKING:
    pass


DIAGNOSTICS_DEFINITIONS_VERSION = "association-diagnostics-v3"
"""Versioned identity of the diagnostic definitions in this module.

``v3``: a frame reports ``range_limit_candidate_support_count``, the associated support
whose optical-axis depth reaches ``max_range_m * cos(theta_max)``, and ``min_range_slack_m``, the
smallest ``max_range_m - range`` over that support; both are ``None`` without a range limit.
They make observable the regime where culling on an ``OPTICAL_AXIS`` camera can free, and so
associate, support the full map occludes.

``v2``: a frame's trusted reference now carries a
:class:`ReprojectionAttempt` with an explicit :class:`ReprojectionOutcome`, so a missing
residual is counted as one of three distinct facts instead of a single absence; the new
``REFERENCE_NOT_EVALUATED`` warning separates a candidate policy that evaluated no reference
geometry from a camera that cannot project it; and the invalid rate is measured over the
evaluated population rather than the declared one. The identity moves because the definitions
changed, not because the code did: two runs whose diagnostics mean different things must never
share it.
"""

_DEPTH_FLOOR_ULPS = 4
"""ULPs of ``max_range_m`` the depth floor is lowered by, so rounding can only add candidates."""

_NONE_PROJECTS = "no evaluated trusted reference correspondence could be projected"
_NOT_EVALUATED = (
    "the frame's candidate policy evaluated none of the trusted reference geometry, so the frame "
    "says nothing about it"
)


class FindingSeverity(Enum):
    """How serious a diagnostic finding is.

    Attributes:
        WARNING: A configured tolerance was exceeded; the association ran but should be read
            with care.
        FAILURE: The frame cannot support a meaningful association, or a reference that was
            provided could not be evaluated at all.
    """

    WARNING = "warning"
    FAILURE = "failure"


class FindingCode(Enum):
    """The conditions a frame diagnostic can flag.

    Attributes:
        POSE_TIME_DELTA_EXCEEDS_TOLERANCE: The pose used is further in time from the frame
            than the tolerance allows.
        FRAME_OUTSIDE_MAP_TIME_WINDOW: The frame lies outside the geometry's acquisition
            window by more than the tolerance allows.
        NOTHING_VISIBLE_IN_IMAGE: No map point is visible in the supported prepared image.
        REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE: The 95th percentile residual against the
            trusted reference exceeds the tolerance.
        REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE: Too many reference correspondences could
            not be projected.
        NO_REFERENCE_CORRESPONDENCE_PROJECTS: A trusted reference was provided, the frame
            evaluated its geometry, and the camera model could project none of it.
        REFERENCE_NOT_EVALUATED: A trusted reference was provided but the frame's candidate
            policy evaluated none of its geometry, so the frame says nothing about it. That is
            a limitation of the evaluated population, not a projection failure, which is why it
            is a warning and not a failure.
    """

    POSE_TIME_DELTA_EXCEEDS_TOLERANCE = "pose_time_delta_exceeds_tolerance"
    FRAME_OUTSIDE_MAP_TIME_WINDOW = "frame_outside_map_time_window"
    NOTHING_VISIBLE_IN_IMAGE = "nothing_visible_in_image"
    REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE = "reprojection_residual_exceeds_tolerance"
    REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE = "reprojection_invalid_rate_exceeds_tolerance"
    NO_REFERENCE_CORRESPONDENCE_PROJECTS = "no_reference_correspondence_projects"
    REFERENCE_NOT_EVALUATED = "reference_not_evaluated"


class ReprojectionOutcome(Enum):
    """What became of a frame's trusted reference, when one was supplied.

    Attributes:
        NO_REFERENCE: No trusted reference was supplied for the frame.
        NOT_EVALUATED: One was, but the frame's candidate policy evaluated none of its
            geometry, so the frame says nothing about it.
        NONE_PROJECTABLE: The frame evaluated its geometry and the camera model could project
            none of it.
        MEASURED: A residual was measured.
    """

    NO_REFERENCE = "no_reference"
    NOT_EVALUATED = "not_evaluated"
    NONE_PROJECTABLE = "none_projectable"
    MEASURED = "measured"


@dataclass(frozen=True, kw_only=True)
class ReprojectionAttempt:
    """What one frame did with a trusted reference, whether or not a residual came out.

    The *attempt* is separate from the *residual* on purpose. "No residual" used to be a
    single ``None``, which collapsed three different facts -- no reference, a reference the
    candidate policy never evaluated, and a reference the camera could not project -- and
    threw away their counts, so a downstream evaluator could neither tell them apart nor
    report how many correspondences were involved. The attempt is always present and always
    counted; the residual exists only when something projected.

    Attributes:
        outcome: Which of the four things happened.
        reference_id: The trusted correspondence set, or ``None`` when none was supplied.
        correspondence_count: Correspondences the reference declared.
        evaluated_count: Of those, the ones whose geometry the frame's candidate policy
            evaluated. Zero means the frame says nothing about the reference.
        invalid_count: Of the evaluated ones, those the camera model could not project.
    """

    outcome: ReprojectionOutcome
    reference_id: str | None
    correspondence_count: int
    evaluated_count: int
    invalid_count: int

    def __post_init__(self) -> None:
        """Validate that the counts narrow and match the outcome.

        Raises:
            ValueError: If a count is negative, they do not narrow from declared through
                evaluated to invalid, or they contradict ``outcome``. Every state the enum
                declares is checked, including the two that only a wrong producer could
                build: counts on a frame that had no reference, and a supplied reference with
                no correspondences in it.
        """
        if min(self.correspondence_count, self.evaluated_count, self.invalid_count) < 0:
            raise ValueError("reprojection counts must not be negative")
        if not self.invalid_count <= self.evaluated_count <= self.correspondence_count:
            raise ValueError(
                f"reprojection counts must narrow: {self.invalid_count} invalid of "
                f"{self.evaluated_count} evaluated of {self.correspondence_count} declared"
            )
        if (self.outcome is ReprojectionOutcome.NO_REFERENCE) != (self.reference_id is None):
            raise ValueError("a reference_id exists exactly when a reference was supplied")
        if self.outcome is ReprojectionOutcome.NO_REFERENCE:
            # A narrowing acima faz evaluated e invalid seguirem de correspondence_count.
            if self.correspondence_count:
                raise ValueError(
                    f"{self.outcome.value} means no reference existed, so it can count no "
                    f"correspondences, got {self.correspondence_count}"
                )
        elif self.correspondence_count == 0:
            raise ValueError(
                f"{self.outcome.value} names a supplied reference, which always declares at "
                "least one correspondence"
            )
        if self.outcome is ReprojectionOutcome.NOT_EVALUATED and self.evaluated_count:
            raise ValueError(
                f"{self.outcome.value} means nothing was evaluated, got {self.evaluated_count}"
            )
        if self.outcome is ReprojectionOutcome.NONE_PROJECTABLE and (
            self.evaluated_count == 0 or self.invalid_count != self.evaluated_count
        ):
            raise ValueError(
                f"{self.outcome.value} means every evaluated correspondence is invalid, got "
                f"{self.invalid_count} of {self.evaluated_count}"
            )
        if self.outcome is ReprojectionOutcome.MEASURED and self.invalid_count >= (
            self.evaluated_count
        ):
            raise ValueError(
                f"{self.outcome.value} needs an evaluated correspondence that projects"
            )

    @property
    def unevaluated_count(self) -> int:
        """Correspondences the frame's candidate policy did not evaluate."""
        return self.correspondence_count - self.evaluated_count

    @property
    def invalid_rate(self) -> float | None:
        """Share of the evaluated correspondences that could not be projected.

        ``None`` when nothing was evaluated, which is not the same as a rate of zero.
        """
        return None if self.evaluated_count == 0 else self.invalid_count / self.evaluated_count

    def to_record(self) -> dict[str, Any]:
        """Return the attempt as JSON primitives, with both populations named."""
        return {
            "outcome": self.outcome.value,
            "reference_id": self.reference_id,
            "correspondence_count": self.correspondence_count,
            "evaluated_count": self.evaluated_count,
            "unevaluated_count": self.unevaluated_count,
            "invalid_count": self.invalid_count,
            "invalid_rate": self.invalid_rate,
        }


@dataclass(frozen=True, kw_only=True)
class DiagnosticFinding:
    """One explicit warning or failure.

    Attributes:
        code: What was flagged.
        severity: How serious it is.
        message: A human-readable description with the numbers behind it.
        observed: The measured value, when there is one.
        tolerance: The tolerance it was compared with, when there is one.
    """

    code: FindingCode
    severity: FindingSeverity
    message: str
    observed: float | None
    tolerance: float | None


@dataclass(frozen=True, kw_only=True)
class DiagnosticTolerances:
    """The tolerances a frame is checked against. There are no defaults.

    A check is only as meaningful as the profile it comes from, so each tolerance is chosen
    per run; ``None`` disables that check.

    Attributes:
        max_pose_time_delta_ns: Largest accepted distance from the frame timestamp to the
            nearest contributing pose, in nanoseconds.
        max_map_window_offset_ns: Largest accepted distance from the frame timestamp to the
            geometry's acquisition window, in nanoseconds.
        max_reprojection_p95_px: Largest accepted 95th percentile residual, in pixels.
        max_reprojection_invalid_rate: Largest accepted share of reference correspondences that
            could not be projected, within ``[0, 1]``.
    """

    max_pose_time_delta_ns: int | None
    max_map_window_offset_ns: int | None
    max_reprojection_p95_px: float | None
    max_reprojection_invalid_rate: float | None

    def __post_init__(self) -> None:
        """Validate the tolerances.

        Raises:
            ValueError: If a tolerance is negative or not finite, or the invalid rate is above 1.
        """
        for name in (
            "max_pose_time_delta_ns",
            "max_map_window_offset_ns",
            "max_reprojection_p95_px",
            "max_reprojection_invalid_rate",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and not negative, got {value!r}")
        rate = self.max_reprojection_invalid_rate
        if rate is not None and rate > 1.0:
            raise ValueError(f"max_reprojection_invalid_rate must be at most 1, got {rate!r}")


@dataclass(frozen=True, kw_only=True)
class DenseSamplingSummary:
    """How one dense feature map was sampled in a frame, for comparing feature paths.

    Attributes:
        feature_id: The dense feature.
        source_artifact_id: The artifact that owns it.
        embedding_space_id: Its feature space.
        interpolation: The interpolation policy used.
        coordinate_transform_id: The transform that produced its sampling geometry.
        sampling_fingerprint: Hash of its complete sampling geometry.
        enhanced: ``True`` when the map is the output of a resolution enhancement.
        enhancement_source_feature_id: The native feature an enhanced map derives from.
        eligible_count: Visible points that were eligible.
        sampled_count: Of those, the ones the grid served.
        out_of_support_count: Of those, the ones it could not serve.
    """

    feature_id: str
    source_artifact_id: str
    embedding_space_id: str
    interpolation: InterpolationPolicy
    coordinate_transform_id: str
    sampling_fingerprint: str
    enhanced: bool
    enhancement_source_feature_id: str | None
    eligible_count: int
    sampled_count: int
    out_of_support_count: int


@dataclass(frozen=True, kw_only=True)
class FrameDiagnostics:
    """The diagnostic report of one association frame.

    The geometry and calibration part is independent of the dense feature path, so a native
    and an enhanced run over the same inputs report identical values there.

    Attributes:
        definitions_version: Version of the diagnostic definitions.
        source_observation_id: The camera frame.
        image_timestamp: When the frame was acquired.
        map_id: The geometric map projected.
        map_time_bounds: The acquisition window of the geometry.
        map_window_offset_ns: Signed distance from the frame timestamp to that window, in
            nanoseconds; ``0`` inside it and ``None`` when the clock domains differ.
        calibration_ref: The exact calibration and camera used.
        camera: The calibrated camera.
        pose_ref: The exact pose used; its ``time_delta_ns`` is the temporal alignment.
        extrinsic: The static body-to-camera extrinsic used.
        image_transform_id: The raw-to-prepared image chain.
        prepared_image_size: ``(width, height)`` of the prepared image.
        visibility_policy_id: The occlusion rule applied.
        visibility_policy_fingerprint: Hash of that rule's parameters.
        depth_metric: How depth is measured.
        point_count: Map points evaluated.
        state_counts: Points per decided visibility state.
        visible_count: Points that are supported and not occluded.
        visible_depth_m: Depth of the visible points, when there are any.
        membership: The region membership summary, when membership was evaluated.
        range_limit_candidate_support_count: Associated support whose optical-axis depth
            ``z`` is at least ``max_range_m * cos(theta_max)``, with ``theta_max`` the frame's
            :attr:`~FrameProjection.max_ray_angle_rad`. An element the range cut excludes and
            that lands in the prepared image, where every occluder lands, is farther than
            ``max_range_m`` at an angle within ``theta_max``, so its ``z`` is above that
            floor, and it can only have hidden support deeper than itself: zero
            is **conclusive**, no associated support lies where the cut could have changed
            its occlusion. A non-zero value only counts candidates; it does not say those
            points are near ``max_range_m`` or were affected. ``None`` when the frame has no
            range limit or membership was not evaluated -- not applicable, never zero.
        min_range_slack_m: Smallest ``max_range_m - range`` over the associated support, in
            meters; auxiliary to the count, which it does not replace. ``None`` when there is
            no range limit or no associated support.
        reprojection: Residual against a trusted reference, when one was measured.
        reprojection_attempt: What the frame did with its trusted reference, always present
            and always counted, so a missing residual is still an explainable, countable fact
            rather than an absence.
        dense_sampling: One summary per dense feature map sampled.
        findings: Explicit warnings and failures.
    """

    definitions_version: str
    source_observation_id: SourceObservationId
    image_timestamp: SourceTimestamp
    map_id: MapId
    map_time_bounds: TimeBounds
    map_window_offset_ns: int | None
    calibration_ref: CalibrationRef
    camera: CameraIdentity
    pose_ref: PoseRef
    extrinsic: ExtrinsicRef
    image_transform_id: str
    prepared_image_size: tuple[int, int]
    visibility_policy_id: str
    visibility_policy_fingerprint: str
    depth_metric: DepthMetric
    point_count: int
    state_counts: dict[VisibilityState, int]
    visible_count: int
    visible_depth_m: ValueSummary | None
    membership: MembershipStatistics | None
    range_limit_candidate_support_count: int | None
    min_range_slack_m: float | None
    reprojection: ReprojectionStatistics | None
    reprojection_attempt: ReprojectionAttempt
    dense_sampling: tuple[DenseSamplingSummary, ...]
    findings: tuple[DiagnosticFinding, ...]

    @property
    def failed(self) -> bool:
        """Whether any finding is a failure."""
        return any(finding.severity is FindingSeverity.FAILURE for finding in self.findings)

    def to_record(self) -> dict[str, Any]:
        """Return the report as JSON primitives, keeping every raw component."""
        return {
            "definitions_version": self.definitions_version,
            "source_observation_id": str(self.source_observation_id),
            "image_timestamp": self.image_timestamp.to_record(),
            "map": {
                "map_id": str(self.map_id),
                "time_bounds": {
                    "start": self.map_time_bounds.start.to_record(),
                    "end": self.map_time_bounds.end.to_record(),
                },
                "window_offset_ns": self.map_window_offset_ns,
            },
            "calibration_ref": encode_calibration_ref(self.calibration_ref),
            "camera": {
                "calibration_id": str(self.camera.calibration_id),
                "content_hash": self.camera.content_hash,
                "camera_frame": str(self.camera.camera_frame),
                "camera_model_kind": self.camera.camera_model_kind,
                "image_size": list(self.camera.image_size),
            },
            "pose_ref": encode_pose_ref(self.pose_ref),
            "extrinsic": {
                "calibration_identity": self.extrinsic.calibration_identity,
                "parent_frame": str(self.extrinsic.parent_frame),
                "child_frame": str(self.extrinsic.child_frame),
            },
            "image_transform_id": self.image_transform_id,
            "prepared_image_size": list(self.prepared_image_size),
            "visibility_policy": {
                "policy_id": self.visibility_policy_id,
                "fingerprint": self.visibility_policy_fingerprint,
            },
            "depth_metric": self.depth_metric.value,
            "visibility": {
                "point_count": self.point_count,
                "behind_camera": self.state_counts[VisibilityState.BEHIND_CAMERA],
                "outside_image": self.state_counts[VisibilityState.OUTSIDE_IMAGE],
                "outside_valid_support": self.state_counts[VisibilityState.OUTSIDE_VALID_SUPPORT],
                "occluded": self.state_counts[VisibilityState.OCCLUDED],
                "visible": self.visible_count,
                "visible_depth_m": None
                if self.visible_depth_m is None
                else dataclasses.asdict(self.visible_depth_m),
            },
            "membership": None if self.membership is None else dataclasses.asdict(self.membership),
            "range_limit_candidate_support_count": self.range_limit_candidate_support_count,
            "min_range_slack_m": self.min_range_slack_m,
            "reprojection": None
            if self.reprojection is None
            else dataclasses.asdict(self.reprojection),
            "reprojection_attempt": self.reprojection_attempt.to_record(),
            "dense_sampling": [
                {**dataclasses.asdict(summary), "interpolation": summary.interpolation.value}
                for summary in self.dense_sampling
            ],
            "findings": [
                {
                    "code": finding.code.value,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "observed": finding.observed,
                    "tolerance": finding.tolerance,
                }
                for finding in self.findings
            ],
        }


def diagnose_frame(
    resolution: VisibilityResolution,
    *,
    tolerances: DiagnosticTolerances,
    membership: FrameMembership | None = None,
    correspondences: TrustedCorrespondences | None = None,
    dense_samples: Sequence[DenseFeatureSamples] = (),
) -> FrameDiagnostics:
    """Build the diagnostic report of one association frame.

    Args:
        resolution: The visibility of the frame's points.
        tolerances: The tolerances the frame is checked against.
        membership: The frame's region membership, when it was evaluated.
        correspondences: A trusted reference to measure the reprojection residual against.
        dense_samples: The dense feature samplings of the frame, one per feature map.

    Returns:
        The report, with explicit findings for anything outside the tolerances.
    """
    import numpy as np

    frame = resolution.frame
    findings: list[DiagnosticFinding] = []
    window_offset = _window_offset_ns(frame.image_timestamp, frame.map_time_bounds)

    delta = frame.pose_ref.time_delta_ns
    limit = tolerances.max_pose_time_delta_ns
    if limit is not None and delta > limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.POSE_TIME_DELTA_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the pose used is {delta} ns from the frame, above the {limit} ns tolerance"
                ),
                observed=delta,
                tolerance=limit,
            )
        )
    window_limit = tolerances.max_map_window_offset_ns
    if window_limit is not None and window_offset is not None and abs(window_offset) > window_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.FRAME_OUTSIDE_MAP_TIME_WINDOW,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the frame is {abs(window_offset)} ns outside the geometry's acquisition "
                    f"window, above the {window_limit} ns tolerance"
                ),
                observed=abs(window_offset),
                tolerance=window_limit,
            )
        )
    if resolution.visible_count == 0:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.NOTHING_VISIBLE_IN_IMAGE,
                severity=FindingSeverity.FAILURE,
                message="no map point is visible in the supported prepared image",
                observed=0,
                tolerance=None,
            )
        )

    reprojection = None
    attempt = ReprojectionAttempt(
        outcome=ReprojectionOutcome.NO_REFERENCE,
        reference_id=None,
        correspondence_count=0,
        evaluated_count=0,
        invalid_count=0,
    )
    if correspondences is not None:
        reprojection = reprojection_statistics(frame, correspondences)
        # "Não avaliado" e "não projetável" são fatos diferentes: o primeiro é a política de
        # candidatos, o segundo é a câmera. Confundi-los transformava um recorte de alcance em
        # falha de calibração, e colapsá-los em None fazia o evaluator perder as contagens.
        declared = int(correspondences.geometry_indices.shape[0])
        evaluated = evaluated_correspondences(frame, correspondences)
        if reprojection is not None:
            outcome, invalid = ReprojectionOutcome.MEASURED, reprojection.invalid_count
        elif evaluated == 0:
            outcome, invalid = ReprojectionOutcome.NOT_EVALUATED, 0
        else:
            outcome, invalid = ReprojectionOutcome.NONE_PROJECTABLE, evaluated
        attempt = ReprojectionAttempt(
            outcome=outcome,
            reference_id=correspondences.reference_id,
            correspondence_count=declared,
            evaluated_count=evaluated,
            invalid_count=invalid,
        )
        findings.extend(
            _reprojection_findings(
                reprojection,
                tolerances,
                evaluated_count=evaluated,
                correspondence_count=declared,
            )
        )

    visible_depth = None
    if resolution.visible_count:
        depth = resolution.depth_m[resolution.visible]
        visible_depth = ValueSummary(
            count=int(depth.shape[0]),
            minimum=float(depth.min()),
            median=float(np.median(depth)),
            maximum=float(depth.max()),
        )
    candidate_support, range_slack = _range_limit_support(frame, membership)

    return FrameDiagnostics(
        definitions_version=DIAGNOSTICS_DEFINITIONS_VERSION,
        source_observation_id=frame.source_observation_id,
        image_timestamp=frame.image_timestamp,
        map_id=frame.map_id,
        map_time_bounds=frame.map_time_bounds,
        map_window_offset_ns=window_offset,
        calibration_ref=frame.calibration_ref,
        camera=frame.camera,
        pose_ref=frame.pose_ref,
        extrinsic=frame.extrinsic,
        image_transform_id=frame.image_transform.transform_id,
        prepared_image_size=frame.image_transform.prepared_size,
        visibility_policy_id=resolution.policy.policy_id,
        visibility_policy_fingerprint=resolution.policy.fingerprint(),
        depth_metric=resolution.depth_metric,
        point_count=len(frame.projectable),
        state_counts=resolution.state_counts(),
        visible_count=resolution.visible_count,
        visible_depth_m=visible_depth,
        membership=None if membership is None else membership.statistics(),
        range_limit_candidate_support_count=candidate_support,
        min_range_slack_m=range_slack,
        reprojection=reprojection,
        reprojection_attempt=attempt,
        dense_sampling=tuple(_dense_summary(samples) for samples in dense_samples),
        findings=tuple(findings),
    )


def _reprojection_findings(
    statistics: ReprojectionStatistics | None,
    tolerances: DiagnosticTolerances,
    *,
    evaluated_count: int,
    correspondence_count: int,
) -> list[DiagnosticFinding]:
    """Findings for one frame's reprojection, keeping the three outcomes apart.

    ``all unevaluated`` is a warning about the evaluated population; ``evaluated but none
    projectable`` is a failure about the camera; anything else is measured against tolerances.
    """
    if statistics is None:
        if evaluated_count == 0:
            return [
                DiagnosticFinding(
                    code=FindingCode.REFERENCE_NOT_EVALUATED,
                    severity=FindingSeverity.WARNING,
                    message=(
                        f"{_NOT_EVALUATED}: none of its {correspondence_count} correspondences "
                        "name geometry this frame evaluated"
                    ),
                    observed=correspondence_count,
                    tolerance=None,
                )
            ]
        return [
            DiagnosticFinding(
                code=FindingCode.NO_REFERENCE_CORRESPONDENCE_PROJECTS,
                severity=FindingSeverity.FAILURE,
                message=(
                    f"{_NONE_PROJECTS}: {evaluated_count} of {correspondence_count} were "
                    "evaluated and none of them projects"
                ),
                observed=evaluated_count,
                tolerance=None,
            )
        ]
    findings: list[DiagnosticFinding] = []
    p95_limit = tolerances.max_reprojection_p95_px
    if p95_limit is not None and statistics.p95_px > p95_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.REPROJECTION_RESIDUAL_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"the 95th percentile reprojection residual is {statistics.p95_px:.3f} px, "
                    f"above the {p95_limit} px tolerance"
                ),
                observed=statistics.p95_px,
                tolerance=p95_limit,
            )
        )
    # Sobre a população avaliada, nunca sobre a declarada: correspondências que a política de
    # candidatos excluiu diluiriam a taxa.
    rate = statistics.invalid_rate
    rate_limit = tolerances.max_reprojection_invalid_rate
    if rate_limit is not None and rate > rate_limit:
        findings.append(
            DiagnosticFinding(
                code=FindingCode.REPROJECTION_INVALID_RATE_EXCEEDS_TOLERANCE,
                severity=FindingSeverity.WARNING,
                message=(
                    f"{statistics.invalid_count} of the {statistics.evaluated_count} evaluated "
                    f"reference correspondences could not be projected, above the {rate_limit} "
                    f"tolerance ({statistics.unevaluated_count} more were not evaluated and do "
                    "not count either way)"
                ),
                observed=rate,
                tolerance=rate_limit,
            )
        )
    return findings


def _range_limit_support(
    frame: FrameProjection, membership: FrameMembership | None
) -> tuple[int | None, float | None]:
    """Count the associated support the range cut could affect, and its smallest range slack.

    An element the cut excludes has ``range > max_range_m``; if it lands in the prepared image,
    its angle to the axis is at most ``theta_max``, so its ``z`` is above
    ``max_range_m * cos(theta_max)``. Occlusion only hides what is deeper than its support, so
    only associated support at or beyond that floor could have been hidden by an excluded
    element. The floor has no physical margin: a margin would leave support between the floor
    and the floor plus the margin uncounted, and a zero would stop being conclusive. It is only
    lowered by a few ULPs, so that rounding can add candidates and never remove one.

    Returns:
        ``(count, slack)``: ``(None, None)`` without a range limit or membership, and a
        ``None`` slack when no support is associated.
    """
    import numpy as np

    max_range_m = frame.candidates.policy.max_range_m
    if max_range_m is None or membership is None:
        return None, None
    associated = np.unique(membership.point_indices)
    if associated.size == 0:
        return 0, None
    floor = max_range_m * math.cos(frame.max_ray_angle_rad)
    tolerance = _DEPTH_FLOOR_ULPS * math.ulp(max_range_m)
    count = int((frame.camera_depth_m[associated] >= floor - tolerance).sum())
    return count, float(max_range_m - frame.camera_range_m[associated].max())


def _window_offset_ns(timestamp: SourceTimestamp, bounds: TimeBounds) -> int | None:
    """Return signed distance to the map window, or None for incompatible clocks."""
    if timestamp.clock_id != bounds.start.clock_id:
        return None
    instant = timestamp.total_nanoseconds()
    start = bounds.start.total_nanoseconds()
    end = bounds.end.total_nanoseconds()
    if instant < start:
        return instant - start
    if instant > end:
        return instant - end
    return 0


def _dense_summary(samples: DenseFeatureSamples) -> DenseSamplingSummary:
    """Summarize one dense channel while retaining its identity and lineage."""
    provenance = samples.provenance
    enhancement = provenance.enhancement
    return DenseSamplingSummary(
        feature_id=str(provenance.feature_id),
        source_artifact_id=provenance.source_artifact_id,
        embedding_space_id=provenance.embedding_space_id,
        interpolation=provenance.interpolation,
        coordinate_transform_id=provenance.coordinate_transform_id,
        sampling_fingerprint=provenance.sampling_fingerprint,
        enhanced=enhancement is not None,
        enhancement_source_feature_id=None
        if enhancement is None
        else str(enhancement.source_feature_id),
        eligible_count=int(samples.eligible_indices.shape[0]),
        sampled_count=samples.sampled_count,
        out_of_support_count=samples.out_of_support_count,
    )
