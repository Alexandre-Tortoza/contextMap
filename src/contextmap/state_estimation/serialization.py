"""Codificação JSON dos contratos de State Estimation.

Os registros contêm somente primitivas JSON, então uma trajetória persistida abre sem ROS,
sem estimador e sem NumPy. Poses e metadados da trajetória são codificados separadamente
porque o run artifact guarda uma pose por linha e os metadados uma única vez. Este módulo
também codifica os relatórios que acompanham a trajetória (preflight, movimento, qualidade e
diagnósticos); o layout de arquivos, o manifest e o ciclo de escrita ficam em ``run_artifact``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from contextmap.ingestion import FrameId, SequenceArtifactId, SourceObservationId
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    Trajectory,
    TrajectoryGap,
    TrajectoryId,
    TrajectoryProvenance,
)
from contextmap.state_estimation.motion import DistributionSummary, MotionSummary
from contextmap.state_estimation.ports import EstimationDiagnostic
from contextmap.state_estimation.preflight import GeometryPreflightReport, TransformCheck

_NANOSECONDS_PER_SECOND = 1_000_000_000


class InvalidTrajectoryRecordError(ValueError):
    """Raised when a persisted trajectory record omits a field the contract requires.

    Presence is part of the contract: a field that may be ``null`` (such as the auxiliary
    lineage) is still required to be written, so an omitted field means a malformed record,
    not "no value".
    """


def encode_pose_estimate(pose: PoseEstimate) -> dict[str, Any]:
    """Encode a pose into a JSON-serializable record.

    Args:
        pose: The pose to encode.

    Returns:
        A record whose ``translation_m`` is in meters and whose
        ``orientation_xyzw`` follows the ``(x, y, z, w)`` order.
    """
    return {
        "estimate_id": str(pose.estimate_id),
        "timestamp": _encode_timestamp(pose.timestamp),
        "parent_frame": str(pose.parent_frame),
        "child_frame": str(pose.child_frame),
        "translation_m": list(pose.translation_m),
        "orientation_xyzw": list(pose.orientation),
        "validity": pose.validity.value,
        "covariance": None if pose.covariance is None else list(pose.covariance),
        "provenance": {
            "source_observation_ids": [
                str(item) for item in pose.provenance.source_observation_ids
            ],
            "conversions_applied": list(pose.provenance.conversions_applied),
            "derived_from": [str(item) for item in pose.provenance.derived_from],
        },
    }


def decode_pose_estimate(record: Mapping[str, Any]) -> PoseEstimate:
    """Decode a pose from a record produced by :func:`encode_pose_estimate`.

    Args:
        record: The persisted record.

    Returns:
        The validated pose.

    Raises:
        ValueError: If the record is malformed or violates the pose contract.
    """
    tx, ty, tz = record["translation_m"]
    qx, qy, qz, qw = record["orientation_xyzw"]
    covariance = record["covariance"]
    provenance = record["provenance"]
    return PoseEstimate(
        estimate_id=PoseEstimateId(record["estimate_id"]),
        timestamp=_decode_timestamp(record["timestamp"]),
        parent_frame=FrameId(record["parent_frame"]),
        child_frame=FrameId(record["child_frame"]),
        translation_m=(tx, ty, tz),
        orientation=(qx, qy, qz, qw),
        validity=PoseValidity(record["validity"]),
        provenance=PoseProvenance(
            source_observation_ids=tuple(
                SourceObservationId(item) for item in provenance["source_observation_ids"]
            ),
            conversions_applied=tuple(provenance["conversions_applied"]),
            derived_from=tuple(PoseEstimateId(item) for item in provenance["derived_from"]),
        ),
        covariance=None if covariance is None else tuple(covariance),
    )


def encode_trajectory_metadata(trajectory: Trajectory) -> dict[str, Any]:
    """Encode everything about a trajectory except its poses.

    Args:
        trajectory: The trajectory to encode.

    Returns:
        A JSON-serializable record; combine it with the poses in
        :func:`decode_trajectory`.
    """
    provenance = trajectory.provenance
    return {
        "trajectory_id": str(trajectory.trajectory_id),
        "reference_frame": str(trajectory.reference_frame),
        "body_frame": str(trajectory.body_frame),
        "gaps": [encode_gap(gap) for gap in trajectory.gaps],
        "provenance": {
            "estimator": encode_estimator_provenance(provenance.estimator),
            "sequence_artifact_id": str(provenance.sequence_artifact_id),
            "selection_id": provenance.selection_id,
            "calibration_identity": provenance.calibration_identity,
            "code_version": provenance.code_version,
            "auxiliary_sequence_artifact_id": (
                None
                if provenance.auxiliary_sequence_artifact_id is None
                else str(provenance.auxiliary_sequence_artifact_id)
            ),
            "auxiliary_selection_id": provenance.auxiliary_selection_id,
        },
    }


def decode_trajectory(metadata: Mapping[str, Any], poses: Sequence[PoseEstimate]) -> Trajectory:
    """Rebuild a trajectory from its metadata record and its poses.

    Args:
        metadata: Record produced by :func:`encode_trajectory_metadata`.
        poses: The poses, in trajectory order.

    Returns:
        The validated trajectory.

    Raises:
        InvalidTrajectoryRecordError: If the metadata omits a field the contract requires.
        ValueError: If the poses and metadata violate the trajectory contract.
    """
    try:
        return _decode_trajectory(metadata, poses)
    except KeyError as error:
        raise InvalidTrajectoryRecordError(
            f"trajectory metadata is missing the required field {error.args[0]!r}"
        ) from error


def _decode_trajectory(metadata: Mapping[str, Any], poses: Sequence[PoseEstimate]) -> Trajectory:
    provenance = metadata["provenance"]
    return Trajectory(
        trajectory_id=TrajectoryId(metadata["trajectory_id"]),
        reference_frame=FrameId(metadata["reference_frame"]),
        body_frame=FrameId(metadata["body_frame"]),
        poses=tuple(poses),
        gaps=tuple(decode_gap(gap) for gap in metadata["gaps"]),
        provenance=TrajectoryProvenance(
            estimator=decode_estimator_provenance(provenance["estimator"]),
            sequence_artifact_id=SequenceArtifactId(provenance["sequence_artifact_id"]),
            selection_id=provenance["selection_id"],
            calibration_identity=provenance["calibration_identity"],
            code_version=provenance["code_version"],
            auxiliary_sequence_artifact_id=(
                None
                if provenance["auxiliary_sequence_artifact_id"] is None
                else SequenceArtifactId(provenance["auxiliary_sequence_artifact_id"])
            ),
            auxiliary_selection_id=provenance["auxiliary_selection_id"],
        ),
    )


def encode_estimator_provenance(estimator: EstimatorProvenance) -> dict[str, Any]:
    """Encode the backend identity shared by the trajectory metadata and the run manifest.

    Args:
        estimator: The backend and configuration identity.

    Returns:
        A JSON-serializable record.
    """
    return {
        "backend_id": estimator.backend_id,
        "backend_version": estimator.backend_version,
        "configuration_fingerprint": estimator.configuration_fingerprint,
    }


def decode_estimator_provenance(record: Mapping[str, Any]) -> EstimatorProvenance:
    """Decode a record produced by :func:`encode_estimator_provenance`."""
    return EstimatorProvenance(
        backend_id=record["backend_id"],
        backend_version=record["backend_version"],
        configuration_fingerprint=record["configuration_fingerprint"],
    )


def encode_gap(gap: TrajectoryGap) -> dict[str, Any]:
    """Encode a trajectory gap into a JSON-serializable record."""
    return {
        "previous_estimate_id": str(gap.previous_estimate_id),
        "next_estimate_id": str(gap.next_estimate_id),
        "duration_ns": gap.duration_ns,
    }


def decode_gap(record: Mapping[str, Any]) -> TrajectoryGap:
    """Decode a record produced by :func:`encode_gap`."""
    return TrajectoryGap(
        previous_estimate_id=PoseEstimateId(record["previous_estimate_id"]),
        next_estimate_id=PoseEstimateId(record["next_estimate_id"]),
        duration_ns=record["duration_ns"],
    )


def encode_diagnostic(diagnostic: EstimationDiagnostic) -> dict[str, Any]:
    """Encode one backend diagnostic event."""
    return {
        "severity": diagnostic.severity.value,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "observation_id": None
        if diagnostic.observation_id is None
        else str(diagnostic.observation_id),
    }


def encode_transform_check(check: TransformCheck) -> dict[str, Any]:
    """Encode one numeric or availability check on a static transform."""
    return {
        "subject": check.subject,
        "check": check.check,
        "passed": check.passed,
        "detail": check.detail,
    }


def encode_frame_summary(report: GeometryPreflightReport) -> dict[str, Any]:
    """Encode the frames the run relied on, with the conventions the checks assumed."""
    graph = report.frame_graph
    return {
        "dynamic": {
            "reference_frame": str(graph.reference_frame),
            "body_frame": str(graph.body_frame),
        },
        "static_frames": [str(frame) for frame in graph.static_frames],
        "static_edges": [[str(parent), str(child)] for parent, child in graph.static_edges],
        "component_count": graph.component_count,
        "calibration_identity": report.calibration_identity,
        "conventions": dict(report.conventions),
    }


def encode_preflight_report(report: GeometryPreflightReport) -> dict[str, Any]:
    """Encode the complete geometry preflight report."""
    return {
        "status": report.status.value,
        "required_inputs": sorted(report.required_inputs),
        "available_inputs": sorted(report.available_inputs),
        "calibration_identity": report.calibration_identity,
        "frame_graph": encode_frame_summary(report),
        "transform_checks": [encode_transform_check(check) for check in report.transform_checks],
        "clock_checks": [
            {"subject": c.subject, "passed": c.passed, "detail": c.detail}
            for c in report.clock_checks
        ],
        "blockers": [{"code": f.code, "message": f.message} for f in report.blockers],
        "warnings": [{"code": f.code, "message": f.message} for f in report.warnings],
        "downstream": [
            {
                "capability": item.capability,
                "ready": item.ready,
                "missing": [{"code": f.code, "message": f.message} for f in item.missing],
            }
            for item in report.downstream
        ],
        "conventions": dict(report.conventions),
    }


def encode_quality(
    trajectory: Trajectory, *, consumed_observation_count: int, rejected_observation_count: int
) -> dict[str, Any]:
    """Encode sampling, gap and flagging statistics of a trajectory with the backend counts."""
    summary = trajectory.quality_summary()
    duration_s = summary.duration_ns / _NANOSECONDS_PER_SECOND
    return {
        "pose_count": summary.pose_count,
        "duration_ns": summary.duration_ns,
        "sample_rate_hz": (summary.pose_count - 1) / duration_s if duration_s > 0 else None,
        "min_interval_ns": summary.min_interval_ns,
        "median_interval_ns": summary.median_interval_ns,
        "max_interval_ns": summary.max_interval_ns,
        "gap_count": summary.gap_count,
        "gaps": [encode_gap(gap) for gap in trajectory.gaps],
        "degraded_pose_count": summary.degraded_pose_count,
        "pose_with_covariance_count": summary.pose_with_covariance_count,
        "consumed_observation_count": consumed_observation_count,
        "rejected_observation_count": rejected_observation_count,
    }


def encode_motion(motion: MotionSummary) -> dict[str, Any]:
    """Encode the per-interval motion distributions of a trajectory."""
    return {
        "translation_delta_m": _encode_distribution(motion.translation_delta_m),
        "orientation_delta_rad": _encode_distribution(motion.orientation_delta_rad),
        "linear_speed_mps": _encode_distribution(motion.linear_speed_mps),
        "angular_speed_radps": _encode_distribution(motion.angular_speed_radps),
    }


def _encode_distribution(summary: DistributionSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "p95": summary.p95,
        "maximum": summary.maximum,
    }


def _encode_timestamp(timestamp: SourceTimestamp) -> dict[str, Any]:
    return {
        "seconds": timestamp.seconds,
        "nanoseconds": timestamp.nanoseconds,
        "clock_id": timestamp.clock_id,
    }


def _decode_timestamp(record: Mapping[str, Any]) -> SourceTimestamp:
    return SourceTimestamp(
        seconds=record["seconds"],
        nanoseconds=record["nanoseconds"],
        clock_id=record["clock_id"],
    )
