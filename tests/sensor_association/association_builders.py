"""Deterministic builders for Sensor Association tests."""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.region_perception import RegionId
from contextmap.sensor_association import (
    AssociationProvenance,
    CalibrationRef,
    DepthMetric,
    PoseRef,
    ProjectionSummary,
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    spatial_observation_id_for,
)
from contextmap.state_estimation import (
    LookupOutcome,
    PoseEstimateId,
    StateEstimationRunId,
    TrajectoryId,
)

MAP_ID = MapId("map-0001")


def refs(indexes: Sequence[int], *, map_id: MapId = MAP_ID) -> tuple[GeometryReference, ...]:
    return tuple(
        GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=i))
        for i in indexes
    )


def make_calibration_ref() -> CalibrationRef:
    return CalibrationRef(
        calibration_identity="sha256:calibration",
        camera_calibration_id=CalibrationReferenceId("camera0"),
        camera_model_kind="mei",
        camera_frame=FrameId("rgb_camera"),
    )


def make_pose_ref(*, interpolated: bool = True) -> PoseRef:
    if interpolated:
        return PoseRef(
            trajectory_id=TrajectoryId("traj"),
            state_estimation_run_id=StateEstimationRunId("run-0001"),
            source_estimate_ids=(
                PoseEstimateId("traj--pose-000007"),
                PoseEstimateId("traj--pose-000008"),
            ),
            lookup_outcome=LookupOutcome.INTERPOLATED,
            time_delta_ns=20_000_000,
            interpolation_fraction=0.25,
        )
    return PoseRef(
        trajectory_id=TrajectoryId("traj"),
        state_estimation_run_id=None,
        source_estimate_ids=(PoseEstimateId("traj--pose-000007"),),
        lookup_outcome=LookupOutcome.EXACT,
        time_delta_ns=0,
        interpolation_fraction=None,
    )


def make_provenance(*, map_id: MapId = MAP_ID) -> AssociationProvenance:
    return AssociationProvenance(
        geometric_map_id=map_id,
        association_run_id="association-run-0001",
        region_artifact_id="run-0001",
        sequence_artifact_id=SequenceArtifactId("sequence-0001"),
        visibility_policy_id="conservative-depth-support-v1",
        membership_policy_id="mask-membership-v1",
        configuration_fingerprint="sha256:association-config",
        code_version="test",
    )


def make_summary(*, considered: int = 100, visible: int = 60) -> ProjectionSummary:
    return ProjectionSummary(
        camera_model_kind="mei",
        depth_metric=DepthMetric.RAY_RANGE,
        image_transform_id="sha256:raw-to-prepared",
        prepared_image_size=(640, 480),
        considered_count=considered,
        visible_count=visible,
    )


def make_spatial_observation(
    *,
    region: str = "region-0001",
    support: Sequence[int] = (3, 5, 9),
    map_id: MapId = MAP_ID,
    occluded: int = 4,
) -> SpatialObservation:
    references = refs(support, map_id=map_id)
    return SpatialObservation(
        spatial_observation_id=spatial_observation_id_for(
            association_run_id="association-run-0001",
            source_observation_id=SourceObservationId("frame-0124"),
            region_id=RegionId(region),
        ),
        source_observation_id=SourceObservationId("frame-0124"),
        region_artifact_id="run-0001",
        region_id=RegionId(region),
        geometry_support=references,
        projection_summary=make_summary(),
        calibration_ref=make_calibration_ref(),
        pose_ref=make_pose_ref(),
        visibility=VisibilityDiagnostics(
            counts={
                VisibilityState.ASSOCIATED: len(references),
                VisibilityState.OCCLUDED: occluded,
            }
        ),
        provenance=make_provenance(map_id=map_id),
    )
