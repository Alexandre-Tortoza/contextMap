"""Build spatial observations from region membership with explicit producer provenance."""

from __future__ import annotations

from contextmap.sensor_association.membership.models import FrameMembership
from contextmap.sensor_association.models import (
    AssociationProvenance,
    ProjectionSummary,
    SpatialObservation,
    VisibilityDiagnostics,
    VisibilityState,
    spatial_observation_id_for,
)


def build_spatial_observations(
    membership: FrameMembership,
    *,
    association_run_id: str,
    configuration_fingerprint: str | None,
    code_version: str | None,
) -> tuple[SpatialObservation, ...]:
    """Turn the evaluated regions into canonical spatial observations.

    Each observation carries references to the geometry, the region's features and its
    claims, plus the calibration, pose and policies used; nothing is copied into it.

    Features a region refers to are its own region-scoped features and the dense feature
    maps of the same result (which can be sampled over the region); global features and
    scene-level claims describe the whole image, not the region, and are left out.

    Args:
        membership: The membership of one frame.
        association_run_id: Execution producing these spatial observations.
        configuration_fingerprint: Hash of the association configuration, when known.
        code_version: Code revision that produced the observations, when known.

    Returns:
        One observation per evaluated region, in the regional selection's order.
    """
    resolution = membership.resolution
    frame = membership.frame
    summary = ProjectionSummary(
        camera_model_kind=frame.camera.camera_model_kind,
        depth_metric=resolution.depth_metric,
        image_transform_id=frame.image_transform.transform_id,
        prepared_image_size=frame.image_transform.prepared_size,
        considered_count=len(resolution.visible),
        visible_count=resolution.visible_count,
    )
    observations: list[SpatialObservation] = []
    for region in membership.regions:
        # A proveniência registra a regra aplicada à região, não a política configurada do run.
        provenance = AssociationProvenance(
            geometric_map_id=frame.map_id,
            association_run_id=association_run_id,
            region_artifact_id=membership.region_artifact_id,
            sequence_artifact_id=membership.sequence_artifact_id,
            visibility_policy_id=resolution.policy.policy_id,
            membership_policy_id=region.membership_rule_id,
            configuration_fingerprint=configuration_fingerprint,
            code_version=code_version,
        )
        counts = {
            VisibilityState.ASSOCIATED: region.associated_count,
            VisibilityState.OCCLUDED: region.occluded_count,
            VisibilityState.OUTSIDE_VALID_SUPPORT: region.outside_valid_support_count,
        }
        observations.append(
            SpatialObservation(
                spatial_observation_id=spatial_observation_id_for(
                    association_run_id=association_run_id,
                    source_observation_id=frame.source_observation_id,
                    region_id=region.region_id,
                ),
                source_observation_id=frame.source_observation_id,
                region_artifact_id=membership.region_artifact_id,
                region_id=region.region_id,
                geometry_support=tuple(
                    frame.map_reference(int(index)) for index in region.associated_indices
                ),
                projection_summary=summary,
                calibration_ref=frame.calibration_ref,
                pose_ref=frame.pose_ref,
                visibility=VisibilityDiagnostics(counts=counts),
                provenance=provenance,
            )
        )
    return tuple(observations)
