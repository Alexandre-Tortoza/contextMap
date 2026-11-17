import dataclasses

import pytest
from projection_builders import SEQUENCE_ID, TRAJECTORY_ID, make_lookup
from run_builders import (
    ENHANCED,
    NATIVE,
    OCCLUSION,
    CollectingSink,
    frame_id,
    frame_input,
    make_request,
    run_collecting,
)

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import SequenceArtifactId, SourceObservationId
from contextmap.sensor_association import VisibilityState
from contextmap.sensor_association.dense_sampling import InterpolationPolicy
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.projection_chain.frame_projection import RejectedProjection
from contextmap.sensor_association.service import DenseChannel, SensorAssociationService
from contextmap.state_estimation import LookupRejection

SERVICE = SensorAssociationService()


def test_request_sequence_must_match_the_geometry_and_trajectory() -> None:
    request = dataclasses.replace(
        make_request(), sequence_artifact_id=SequenceArtifactId("other-sequence")
    )

    sink = CollectingSink()
    with pytest.raises(AssociationInputError, match="other-sequence"):
        SERVICE.run(request, sink=sink)
    assert sink.frames == []


def test_the_trajectory_sequence_is_checked_on_its_own_and_not_only_through_the_map() -> None:
    request = make_request()
    trajectory = request.trajectory.trajectory
    other = dataclasses.replace(
        trajectory,
        provenance=dataclasses.replace(
            trajectory.provenance, sequence_artifact_id=SequenceArtifactId("other-sequence")
        ),
    )

    sink = CollectingSink()
    with pytest.raises(AssociationInputError, match="trajectory comes from sequence"):
        SERVICE.run(
            dataclasses.replace(request, trajectory=make_lookup(other)),
            sink=sink,
        )
    assert sink.frames == []


@pytest.mark.parametrize("time_ns", [0, 10_000_000_000], ids=["accepted", "rejected-pose"])
def test_regional_selection_is_required_before_pose_lookup(time_ns: int) -> None:
    frame = frame_input(0, time_ns=time_ns)
    frame = dataclasses.replace(frame, region_artifact_id="")

    sink = CollectingSink()
    with pytest.raises(AssociationInputError, match="region_artifact_id"):
        SERVICE.run(make_request(frames=[frame]), sink=sink)
    assert sink.frames == []


def test_region_observation_mismatch_is_not_hidden_by_a_rejected_pose() -> None:
    frame = frame_input(0, time_ns=10_000_000_000)
    frame = dataclasses.replace(
        frame,
        regions=(
            dataclasses.replace(
                frame.regions[0], source_observation_id=SourceObservationId("other-frame")
            ),
            *frame.regions[1:],
        ),
    )

    sink = CollectingSink()
    with pytest.raises(AssociationInputError, match="region source observation"):
        SERVICE.run(make_request(frames=[frame]), sink=sink)
    assert sink.frames == []


def test_every_frame_is_associated_with_its_observations_quality_and_diagnostics() -> None:
    outcome, frames = run_collecting(make_request())

    assert outcome.frame_count == 2
    assert [frame.source_observation_id for frame in frames] == [frame_id(0), frame_id(1)]
    first = frames[0]
    assert [o.region_id for o in first.observations] == ["region-A", "region-B"]
    assert [q.spatial_observation_id for q in first.qualities] == [
        o.spatial_observation_id for o in first.observations
    ]
    assert first.diagnostics.source_observation_id == first.source_observation_id
    assert first.diagnostics.membership is not None
    # A região A vê o ponto da frente e o oculto atrás dele.
    observation_a = first.observations[0]
    assert observation_a.visibility.counts[VisibilityState.OCCLUDED] == 1
    assert len(observation_a.geometry_support) == 2


def test_the_observations_of_a_frame_record_the_run_configuration() -> None:
    outcome, frames = run_collecting(make_request())

    provenance = frames[0].observations[0].provenance
    assert provenance.configuration_fingerprint == outcome.configuration_fingerprint
    assert provenance.code_version == "test"
    assert provenance.visibility_policy_id == OCCLUSION.policy_id


def test_the_outcome_names_the_upstream_artifacts_the_run_consumed() -> None:
    outcome, _ = run_collecting(make_request())

    assert outcome.geometric_map.map_id == MapId("map-0001")
    assert outcome.sequence_artifact_id == SEQUENCE_ID
    assert outcome.selection_id == "full-sequence"
    assert outcome.trajectory_id == TRAJECTORY_ID
    assert outcome.calibration_identity.startswith("sha256:")
    assert outcome.region_artifact_ids == ("run-0001",)
    assert outcome.occlusion_policy == OCCLUSION


def test_a_frame_whose_pose_is_rejected_is_reported_and_the_run_continues() -> None:
    late = frame_input(1, time_ns=10_000_000_000)

    outcome, frames = run_collecting(make_request(frames=[frame_input(0), late]))

    assert [frame.source_observation_id for frame in frames] == [frame_id(0)]
    (rejected,) = outcome.rejected
    assert isinstance(rejected, RejectedProjection)
    assert rejected.source_observation_id == frame_id(1)
    assert rejected.rejection is LookupRejection.OUT_OF_RANGE


def test_native_and_enhanced_dense_channels_stay_distinct_evidence() -> None:
    outcome, frames = run_collecting(make_request(channels=[NATIVE, ENHANCED]))

    samples = frames[0].dense_samples
    assert set(samples) == {"dino-native", "dino-enhanced"}
    native, enhanced = samples["dino-native"], samples["dino-enhanced"]
    assert native.provenance.enhancement is None
    assert enhanced.provenance.enhancement is not None
    assert native.provenance.interpolation is InterpolationPolicy.NEAREST
    assert enhanced.provenance.interpolation is InterpolationPolicy.BILINEAR
    assert native.provenance.feature_id != enhanced.provenance.feature_id
    assert outcome.dense_channels == (NATIVE, ENHANCED)


def test_a_run_with_no_dense_channel_samples_no_feature() -> None:
    outcome, frames = run_collecting(make_request())

    assert frames[0].dense_samples == {}
    assert outcome.dense_channels == ()


def test_a_frame_without_the_dense_map_of_a_declared_channel_is_rejected() -> None:
    incomplete = dataclasses.replace(frame_input(0, channels=[NATIVE]), dense_maps={})

    with pytest.raises(AssociationInputError, match="dino-native"):
        run_collecting(make_request(frames=[incomplete], channels=[NATIVE]))


def test_channel_identities_must_be_unique() -> None:
    with pytest.raises(AssociationInputError, match="channel"):
        run_collecting(make_request(channels=[NATIVE, NATIVE]))


def test_a_frame_cannot_appear_twice() -> None:
    with pytest.raises(AssociationInputError, match="frame"):
        run_collecting(make_request(frames=[frame_input(0), frame_input(0)]))


def test_a_trusted_reference_gives_the_quality_and_the_diagnostics_a_residual() -> None:
    _, frames = run_collecting(make_request(frames=[frame_input(0, with_reference=True)]))

    frame = frames[0]
    assert frame.diagnostics.reprojection is not None
    assert frame.diagnostics.reprojection.median_px == pytest.approx(1.0)
    assert all(q.reprojection == frame.diagnostics.reprojection for q in frame.qualities)


def test_the_configuration_fingerprint_is_deterministic_and_tracks_the_configuration() -> None:
    base = run_collecting(make_request())[0].configuration_fingerprint

    assert base == run_collecting(make_request())[0].configuration_fingerprint
    assert base.startswith("sha256:")
    wider = dataclasses.replace(OCCLUSION, neighborhood_radius_cells=3)
    assert base != run_collecting(make_request(occlusion=wider))[0].configuration_fingerprint
    assert base != run_collecting(make_request(channels=[NATIVE]))[0].configuration_fingerprint
    bilinear = DenseChannel(channel_id="dino-native", interpolation=InterpolationPolicy.BILINEAR)
    assert (
        run_collecting(make_request(channels=[NATIVE]))[0].configuration_fingerprint
        != run_collecting(make_request(channels=[bilinear]))[0].configuration_fingerprint
    )
