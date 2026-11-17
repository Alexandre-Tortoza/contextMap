import dataclasses
import random
from collections.abc import Iterable, Sequence

import pytest
from fusion_builders import frame_timestamp, make_interpreter, spatial_id
from observation_builders import (
    make_perception_run,
    make_spatial_observation,
    timestamps,
)

from contextmap.region_perception import RegionPerceptionRunManifest
from contextmap.semantic_fusion import (
    PHYSICAL_OBSERVATION_GROUPING_POLICY_ID,
    PhysicalObservationGrouping,
    group_by_physical_observation,
)
from contextmap.sensor_association import SpatialObservation, SpatialObservationId

THREE_RUNS = ("run-a", "run-b", "run-c")


def _group(
    observations: Iterable[SpatialObservation],
    *,
    runs: Sequence[RegionPerceptionRunManifest] | None = None,
    frames: Iterable[str] = ("frame-0120", "frame-0121"),
) -> PhysicalObservationGrouping:
    return group_by_physical_observation(
        observations,
        selected_region_runs=[make_perception_run(run) for run in THREE_RUNS]
        if runs is None
        else runs,
        acquisition_timestamps=timestamps(frames),
    )


def test_one_frame_interpreted_once_is_one_physical_observation_and_one_inference() -> None:
    grouping = _group([make_spatial_observation("run-a", "frame-0120")])

    assert grouping.physical_observation_count == 1
    assert grouping.inference_result_count == 1
    assert grouping.perception_run_count == 1
    group = grouping.groups[0]
    assert group.physical_observation_id == "frame-0120"
    assert group.acquisition_timestamp == frame_timestamp("frame-0120")
    assert group.spatial_observation_ids == (spatial_id("run-a", "frame-0120"),)
    assert group.semantic_interpretation_artifact_ids == ()
    assert group.region_artifact_ids == ("run-a",)


def test_three_runs_over_one_frame_stay_one_physical_observation() -> None:
    grouping = _group([make_spatial_observation(run, "frame-0120") for run in THREE_RUNS])

    assert grouping.physical_observation_count == 1
    assert grouping.inference_result_count == 3
    assert grouping.perception_run_count == 3
    group = grouping.groups[0]
    assert group.inference_result_count == 3
    assert group.region_artifact_ids == THREE_RUNS
    assert group.semantic_interpretation_artifact_ids == ()


def test_regions_of_one_inference_result_are_one_inference_result() -> None:
    grouping = _group(
        [
            make_spatial_observation("run-a", "frame-0120", "region-0001"),
            make_spatial_observation("run-a", "frame-0120", "region-0002"),
        ]
    )

    assert grouping.physical_observation_count == 1
    assert grouping.inference_result_count == 1
    assert len(grouping.groups[0].spatial_observation_ids) == 2


def test_evidence_from_separate_frames_remains_separate_physical_observations() -> None:
    grouping = _group(
        [
            make_spatial_observation("run-a", "frame-0121"),
            make_spatial_observation("run-a", "frame-0120"),
        ]
    )

    assert grouping.physical_observation_count == 2
    assert grouping.inference_result_count == 2
    assert grouping.perception_run_count == 1
    assert [group.physical_observation_id for group in grouping.groups] == [
        "frame-0120",
        "frame-0121",
    ]


def test_repeated_inference_is_kept_and_not_turned_into_independent_votes() -> None:
    grouping = _group(
        [make_spatial_observation(run, "frame-0120") for run in THREE_RUNS]
        + [make_spatial_observation("run-a", "frame-0121")]
    )

    counts = {
        group.physical_observation_id: group.inference_result_count for group in grouping.groups
    }
    assert counts == {"frame-0120": 3, "frame-0121": 1}
    assert grouping.physical_observation_count == 2
    assert grouping.inference_result_count == 4


def test_the_grouping_does_not_depend_on_the_order_of_the_inputs() -> None:
    observations = [
        make_spatial_observation(run, frame, region)
        for run in THREE_RUNS
        for frame in ("frame-0120", "frame-0121")
        for region in ("region-0001", "region-0002")
    ]
    expected = _group(observations)

    for seed in range(5):
        shuffled = list(observations)
        random.Random(seed).shuffle(shuffled)
        assert _group(shuffled) == expected
    reversed_runs = [make_perception_run(run) for run in reversed(THREE_RUNS)]
    assert _group(reversed(observations), runs=reversed_runs) == expected


def test_runs_with_the_same_backend_are_repeated_inference_and_not_new_variants() -> None:
    other = dataclasses.replace(make_interpreter(), model="gemini-3-flash")
    runs = [
        make_perception_run("run-a"),
        make_perception_run("run-b"),
        make_perception_run("run-c", interpreter=other),
    ]

    grouping = _group([make_spatial_observation(r.run_id, "frame-0120") for r in runs], runs=runs)

    assert grouping.perception_run_count == 3
    assert grouping.inference_variant_count == 2


def test_a_backend_with_another_configuration_is_another_variant() -> None:
    tuned = dataclasses.replace(make_interpreter(), configuration_fingerprint="sha256:tuned")
    runs = [make_perception_run("run-a"), make_perception_run("run-b", interpreter=tuned)]

    grouping = _group([make_spatial_observation(r.run_id, "frame-0120") for r in runs], runs=runs)

    assert grouping.inference_variant_count == 2


def test_the_grouping_names_its_policy_and_the_runs_it_was_given() -> None:
    grouping = _group([make_spatial_observation("run-b", "frame-0120")])

    assert grouping.grouping_policy_id == PHYSICAL_OBSERVATION_GROUPING_POLICY_ID
    assert grouping.selected_region_artifact_ids == THREE_RUNS
    assert grouping.perception_run_count == 1


def test_no_observations_is_an_empty_grouping() -> None:
    grouping = _group([])

    assert grouping.groups == ()
    assert grouping.physical_observation_count == 0
    assert grouping.inference_variant_count == 0


def test_an_observation_from_a_run_that_was_not_selected_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"run-b.*not among the selected"):
        _group(
            [make_spatial_observation("run-a"), make_spatial_observation("run-b")],
            runs=[make_perception_run("run-a")],
        )


def test_a_selection_must_name_at_least_one_run() -> None:
    with pytest.raises(ValueError, match="at least one perception run"):
        _group([], runs=[])


def test_a_run_selected_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="selected more than once"):
        _group([], runs=[make_perception_run("run-a"), make_perception_run("run-a")])


def test_runs_over_different_sequences_cannot_share_a_grouping() -> None:
    runs = [make_perception_run("run-a"), make_perception_run("run-b", sequence="sequence-0002")]

    with pytest.raises(ValueError, match="one sequence artifact"):
        _group([], runs=runs)


def test_an_observation_from_another_sequence_than_its_run_is_rejected() -> None:
    with pytest.raises(ValueError, match="sequence-0002"):
        _group([make_spatial_observation("run-a", sequence="sequence-0002")])


def test_the_same_spatial_observation_twice_is_rejected() -> None:
    observation = make_spatial_observation("run-a", "frame-0120")

    with pytest.raises(ValueError, match="duplicate spatial observation"):
        _group([observation, observation])


def test_alternative_associations_for_same_regional_evidence_are_ambiguous() -> None:
    first = make_spatial_observation("run-a", "frame-0120")
    second = dataclasses.replace(
        first,
        spatial_observation_id=SpatialObservationId("alternative-association"),
        provenance=dataclasses.replace(first.provenance, association_run_id="alternative"),
    )
    with pytest.raises(ValueError, match="ambiguous association"):
        _group([first, second])


def test_a_regional_producer_describes_separate_physical_observations() -> None:
    grouped = _group(
        [
            make_spatial_observation("run-a", "frame-0120"),
            make_spatial_observation("run-a", "frame-0121"),
        ]
    )
    assert grouped.physical_observation_count == 2


def test_a_frame_without_an_acquisition_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"no acquisition timestamp.*frame-0122"):
        _group([make_spatial_observation("run-a", "frame-0122")])


def test_the_grouping_rejects_groups_from_runs_it_was_not_given() -> None:
    valid = _group([make_spatial_observation("run-a", "frame-0120")])

    with pytest.raises(ValueError, match="selected_region_artifact_ids"):
        dataclasses.replace(valid, selected_region_artifact_ids=("run-z",))


def test_the_grouping_keeps_groups_sorted_by_physical_observation() -> None:
    valid = _group(
        [
            make_spatial_observation("run-a", "frame-0120"),
            make_spatial_observation("run-a", "frame-0121"),
        ]
    )

    with pytest.raises(ValueError, match="groups must be sorted and unique"):
        dataclasses.replace(valid, groups=tuple(reversed(valid.groups)))
