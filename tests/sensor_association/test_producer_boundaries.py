"""Association evidence owns geometry links and selects its regional producer explicitly."""

from dataclasses import fields
from pathlib import Path

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.sensor_association import (
    AssociationFrameInput,
    SpatialObservation,
    spatial_observation_id_for,
)


def test_spatial_observation_does_not_copy_foreign_evidence_references() -> None:
    """Claims and features are joined downstream through region and physical observation."""
    names = {field.name for field in fields(SpatialObservation)}
    assert "region_artifact_id" in names
    assert "perception_result_id" not in names
    assert "visual_feature_refs" not in names
    assert "semantic_claim_refs" not in names


def test_spatial_identity_includes_its_own_association_execution() -> None:
    """Repeating association creates distinct evidence while reusing the region unchanged."""
    first = spatial_observation_id_for(
        association_run_id="association-a",
        source_observation_id=SourceObservationId("frame"),
        region_id=RegionId("region"),
    )
    second = spatial_observation_id_for(
        association_run_id="association-b",
        source_observation_id=SourceObservationId("frame"),
        region_id=RegionId("region"),
    )
    assert first != second


def test_frame_input_selects_regions_without_requiring_semantic_or_visual_artifacts() -> None:
    """The geometric path has a regional input; dense sampling adds optional visual evidence."""
    names = {field.name for field in fields(AssociationFrameInput)}
    assert {"region_artifact_id", "regions", "visual_artifact_id", "visual_evidence"} <= names
    assert "perception_result" not in names


def test_dense_sampling_rejects_a_map_from_an_unselected_visual_producer() -> None:
    """Same feature metadata does not authorize consuming another producer's map."""
    from dataclasses import replace

    import pytest
    from run_builders import NATIVE, frame_input, make_request, run_collecting

    from contextmap.sensor_association import AssociationInputError

    frame = frame_input(0, channels=(NATIVE,))
    wrong = replace(frame.dense_maps[NATIVE.channel_id], source_artifact_id="unselected")
    frame = replace(frame, dense_maps={NATIVE.channel_id: wrong})
    with pytest.raises(AssociationInputError, match="visual artifact"):
        run_collecting(make_request(frames=(frame,), channels=(NATIVE,)))


def test_association_run_id_is_required_before_projection() -> None:
    """Association evidence must name the producing execution, including an empty run."""
    from dataclasses import replace

    import pytest
    from run_builders import make_request, run_collecting

    from contextmap.sensor_association import AssociationInputError

    with pytest.raises(AssociationInputError, match="association_run_id"):
        run_collecting(replace(make_request(frames=()), association_run_id=""))


def test_writer_refuses_an_outcome_from_another_association_run(tmp_path: Path) -> None:
    """A manifest cannot relabel evidence produced by a different association execution."""
    import pytest
    from run_builders import make_request, run_collecting

    from contextmap.sensor_association import (
        RunArtifactError,
        SensorAssociationRunId,
        SensorAssociationRunWriter,
    )

    outcome, _ = run_collecting(make_request(frames=()))
    writer = SensorAssociationRunWriter(
        output_dir=tmp_path / "association",
        sequence_name="sequence",
        run_id=SensorAssociationRunId("another-run"),
        run_index=1,
    )
    with (
        writer.transaction() as transaction,
        pytest.raises(RunArtifactError, match="association run"),
    ):
        transaction.finalize(outcome)


def test_sensor_api_exposes_no_semantic_or_feature_reference_contracts() -> None:
    """Foreign evidence references are owned by fusion and visual interpretation."""
    import contextmap.sensor_association as association

    assert not hasattr(association, "VisualFeatureRef")
    assert not hasattr(association, "SemanticClaimRef")
