"""Independent producer execution over shared frozen regional evidence."""

from pathlib import Path
from typing import Any, cast

import pytest
from runtime_perception import publish_region_run, stub_perception_executor, stub_semantic_executor
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.ingestion import (
    FrameId,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.artifacts import inventory_digest
from contextmap.semantic_interpretation import SemanticInterpretationRunReader
from contextmap.shared import SourceTimestamp
from contextmap.visual_perception import VisualEvidenceRunReader

pytestmark = pytest.mark.usefixtures("fake_pillow")


def _sequence(workspace: Path, *, frames: int = 2) -> ArtifactRef:
    output = workspace / "S1" / "run-0001" / "ingestion"
    with SequenceArtifactWriter(
        output_dir=output, sequence_name="S1", artifact_id=SequenceArtifactId("sequence-1")
    ) as writer:
        for index in range(frames):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index}"),
                    sensor_id=SensorId("cam"),
                    frame_id=FrameId("optical"),
                    timestamp=SourceTimestamp(seconds=index, nanoseconds=0, clock_id="fixture"),
                    provenance=SourceProvenance(source_type="fixture", source_path="images"),
                    width=2,
                    height=2,
                    encoding=ImageEncoding.RGB8,
                    data=bytes([10, 20, 30] * 4),
                )
            )
        manifest = writer.finalize()
    return ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id="sequence-1",
        content_hash=inventory_digest(manifest.file_inventory),
        location=str(output.relative_to(workspace)),
    )


def test_visual_and_semantic_execute_without_recopying_the_frozen_regions(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    regions = publish_region_run(tmp_path, sequence)
    inputs = {"sequence": (sequence,), "regions": (regions,)}
    visual_output = tmp_path / "S1" / "run-0001" / "visual_perception"
    semantic_output = tmp_path / "S1" / "run-0001" / "semantic_interpretation"
    visual = stub_perception_executor().execute(
        StageRequest(
            stage_id="visual_perception",
            inputs=inputs,
            components={},
            config_digest="sha256:visual",
            workspace=tmp_path,
            output_dir=visual_output,
        )
    )
    semantic = stub_semantic_executor().execute(
        StageRequest(
            stage_id="semantic_interpretation",
            inputs=inputs,
            components={},
            config_digest="sha256:semantic",
            workspace=tmp_path,
            output_dir=semantic_output,
        )
    )
    assert regions.location is not None
    regional = RegionPerceptionRunReader(tmp_path / regions.location)
    visual_reader = VisualEvidenceRunReader(visual_output)
    semantic_reader = SemanticInterpretationRunReader(semantic_output)
    assert visual.contract == "VisualEvidenceArtifact"
    assert semantic.contract == "SemanticInterpretationArtifact"
    assert visual_reader.manifest.region_artifact_id == regional.manifest.run_id
    assert semantic_reader.manifest.region_artifact_id == regional.manifest.run_id
    assert len(visual_reader.list_results()) == 2
    assert len(semantic_reader.list_observations()) == 2
    regional_ids = {region.region_id for region in regional.read_regions()}
    assert {
        feature.region_id
        for result in visual_reader.iter_results()
        for feature in result.features
        if feature.region_id is not None
    } == regional_ids
    assert not (visual_output / "outputs" / "masks").exists()
    assert not (semantic_output / "outputs" / "masks").exists()
    assert visual_reader.verify_integrity() == []
    assert semantic_reader.verify_integrity() == []


def test_association_selection_refuses_two_runs_over_the_same_region() -> None:
    from types import SimpleNamespace

    import pytest

    from contextmap.runtime.executors import ExecutorError, _check_one_association_per_region

    first = SimpleNamespace(
        manifest=SimpleNamespace(run_id="association-a", region_artifact_ids=("regions-1",))
    )
    second = SimpleNamespace(
        manifest=SimpleNamespace(run_id="association-b", region_artifact_ids=("regions-1",))
    )
    with pytest.raises(ExecutorError, match="one association per regional"):
        _check_one_association_per_region(cast(Any, [first, second]), {"regions-1"})


def test_stage_major_releases_each_feature_backend_once(tmp_path: Path) -> None:
    from runtime_perception import _FakeFeatureExtractor

    from contextmap.runtime.executors import PerceptionSchedule, VisualPerceptionExecutor
    from contextmap.visual_perception import FeatureScope

    released: list[str] = []

    class Features(_FakeFeatureExtractor):
        def release(self) -> None:
            released.append(self._scope.value)

    sequence = _sequence(tmp_path)
    regions = publish_region_run(tmp_path, sequence)
    VisualPerceptionExecutor(
        dense_features=lambda scope: Features(FeatureScope.DENSE, str(scope.run_id)),
        region_features=lambda scope: Features(FeatureScope.REGION, str(scope.run_id)),
        perception_schedule=PerceptionSchedule.STAGE_MAJOR,
    ).execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence,), "regions": (regions,)},
            components={},
            config_digest="sha256:stage-major",
            workspace=tmp_path,
            output_dir=tmp_path / "S1" / "run-0001" / "visual_perception",
        )
    )
    assert released == ["dense", "region"]


def test_failed_regional_source_skips_region_features_but_keeps_dense_evidence(
    tmp_path: Path,
) -> None:
    import json

    from runtime_perception import _FakeRegionDiscovery

    class FailedDiscovery(_FakeRegionDiscovery):
        def discover(self, image: object) -> list[Any]:
            raise RuntimeError("discovery unavailable")

    sequence = _sequence(tmp_path)
    regions = publish_region_run(tmp_path, sequence, discovery=FailedDiscovery())
    output = tmp_path / "S1" / "run-0001" / "visual_perception"
    stub_perception_executor().execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence,), "regions": (regions,)},
            components={},
            config_digest="sha256:failed-source",
            workspace=tmp_path,
            output_dir=output,
        )
    )
    statuses = [
        (row["stage_id"], row["status"])
        for row in (
            json.loads(line)
            for line in (output / "metrics" / "stage-timings.jsonl").read_text().splitlines()
        )
    ]
    assert ("region_feature_extraction", "skipped") in statuses
    assert ("dense_feature_extraction", "succeeded") in statuses


def test_dense_payload_cache_retains_at_most_one_physical_observation(tmp_path: Path) -> None:
    import gc
    import weakref
    from dataclasses import replace

    import numpy as np
    from runtime_perception import _FakeFeatureExtractor

    from contextmap.runtime.executors import _DeferredFeaturePayloadSink
    from contextmap.visual_perception import (
        FeatureId,
        FeatureScope,
        PreparedImage,
        VisualEvidenceRunWriter,
    )

    sink = _DeferredFeaturePayloadSink()
    sink.bind(
        VisualEvidenceRunWriter(
            output_dir=tmp_path / "visual",
            run_id="visual",
            sequence_artifact_id="sequence",
            region_artifact_id="regions",
            selection_id="selection",
            configuration_digest="config",
        )
    )
    first = _FakeFeatureExtractor(FeatureScope.DENSE).extract(
        PreparedImage(
            source_observation_id=SourceObservationId("frame-1"),
            payload_reference="image",
            width=2,
            height=2,
        )
    )[0]
    array = np.zeros((2,), dtype=np.float32)
    retained = weakref.ref(array)
    sink.add_feature_payload(first, first.source_observation_id, array)
    del array
    second = replace(
        first,
        feature_id=FeatureId("second"),
        payload_reference="second.npy",
        source_observation_id=SourceObservationId("frame-2"),
    )
    sink.add_feature_payload(second, second.source_observation_id, np.ones((2,), dtype=np.float32))
    gc.collect()
    assert retained() is None
    np.testing.assert_array_equal(
        sink.current_payload(first.source_observation_id, first.feature_id),
        np.zeros((2,), dtype=np.float32),
    )
