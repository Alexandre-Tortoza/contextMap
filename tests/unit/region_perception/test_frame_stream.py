"""Per-frame preparation metadata distinguishes empty evidence from failure."""

from pathlib import Path

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    RegionPerceptionRunError,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
)
from contextmap.visual_perception import PreparedImage

from .test_artifacts import _manifest


def _image(observation: str = "obs_001") -> PreparedImage:
    """An addressed physical frame with no temporary payload location."""
    return PreparedImage(
        source_observation_id=SourceObservationId(observation),
        payload_reference=f"source-observation:{observation}",
        width=10,
        height=20,
    )


def test_frames_roundtrip_even_when_they_have_no_regions(tmp_path: Path) -> None:
    """Successful empty and failed discovery remain separate auditable outcomes."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest(source_observation_ids=("obs_001", "obs_002")))
    writer.write_frame(_image())
    writer.write_frame(_image("obs_002"), outcome="failed", failure_detail="invalid projection")
    writer.finalize()
    reader = RegionPerceptionRunReader(tmp_path / "run")
    frames = reader.read_frames()
    assert len(frames) == 2
    assert frames[0].image == _image()
    assert frames[0].outcome == "succeeded"
    assert frames[0].failure_detail is None
    assert frames[1].outcome == "failed"
    assert frames[1].failure_detail == "invalid projection"
    assert reader.verify_integrity() == []


def test_duplicate_frame_is_rejected(tmp_path: Path) -> None:
    """One producer may emit only one outcome per selected physical observation."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_frame(_image())
    with pytest.raises(RegionPerceptionRunError, match="duplicate"):
        writer.write_frame(_image())


def test_failed_frame_requires_explicit_failure_detail(tmp_path: Path) -> None:
    """Failure cannot be recorded as indistinguishable absence."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    with pytest.raises(ValueError, match="failure_detail"):
        writer.write_frame(_image(), outcome="failed")


def test_constraints_are_externalized_and_roundtrip(tmp_path: Path) -> None:
    """Preparation constraints remain auditable without inline mask pixels."""
    import json
    from dataclasses import replace

    import numpy as np

    from contextmap.region_perception import InlineMask
    from contextmap.visual_perception import ValidRegion

    image = replace(
        _image(),
        valid_region=ValidRegion(
            mask=InlineMask(np.ones((20, 10), dtype=bool)),
            reason="camera field",
            source="calibration",
        ),
    )
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_frame(image)
    writer.finalize()
    record = json.loads((tmp_path / "run/outputs/frames.jsonl").read_text())
    assert "mask" not in record["image"]["valid_region"]
    assert record["image"]["valid_region"]["mask_reference"].startswith("outputs/masks/")
    restored = RegionPerceptionRunReader(tmp_path / "run").read_frames()[0].image
    assert image.valid_region is not None
    assert restored.valid_region is not None
    assert restored.valid_region.mask == image.valid_region.mask


def test_finalize_refuses_missing_selected_frame_outcome(tmp_path: Path) -> None:
    """A selected frame must be accounted for even when discovery produced nothing."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    with pytest.raises(RegionPerceptionRunError, match="frame coverage"):
        writer.finalize()


def test_regions_cannot_be_added_after_finalization(tmp_path: Path) -> None:
    """The producer cannot mutate its in-memory boundary after publishing it."""
    from .test_artifacts import _region

    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_frame(_image())
    writer.finalize()
    with pytest.raises(RegionPerceptionRunError, match="finalize"):
        writer.write_region(_region("new"))


def test_frames_can_be_streamed_and_loaded_by_physical_observation(tmp_path: Path) -> None:
    """Consumers select one physical frame without materializing the full frame stream."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest(source_observation_ids=("obs_001", "obs_002")))
    writer.write_frame(_image())
    writer.write_frame(_image("obs_002"))
    writer.finalize()
    reader = RegionPerceptionRunReader(tmp_path / "run")
    stream = reader.iter_frames()
    assert iter(stream) is stream
    assert next(stream).image == _image()
    assert reader.frame(SourceObservationId("obs_002")).image == _image("obs_002")
    with pytest.raises(RegionPerceptionRunError, match="selected frame"):
        reader.frame(SourceObservationId("unselected"))
