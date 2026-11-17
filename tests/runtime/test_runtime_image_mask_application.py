"""The perception image-preparation step applies the configured per-camera masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.runtime.configuration import CameraPreparationConfig
from contextmap.runtime.executors import ExecutorError, _materialize_prepared_image
from contextmap.runtime.image_masks import resolve_image_masks
from contextmap.shared import SourceTimestamp

_CAMERA = "cam-a"


def _observation() -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId("frame-0000"),
        sensor_id=SensorId("camera_1"),
        frame_id=FrameId("camera_1_optical"),
        timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:header"),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures"),
        calibration_id=CalibrationReferenceId(_CAMERA),
        width=2,
        height=2,
        encoding=ImageEncoding.RGB8,
        data=np.zeros((2, 2, 3), dtype=np.uint8).tobytes(),
    )


def _masks(tmp_path: Path, *, shape: tuple[int, int]) -> dict:
    ego = np.zeros(shape, dtype=np.bool_)
    ego[shape[0] - 1, :] = True
    np.save(tmp_path / "ego.npy", ego, allow_pickle=False)
    valid = np.ones(shape, dtype=np.bool_)
    np.save(tmp_path / "valid.npy", valid, allow_pickle=False)
    config = {
        _CAMERA: CameraPreparationConfig(valid_region="valid.npy", exclusions={"ego": "ego.npy"})
    }
    return dict(resolve_image_masks(config, base_dir=tmp_path))


@pytest.mark.usefixtures("fake_pillow")
def test_prepared_image_carries_the_configured_regions(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    masks = _masks(tmp_path, shape=(2, 2))

    prepared = _materialize_prepared_image(_observation(), root, masks)

    assert prepared.valid_region is not None
    assert tuple(r.name for r in prepared.exclusion_regions) == ("ego",)
    assert prepared.exclusion_regions[0].mask.value_at(0, 1) is True


@pytest.mark.usefixtures("fake_pillow")
def test_without_masks_the_image_is_unconstrained(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()

    prepared = _materialize_prepared_image(_observation(), root, {})

    assert prepared.valid_region is None
    assert prepared.exclusion_regions == ()


@pytest.mark.usefixtures("fake_pillow")
def test_dimension_mismatch_is_a_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    masks = _masks(tmp_path, shape=(4, 4))  # image is 2x2

    with pytest.raises(ExecutorError, match=r"4x4|2x2|not resampled"):
        _materialize_prepared_image(_observation(), root, masks)
