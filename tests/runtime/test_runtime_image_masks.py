"""Loading per-camera image-preparation masks (valid region and ego exclusion) from files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from contextmap.runtime.configuration import CameraPreparationConfig
from contextmap.runtime.image_masks import (
    CameraImageMasks,
    exclusion_region_from_file,
    resolve_image_masks,
    valid_region_from_file,
)
from contextmap.visual_perception import ExclusionRegion, ValidRegion


def _write_mask(path: Path, pixels: np.ndarray) -> Path:
    np.save(path, pixels, allow_pickle=False)
    return path.with_suffix(".npy") if path.suffix != ".npy" else path


def test_valid_region_from_file_builds_region_with_pixels(tmp_path: Path) -> None:
    pixels = np.zeros((4, 6), dtype=np.bool_)
    pixels[1:3, 2:5] = True
    target = _write_mask(tmp_path / "valid", pixels)

    region = valid_region_from_file(target, reason="fisheye valid circle", source="corridor-02")

    assert isinstance(region, ValidRegion)
    assert region.reason == "fisheye valid circle"
    assert region.source == "corridor-02"
    assert region.mask.width == 6
    assert region.mask.height == 4
    assert np.array_equal(region.mask.as_array(), pixels)


def test_exclusion_region_from_file_builds_named_region(tmp_path: Path) -> None:
    pixels = np.zeros((4, 6), dtype=np.bool_)
    pixels[3, :] = True
    target = _write_mask(tmp_path / "ego", pixels)

    region = exclusion_region_from_file(
        target, name="ego", reason="recording rig", source="corridor-02"
    )

    assert isinstance(region, ExclusionRegion)
    assert region.name == "ego"
    assert region.reason == "recording rig"
    assert region.source == "corridor-02"
    assert np.array_equal(region.mask.as_array(), pixels)


def test_non_boolean_mask_is_rejected(tmp_path: Path) -> None:
    target = _write_mask(tmp_path / "gray", np.full((4, 6), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="bool"):
        exclusion_region_from_file(target, name="ego", reason="r", source="s")


def test_non_two_dimensional_mask_is_rejected(tmp_path: Path) -> None:
    target = _write_mask(tmp_path / "cube", np.zeros((2, 3, 4), dtype=np.bool_))

    with pytest.raises(ValueError, match="two-dimensional"):
        valid_region_from_file(target, reason="r", source="s")


def _bool(path: Path, h: int, w: int) -> None:
    np.save(path, np.zeros((h, w), dtype=np.bool_), allow_pickle=False)


def test_resolve_image_masks_decodes_each_camera(tmp_path: Path) -> None:
    _bool(tmp_path / "valid.npy", 4, 6)
    _bool(tmp_path / "ego.npy", 4, 6)
    config = {
        "cam-a": CameraPreparationConfig(valid_region="valid.npy", exclusions={"ego": "ego.npy"})
    }

    resolved = resolve_image_masks(config, base_dir=tmp_path)

    masks = resolved["cam-a"]
    assert isinstance(masks, CameraImageMasks)
    assert masks.valid_region is not None and masks.valid_region.mask.width == 6
    assert tuple(e.name for e in masks.exclusions) == ("ego",)


def test_constraints_for_matching_dimensions_returns_regions(tmp_path: Path) -> None:
    _bool(tmp_path / "ego.npy", 4, 6)
    config = {"cam-a": CameraPreparationConfig(valid_region=None, exclusions={"ego": "ego.npy"})}
    masks = resolve_image_masks(config, base_dir=tmp_path)["cam-a"]

    valid, exclusions = masks.constraints_for(width=6, height=4, camera="cam-a")

    assert valid is None
    assert tuple(e.name for e in exclusions) == ("ego",)


def test_constraints_for_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    _bool(tmp_path / "ego.npy", 4, 6)
    config = {"cam-a": CameraPreparationConfig(valid_region=None, exclusions={"ego": "ego.npy"})}
    masks = resolve_image_masks(config, base_dir=tmp_path)["cam-a"]

    with pytest.raises(ValueError, match=r"dimension|size|6x4|8x8"):
        masks.constraints_for(width=8, height=8, camera="cam-a")
