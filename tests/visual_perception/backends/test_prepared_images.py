"""Unit tests of how a discovery pass is materialized from a run's prepared image (#661).

Pillow is not a dependency of the project, so ``PIL.Image`` is a small fake here that decodes
an ``np.save`` payload: what is under test is which bytes are read and verified, which window is
cut and what is refused, not image decoding.
"""

from __future__ import annotations

import hashlib
import io
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox,
    DiscoveryInput,
    DiscoveryPass,
    DiscoveryPassConfig,
    PassKind,
    TilingConfig,
)
from contextmap.region_perception.adapters.prepared_images import (
    materialize_discovery_pass,
    read_prepared_image,
    require_unscaled_passes,
)
from contextmap.region_perception.discovery.region_discovery import (
    validate_materialized_discovery_image,
)
from contextmap.visual_perception import ArtifactReference, PreparedImage

REFERENCE = "frame-0001.png"


class _Image:
    """The part of a Pillow image the materializer uses, over an HWC ``uint8`` array."""

    def __init__(self, pixels: np.ndarray) -> None:
        self.pixels = pixels
        self.mode = "RGB" if pixels.ndim == 3 else "L"

    def __enter__(self) -> _Image:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    @property
    def size(self) -> tuple[int, int]:
        return int(self.pixels.shape[1]), int(self.pixels.shape[0])

    def convert(self, mode: str) -> _Image:
        assert mode == "RGB"
        if self.mode == "L":
            return _Image(np.repeat(self.pixels[:, :, None], 3, axis=2))
        return _Image(self.pixels.copy())

    def crop(self, box: tuple[int, int, int, int]) -> _Image:
        assert all(isinstance(value, int) for value in box), box
        left, upper, right, lower = box
        return _Image(self.pixels[upper:lower, left:right])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.pixels


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Install the fake ``PIL.Image``; the list records every payload it decoded."""
    decoded: list[bytes] = []

    def open_image(source: io.BytesIO) -> _Image:
        payload = source.getvalue()
        decoded.append(payload)
        return _Image(np.load(io.BytesIO(payload)))

    module = types.ModuleType("PIL.Image")
    module.__dict__["open"] = open_image
    monkeypatch.setitem(sys.modules, "PIL.Image", module)
    return decoded


def _frame(width: int = 6, height: int = 4) -> np.ndarray:
    """A frame whose every pixel names its own position: ``(x, y, 7)``."""
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack((columns, rows, np.full_like(rows, 7)), axis=-1).astype(np.uint8)


def _prepared(root: Path, pixels: np.ndarray, *, recorded: bytes | None = None) -> PreparedImage:
    """Write ``pixels`` under ``root`` and describe them, with the SHA-256 of ``recorded``."""
    buffer = io.BytesIO()
    np.save(buffer, pixels)
    payload = buffer.getvalue()
    (root / REFERENCE).write_bytes(payload)
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference=REFERENCE,
        payload_artifact=ArtifactReference(
            uri=REFERENCE,
            sha256=hashlib.sha256(payload if recorded is None else recorded).hexdigest(),
            media_type="image/png",
        ),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
    )


def _input(
    image: PreparedImage, window: BoundingBox | None = None, scale: float = 1.0
) -> DiscoveryInput:
    full = BoundingBox(x_min=0, y_min=0, x_max=image.width, y_max=image.height)
    return DiscoveryInput(
        prepared_image=image,
        discovery_pass=DiscoveryPass(
            pass_id="full-frame" if window is None else "tile-0000",
            kind=PassKind.FULL_FRAME if window is None else PassKind.TILE,
            window=window or full,
            scale=scale,
        ),
        perception_run_id="run-0001",
        perception_result_id="run-0001--frame-0001",
    )


def test_a_full_frame_pass_is_the_verified_prepared_image_as_rgb(
    tmp_path: Path, opened: list[bytes]
) -> None:
    pixels = _frame()
    discovery_input = _input(_prepared(tmp_path, pixels))

    image = materialize_discovery_pass(tmp_path, discovery_input)

    validate_materialized_discovery_image(image, discovery_input)
    assert np.array_equal(np.asarray(image), pixels)
    assert opened == [(tmp_path / REFERENCE).read_bytes()]


def test_a_tile_pass_is_exactly_its_window_of_the_prepared_image(
    tmp_path: Path, opened: list[bytes]
) -> None:
    pixels = _frame()
    window = BoundingBox(x_min=2, y_min=1, x_max=5, y_max=4)
    discovery_input = _input(_prepared(tmp_path, pixels), window)

    image = materialize_discovery_pass(tmp_path, discovery_input)

    validate_materialized_discovery_image(image, discovery_input)
    assert np.array_equal(np.asarray(image), pixels[1:4, 2:5])


def test_a_grayscale_prepared_image_is_presented_as_rgb(
    tmp_path: Path, opened: list[bytes]
) -> None:
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)

    image = materialize_discovery_pass(tmp_path, _input(_prepared(tmp_path, gray)))

    assert np.array_equal(np.asarray(image), np.repeat(gray[:, :, None], 3, axis=2))


def test_a_scaled_pass_is_refused_before_anything_is_read(
    tmp_path: Path, opened: list[bytes]
) -> None:
    """No resampling rule for discovery passes is defined, so none is applied silently."""
    image = _prepared(tmp_path, _frame())

    with pytest.raises(ValueError, match="no resampling rule"):
        materialize_discovery_pass(tmp_path, _input(image, scale=0.5))
    assert opened == []


def test_a_pass_plan_with_a_scaled_tiling_is_refused_before_any_pass() -> None:
    """The same refusal, over the plan, so a run never fails every frame for it."""
    unscaled = TilingConfig(tile_width=64, tile_height=64)
    scaled = TilingConfig(tile_width=64, tile_height=64, scale=0.5)

    require_unscaled_passes(None)
    require_unscaled_passes(DiscoveryPassConfig(tiling=unscaled))
    with pytest.raises(ValueError, match=r"scale 0\.5 cannot be materialized"):
        require_unscaled_passes(DiscoveryPassConfig(tiling=unscaled, additional_tilings=(scaled,)))


def test_a_payload_that_is_not_the_one_recorded_is_never_decoded(
    tmp_path: Path, opened: list[bytes]
) -> None:
    image = _prepared(tmp_path, _frame(), recorded=b"the bytes the run recorded")

    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        materialize_discovery_pass(tmp_path, _input(image))
    assert opened == []


def test_a_payload_that_decodes_to_other_dimensions_is_refused(
    tmp_path: Path, opened: list[bytes]
) -> None:
    image = _prepared(tmp_path, _frame())
    lying = PreparedImage(
        source_observation_id=image.source_observation_id,
        payload_reference=image.payload_reference,
        payload_artifact=image.payload_artifact,
        width=image.width + 1,
        height=image.height,
    )

    with pytest.raises(ValueError, match="decodes to"):
        materialize_discovery_pass(tmp_path, _input(lying, BoundingBox(0, 0, 6, 4)))


def test_a_window_outside_the_prepared_image_is_refused(
    tmp_path: Path, opened: list[bytes]
) -> None:
    image = _prepared(tmp_path, _frame())

    with pytest.raises(ValueError, match="leaves the 6x4 prepared image"):
        materialize_discovery_pass(tmp_path, _input(image, BoundingBox(3, 0, 8, 4)))
    assert opened == []


@pytest.mark.parametrize("reference", ["/etc/frame.png", "../frame.png"])
def test_a_reference_outside_the_root_is_refused_before_reading(
    tmp_path: Path, reference: str
) -> None:
    image = PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference=reference,
        payload_artifact=ArtifactReference(uri=reference, sha256="0" * 64, media_type="image/png"),
        width=2,
        height=2,
    )

    with pytest.raises(ValueError, match="outside the prepared-image root"):
        read_prepared_image(tmp_path, image)


def test_a_prepared_image_without_a_recorded_sha256_cannot_be_verified(tmp_path: Path) -> None:
    image = PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference=REFERENCE,
        width=2,
        height=2,
    )

    with pytest.raises(ValueError, match="no recorded sha256"):
        read_prepared_image(tmp_path, image)
