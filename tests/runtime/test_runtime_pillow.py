"""The fake ``PIL.Image`` of the executor tests, and what only the real Pillow can cover (#638).

The perception executor tests run on :mod:`runtime_pillow`, so the CI, which installs no Pillow,
executes them. These tests keep that fake honest: it refuses every image API misuse real Pillow
refuses, within what the executor does, and models the operations the executor performs.

Two tests need the real Pillow and are skipped, explicitly, without it: one runs the same misuse
cases against it, the other covers what the fake never does, encoding the prepared image as a
PNG.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from runtime_pillow import fake_pillow_module

_NEEDS_PILLOW = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None,
    reason="needs the real Pillow, which the base and dev installs do not include",
)

# O que o executor faz com a API de imagem, usado errado de formas que o Pillow real recusa.
_MISUSES: dict[str, Callable[[Any], object]] = {
    "fromarray-2d-as-rgb": lambda image: image.fromarray(np.zeros((2, 2), np.uint8), mode="RGB"),
    "fromarray-rgb-as-l": lambda image: image.fromarray(np.zeros((2, 2, 3), np.uint8), mode="L"),
    "fromarray-8-bit-as-16-bit": lambda image: image.fromarray(
        np.zeros((2, 2), np.uint8), mode="I;16"
    ),
    "fromarray-unknown-dtype": lambda image: image.fromarray(np.zeros((2, 2, 3), np.float64)),
    "crop-right-before-left": lambda image: image.new("RGB", (4, 4)).crop((3, 0, 1, 2)),
    "crop-lower-before-upper": lambda image: image.new("RGB", (4, 4)).crop((0, 3, 2, 1)),
    "convert-to-unknown-mode": lambda image: image.new("RGB", (2, 2)).convert("XYZ"),
    "new-of-unknown-mode": lambda image: image.new("XYZ", (2, 2)),
}

# Pixels brutos de cada encoding que o executor materializa e os pixels que o PNG deve conter.
_ENCODED_FRAMES: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {
    "BGR8": (
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3)[:, :, ::-1],
    ),
    "MONO8": (np.arange(4, dtype=np.uint8).reshape(2, 2),) * 2,
    "MONO16": (np.array([[0, 300], [4000, 65535]], dtype=np.uint16),) * 2,
}


def _real_pillow() -> ModuleType:
    return importlib.import_module("PIL.Image")


@pytest.mark.parametrize("misuse", sorted(_MISUSES))
def test_the_fake_refuses_an_image_misuse_real_pillow_refuses(misuse: str) -> None:
    """A test running on the fake cannot hide a misuse the executor would hit in production."""
    with pytest.raises((ValueError, TypeError)):
        _MISUSES[misuse](fake_pillow_module())


def test_the_fake_models_the_operations_the_executor_performs(tmp_path: Path) -> None:
    image = fake_pillow_module()
    bgr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    target = tmp_path / "frame.png"

    image.fromarray(bgr[:, :, ::-1], mode="RGB").save(target)
    reopened = image.open(target).convert("RGB")

    assert reopened.size == (3, 2)
    np.testing.assert_array_equal(np.asarray(reopened), bgr[:, :, ::-1])
    # Caixa (left, upper, right, lower), com zeros fora da imagem, como no Pillow.
    np.testing.assert_array_equal(
        np.asarray(reopened.crop((2, 1, 4, 2))), [[bgr[1, 2, ::-1], [0, 0, 0]]]
    )
    mono = image.fromarray(np.full((2, 3), 7, np.uint8), mode="L")
    np.testing.assert_array_equal(np.asarray(mono.convert("RGB")), np.full((2, 3, 3), 7))
    np.testing.assert_array_equal(np.asarray(image.new("RGB", (3, 2), (1, 2, 3)))[1, 2], [1, 2, 3])


def test_the_fake_opens_an_encoded_frame_as_a_context_manager(tmp_path: Path) -> None:
    """The semantic bridge decodes the prepared frame's exact bytes in a ``with`` block (#524)."""
    import io

    image = fake_pillow_module()
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    target = tmp_path / "frame.png"
    image.fromarray(rgb, mode="RGB").save(target)

    with image.open(io.BytesIO(target.read_bytes())) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8)

    np.testing.assert_array_equal(pixels, rgb)


@_NEEDS_PILLOW
@pytest.mark.filterwarnings("ignore:'mode' parameter:DeprecationWarning")
def test_real_pillow_refuses_every_misuse_the_fake_refuses() -> None:
    """The fake's contract, checked against the library it stands in for."""
    accepted = []
    for name, misuse in sorted(_MISUSES.items()):
        try:
            misuse(_real_pillow())
        except (ValueError, TypeError):
            continue
        accepted.append(name)

    assert accepted == [], f"real Pillow accepts what the fake refuses: {accepted}"


@_NEEDS_PILLOW
@pytest.mark.filterwarnings("ignore:'mode' parameter:DeprecationWarning")
def test_the_prepared_image_is_a_real_png_of_the_observation(tmp_path: Path) -> None:
    """What the fake never does: the executor writes a real, decodable PNG of the pixels."""
    from contextmap.ingestion import (
        FrameId,
        ImageEncoding,
        ImageObservation,
        SensorId,
        SourceObservationId,
        SourceProvenance,
    )
    from contextmap.runtime.executors import _materialize_prepared_image
    from contextmap.shared import SourceTimestamp

    for encoding, (raw, expected) in _ENCODED_FRAMES.items():
        root = tmp_path / encoding
        root.mkdir()
        observation = ImageObservation(
            observation_id=SourceObservationId("frame-0000"),
            sensor_id=SensorId("camera_1"),
            frame_id=FrameId("camera_1_optical"),
            timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:header"),
            provenance=SourceProvenance(source_type="fixture", source_path="fixtures/images"),
            width=2,
            height=2,
            encoding=ImageEncoding[encoding],
            data=raw.tobytes(),
        )

        prepared = _materialize_prepared_image(observation, root)

        written = root / prepared.payload_reference
        assert written.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), encoding
        assert prepared.payload_artifact is not None
        assert prepared.payload_artifact.sha256 == hashlib.sha256(written.read_bytes()).hexdigest()
        with _real_pillow().open(written) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), expected, err_msg=encoding)
