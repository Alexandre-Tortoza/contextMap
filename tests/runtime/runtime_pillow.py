"""A stand-in for the ``PIL.Image`` operations the perception executor performs (#638).

Pillow is not a dependency of the project and ``.[dev]`` does not install it, yet the
``VisualPerceptionExecutor`` materializes each prepared image and each semantic view through
``PIL.Image`` (``fromarray``, ``open``, also in a ``with`` block, ``convert``, ``crop``,
``save``). Installing
:func:`fake_pillow` lets the executor's own orchestration run without Pillow instead of the tests
being skipped. Pixels are kept as NumPy arrays and written with ``np.save``; nothing here encodes
a PNG, which only a test with the real Pillow can cover (``test_runtime_pillow.py``).

The fake refuses every misuse of those operations that real Pillow refuses (an array that does
not hold the mode it is given, a crop box with its corners swapped, an unknown mode), so a test
that passes on it cannot hide an image API misuse the executor would hit in production;
``test_runtime_pillow.py`` checks that contract against real Pillow whenever it is installed. A
conversion the fake does not model fails loudly instead of guessing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

# Os modos que o executor produz ou pede: canais por pixel (None = imagem 2D) e dtype dos pixels.
_MODES: dict[str, tuple[int | None, type[np.generic]]] = {
    "RGB": (3, np.uint8),
    "L": (None, np.uint8),
    "I;16": (None, np.uint16),
}


class FakePillowImage:
    """An image in one of :data:`_MODES`, over the NumPy array of its pixels."""

    def __init__(self, pixels: NDArray[Any], mode: str) -> None:
        """Hold ``pixels`` (``(height, width[, channels])``) as an image of ``mode``.

        Raises:
            ValueError: If ``mode`` is unknown or the pixels do not hold it exactly.
        """
        if mode not in _MODES:
            raise ValueError(f"unrecognized image mode: {mode}")
        if _mode_of(pixels) != mode:
            raise ValueError(
                f"pixels of shape {pixels.shape} and dtype {pixels.dtype} are not a {mode} image"
            )
        self._pixels = pixels
        self.mode = mode

    @property
    def size(self) -> tuple[int, int]:
        """Return ``(width, height)``, the order Pillow uses."""
        return int(self._pixels.shape[1]), int(self._pixels.shape[0])

    def __enter__(self) -> FakePillowImage:
        """Serve ``with Image.open(...) as image``, as a Pillow image does."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave the ``with`` block; the fake holds no file to close."""

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[Any]:
        """Expose the pixels to ``np.asarray``, as Pillow does."""
        return self._pixels if dtype is None else self._pixels.astype(dtype)

    def save(self, path: str | Path) -> None:
        """Write the pixels with ``np.save``, whatever the file extension says."""
        with Path(path).open("wb") as handle:
            np.save(handle, self._pixels)

    def convert(self, mode: str) -> FakePillowImage:
        """Convert to ``mode``, as far as the fake models it.

        Raises:
            ValueError: If ``mode`` is not an image mode, as Pillow does.
            NotImplementedError: If the conversion is one the fake does not model.
        """
        if mode not in _MODES:
            raise ValueError("image has wrong mode")
        if mode == self.mode:
            return FakePillowImage(self._pixels.copy(), mode)
        if (self.mode, mode) == ("L", "RGB"):
            return FakePillowImage(np.repeat(self._pixels[:, :, None], 3, axis=2), mode)
        raise NotImplementedError(f"the fake Pillow does not model {self.mode} -> {mode}")

    def crop(self, box: tuple[int, int, int, int]) -> FakePillowImage:
        """Cut ``(left, upper, right, lower)``, padding with zeros outside the image like Pillow.

        Raises:
            ValueError: If the box is not four integers, or a far corner lies before its near
                one, which Pillow also refuses.
        """
        if len(box) != 4 or not all(isinstance(value, int | np.integer) for value in box):
            raise ValueError(f"crop box must be four integers (left, upper, right, lower): {box}")
        left, upper, right, lower = (int(value) for value in box)
        if right < left:
            raise ValueError("Coordinate 'right' is less than 'left'")
        if lower < upper:
            raise ValueError("Coordinate 'lower' is less than 'upper'")
        width, height = self.size
        shape = (lower - upper, right - left, *self._pixels.shape[2:])
        cropped = np.zeros(shape, dtype=self._pixels.dtype)
        top, bottom = max(upper, 0), min(lower, height)
        start, end = max(left, 0), min(right, width)
        if top < bottom and start < end:
            cropped[top - upper : bottom - upper, start - left : end - left] = self._pixels[
                top:bottom, start:end
            ]
        return FakePillowImage(cropped, self.mode)


def fromarray(pixels: Any, mode: str | None = None) -> FakePillowImage:
    """Wrap an array as an image; given ``mode``, the array has to hold exactly that mode.

    Raises:
        TypeError: If no mode is given and the array holds none of :data:`_MODES`, like
            Pillow's "Cannot handle this data type".
        ValueError: If ``mode`` is given and the array does not hold it. Pillow refuses an
            array too small for the mode and reads a larger one as garbage; the fake refuses
            both.
    """
    array = np.asarray(pixels)
    if mode is None:
        mode = _mode_of(array)
        if mode is None:
            raise TypeError(f"Cannot handle this data type: {array.shape}, {array.dtype}")
    return FakePillowImage(array, mode)


def open_image(path: str | Path) -> FakePillowImage:
    """Read back an image :meth:`FakePillowImage.save` wrote (``PIL.Image.open``)."""
    array = np.load(path)
    mode = _mode_of(array)
    if mode is None:
        raise ValueError(f"not an image written by the fake Pillow: {path}")
    return FakePillowImage(array, mode)


def new(mode: str, size: tuple[int, int], color: Any = 0) -> FakePillowImage:
    """Create a ``size`` (``(width, height)``) image of ``mode`` filled with ``color``.

    Raises:
        ValueError: If ``mode`` is unknown, as Pillow does.
    """
    if mode not in _MODES:
        raise ValueError("unrecognized image mode")
    channels, dtype = _MODES[mode]
    width, height = size
    shape = (height, width) if channels is None else (height, width, channels)
    return FakePillowImage(np.full(shape, color, dtype=dtype), mode)


def fake_pillow_module() -> types.ModuleType:
    """Return a module standing in for ``PIL.Image``, with only what the executor uses."""
    module = types.ModuleType("PIL.Image")
    module.__dict__.update(fromarray=fromarray, open=open_image, new=new)
    return module


@pytest.fixture
def fake_pillow(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Serve the fake as ``PIL.Image`` for one test, so the executor runs without Pillow."""
    module = fake_pillow_module()
    monkeypatch.setitem(sys.modules, "PIL.Image", module)
    return module


def _mode_of(array: NDArray[Any]) -> str | None:
    """Return the mode an array holds, or ``None`` when it holds none of :data:`_MODES`."""
    for mode, (channels, dtype) in _MODES.items():
        if array.dtype != dtype:
            continue
        if channels is None and array.ndim == 2:
            return mode
        if channels is not None and array.ndim == 3 and array.shape[2] == channels:
            return mode
    return None
