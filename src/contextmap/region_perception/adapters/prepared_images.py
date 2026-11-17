"""Read a run's prepared images and materialize what a backend runtime consumes from them.

A ``PreparedImage`` names its payload by a reference relative to a root that only the run
knows. These functions resolve the reference against that root on the adapter/composition
side, so a model runtime supplied by a provider receives exactly what its SDK consumes, never a
reference it could not resolve (#661).
"""

from __future__ import annotations

import io
from hashlib import sha256
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any

from contextmap.visual_perception import PreparedImage

from ..discovery.region_discovery import DiscoveryInput, DiscoveryPassConfig


def read_prepared_image(root: Path, image: PreparedImage) -> bytes:
    """Read the prepared image under ``root`` and verify it is the one ``image`` names.

    The payload is read once, hashed and handed back, so the verified bytes are the consumed
    bytes and the file cannot change between the check and the use.

    Args:
        root: Directory that resolves ``image.payload_reference``.
        image: Prepared image whose reference and recorded SHA-256 identify the bytes.

    Returns:
        The payload bytes, whose SHA-256 equals ``image.payload_artifact.sha256``.

    Raises:
        ValueError: If the prepared image records no SHA-256, its reference points outside
            ``root``, or the bytes do not match the recorded SHA-256.
        OSError: If the payload cannot be read.
    """
    artifact = image.payload_artifact
    if artifact is None:
        raise ValueError("the prepared image has no recorded sha256 to verify")
    reference = PurePosixPath(image.payload_reference)
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError(
            f"prepared image {image.payload_reference!r} points outside the prepared-image root"
        )
    payload = (root / reference).read_bytes()
    if sha256(payload).hexdigest() != artifact.sha256:
        raise ValueError(
            f"prepared image {image.payload_reference!r} does not match its recorded sha256"
        )
    return payload


def require_unscaled_passes(pass_config: DiscoveryPassConfig | None) -> None:
    """Refuse a pass plan with a tiling whose passes :func:`materialize_discovery_pass` refuses.

    A tiling at a scale other than 1 presents its windows resampled, and no resampling rule is
    defined for discovery passes. Checking the plan before any run keeps a configuration from
    failing the discovery of every frame for that same reason.

    Args:
        pass_config: The discovery pass plan; ``None`` is the default, full-frame-only plan.

    Raises:
        ValueError: If any tiling of the plan has a scale other than 1.
    """
    if pass_config is None:
        return
    scales = sorted({tiling.scale for tiling in pass_config.tilings if tiling.scale != 1.0})
    if scales:
        listed = ", ".join(format(scale, "g") for scale in scales)
        raise ValueError(
            f"a tiling at scale {listed} cannot be materialized: no resampling rule for "
            "discovery passes is defined, so only unscaled passes are supported"
        )


def materialize_discovery_pass(root: Path, discovery_input: DiscoveryInput) -> Any:
    """Materialize the exact image one discovery pass presents to the backend.

    The prepared image is read under ``root`` and verified by :func:`read_prepared_image`,
    decoded to RGB with Pillow and cut to the pass window. The result is a Pillow RGB image of
    ``input_width x input_height`` pixels: what the official SAM3 and Florence-2 processors
    consume, and what SAM2 reads as an HWC ``uint8`` array.

    A pass whose input is not its window at scale 1 (``TilingConfig.scale`` other than 1) is
    refused before anything is read: presenting it would need a resampling rule, and none is
    defined nor recorded in the discovery identity.

    Args:
        root: Directory that resolves the prepared image's payload reference.
        discovery_input: The pass, and the prepared image it is cut from.

    Returns:
        A Pillow RGB image of the pass's input dimensions.

    Raises:
        ValueError: If the pass is scaled, its window is not whole pixels inside the prepared
            image, the payload cannot be verified, or it decodes to other dimensions than the
            prepared image records.
        OSError: If the payload cannot be read.
        ModuleNotFoundError: If Pillow is not installed.
    """
    prepared = discovery_input.prepared_image
    discovery_pass = discovery_input.discovery_pass
    window = discovery_pass.window
    corners = (window.x_min, window.y_min, window.x_max, window.y_max)
    if not all(float(value).is_integer() for value in corners):
        raise ValueError(f"discovery pass {discovery_pass.pass_id!r} window is not whole pixels")
    left, upper, right, lower = (int(value) for value in corners)
    if right > prepared.width or lower > prepared.height:
        raise ValueError(
            f"discovery pass {discovery_pass.pass_id!r} window leaves the "
            f"{prepared.width}x{prepared.height} prepared image"
        )
    if (discovery_pass.input_width, discovery_pass.input_height) != (right - left, lower - upper):
        raise ValueError(
            f"discovery pass {discovery_pass.pass_id!r} is scaled to "
            f"{discovery_pass.input_width}x{discovery_pass.input_height}; no resampling rule "
            "for discovery passes is defined, so only unscaled passes can be materialized"
        )
    payload = read_prepared_image(root, prepared)
    image_module = import_module("PIL.Image")
    with image_module.open(io.BytesIO(payload)) as opened:
        rgb = opened.convert("RGB")
    if tuple(rgb.size) != (prepared.width, prepared.height):
        raise ValueError(
            f"prepared image {prepared.payload_reference!r} decodes to {tuple(rgb.size)}, not "
            f"the recorded {(prepared.width, prepared.height)}"
        )
    return rgb.crop((left, upper, right, lower))
