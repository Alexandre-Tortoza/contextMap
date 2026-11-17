"""Load per-camera image-preparation masks from files into domain regions.

Image preparation may carry two explicit spatial constraints in the prepared-image
coordinate space: a :class:`~contextmap.visual_perception.ValidRegion` (the usable sensor
area, e.g. a fisheye's valid circle) and named
:class:`~contextmap.visual_perception.ExclusionRegion` masks (e.g. the recording rig that
occupies the same pixels every frame). Both are static per-camera assets, authored once and
referenced by the run configuration.

This is a composition-root adapter: it decodes a mask file into the domain contract, so the
executors receive an already-validated region and never a file path or an image library. The
asset format is NumPy's native ``.npy`` container holding a two-dimensional boolean array in
the prepared-image space, because Pillow is not a project dependency (the perception executor
imports it lazily) while NumPy already is, and the project's own masks persist as ``.npy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from contextmap.region_perception import InlineMask
from contextmap.runtime.configuration import CameraPreparationConfig
from contextmap.visual_perception import ExclusionRegion, ValidRegion


def load_mask(path: str | Path) -> InlineMask:
    """Read a boolean mask asset and return it as an :class:`InlineMask`.

    Args:
        path: File holding a two-dimensional boolean ``.npy`` array in prepared-image space.

    Returns:
        The mask as an immutable :class:`InlineMask`.

    Raises:
        ValueError: If the asset is not a two-dimensional boolean array. The message names
            the offending file, because a malformed mask asset is a configuration problem the
            operator must fix.
    """
    import numpy as np

    file = Path(path)
    pixels = np.load(file, allow_pickle=False)
    if pixels.dtype != np.bool_:
        raise ValueError(f"mask asset {file} must hold bool values, got dtype {pixels.dtype}")
    if pixels.ndim != 2:
        raise ValueError(
            f"mask asset {file} must be a two-dimensional image, got shape {pixels.shape}"
        )
    return InlineMask(pixels)


def valid_region_from_file(path: str | Path, *, reason: str, source: str) -> ValidRegion:
    """Build a :class:`ValidRegion` from a mask asset.

    Args:
        path: The boolean mask asset (foreground = eligible pixels).
        reason: Auditable motivation for the constraint.
        source: Auditable origin of the asset.

    Returns:
        The eligibility region in prepared-image space.
    """
    return ValidRegion(mask=load_mask(path), reason=reason, source=source)


def exclusion_region_from_file(
    path: str | Path, *, name: str, reason: str, source: str
) -> ExclusionRegion:
    """Build a named :class:`ExclusionRegion` from a mask asset.

    Args:
        path: The boolean mask asset (foreground = excluded pixels).
        name: Stable diagnostic name of the exclusion, e.g. ``"ego"``.
        reason: Auditable motivation for the exclusion.
        source: Auditable origin of the asset.

    Returns:
        The exclusion region in prepared-image space.
    """
    return ExclusionRegion(name=name, mask=load_mask(path), reason=reason, source=source)


@dataclass(frozen=True, kw_only=True)
class CameraImageMasks:
    """The decoded image-preparation constraints of one camera.

    Attributes:
        valid_region: The eligibility region, or ``None`` when the whole image is eligible.
        exclusions: The named exclusion regions, in configuration order.
    """

    valid_region: ValidRegion | None
    exclusions: tuple[ExclusionRegion, ...]

    def constraints_for(
        self, *, width: int, height: int, camera: str
    ) -> tuple[ValidRegion | None, tuple[ExclusionRegion, ...]]:
        """Return the constraints for a prepared image of ``(width, height)``.

        Args:
            width: Prepared image width in pixels.
            height: Prepared image height in pixels.
            camera: Camera identity, for an actionable error.

        Returns:
            The valid region (or ``None``) and the exclusion regions.

        Raises:
            ValueError: If any mask's dimensions differ from the prepared image. A mask is
                authored in the prepared-image space and is never resampled, so a mismatch is
                a configuration error the operator must fix.
        """
        regions: list[tuple[str, InlineMask]] = []
        if self.valid_region is not None:
            regions.append(("valid_region", self.valid_region.mask))
        regions.extend((f"exclusion {region.name!r}", region.mask) for region in self.exclusions)
        for label, mask in regions:
            if (mask.width, mask.height) != (width, height):
                raise ValueError(
                    f"camera {camera!r} {label} mask is {mask.width}x{mask.height}, but its "
                    f"prepared image is {width}x{height}: masks are not resampled"
                )
        return self.valid_region, self.exclusions


def resolve_image_masks(
    image_preparation: Mapping[str, CameraPreparationConfig], *, base_dir: Path | None = None
) -> Mapping[str, CameraImageMasks]:
    """Decode every camera's configured masks into domain regions.

    Args:
        image_preparation: The parsed ``[image_preparation]`` section, keyed by camera
            calibration id.
        base_dir: Directory relative mask references resolve against; absolute references are
            used as given. ``None`` resolves against the current working directory.

    Returns:
        The decoded masks per camera, ready for :meth:`CameraImageMasks.constraints_for`.
    """
    root = Path.cwd() if base_dir is None else Path(base_dir)

    def _path(reference: str) -> Path:
        candidate = Path(reference)
        return candidate if candidate.is_absolute() else root / candidate

    cameras: dict[str, CameraImageMasks] = {}
    for camera, prep in image_preparation.items():
        valid_region = (
            None
            if prep.valid_region is None
            else valid_region_from_file(
                _path(prep.valid_region), reason="configured valid region", source=camera
            )
        )
        exclusions = tuple(
            exclusion_region_from_file(
                _path(reference), name=name, reason="configured exclusion", source=camera
            )
            for name, reference in prep.exclusions.items()
        )
        cameras[camera] = CameraImageMasks(valid_region=valid_region, exclusions=exclusions)
    return MappingProxyType(cameras)
