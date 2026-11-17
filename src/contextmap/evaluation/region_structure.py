"""Ground-truth-free structural metrics of one frame's region masks.

Region Discovery can return masks that are individually plausible and collectively unusable:
speckled into hundreds of pieces, stacked several deep over the same pixels, or repeated with
almost the same extent. None of that needs an annotation to be measured. This module measures
it for one frame, from the masks alone, so that discovery configurations (for example SAM2
quality thresholds) can be compared on structure before any label exists.

Every metric has an explicit definition (see ``docs/region-structure.md``):

* a **component** is a set of mask pixels connected through edge neighbours
  (:data:`MASK_CONNECTIVITY` = 4; diagonal contact does not join two components);
* a mask is **fragmented** when it has more than one component;
* the **fraction of mask A inside mask B** is ``|A ∩ B| / |A|``; a mask is **contained in
  another** when that fraction reaches ``containment_threshold`` for some other mask;
* an unordered pair is **contained** when the smaller mask is inside the larger by at least
  ``containment_threshold`` (``|A ∩ B| / min(|A|, |B|)``), and **duplicate** when its IoU
  reaches ``duplicate_iou_threshold``; the two counts are independent (a duplicate pair is
  usually also contained);
* the **nesting depth** of a pixel is how many masks cover it.

Optionally, :func:`known_object_recall` measures whether annotated objects are still found by
some mask, which is what a stricter discovery configuration can lose.

The report never mixes these numbers into one score: each is a separate, named quantity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from contextmap.evaluation._validation import require_unique
from contextmap.evaluation.annotations import RegionAnnotation
from contextmap.region_perception import (
    BoundingBox,
    InlineMask,
)

REGION_STRUCTURE_METRIC_VERSION = "1.0.0"
"""Bumped whenever a metric definition changes, so reports stay comparable."""

MASK_CONNECTIVITY = 4
"""Pixel connectivity of a mask component: edge neighbours only.

It is the default structuring element of ``scipy.ndimage.label`` in two dimensions, which the
semantic-geometry investigation (``p6_mask_quality.py``) used, so its numbers are reproduced.
"""

AREA_QUANTILE_METHOD: Final = "linear"
"""NumPy quantile method used for the area percentiles."""

_PIXEL_BLOCK = 65_536
"""Pixels per block in the pairwise-intersection product.

Each block's partial sums stay below ``2**24``, so the ``float32`` product is exact.
"""


@dataclass(frozen=True, kw_only=True)
class MaskStructure:
    """Structure of one mask within its frame.

    Attributes:
        region_id: The region the mask belongs to.
        area_px: Foreground pixels.
        component_count: Edge-connected components.
        max_fraction_inside_other: Largest ``|A ∩ B| / |A|`` over the other masks ``B``;
            ``0.0`` when the mask shares no pixel with any other.
        most_containing_region_id: The mask ``B`` reaching that fraction (the first in region
            order on ties); ``None`` when the mask shares no pixel with any other.
    """

    region_id: str
    area_px: int
    component_count: int
    max_fraction_inside_other: float
    most_containing_region_id: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "region_id": self.region_id,
            "area_px": self.area_px,
            "component_count": self.component_count,
            "max_fraction_inside_other": self.max_fraction_inside_other,
            "most_containing_region_id": self.most_containing_region_id,
        }


@dataclass(frozen=True, kw_only=True)
class RegionStructureReport:
    """Structural metrics of one frame's masks.

    Rates whose population is empty are ``None``, never ``0.0``.

    Attributes:
        metric_version: :data:`REGION_STRUCTURE_METRIC_VERSION`.
        containment_threshold: Fraction that makes one mask "inside" another.
        duplicate_iou_threshold: IoU that makes a pair a duplicate.
        mask_count: Masks measured.
        proposal_count: Backend proposals the masks were selected from, when given.
        accepted_mask_ratio: ``mask_count / proposal_count``; ``None`` without a proposal count.
        area_px_p10: 10th percentile of the mask areas, in pixels.
        area_px_median: Median mask area, in pixels.
        area_px_p90: 90th percentile of the mask areas, in pixels.
        fragmented_mask_count: Masks with more than one component.
        fragmentation_rate: ``fragmented_mask_count / mask_count``.
        masks_contained_in_another: Masks whose ``max_fraction_inside_other`` reaches
            ``containment_threshold``.
        overlapping_pair_count: Unordered pairs sharing at least one pixel.
        contained_pair_count: Unordered pairs with ``|A ∩ B| / min(|A|, |B|)`` at or above
            ``containment_threshold``.
        duplicate_pair_count: Unordered pairs with IoU at or above ``duplicate_iou_threshold``.
        covered_pixel_count: Pixels covered by at least one mask.
        pixels_with_multiple_masks: Covered pixels with nesting depth two or more.
        multiple_mask_pixel_fraction: ``pixels_with_multiple_masks / covered_pixel_count``.
        nesting_depth_histogram: ``(depth, pixels)`` for every depth from one that occurs.
        max_nesting_depth: Largest number of masks covering one pixel (``0`` without masks).
        usable_pixel_count: Pixels of the usable area, when one was given.
        usable_area_coverage: Share of the usable area covered by at least one mask.
        masks: Per-mask structure, in region order.
    """

    metric_version: str
    containment_threshold: float
    duplicate_iou_threshold: float
    mask_count: int
    proposal_count: int | None
    accepted_mask_ratio: float | None
    area_px_p10: float | None
    area_px_median: float | None
    area_px_p90: float | None
    fragmented_mask_count: int
    fragmentation_rate: float | None
    masks_contained_in_another: int
    overlapping_pair_count: int
    contained_pair_count: int
    duplicate_pair_count: int
    covered_pixel_count: int
    pixels_with_multiple_masks: int
    multiple_mask_pixel_fraction: float | None
    nesting_depth_histogram: tuple[tuple[int, int], ...]
    max_nesting_depth: int
    usable_pixel_count: int | None
    usable_area_coverage: float | None
    masks: tuple[MaskStructure, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report, with the definitions it was measured under."""
        return {
            "metric_version": self.metric_version,
            "definition": {
                "connectivity": MASK_CONNECTIVITY,
                "containment_threshold": self.containment_threshold,
                "duplicate_iou_threshold": self.duplicate_iou_threshold,
                "area_quantile_method": AREA_QUANTILE_METHOD,
            },
            "mask_count": self.mask_count,
            "proposal_count": self.proposal_count,
            "accepted_mask_ratio": self.accepted_mask_ratio,
            "area_px_p10": self.area_px_p10,
            "area_px_median": self.area_px_median,
            "area_px_p90": self.area_px_p90,
            "fragmented_mask_count": self.fragmented_mask_count,
            "fragmentation_rate": self.fragmentation_rate,
            "masks_contained_in_another": self.masks_contained_in_another,
            "overlapping_pair_count": self.overlapping_pair_count,
            "contained_pair_count": self.contained_pair_count,
            "duplicate_pair_count": self.duplicate_pair_count,
            "covered_pixel_count": self.covered_pixel_count,
            "pixels_with_multiple_masks": self.pixels_with_multiple_masks,
            "multiple_mask_pixel_fraction": self.multiple_mask_pixel_fraction,
            "nesting_depth_histogram": {
                str(depth): pixels for depth, pixels in self.nesting_depth_histogram
            },
            "max_nesting_depth": self.max_nesting_depth,
            "usable_pixel_count": self.usable_pixel_count,
            "usable_area_coverage": self.usable_area_coverage,
            "masks": [mask.to_dict() for mask in self.masks],
        }


@dataclass(frozen=True, kw_only=True)
class ObjectMatch:
    """How well the best mask covers one annotated object.

    Attributes:
        object_id: The annotated object.
        geometry: ``"mask"`` when the annotation's mask was compared with each mask, or
            ``"box"`` when only a box was annotated and it was compared with each mask's tight
            box.
        best_iou: Largest IoU with any mask; ``0.0`` when none overlaps.
        best_region_id: The mask reaching it; ``None`` when none overlaps.
        recalled: Whether ``best_iou`` reaches the match threshold.
    """

    object_id: str
    geometry: str
    best_iou: float
    best_region_id: str | None
    recalled: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "object_id": self.object_id,
            "geometry": self.geometry,
            "best_iou": self.best_iou,
            "best_region_id": self.best_region_id,
            "recalled": self.recalled,
        }


@dataclass(frozen=True, kw_only=True)
class KnownObjectRecall:
    """Which annotated objects some mask still finds.

    Attributes:
        metric_version: :data:`REGION_STRUCTURE_METRIC_VERSION`.
        match_iou_threshold: Best IoU at or above which an object counts as recalled.
        object_count: Annotated objects.
        recalled_count: Objects whose best IoU reaches the threshold.
        recall: ``recalled_count / object_count``; ``None`` without objects.
        matches: One match per object, in annotation order.
    """

    metric_version: str
    match_iou_threshold: float
    object_count: int
    recalled_count: int
    recall: float | None
    matches: tuple[ObjectMatch, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""
        return {
            "metric_version": self.metric_version,
            "match_iou_threshold": self.match_iou_threshold,
            "object_count": self.object_count,
            "recalled_count": self.recalled_count,
            "recall": self.recall,
            "matches": [match.to_dict() for match in self.matches],
        }


def connected_component_count(mask: InlineMask) -> int:
    """Count the edge-connected components of a mask (:data:`MASK_CONNECTIVITY`)."""
    return _component_count(mask.as_array())


def measure_region_structure(
    masks: Mapping[str, InlineMask],
    *,
    containment_threshold: float,
    duplicate_iou_threshold: float,
    proposal_count: int | None = None,
    usable_area: InlineMask | None = None,
) -> RegionStructureReport:
    """Measure the structure of one frame's masks.

    Args:
        masks: The frame's masks by region id, all in the same image space. Pass only the
            regions under evaluation (typically the accepted ones).
        containment_threshold: Fraction, in ``(0, 1]``, that makes one mask inside another.
        duplicate_iou_threshold: IoU, in ``(0, 1]``, that makes a pair a duplicate.
        proposal_count: Backend proposals the masks were selected from, for the accepted
            ratio; at least ``len(masks)``.
        usable_area: Pixels where a region is meaningful (valid region minus exclusions), for
            the coverage; same image space as the masks.

    Returns:
        The frame's structural report.

    Raises:
        ValueError: If a threshold is outside ``(0, 1]``, a mask is empty, the masks or the
            usable area disagree on dimensions, or ``proposal_count`` is below the mask count.
    """
    _require_fraction("containment_threshold", containment_threshold)
    _require_fraction("duplicate_iou_threshold", duplicate_iou_threshold)
    region_ids = sorted(masks)
    shape = _common_shape([masks[region_id] for region_id in region_ids], usable_area)
    for region_id in region_ids:
        if masks[region_id].area == 0:
            raise ValueError(f"mask of region {region_id!r} is empty")
    if proposal_count is not None and proposal_count < len(region_ids):
        raise ValueError(
            f"proposal_count ({proposal_count}) must be at least the mask count ({len(region_ids)})"
        )

    count = len(region_ids)
    pixel_count = shape[0] * shape[1] if shape is not None else 0
    stack = np.zeros((count, pixel_count), dtype=bool)
    for row, region_id in enumerate(region_ids):
        stack[row] = masks[region_id].as_array().reshape(-1)
    areas = stack.sum(axis=1).astype(np.int64)
    components = [_component_count(masks[region_id].as_array()) for region_id in region_ids]

    intersections = _pairwise_intersections(stack)
    inside = intersections / areas[:, None] if count else np.zeros((0, 0))
    np.fill_diagonal(inside, 0.0)
    unions = areas[:, None] + areas[None, :] - intersections
    upper = np.triu_indices(count, 1)
    pair_intersections = intersections[upper]
    pair_iou = pair_intersections / unions[upper] if count else np.zeros(0)
    pair_containment = pair_intersections / np.minimum(areas[upper[0]], areas[upper[1]])

    mask_rows = []
    for row, region_id in enumerate(region_ids):
        best = int(np.argmax(inside[row])) if count > 1 else 0
        fraction = float(inside[row, best]) if count > 1 else 0.0
        mask_rows.append(
            MaskStructure(
                region_id=region_id,
                area_px=int(areas[row]),
                component_count=components[row],
                max_fraction_inside_other=fraction,
                most_containing_region_id=region_ids[best] if fraction > 0 else None,
            )
        )

    depth = stack.sum(axis=0)
    covered = depth > 0
    covered_count = int(covered.sum())
    histogram = np.bincount(depth[covered]) if covered_count else np.zeros(1, dtype=np.int64)
    multiple = int((depth >= 2).sum())
    fragmented = sum(1 for value in components if value > 1)
    quantiles: list[float | None] = (
        [
            float(value)
            for value in np.quantile(
                areas.astype(np.float64), [0.1, 0.5, 0.9], method=AREA_QUANTILE_METHOD
            )
        ]
        if count
        else [None, None, None]
    )
    usable_count: int | None = None
    usable_coverage: float | None = None
    if usable_area is not None:
        usable = usable_area.as_array().reshape(-1)
        usable_count = int(usable.sum())
        usable_coverage = float(covered[usable].mean()) if usable_count else None

    return RegionStructureReport(
        metric_version=REGION_STRUCTURE_METRIC_VERSION,
        containment_threshold=containment_threshold,
        duplicate_iou_threshold=duplicate_iou_threshold,
        mask_count=count,
        proposal_count=proposal_count,
        accepted_mask_ratio=(
            count / proposal_count if proposal_count is not None and proposal_count else None
        ),
        area_px_p10=quantiles[0],
        area_px_median=quantiles[1],
        area_px_p90=quantiles[2],
        fragmented_mask_count=fragmented,
        fragmentation_rate=fragmented / count if count else None,
        masks_contained_in_another=sum(
            1 for mask in mask_rows if mask.max_fraction_inside_other >= containment_threshold
        ),
        overlapping_pair_count=int((pair_intersections > 0).sum()),
        contained_pair_count=int((pair_containment >= containment_threshold).sum()),
        duplicate_pair_count=int((pair_iou >= duplicate_iou_threshold).sum()),
        covered_pixel_count=covered_count,
        pixels_with_multiple_masks=multiple,
        multiple_mask_pixel_fraction=multiple / covered_count if covered_count else None,
        nesting_depth_histogram=tuple(
            (depth_value, int(pixels))
            for depth_value, pixels in enumerate(histogram)
            if depth_value > 0 and pixels
        ),
        max_nesting_depth=int(depth.max()) if pixel_count else 0,
        usable_pixel_count=usable_count,
        usable_area_coverage=usable_coverage,
        masks=tuple(mask_rows),
    )


def known_object_recall(
    masks: Mapping[str, InlineMask],
    objects: Sequence[RegionAnnotation],
    *,
    match_iou_threshold: float,
) -> KnownObjectRecall:
    """Measure which annotated objects are still found by some mask.

    An object annotated with a mask is compared with each mask by pixel IoU. An object
    annotated only with a box is compared, by box IoU, with each mask's tight half-open box:
    a box annotation says nothing about which pixels inside it belong to the object.

    Args:
        masks: The frame's masks by region id.
        objects: Annotated objects with unique ids, by mask, box or both (the mask wins).
        match_iou_threshold: Best IoU, in ``(0, 1]``, at which an object counts as recalled.

    Returns:
        The best match of every object and the recall.

    Raises:
        ValueError: If the threshold is outside ``(0, 1]``, object ids repeat, or an object
            mask disagrees with the masks on dimensions.
    """
    _require_fraction("match_iou_threshold", match_iou_threshold)
    require_unique("annotated object id", (item.region_id for item in objects))
    region_ids = sorted(masks)
    pixels = {region_id: masks[region_id].as_array() for region_id in region_ids}
    boxes = {region_id: _tight_box(pixels[region_id]) for region_id in region_ids}

    matches = []
    for item in objects:
        if item.mask is not None:
            geometry = "mask"
            annotated = item.mask.as_array()
            scores = []
            for region_id in region_ids:
                if pixels[region_id].shape != annotated.shape:
                    raise ValueError(
                        f"mask of object {item.region_id!r} disagrees with region "
                        f"{region_id!r} on dimensions"
                    )
                scores.append(_mask_iou(annotated, pixels[region_id]))
        else:
            geometry = "box"
            assert item.box is not None  # RegionAnnotation exige máscara ou caixa.
            scores = [_box_iou(item.box, boxes[region_id]) for region_id in region_ids]
        best = int(np.argmax(scores)) if scores else 0
        best_iou = float(scores[best]) if scores else 0.0
        matches.append(
            ObjectMatch(
                object_id=item.region_id,
                geometry=geometry,
                best_iou=best_iou,
                best_region_id=region_ids[best] if best_iou > 0 else None,
                recalled=best_iou >= match_iou_threshold,
            )
        )
    recalled = sum(1 for match in matches if match.recalled)
    return KnownObjectRecall(
        metric_version=REGION_STRUCTURE_METRIC_VERSION,
        match_iou_threshold=match_iou_threshold,
        object_count=len(matches),
        recalled_count=recalled,
        recall=recalled / len(matches) if matches else None,
        matches=tuple(matches),
    )


def _require_fraction(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")


def _common_shape(
    masks: Sequence[InlineMask], usable_area: InlineMask | None
) -> tuple[int, int] | None:
    """Return the one image shape shared by the masks and the usable area.

    Raises:
        ValueError: If two masks, or a mask and the usable area, disagree on dimensions.
    """
    shapes = {(mask.height, mask.width) for mask in masks}
    if len(shapes) > 1:
        raise ValueError(f"masks disagree on dimensions: {sorted(shapes)}")
    shape = next(iter(shapes)) if shapes else None
    if usable_area is not None:
        usable_shape = (usable_area.height, usable_area.width)
        if shape is not None and usable_shape != shape:
            raise ValueError(
                f"usable_area dimensions {usable_shape} disagree with the masks {shape}"
            )
        shape = usable_shape
    return shape


def _component_count(pixels: NDArray[np.bool_]) -> int:
    """Count edge-connected components through the horizontal runs of each row.

    Two runs of adjacent rows touch when their column intervals overlap; the components are
    the connected components of that run graph, found by minimum-label propagation.
    """
    height, width = pixels.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = pixels
    steps = np.diff(padded, axis=1)
    run_rows, run_starts = np.nonzero(steps == 1)
    _, run_ends = np.nonzero(steps == -1)
    run_count = len(run_rows)
    if run_count == 0:
        return 0

    # Chaves linha-major crescentes: os runs de uma linha ocupam um trecho contíguo, então os
    # runs da linha de cima que tocam [start, end) formam o intervalo [first, last).
    stride = width + 2
    start_keys = run_rows * stride + run_starts
    end_keys = run_rows * stride + run_ends
    above = (run_rows - 1) * stride
    first = np.searchsorted(end_keys, above + run_starts, side="right")
    last = np.searchsorted(start_keys, above + run_ends, side="left")
    counts = np.maximum(last - first, 0)
    sources = np.repeat(np.arange(run_count), counts)
    offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    targets = np.repeat(first, counts) + offsets

    labels = np.arange(run_count)
    while True:
        previous = labels
        lowest = np.minimum(labels[sources], labels[targets])
        labels = labels.copy()
        np.minimum.at(labels, sources, lowest)
        np.minimum.at(labels, targets, lowest)
        # Atalho (pointer jumping): cada rótulo aponta para o rótulo do seu rótulo.
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        if np.array_equal(labels, previous):
            return int(np.unique(labels).size)


def _pairwise_intersections(stack: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Return ``|A ∩ B|`` for every pair of rows of a ``(masks, pixels)`` stack."""
    count = len(stack)
    if count == 0:
        return np.zeros((0, 0), dtype=np.int64)
    covered = stack[:, stack.any(axis=0)]
    total = np.zeros((count, count), dtype=np.float64)
    for begin in range(0, covered.shape[1], _PIXEL_BLOCK):
        block = covered[:, begin : begin + _PIXEL_BLOCK].astype(np.float32)
        total += block @ block.T
    return np.rint(total).astype(np.int64)


def _mask_iou(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> float:
    union = int(np.count_nonzero(first | second))
    return int(np.count_nonzero(first & second)) / union if union else 0.0


def _tight_box(pixels: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Return the half-open ``(x_min, y_min, x_max, y_max)`` box of a non-empty mask."""
    rows = np.flatnonzero(pixels.any(axis=1))
    columns = np.flatnonzero(pixels.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def _box_iou(box: BoundingBox, other: tuple[int, int, int, int]) -> float:
    x_min, y_min, x_max, y_max = other
    width = min(box.x_max, x_max) - max(box.x_min, x_min)
    height = min(box.y_max, y_max) - max(box.y_min, y_min)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = box.area + (x_max - x_min) * (y_max - y_min) - intersection
    return float(intersection / union)
