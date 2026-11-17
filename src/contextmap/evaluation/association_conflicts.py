"""Conflict taxonomy of 2D->3D region membership, measured without labels.

Sensor Association assigns a visible map point to **every** accepted mask that contains its
pixel: membership is evidence, and nothing there decides which region owns a point. When masks
overlap, one point therefore supports several regions. Whether that is a problem depends on
how the masks overlap, and on whether the points a mask collects lie on one surface. This
module measures both for one frame, from the membership, the masks and the depth of each
point, so that a later ownership or filtering policy can target the class that dominates
instead of a guess (see ``docs/association-conflicts.md``).

Definitions:

* a **multi-mask point** is an associated point that belongs to two or more regions;
* every unordered pair of regions sharing at least one point is classified from its 2D masks:
  **duplicate** when their IoU reaches ``duplicate_iou_threshold``; otherwise **nested** when
  the smaller mask is inside the larger by at least ``nested_containment_threshold``
  (``|A ∩ B| / min(|A|, |B|)``); otherwise **partial**. Duplicate wins over nested because
  two near-identical masks are always mutually nested;
* a multi-mask point takes the **most severe** class among the pairs it belongs to, in the
  order partial > duplicate > nested: a point shared only by a part and its whole is
  "nested", not a conflict;
* the **depth consistency** of a region is the fraction of its associated points outside its
  main depth layer (:func:`off_main_layer_fraction`);
* the **occlusion proximity** is the fraction of occluded points whose depth lies within
  ``proximity_m`` of the supporting depth that occluded them
  (:func:`measure_occlusion_proximity`).

Numbers of different kinds are reported side by side and never combined into one score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextmap.region_perception import (
    InlineMask,
)

ASSOCIATION_CONFLICT_METRIC_VERSION = "1.0.0"
"""Bumped whenever a metric definition changes, so reports stay comparable."""


class ConflictClass(Enum):
    """How two regions sharing points overlap in the image.

    The member order is the severity order, from the least to the most severe.
    """

    NESTED = "nested"
    DUPLICATE = "duplicate"
    PARTIAL = "partial"


_SEVERITY = {conflict: rank for rank, conflict in enumerate(ConflictClass)}
_BY_SEVERITY = tuple(ConflictClass)


@dataclass(frozen=True, kw_only=True)
class DepthLayerCriterion:
    """How the depths of one region's points split into layers.

    Sorted depths start a new layer wherever two consecutive values are farther apart than
    ``max(gap_m, gap_ratio * nearer_depth)``. The main layer is the nearest one holding at
    least ``main_layer_min_share`` of the points; when none does, the largest one (the
    nearest on ties). Regions with fewer than ``min_points`` points are not measured.

    Attributes:
        gap_m: Smallest depth jump, in meters, that separates two layers.
        gap_ratio: Depth jump, as a fraction of the nearer depth, that separates two layers.
        main_layer_min_share: Share of the points, in ``(0, 1]``, a layer needs to be main.
        min_points: Fewest points a region needs to be measured.
    """

    gap_m: float
    gap_ratio: float
    main_layer_min_share: float
    min_points: int

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a gap is negative or not finite, the share is outside ``(0, 1]``,
                or ``min_points`` is below one.
        """
        for name in ("gap_m", "gap_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if not 0.0 < self.main_layer_min_share <= 1.0:
            raise ValueError(
                f"main_layer_min_share must be in (0, 1], got {self.main_layer_min_share!r}"
            )
        if self.min_points < 1:
            raise ValueError(f"min_points must be at least 1, got {self.min_points!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "gap_m": self.gap_m,
            "gap_ratio": self.gap_ratio,
            "main_layer_min_share": self.main_layer_min_share,
            "min_points": self.min_points,
        }


@dataclass(frozen=True, kw_only=True)
class MaskPairConflict:
    """One pair of regions that share associated points.

    Attributes:
        region_ids: The two regions, in sorted order.
        shared_point_count: Points associated with both.
        iou_2d: Pixel IoU of their masks.
        containment_2d: ``|A ∩ B| / min(|A|, |B|)`` of their masks.
        conflict_class: The pair's class.
    """

    region_ids: tuple[str, str]
    shared_point_count: int
    iou_2d: float
    containment_2d: float
    conflict_class: ConflictClass

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "region_ids": list(self.region_ids),
            "shared_point_count": self.shared_point_count,
            "iou_2d": self.iou_2d,
            "containment_2d": self.containment_2d,
            "conflict_class": self.conflict_class.value,
        }


@dataclass(frozen=True, kw_only=True)
class MaskDepthConsistency:
    """Depth consistency of the points one region collected.

    Attributes:
        region_id: The region.
        associated_point_count: Points associated with it.
        off_main_layer_fraction: Fraction outside its main depth layer; ``None`` below the
            criterion's ``min_points``.
    """

    region_id: str
    associated_point_count: int
    off_main_layer_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record."""
        return {
            "region_id": self.region_id,
            "associated_point_count": self.associated_point_count,
            "off_main_layer_fraction": self.off_main_layer_fraction,
        }


@dataclass(frozen=True, kw_only=True)
class AssociationConflictReport:
    """Multi-membership, its conflict classes and the depth consistency of one frame.

    Rates whose population is empty are ``None``, never ``0.0``. The ``*_by_class`` fields
    list every class, in :class:`ConflictClass` order.

    Attributes:
        metric_version: :data:`ASSOCIATION_CONFLICT_METRIC_VERSION`.
        nested_containment_threshold: Containment that makes a pair nested.
        duplicate_iou_threshold: IoU that makes a pair a duplicate.
        depth_layers: The depth-layer criterion.
        associated_point_count: Distinct points associated with at least one region.
        membership_count: (region, point) memberships.
        multi_mask_point_count: Points associated with two or more regions.
        multi_mask_point_rate: ``multi_mask_point_count / associated_point_count``.
        mean_regions_per_point: ``membership_count / associated_point_count``.
        max_regions_per_point: Most regions one point belongs to (``0`` without points).
        regions_per_point_histogram: ``(regions, points)`` for every count that occurs.
        pair_count_by_class: Region pairs of each class.
        pair_shared_points_by_class: Sum, over the pairs of each class, of their shared
            points; a point shared by several pairs counts once per pair.
        multi_mask_points_by_class: Multi-mask points by their most severe class; the counts
            add up to ``multi_mask_point_count``.
        multi_mask_point_rate_by_class: Those counts over ``associated_point_count``.
        depth_layer_mask_count: Regions with enough points to be measured.
        off_main_layer_fraction_weighted: Mean off-main-layer fraction of the measured
            regions, weighted by their associated points.
        pairs: Every classified pair, in region order.
        masks: The depth consistency of every region in the membership, in region order.
    """

    metric_version: str
    nested_containment_threshold: float
    duplicate_iou_threshold: float
    depth_layers: DepthLayerCriterion
    associated_point_count: int
    membership_count: int
    multi_mask_point_count: int
    multi_mask_point_rate: float | None
    mean_regions_per_point: float | None
    max_regions_per_point: int
    regions_per_point_histogram: tuple[tuple[int, int], ...]
    pair_count_by_class: tuple[tuple[ConflictClass, int], ...]
    pair_shared_points_by_class: tuple[tuple[ConflictClass, int], ...]
    multi_mask_points_by_class: tuple[tuple[ConflictClass, int], ...]
    multi_mask_point_rate_by_class: tuple[tuple[ConflictClass, float | None], ...]
    depth_layer_mask_count: int
    off_main_layer_fraction_weighted: float | None
    pairs: tuple[MaskPairConflict, ...]
    masks: tuple[MaskDepthConsistency, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report, with the definitions it was measured under."""

        def by_class(values: tuple[tuple[ConflictClass, Any], ...]) -> dict[str, Any]:
            return {conflict.value: value for conflict, value in values}

        return {
            "metric_version": self.metric_version,
            "definition": {
                "nested_containment_threshold": self.nested_containment_threshold,
                "duplicate_iou_threshold": self.duplicate_iou_threshold,
                "class_precedence": ["duplicate", "nested", "partial"],
                "point_severity": [conflict.value for conflict in reversed(_BY_SEVERITY)],
                "depth_layers": self.depth_layers.to_dict(),
            },
            "associated_point_count": self.associated_point_count,
            "membership_count": self.membership_count,
            "multi_mask_point_count": self.multi_mask_point_count,
            "multi_mask_point_rate": self.multi_mask_point_rate,
            "mean_regions_per_point": self.mean_regions_per_point,
            "max_regions_per_point": self.max_regions_per_point,
            "regions_per_point_histogram": {
                str(regions): points for regions, points in self.regions_per_point_histogram
            },
            "pair_count_by_class": by_class(self.pair_count_by_class),
            "pair_shared_points_by_class": by_class(self.pair_shared_points_by_class),
            "multi_mask_points_by_class": by_class(self.multi_mask_points_by_class),
            "multi_mask_point_rate_by_class": by_class(self.multi_mask_point_rate_by_class),
            "depth_layer_mask_count": self.depth_layer_mask_count,
            "off_main_layer_fraction_weighted": self.off_main_layer_fraction_weighted,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "masks": [mask.to_dict() for mask in self.masks],
        }


@dataclass(frozen=True, kw_only=True)
class OcclusionProximityReport:
    """How close the occluded points are to the depth that occluded them.

    Attributes:
        metric_version: :data:`ASSOCIATION_CONFLICT_METRIC_VERSION`.
        proximity_m: Largest ``depth - support_depth``, in meters, counted as near.
        occluded_point_count: Occluded points.
        near_support_point_count: Occluded points within ``proximity_m`` of their support.
        near_support_fraction: ``near_support_point_count / occluded_point_count``.
    """

    metric_version: str
    proximity_m: float
    occluded_point_count: int
    near_support_point_count: int
    near_support_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""
        return {
            "metric_version": self.metric_version,
            "proximity_m": self.proximity_m,
            "occluded_point_count": self.occluded_point_count,
            "near_support_point_count": self.near_support_point_count,
            "near_support_fraction": self.near_support_fraction,
        }


def off_main_layer_fraction(
    depths_m: NDArray[np.floating[Any]], criterion: DepthLayerCriterion
) -> float | None:
    """Return the fraction of one region's points outside its main depth layer.

    Args:
        depths_m: Depth of each associated point, in meters, under one depth metric.
        criterion: How depths split into layers and which layer is main.

    Returns:
        The fraction in ``[0, 1)``, or ``None`` below ``criterion.min_points`` points.
    """
    count = len(depths_m)
    if count < criterion.min_points:
        return None
    ordered = np.sort(np.asarray(depths_m, dtype=np.float64))
    jumps = np.diff(ordered) > np.maximum(criterion.gap_m, criterion.gap_ratio * ordered[:-1])
    layers = np.concatenate([[0], np.cumsum(jumps)])
    sizes = np.bincount(layers)
    dominant = np.flatnonzero(sizes >= criterion.main_layer_min_share * count)
    main = int(dominant[0]) if dominant.size else int(np.argmax(sizes))
    return float(1.0 - sizes[main] / count)


def measure_association_conflicts(
    membership: Mapping[str, NDArray[np.integer[Any]]],
    masks: Mapping[str, InlineMask],
    point_depth_m: NDArray[np.floating[Any]],
    *,
    nested_containment_threshold: float,
    duplicate_iou_threshold: float,
    depth_layers: DepthLayerCriterion,
) -> AssociationConflictReport:
    """Measure multi-membership, classify its region pairs and the depth consistency.

    Args:
        membership: For each region, the global indices of its associated points (unique).
        masks: The 2D mask of every region in ``membership``, all in one image space.
        point_depth_m: Depth of every point, in meters, indexed by global point index, under
            one depth metric (the association's own, for example the ray range).
        nested_containment_threshold: Containment, in ``(0, 1]``, that makes a pair nested.
        duplicate_iou_threshold: IoU, in ``(0, 1]``, that makes a pair a duplicate.
        depth_layers: The depth-layer criterion.

    Returns:
        The frame's conflict report.

    Raises:
        ValueError: If a threshold is outside ``(0, 1]``, a region has no mask, masks disagree
            on dimensions, a region repeats a point, or a point index has no finite depth.
    """
    for name, value in (
        ("nested_containment_threshold", nested_containment_threshold),
        ("duplicate_iou_threshold", duplicate_iou_threshold),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    region_ids = sorted(membership)
    members = [
        _validated_members(region_id, membership, masks, point_depth_m) for region_id in region_ids
    ]
    shapes = {masks[region_id].as_array().shape for region_id in region_ids}
    if len(shapes) > 1:
        raise ValueError(f"masks disagree on dimensions: {sorted(shapes)}")

    points = np.concatenate(members) if members else np.zeros(0, dtype=np.int64)
    regions = np.repeat(np.arange(len(region_ids)), [len(item) for item in members])
    order = np.lexsort((regions, points))
    points, regions = points[order], regions[order]
    _, starts, regions_per_point = np.unique(points, return_index=True, return_counts=True)
    associated = len(starts)

    pair_keys, pair_points = _shared_pairs(regions, starts, regions_per_point, len(region_ids))
    unique_keys, pair_of_incidence, shared_counts = np.unique(
        pair_keys, return_inverse=True, return_counts=True
    )
    pairs = []
    for key, shared in zip(unique_keys.tolist(), shared_counts.tolist(), strict=True):
        first, second = divmod(key, len(region_ids))
        pairs.append(
            _classified_pair(
                (region_ids[first], region_ids[second]),
                shared,
                masks,
                nested_containment_threshold=nested_containment_threshold,
                duplicate_iou_threshold=duplicate_iou_threshold,
            )
        )

    severity = np.full(associated, -1, dtype=np.int64)
    pair_severity = np.array([_SEVERITY[pair.conflict_class] for pair in pairs], dtype=np.int64)
    if len(pair_points):
        np.maximum.at(severity, pair_points, pair_severity[pair_of_incidence])
    points_by_class = {
        conflict: int((severity == rank).sum()) for conflict, rank in _SEVERITY.items()
    }
    multi = int((regions_per_point >= 2).sum())

    depth_rows = []
    weighted_sum, weight = 0.0, 0
    for region_id, item in zip(region_ids, members, strict=True):
        fraction = off_main_layer_fraction(point_depth_m[item], depth_layers)
        depth_rows.append(
            MaskDepthConsistency(
                region_id=region_id,
                associated_point_count=len(item),
                off_main_layer_fraction=fraction,
            )
        )
        if fraction is not None:
            weighted_sum += fraction * len(item)
            weight += len(item)

    histogram = np.bincount(regions_per_point) if associated else np.zeros(1, dtype=np.int64)
    return AssociationConflictReport(
        metric_version=ASSOCIATION_CONFLICT_METRIC_VERSION,
        nested_containment_threshold=nested_containment_threshold,
        duplicate_iou_threshold=duplicate_iou_threshold,
        depth_layers=depth_layers,
        associated_point_count=associated,
        membership_count=len(points),
        multi_mask_point_count=multi,
        multi_mask_point_rate=multi / associated if associated else None,
        mean_regions_per_point=len(points) / associated if associated else None,
        max_regions_per_point=int(regions_per_point.max()) if associated else 0,
        regions_per_point_histogram=tuple(
            (regions_count, int(count))
            for regions_count, count in enumerate(histogram)
            if regions_count > 0 and count
        ),
        pair_count_by_class=tuple(
            (conflict, sum(1 for pair in pairs if pair.conflict_class is conflict))
            for conflict in ConflictClass
        ),
        pair_shared_points_by_class=tuple(
            (
                conflict,
                sum(pair.shared_point_count for pair in pairs if pair.conflict_class is conflict),
            )
            for conflict in ConflictClass
        ),
        multi_mask_points_by_class=tuple(
            (conflict, points_by_class[conflict]) for conflict in ConflictClass
        ),
        multi_mask_point_rate_by_class=tuple(
            (conflict, points_by_class[conflict] / associated if associated else None)
            for conflict in ConflictClass
        ),
        depth_layer_mask_count=sum(
            1 for row in depth_rows if row.off_main_layer_fraction is not None
        ),
        off_main_layer_fraction_weighted=weighted_sum / weight if weight else None,
        pairs=tuple(pairs),
        masks=tuple(depth_rows),
    )


def measure_occlusion_proximity(
    depth_m: NDArray[np.floating[Any]],
    support_depth_m: NDArray[np.floating[Any]],
    occluded: NDArray[np.bool_],
    *,
    proximity_m: float,
) -> OcclusionProximityReport:
    """Measure how many occluded points lie close behind the depth that occluded them.

    An occluded point close to its supporting depth is more likely the same surface seen at a
    grazing angle (or thickened by accumulation) than a surface truly hidden behind another;
    this measures how often that happens, it does not decide it.

    Args:
        depth_m: ``(N,)`` depth of each point, in meters.
        support_depth_m: ``(N,)`` nearest supporting depth around each point, in meters,
            under the same metric; may be ``NaN`` for points that are not occluded.
        occluded: ``(N,)`` whether each point was occluded.
        proximity_m: Largest ``depth - support_depth``, in meters, counted as near.

    Returns:
        The occluded, near and fraction counts.

    Raises:
        ValueError: If the arrays disagree in shape, ``proximity_m`` is negative or not
            finite, or an occluded point lacks a finite depth or support depth.
    """
    if not math.isfinite(proximity_m) or proximity_m < 0:
        raise ValueError(f"proximity_m must be finite and non-negative, got {proximity_m!r}")
    if not (depth_m.shape == support_depth_m.shape == occluded.shape) or depth_m.ndim != 1:
        raise ValueError("depth_m, support_depth_m and occluded must share one (N,) shape")
    hidden = np.asarray(occluded, dtype=bool)
    gap = depth_m[hidden] - support_depth_m[hidden]
    if not np.isfinite(gap).all():
        raise ValueError("every occluded point needs a finite depth and support depth")
    near = int((gap <= proximity_m).sum())
    count = int(hidden.sum())
    return OcclusionProximityReport(
        metric_version=ASSOCIATION_CONFLICT_METRIC_VERSION,
        proximity_m=proximity_m,
        occluded_point_count=count,
        near_support_point_count=near,
        near_support_fraction=near / count if count else None,
    )


def _validated_members(
    region_id: str,
    membership: Mapping[str, NDArray[np.integer[Any]]],
    masks: Mapping[str, InlineMask],
    point_depth_m: NDArray[np.floating[Any]],
) -> NDArray[np.int64]:
    """Return one region's point indices after checking its mask, uniqueness and depths.

    Raises:
        ValueError: If the region has no mask, repeats a point, or names a point without a
            finite depth.
    """
    if region_id not in masks:
        raise ValueError(f"region {region_id!r} has associated points but no mask")
    item = np.asarray(membership[region_id], dtype=np.int64)
    if item.ndim != 1:
        raise ValueError(f"points of region {region_id!r} must be a flat index array")
    if len(np.unique(item)) != len(item):
        raise ValueError(f"region {region_id!r} repeats a point index")
    if len(item) and (item.min() < 0 or item.max() >= len(point_depth_m)):
        raise ValueError(f"region {region_id!r} names a point outside point_depth_m")
    if not np.isfinite(point_depth_m[item]).all():
        raise ValueError(f"region {region_id!r} names a point without a finite depth")
    return item


def _shared_pairs(
    regions: NDArray[np.int64],
    starts: NDArray[np.int64],
    regions_per_point: NDArray[np.int64],
    region_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Enumerate, for every multi-mask point, each pair of its regions.

    ``regions`` is sorted by point and, within a point, by region, so the regions of point
    ``p`` are ``regions[starts[p] : starts[p] + regions_per_point[p]]`` in ascending order.

    Returns:
        ``(pair_keys, pair_points)``: one entry per (point, region pair), the key encoding the
        pair as ``first * region_count + second`` with ``first < second``.
    """
    keys: list[NDArray[np.int64]] = []
    owners: list[NDArray[np.int64]] = []
    for size in np.unique(regions_per_point).tolist():
        if size < 2:
            continue
        points = np.flatnonzero(regions_per_point == size)
        block = regions[starts[points][:, None] + np.arange(size)]
        for first in range(size):
            for second in range(first + 1, size):
                keys.append(block[:, first] * region_count + block[:, second])
                owners.append(points)
    if not keys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(keys), np.concatenate(owners)


def _classified_pair(
    region_ids: tuple[str, str],
    shared: int,
    masks: Mapping[str, InlineMask],
    *,
    nested_containment_threshold: float,
    duplicate_iou_threshold: float,
) -> MaskPairConflict:
    first, second = (masks[region_id].as_array() for region_id in region_ids)
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    smaller = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    iou = intersection / union if union else 0.0
    containment = intersection / smaller if smaller else 0.0
    if iou >= duplicate_iou_threshold:
        conflict = ConflictClass.DUPLICATE
    elif containment >= nested_containment_threshold:
        conflict = ConflictClass.NESTED
    else:
        conflict = ConflictClass.PARTIAL
    return MaskPairConflict(
        region_ids=region_ids,
        shared_point_count=shared,
        iou_2d=iou,
        containment_2d=containment,
        conflict_class=conflict,
    )
