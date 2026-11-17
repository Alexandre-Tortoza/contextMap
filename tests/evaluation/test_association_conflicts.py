"""Conflict taxonomy of 2D->3D membership (C5), on small synthetic scenes."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from contextmap.evaluation.association_conflicts import (
    AssociationConflictReport,
    ConflictClass,
    DepthLayerCriterion,
    measure_association_conflicts,
    measure_occlusion_proximity,
    off_main_layer_fraction,
)
from contextmap.region_perception import InlineMask

HEIGHT, WIDTH = 10, 20
LAYERS = DepthLayerCriterion(gap_m=0.3, gap_ratio=0.08, main_layer_min_share=0.2, min_points=30)


def _mask(x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pixels[y0:y1, x0:x1] = True
    return InlineMask(pixels)


def _members(*indices: int) -> NDArray[np.int64]:
    return np.array(indices, dtype=np.int64)


def _conflicts(
    membership: Mapping[str, NDArray[np.int64]],
    masks: Mapping[str, InlineMask],
    depth: NDArray[np.float64] | None = None,
) -> AssociationConflictReport:
    return measure_association_conflicts(
        membership,
        masks,
        np.full(64, 2.0) if depth is None else depth,
        nested_containment_threshold=0.9,
        duplicate_iou_threshold=0.8,
        depth_layers=LAYERS,
    )


def test_a_point_in_three_masks_counts_three_regions() -> None:
    masks = {"a": _mask(0, 0, 10, 10), "b": _mask(0, 0, 4, 4), "c": _mask(8, 0, 14, 4)}
    report = _conflicts({"a": _members(0, 1, 2), "b": _members(0, 1), "c": _members(0)}, masks)

    assert report.associated_point_count == 3
    assert report.membership_count == 6
    assert report.multi_mask_point_count == 2
    assert report.multi_mask_point_rate == pytest.approx(2 / 3)
    assert report.max_regions_per_point == 3
    assert report.mean_regions_per_point == pytest.approx(2.0)
    assert dict(report.regions_per_point_histogram) == {1: 1, 2: 1, 3: 1}


def test_a_point_shared_by_nested_masks_is_nested_not_a_conflict() -> None:
    masks = {"door": _mask(0, 0, 10, 10), "handle": _mask(2, 2, 4, 4)}
    report = _conflicts({"door": _members(0, 1, 2), "handle": _members(1)}, masks)

    pair = report.pairs[0]
    assert pair.region_ids == ("door", "handle")
    assert pair.shared_point_count == 1
    assert pair.containment_2d == pytest.approx(1.0)
    assert pair.conflict_class is ConflictClass.NESTED
    assert dict(report.multi_mask_points_by_class) == {
        ConflictClass.NESTED: 1,
        ConflictClass.DUPLICATE: 0,
        ConflictClass.PARTIAL: 0,
    }


def test_near_identical_masks_are_a_duplicate_pair_even_though_nested() -> None:
    masks = {"a": _mask(0, 0, 10, 10), "b": _mask(0, 0, 10, 9)}
    report = _conflicts({"a": _members(0, 1), "b": _members(0, 1)}, masks)

    assert report.pairs[0].iou_2d == pytest.approx(0.9)
    assert report.pairs[0].conflict_class is ConflictClass.DUPLICATE
    assert dict(report.pair_shared_points_by_class)[ConflictClass.DUPLICATE] == 2


def test_partially_overlapping_masks_are_a_partial_conflict() -> None:
    masks = {"a": _mask(0, 0, 10, 10), "b": _mask(5, 0, 15, 10)}
    report = _conflicts({"a": _members(0, 1), "b": _members(1, 2)}, masks)

    assert report.pairs[0].conflict_class is ConflictClass.PARTIAL
    assert dict(report.pair_count_by_class)[ConflictClass.PARTIAL] == 1


def test_a_point_takes_the_most_severe_class_of_its_pairs() -> None:
    # outer ⊃ inner (aninhado) e side cruza outer (parcial): o ponto 0 está nos três.
    masks = {"outer": _mask(0, 0, 10, 10), "inner": _mask(2, 2, 4, 4), "side": _mask(8, 0, 18, 10)}
    report = _conflicts(
        {"outer": _members(0, 1), "inner": _members(0, 1), "side": _members(0)}, masks
    )

    by_class = dict(report.multi_mask_points_by_class)
    assert by_class[ConflictClass.PARTIAL] == 1
    assert by_class[ConflictClass.NESTED] == 1
    assert sum(by_class.values()) == report.multi_mask_point_count
    rates = dict(report.multi_mask_point_rate_by_class)
    assert rates[ConflictClass.PARTIAL] == pytest.approx(0.5)


def test_pairs_that_share_no_point_are_not_classified() -> None:
    masks = {"a": _mask(0, 0, 10, 10), "b": _mask(5, 0, 15, 10)}
    report = _conflicts({"a": _members(0), "b": _members(1)}, masks)

    assert report.pairs == ()
    assert report.multi_mask_point_rate == 0.0


def test_two_depth_layers_put_the_far_layer_off_the_main_one() -> None:
    depths = np.concatenate([np.full(40, 2.0), np.full(10, 5.0)])

    assert off_main_layer_fraction(depths, LAYERS) == pytest.approx(0.2)


def test_the_main_layer_is_the_nearest_with_enough_share_not_the_largest() -> None:
    depths = np.concatenate([np.full(10, 2.0), np.full(30, 5.0)])

    assert off_main_layer_fraction(depths, LAYERS) == pytest.approx(0.75)


def test_a_continuous_surface_is_one_layer() -> None:
    # Parede rasante: alcances crescentes sem salto maior que max(0,3 m; 8 %).
    depths = np.linspace(1.0, 6.0, 60)

    assert off_main_layer_fraction(depths, LAYERS) == 0.0


def test_too_few_points_have_no_layer_measurement() -> None:
    assert off_main_layer_fraction(np.full(29, 2.0), LAYERS) is None


def test_without_a_dominant_layer_the_largest_one_is_main() -> None:
    # Seis camadas de 1/6 cada: nenhuma chega a 20 %; a maior (a mais próxima no empate) vale.
    depths = np.repeat(np.array([1.0, 3.0, 6.0, 10.0, 15.0, 22.0]), [6, 6, 6, 6, 6, 7])

    assert off_main_layer_fraction(depths, LAYERS) == pytest.approx(1 - 7 / 37)


def test_depth_consistency_is_weighted_by_associated_points() -> None:
    depth = np.concatenate([np.full(40, 2.0), np.full(10, 5.0), np.full(30, 3.0), [9.0]])
    masks = {"front": _mask(0, 0, 10, 10), "wall": _mask(10, 0, 20, 10), "tiny": _mask(0, 0, 1, 1)}
    membership = {
        "front": np.arange(0, 50),
        "wall": np.arange(50, 80),
        "tiny": _members(80),
    }

    report = _conflicts(membership, masks, depth)

    by_region = {row.region_id: row for row in report.masks}
    assert by_region["front"].off_main_layer_fraction == pytest.approx(0.2)
    assert by_region["wall"].off_main_layer_fraction == 0.0
    assert by_region["tiny"].off_main_layer_fraction is None
    assert report.depth_layer_mask_count == 2
    assert report.off_main_layer_fraction_weighted == pytest.approx(0.2 * 50 / 80)


def test_report_is_json_encodable_and_declares_its_definition() -> None:
    masks = {"door": _mask(0, 0, 10, 10), "handle": _mask(2, 2, 4, 4)}
    report = _conflicts({"door": _members(0, 1), "handle": _members(1)}, masks)

    record = json.loads(json.dumps(report.to_dict()))

    assert record["definition"]["nested_containment_threshold"] == 0.9
    assert record["definition"]["depth_layers"]["min_points"] == 30
    assert record["multi_mask_points_by_class"] == {"nested": 1, "duplicate": 0, "partial": 0}
    assert record["pairs"][0]["conflict_class"] == "nested"


@pytest.mark.parametrize(
    ("membership", "masks", "message"),
    [
        ({"a": _members(0)}, {}, "mask"),
        ({"a": _members(0, 0)}, {"a": _mask(0, 0, 2, 2)}, "repeat"),
        ({"a": _members(64)}, {"a": _mask(0, 0, 2, 2)}, "depth"),
    ],
)
def test_invalid_membership_is_rejected(
    membership: dict[str, NDArray[np.int64]], masks: dict[str, InlineMask], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _conflicts(membership, masks)


def test_occlusion_proximity_counts_occluded_points_near_their_support() -> None:
    depth = np.array([1.0, 1.4, 1.6, 3.0, 2.0])
    support = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
    occluded = np.array([False, True, True, True, False])

    report = measure_occlusion_proximity(depth, support, occluded, proximity_m=0.5)

    assert report.occluded_point_count == 3
    assert report.near_support_point_count == 1
    assert report.near_support_fraction == pytest.approx(1 / 3)
    assert json.loads(json.dumps(report.to_dict()))["proximity_m"] == 0.5


def test_occlusion_proximity_without_occlusion_has_no_population() -> None:
    report = measure_occlusion_proximity(
        np.array([1.0]), np.array([1.0]), np.array([False]), proximity_m=0.5
    )

    assert report.near_support_fraction is None


def test_occluded_points_need_a_support_depth() -> None:
    with pytest.raises(ValueError, match="support"):
        measure_occlusion_proximity(
            np.array([2.0]), np.array([np.nan]), np.array([True]), proximity_m=0.5
        )
