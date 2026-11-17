"""Structural metrics of one frame's region masks (C4), on small synthetic masks."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from contextmap.evaluation.annotations import RegionAnnotation
from contextmap.evaluation.region_structure import (
    MASK_CONNECTIVITY,
    RegionStructureReport,
    connected_component_count,
    known_object_recall,
    measure_region_structure,
)
from contextmap.region_perception import BoundingBox, InlineMask

HEIGHT, WIDTH = 12, 16


def _mask(*boxes: tuple[int, int, int, int]) -> InlineMask:
    """Union of half-open ``(x0, y0, x1, y1)`` rectangles on the test canvas."""
    pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        pixels[y0:y1, x0:x1] = True
    return InlineMask(pixels)


def _structure(masks: dict[str, InlineMask], **kwargs: Any) -> RegionStructureReport:
    options: dict[str, Any] = {"containment_threshold": 0.8, "duplicate_iou_threshold": 0.8}
    options.update(kwargs)
    return measure_region_structure(masks, **options)


def test_a_ring_is_one_component() -> None:
    ring = np.zeros((HEIGHT, WIDTH), dtype=bool)
    ring[2:9, 3:10] = True
    ring[4:7, 5:8] = False

    assert connected_component_count(InlineMask(ring)) == 1


def test_speckle_counts_every_isolated_pixel() -> None:
    speckle = np.zeros((HEIGHT, WIDTH), dtype=bool)
    speckle[::3, ::3] = True

    assert connected_component_count(InlineMask(speckle)) == int(speckle.sum())


def test_diagonal_neighbours_are_separate_components_under_edge_connectivity() -> None:
    diagonal = np.zeros((HEIGHT, WIDTH), dtype=bool)
    diagonal[1, 1] = diagonal[2, 2] = True

    assert MASK_CONNECTIVITY == 4
    assert connected_component_count(InlineMask(diagonal)) == 2


def test_a_snake_joining_runs_through_later_rows_is_one_component() -> None:
    # Duas colunas só se unem pela linha de baixo: exige propagar a união entre linhas.
    snake = np.zeros((HEIGHT, WIDTH), dtype=bool)
    snake[0:10, 1] = True
    snake[0:10, 5] = True
    snake[0:10, 9] = True
    snake[9, 1:6] = True
    snake[0, 5:10] = True

    assert connected_component_count(InlineMask(snake)) == 1


def test_nested_masks_give_nesting_depth_two_and_one_contained_pair() -> None:
    report = _structure({"outer": _mask((0, 0, 8, 8)), "inner": _mask((2, 2, 4, 4))})

    assert report.max_nesting_depth == 2
    assert dict(report.nesting_depth_histogram) == {1: 60, 2: 4}
    assert report.covered_pixel_count == 64
    assert report.pixels_with_multiple_masks == 4
    assert report.multiple_mask_pixel_fraction == pytest.approx(4 / 64)
    assert report.contained_pair_count == 1
    assert report.masks_contained_in_another == 1
    assert report.duplicate_pair_count == 0
    inner = next(mask for mask in report.masks if mask.region_id == "inner")
    assert inner.max_fraction_inside_other == pytest.approx(1.0)
    assert inner.most_containing_region_id == "outer"


def test_identical_masks_are_a_duplicate_pair_and_both_are_contained() -> None:
    report = _structure({"a": _mask((0, 0, 4, 4)), "b": _mask((0, 0, 4, 4))})

    assert report.duplicate_pair_count == 1
    assert report.contained_pair_count == 1
    assert report.masks_contained_in_another == 2


def test_partial_overlap_is_neither_contained_nor_duplicate() -> None:
    report = _structure({"a": _mask((0, 0, 4, 4)), "b": _mask((2, 0, 6, 4))})

    assert report.overlapping_pair_count == 1
    assert report.contained_pair_count == 0
    assert report.duplicate_pair_count == 0
    assert report.masks_contained_in_another == 0


def test_fragmentation_rate_counts_masks_with_more_than_one_component() -> None:
    report = _structure({"solid": _mask((0, 0, 3, 3)), "split": _mask((5, 0, 6, 1), (8, 0, 9, 1))})

    assert report.fragmented_mask_count == 1
    assert report.fragmentation_rate == pytest.approx(0.5)
    components = {mask.region_id: mask.component_count for mask in report.masks}
    assert components == {"solid": 1, "split": 2}


def test_area_quantiles_use_linear_interpolation() -> None:
    masks = {f"m{side}": _mask((0, 0, side, 1)) for side in (1, 2, 3, 4, 5)}

    report = _structure(masks)

    assert report.mask_count == 5
    assert report.area_px_median == pytest.approx(3.0)
    assert report.area_px_p10 == pytest.approx(1.4)
    assert report.area_px_p90 == pytest.approx(4.6)


def test_accepted_mask_ratio_needs_the_proposal_count() -> None:
    masks = {"a": _mask((0, 0, 2, 2)), "b": _mask((4, 4, 6, 6))}

    assert _structure(masks).accepted_mask_ratio is None
    assert _structure(masks, proposal_count=8).accepted_mask_ratio == pytest.approx(0.25)
    with pytest.raises(ValueError, match="proposal_count"):
        _structure(masks, proposal_count=1)


def test_usable_area_coverage_is_the_covered_share_of_usable_pixels() -> None:
    usable = _mask((0, 0, 4, 4))
    report = _structure({"a": _mask((0, 0, 2, 4))}, usable_area=usable)

    assert report.usable_pixel_count == 16
    assert report.usable_area_coverage == pytest.approx(0.5)
    assert _structure({"a": _mask((0, 0, 2, 4))}).usable_area_coverage is None


def test_an_empty_frame_reports_no_population_instead_of_zero() -> None:
    report = _structure({})

    assert report.mask_count == 0
    assert report.fragmentation_rate is None
    assert report.area_px_median is None
    assert report.multiple_mask_pixel_fraction is None
    assert report.max_nesting_depth == 0


def test_report_is_json_encodable_and_declares_its_definition() -> None:
    report = _structure(
        {"outer": _mask((0, 0, 8, 8)), "inner": _mask((2, 2, 4, 4))},
        proposal_count=5,
        usable_area=_mask((0, 0, 16, 12)),
    )

    record = json.loads(json.dumps(report.to_dict()))

    assert record["metric_version"] == report.metric_version
    assert record["definition"]["connectivity"] == 4
    assert record["definition"]["containment_threshold"] == 0.8
    assert record["nesting_depth_histogram"] == {"1": 60, "2": 4}
    assert [mask["region_id"] for mask in record["masks"]] == ["inner", "outer"]


@pytest.mark.parametrize(
    ("masks", "kwargs", "message"),
    [
        ({"a": InlineMask(np.zeros((2, 2), dtype=bool))}, {}, "empty"),
        (
            {"a": _mask((0, 0, 1, 1)), "b": InlineMask(np.ones((2, 2), dtype=bool))},
            {},
            "dimensions",
        ),
        ({"a": _mask((0, 0, 1, 1))}, {"containment_threshold": 0.0}, "containment_threshold"),
        ({"a": _mask((0, 0, 1, 1))}, {"duplicate_iou_threshold": 1.5}, "duplicate_iou"),
        (
            {"a": _mask((0, 0, 1, 1))},
            {"usable_area": InlineMask(np.ones((2, 2), dtype=bool))},
            "usable_area",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    masks: dict[str, InlineMask], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _structure(masks, **kwargs)


def test_known_object_recall_matches_masks_by_mask_iou() -> None:
    masks = {"door": _mask((0, 0, 4, 8)), "handle": _mask((6, 2, 8, 4))}
    objects = [
        RegionAnnotation(region_id="door-1", mask=_mask((0, 0, 4, 8))),
        RegionAnnotation(region_id="window-1", mask=_mask((10, 0, 16, 2))),
    ]

    recall = known_object_recall(masks, objects, match_iou_threshold=0.5)

    assert recall.object_count == 2
    assert recall.recalled_count == 1
    assert recall.recall == pytest.approx(0.5)
    by_object = {match.object_id: match for match in recall.matches}
    assert by_object["door-1"].best_iou == pytest.approx(1.0)
    assert by_object["door-1"].best_region_id == "door"
    assert by_object["door-1"].geometry == "mask"
    assert by_object["window-1"].best_iou == 0.0
    assert by_object["window-1"].best_region_id is None
    assert not by_object["window-1"].recalled


def test_box_only_objects_match_the_tight_box_of_each_mask() -> None:
    # Um L cheio só 3/4 da caixa: a caixa justa ainda coincide com a anotação.
    masks = {"l-shape": _mask((0, 0, 2, 4), (0, 2, 4, 4))}
    objects = [
        RegionAnnotation(region_id="object", box=BoundingBox(x_min=0, y_min=0, x_max=4, y_max=4))
    ]

    recall = known_object_recall(masks, objects, match_iou_threshold=0.5)

    assert recall.matches[0].geometry == "box"
    assert recall.matches[0].best_iou == pytest.approx(1.0)
    assert recall.recall == pytest.approx(1.0)


def test_known_object_recall_without_objects_has_no_population() -> None:
    recall = known_object_recall({"a": _mask((0, 0, 2, 2))}, [], match_iou_threshold=0.5)

    assert recall.recall is None
    assert json.loads(json.dumps(recall.to_dict()))["recall"] is None
