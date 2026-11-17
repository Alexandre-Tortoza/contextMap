"""Versioned one-to-one join between reference regions and run regions (#658)."""

from collections.abc import Iterable

import numpy as np
import pytest

from contextmap.evaluation.annotations import (
    Coverage,
    FrameRegionAnnotation,
    RegionAnnotation,
)
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.region_join import (
    RegionJoinError,
    RegionJoinPolicy,
    join_frame_regions,
    summarize_region_joins,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox, BoundingBox2D, InlineMask, Region2D, RegionId
from contextmap.shared import BackendProvenance

SAMPLE = ReferenceSampleId("sample-0001")
FRAME = SourceObservationId("frame-0001")
WIDTH, HEIGHT = 6, 4
DISCOVERY = BackendProvenance(
    backend_id="fake_discovery", capability="region_discovery", provider="f", model="f", version="1"
)
POLICY = RegionJoinPolicy(policy_id="reference-region-join/test", min_iou=0.3)


def _mask(pixels: Iterable[tuple[int, int]]) -> InlineMask:
    array = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for x, y in pixels:
        array[y, x] = True
    return InlineMask(array)


def _row(y: int, *columns: int) -> set[tuple[int, int]]:
    return {(x, y) for x in columns}


def _reference(region_id: str, pixels: Iterable[tuple[int, int]]) -> RegionAnnotation:
    return RegionAnnotation(region_id=region_id, mask=_mask(pixels))


def _frame(*regions: RegionAnnotation) -> FrameRegionAnnotation:
    return FrameRegionAnnotation(
        sample_id=SAMPLE,
        observation_id=FRAME,
        image_width=WIDTH,
        image_height=HEIGHT,
        coverage=Coverage.PARTIAL,
        regions=regions,
    )


def _run(region_id: str, pixels: Iterable[tuple[int, int]]) -> Region2D:
    mask = _mask(pixels)
    ys, xs = np.nonzero(mask.as_array())
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=BoundingBox2D(
            x=float(xs.min()),
            y=float(ys.min()),
            width=float(xs.max() - xs.min() + 1),
            height=float(ys.max() - ys.min() + 1),
        ),
        provenance=DISCOVERY,
        source_observation_id=FRAME,
        image_width=WIDTH,
        image_height=HEIGHT,
        mask=mask,
    )


# Referência "a" coincide com o run r1; "b" cobre metade de r1 (IoU 0.5) e 2/5 de r2 (IoU 0.4).
REF_A = _reference("ref-a", _row(0, 0, 1, 2, 3))
REF_B = _reference("ref-b", _row(0, 0, 1))
RUN_1 = _run("r1", _row(0, 0, 1, 2, 3))
RUN_2 = _run("r2", _row(0, 0, 1) | _row(1, 0, 1, 2))


def test_the_join_is_one_to_one_greedy_by_descending_iou() -> None:
    join = join_frame_regions(POLICY, _frame(REF_A, REF_B), (RUN_1, RUN_2))

    # O melhor IoU de "b" é r1, mas r1 já foi para "a" (IoU 1): "b" fica com r2 (IoU 0.4).
    assert [(item.reference_region_id, item.run_region_id) for item in join.matches] == [
        ("ref-a", "r1"),
        ("ref-b", "r2"),
    ]
    assert join.matches[1].iou == pytest.approx(0.4)
    assert join.reference_region_of(RegionId("r2")) == "ref-b"
    assert (join.unmatched_reference_region_ids, join.unmatched_run_region_ids) == ((), ())
    assert (join.policy_id, join.sample_id, join.observation_id) == (
        POLICY.policy_id,
        SAMPLE,
        FRAME,
    )


def test_a_pair_below_the_policy_threshold_is_never_joined() -> None:
    strict = RegionJoinPolicy(policy_id="reference-region-join/strict", min_iou=0.45)

    join = join_frame_regions(strict, _frame(REF_A, REF_B), (RUN_1, RUN_2))

    assert [(item.reference_region_id, item.run_region_id) for item in join.matches] == [
        ("ref-a", "r1")
    ]
    assert join.unmatched_reference_region_ids == ("ref-b",)
    assert join.unmatched_run_region_ids == ("r2",)
    assert join.reference_region_of(RegionId("r2")) is None


def test_equal_ious_are_broken_by_reference_then_run_region_id() -> None:
    same = _row(2, 1, 2, 3)

    by_reference = join_frame_regions(
        POLICY, _frame(_reference("ref-z", same), _reference("ref-y", same)), (_run("r1", same),)
    )
    by_run = join_frame_regions(
        POLICY, _frame(_reference("ref-a", same)), (_run("r9", same), _run("r3", same))
    )

    assert [item.reference_region_id for item in by_reference.matches] == ["ref-y"]
    assert by_reference.unmatched_reference_region_ids == ("ref-z",)
    assert [item.run_region_id for item in by_run.matches] == ["r3"]
    assert by_run.unmatched_run_region_ids == ("r9",)


def test_boxes_are_rasterized_at_pixel_centres_on_both_sides() -> None:
    reference = RegionAnnotation(
        region_id="ref-box", box=BoundingBox(x_min=0.0, y_min=0.0, x_max=2.0, y_max=2.0)
    )
    box_only = Region2D(
        region_id=RegionId("r-box"),
        bounding_box=BoundingBox2D(x=0.2, y=0.2, width=1.6, height=1.6),
        provenance=DISCOVERY,
        source_observation_id=FRAME,
        image_width=WIDTH,
        image_height=HEIGHT,
    )

    join = join_frame_regions(POLICY, _frame(reference), (box_only,))

    # Os centros 0.5 e 1.5 caem nas duas caixas: os mesmos quatro pixels, IoU 1.
    assert join.matches[0].iou == 1.0


def test_a_persisted_region_whose_mask_was_not_loaded_is_refused() -> None:
    persisted = Region2D(
        region_id=RegionId("r1"),
        bounding_box=BoundingBox2D(x=0.0, y=0.0, width=4.0, height=1.0),
        provenance=DISCOVERY,
        mask_reference="outputs/masks/frame-0001/r1.npy",
        source_observation_id=FRAME,
        image_width=WIDTH,
        image_height=HEIGHT,
    )

    # Ler a caixa no lugar da máscara mudaria o IoU sem aviso.
    with pytest.raises(RegionJoinError, match="mask"):
        join_frame_regions(POLICY, _frame(REF_A), (persisted,))


def test_a_run_region_of_another_observation_or_image_is_refused() -> None:
    other_frame = Region2D(
        region_id=RegionId("r1"),
        bounding_box=BoundingBox2D(x=0.0, y=0.0, width=1.0, height=1.0),
        provenance=DISCOVERY,
        source_observation_id=SourceObservationId("frame-0002"),
        image_width=WIDTH,
        image_height=HEIGHT,
    )
    other_size = Region2D(
        region_id=RegionId("r1"),
        bounding_box=BoundingBox2D(x=0.0, y=0.0, width=1.0, height=1.0),
        provenance=DISCOVERY,
        source_observation_id=FRAME,
        image_width=WIDTH * 2,
        image_height=HEIGHT,
    )

    with pytest.raises(RegionJoinError, match="observation"):
        join_frame_regions(POLICY, _frame(REF_A), (other_frame,))
    with pytest.raises(RegionJoinError, match="dimensions"):
        join_frame_regions(POLICY, _frame(REF_A), (other_size,))


def test_the_summary_counts_matched_and_unmatched_regions_of_one_policy() -> None:
    first = join_frame_regions(POLICY, _frame(REF_A, REF_B), (RUN_1, RUN_2))
    second = join_frame_regions(
        POLICY, _frame(_reference("ref-c", _row(3, 4, 5))), (_run("r7", _row(3, 0)),)
    )

    summary = summarize_region_joins((first, second))

    assert summary.policy_id == POLICY.policy_id
    assert (summary.frame_count, summary.matched_count) == (2, 2)
    assert (summary.unmatched_reference_count, summary.unmatched_run_count) == (1, 1)
    other = join_frame_regions(
        RegionJoinPolicy(policy_id="reference-region-join/other", min_iou=0.3),
        _frame(REF_A),
        (RUN_1,),
    )
    with pytest.raises(RegionJoinError, match="policy"):
        summarize_region_joins((first, other))


@pytest.mark.parametrize("min_iou", [0.0, -0.1, 1.5, float("nan")])
def test_the_policy_declares_an_iou_threshold_in_the_unit_interval(min_iou: float) -> None:
    with pytest.raises(ValueError, match="min_iou"):
        RegionJoinPolicy(policy_id="reference-region-join/test", min_iou=min_iou)
