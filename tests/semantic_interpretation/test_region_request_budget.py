"""The region request budget: only the largest regions of a frame are interpreted, by rule."""

from dataclasses import replace

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    InlineMask,
    MaskStoreError,
    Region2D,
    RegionId,
)
from contextmap.semantic_interpretation import (
    RegionRequestBudget,
    SemanticModePrompt,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.shared import BackendProvenance

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="0"
)
_PROMPT = SemanticModePrompt(template_id="region/v1", output_schema="semantic-response/1")
_VIEWS = SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,))


def _region(region_id: str, area: int) -> Region2D:
    mask = np.zeros((40, 40), dtype=bool)
    mask.reshape(-1)[:area] = True
    return Region2D(
        region_id=RegionId(region_id),
        bounding_box=BoundingBox2D(x=0, y=0, width=40, height=40),
        provenance=_PROVENANCE,
        mask=InlineMask(mask),
        image_width=40,
        image_height=40,
        source_observation_id=SourceObservationId("frame-0000"),
    )


def test_the_budget_keeps_the_largest_regions_and_breaks_ties_by_region_id() -> None:
    regions = [
        _region("region-0001", 30),
        _region("region-0002", 90),
        _region("region-0003", 60),
        _region("region-0004", 60),
    ]

    selected = RegionRequestBudget(max_regions=3).select(regions)

    assert selected == frozenset({RegionId("region-0002"), RegionId("region-0003"), "region-0004"})


def test_a_frame_within_the_budget_keeps_every_region() -> None:
    regions = [_region("region-0001", 30), _region("region-0002", 90)]

    assert RegionRequestBudget(max_regions=8).select(regions) == {"region-0001", "region-0002"}


def test_the_budget_refuses_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="max_regions"):
        RegionRequestBudget(max_regions=0)


def test_the_budget_is_part_of_the_request_policy_identity() -> None:
    without = SemanticRequestPolicy(views=_VIEWS, region=_PROMPT)
    with_budget = SemanticRequestPolicy(
        views=_VIEWS, region=_PROMPT, region_budget=RegionRequestBudget(max_regions=8)
    )

    assert with_budget.to_document()["region_budget"] == {
        "policy_id": "largest-area-regions-v1",
        "max_regions": 8,
    }
    assert without.to_document()["region_budget"] is None
    assert with_budget.fingerprint() != without.fingerprint()


def test_a_budget_needs_region_requests_to_limit() -> None:
    with pytest.raises(ValueError, match="region"):
        SemanticRequestPolicy(
            views=_VIEWS,
            scene=_PROMPT,
            region=None,
            region_budget=RegionRequestBudget(max_regions=8),
        )


class _Resolver:
    def __init__(self, masks: dict[str, InlineMask]) -> None:
        self._masks = masks
        self.loads: list[str] = []

    def resolve_mask(self, reference: str) -> InlineMask:
        self.loads.append(reference)
        return self._masks[reference]


def _externalized(regions: list[Region2D]) -> tuple[list[Region2D], _Resolver]:
    masks = {f"outputs/masks/{r.region_id}.npy": r.mask for r in regions if r.mask is not None}
    return (
        [
            replace(region, mask=None, mask_reference=f"outputs/masks/{region.region_id}.npy")
            for region in regions
        ],
        _Resolver(masks),
    )


def test_a_referenced_mask_ranks_exactly_as_the_inline_mask_did() -> None:
    inline = [
        _region("region-0001", 30),
        _region("region-0002", 90),
        _region("region-0003", 60),
        _region("region-0004", 60),
    ]
    referenced, resolver = _externalized(inline)
    budget = RegionRequestBudget(max_regions=3)

    assert budget.select(referenced, mask_resolver=resolver) == budget.select(inline)


def test_a_recorded_area_ranks_without_loading_any_pixels() -> None:
    inline = [replace(_region("region-0001", 30), area_pixels=500.0), _region("region-0002", 90)]
    referenced, resolver = _externalized(inline)

    selected = RegionRequestBudget(max_regions=1).select(referenced, mask_resolver=resolver)

    assert selected == {"region-0001"}
    assert resolver.loads == ["outputs/masks/region-0002.npy"]


def test_a_referenced_region_without_a_resolver_is_an_explicit_error() -> None:
    referenced, _ = _externalized([_region("region-0001", 30)])

    with pytest.raises(MaskStoreError, match="no resolver"):
        RegionRequestBudget(max_regions=1).select(referenced)
