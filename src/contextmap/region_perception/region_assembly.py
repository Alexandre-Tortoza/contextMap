"""Ordered composition of independent regional evidence sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from contextmap.region_perception.region_evidence import Region2D

if TYPE_CHECKING:
    from contextmap.region_perception.grounding.region_grounding import RegionGroundingExecution
    from contextmap.region_perception.refinement.region_refinement import RegionRefinementExecution

REGION_SET_ASSEMBLY_POLICY = "region-set-assembly/1"
"""Discovery first, then query-ordered boxes, then prompt-ordered accepted refined masks."""


def assemble_regions(
    *,
    discovered: Sequence[Region2D],
    groundings: Sequence[RegionGroundingExecution] = (),
    refinements: Sequence[RegionRefinementExecution] = (),
    policy_id: str = REGION_SET_ASSEMBLY_POLICY,
) -> tuple[Region2D, ...]:
    """Preserve each successful source's evidence in order, rejecting identity collisions.

    Regions are never merged here: a grounded box and a refined mask make distinct
    scientific statements, even when their extents overlap.
    """
    if policy_id != REGION_SET_ASSEMBLY_POLICY:
        raise ValueError(f"unsupported region assembly policy: {policy_id!r}")
    regions = (
        *discovered,
        *(region for execution in groundings for region in execution.regions),
        *(region for execution in refinements for region in execution.regions),
    )
    seen: set[str] = set()
    for region in regions:
        if not isinstance(region, Region2D):
            raise TypeError("region assembly requires Region2D values")
        if str(region.region_id) in seen:
            raise ValueError(
                f"the region set holds region {region.region_id!r} twice; "
                "region sources must keep their identities apart"
            )
        seen.add(str(region.region_id))
    return regions
