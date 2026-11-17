"""Versioned one-to-one join between reference regions and the regions of a perception run.

A reference annotation names regions of the **reference** (family ``regions``), while every
claim and score of a run names a ``region_id`` local to its perception run. Before a reference
expectation can grade a claim, each run region must be joined to at most one reference region,
and the same join must serve every compared arm (#653, gap 2; #658).

The join measures overlap with the Region Discovery rule (``region.iou.mean``): pixel sets,
an inline mask giving its foreground and a box giving the pixels whose centre lies inside it.
It is greedy and one-to-one: every pair at or above :attr:`RegionJoinPolicy.min_iou` is taken
by descending IoU, ties broken by reference region id and then run region id, and a pair is
skipped when either side is already joined. A run region left unmatched has no reference
expectation; a reference region left unmatched is counted, never dropped. See
``src/contextmap/evaluation/docs/semantic-scoring.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.annotations import FrameRegionAnnotation, RegionAnnotation
from contextmap.evaluation.reference_set import ReferenceSampleId
from contextmap.evaluation.region_discovery import (
    box_pixels,
    mask_pixels,
    pixel_set_iou,
    region_pixels,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import Region2D, RegionId


class RegionJoinError(ValueError):
    """Raised when reference and run regions cannot be joined as declared."""


@dataclass(frozen=True, kw_only=True)
class RegionJoinPolicy:
    """The declared rule that joins reference regions to run regions; no hidden default.

    Attributes:
        policy_id: Version of the policy, cited by every join and report.
        min_iou: A pair is joined only at or above this IoU, in ``(0, 1]``.
    """

    policy_id: str
    min_iou: float

    def __post_init__(self) -> None:
        """Require an identity and an IoU threshold in ``(0, 1]``."""
        require_text("policy_id", self.policy_id)
        if not (math.isfinite(self.min_iou) and 0.0 < self.min_iou <= 1.0):
            raise ValueError("min_iou must be an IoU in (0, 1]")


@dataclass(frozen=True, kw_only=True)
class RegionMatch:
    """One reference region joined to one run region, with the IoU that joined them."""

    reference_region_id: str
    run_region_id: RegionId
    iou: float


@dataclass(frozen=True, kw_only=True)
class FrameRegionJoin:
    """The join of one annotated observation under one policy.

    Attributes:
        policy_id: The policy that produced it.
        sample_id: The reference sample the observation belongs to.
        observation_id: The joined observation.
        matches: The joined pairs, in the order the greedy assignment took them.
        unmatched_reference_region_ids: Reference regions no run region was joined to.
        unmatched_run_region_ids: Run regions without a reference region; they carry no
            reference expectation.
    """

    policy_id: str
    sample_id: ReferenceSampleId
    observation_id: SourceObservationId
    matches: tuple[RegionMatch, ...]
    unmatched_reference_region_ids: tuple[str, ...]
    unmatched_run_region_ids: tuple[RegionId, ...]

    def reference_region_of(self, run_region_id: RegionId) -> str | None:
        """Return the reference region joined to a run region, or ``None`` when unmatched."""
        return next(
            (
                match.reference_region_id
                for match in self.matches
                if match.run_region_id == run_region_id
            ),
            None,
        )


@dataclass(frozen=True, kw_only=True)
class RegionJoinSummary:
    """How many regions a policy joined over a set of observations, and how many it left."""

    policy_id: str
    frame_count: int
    matched_count: int
    unmatched_reference_count: int
    unmatched_run_count: int


def join_frame_regions(
    policy: RegionJoinPolicy,
    reference: FrameRegionAnnotation,
    run_regions: Sequence[Region2D],
) -> FrameRegionJoin:
    """Join the run regions of one observation to its reference regions.

    Args:
        policy: The join rule.
        reference: The region annotation of the observation.
        run_regions: The observation's run regions. A region persisted with a
            ``mask_reference`` must carry its mask inline, loaded from the run's mask store.

    Raises:
        RegionJoinError: For a run region of another observation or image size, a repeated
            run region id, a persisted mask that was not loaded, or a reference mask of
            another image size.
    """
    size = (reference.image_width, reference.image_height)
    references = {region.region_id: _reference_pixels(region, size) for region in reference.regions}
    for region in run_regions:
        _check_run_region(region, reference)
    try:
        require_unique("run region id", (region.region_id for region in run_regions))
    except ValueError as error:
        raise RegionJoinError(str(error)) from error
    runs = {region.region_id: region_pixels(region) for region in run_regions}
    pairs = sorted(
        (
            (pixel_set_iou(reference_pixels, run_pixels), reference_id, run_id)
            for reference_id, reference_pixels in references.items()
            for run_id, run_pixels in runs.items()
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matches: list[RegionMatch] = []
    joined_references: set[str] = set()
    joined_runs: set[RegionId] = set()
    for iou, reference_id, run_id in pairs:
        if iou < policy.min_iou:
            break
        if reference_id in joined_references or run_id in joined_runs:
            continue
        joined_references.add(reference_id)
        joined_runs.add(run_id)
        matches.append(RegionMatch(reference_region_id=reference_id, run_region_id=run_id, iou=iou))
    return FrameRegionJoin(
        policy_id=policy.policy_id,
        sample_id=reference.sample_id,
        observation_id=reference.observation_id,
        matches=tuple(matches),
        unmatched_reference_region_ids=tuple(
            region_id for region_id in references if region_id not in joined_references
        ),
        unmatched_run_region_ids=tuple(
            region_id for region_id in runs if region_id not in joined_runs
        ),
    )


def summarize_region_joins(joins: Sequence[FrameRegionJoin]) -> RegionJoinSummary:
    """Count the joined and unmatched regions of joins made under one policy.

    Raises:
        RegionJoinError: If there is no join, or the joins come from different policies.
    """
    if not joins:
        raise RegionJoinError("a join summary needs at least one joined observation")
    policies = sorted({join.policy_id for join in joins})
    if len(policies) != 1:
        raise RegionJoinError(f"the joins come from more than one join policy: {policies}")
    return RegionJoinSummary(
        policy_id=policies[0],
        frame_count=len(joins),
        matched_count=sum(len(join.matches) for join in joins),
        unmatched_reference_count=sum(len(join.unmatched_reference_region_ids) for join in joins),
        unmatched_run_count=sum(len(join.unmatched_run_region_ids) for join in joins),
    )


def _reference_pixels(region: RegionAnnotation, size: tuple[int, int]) -> set[tuple[int, int]]:
    """Return a reference region's pixels: its mask when it has one, otherwise its box."""
    width, height = size
    if region.mask is not None:
        if (region.mask.width, region.mask.height) != size:
            raise RegionJoinError(
                f"reference region {region.region_id!r} has a {region.mask.width}x"
                f"{region.mask.height} mask on a {width}x{height} image"
            )
        return mask_pixels(region.mask)
    box = region.box
    assert box is not None  # RegionAnnotation exige máscara ou caixa.
    return box_pixels(
        x_min=box.x_min,
        y_min=box.y_min,
        x_max=box.x_max,
        y_max=box.y_max,
        width=width,
        height=height,
    )


def _check_run_region(region: Region2D, reference: FrameRegionAnnotation) -> None:
    """Refuse a run region that cannot be compared with the reference image."""
    if region.source_observation_id not in (None, reference.observation_id):
        raise RegionJoinError(
            f"run region {region.region_id!r} belongs to observation "
            f"{region.source_observation_id!r}, not {reference.observation_id!r}"
        )
    size = (reference.image_width, reference.image_height)
    if region.image_width is not None and (region.image_width, region.image_height) != size:
        raise RegionJoinError(
            f"run region {region.region_id!r} has image dimensions "
            f"{region.image_width}x{region.image_height}, the reference {size[0]}x{size[1]}"
        )
    if region.mask is not None and (region.mask.width, region.mask.height) != size:
        raise RegionJoinError(
            f"run region {region.region_id!r} has a mask of other image dimensions than the "
            "reference"
        )
    if region.mask is None and region.mask_reference is not None:
        raise RegionJoinError(
            f"run region {region.region_id!r} was persisted with a mask that is not loaded; "
            "load it from the run's mask store, or its box would be read in its place"
        )
