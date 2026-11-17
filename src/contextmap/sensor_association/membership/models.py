"""Contratos internos de pertencimento entre regiões e geometria.

Define suporte, estatísticas, skips e resolução substituível de máscaras persistidas.
Não executa a regra de máscara nem a seleção de componentes 3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SequenceArtifactId, SourceObservationId
from contextmap.region_perception import InlineMask, RegionId
from contextmap.sensor_association.models import FootprintKind, VisibilityState
from contextmap.sensor_association.projection_chain.frame_projection import FrameProjection
from contextmap.sensor_association.projection_chain.visibility import VisibilityResolution

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from contextmap.sensor_association.membership.box_depth_component import (
        BoxDepthComponentPolicy,
    )


MEMBERSHIP_POLICY_ID = "mask-membership-v1"
"""Versioned identity of the membership rule: a visible point belongs to every accepted
region whose inline mask is foreground at the pixel whose center is nearest to the
point's prepared-image pixel."""

COMPOSITE_MEMBERSHIP_POLICY_ID = "mask-or-box-depth-component-v1"
"""Identity of the membership configuration of a run with the box rule enabled: regions with
a mask follow :data:`MEMBERSHIP_POLICY_ID` and box-only regions follow
:class:`BoxDepthComponentPolicy`. It is the *configured* policy of the frame; the rule
*applied* to each region is :attr:`RegionMembership.membership_rule_id`."""


def configured_membership_policy_id(box_policy: BoxDepthComponentPolicy | None) -> str:
    """Return the membership policy a run is configured with.

    Args:
        box_policy: The box rule, or ``None`` when only the mask rule is enabled.

    Returns:
        :data:`MEMBERSHIP_POLICY_ID` without a box rule, else
        :data:`COMPOSITE_MEMBERSHIP_POLICY_ID`.
    """
    return MEMBERSHIP_POLICY_ID if box_policy is None else COMPOSITE_MEMBERSHIP_POLICY_ID


COVERAGE_DEFINITIONS_VERSION = "region-support-metrics-v2"
"""Versioned identity of the coverage and support definitions documented on
:class:`RegionMembership` and :class:`MembershipStatistics`.

``v2``: the footprint is the mask **or**, under :class:`BoxDepthComponentPolicy`, the box of a
region without mask (``footprint_kind``), and the visible footprint points the box rule leaves
out are counted (``excluded_visible_count``) instead of disappearing from the footprint."""


class SkipReason(Enum):
    """Why a region of a regional selection was not evaluated.

    Attributes:
        REJECTED: The candidate was rejected by normalization, so it is not part of the
            frozen canonical region set.
        NO_INLINE_MASK: The region has no mask at all: a box-only region, or a region
            whose mask was persisted by reference without a ``mask_loader``
            supplied to :func:`associate_regions` to resolve it.
        EMPTY_MASK: The accepted region's mask has no foreground pixel, so it has no
            footprint to support geometry and its per-pixel statistics are undefined.
        EMPTY_BOX: Under the box rule, the box of a region without mask holds no pixel
            center of the prepared image.
        INSUFFICIENT_BOX_SUPPORT: Under the box rule, no 3D component seen by the box core
            reaches the policy's minimum support.
        AMBIGUOUS_BOX_SUPPORT: Under the box rule, a second component sees almost as much
            of the box core as the best one, so neither is the target with confidence.
    """

    REJECTED = "rejected"
    NO_INLINE_MASK = "no_inline_mask"
    EMPTY_MASK = "empty_mask"
    EMPTY_BOX = "empty_box"
    INSUFFICIENT_BOX_SUPPORT = "insufficient_box_support"
    AMBIGUOUS_BOX_SUPPORT = "ambiguous_box_support"


class RegionMaskLoader(Protocol):
    """Adapter boundary for resolving a frozen region's mask kept by reference.

    A region's mask is no longer always inlined in its
    :class:`~contextmap.region_perception.Region2D.mask` field
    once a run is persisted and reopened: it may instead be a
    ``mask_reference`` pointing into a lazily-loaded store. Mask
    membership only needs *some* way to resolve that reference back into
    pixels for the owning observation — never a particular storage
    backend — so this is a minimal, sensor-association-local Protocol.
    :meth:`~contextmap.region_perception.artifacts.mask_store.MaskStoreReader.load`
    already satisfies this shape and is the composition root's usual
    choice.
    """

    def load(self, source_observation_id: SourceObservationId, region_id: RegionId) -> InlineMask:
        """Load and hash-verify one region's full-image mask."""
        ...


@dataclass(frozen=True, kw_only=True)
class SkippedRegion:
    """A region that was not evaluated, kept so its absence is explainable.

    Attributes:
        region_id: The region.
        reason: Why it was skipped.
    """

    region_id: RegionId
    reason: SkipReason


@dataclass(frozen=True, kw_only=True, eq=False)
class RegionMembership:
    """The geometry one frozen region supports, with its support statistics.

    Definitions (version :data:`COVERAGE_DEFINITIONS_VERSION`):

    * *footprint*: the pixels of the region's mask or, for a region without mask under the
      box rule, of its box (``footprint_kind``); the *footprint points* are the points whose
      prepared pixel falls inside it, whatever their visibility state (behind-camera and
      outside-image points have no pixel and are never in it);
    * ``associated_count``: footprint points that support the region (every visible one
      under the mask rule; those of the selected component under the box rule);
    * ``excluded_visible_count``: visible footprint points the box rule left out; always
      ``0`` for a mask;
    * ``occluded_count``, ``outside_valid_support_count``: footprint points that are hidden
      or unsupported;
    * ``covered_pixel_count``: distinct footprint pixels holding at least one associated
      point;
    * ``support_pixel_coverage``: ``covered_pixel_count / footprint_area_px``, the share of
      the footprint's pixels that have supported 3D geometry (low on a sparse map);
    * ``visible_share``: visible footprint points over all footprint points; ``None`` for an
      empty footprint.

    Attributes:
        region_id: The region, as frozen by Visual Perception.
        footprint_kind: Whether the footprint is the mask or the box.
        footprint_area_px: Number of pixels of the footprint.
        associated_indices: Sorted positions, in the frame, of the associated geometry.
        excluded_visible_count: Visible footprint points that are not associated.
        occluded_count: Footprint points hidden by a nearer surface.
        outside_valid_support_count: Footprint points outside the valid support.
        covered_pixel_count: Distinct footprint pixels holding an associated point.
        membership_rule_id: The rule actually applied to this region:
            :data:`MEMBERSHIP_POLICY_ID` for a mask, ``box-depth-component-v1`` for a
            box-only region. Unlike the frame's configured policy, it is per region.
    """

    region_id: RegionId
    footprint_kind: FootprintKind
    footprint_area_px: int
    associated_indices: NDArray[Any]
    excluded_visible_count: int
    occluded_count: int
    outside_valid_support_count: int
    covered_pixel_count: int
    membership_rule_id: str

    @property
    def associated_count(self) -> int:
        """Number of geometry elements that support the region."""
        return int(self.associated_indices.shape[0])

    @property
    def footprint_point_count(self) -> int:
        """Number of points, of any state, whose prepared pixel is inside the footprint."""
        return (
            self.associated_count
            + self.excluded_visible_count
            + self.occluded_count
            + self.outside_valid_support_count
        )

    @property
    def support_pixel_coverage(self) -> float:
        """Share of footprint pixels that hold at least one associated point."""
        return self.covered_pixel_count / self.footprint_area_px if self.footprint_area_px else 0.0

    @property
    def visible_share(self) -> float | None:
        """Share of the footprint that was visible; ``None`` when the footprint is empty."""
        footprint = self.footprint_point_count
        visible = self.associated_count + self.excluded_visible_count
        return None if footprint == 0 else visible / footprint


@dataclass(frozen=True, kw_only=True)
class MembershipStatistics:
    """Auditable counts for one frame (version :data:`COVERAGE_DEFINITIONS_VERSION`).

    The five state counts partition ``point_count``; ``visible_count`` splits into
    ``associated_count`` and ``visible_unassigned_count``.

    Attributes:
        definitions_version: The version of these definitions.
        point_count: Map points evaluated for the frame.
        behind_camera_count: Points the camera model cannot project.
        outside_image_count: Points that project outside the prepared image.
        outside_valid_support_count: Points outside the valid region or inside an exclusion.
        occluded_count: Points hidden by a nearer surface.
        visible_count: Points that are supported and not occluded.
        associated_count: Visible points inside at least one evaluated mask.
        visible_unassigned_count: Visible points inside no evaluated mask.
        overlap_point_count: Visible points inside two or more masks.
        membership_pair_count: ``(point, region)`` memberships.
        points_by_region_count: For each number of regions ``k``, how many visible points
            belong to exactly ``k`` regions; ``0`` counts the unassigned ones.
    """

    definitions_version: str
    point_count: int
    behind_camera_count: int
    outside_image_count: int
    outside_valid_support_count: int
    occluded_count: int
    visible_count: int
    associated_count: int
    visible_unassigned_count: int
    overlap_point_count: int
    membership_pair_count: int
    points_by_region_count: dict[int, int]


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameMembership:
    """Which evaluated regions support each visible point of one frame.

    Attributes:
        resolution: The visibility the membership was computed on.
        region_artifact_id: The selected regional producer.
        sequence_artifact_id: The canonical source sequence.
        membership_policy_id: The membership policy the run was configured with (see
            :func:`configured_membership_policy_id`); the rule applied to each region is in
            :attr:`RegionMembership.membership_rule_id`.
        regions: Evaluated regions, in the order of the regional selection.
        skipped: Regions that were not evaluated, with the reason.
        point_indices: ``(M,)`` frame position of each membership, sorted by point then
            region.
        point_region_slots: ``(M,)`` index into ``regions`` of each membership.
        visible_unassigned: ``(N,)`` visible points that are inside no evaluated mask.
    """

    resolution: VisibilityResolution
    region_artifact_id: str
    sequence_artifact_id: SequenceArtifactId
    membership_policy_id: str
    regions: tuple[RegionMembership, ...]
    skipped: tuple[SkippedRegion, ...]
    point_indices: NDArray[Any]
    point_region_slots: NDArray[Any]
    visible_unassigned: NDArray[Any]

    @property
    def frame(self) -> FrameProjection:
        """The projection the memberships refer to."""
        return self.resolution.frame

    def points_of(self, region_id: RegionId) -> NDArray[Any]:
        """Return the sorted frame positions of the geometry a region supports.

        Raises:
            KeyError: If the region was not evaluated.
        """
        for region in self.regions:
            if region.region_id == region_id:
                return region.associated_indices
        raise KeyError(region_id)

    def regions_of(self, point_index: int) -> tuple[RegionId, ...]:
        """Return every evaluated region that contains a point, in result order.

        An empty tuple means the point is inside no mask; overlaps keep all regions.
        """
        import numpy as np

        low = int(np.searchsorted(self.point_indices, point_index, side="left"))
        high = int(np.searchsorted(self.point_indices, point_index, side="right"))
        return tuple(
            self.regions[int(slot)].region_id for slot in self.point_region_slots[low:high]
        )

    def statistics(self) -> MembershipStatistics:
        """Summarize the frame: every state, the association split and the overlaps."""
        import numpy as np

        counts = self.resolution.state_counts()
        visible = self.resolution.visible_count
        unassigned = int(self.visible_unassigned.sum())
        per_point = np.bincount(self.point_indices, minlength=len(self.visible_unassigned))
        overlap = int((per_point >= 2).sum())
        histogram: dict[int, int] = {0: unassigned}
        for regions_per_point in per_point[per_point > 0].tolist():
            histogram[regions_per_point] = histogram.get(regions_per_point, 0) + 1
        return MembershipStatistics(
            definitions_version=COVERAGE_DEFINITIONS_VERSION,
            point_count=len(self.visible_unassigned),
            behind_camera_count=counts[VisibilityState.BEHIND_CAMERA],
            outside_image_count=counts[VisibilityState.OUTSIDE_IMAGE],
            outside_valid_support_count=counts[VisibilityState.OUTSIDE_VALID_SUPPORT],
            occluded_count=counts[VisibilityState.OCCLUDED],
            visible_count=visible,
            associated_count=visible - unassigned,
            visible_unassigned_count=unassigned,
            overlap_point_count=overlap,
            membership_pair_count=int(self.point_indices.shape[0]),
            points_by_region_count={k: histogram[k] for k in sorted(histogram)},
        )
