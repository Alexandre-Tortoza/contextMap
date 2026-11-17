"""Pertencimento da geometria visível às regiões congeladas de um frame.

Lê máscaras no espaço preparado e aplica a regra de caixa quando habilitada.
Regiões sobrepostas permanecem válidas e nenhum vencedor semântico é escolhido.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import SequenceArtifactId
from contextmap.region_perception import Region2D
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.membership.box_depth_component import (
    BoxDepthComponentPolicy,
    box_membership,
)
from contextmap.sensor_association.membership.models import (
    MEMBERSHIP_POLICY_ID,
    FrameMembership,
    RegionMaskLoader,
    RegionMembership,
    SkippedRegion,
    SkipReason,
    configured_membership_policy_id,
)
from contextmap.sensor_association.models import FootprintKind
from contextmap.sensor_association.projection_chain.visibility import VisibilityResolution

if TYPE_CHECKING:
    from numpy.typing import NDArray


def associate_regions(
    resolution: VisibilityResolution,
    regions: Sequence[Region2D],
    *,
    region_artifact_id: str,
    sequence_artifact_id: SequenceArtifactId,
    mask_loader: RegionMaskLoader | None = None,
    box_policy: BoxDepthComponentPolicy | None = None,
) -> FrameMembership:
    """Find which frozen regions support each visible projected point.

    Args:
        resolution: The visibility of the frame's points.
        regions: Frozen regions of the same physical observation.
        region_artifact_id: Explicitly selected regional producer.
        sequence_artifact_id: Canonical sequence shared by geometry and regions.
        mask_loader: Resolves a region's mask when it was persisted by
            reference instead of being inlined on ``Region2D.mask``, for example a reopened run's
            ``RegionPerceptionRunReader.mask_store()``. ``None`` (the default)
            keeps the previous behavior: a region without an inline mask
            is skipped with :attr:`SkipReason.NO_INLINE_MASK`.
        box_policy: Enables the box rule for regions that, once ``mask_loader`` was tried,
            still have no mask. ``None`` (the default) keeps them skipped with
            :attr:`SkipReason.NO_INLINE_MASK`; regions with a mask always follow the mask rule.

    An accepted region whose mask has no foreground pixel is skipped with
    :attr:`SkipReason.EMPTY_MASK`: every :class:`RegionMembership` has ``footprint_area_px >= 1``,
    which the per-pixel statistics and the observation quality require.

    Returns:
        The region-to-geometry and geometry-to-region indexes, the support statistics and
        the regions that were skipped.

    Raises:
        AssociationInputError: If the regional selection belongs to another observation, or
            an evaluated region's mask is not in the prepared image's coordinate space.
    """
    import numpy as np

    frame = resolution.frame
    if not region_artifact_id:
        raise AssociationInputError(
            "region_artifact_id must identify the selected regional producer"
        )
    if any(
        region.source_observation_id not in (None, frame.source_observation_id)
        for region in regions
    ):
        raise AssociationInputError("region source observation does not match projected frame")
    if len({region.region_id for region in regions}) != len(regions):
        raise AssociationInputError("duplicate regional evidence identity")
    width, height = frame.image_transform.prepared_size
    in_image = np.flatnonzero(frame.in_prepared_image)
    # O pixel de índice i cobre [i - 0.5, i + 0.5): o índice do centro c é floor(c + 0.5).
    columns = np.floor(frame.prepared_pixels[in_image, 0] + 0.5).astype(np.int64)
    rows = np.floor(frame.prepared_pixels[in_image, 1] + 0.5).astype(np.int64)
    visible = resolution.visible[in_image]
    occluded = resolution.occluded[in_image]
    unsupported = resolution.outside_valid_support[in_image]

    evaluated_regions: list[RegionMembership] = []
    skipped: list[SkippedRegion] = []
    pairs_points: list[NDArray[Any]] = []
    pairs_slots: list[NDArray[Any]] = []
    for region in regions:
        if not region.is_accepted:
            skipped.append(SkippedRegion(region_id=region.region_id, reason=SkipReason.REJECTED))
            continue
        region_mask = region.mask
        if region_mask is None and region.mask_reference is not None and mask_loader is not None:
            region_mask = mask_loader.load(frame.source_observation_id, region.region_id)
        if region_mask is None and box_policy is not None:
            outcome = box_membership(
                region,
                box_policy,
                frame=frame,
                in_image=in_image,
                columns=columns,
                rows=rows,
                visible=visible,
                occluded=occluded,
                unsupported=unsupported,
            )
            if isinstance(outcome, SkipReason):
                skipped.append(SkippedRegion(region_id=region.region_id, reason=outcome))
                continue
            slot = len(evaluated_regions)
            evaluated_regions.append(outcome)
            pairs_points.append(outcome.associated_indices)
            pairs_slots.append(np.full(outcome.associated_indices.shape, slot, dtype=np.int64))
            continue
        if region_mask is None:
            skipped.append(
                SkippedRegion(region_id=region.region_id, reason=SkipReason.NO_INLINE_MASK)
            )
            continue
        if (region_mask.width, region_mask.height) != (width, height):
            raise AssociationInputError(
                f"the mask of region {region.region_id!r} is {region_mask.width}x"
                f"{region_mask.height} but the prepared image is {width}x{height}: regions must be "
                f"expressed in the prepared image the geometry was projected into"
            )
        mask = region_mask.as_array()
        footprint_area_px = region_mask.area
        if footprint_area_px == 0:
            skipped.append(SkippedRegion(region_id=region.region_id, reason=SkipReason.EMPTY_MASK))
            continue
        inside = mask[rows, columns]
        associated = inside & visible
        associated_indices = in_image[associated]
        covered = np.unique(rows[associated] * width + columns[associated]).shape[0]
        slot = len(evaluated_regions)
        evaluated_regions.append(
            RegionMembership(
                region_id=region.region_id,
                footprint_kind=FootprintKind.MASK,
                footprint_area_px=footprint_area_px,
                associated_indices=associated_indices,
                excluded_visible_count=0,
                occluded_count=int((inside & occluded).sum()),
                outside_valid_support_count=int((inside & unsupported).sum()),
                covered_pixel_count=int(covered),
                membership_rule_id=MEMBERSHIP_POLICY_ID,
            )
        )
        pairs_points.append(associated_indices)
        pairs_slots.append(np.full(associated_indices.shape, slot, dtype=np.int64))

    if pairs_points:
        points = np.concatenate(pairs_points)
        slots = np.concatenate(pairs_slots)
    else:
        points = np.empty(0, dtype=np.int64)
        slots = np.empty(0, dtype=np.int64)
    order = np.lexsort((slots, points))
    points, slots = points[order], slots[order]
    assigned = np.zeros(len(resolution.visible), dtype=bool)
    assigned[points] = True
    return FrameMembership(
        resolution=resolution,
        region_artifact_id=region_artifact_id,
        sequence_artifact_id=sequence_artifact_id,
        membership_policy_id=configured_membership_policy_id(box_policy),
        regions=tuple(evaluated_regions),
        skipped=tuple(skipped),
        point_indices=points,
        point_region_slots=slots,
        visible_unassigned=resolution.visible & ~assigned,
    )
