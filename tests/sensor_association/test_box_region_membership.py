"""Box membership (``box-depth-component-v1``): a box-only region gets the 3D component its
box core sees, never every point of the box."""

import numpy as np
import pytest
from perception_builders import make_region, make_result, rect_mask
from projection_builders import scene_frame

from contextmap.ingestion import SequenceArtifactId
from contextmap.region_perception import RegionId
from contextmap.sensor_association import BoxDepthComponentPolicy, FootprintKind
from contextmap.sensor_association.membership import (
    COMPOSITE_MEMBERSHIP_POLICY_ID,
    MEMBERSHIP_POLICY_ID,
    FrameMembership,
    SkippedRegion,
    SkipReason,
    associate_regions,
    build_spatial_observations,
)
from contextmap.sensor_association.projection_chain.frame_projection import FrameProjection
from contextmap.sensor_association.projection_chain.visibility import (
    OcclusionPolicy,
    resolve_visibility,
)
from contextmap.sensor_association.quality_derivation import derive_observation_quality

OCCLUSION = OcclusionPolicy(
    cell_size_px=4, neighborhood_radius_cells=2, depth_margin_m=0.1, depth_margin_ratio=0.02
)
BOX_POLICY = BoxDepthComponentPolicy(
    cell_size_m=0.15, core_fraction=0.5, minimum_support_points=3, ambiguity_ratio=0.5
)
A, B = RegionId("region-A"), RegionId("region-B")
BOX = (260.0, 180.0, 120.0, 120.0)
"""``x 260..380``, ``y 180..300``; its core (half of each side) is ``x 290..350``,
``y 210..270``."""


def _object_grid(
    x0: int, x1: int, y0: int, y1: int, depth: float
) -> list[tuple[float, float, float]]:
    return [(u, v, depth) for u in range(x0, x1, 10) for v in range(y0, y1, 10)]


def _associate(
    frame: FrameProjection,
    *regions: object,
    policy: BoxDepthComponentPolicy | None = BOX_POLICY,
) -> FrameMembership:
    result = make_result(list(regions))  # type: ignore[arg-type]
    return associate_regions(
        resolve_visibility(frame, OCCLUSION),
        (result).regions,
        box_policy=policy,
        region_artifact_id="run-0001",
        sequence_artifact_id=SequenceArtifactId("sequence-0001"),
    )


def test_a_box_supports_the_component_its_core_sees_and_not_the_background_wall() -> None:
    target = _object_grid(300, 345, 220, 265, depth=2.0)
    # Parede ao fundo, nas margens da caixa e longe dos pixels do objeto, para continuar
    # visível: sem a regra por componente ela entraria no suporte.
    wall = [(265.0, 185.0, 5.0), (375.0, 185.0, 5.0), (265.0, 295.0, 5.0), (375.0, 295.0, 5.0)]
    frame = scene_frame(*target, *wall)

    membership = _associate(frame, make_region("region-A", None, box=BOX))

    (region,) = membership.regions
    assert region.footprint_kind is FootprintKind.BOX
    assert region.associated_indices.tolist() == list(range(len(target)))
    assert region.excluded_visible_count == len(wall)
    assert region.footprint_point_count == len(target) + len(wall)
    assert region.footprint_area_px == 120 * 120
    assert membership.membership_policy_id == COMPOSITE_MEMBERSHIP_POLICY_ID
    assert membership.visible_unassigned[len(target) :].all()


def test_two_comparable_components_in_the_core_make_the_box_ambiguous() -> None:
    # Dois aglomerados a profundidades diferentes, separados em pixel o bastante para que um
    # não oculte o outro, ambos no núcleo da caixa: nenhum é o alvo com segurança.
    near = _object_grid(292, 312, 215, 265, depth=2.0)
    far = _object_grid(330, 350, 215, 265, depth=4.0)
    frame = scene_frame(*near, *far)

    membership = _associate(frame, make_region("region-A", None, box=BOX))

    assert membership.regions == ()
    assert membership.skipped == (
        SkippedRegion(region_id=A, reason=SkipReason.AMBIGUOUS_BOX_SUPPORT),
    )


def test_a_box_whose_best_component_is_too_small_is_skipped_as_insufficient() -> None:
    frame = scene_frame((310.0, 230.0, 2.0), (320.0, 230.0, 2.0))

    membership = _associate(frame, make_region("region-A", None, box=BOX))

    assert membership.regions == ()
    assert membership.skipped == (
        SkippedRegion(region_id=A, reason=SkipReason.INSUFFICIENT_BOX_SUPPORT),
    )


def test_a_masked_region_follows_the_mask_rule_under_the_box_policy() -> None:
    frame = scene_frame((100, 100, 3.0), (200, 100, 3.0), (300, 100, 3.0))
    region = make_region("region-B", rect_mask(640, 480, 90, 90, 210, 110))

    with_box_policy = _associate(frame, region)
    mask_only = _associate(frame, region, policy=None)

    assert with_box_policy.points_of(B).tolist() == mask_only.points_of(B).tolist() == [0, 1]
    (masked,) = with_box_policy.regions
    assert masked.footprint_kind is FootprintKind.MASK
    assert masked.excluded_visible_count == 0
    assert mask_only.membership_policy_id == MEMBERSHIP_POLICY_ID


def test_without_the_box_policy_a_box_only_region_stays_unevaluated() -> None:
    frame = scene_frame(*_object_grid(300, 345, 220, 265, depth=2.0))

    membership = _associate(frame, make_region("region-A", None, box=BOX), policy=None)

    assert membership.skipped == (SkippedRegion(region_id=A, reason=SkipReason.NO_INLINE_MASK),)


def test_box_observations_name_the_box_policy_and_measure_quality_per_box_pixel() -> None:
    target = _object_grid(300, 345, 220, 265, depth=2.0)
    frame = scene_frame(*target, (265.0, 185.0, 5.0))
    membership = _associate(frame, make_region("region-A", None, box=BOX))

    (observation,) = build_spatial_observations(
        membership,
        configuration_fingerprint=None,
        code_version=None,
        association_run_id="association-run-0001",
    )
    (quality,) = derive_observation_quality(membership, (observation,))

    assert observation.provenance.membership_policy_id == "box-depth-component-v1"
    assert len(observation.geometry_support) == len(target)
    assert quality.footprint_kind is FootprintKind.BOX
    assert quality.footprint_area_px == 120 * 120
    assert quality.footprint_count == len(target) + 1
    assert quality.support_density_per_footprint_pixel == pytest.approx(len(target) / 14400)


def test_each_observation_names_the_rule_applied_to_its_region_not_the_run_policy() -> None:
    target = _object_grid(300, 345, 220, 265, depth=2.0)
    frame = scene_frame((100, 100, 3.0), (200, 100, 3.0), *target)
    masked = make_region("region-B", rect_mask(640, 480, 90, 90, 210, 110))
    box_only = make_region("region-A", None, box=BOX)
    membership = _associate(frame, masked, box_only)

    first, second = build_spatial_observations(
        membership,
        configuration_fingerprint=None,
        code_version=None,
        association_run_id="association-run-0001",
    )

    assert [r.membership_rule_id for r in membership.regions] == [
        MEMBERSHIP_POLICY_ID,
        BoxDepthComponentPolicy.policy_id,
    ]
    assert first.provenance.membership_policy_id == MEMBERSHIP_POLICY_ID == "mask-membership-v1"
    assert second.provenance.membership_policy_id == "box-depth-component-v1"
    assert membership.membership_policy_id == COMPOSITE_MEMBERSHIP_POLICY_ID


def test_box_membership_is_deterministic() -> None:
    target = _object_grid(300, 345, 220, 265, depth=2.0)
    frame = scene_frame(*target, (265.0, 185.0, 5.0), (375.0, 295.0, 5.0))
    region = make_region("region-A", None, box=BOX)

    first, second = _associate(frame, region), _associate(frame, region)

    assert np.array_equal(first.point_indices, second.point_indices)
    assert np.array_equal(first.visible_unassigned, second.visible_unassigned)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_size_m", 0.0),
        ("cell_size_m", float("nan")),
        ("core_fraction", 0.0),
        ("core_fraction", 1.5),
        ("minimum_support_points", 0),
        ("ambiguity_ratio", 0.0),
        ("ambiguity_ratio", 1.5),
    ],
)
def test_the_box_policy_refuses_meaningless_parameters(field: str, value: float) -> None:
    parameters = {
        "cell_size_m": 0.15,
        "core_fraction": 0.5,
        "minimum_support_points": 3,
        "ambiguity_ratio": 0.5,
    }
    parameters[field] = value

    with pytest.raises(ValueError, match=field):
        BoxDepthComponentPolicy(**parameters)  # type: ignore[arg-type]


def test_the_box_policy_record_carries_its_versioned_identity() -> None:
    record = BOX_POLICY.to_record()

    assert record == {
        "policy_id": "box-depth-component-v1",
        "cell_size_m": 0.15,
        "core_fraction": 0.5,
        "minimum_support_points": 3,
        "ambiguity_ratio": 0.5,
    }
