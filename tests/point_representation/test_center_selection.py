"""The versioned rule that picks which map geometry a runtime run represents (issue #663)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from observation_builders import make_spatial_observation
from pointrep_geometry import MAP_ID, LinearScanSource, make_geometry_point

from contextmap.geometric_mapping import GeometryReference, MapId, geometry_id_for
from contextmap.point_representation import (
    OBSERVATION_SUPPORT_ANCHOR_POLICY_ID,
    ObservationSupportAnchorPolicy,
)
from contextmap.shared import Vector3


def _source(coordinates: Sequence[Vector3]) -> LinearScanSource:
    return LinearScanSource(
        [make_geometry_point(index, point) for index, point in enumerate(coordinates)]
    )


def _ref(index: int, *, map_id: MapId = MAP_ID) -> GeometryReference:
    return GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index))


def test_the_policy_has_its_versioned_identity() -> None:
    assert ObservationSupportAnchorPolicy().policy_id == OBSERVATION_SUPPORT_ANCHOR_POLICY_ID
    assert OBSERVATION_SUPPORT_ANCHOR_POLICY_ID == "observation-support-anchor-v1"


def test_an_observation_is_anchored_at_the_member_nearest_the_centroid_of_its_support() -> None:
    # Centróide em x = 3,2: o membro mais próximo é o de x = 3, não o de índice central.
    source = _source([(x, 0.0, 0.0) for x in (0.0, 1.0, 2.0, 3.0, 10.0)])
    observation = make_spatial_observation(support=(0, 1, 2, 3, 4))

    centers = ObservationSupportAnchorPolicy().select_centers([observation], source)

    assert centers == (_ref(3),)


def test_equidistant_members_are_decided_by_geometry_id() -> None:
    source = _source([(2.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    observation = make_spatial_observation(support=(0, 1))

    assert ObservationSupportAnchorPolicy().select_centers([observation], source) == (_ref(0),)


def test_one_anchor_reached_by_several_observations_is_one_center_in_geometry_order() -> None:
    source = _source([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (9.0, 0.0, 0.0)])
    late = make_spatial_observation(frame="frame-0130", support=(2, 3))
    first = make_spatial_observation(frame="frame-0120", support=(0, 1, 2))
    again = make_spatial_observation(frame="frame-0125", support=(0, 1, 2))

    centers = ObservationSupportAnchorPolicy().select_centers([late, first, again], source)

    assert centers == (_ref(1), _ref(2))


def test_an_observation_without_geometry_support_anchors_nothing() -> None:
    source = _source([(0.0, 0.0, 0.0)])
    empty = make_spatial_observation(support=())

    assert ObservationSupportAnchorPolicy().select_centers([empty], source) == ()


def test_an_observation_of_another_map_is_refused() -> None:
    source = _source([(0.0, 0.0, 0.0)])
    other = make_spatial_observation(support=(0,), map_id=MapId("map-other"))

    with pytest.raises(ValueError, match="map-other"):
        ObservationSupportAnchorPolicy().select_centers([other], source)
