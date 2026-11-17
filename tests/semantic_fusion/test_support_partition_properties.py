"""Properties of fusion support construction: a partition of the observations, nothing more.

``FusionSupportBuild`` documents that every input observation is in exactly one support or in
``excluded``, but its own validation cannot check that the inputs were all placed, since it never
sees them. These properties check it against the inputs, for any geometry the observations
reference (empty, identical, nested, disjoint) and any policy, including a ``min_overlap`` equal
to an overlap the scene actually has, where the inclusive threshold decides. The supports are then
compared with the connected components of a plain set-based enumeration of the overlap links.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from fusion_builders import MAP_ID
from geometry_fake import InMemoryGeometrySource
from hypothesis import given
from hypothesis import strategies as st
from observation_builders import make_spatial_observation, timestamps

from contextmap.semantic_fusion import (
    FusionSupportBuild,
    GeometryOverlapSupportPolicy,
    build_fusion_supports,
)
from contextmap.sensor_association import SpatialObservation, SpatialObservationId

FRAMES = ("frame-0120", "frame-0121", "frame-0122", "frame-0123")
UNIVERSE = 12
SOURCE = InMemoryGeometrySource(
    MAP_ID, {index: (index * 0.1, 0.0, 0.0) for index in range(UNIVERSE)}
)

View = tuple[str, int, frozenset[int]]
"""An observation as the strategy draws it: frame, region number and referenced geometry."""

views = st.lists(
    st.tuples(
        st.sampled_from(FRAMES),
        st.integers(min_value=1, max_value=3),
        st.frozensets(st.integers(min_value=0, max_value=UNIVERSE - 1), max_size=UNIVERSE),
    ),
    unique_by=lambda view: (view[0], view[1]),
    max_size=10,
)


@st.composite
def policies(draw: st.DrawFn, scene: Sequence[View]) -> GeometryOverlapSupportPolicy:
    """Any valid policy, with a ``min_overlap`` that is often an overlap of the scene itself."""
    overlaps = sorted({_jaccard(a[2], b[2]) for a, b in combinations(scene, 2)} - {0.0})
    candidates = st.floats(min_value=0.0, max_value=1.0, exclude_min=True)
    if overlaps:
        candidates = st.sampled_from(overlaps) | candidates
    return GeometryOverlapSupportPolicy(
        min_geometry_count=draw(st.integers(min_value=1, max_value=6)),
        min_overlap=draw(candidates),
    )


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = len(left | right)
    return 0.0 if union == 0 else len(left & right) / union


def _observation(view: View) -> SpatialObservation:
    frame, region, support = view
    return make_spatial_observation("run-a", frame, f"region-{region:04d}", support=sorted(support))


def _build(scene: Sequence[View], policy: GeometryOverlapSupportPolicy) -> FusionSupportBuild:
    return build_fusion_supports(
        [_observation(view) for view in scene],
        geometry=SOURCE,
        acquisition_timestamps=timestamps(FRAMES),
        policy=policy,
    )


def _components(scene: Sequence[View], policy: GeometryOverlapSupportPolicy) -> set[frozenset[str]]:
    """The connected components of the overlap links, by plain enumeration of every pair."""
    eligible = [view for view in scene if len(view[2]) >= policy.min_geometry_count]
    component = {index: {index} for index in range(len(eligible))}
    for left, right in combinations(range(len(eligible)), 2):
        if _jaccard(eligible[left][2], eligible[right][2]) >= policy.min_overlap:
            merged = component[left] | component[right]
            for member in merged:
                component[member] = merged
    return {
        frozenset(str(_observation(eligible[index]).spatial_observation_id) for index in members)
        for members in map(frozenset, component.values())
    }


@given(st.data())
def test_every_observation_is_in_exactly_one_support_or_excluded_for_being_too_small(
    data: st.DataObject,
) -> None:
    scene = data.draw(views)
    policy = data.draw(policies(scene))

    build = _build(scene, policy)

    identities = {_observation(view).spatial_observation_id: len(view[2]) for view in scene}
    placed: list[SpatialObservationId] = [
        identity for support in build.supports for identity in support.spatial_observation_ids
    ]
    excluded = {item.spatial_observation_id: item for item in build.excluded}
    # Partição: cada entrada exatamente uma vez, num suporte ou em ``excluded``.
    assert sorted([*placed, *excluded]) == sorted(identities)
    # E ``excluded`` são exatamente as pequenas demais, cada uma com a sua contagem.
    assert set(excluded) == {
        identity for identity, count in identities.items() if count < policy.min_geometry_count
    }
    for identity, item in excluded.items():
        assert (item.geometry_count, item.minimum_geometry_count) == (
            identities[identity],
            policy.min_geometry_count,
        )


@given(st.data())
def test_the_supports_are_the_components_of_the_overlap_links(data: st.DataObject) -> None:
    scene = data.draw(views)
    policy = data.draw(policies(scene))

    build = _build(scene, policy)

    assert {
        frozenset(str(identity) for identity in support.spatial_observation_ids)
        for support in build.supports
    } == _components(scene, policy)
