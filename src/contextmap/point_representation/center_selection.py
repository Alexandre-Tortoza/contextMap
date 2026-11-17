"""Versioned selection of the geometry a Point Representation run represents.

:class:`~contextmap.point_representation.RepresentationService` represents exactly the centers
it is asked for; which centers that is decides both what the run describes and what it costs,
so the choice is a versioned policy of this capability, never a rule of whoever runs it.

``observation-support-anchor-v1`` is a **heuristic**. It anchors one representation in each
spatial observation of an association run: the member of the observation's geometry support
nearest the centroid of that support. It does not claim the anchor is the object's center, a
salient point or a representative sample of the support; it only makes the representations of
a run follow what the camera actually observed, at a cost bounded by the number of observations
instead of by the size of the map. See ``docs/center-selection.md`` for the rule, why it was
chosen and its limits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from contextmap.geometric_mapping import GeometryReference, GeometrySource
from contextmap.sensor_association import SpatialObservation

OBSERVATION_SUPPORT_ANCHOR_POLICY_ID = "observation-support-anchor-v1"
"""Versioned identity of the anchor rule of :class:`ObservationSupportAnchorPolicy`."""


@dataclass(frozen=True, kw_only=True)
class ObservationSupportAnchorPolicy:
    """One anchor per spatial observation: the support member nearest the support centroid.

    The policy has no parameter: every change to the rule is a new ``policy_id``.
    """

    @property
    def policy_id(self) -> str:
        """Versioned identity of the rule, recorded by every run it selects centers for."""
        return OBSERVATION_SUPPORT_ANCHOR_POLICY_ID

    def select_centers(
        self, observations: Iterable[SpatialObservation], source: GeometrySource
    ) -> tuple[GeometryReference, ...]:
        """Select the centers of one run from the observations of one association run.

        For each observation, the centroid of its ``geometry_support`` is the mean of the
        members' authoritative coordinates, in meters, in the map frame; its anchor is the member
        at the smallest Euclidean distance from that centroid, with ties decided by the smallest
        ``geometry_id``. An observation without geometry support anchors nothing. The same anchor
        reached by several observations is one center.

        Args:
            observations: The spatial observations of one association run.
            source: The map those observations reference.

        Returns:
            The distinct anchors, ordered by ``geometry_id``, so the request -- and with it the
            identity of each representation -- does not depend on the order of ``observations``.

        Raises:
            ValueError: If an observation references another map than ``source``.
            KeyError: If a support member is not in ``source`` (from the ``GeometrySource`` port).
        """
        map_id = source.geometric_map.map_id
        anchors: set[GeometryReference] = set()
        for observation in observations:
            if observation.provenance.geometric_map_id != map_id:
                raise ValueError(
                    f"spatial observation {observation.spatial_observation_id!r} references map "
                    f"{observation.provenance.geometric_map_id!r}, not the map {map_id!r} being "
                    "represented"
                )
            if observation.geometry_support:
                anchors.add(_nearest_to_centroid(observation.geometry_support, source))
        return tuple(sorted(anchors, key=lambda reference: reference.geometry_id))


def _nearest_to_centroid(
    support: tuple[GeometryReference, ...], source: GeometrySource
) -> GeometryReference:
    """Return the member of ``support`` nearest its centroid, ties by ``geometry_id``."""
    coordinates = [(reference, source.get(reference).coordinates_m) for reference in support]
    count = len(coordinates)
    # math.fsum: o centróide não depende da ordem dos membros por erro de arredondamento.
    centroid = tuple(
        math.fsum(point[axis] for _, point in coordinates) / count for axis in range(3)
    )
    return min(
        coordinates,
        key=lambda item: (
            math.fsum((item[1][axis] - centroid[axis]) ** 2 for axis in range(3)),
            item[0].geometry_id,
        ),
    )[0]
