"""Seleção da geometria que um frame avalia antes da projeção.

O alcance é medido do centro óptico em metros e preserva os índices globais do mapa.
A fronteira da esfera usa aritmética exata quando o arredondamento pode mudar a decisão.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

from contextmap.geometric_mapping import (
    DEFAULT_BLOCK_POINTS,
    Bounds3D,
    GeometricMap,
    GeometryBlockSource,
    GeometryReference,
    geometry_id_for,
)
from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

CANDIDATE_GEOMETRY_POLICY_ID = "camera-range-candidates-v1"
"""Versioned identity of the candidate rule: the geometry whose distance to the camera
optical centre is at most ``max_range_m``, or the whole map when it is ``None``."""


@dataclass(frozen=True, kw_only=True)
class CandidateGeometryPolicy:
    """Which geometry a frame evaluates, and how much of it is read at a time.

    There is no default range: how far geometry may still be evidence depends on the
    sensor, the scene and the question being asked, so it is chosen and recorded per run.

    Attributes:
        max_range_m: Largest distance, in meters, from the camera optical centre at which
            a map element is still evaluated. ``None`` evaluates the whole map, which is
            the full-map baseline.
        block_points: Rows read per block while selecting, which bounds the selection's
            own working set independently of the map's size.
    """

    policy_id: ClassVar[str] = CANDIDATE_GEOMETRY_POLICY_ID

    max_range_m: float | None
    block_points: int = DEFAULT_BLOCK_POINTS

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``max_range_m`` is not a positive finite distance (when it is
                given), or ``block_points`` is not positive.
        """
        if self.max_range_m is not None and (
            not math.isfinite(self.max_range_m) or self.max_range_m <= 0
        ):
            raise ValueError(
                f"max_range_m must be a positive finite distance or None, got {self.max_range_m!r}"
            )
        if self.block_points < 1:
            raise ValueError(f"block_points must be at least 1, got {self.block_points}")

    def to_record(self) -> dict[str, Any]:
        """Return the policy as JSON primitives, with its versioned identity."""
        return {
            "policy_id": self.policy_id,
            "max_range_m": self.max_range_m,
            "block_points": self.block_points,
        }

    def fingerprint(self) -> str:
        """Return ``"sha256:<hex>"`` over the policy identity and parameters."""
        payload = json.dumps(self.to_record(), sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def locate_global_indices(
    global_indices: NDArray[Any], wanted: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Locate map elements in a strictly increasing index array by their global index.

    This is the one implementation of the lookup: both :meth:`CandidateGeometryCloud.rows_for`
    and the projection's own read it, so the identity rule cannot drift between them.

    Args:
        global_indices: ``(N,)`` strictly increasing global indices held, integer dtype.
        wanted: ``(K,)`` global indices to look up, in any order.

    Returns:
        ``(rows, found)``: the row of each requested element and whether the array holds it.
        A row is meaningless where ``found`` is ``False``.

    Raises:
        ValueError: If ``wanted`` is not an integer array. The lookup matches by equality, so
            a float or ``NaN`` index would match nothing and be reported as geometry the
            candidate policy excluded, which is a different fact from a malformed request.
            ``NaN`` also defeats every bounds comparison upstream.
    """
    import numpy as np

    requested = np.asarray(wanted)
    if not np.issubdtype(requested.dtype, np.integer):
        raise ValueError(
            f"global indices to look up must be an integer array, got dtype {requested.dtype}: "
            "a non-integer index names no map element and must not be read as excluded geometry"
        )
    count = global_indices.shape[0]
    if count == 0:
        return np.zeros(requested.shape, dtype=np.int64), np.zeros(requested.shape, dtype=bool)
    rows = np.clip(np.searchsorted(global_indices, requested), 0, count - 1)
    return rows, global_indices[rows] == requested


@dataclass(frozen=True, kw_only=True, eq=False)
class CandidateGeometryCloud:
    """The geometry one frame evaluates, ready for vectorized projection.

    Attributes:
        geometric_map: The map's identity, frame and provenance.
        coordinates_m: ``(N, 3)`` authoritative map-frame coordinates, in meters.
        global_indices: ``(N,)`` global index in the map of each row, strictly
            increasing. Row ``i`` is the geometry ``global_indices[i]``, never ``i``.
    """

    geometric_map: GeometricMap
    coordinates_m: NDArray[Any]
    global_indices: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the arrays and the identity they carry.

        Raises:
            ValueError: If the coordinates are not ``(N, 3)``, the indices are not an integer
                array, there is not one index per row, an index falls outside the map, or the
                indices are not strictly increasing. Strict increase is what lets a persisted
                geometry support be looked up by binary search and keeps its order that of the
                map; the integer dtype keeps :meth:`reference` from silently truncating a float
                into the identity of a different element.
        """
        import numpy as np

        if not np.issubdtype(self.global_indices.dtype, np.integer):
            raise ValueError(
                f"global_indices must be an integer array, got dtype "
                f"{self.global_indices.dtype}: a geometry index is an identity, never a rounded "
                "measurement"
            )
        if self.coordinates_m.ndim != 2 or self.coordinates_m.shape[1] != 3:
            raise ValueError(
                f"coordinates_m must have shape (N, 3), got {self.coordinates_m.shape}"
            )
        if self.global_indices.shape != (self.coordinates_m.shape[0],):
            raise ValueError(
                f"a candidate cloud needs one global index per row: "
                f"{self.global_indices.shape} indices for {self.coordinates_m.shape[0]} rows"
            )
        if self.global_indices.size:
            if int(self.global_indices.min()) < 0 or int(self.global_indices.max()) >= (
                self.geometric_map.point_count
            ):
                raise ValueError(
                    f"candidate indices name geometry outside the map's "
                    f"{self.geometric_map.point_count} elements"
                )
            if not bool((np.diff(self.global_indices) > 0).all()):
                raise ValueError("global_indices must be strictly increasing")

    def __len__(self) -> int:
        """Return the number of candidate elements."""
        return int(self.coordinates_m.shape[0])

    def reference(self, row: int) -> GeometryReference:
        """Return the persistent reference of one candidate row.

        Args:
            row: Position in this cloud, which is **not** a geometry index.
        """
        map_id = self.geometric_map.map_id
        index = int(self.global_indices[row])
        return GeometryReference(
            map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=index)
        )

    def references(self, rows: Iterable[int]) -> tuple[GeometryReference, ...]:
        """Return the references of the given rows, in the given order."""
        return tuple(self.reference(int(row)) for row in rows)

    def rows_for(self, global_indices: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Locate map elements in this cloud by their global index.

        Args:
            global_indices: ``(K,)`` global indices to look up, in any order.

        Returns:
            ``(rows, found)``: the row of each requested element and whether the cloud
            holds it at all. A row is meaningless where ``found`` is ``False``; an
            element the candidate policy excluded is reported, never silently mapped
            onto a neighbour.

        Raises:
            ValueError: If ``global_indices`` is not an integer array.
        """
        return locate_global_indices(self.global_indices, global_indices)


@dataclass(frozen=True, kw_only=True)
class CandidateGeometryReport:
    """What the candidate step of one frame decided and what it cost.

    It is the auditable half of a selection and holds no array, so a frame's record can
    travel downstream and be persisted without keeping the candidate coordinates alive.

    Attributes:
        policy: The policy applied.
        map_point_count: Elements the whole map holds.
        queried_count: Elements the spatial query returned, before the exact range test;
            it is how much the enclosing box over-selected.
        candidate_count: Elements the frame actually evaluates.
        query_seconds: Wall-clock time the selection took.
    """

    policy: CandidateGeometryPolicy
    map_point_count: int
    queried_count: int
    candidate_count: int
    query_seconds: float

    def __post_init__(self) -> None:
        """Validate the counts.

        Raises:
            ValueError: If a count is negative, or the counts do not narrow from the map
                through the query to the candidates.
        """
        if min(self.map_point_count, self.queried_count, self.candidate_count) < 0:
            raise ValueError("candidate counts must not be negative")
        if not self.candidate_count <= self.queried_count <= self.map_point_count:
            raise ValueError(
                f"candidate selection must narrow: {self.candidate_count} candidates of "
                f"{self.queried_count} queried of {self.map_point_count} map elements"
            )

    @property
    def candidate_fraction(self) -> float:
        """Share of the map this frame evaluates; ``0.0`` for an empty map."""
        return self.candidate_count / self.map_point_count if self.map_point_count else 0.0

    def to_record(self) -> dict[str, Any]:
        """Return the deterministic per-frame candidate facts as JSON primitives.

        ``query_seconds`` is deliberately absent: it is a wall-clock measurement, not a
        fact about the geometry, and a contractual record that held it would stop a rerun
        from reproducing the artifact's hashes. Timing is reported in ``metrics/``.
        """
        return {
            "policy": self.policy.to_record(),
            "policy_fingerprint": self.policy.fingerprint(),
            "map_point_count": self.map_point_count,
            "queried_count": self.queried_count,
            "candidate_count": self.candidate_count,
            "candidate_fraction": self.candidate_fraction,
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class CandidateGeometrySelection:
    """The candidates of one frame, with the report of how they were chosen.

    Attributes:
        cloud: The selected geometry, with its global identities.
        report: The counts, the policy and the cost of the selection.
    """

    cloud: CandidateGeometryCloud
    report: CandidateGeometryReport

    def __post_init__(self) -> None:
        """Validate that the report describes the cloud.

        Raises:
            ValueError: If the report's candidate count is not the cloud's size.
        """
        if self.report.candidate_count != len(self.cloud):
            raise ValueError(
                f"the report counts {self.report.candidate_count} candidates but the cloud "
                f"holds {len(self.cloud)}"
            )

    @property
    def candidate_count(self) -> int:
        """Elements this frame evaluates."""
        return len(self.cloud)

    @property
    def candidate_fraction(self) -> float:
        """Share of the map this frame evaluates; ``0.0`` for an empty map."""
        return self.report.candidate_fraction

    @property
    def map_point_count(self) -> int:
        """Elements the whole map holds."""
        return self.report.map_point_count

    @property
    def queried_count(self) -> int:
        """Elements the spatial query returned, before the exact range test."""
        return self.report.queried_count

    @property
    def query_seconds(self) -> float:
        """Wall-clock time the selection took."""
        return self.report.query_seconds


def select_candidate_geometry(
    source: GeometryBlockSource, *, camera_center_m: Vector3, policy: CandidateGeometryPolicy
) -> CandidateGeometrySelection:
    """Choose the geometry one camera frame evaluates.

    Args:
        source: The block read boundary of the persistent map.
        camera_center_m: The camera optical centre, in the map frame, in meters.
        policy: The candidate rule.

    Returns:
        The candidates, their global identities and what the selection cost.
    """
    import numpy as np

    started = time.perf_counter()
    geometric_map = source.geometric_map
    centre = np.array(camera_center_m, dtype=np.float64)
    radius = policy.max_range_m
    bounds = None if radius is None else _enclosing_box(geometric_map, centre, radius)
    index_parts: list[NDArray[Any]] = []
    coordinate_parts: list[NDArray[Any]] = []
    queried = 0
    for block in source.iter_blocks(bounds=bounds, block_points=policy.block_points):
        queried += len(block)
        if radius is None:
            index_parts.append(block.indices)
            coordinate_parts.append(block.coordinates_m)
            continue
        within = _within_range(block.coordinates_m, camera_center_m, radius)
        if not within.any():
            continue
        index_parts.append(block.indices[within])
        coordinate_parts.append(block.coordinates_m[within])
    indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int64)
    coordinates = (
        np.concatenate(coordinate_parts) if coordinate_parts else np.empty((0, 3), dtype=np.float64)
    )
    cloud = CandidateGeometryCloud(
        geometric_map=geometric_map, coordinates_m=coordinates, global_indices=indices
    )
    return CandidateGeometrySelection(
        cloud=cloud,
        report=CandidateGeometryReport(
            policy=policy,
            map_point_count=geometric_map.point_count,
            queried_count=queried,
            candidate_count=len(cloud),
            query_seconds=time.perf_counter() - started,
        ),
    )


_RANGE_FILTER_RELATIVE_MARGIN = 2.0**-40
"""Relative half-width, around ``max_range_m ** 2``, of the band the float filter leaves undecided.

The float squared distance of a point is within about ``6 * 2**-53`` of its exact value, so a
margin of ``2**-40`` puts every point the float filter decides far from the boundary, and only
points that close to the sphere (a few picometres at 5 m) go to the exact decision."""

_RANGE_FILTER_ABSOLUTE_MARGIN_M2 = 2.0**-1000
"""Absolute half-width of the undecided band, in square meters, covering squares that underflow."""


def _within_range(coordinates_m: NDArray[Any], centre: Vector3, radius: float) -> NDArray[Any]:
    """Decide, exactly, which points lie within ``radius`` of ``centre``.

    The criterion is the exact test ``||P - C|| <= radius`` on the float values themselves, as
    real numbers. Comparing squared distances in floating point is not that test: rounding the
    offsets, their squares and their sum disagrees with it by one unit in the last place, in both
    directions, for points on the sphere. The float comparison therefore only decides the
    points clearly inside or outside, with a margin far above its rounding error, and the few
    points in the band around the sphere are decided with exact rational arithmetic.

    Args:
        coordinates_m: ``(N, 3)`` map-frame coordinates, in meters.
        centre: The camera optical centre, in the same frame, in meters.
        radius: The range, a positive finite distance in meters.

    Returns:
        ``(N,)`` boolean mask of the points within the range, boundary included.
    """
    import numpy as np

    offsets = coordinates_m - np.array(centre, dtype=np.float64)
    squared = np.einsum("ij,ij->i", offsets, offsets)
    radius_squared = radius * radius
    band = radius_squared * _RANGE_FILTER_RELATIVE_MARGIN + _RANGE_FILTER_ABSOLUTE_MARGIN_M2
    within: NDArray[Any] = squared <= radius_squared - band
    undecided = ~within & (squared <= radius_squared + band)
    if undecided.any():
        exact_radius_squared = Fraction(radius) ** 2
        exact_centre = [Fraction(value) for value in centre]
        for row in np.flatnonzero(undecided):
            # Fraction de um float é o seu valor exato: a soma abaixo não arredonda nada.
            exact_squared = sum(
                (Fraction(float(value)) - origin) ** 2
                for value, origin in zip(coordinates_m[row], exact_centre, strict=True)
            )
            within[row] = exact_squared <= exact_radius_squared
    return within


def _enclosing_box(geometric_map: GeometricMap, centre: NDArray[Any], radius: float) -> Bounds3D:
    """Return the axis-aligned box that encloses the candidate sphere.

    The box is what the spatial query can prune with; it over-selects on purpose, and
    :func:`select_candidate_geometry` narrows it to the sphere. Over-selection is safe, false
    exclusion is not.

    The box is conservative for the exact test at the boundary too: a point in the sphere has
    ``C - r <= P <= C + r`` on every axis, as real numbers, and rounding is monotonic and leaves a
    float coordinate unchanged, so ``fl(C - r) <= P <= fl(C + r)`` as well; the query compares
    the faces inclusively, without arithmetic.
    """
    low = centre - radius
    high = centre + radius
    return Bounds3D(
        frame_id=geometric_map.frame_id,
        minimum_m=(float(low[0]), float(low[1]), float(low[2])),
        maximum_m=(float(high[0]), float(high[1]), float(high[2])),
    )
