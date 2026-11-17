"""A política versionada que produz os resumos espaciais de uma entidade, e sua verificação.

Aqui ficam as escolhas *científicas* do resumo baseline `entity-geometry-summary-v1`: quais
limiares tornam um suporte esparso, a que raio dois pontos contam como ligados, quando uma
orientação é justificada e qual o maior suporte cujos componentes vale a pena contar. Nenhuma
delas tem default — um perfil as declara — e todas entram no fingerprint da configuração.

A matemática que executa essas decisões é de `support_shape.py`; os contratos que guardam o
resultado são de `geometry.py`. Este módulo é o único que conhece os três.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.geometric_mapping import Bounds3D, GeometryReference, GeometrySource
from contextmap.semantic_mapping.fingerprint import canonical_digest, geometry_set_digest
from contextmap.semantic_mapping.geometry import (
    EmptyGeometrySupportError,
    EntityGeometry,
    EntityOrientation,
    GeometryDiagnostic,
    GeometryDiagnosticKind,
    GeometryResolutionError,
    SpatialSummaryProvenance,
    SupportStatistics,
    resolve_geometry,
)
from contextmap.semantic_mapping.support_shape import (
    DEFAULT_MAX_CONNECTIVITY_POINTS,
    ORIENTATION_METHOD,
    SUMMARY_TOLERANCE_M,
    centroid_of,
    component_sizes,
    principal_axes,
    vector_of,
)
from contextmap.shared import Vector3

GEOMETRY_SUMMARY_ALGORITHM_ID = "entity-geometry-summary-v1"
"""Versioned identity of the baseline spatial summary described in this module."""

_NUMERICAL_CONVENTIONS = (
    "float64; centroid=arithmetic mean (fsum), clamped into the bounds; bounds=min/max; "
    "extent=max-min; volume=extent product; covariance normalized by n; axes sorted by "
    "descending variance, sign fixed so the largest component is positive, third axis is the "
    "cross product of the first two"
)

_NO_FILTERING = "none: the whole support is used, nothing is filtered or downsampled"


@dataclass(frozen=True, kw_only=True)
class OrientationPolicy:
    """When an orientation is mathematically justified.

    There are no defaults: the thresholds are scientific choices that a profile declares.

    Attributes:
        min_points: Fewest points the support needs for its axes to mean anything.
        min_variance_ratio: How much larger than the next each variance must be, so the axes
            are distinct; ``1`` accepts any ordering and larger values demand a clearer one.
    """

    min_points: int
    min_variance_ratio: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If ``min_points`` is below three or the ratio is not finite and above
                one.
        """
        if self.min_points < 3:
            raise ValueError(f"min_points must be at least 3, got {self.min_points}")
        if not (math.isfinite(self.min_variance_ratio) and self.min_variance_ratio > 1.0):
            raise ValueError(
                f"min_variance_ratio must be finite and above 1, got {self.min_variance_ratio!r}"
            )


@dataclass(frozen=True, kw_only=True)
class GeometrySummaryPolicy:
    """Explicit configuration of the baseline spatial summary.

    There are no defaults: the thresholds are scientific choices that a profile declares.

    Attributes:
        sparse_point_threshold: Supports with fewer points are flagged as sparse.
        connectivity_radius_m: Distance under which two points are linked, in meters.
        orientation: When to derive an orientation, or ``None`` to never derive one.
        max_connectivity_points: Largest support whose connected components are counted. A
            larger support is not linked, so its components are ``None`` and a diagnostic says
            so; it bounds the cost of the connectivity check (see
            :data:`DEFAULT_MAX_CONNECTIVITY_POINTS`).
    """

    sparse_point_threshold: int
    connectivity_radius_m: float
    orientation: OrientationPolicy | None = None
    max_connectivity_points: int = DEFAULT_MAX_CONNECTIVITY_POINTS

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ValueError: If the sparse threshold or the connectivity limit is below one or the
                radius is not finite and positive.
        """
        if self.sparse_point_threshold < 1:
            raise ValueError(
                f"sparse_point_threshold must be at least 1, got {self.sparse_point_threshold}"
            )
        if self.max_connectivity_points < 1:
            raise ValueError(
                f"max_connectivity_points must be at least 1, got {self.max_connectivity_points}"
            )
        if not (math.isfinite(self.connectivity_radius_m) and self.connectivity_radius_m > 0.0):
            raise ValueError(
                f"connectivity_radius_m must be finite and positive, got "
                f"{self.connectivity_radius_m!r}"
            )

    def fingerprint(self) -> str:
        """Hash the algorithm identity and thresholds, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        return canonical_digest(
            {
                "algorithm_id": GEOMETRY_SUMMARY_ALGORITHM_ID,
                "sparse_point_threshold": self.sparse_point_threshold,
                "connectivity_radius_m": self.connectivity_radius_m,
                "max_connectivity_points": self.max_connectivity_points,
                "orientation": None
                if self.orientation is None
                else {
                    "min_points": self.orientation.min_points,
                    "min_variance_ratio": self.orientation.min_variance_ratio,
                },
            }
        )


def summarize_geometry(
    references: Sequence[GeometryReference],
    *,
    source: GeometrySource,
    policy: GeometrySummaryPolicy,
) -> EntityGeometry:
    """Build the geometry of an entity from its supporting references.

    Args:
        references: The support, in any order; it is sorted into canonical order.
        source: The read boundary of the geometric map the references belong to.
        policy: The thresholds of the baseline summary.

    Returns:
        The entity geometry, with every derived summary in the map frame.

    Raises:
        EmptyGeometrySupportError: If there is no support.
        GeometryResolutionError: If a reference cannot be resolved against the map.
        ValueError: If a reference is repeated.
    """
    if not references:
        raise EmptyGeometrySupportError("an entity needs at least one geometry reference")
    ordered = tuple(sorted(references, key=lambda reference: reference.geometry_id))
    points = resolve_geometry(ordered, source=source)
    frame = source.geometric_map.frame_id
    coordinates = [point.coordinates_m for point in points]
    bounds = Bounds3D.enclosing(coordinates, frame_id=frame)
    centroid = centroid_of(coordinates, bounds)
    extent = vector_of(
        high - low for low, high in zip(bounds.minimum_m, bounds.maximum_m, strict=True)
    )
    volume = math.prod(extent)
    # O custo da conectividade cresce ~quadraticamente com a densidade local: acima do limite da
    # política os componentes não são contados, e o diagnóstico diz isso em vez de demorar sem teto.
    sizes = (
        None
        if len(points) > policy.max_connectivity_points
        else component_sizes(coordinates, policy.connectivity_radius_m)
    )
    statistics = SupportStatistics(
        point_count=len(points),
        volume_m3=volume,
        density_per_m3=None if volume == 0.0 else len(points) / volume,
        component_count=None if sizes is None else len(sizes),
        largest_component_fraction=None if sizes is None else max(sizes) / len(points),
    )
    diagnostics = {item.kind: item for item in _diagnostics(statistics, extent, policy)}
    orientation = None
    if policy.orientation is not None:
        axes, reason = principal_axes(
            coordinates,
            centroid,
            min_points=policy.orientation.min_points,
            min_variance_ratio=policy.orientation.min_variance_ratio,
        )
        orientation = (
            None
            if axes is None
            else EntityOrientation(
                axes=axes.axes, variances_m2=axes.variances_m2, method=ORIENTATION_METHOD
            )
        )
        if orientation is None and reason is not None:
            item = GeometryDiagnostic(
                kind=GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED, detail=reason
            )
            diagnostics[item.kind] = item
    return EntityGeometry(
        geometry_refs=ordered,
        map_frame=frame,
        centroid_m=centroid,
        bounds=bounds,
        extent_m=extent,
        statistics=statistics,
        summary=SpatialSummaryProvenance(
            algorithm_id=GEOMETRY_SUMMARY_ALGORITHM_ID,
            map_frame=frame,
            input_geometry_count=len(ordered),
            input_geometry_digest=geometry_set_digest(ordered),
            numerical_conventions=_NUMERICAL_CONVENTIONS,
            filtering=_NO_FILTERING,
            configuration_fingerprint=policy.fingerprint(),
        ),
        orientation=orientation,
        diagnostics=tuple(sorted(diagnostics.values(), key=lambda item: item.kind.value)),
    )


def verify_geometry_summary(
    geometry: EntityGeometry, *, source: GeometrySource, policy: GeometrySummaryPolicy
) -> list[str]:
    """Recompute the summaries of an entity from the persistent geometry and compare.

    Args:
        geometry: The geometry of an entity.
        source: The read boundary of the geometric map it references.
        policy: The policy the summaries are expected to follow.

    Returns:
        Human-readable problems; empty means the summaries are the ones the persistent
        geometry and the policy produce. An unresolvable support is reported, not raised.
    """
    try:
        expected = summarize_geometry(geometry.geometry_refs, source=source, policy=policy)
    except GeometryResolutionError as error:
        return [f"the support cannot be resolved: {error}"]
    problems: list[str] = []
    if geometry.summary.configuration_fingerprint != expected.summary.configuration_fingerprint:
        problems.append("the summary was computed under a different policy")
    _compare_vector("centroid_m", geometry.centroid_m, expected.centroid_m, problems)
    _compare_vector(
        "bounds.minimum_m", geometry.bounds.minimum_m, expected.bounds.minimum_m, problems
    )
    _compare_vector(
        "bounds.maximum_m", geometry.bounds.maximum_m, expected.bounds.maximum_m, problems
    )
    _compare_vector("extent_m", geometry.extent_m, expected.extent_m, problems)
    if geometry.statistics.point_count != expected.statistics.point_count:
        problems.append("statistics.point_count differs from the persistent geometry")
    if geometry.statistics.component_count != expected.statistics.component_count:
        problems.append("statistics.component_count differs from the persistent geometry")
    if (geometry.orientation is None) != (expected.orientation is None):
        problems.append("the presence of an orientation differs from the persistent geometry")
    elif geometry.orientation is not None and expected.orientation is not None:
        _compare_vector(
            "orientation.variances_m2",
            geometry.orientation.variances_m2,
            expected.orientation.variances_m2,
            problems,
        )
    if geometry.diagnostic_kinds() != expected.diagnostic_kinds():
        problems.append("the diagnostics differ from the persistent geometry")
    return problems


def _compare_vector(name: str, got: Vector3, want: Vector3, problems: list[str]) -> None:
    if any(
        not math.isclose(a, b, abs_tol=SUMMARY_TOLERANCE_M) for a, b in zip(got, want, strict=True)
    ):
        problems.append(f"{name} {got!r} differs from the persistent geometry, {want!r}")


def _diagnostics(
    statistics: SupportStatistics, extent: Vector3, policy: GeometrySummaryPolicy
) -> list[GeometryDiagnostic]:
    found: list[GeometryDiagnostic] = []
    if statistics.point_count < policy.sparse_point_threshold:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.SPARSE_SUPPORT,
                detail=(
                    f"{statistics.point_count} points, fewer than the "
                    f"{policy.sparse_point_threshold} the policy considers reliable"
                ),
            )
        )
    if statistics.component_count is None:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.CONNECTIVITY_NOT_COMPUTED,
                detail=(
                    f"{statistics.point_count} points, above the connectivity limit of "
                    f"{policy.max_connectivity_points} the policy accepts: the connected "
                    f"components were not counted"
                ),
            )
        )
    elif statistics.component_count > 1:
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.DISCONNECTED_SUPPORT,
                detail=(
                    f"{statistics.component_count} connected components at "
                    f"{policy.connectivity_radius_m} m; the largest holds "
                    f"{statistics.largest_component_fraction:.3f} of the points"
                ),
            )
        )
    if any(side == 0.0 for side in extent):
        found.append(
            GeometryDiagnostic(
                kind=GeometryDiagnosticKind.DEGENERATE_EXTENT,
                detail=(
                    "the bounds are flat on at least one axis: volume and density are not "
                    "meaningful"
                ),
            )
        )
    return found
