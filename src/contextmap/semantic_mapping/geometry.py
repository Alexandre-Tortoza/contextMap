"""O suporte 3D de uma entidade e os contratos dos resumos derivados dele.

Uma entidade guarda a geometria persistente **exata** que justifica sua existência espacial, como
referências para um mapa geométrico imutável. Essa geometria permanece a autoridade: centroide,
limites, extensão, estatísticas e orientação são resumos *derivados* dela, expressos no mesmo frame
do mapa, e registram como foram derivados para que possam ser recalculados e comparados. Nada aqui
copia XYZ para dentro da entidade, e um centroide nunca é a única representação espacial.

Este módulo possui os contratos e a resolução das referências contra o mapa. Os limiares que
produzem os resumos, o algoritmo que os calcula e a verificação pertencem a `geometry_summary.py`
e a `support_shape.py`: um contrato nunca importa a política que o preenche.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from contextmap.geometric_mapping import (
    Bounds3D,
    GeometryPoint,
    GeometryReference,
    GeometrySource,
    MapId,
)
from contextmap.ingestion import FrameId
from contextmap.semantic_mapping.fingerprint import geometry_set_digest
from contextmap.semantic_mapping.invariants import require_canonical, require_present
from contextmap.semantic_mapping.support_shape import SUMMARY_TOLERANCE_M, cross, dot, norm
from contextmap.shared import Vector3

_UNIT_TOLERANCE = 1e-6


class EmptyGeometrySupportError(ValueError):
    """Raised when an entity would have no 3D support."""


class GeometryResolutionError(Exception):
    """Raised when geometry references cannot be resolved against the map that owns them."""


class GeometryDiagnosticKind(Enum):
    """Why a spatial summary should be read with care.

    Attributes:
        SPARSE_SUPPORT: The support has fewer points than the policy considers reliable.
        DISCONNECTED_SUPPORT: The support falls apart into several connected components.
        DEGENERATE_EXTENT: The bounds are flat on some axis, so volume and density are not
            meaningful.
        ORIENTATION_NOT_JUSTIFIED: An orientation was requested but its principal axes are not
            well defined for this support.
        CONNECTIVITY_NOT_COMPUTED: The support has more points than the policy's connectivity
            limit, so its connected components were not counted; it may or may not be connected.
    """

    SPARSE_SUPPORT = "sparse_support"
    DISCONNECTED_SUPPORT = "disconnected_support"
    DEGENERATE_EXTENT = "degenerate_extent"
    ORIENTATION_NOT_JUSTIFIED = "orientation_not_justified"
    CONNECTIVITY_NOT_COMPUTED = "connectivity_not_computed"


@dataclass(frozen=True, kw_only=True)
class GeometryDiagnostic:
    """A caveat about a spatial summary, stated instead of hidden.

    Attributes:
        kind: What kind of caveat this is.
        detail: A deterministic, human-readable explanation.
    """

    kind: GeometryDiagnosticKind
    detail: str

    def __post_init__(self) -> None:
        """Require an explanation.

        Raises:
            ValueError: If the detail is empty.
        """
        require_present(self, "detail")


@dataclass(frozen=True, kw_only=True)
class SupportStatistics:
    """Size, density and connectivity of the support.

    Attributes:
        point_count: Geometry elements in the support.
        volume_m3: Volume of the bounds, in cubic meters.
        density_per_m3: ``point_count`` over ``volume_m3``; ``None`` when the bounds are flat.
        component_count: Connected components at the policy's connectivity radius; ``None`` when
            the support is above the policy's connectivity limit and they were not counted,
            which is not the same as one component.
        largest_component_fraction: Share of the points in the largest component, in ``(0, 1]``;
            ``None`` exactly when ``component_count`` is.
    """

    point_count: int
    volume_m3: float
    density_per_m3: float | None
    component_count: int | None
    largest_component_fraction: float | None

    def __post_init__(self) -> None:
        """Validate that the figures are coherent with each other.

        Raises:
            ValueError: If the count or components are not positive, only one of the two
                connectivity figures is present, a figure is not finite or negative, the density
                does not match the volume, or the largest component fraction is outside
                ``(0, 1]``.
        """
        if (self.component_count is None) != (self.largest_component_fraction is None):
            raise ValueError(
                "component_count and largest_component_fraction must be both present or both None"
            )
        if self.point_count < 1 or (self.component_count is not None and self.component_count < 1):
            raise ValueError("point_count and component_count must be at least 1")
        if self.component_count is not None and self.component_count > self.point_count:
            raise ValueError("component_count cannot exceed point_count")
        if not (math.isfinite(self.volume_m3) and self.volume_m3 >= 0.0):
            raise ValueError(f"volume_m3 must be finite and not negative, got {self.volume_m3!r}")
        if self.volume_m3 == 0.0:
            if self.density_per_m3 is not None:
                raise ValueError("density_per_m3 must be None when the volume is zero")
        elif self.density_per_m3 is None or not math.isclose(
            self.density_per_m3, self.point_count / self.volume_m3, rel_tol=1e-9
        ):
            raise ValueError("density_per_m3 must equal point_count over volume_m3")
        if self.largest_component_fraction is not None and not (
            0.0 < self.largest_component_fraction <= 1.0
        ):
            raise ValueError(
                f"largest_component_fraction must be within (0, 1], got "
                f"{self.largest_component_fraction!r}"
            )


@dataclass(frozen=True, kw_only=True)
class EntityOrientation:
    """The principal axes of an entity's support, when they are well defined.

    Attributes:
        axes: Three unit axes in the map frame, orthogonal and right-handed, sorted by
            descending variance.
        variances_m2: The variance of the support along each axis, in square meters,
            non-increasing.
        method: The versioned method that derived the axes.
    """

    axes: tuple[Vector3, Vector3, Vector3]
    variances_m2: Vector3
    method: str

    def __post_init__(self) -> None:
        """Validate that the axes are an orthonormal right-handed frame and the variances ordered.

        Raises:
            ValueError: If the method is empty, an axis is not a finite unit vector, the axes
                are not orthogonal or right-handed, or the variances are negative, not finite
                or not sorted in descending order.
        """
        require_present(self, "method")
        first, second, third = self.axes
        for axis in self.axes:
            if not all(math.isfinite(value) for value in axis) or not math.isclose(
                norm(axis), 1.0, abs_tol=_UNIT_TOLERANCE
            ):
                raise ValueError(f"orientation axes must be finite unit vectors, got {axis!r}")
        if any(
            abs(dot(left, right)) > _UNIT_TOLERANCE
            for left, right in ((first, second), (first, third), (second, third))
        ):
            raise ValueError("orientation axes must be orthogonal")
        if dot(cross(first, second), third) < 1.0 - _UNIT_TOLERANCE:
            raise ValueError("orientation axes must be right-handed")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.variances_m2):
            raise ValueError(
                f"variances must be finite and not negative, got {self.variances_m2!r}"
            )
        if not (self.variances_m2[0] >= self.variances_m2[1] >= self.variances_m2[2]):
            raise ValueError("variances must be sorted in descending order")


@dataclass(frozen=True, kw_only=True)
class SpatialSummaryProvenance:
    """How the derived spatial summaries of an entity were computed.

    Attributes:
        algorithm_id: Versioned algorithm.
        map_frame: The frame every derived value is expressed in.
        input_geometry_count: Geometry elements the summaries were computed from.
        input_geometry_digest: Digest of the exact set of geometry identities used.
        numerical_conventions: The numerical conventions the values follow.
        filtering: Whether any filtering or downsampling was applied, stated explicitly.
        configuration_fingerprint: Hash of the policy the summaries were computed under.
    """

    algorithm_id: str
    map_frame: FrameId
    input_geometry_count: int
    input_geometry_digest: str
    numerical_conventions: str
    filtering: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        """Require every field to say something.

        Raises:
            ValueError: If a text field is empty or the input set is empty.
        """
        require_present(
            self,
            "algorithm_id",
            "map_frame",
            "input_geometry_digest",
            "numerical_conventions",
            "filtering",
            "configuration_fingerprint",
        )
        if self.input_geometry_count < 1:
            raise ValueError("input_geometry_count must be at least 1")


@dataclass(frozen=True, kw_only=True)
class EntityGeometry:
    """The persistent geometry that supports one entity, with reproducible summaries.

    Attributes:
        geometry_refs: References to the supporting geometry, sorted by ``geometry_id``,
            unique and never empty: an entity without real 3D support is not an entity. This
            is the authoritative support; every other field is derived from it.
        map_frame: The frame of the map the geometry belongs to.
        centroid_m: Centroid of the support, in meters, in ``map_frame``.
        bounds: Tight axis-aligned box of the support, in ``map_frame``.
        extent_m: Side lengths of ``bounds``, in meters.
        statistics: Size, density and connectivity of the support.
        summary: How the summaries were derived.
        orientation: Principal axes of the support, only when the policy asked for them and they
            are well defined.
        diagnostics: Caveats about the summaries, sorted by kind and unique.
    """

    geometry_refs: tuple[GeometryReference, ...]
    map_frame: FrameId
    centroid_m: Vector3
    bounds: Bounds3D
    extent_m: Vector3
    statistics: SupportStatistics
    summary: SpatialSummaryProvenance
    orientation: EntityOrientation | None = None
    diagnostics: tuple[GeometryDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the support is present and every summary agrees with it.

        Raises:
            EmptyGeometrySupportError: If the support is empty.
            ValueError: If the frame is empty, the support is not sorted by geometry and unique
                or spans more than one map, a summary is not in ``map_frame``, the centroid is not
                finite or lies outside the bounds, the extent is not the bounds' size, the
                statistics or the provenance disagree with the support, or the diagnostics
                contradict the statistics.
        """
        require_present(self, "map_frame")
        if not self.geometry_refs:
            raise EmptyGeometrySupportError(
                "geometry_refs must not be empty: an entity needs 3D support"
            )
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda reference: (reference.geometry_id,),
            detail="by geometry_id ",
        )
        maps = {reference.map_id for reference in self.geometry_refs}
        if len(maps) > 1:
            raise ValueError(f"geometry_refs must belong to one map, got {sorted(maps)!r}")
        self._require_summaries_in_frame()
        self._require_summaries_match_support()
        self._require_diagnostics_match()

    @property
    def geometric_map_id(self) -> MapId:
        """The immutable geometric map the support belongs to."""
        return self.geometry_refs[0].map_id

    def diagnostic_kinds(self) -> frozenset[GeometryDiagnosticKind]:
        """The kinds of caveat attached to the summaries."""
        return frozenset(item.kind for item in self.diagnostics)

    def _require_summaries_in_frame(self) -> None:
        if self.bounds.frame_id != self.map_frame:
            raise ValueError(
                f"bounds are expressed in {self.bounds.frame_id!r}, not the map frame "
                f"{self.map_frame!r}"
            )
        if self.summary.map_frame != self.map_frame:
            raise ValueError(
                f"the summary was derived in {self.summary.map_frame!r}, not the map frame "
                f"{self.map_frame!r}"
            )

    def _require_summaries_match_support(self) -> None:
        if not all(math.isfinite(value) for value in self.centroid_m):
            raise ValueError(f"centroid_m must be finite, got {self.centroid_m!r}")
        if not self.bounds.contains(self.centroid_m, frame_id=self.bounds.frame_id):
            raise ValueError(
                f"centroid_m {self.centroid_m!r} must lie inside the bounds "
                f"{self.bounds.minimum_m!r}..{self.bounds.maximum_m!r}"
            )
        expected = tuple(
            high - low
            for low, high in zip(self.bounds.minimum_m, self.bounds.maximum_m, strict=True)
        )
        if any(
            not math.isclose(got, want, abs_tol=SUMMARY_TOLERANCE_M)
            for got, want in zip(self.extent_m, expected, strict=True)
        ):
            raise ValueError(
                f"extent_m {self.extent_m!r} must be the size of the bounds, {expected!r}"
            )
        count = len(self.geometry_refs)
        if self.statistics.point_count != count or self.summary.input_geometry_count != count:
            raise ValueError(
                f"statistics and summary must describe the {count} geometry references, got "
                f"{self.statistics.point_count} and {self.summary.input_geometry_count}"
            )
        if self.summary.input_geometry_digest != geometry_set_digest(self.geometry_refs):
            raise ValueError("the summary was not derived from exactly these geometry references")
        volume = math.prod(self.extent_m)
        if not math.isclose(
            self.statistics.volume_m3, volume, rel_tol=1e-9, abs_tol=SUMMARY_TOLERANCE_M
        ):
            raise ValueError(
                f"volume_m3 {self.statistics.volume_m3!r} must be the volume of the bounds, "
                f"{volume!r}"
            )

    def _require_diagnostics_match(self) -> None:
        require_canonical(
            "diagnostics", self.diagnostics, lambda item: (item.kind.value,), detail="by kind "
        )
        kinds = self.diagnostic_kinds()
        components = self.statistics.component_count
        disconnected = components is not None and components > 1
        if disconnected != (GeometryDiagnosticKind.DISCONNECTED_SUPPORT in kinds):
            raise ValueError(
                "the disconnected_support diagnostic must be present exactly when the support "
                "has more than one connected component"
            )
        if (components is None) != (GeometryDiagnosticKind.CONNECTIVITY_NOT_COMPUTED in kinds):
            raise ValueError(
                "the connectivity_not_computed diagnostic must be present exactly when the "
                "connected components were not counted"
            )
        flat = any(side == 0.0 for side in self.extent_m)
        if flat != (GeometryDiagnosticKind.DEGENERATE_EXTENT in kinds):
            raise ValueError(
                "the degenerate_extent diagnostic must be present exactly when the bounds are "
                "flat on some axis"
            )
        if (
            self.orientation is not None
            and GeometryDiagnosticKind.ORIENTATION_NOT_JUSTIFIED in kinds
        ):
            raise ValueError(
                "an orientation cannot coexist with an orientation_not_justified diagnostic"
            )


def resolve_geometry(
    references: Sequence[GeometryReference], *, source: GeometrySource
) -> tuple[GeometryPoint, ...]:
    """Resolve references to their authoritative geometry, in the order given.

    Args:
        references: References into the map ``source`` serves.
        source: The read boundary of the geometric map.

    Returns:
        The geometry of every reference, from the persisted map.

    Raises:
        GeometryResolutionError: If a reference belongs to another map, does not exist in this
            one, or resolves to geometry expressed in a frame other than the map's.
    """
    geometric_map = source.geometric_map
    foreign = sorted({reference.map_id for reference in references} - {geometric_map.map_id})
    if foreign:
        raise GeometryResolutionError(
            f"references belong to map(s) {foreign!r}, but the source serves "
            f"{geometric_map.map_id!r}"
        )
    points: list[GeometryPoint] = []
    missing: list[str] = []
    for reference in references:
        try:
            points.append(source.get(reference))
        except KeyError:
            missing.append(reference.geometry_id)
    if missing:
        raise GeometryResolutionError(
            f"{len(missing)} geometry reference(s) do not exist in map {geometric_map.map_id!r}: "
            f"{missing[:3]!r}"
        )
    wrong_frame = sorted({point.map_frame for point in points} - {geometric_map.frame_id})
    if wrong_frame:
        raise GeometryResolutionError(
            f"geometry is expressed in {wrong_frame!r}, not the map frame "
            f"{geometric_map.frame_id!r}"
        )
    return tuple(points)
