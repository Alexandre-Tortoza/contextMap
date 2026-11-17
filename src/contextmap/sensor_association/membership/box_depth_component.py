"""Suporte de regiões sem máscara por componentes de profundidade 3D.

Seleciona suporte sob política explícita e recusa insuficiência ou ambiguidade.
Uma caixa também cobre fundo e oclusores; seus pontos não são aceitos em bloco.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from contextmap.region_perception import Region2D
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.membership.models import RegionMembership, SkipReason
from contextmap.sensor_association.models import FootprintKind
from contextmap.sensor_association.projection_chain.frame_projection import FrameProjection

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class BoxDepthComponentPolicy:
    """Membership for regions that have a box and no mask (``box-depth-component-v1``).

    A box also covers background and occluders, so its points are never accepted as
    a whole. The visible points whose prepared pixel lies in the box are grouped into 3D
    components: cells of ``cell_size_m`` in the camera frame, connected when they touch
    (26-neighbourhood). The component with the most points in the box *core* -- the central
    ``core_fraction`` of each side -- is the support; the rest of the box stays visible and
    unassigned. A box is rejected, with its reason, when no component has
    ``minimum_support_points`` with at least one in the core, or when the second component's
    core count reaches ``ambiguity_ratio`` times the best one's. A region with a mask keeps
    the mask rule.

    There are no defaults: the cell size depends on the map density.

    Attributes:
        cell_size_m: Side of a connectivity cell, in meters.
        core_fraction: Share of the box width and height that forms its core, in ``(0, 1]``.
        minimum_support_points: Fewest points the selected component may hold.
        ambiguity_ratio: Second-best over best core count from which the box is ambiguous,
            in ``(0, 1]``.
    """

    policy_id: ClassVar[str] = "box-depth-component-v1"

    cell_size_m: float
    core_fraction: float
    minimum_support_points: int
    ambiguity_ratio: float

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is outside its range or not finite.
        """
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0:
            raise ValueError(f"cell_size_m must be finite and positive, got {self.cell_size_m!r}")
        for name in ("core_fraction", "ambiguity_ratio"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 < value <= 1.0):
                raise ValueError(f"{name} must be within (0, 1], got {value!r}")
        if self.minimum_support_points < 1:
            raise ValueError(
                f"minimum_support_points must be at least 1, got {self.minimum_support_points}"
            )

    def to_record(self) -> dict[str, Any]:
        """Return the policy as JSON primitives, with its versioned identity."""
        return {
            "policy_id": self.policy_id,
            "cell_size_m": self.cell_size_m,
            "core_fraction": self.core_fraction,
            "minimum_support_points": self.minimum_support_points,
            "ambiguity_ratio": self.ambiguity_ratio,
        }


def box_membership(
    region: Region2D,
    policy: BoxDepthComponentPolicy,
    *,
    frame: FrameProjection,
    in_image: NDArray[Any],
    columns: NDArray[Any],
    rows: NDArray[Any],
    visible: NDArray[Any],
    occluded: NDArray[Any],
    unsupported: NDArray[Any],
) -> RegionMembership | SkipReason:
    """Apply the box rule of :class:`BoxDepthComponentPolicy` to one region without mask.

    The arrays are those of :func:`associate_regions`, restricted to the in-image rows.

    Raises:
        AssociationInputError: If the region declares an image size other than the prepared
            image's, so its box is not in the space the geometry was projected into.
    """
    import numpy as np

    width, height = frame.image_transform.prepared_size
    if (region.image_width, region.image_height) not in ((None, None), (width, height)):
        raise AssociationInputError(
            f"the box of region {region.region_id!r} is in a {region.image_width}x"
            f"{region.image_height} image but the prepared image is {width}x{height}: regions "
            f"must be expressed in the prepared image the geometry was projected into"
        )
    box = region.bounding_box
    x0, x1 = _pixel_span(box.x_min, box.x_max, width)
    y0, y1 = _pixel_span(box.y_min, box.y_max, height)
    if x0 >= x1 or y0 >= y1:
        return SkipReason.EMPTY_BOX
    inside = (columns >= x0) & (columns < x1) & (rows >= y0) & (rows < y1)
    candidates = np.flatnonzero(inside & visible)
    core_half_width = policy.core_fraction * (box.x_max - box.x_min) / 2
    core_half_height = policy.core_fraction * (box.y_max - box.y_min) / 2
    center_x, center_y = (box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2
    cx0, cx1 = _pixel_span(center_x - core_half_width, center_x + core_half_width, width)
    cy0, cy1 = _pixel_span(center_y - core_half_height, center_y + core_half_height, height)
    candidate_columns, candidate_rows = columns[candidates], rows[candidates]
    in_core = (
        (candidate_columns >= cx0)
        & (candidate_columns < cx1)
        & (candidate_rows >= cy0)
        & (candidate_rows < cy1)
    )
    labels = _cell_components(frame.camera_points_m[in_image[candidates]], policy.cell_size_m)
    if labels.size == 0:
        return SkipReason.INSUFFICIENT_BOX_SUPPORT
    sizes = np.bincount(labels)
    core_counts = np.bincount(labels, weights=in_core, minlength=sizes.shape[0])
    # Ordem total e determinística: mais pontos no núcleo, depois mais pontos, depois o
    # rótulo menor (o rótulo é a menor célula ocupada do componente).
    order = np.lexsort((np.arange(sizes.shape[0]), -sizes, -core_counts))
    best = int(order[0])
    if core_counts[best] < 1 or sizes[best] < policy.minimum_support_points:
        return SkipReason.INSUFFICIENT_BOX_SUPPORT
    if order.shape[0] > 1 and core_counts[order[1]] >= policy.ambiguity_ratio * core_counts[best]:
        return SkipReason.AMBIGUOUS_BOX_SUPPORT
    selected = candidates[labels == best]
    associated_indices = in_image[selected]
    covered = np.unique(rows[selected] * width + columns[selected]).shape[0]
    return RegionMembership(
        region_id=region.region_id,
        footprint_kind=FootprintKind.BOX,
        footprint_area_px=(x1 - x0) * (y1 - y0),
        associated_indices=associated_indices,
        excluded_visible_count=int(candidates.shape[0] - selected.shape[0]),
        occluded_count=int((inside & occluded).sum()),
        outside_valid_support_count=int((inside & unsupported).sum()),
        covered_pixel_count=int(covered),
        membership_rule_id=BoxDepthComponentPolicy.policy_id,
    )


def _pixel_span(low: float, high: float, size: int) -> tuple[int, int]:
    """Return the half-open range of pixel indices whose center ``i + 0.5`` is in ``[low, high)``.

    It is the rasterization of a box used by region normalization, clipped to the image.
    """
    start = max(0, math.ceil(low - 0.5))
    stop = min(size, math.ceil(high - 0.5))
    return start, stop


def _cell_components(points_m: NDArray[Any], cell_size_m: float) -> NDArray[Any]:
    """Label each point with its 3D component: touching occupied cells are connected.

    Points are binned into cubic cells of ``cell_size_m``; two occupied cells are connected
    when they share a face, an edge or a corner (26-neighbourhood). A component's label is
    the index of its smallest occupied cell in lexicographic order, so labels are dense,
    deterministic and do not depend on the order of the points.

    Returns:
        ``(N,)`` integer labels in ``[0, number of components)``.
    """
    import numpy as np

    if points_m.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    cells = np.floor(points_m / cell_size_m).astype(np.int64)
    occupied, point_cell = np.unique(cells, axis=0, return_inverse=True)
    point_cell = point_cell.reshape(-1)
    # Chave escalar por célula, com folga de uma célula em cada eixo para os vizinhos.
    origin = occupied.min(axis=0) - 1
    extent = occupied.max(axis=0) - origin + 2
    keys = _cell_keys(occupied - origin, extent)
    parent = np.arange(occupied.shape[0], dtype=np.int64)
    sources: list[NDArray[Any]] = []
    targets: list[NDArray[Any]] = []
    for offset in _HALF_NEIGHBOURHOOD:
        neighbour = _cell_keys(occupied - origin + np.array(offset), extent)
        position = np.searchsorted(keys, neighbour)
        position = np.minimum(position, keys.shape[0] - 1)
        found = keys[position] == neighbour
        sources.append(np.flatnonzero(found))
        targets.append(position[found])
    left = np.concatenate(sources)
    right = np.concatenate(targets)
    # Propagação do menor rótulo pelas arestas, com salto de ponteiros, até estabilizar.
    while True:
        previous = parent.copy()
        smallest = np.minimum(parent[left], parent[right])
        np.minimum.at(parent, left, smallest)
        np.minimum.at(parent, right, smallest)
        parent = parent[parent]
        if np.array_equal(parent, previous):
            break
    _, dense = np.unique(parent, return_inverse=True)
    labels: NDArray[Any] = dense.reshape(-1)[point_cell]
    return labels


def _cell_keys(cells: NDArray[Any], extent: NDArray[Any]) -> NDArray[Any]:
    """Encode non-negative cell coordinates within ``extent`` as one sortable integer."""
    keys: NDArray[Any] = (cells[:, 0] * extent[1] + cells[:, 1]) * extent[2] + cells[:, 2]
    return keys


_HALF_NEIGHBOURHOOD: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)
"""The 13 offsets of the 26-neighbourhood that come after the cell itself; the other 13 are
their opposites, so each touching pair is visited once."""
