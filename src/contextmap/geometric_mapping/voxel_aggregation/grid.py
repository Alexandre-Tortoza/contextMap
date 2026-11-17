"""Especificação e indexação da grade de voxels no frame do mapa.

Declara origem, resolução e fronteiras semiabertas com validação de chaves exatas.
Não persiste agregados; NumPy é carregado apenas nas operações numéricas.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.ingestion import FrameId
from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

VOXEL_INDEXING = "floor-half-open"
"""Voxel ``k`` covers ``[origin + k * cell_m, origin + (k + 1) * cell_m)`` on every axis."""

# Uma chave só é exata enquanto o quociente cabe na mantissa do float64.
_MAX_EXACT_KEY = 2**53
_EPSILON = 2.0**-52


class VoxelAggregationError(ValueError):
    """Raised when a raw map cannot be aggregated without breaking an invariant."""


@dataclass(frozen=True, kw_only=True)
class VoxelGridSpec:
    """The identity of a voxel grid: frame, origin, resolution and indexing convention.

    Attributes:
        frame_id: The map frame the grid is laid in; it must be the raw map's frame.
        origin_m: The corner of voxel ``(0, 0, 0)``, in meters, in ``frame_id``.
        cell_m: Voxel edge length, in meters.
    """

    frame_id: FrameId
    origin_m: Vector3
    cell_m: float

    def __post_init__(self) -> None:
        """Require a usable grid.

        Raises:
            ValueError: If the frame is empty, the origin is not finite, or ``cell_m``
                is not positive and finite.
        """
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if len(self.origin_m) != 3 or not all(math.isfinite(value) for value in self.origin_m):
            raise ValueError(f"origin_m must be three finite coordinates, got {self.origin_m}")
        if not (math.isfinite(self.cell_m) and self.cell_m > 0):
            raise ValueError(f"cell_m must be positive and finite, got {self.cell_m}")

    @property
    def indexing(self) -> str:
        """The indexing convention; part of the grid identity."""
        return VOXEL_INDEXING

    def keys_of(self, coordinates_m: NDArray[Any]) -> NDArray[Any]:
        """Return the voxel key of every coordinate.

        Args:
            coordinates_m: ``(N, 3)`` coordinates in ``frame_id``, in meters.

        Returns:
            ``(N, 3)`` ``int64`` keys, ``floor((p - origin) / cell_m)`` per axis.

        Raises:
            VoxelAggregationError: If a coordinate is not finite or so far from the
                origin, relative to ``cell_m``, that its key would not be exact.
        """
        import numpy as np

        quotients = (np.asarray(coordinates_m, dtype=np.float64) - self._origin()) / self.cell_m
        floors = np.floor(quotients)
        if floors.size and not (
            np.isfinite(floors).all() and np.abs(floors).max() < _MAX_EXACT_KEY
        ):
            raise VoxelAggregationError(
                f"a coordinate is not finite or too far from the grid origin for an exact "
                f"voxel key at cell_m={self.cell_m!r}"
            )
        keys: NDArray[Any] = floors.astype(np.int64)
        return keys

    def anchors_of(self, keys: NDArray[Any]) -> NDArray[Any]:
        """Return the lower corner of every voxel, ``origin + key * cell_m``, in meters."""
        import numpy as np

        anchors: NDArray[Any] = self._origin() + np.asarray(keys, dtype=np.float64) * self.cell_m
        return anchors

    def to_record(self) -> dict[str, Any]:
        """Describe the grid exactly, for configuration and fingerprints."""
        return {
            "frame_id": str(self.frame_id),
            "origin_m": [float(value) for value in self.origin_m],
            "cell_m": self.cell_m,
            "indexing": self.indexing,
        }

    def fingerprint(self) -> str:
        """Hash the grid identity alone: frame, origin, resolution and indexing.

        This is deliberately narrower than a policy fingerprint. Two aggregations laid on
        the same grid share it even when what they compute per voxel differs, which is what
        lets a derived representation address a voxel by space instead of by run.

        Returns:
            ``"sha256:<hex>"`` of :meth:`to_record`.
        """
        payload = json.dumps(self.to_record(), sort_keys=True)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _origin(self) -> NDArray[Any]:
        import numpy as np

        origin: NDArray[Any] = np.array(
            self.origin_m,
            dtype=np.float64,
        )
        return origin


def centroid_tolerance_m(
    point_counts: NDArray[Any], cell_m: float, centroids_m: NDArray[Any]
) -> NDArray[Any]:
    """Bound, per axis, how much order or chunking can move a centroid.

    Each offset ``p - anchor`` is computed identically whatever the order, and lies
    within one voxel; only the order of the ``n`` additions changes. Two summation
    orders of ``n`` terms of magnitude ``<= cell_m`` differ by at most
    ``n * n * eps * cell_m``, so the mean differs by at most ``n * eps * cell_m``, plus
    one rounding of the final ``anchor + mean`` on each side.

    Args:
        point_counts: ``(N,)`` raw points behind each aggregate.
        cell_m: The voxel edge length.
        centroids_m: ``(N, 3)`` centroids.

    Returns:
        ``(N, 3)`` tolerances in meters.
    """
    import numpy as np

    counts = np.asarray(
        point_counts,
        dtype=np.float64,
    )[:, None]
    tolerance: NDArray[Any] = 2.0 * counts * _EPSILON * cell_m + 2.0 * _EPSILON * np.abs(
        centroids_m
    )
    return tolerance
