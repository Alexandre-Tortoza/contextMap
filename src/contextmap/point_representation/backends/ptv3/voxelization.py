"""Pure-NumPy mapping of one prepared support to the backbone's sparse input and back.

Sparse convolution needs one entry per occupied grid cell, so :func:`voxelize` merges the points of
a support that share a cell into their centroid and keeps the point-to-voxel mapping;
:func:`pool_voxel_features` uses that mapping to reduce per-voxel features to the single vector
that represents the support. Both are deterministic and independent of the order of the points.

Nothing here touches torch, a device or a checkpoint. NumPy is imported inside the functions that
use it, and its typing symbols only under ``TYPE_CHECKING``, so importing this module loads no
array library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

XYZ_CHANNELS = 3
"""Spatial channels of one point; the backbone input beyond these is zero padding."""


@dataclass(frozen=True, eq=False)
class Voxelization:
    """A support reduced to one entry per occupied grid cell.

    Attributes:
        grid_coord: Non-negative integer cell of each voxel, anchored at the support's minimum
            corner, shape ``(V, 3)``, sorted lexicographically.
        centroid_m: Mean of the points inside each voxel, in the input units, shape ``(V, 3)``.
        inverse: Voxel index of each input point, shape ``(N,)``.
        counts: Number of input points in each voxel, shape ``(V,)``.
    """

    grid_coord: NDArray[Any]
    centroid_m: NDArray[Any]
    inverse: NDArray[Any]
    counts: NDArray[Any]


def voxelize(coordinates_m: Sequence[Vector3], grid_size_m: float) -> Voxelization:
    """Merge the points of one support that fall in the same grid cell.

    Sparse convolution needs unique voxel indices, so points sharing a cell are represented by
    their centroid; the mapping back to the input points is kept to pool the features afterwards.
    The result does not depend on the order of the points.

    Args:
        coordinates_m: One ``(x, y, z)`` per supporting element.
        grid_size_m: Cell edge, in the units of ``coordinates_m``.

    Returns:
        The voxels, their centroids and the point-to-voxel mapping.

    Raises:
        ValueError: If there are no points, the shape is not ``(N, 3)`` or a value is not finite.
    """
    import numpy as np

    points = np.asarray(coordinates_m, dtype=np.float64)
    if points.size == 0:
        raise ValueError("a support needs at least one point")
    if points.ndim != 2 or points.shape[1] != XYZ_CHANNELS:
        raise ValueError(f"coordinates must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    cells = np.floor((points - points.min(axis=0)) / grid_size_m).astype(np.int64)
    grid_coord, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    centroid_m = np.zeros((len(grid_coord), XYZ_CHANNELS), dtype=np.float64)
    np.add.at(centroid_m, inverse, points)
    centroid_m /= counts[:, None]
    return Voxelization(
        grid_coord=grid_coord, centroid_m=centroid_m, inverse=inverse, counts=counts
    )


def pool_voxel_features(
    voxel_features: NDArray[Any], voxels: Voxelization, *, center_index: int, pooling: str
) -> NDArray[Any]:
    """Reduce per-voxel features to the one vector that represents the support.

    Args:
        voxel_features: One feature row per voxel, shape ``(V, C)``.
        voxels: The voxelization the features were computed on.
        center_index: Position of the support's center element among the input points.
        pooling: ``"center"`` returns the feature of the voxel holding the center element;
            ``"mean"`` averages over the input points (a voxel counts once per point it holds).

    Returns:
        The pooled vector, shape ``(C,)``.

    Raises:
        ValueError: If ``pooling`` is unknown, ``center_index`` is outside the support or the
            features do not cover every voxel.
    """
    if pooling not in ("center", "mean"):
        raise ValueError(f"pooling must be 'center' or 'mean', got {pooling!r}")
    if not 0 <= center_index < len(voxels.inverse):
        raise ValueError(
            f"center_index {center_index} is outside a support of {len(voxels.inverse)} points"
        )
    if voxel_features.shape[0] != len(voxels.counts):
        raise ValueError(
            f"got {voxel_features.shape[0]} feature rows for {len(voxels.counts)} voxels"
        )
    pooled: NDArray[Any]
    if pooling == "center":
        pooled = voxel_features[voxels.inverse[center_index]]
    else:
        pooled = (voxel_features * voxels.counts[:, None]).sum(axis=0) / voxels.counts.sum()
    return pooled
