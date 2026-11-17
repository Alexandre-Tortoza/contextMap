"""A forma de um suporte de pontos: centroide, conectividade e eixos principais.

Este módulo é o **mecanismo** por trás do resumo espacial de uma entidade, e muda por custo, não
por ciência: a grade de conectividade, o union-find e o solver de Jacobi existem para contar
componentes e derivar eixos sem NumPy, e podem ser otimizados sem que nenhuma decisão científica
mude. Os limiares que decidem *quando* aplicar cada coisa pertencem a `geometry_summary.py`.

Por isso ele é folha: recebe coordenadas e números, devolve coordenadas e números, e não conhece
nenhum contrato, política ou provenance desta capability. Quem embala o resultado em
`EntityOrientation` ou `SupportStatistics` é a camada de cima.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import product
from typing import NamedTuple

from contextmap.geometric_mapping import Bounds3D
from contextmap.shared import Vector3

SUMMARY_TOLERANCE_M = 1e-9
"""Absolute tolerance, in metres, for comparing a derived summary with a recomputed one."""


class PrincipalAxes(NamedTuple):
    """The principal axes of a support and the variance along each, largest first."""

    axes: tuple[Vector3, Vector3, Vector3]
    variances_m2: Vector3


DEFAULT_MAX_CONNECTIVITY_POINTS = 10_000
"""Largest support whose connected components are counted unless the policy says otherwise.

The pairwise comparison inside each 27-cell neighborhood is roughly quadratic in the local density.
One synthetic run in pure Python (random points in a 2 x 2 x 1 m box, 0.3 m radius) took about 4.5 s
at 10,000 points and 42 s at 30,000, and no real support has been measured: the limit keeps one
entity in the order of seconds. It is a cost guard, not a scientific threshold, so a profile that
accepts the cost raises it explicitly, and the value enters the configuration fingerprint.
"""

ORIENTATION_METHOD = "pca-covariance-jacobi-v1"
"""Versioned identity of the orientation derivation implemented here."""


def vector_of(values: Iterable[float]) -> Vector3:
    """Pack three coordinate values into a :data:`Vector3`."""
    x, y, z = values
    return (x, y, z)


def centroid_of(coordinates: Sequence[Vector3], bounds: Bounds3D) -> Vector3:
    """Exact arithmetic mean of the support, clamped into ``bounds``."""
    count = len(coordinates)
    mean = tuple(math.fsum(point[axis] for point in coordinates) / count for axis in range(3))
    # A média está matematicamente dentro da caixa; o clamp só corrige o erro de arredondamento.
    x, y, z = (
        min(max(value, low), high)
        for value, low, high in zip(mean, bounds.minimum_m, bounds.maximum_m, strict=True)
    )
    return (x, y, z)


def component_sizes(coordinates: Sequence[Vector3], radius_m: float) -> list[int]:
    """Size of each connected component when points closer than ``radius_m`` are linked.

    Points are bucketed in a grid of cell size ``radius_m``, so only the 27 neighboring cells
    of a point can hold a neighbor within the radius.
    """
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (x, y, z) in enumerate(coordinates):
        cells[
            (math.floor(x / radius_m), math.floor(y / radius_m), math.floor(z / radius_m))
        ].append(index)
    parent = list(range(len(coordinates)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    radius_squared = radius_m * radius_m
    for (cx, cy, cz), members in cells.items():
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            for other in cells.get((cx + dx, cy + dy, cz + dz), ()):
                for member in members:
                    if (
                        other > member
                        and _squared_distance(coordinates[member], coordinates[other])
                        <= radius_squared
                    ):
                        parent[find(other)] = find(member)
    sizes: dict[int, int] = defaultdict(int)
    for index in range(len(coordinates)):
        sizes[find(index)] += 1
    return sorted(sizes.values(), reverse=True)


def principal_axes(
    coordinates: Sequence[Vector3],
    centroid: Vector3,
    *,
    min_points: int,
    min_variance_ratio: float,
) -> tuple[PrincipalAxes | None, str | None]:
    """Derive the principal axes of a support, or say why they are not well defined.

    Args:
        coordinates: The support, in metres.
        centroid: The centroid of the support, in metres.
        min_points: Fewest points an orientation needs.
        min_variance_ratio: Least ratio between consecutive variances for the axes to count as
            distinct.

    Returns:
        The axes and their variances, or ``None`` and the reason they are not well defined.
    """
    count = len(coordinates)
    if count < min_points:
        return None, f"{count} points, fewer than the {min_points} an orientation needs"
    covariance = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(row, 3):
            value = (
                math.fsum(
                    (point[row] - centroid[row]) * (point[column] - centroid[column])
                    for point in coordinates
                )
                / count
            )
            covariance[row][column] = covariance[column][row] = value
    variances, vectors = _symmetric_eigen(covariance)
    order = sorted(range(3), key=lambda index: -variances[index])
    ordered_variances = [max(variances[index], 0.0) for index in order]
    for larger, smaller in ((0, 1), (1, 2)):
        if not _distinct(ordered_variances[larger], ordered_variances[smaller], min_variance_ratio):
            return None, (
                f"variances {tuple(ordered_variances)!r} are not distinct enough: the ratio "
                f"between consecutive variances must be at least {min_variance_ratio}"
            )
    first = _canonical_sign(vectors[order[0]])
    second = _canonical_sign(vectors[order[1]])
    third = cross(first, second)
    return (
        PrincipalAxes(axes=(first, second, third), variances_m2=vector_of(ordered_variances)),
        None,
    )


def _distinct(larger: float, smaller: float, min_variance_ratio: float) -> bool:
    """Whether two consecutive variances are far enough apart to define an axis."""
    if larger <= 0.0:
        return False
    return smaller == 0.0 or larger / smaller >= min_variance_ratio


def _canonical_sign(axis: Vector3) -> Vector3:
    """Flip an axis so its largest component is positive, so equivalent axes encode identically."""
    dominant = max(axis, key=abs)
    return axis if dominant >= 0.0 else (-axis[0], -axis[1], -axis[2])


def _symmetric_eigen(matrix: list[list[float]]) -> tuple[list[float], list[Vector3]]:
    """Eigenvalues and unit eigenvectors of a symmetric 3x3 matrix, by cyclic Jacobi rotations."""
    a = [row[:] for row in matrix]
    v = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    scale = sum(a[i][i] ** 2 for i in range(3)) or 1.0
    for _ in range(64):
        if a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2 <= 1e-32 * scale:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if a[p][q] == 0.0:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            other = 3 - p - q
            app, aqq, apq = a[p][p], a[q][q], a[p][q]
            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = a[q][p] = 0.0
            aop, aoq = a[other][p], a[other][q]
            a[other][p] = a[p][other] = c * aop - s * aoq
            a[other][q] = a[q][other] = s * aop + c * aoq
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    eigenvalues = [a[i][i] for i in range(3)]
    eigenvectors = [(v[0][i], v[1][i], v[2][i]) for i in range(3)]
    return eigenvalues, eigenvectors


def _squared_distance(left: Vector3, right: Vector3) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def dot(left: Vector3, right: Vector3) -> float:
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def cross(left: Vector3, right: Vector3) -> Vector3:
    """Cross product of two vectors, right-handed."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Vector3) -> float:
    """Euclidean length of a vector."""
    return math.sqrt(dot(vector, vector))
