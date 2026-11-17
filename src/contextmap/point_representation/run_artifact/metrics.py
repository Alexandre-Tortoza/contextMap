"""How the figures of a Point Representation run are computed.

Counts by failure reason, the distribution of support sizes and representation norms, and
the running per-component statistics of the stored vectors. None of these is a measure of
representation quality: they describe the evidence a run holds.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from contextmap.point_representation.models import FailureReason


class ComponentStatistics:
    """Running per-component minimum, maximum and mean over the defined values."""

    def __init__(self, dimension: int) -> None:
        """Start empty statistics for vectors of ``dimension`` components."""
        self._count = [0] * dimension
        self._sum = [0.0] * dimension
        self._minimum = [math.inf] * dimension
        self._maximum = [-math.inf] * dimension

    def observe(self, values: tuple[float, ...], undefined: set[int]) -> None:
        """Fold one vector in, skipping the components listed as undefined."""
        for index, value in enumerate(values):
            if index in undefined:
                continue
            self._count[index] += 1
            self._sum[index] += value
            self._minimum[index] = min(self._minimum[index], value)
            self._maximum[index] = max(self._maximum[index], value)

    def summary(self, feature_names: tuple[str, ...]) -> list[dict[str, Any]]:
        """Per-component record: index, name, defined count, minimum, maximum and mean."""
        return [
            {
                "index": index,
                "name": feature_names[index] if feature_names else None,
                "defined_count": count,
                "minimum": self._minimum[index] if count else None,
                "maximum": self._maximum[index] if count else None,
                "mean": self._sum[index] / count if count else None,
            }
            for index, count in enumerate(self._count)
        ]


def distribution(values: list[int] | list[float]) -> dict[str, Any]:
    """Count, minimum, median, 95th percentile (nearest rank) and maximum."""
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum": ordered[-1],
    }


def failure_reason_counts(counts: Mapping[FailureReason, int]) -> dict[str, int]:
    """Failed-support counts keyed by the reason's value, ordered by that value."""
    return {
        reason.value: count for reason, count in sorted(counts.items(), key=lambda i: i[0].value)
    }
