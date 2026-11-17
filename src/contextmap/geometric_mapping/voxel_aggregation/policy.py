"""Aggregation policy: identity, semantics, and configuration.

Defines the policy that produces voxel aggregates: the grid identity, representative,
statistics, and lineage granularity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .grid import VoxelGridSpec

INTER_SCAN_VOXEL_POLICY_ID = "inter-scan-voxel-centroid"
"""Identity of the aggregation rule, independent of its parameters."""

INTER_SCAN_VOXEL_POLICY_VERSION = "0.1.0"
"""Version of what the rule computes; a change of semantics requires a new version."""

_REPRESENTATIVE = "centroid"
_STATISTICS = (
    "point_count",
    "scan_count",
    "observation_count",
    "offset_sum_m",
    "minimum_m",
    "maximum_m",
    "first_observed_ns",
    "last_observed_ns",
)
_LINEAGE = "scan-ordinal-and-point-count"


@dataclass(frozen=True, kw_only=True)
class InterScanVoxelPolicy:
    """Aggregate the raw points of every scan that fall in one voxel into their centroid.

    Only the grid varies between executions of this policy version; the representative
    (the centroid), the statistics and the lineage granularity are fixed by
    :data:`INTER_SCAN_VOXEL_POLICY_VERSION`.

    Attributes:
        grid: The voxel grid.
    """

    grid: VoxelGridSpec

    @property
    def rule(self) -> str:
        """The rule a derived map declares, exact in the voxel size."""
        return f"{INTER_SCAN_VOXEL_POLICY_ID}-{self.grid.cell_m!r}m"

    def to_record(self) -> dict[str, Any]:
        """Describe everything that changes the derived representation."""
        return {
            "policy_id": INTER_SCAN_VOXEL_POLICY_ID,
            "policy_version": INTER_SCAN_VOXEL_POLICY_VERSION,
            "grid": self.grid.to_record(),
            "representative": _REPRESENTATIVE,
            "statistics": list(_STATISTICS),
            "lineage": _LINEAGE,
            "rule": self.rule,
        }

    def fingerprint(self) -> str:
        """Hash the whole policy: the grid *and* what the aggregation computes per voxel.

        It changes when the grid changes and also when this policy version changes, so it
        identifies the derivation, not the grid. A consumer that needs the identity of the
        grid alone uses :meth:`~contextmap.geometric_mapping.VoxelGridSpec.fingerprint`.

        Returns:
            ``"sha256:<hex>"`` of :meth:`to_record`.
        """
        payload = json.dumps(self.to_record(), sort_keys=True)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
