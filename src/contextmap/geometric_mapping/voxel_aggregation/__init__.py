"""Inter-scan voxel aggregation: a derived, provenance-preserving view of a raw map.

A raw :class:`~contextmap.geometric_mapping.GeometricMap` persists every accepted
measurement, so a platform that revisits (or stands still in) one place keeps
accumulating near-duplicate points. This module derives, from such a raw map, one
aggregate per occupied voxel of an explicit :class:`VoxelGridSpec`: the centroid of
the raw points in the voxel, the sufficient statistics to update it incrementally, and
the compact lineage that recovers exactly which raw points it summarizes.

It is **not** :class:`~contextmap.geometric_mapping.ScanVoxelPolicy`. That policy runs
inside the mapping, merges only points of the *same* scan and replaces them in the
canonical map. This one merges points of *different* scans, runs offline over a
finished raw map, never modifies it, and produces a separate derived representation.

The aggregation is geometry only. It never merges labels, claims, embeddings,
semantic hypotheses or instance identity; a derived aggregate is evidence about
*where* raw measurements were, not a belief about *what* is there.

Counts keep distinct meanings (see :class:`VoxelAggregate`): ``point_count`` counts
raw measurements, ``scan_count`` accumulated scans (source-index entries) and
``observation_count`` physical source observations. Within one raw map a scan is one
observation, so the last two coincide there; they are kept apart so the contract does
not assume that bijection.

Order and chunking of the input never change keys, counts, bounds, times or lineage.
They can change the floating-point sums by rounding only, within the bound documented
in :func:`centroid_tolerance_m`. See
``src/contextmap/geometric_mapping/docs/inter-scan-aggregation.md``.
"""

from .aggregate import (
    AggregatedGeometry,
    InterScanVoxelAggregator,
    VoxelAggregate,
    VoxelAggregation,
    VoxelContribution,
    aggregate_geometry,
)
from .artifact import (
    AGGREGATE_DTYPE_FIELDS,
    AGGREGATE_RECORD,
    CONTRIBUTION_DTYPE_FIELDS,
    CONTRIBUTION_RECORD,
    SCHEMA_VERSION,
    VoxelAggregationArtifactReader,
    VoxelAggregationArtifactWriter,
    VoxelAggregationManifest,
    VoxelAggregationRunId,
)
from .grid import VoxelAggregationError, VoxelGridSpec, centroid_tolerance_m
from .policy import (
    INTER_SCAN_VOXEL_POLICY_ID,
    INTER_SCAN_VOXEL_POLICY_VERSION,
    InterScanVoxelPolicy,
)

__all__ = [
    "AGGREGATE_DTYPE_FIELDS",
    "AGGREGATE_RECORD",
    "CONTRIBUTION_DTYPE_FIELDS",
    "CONTRIBUTION_RECORD",
    "INTER_SCAN_VOXEL_POLICY_ID",
    "INTER_SCAN_VOXEL_POLICY_VERSION",
    "SCHEMA_VERSION",
    "AggregatedGeometry",
    "InterScanVoxelAggregator",
    "InterScanVoxelPolicy",
    "VoxelAggregate",
    "VoxelAggregation",
    "VoxelAggregationArtifactReader",
    "VoxelAggregationArtifactWriter",
    "VoxelAggregationError",
    "VoxelAggregationManifest",
    "VoxelAggregationRunId",
    "VoxelContribution",
    "VoxelGridSpec",
    "aggregate_geometry",
    "centroid_tolerance_m",
]
