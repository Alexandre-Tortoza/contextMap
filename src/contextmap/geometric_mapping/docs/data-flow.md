# Data Flow: Geometric Mapping

End-to-end data transformation from canonical observations to persistent map artifact.

## Input → Output by Stage

```
SequenceArtifact (canonical)
    + SequenceSelection (frame range)
    + StateEstimationRun (trajectory + poses)
    + CalibrationSet (T_body_sensor)
         ↓
    [Assembly & Validation]  (inputs.py)
         ↓
GeometryMappingPlan
    ├── GeometryMappingSources (data artifacts, selection, trajectory, calibration)
    ├── GeometryMappingPolicy (pose lookup, motion correction, deskew)
    ├── inputs: [PreparedGeometryScan]  (validated scans with pose, frame, layout)
    └── rejections: [RejectedGeometryScan]  (scans rejected + reason)
         ↓
    [Transform to Map Frame]  (transformation.py)
         ↓
MapFrameScan
    ├── observation_id: SourceObservationId
    ├── source_coordinates_m: array (N, 3), original frame
    ├── map_coordinates_m: array (N, 3), map frame
    ├── transform_trace: [TransformStep]  (audit trail: T_map_body · T_body_sensor)
         ↓
    [Streaming Accumulation]  (accumulation.py)
         ↓
GeometryBuildSummary
    ├── geometric_map: GeometricMap
    │   ├── map_id: MapId
    │   ├── frame_id: FrameId
    │   ├── point_count: int  (final count after aggregation if applied)
    │   ├── bounds: Bounds3D
    │   ├── time_bounds: TimeBounds
    │   ├── aggregation_rule: str  (None | "scan_voxel_centroid")
    │   └── provenance: GeometricMapProvenance
    ├── scans: [ScanRecord]  (lineage: origin→geometry, bounds)
    ├── source_point_count: int  (before aggregation)
         ↓
    [Atomic Write to Disk]  (writer.py)
         ↓
GeometricMapArtifact
    ├── run_artifact/
    │   ├── manifest.json  (run, map, sources, config fingerprint, file inventory)
    │   ├── lineage.json  (sequence, selection, trajectory, frames, observation IDs)
    │   ├── config.json  (pose lookup, motion correction, aggregation, debug level)
    │   ├── environment.json  (Python, NumPy, platform versions)
    │   ├── outputs/
    │   │   ├── geometry.bin  (PackedGeometry: points in map frame, scan-indexed)
    │   │   ├── source-index.jsonl  (ScanRecord per scan)
    │   │   └── map-metadata.json  (GeometricMap serialized)
    │   ├── metrics/
    │   │   ├── input-plan.json  (selection, rejections, pose lookups, corrections)
    │   │   ├── mapping.json  (scans, points, reduction, bounds)
    │   │   └── runtime.json  (wall time, peak memory — if measured)
    │   └── __DEBUG__/  (human evidence; not contractual)
    │       ├── scans-by-time.jsonl
    │       ├── trajectory-over-map.csv
    │       ├── bounds-summary.json
    │       ├── warnings.jsonl
    │       └── [transform-traces]  (sample or full)
```

## Cross-Stage Data Contracts

| From | To | Type | What Flows |
| --- | --- | --- | --- |
| Assembly | Transform | `GeometryMappingPlan` | Scans + pose + frame + calibration + policy |
| Transform | Accumulation | `MapFrameScan` iterator | Scan ID, source XYZ, map XYZ, transform trace |
| Accumulation | Write | `GeometryBuildSummary` | Packed geometry, scan records, map metadata |
| Write | Artifact | manifest.json | Run ID, map ID, configuration fingerprint, file hashes |

## Reading Path (Reverse)

```
GeometricMapArtifact (on disk)
         ↓
    [GeometricMapArtifactReader.open(dir)]  (reader.py)
         ↓
GeometricMapArtifactManifest
    ├── Verified: run_id, map_id, sequence, selection, trajectory, calibration
    ├── Verified: all file hashes (SHA-256) and sizes match inventory
    ├── Loaded: GeometricMap  (identity, frame, bounds, time)
    ├── Loaded: ScanRecords  (lineage per scan)
    └── Ready: PackedGeometry (mmap'd, random access)
         ↓
    [GeometrySource protocol]  (ports.py)
    ├── get(ref): GeometryBlock  (coordinates at this reference)
    ├── iter_geometry(bounds): Iterator[GeometryPoint]
    └── query_bounds(): Bounds3D

## Optional: Derived Aggregation

After assembly & persistence, optionally:

```
GeometricMapArtifact (raw)
         ↓
    [Voxel Aggregation]  (voxel_aggregation/aggregate.py)
         ↓
VoxelAggregation
    ├── grid: VoxelGridSpec  (frame, origin, resolution)
    ├── aggregates: dict[voxel_key] → VoxelAggregate
    │   ├── centroid_m: Vector3
    │   ├── point_count: int
    │   ├── source_scan_ids: set (lineage)
    └── observations_per_scan: dict  (recovered)
         ↓
    [Write Derived Artifact]  (writer via voxel_aggregation/)
         ↓
VoxelAggregationArtifact  (separate, keyed to raw map identity)
```

The raw map is **never** modified; aggregation is a separate artifact built from it.

## Validation Timeline

| Stage | Checks | Consequence |
| --- | --- | --- |
| Assembly | Observation, frame, clock, layout, pose lookup, time bounds | RejectedGeometryScan or error |
| Transform | Static frame, calibration, coordinate finiteness | GeometryTransformError |
| Accumulation | Bounds consistency, invariant checks | AccumulationError |
| Write | Manifest schema, file sizes, hashes | AtomicRunDirectory fails, artifact not created |

**Key**: Invalid state is never persisted. Errors prevent artifact creation entirely.

## Identifiers & Provenance Trail

Each entity carries sufficient provenance to trace back:

```
GeometryPoint
    └─ GeometryReference(map_id, geometry_id)
        └─ MapId(<sequence>--<run_id>)
        └─ geometry_id = hash(scan_index, point_index)
            └─ ScanRecord(scan_ordinal, scan_bounds)
                └─ SourceObservationId (from ingestion)
                └─ Transform trace (T_map_body · T_body_source)
                    └─ Trajectory run (state_estimation run_id)
                    └─ Pose lookup policy + actual result
                    └─ Calibration identity (from ingestion)
```

A reader can reconstruct:
- Which observation produced each point
- Which pose (timestamp, interpolation type, confidence) was used
- Which calibration was applied
- The full transform chain and its lineage

## Data Immutability Principle

Once a `GeometricMapArtifact` is finalized:
- The run cannot be re-opened and re-written
- The manifest and all contractual files are read-only
- A second run with the same intent produces a new artifact with a new run_id
- Historical artifacts remain auditable and reproducible

Any correction requires a new run with the corrected inputs or policy.
