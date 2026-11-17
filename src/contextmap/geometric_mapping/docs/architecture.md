# Architecture: geometric_mapping

Ownership, responsibilities, and dependency direction after refactoring.

## Ownership & Responsibilities (Refactored)

### Core Responsibilities
- **geometric_mapping** owns persistent 3D geometry in global map frame
- **models.py** owns canonical contracts (GeometryPoint, GeometricMap, Bounds3D, etc.)
- **inputs.py** owns input assembly with explicit validation pipeline (observation → layout → pose → transform)
- **transformation.py** owns sensor→map frame transformation with traced verification and coordinate validity checks
- **accumulation.py** owns scan streaming and map persistence; map structure and indexing
- **geometry_storage.py** owns packed binary format and random access patterns
- **serialization.py** owns JSON encoding/decoding (format adapter, no ROS/SDK dependencies)
- **ports.py** owns read boundary (GeometrySource protocol) for cross-capability access
- **deskew.py** owns pose-interpolation motion correction implementation
- **motion_correction.py** owns explicit deskew state, policy, and evidence records
- **bulk_geometry.py** owns vectorized block-based geometry reads

### Optional: Voxel Aggregation Module
- **voxel_aggregation/** owns inter-scan voxel aggregation (derived, offline, optional)
  - Not canonical, not required by any canonical stage
  - Provides on-demand spatial querying for evaluation and optimization
  - Preserves lineage to raw map; separate artifact

### Artifact Module (Refactored)
- **run_artifact/manifest.py** owns schema definition, run/map identifiers, types, constants, exceptions, and pure manifest decoding
- **run_artifact/reader.py** owns opening and reading persisted maps; integrity verification; deserialization
- **run_artifact/writer.py** owns writing orchestration, encoding, lineage recording, atomic writes
- **run_artifact/__init__.py** owns public API re-exports for consumers

## Dependency Graph

```text
models.py / contracts.py
    ↓
inputs.py (input assembly + validation)
    ↓
transformation.py (sensor→map transform + verify)
    ↓
accumulation.py + geometry_storage.py
    ↓
run_artifact/ (persist and read)
    ↓
voxel_aggregation/ (optional derived representation)
```

`ports.py`, `serialization.py` and `bulk_geometry.py` support these stages without redefining ownership. Dependency direction should remain acyclic and explicit.

## Cross-Capability Boundaries

### Consumers (via GeometrySource)
- sensor_association: finds visual features at geometry locations
- point_representation: decorates geometry with multi-sensor evidence
- semantic_fusion: attaches semantic evidence to geometry
- semantic_mapping: builds entity geometry from aggregated evidence
- evaluation: validates map quality, numerics, reproducibility
- (planned) entity_resolution, spatial_relations

Consumers depend on the public facade `contextmap.geometric_mapping` and its stable cross-capability contracts, not on internal file layout.

## Validation Strategy

Validation is explicit, early, and categorized:

| Stage | Checks | Location | Consequence |
| --- | --- | --- | --- |
| Observation | type, frame, clock, payload | inputs.py | rejection or error |
| Layout | shape, point count, dtype | inputs.py | rejection or error |
| Pose Lookup | resolution, time bounds | inputs.py | rejection or error |
| Transform Chain | frame connectivity, calibration | inputs.py / transformation.py | rejection or error |
| Coordinate Validity | NaN/Inf checks before persistence | transformation.py | error or explicit drop according to contract |
| Persistence | bounds consistency, scan count, manifest inventory | accumulation.py / run_artifact | explicit failure; no partial final artifact |

Key principle: invalid state is not silently persisted. Errors are surfaced at the closest owner.

## Public API Surface

The module exports via `contextmap.geometric_mapping.__init__.py`:

- **Stable spatial contracts**: GeometryPoint, GeometryReference, GeometricMap, GeometrySource, TransformLineage, MotionCorrectionRecord, etc.
- **Planning**: GeometryMappingPlan, GeometryMappingSources, GeometryMappingPolicy, PreparedGeometryScan, RejectedGeometryScan, build_geometry_mapping_plan, build_geometry_mapping_plan_from_artifacts
- **Transformation**: MapFrameScan, transform_scan_to_map_frame, transform_scans_to_map_frame, verify_transform_trace
- **Accumulation**: GeometryBuildSummary, MapAccumulator, accumulate_plan
- **Artifact I/O**: GeometricMapArtifactWriter, GeometricMapArtifactReader, GeometricMapArtifactManifest
- **Optional Derived**: VoxelGridSpec, InterScanVoxelAggregator, aggregate_geometry, VoxelAggregation
- **Ports**: GeometrySource

Consumers should use the public facade rather than importing implementation modules directly.

## Refactoring Changes

### What Changed
1. `voxel_aggregation` is modeled as an optional derived representation, not as a replaceable backend.
2. `run_artifact` was split into focused manifest, reader, writer and facade modules.
3. The validation pipeline was made explicit and early in inputs/transformation.
4. NumPy imports are explicit at module scope.
5. Planning now separates source lineage from processing decisions through `GeometryMappingSources`, `GeometryMappingPolicy` and `GeometryMappingPlan`.
6. Public symbols for planning, transformation and build summary were intentionally renamed. The v0.x codebase does not keep compatibility shims for the removed names.

### Public API migration

| Removed/old name | Current name |
| --- | --- |
| `GeometryInputPlan` | `GeometryMappingPlan` |
| `GeometryInput` | `PreparedGeometryScan` |
| `GeometryInputRejection` | `RejectedGeometryScan` |
| `assemble_geometry_inputs()` | `build_geometry_mapping_plan()` |
| `assemble_geometry_inputs_from_artifacts()` | `build_geometry_mapping_plan_from_artifacts()` |
| `TransformedScan` | `MapFrameScan` |
| `transform_scan()` | `transform_scan_to_map_frame()` |
| `transform_scans()` | `transform_scans_to_map_frame()` |
| `AccumulatedMap` | `GeometryBuildSummary` |

### What Did Not Change
- Cross-capability spatial contracts such as `GeometrySource`, `GeometryPoint`, `GeometryReference` and `GeometricMap` remain available through `contextmap.geometric_mapping`.
- The raw geometric map remains the authoritative spatial representation; voxel aggregation remains derived and optional.
- Artifact semantics remain centered on immutable, auditable output with lineage, configuration and integrity checks.
- Tests continue to validate behavior and contracts rather than relying on internal file layout.

## Testing & Quality

- **Unit tests** verify behavior of each module in isolation.
- **Contract tests** verify input→output invariants.
- **Integration tests** verify artifact read/write and cross-capability boundaries.
- **Architecture tests** verify ownership and import direction.
- **Evaluation** verifies map quality, numerics and reproducibility separately from the core.

Avoid absolute architectural claims that are not enforced by executable checks. Required boundaries should be encoded in tests where practical.

## Documentation References

- [`code-map.md`](code-map.md) - file ownership, concepts and non-responsibilities
- [`data-flow.md`](data-flow.md) - end-to-end data flow
- [`testing.md`](testing.md) - test strategy and quality gates
- [`contracts.md`](contracts.md) - detailed contract definitions and invariants
- [`motion-correction.md`](motion-correction.md) - deskew state and policy
- [`inputs.md`](inputs.md) - input assembly and validation
- [`transformation.md`](transformation.md) - transform chain and lineage
- [`accumulation.md`](accumulation.md) - streaming and indexing
- [`spatial-access.md`](spatial-access.md) - GeometrySource and block access
- [`artifact.md`](artifact.md) - artifact layout and I/O
- [`inter-scan-aggregation.md`](inter-scan-aggregation.md) - optional derived representation
