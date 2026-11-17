# Testing Strategy: Geometric Mapping

Test organization, pyramid, and quality gates for geometric_mapping.

## Test Pyramid

```
                         ↑
                   [E2E Tests]  (minimal representative path)
                         ↑
              [Integration Tests]  (artifact round-trip)
                         ↑
           [Architecture Tests]  (boundaries, imports, ownership)
                         ↑
             [Contract Tests]  (interchangeable implementations)
                         ↑
              [Unit Tests]  (deterministic behavior)
                        Base
```

## Test Categories

### Unit Tests

**Location**: `tests/geometric_mapping/test_*.py` (deterministic behavior in isolation)

**Scope**: Single module or function with real domain types.

| Test Type | Coverage | Example |
| --- | --- | --- |
| Assembly & validation | Observation errors, rejections, pose lookup | `test_geometric_mapping_inputs.py` (44 tests) |
| Transform & lineage | Sensor→map, trace verification, NaN/Inf checks | `test_geometric_mapping_transformation.py` |
| Deskew | Pose interpolation, temporal correctness | `test_deskew.py` (13 tests) |
| Accumulation | Streaming, indexing, spatial consistency | `test_geometric_mapping_accumulation.py` |
| Motion correction | State tracking, disposition logic | implicit in accumulation tests |
| Serialization | JSON encode/decode round-trip | implicit in artifact tests |

**Stubs/Builders**: `tests/geometric_mapping/input_builders.py`, `tests/geometric_mapping/map_builders.py` — reusable plan/map fixtures.

### Contract Tests

**Location**: `tests/geometric_mapping/test_geometric_mapping_*.py`

**Scope**: Invariants that any implementation must preserve.

| Contract | Test | Verification |
| --- | --- | --- |
| Serialization round-trip | `test_geometric_mapping_artifact.py` | encode → decode → equal |
| Layout discovery | `test_geometric_mapping_inputs.py` | point cloud → PointCloudLayout → extractable coordinates |
| Transform lineage | `test_geometric_mapping_transformation.py` | trace → recomputable frame + time |
| Accumulation consistency | `test_geometric_mapping_accumulation.py` | append-order invariance, bounds correctness |

### Integration Tests

**Location**: `tests/geometric_mapping/test_geometric_mapping_artifact.py` + `test_geometric_mapping_scaling.py`

**Scope**: Full artifact write + read cycle; representative multi-scan maps.

| Test | What It Does |
| --- | --- |
| `test_finalized_map_is_readable` | Write a small map, re-open with GeometricMapArtifactReader, verify contents |
| `test_artifact_round_trip_preserves_identity` | map → artifact → read → GeometryReference → same coordinates |
| `test_map_with_N_scans_and_X_points` | Scaling tests: small, medium, large maps verify serialization |
| `test_debug_evidence_is_written` | Debug level FULL → traces written; NONE → skipped |

### Architecture Tests

**Location**: `tests/architecture/` (boundaries, imports, ownership)

**Scope**: Import direction, no cycles, ownership preservation.

| Check | Why It Matters |
| --- | --- |
| geometric_mapping imports only ingestion + state_estimation | Upstream dependencies declared; no circular coupling |
| inputs.py does not import transformation.py | Layering preserved; inputs are data, transform is logic |
| voxel_aggregation/ does not import outside voxel_aggregation/ | Optional is isolated; can be disabled without breaking core |
| run_artifact/ is the only consumer of geometry_storage | Persistence boundary is clear |
| ports.py has no imports of internal modules | GeometrySource protocol is truly interchangeable |

### Evaluation Tests

**Location**: `tests/evaluation/test_geometric_mapping_evaluation.py` (scientific quality)

**Scope**: Map quality, numerics, reproducibility.

| Test Type | Validates |
| --- | --- |
| Numerical correctness | Transform math precision, no accumulation drift |
| Coordinate consistency | Same input → identical output (deterministic) |
| Serialization fidelity | Round-trip XYZ equals original to machine precision |
| Reference stability | `GeometryReference(map_id, geometry_id)` resolves to same point across reads |

**Not here**: Deep learning model evaluation, backend performance comparisons — those live in model-specific tests.

## Quality Gates

All tests must pass before code lands:

```bash
make check
```

### Local (Pre-Commit)

```bash
# Lint
.venv/bin/ruff check src/contextmap/geometric_mapping tests/

# Type checking
.venv/bin/mypy src/contextmap/geometric_mapping/ --strict

# Tests
.venv/bin/pytest tests/geometric_mapping tests/evaluation/test_geometric_mapping_evaluation.py -v

# Coverage
.venv/bin/pytest --cov=contextmap.geometric_mapping tests/geometric_mapping
```

### CI (Post-Merge)

- Ruff lint (E501, D102, F401, etc.)
- mypy full strict check
- pytest full suite + coverage
- Artifact smoke tests (write + read cycle)

## Test Data & Builders

### Fixtures

All tests use in-memory synthetic data (LidarObservation builders) — no real dataset.

| Module | Purpose |
| --- | --- |
| `input_builders.py` | `make_scan()`, `make_trajectory()`, `assemble_plan()` |
| `map_builders.py` | `make_calibration()`, `make_geometric_map()` |
| `lidar_builders.py` | Low-level scan/observation factories |

### Why No Real Data in Unit Tests?

- **Determinism**: Synthetic data is reproducible, deterministic, testable
- **Coverage**: Unit tests should cover edge cases (empty scans, NaN/Inf, boundary frames) — real data often misses these
- **Speed**: Synthetic data is fast; real data sequences would slow CI
- **Isolation**: Real data couples to dataset, ingestion version, etc.; tests should not

**Real data** is used for evaluation and ablation studies (separate infrastructure).

## Coverage Targets

| Category | Target |
| --- | --- |
| Core functions (inputs, transformation, accumulation) | ≥90% line coverage |
| Error paths | All explicit errors tested (rejections, transform errors, accumulation bounds) |
| Serialization | Round-trip encode/decode for all public contracts |
| Artifact I/O | Write + read cycle; manifest integrity |

Uncovered lines acceptable:
- Defensive branches that should never execute (e.g., "else: raise InternalError")
- Debug evidence paths (MapDebugLevel.FULL traces are smoke-tested, not 100% covered)
- Platform-specific code (Windows vs. Unix paths)

## Test Organization by Module

```
src/contextmap/geometric_mapping/
    contracts.py
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_inputs.py

    inputs.py
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_inputs.py

    transformation.py
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_transformation.py

    deskew.py
        ↓ tests at
    tests/geometric_mapping/test_deskew.py

    accumulation.py
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_accumulation.py

    run_artifact/
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_artifact.py

    voxel_aggregation/
        ↓ tests at
    tests/geometric_mapping/test_geometric_mapping_scaling.py  (integration)

    evaluation module
        ↓ tests at
    tests/evaluation/test_geometric_mapping_evaluation.py
```

## Red→Green→Refactor Workflow

For any new behavior:

1. **Red**: Write a test that fails (asserts new behavior doesn't exist yet)
2. **Green**: Implement minimum code to pass the test
3. **Refactor**: Improve names, structure, remove duplication (tests still pass)

**Exception**: Changes without executable behavior (docstring fixes, renames with no logic change) do not require a failing test.

## Continuous Integration

| Trigger | Tests | Result |
| --- | --- | --- |
| Pre-commit hooks | ruff, mypy, relevant unit tests | Fast feedback (seconds) |
| PR submit | Full Ruff, mypy, pytest suite | Blocks merge if failing |
| Merge to dev | Full suite + integration | Blocks dev if failing |
| Release to main | Full suite + smoke + evaluation | High confidence |

No code merges with failing tests.
