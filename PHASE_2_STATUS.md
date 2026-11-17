# Phase 2: Semantic Scoring Extraction — Current Status

**As of:** 2026-10-01  
**Target:** Complete atomic semantic_scoring extraction with zero legacy coexistence

---

## ✅ Completed (Committed)

### Commit 0560263: Foundation
- `contextmap.semantic_scoring` package created
- `contracts.py`: SemanticScore, SemanticScoreType, ScoreId
- `ports.py`: SemanticScorer protocol
- `__init__.py`: Public API exports

### Commit 6088b9b: Core Implementation
- `artifacts.py`: SemanticScoringRunManifest, Writer, Reader
- Artifact structure: `manifest.json`, `outputs/scores.jsonl`, `metrics/`
- Complete lineage tracking (SemanticInterpretationRunId, VisualFeaturesRunId)

### Commit 37b1c66: API Cleanup
- Removed SemanticScore, SemanticScoreType from visual_perception.__init__.py
- Removed SemanticScorer import from ports
- Removed all SemanticScoring* exports
- Visual perception no longer exposes semantic_scoring symbols

---

## 🔄 Remaining Critical Path

### 1. Extract Adapters (~150 lines)
**Status:** Not started  
**Task:** Move scorer implementations to semantic_scoring/adapters/

Files to extract:
- CLIP scorer implementation
- AlphaCLIP scorer implementation
- Adapter factory/registration

Source: `visual_perception/adapters/features/semantic_scoring.py`  
Destination: `semantic_scoring/adapters/clip.py`, `semantic_scoring/adapters/alphaclip.py`

Validation:
- Adapters implement SemanticScorer protocol
- No dependencies back to visual_perception (except contracts)
- Adapter tests pass

---

### 2. Runtime Composition Updates (~200 lines)
**Status:** Not started  
**Task:** Integrate semantic_scoring as a first-level capability stage

Changes in `contextmap.runtime`:
- Add `semantic_scoring` to composition DAG
- Input stage: Read `SemanticInterpretationRun`, `VisualFeaturesRun`
- Output: Produce `SemanticScoringRun` artifact
- Backend selection: Read config `[semantic_scoring].scorer` (CLIP, AlphaCLIP)
- Adapter factory: Instantiate selected scorer
- Stage identity: Change from `visual_perception::semantic_scoring` → `semantic_scoring::score`
- Reuse key calculation: New fingerprint for semantic_scoring runs

Validation:
- Runtime tests pass
- DAG is acyclic
- semantic_scoring !-> runtime (no circular dependency)

---

### 3. Migrate All Active Consumers (~300 lines across 5 files)
**Status:** Not started  
**Files to update:**

#### semantic_fusion
- Change: `from contextmap.visual_perception import SemanticScore`
- To: `from contextmap.semantic_scoring import SemanticScore`
- Impact: Read from SemanticScoringRun instead of PerceptionResult.semantic_scores
- Test coverage: fusion integration tests

#### semantic_mapping
- Same import migration
- Verify it reads SemanticScoringRun correctly
- Test coverage: mapping integration tests

#### evaluation
- Update imports
- Verify evaluation metrics read from SemanticScoringRun
- Test coverage: evaluation tests

#### runtime composition
- Already covered above

#### tests (unit + integration)
- Update all imports
- Create fixtures using SemanticScoringRunWriter
- Remove references to old PerceptionResult.semantic_scores

---

### 4. Remove Old Semantic Scoring Code from visual_perception (~200 lines)
**Status:** Not started  
**Task:** Complete the extraction by removing all old scoring implementations

Files to remove/clean:
- `visual_perception/models.py`: Remove old SemanticScore, SemanticScoreType definitions
- `visual_perception/ports.py`: Remove SemanticScorer definition
- `visual_perception/adapters/features/semantic_scoring.py`: Delete entire file
- `visual_perception/artifacts/scoring_artifacts.py`: Delete (if exists)
- Any PerceptionResult fields referencing semantic_scores: Set to empty list or remove if no other consumer depends

Validation:
- No `grep "SemanticScore" src/contextmap/visual_perception/` matches
- No `grep "SemanticScorer" src/contextmap/visual_perception/` matches
- PerceptionResult contract remains valid (but no longer aggregates scores)

---

### 5. Tests (Unit + Integration) (~600 lines)
**Status:** Not started  
**Task:** Comprehensive test coverage for semantic_scoring

Create `tests/semantic_scoring/`:
```
tests/semantic_scoring/
├── contracts/
│   ├── test_semantic_score.py
│   │   - ScoreId uniqueness
│   │   - SemanticScore validation (__post_init__)
│   │   - Cosine similarity bounds [-1, 1]
│   │   - Calibrated probability bounds [0, 1]
│   │   - Provenance validation (capability = "semantic_scorer")
│   │   - Embedding space identity preservation
│   │   - Absence of score != zero
│   └── test_score_type.py
│       - SemanticScoreType.COSINE_SIMILARITY semantics
├── adapters/
│   ├── test_clip_adapter.py
│   │   - Scorer interface contract
│   │   - Feature/claim compatibility checking
│   │   - Score production with correct backend provenance
│   │   - Release/cleanup
│   └── test_alphaclip_adapter.py
│       - Same as CLIP
├── artifacts/
│   ├── test_manifest.py
│   │   - Manifest creation and serialization
│   │   - Lineage tracking (interpretation_run_id, features_run_id)
│   └── test_writer_reader.py
│       - Write manifest → read manifest (exact match)
│       - Write scores → read scores (JSONL roundtrip)
│       - Byte-identical before/after extraction (regression)
└── integration/
    ├── test_semantic_scoring_pipeline.py
    │   - End-to-end: claim + feature → SemanticScore
    │   - Score lineage chain
    │   - Runtime stage execution
    └── test_consumer_integration.py
        - semantic_fusion reads SemanticScoringRun
        - semantic_mapping reads SemanticScoringRun
        - evaluation metrics from SemanticScoringRun
```

Validation:
- All tests pass
- Coverage >95% for semantic_scoring core
- Architecture tests pass (dependencies, imports)

---

### 6. Documentation (~400 lines)
**Status:** Not started  
**Task:** Create semantic_scoring capability documentation

Create `src/contextmap/semantic_scoring/docs/`:
```
README.md
    - Capability purpose and scope
    - Non-responsibilities (what it does NOT do)
    - Inputs: SemanticClaim, VisualFeature (from upstream)
    - Outputs: SemanticScore, SemanticScoringRun
    - Key invariants
    - Public API entry point

architecture.md
    - Port definition (SemanticScorer)
    - Adapters (CLIP, AlphaCLIP)
    - Artifact ownership and lineage
    - Runtime composition
    - Score semantics preservation
    - Embedding space identity and compatibility

contracts.md
    - SemanticScore dataclass specification
    - SemanticScoreType enum values and semantics
    - Validation rules
    - Provenance requirements
    - Field semantics (calibrated_probability is optional, not derived)

code-map.md
    - contracts.py → SemanticScore definition + validation
    - ports.py → SemanticScorer interface
    - artifacts.py → Persistence (manifest, writer, reader)
    - adapters/ → Backend implementations
    - What does NOT belong: model SDKs, TensorFlow, PyTorch direct imports
```

Validation:
- Docs explicitly state responsibilities and non-responsibilities
- Score semantics clearly explained (not probability)
- Embedding space identity constraints documented

---

### 7. Repository-Wide Verification (~50 lines)
**Status:** Not started  
**Task:** Ensure zero active production code remains using old ownership

Searches to perform:
```bash
# Should find ZERO production code matches
grep -r "from contextmap.visual_perception import SemanticScore" src/ --exclude-dir=semantic_scoring
grep -r "from contextmap.visual_perception import SemanticScorer" src/ --exclude-dir=semantic_scoring
grep -r "PerceptionResult.semantic_scores" src/ --include="*.py"
grep -r "visual_perception.*SemanticScoring" src/ --include="*.py"

# Should find code only in tests/ or documentation
grep -r "visual_perception.*SemanticScore" tests/ --include="*.py" | wc -l
```

Expected result:
- Zero matches in `src/` (except semantic_scoring, visual_perception models/ports, runtime)
- All matches in `tests/` are updated to use semantic_scoring imports
- All matches in docs/ are accurate

---

### 8. Full Test Suite + make check
**Status:** Not started  
**Task:** Validate complete integration

Run:
```bash
# Focused tests
pytest tests/semantic_scoring/ -v
pytest tests/semantic_fusion/test_consumer_integration.py -v
pytest tests/semantic_mapping/test_consumer_integration.py -v
pytest tests/runtime/test_composition.py -v
pytest tests/artifacts/test_endpoint_reading.py -v

# Full suite
pytest tests/ -v --cov=src/contextmap/semantic_scoring

# Code quality
make check
```

Expected: All pass, no warnings

---

## Timeline Estimate

| Task | Est. Effort | Risk |
|------|-------------|------|
| 1. Extract adapters | 1-2 hours | Low (well-isolated) |
| 2. Runtime updates | 2-3 hours | Medium (DAG changes) |
| 3. Consumer migration | 2-3 hours | Medium (5 locations) |
| 4. Remove old code | 1 hour | Low (cleanup) |
| 5. Tests | 4-6 hours | Medium (coverage) |
| 6. Documentation | 2-3 hours | Low (documentation) |
| 7. Verification | 1 hour | Low (search/validate) |
| 8. Test suite + check | 1-2 hours | Low (existing harness) |
| **Total** | **14-20 hours** | **Medium overall** |

---

## Phase 2 Completion Criteria

Before returning a **Phase 2 Report**, verify:

- [ ] semantic_scoring is independent, importable capability
- [ ] PerceptionResult no longer aggregates semantic scores
- [ ] All active consumers migrated to read SemanticScoringRun
- [ ] Zero production code remains under old visual_perception ownership
- [ ] Runtime DAG explicit with semantic_scoring stage
- [ ] Artifact manifest tracks lineage (interpretation + features runs)
- [ ] Tests pass (unit, contract, integration, architecture, E2E)
- [ ] `make check` passes
- [ ] Documentation complete and accurate
- [ ] Repository-wide ownership verification complete

---

## Phase 2 Report Format (at completion)

1. Final package tree
2. Files moved/created/deleted (count + list)
3. Contracts and ownership
4. Artifact structure and lineage
5. Runtime DAG changes
6. Config migration (old → new keys)
7. Reuse key changes
8. Consumers migrated (all 5)
9. Old code removed (complete list)
10. Test results (unit, integration, E2E, make check)
11. Documentation index
12. Repository-wide ownership verification results
13. Final commit SHA

---

## Next Steps

Proceed with items 1-4 in order:
1. Extract adapters (CLIP, AlphaCLIP)
2. Update runtime composition
3. Migrate consumers
4. Remove old code

Then bundle into single "semantic_scoring extraction complete" commit.

After that, proceed with tests, docs, verification, and final report.
