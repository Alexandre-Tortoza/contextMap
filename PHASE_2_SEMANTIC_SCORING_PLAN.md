# Phase 2: Extract semantic_scoring Capability

**Status:** Ready to implement  
**Complexity:** Low (already fairly isolated)  
**Validation:** semantic_scoring is already separate stage; mostly code organization + artifact creation

---

## Scope

Create independent `contextmap.semantic_scoring` capability by moving scoring-specific contracts, ports, adapters, artifact storage, and runtime integration from visual_perception.

---

## 1. Contracts to Extract

Source: `contextmap.visual_perception.models`

**Move to** `contextmap.semantic_scoring.contracts`:
- `SemanticScore` dataclass
- `SemanticScoreType` enum
- `ScoreId` type alias

These are scoring-specific contracts. No other capability owns them.

---

## 2. Port Definition

Source: `contextmap.visual_perception.ports`

**Move to** `contextmap.semantic_scoring.ports`:
- `SemanticScorer` port (adapter interface)

This port is only implemented by scoring adapters. Consumers don't directly depend on it.

---

## 3. Adapters

Source: `contextmap.visual_perception.adapters/features/semantic_scoring.py`

**Subfolder:** `contextmap.semantic_scoring.adapters/`

Adapters implementing `SemanticScorer`:
- CLIP scorer
- AlphaCLIP scorer

No refactoring needed; these are already isolated.

---

## 4. Artifact

Create: `contextmap.semantic_scoring.artifacts`

**Classes:**
- `SemanticScoringRunManifest` — metadata, lineage, timestamps
- `SemanticScoringRunWriter` — writes scores.jsonl + metadata
- `SemanticScoringRunReader` — reads artifact independently

**Manifest fields:**
```python
@dataclass
class SemanticScoringRunManifest:
    run_id: SemanticScoringRunId
    run_index: int
    semantic_interpretation_run_id: str
    visual_features_run_id: str
    sequence_artifact_id: str
    source_observation_ids: list[str]
    created_at: str
    scorer_backend: BackendProvenance
    configuration_fingerprint: str
    code_version: str | None = None
```

**Output structure:**
```
<workspace>/<dataset>/<run>/semantic_scoring/
├── manifest.json
├── outputs/
│   └── scores.jsonl
└── metrics/
    └── stage-timings.jsonl
```

---

## 5. Runtime Integration

Update: `contextmap.runtime.composition`

**Changes:**
- Add semantic_scoring to runtime DAG
- Input: `SemanticInterpretationRun`, `VisualFeaturesRun`
- Output: `SemanticScoringRun`
- Backend selection: Read config `[semantic_scoring].scorer`
- Adapter initialization: Call scorer adapter factory

---

## 6. Public API

Create: `contextmap.semantic_scoring.__init__.py`

**Exports:**
```python
# Contracts
SemanticScore
SemanticScoreType
ScoreId

# Port
SemanticScorer

# Artifact
SemanticScoringRunManifest
SemanticScoringRunWriter
SemanticScoringRunReader
SemanticScoringRunId

# Adapter factories (if runtime exports these)
# semantic_scoring.adapters.clip_scorer, alph aclip_scorer, etc.
```

---

## 7. Consumer Updates

**Updated imports in:**

1. **contextmap.semantic_fusion**
   - Change: `from contextmap.visual_perception import SemanticScore`
   - To: `from contextmap.semantic_scoring import SemanticScore`
   - Impact: Reads SemanticScoringRun directly instead of PerceptionResult.semantic_scores

2. **contextmap.semantic_mapping**
   - Same import update
   - Reads from SemanticScoringRun

3. **contextmap.evaluation**
   - Import update
   - Reads SemanticScoringRun for evaluation metrics

4. **tests**
   - Update imports
   - Use SemanticScoringRunWriter for test fixtures

---

## 8. Tests

Create: `tests/semantic_scoring/`

**Test structure:**
```
tests/semantic_scoring/
├── contracts/
│   ├── test_semantic_score.py      (SemanticScore invariants)
│   └── test_score_type.py          (SemanticScoreType semantics)
├── adapters/
│   ├── test_clip_scorer.py
│   └── test_alphaclip_scorer.py
├── artifacts/
│   ├── test_manifest.py
│   ├── test_writer.py              (round-trip write/read)
│   └── test_reader.py
└── integration/
    └── test_semantic_scoring_pipeline.py
```

**Regression tests:** Ensure scores are byte-for-byte identical after extraction.

---

## 9. Config Migration

**Old config** (visual_perception):
```toml
[visual_perception.scoring]
backend = "clip"
```

**New config** (semantic_scoring):
```toml
[semantic_scoring]
scorer = "clip"
```

Update all active experiment configs.

---

## 10. Documentation

Create: `contextmap.semantic_scoring.docs/`

**Files:**
- `README.md` — Capability overview
- `architecture.md` — Ports, adapters, lineage
- `contracts.md` — SemanticScore schema
- `adapters.md` — Available scorers

---

## Implementation Order

1. ✅ Approve Phase 2 plan (this document)
2. Create `contextmap.semantic_scoring` package structure
3. Move contracts + ports + adapters
4. Implement SemanticScoringRunArtifact (writer/reader/manifest)
5. Update runtime composition
6. Update consumer imports (semantic_fusion, semantic_mapping, evaluation)
7. Update tests (both semantic_scoring tests + consumer tests)
8. Update experiment configs
9. Run `make check` — all tests must pass
10. Commit with semantic commit message

---

## Expected Changes

| Category | Files | Impact |
|----------|-------|--------|
| New files | ~15 | New capability structure |
| Deleted files | ~5 | semantic_scoring files removed from visual_perception |
| Modified files | ~20 | Imports, runtime, tests, config |
| **Total** | **~40** | One semantic commit |

---

## Validation

Before Phase 3, confirm:
- ✓ SemanticScoringRunArtifact reads/writes correctly
- ✓ Round-trip test: write → read → byte-identical
- ✓ All scorer adapters work with new structure
- ✓ Consumer (semantic_fusion) can read and use SemanticScore
- ✓ Regression: semantic scores identical before/after extraction
- ✓ Tests pass (unit + integration)
- ✓ `make check` passes

---

## Success Criteria

- semantic_scoring is independent, reusable capability
- PerceptionResult no longer references SemanticScore
- Consumers read from SemanticScoringRun, not aggregated result
- No semantic_scoring code remains in visual_perception
- Runtime DAG has explicit semantic_scoring stage
- Documentation reflects new ownership
- All tests pass

---

**Ready to implement Phase 2.**
