# Phase 0: Visual Perception Decomposition Audit

**Date:** 2026-10-01  
**Status:** In Progress  
**Baseline Commit:** 8552d73 (refactor: reorganize file structure into semantic subdomains)

---

## 1. Current Capability Map

### Code Organization

```
src/contextmap/visual_perception/
├── Core contracts/models (9 files)
│   ├── models.py                        (854 lines) — fundamental contracts
│   ├── region_models.py                 (617 lines) — region evidence
│   ├── ports.py                         (326 lines) — port definitions
│   ├── backend_provenance.py            (35 lines)  — backend traceability
│   ├── image_evidence.py                (164 lines) — image contracts
│   ├── evidence_identity.py             (162 lines) — ID generation
│   ├── perception_evidence_set.py       (150 lines) — evidence aggregation
│   ├── image_preparation.py             (188 lines) — image prep
│   └── __init__.py                      (610 lines) — public facade
│
├── regions/                             (6 files)
│   ├── region_discovery.py              (~1000 lines)
│   ├── region_grounding.py              (~500 lines)
│   ├── region_refinement.py             (~500 lines)
│   ├── region_normalization.py          (~400 lines)
│   ├── region_eligibility.py            (~100 lines)
│   ├── region_semantic_hint.py          (~100 lines)
│   └── __init__.py
│
├── features/                            (5 files)
│   ├── region_feature_pooling.py        (~600 lines)
│   ├── embedding_space.py               (~200 lines)
│   ├── feature_diagnostics.py           (~600 lines)
│   ├── feature_enhancement.py           (~150 lines)
│   └── __init__.py
│
├── semantics/                           (6 files)
│   ├── interpretation_request.py        (~400 lines)
│   ├── prompt_policy.py                 (~800 lines) [includes parsing]
│   ├── interpretation_policy.py         (~300 lines)
│   ├── evidence_views.py                (~500 lines)
│   ├── interpretation_execution.py      (~400 lines)
│   ├── interpretation_audit.py          (~300 lines)
│   └── __init__.py
│
├── execution/                           (5 files)
│   ├── pipeline_resolution.py           (~800 lines)
│   ├── pipeline_presets.py              (~300 lines)
│   ├── stage_execution.py               (~600 lines)
│   ├── perception_result_assembly.py    (~300 lines)
│   └── __init__.py
│
├── artifacts/                           (3 files)
│   ├── feature_payload_store.py         (~400 lines)
│   ├── region_mask_store.py             (~400 lines)
│   ├── semantic_scoring_run.py          (~300 lines)
│   ├── perception_serialization.py      (~200 lines)
│   └── __init__.py
│   [NOTE: run_artifact.py still at root — 2600+ lines, massive]
│
├── adapters/                            (26 files)
│   ├── regions/
│   │   ├── sam2.py, sam3.py, florence2.py, locateanything.py
│   │   └── prepared_images.py
│   ├── features/
│   │   ├── dinov2.py, dinov3.py, siglip2.py, clip.py, alphaclip.py
│   │   ├── clip_text.py, alphaclip_text.py
│   │   └── semantic_scoring.py
│   ├── semantics/
│   │   ├── qwen.py, gemini.py, eagle2_5.py, florence2_semantic.py
│   └── [shared adapter utilities]
│       ├── _feature_values.py
│       ├── _huggingface.py
│       ├── _model_placement.py
│       ├── _release.py
│       ├── _semantic_views.py
│
└── docs/                                (15 files)
    ├── README.md, architecture.md, code-map.md, data-flow.md, testing.md
    ├── scientific-invariants.md (NEW)
    └── adapter-specific docs
```

### Lines of Code Summary

| Category | Approx. Lines | Files |
|----------|---------------|-------|
| Models/Contracts | 3,200 | 9 |
| Regions | 2,500 | 6 |
| Features | 1,550 | 5 |
| Semantics | 2,700 | 6 |
| Execution | 2,000 | 5 |
| Artifacts | 1,300 | 4 |
| Adapters | 7,000+ | 26 |
| Docs | 1,500+ | 15 |
| **Total (estimated)** | **22,000+** | **76** |

---

## 2. Contract Ownership Matrix

### Core Types (in models.py)

| Type | Owner | Consumers | Stability | Notes |
|------|-------|-----------|-----------|-------|
| `PerceptionRunId`, `PerceptionResultId` | execution | runtime, tests | High | Artifact identity |
| `RegionId`, `FeatureId`, `ClaimId`, `ScoreId` | execution | all subdomains | High | Local evidence identity |
| `BackendProvenance` | backend_provenance | adapters, all evidence | High | Adapter traceability |
| `PerceptionRun`, `PerceptionResult` | execution | runtime, artifact, tests | Medium | **Candidate for split** |
| `Region2D`, `BoundingBox2D` | region_perception | features, semantics, sensor_association | High | Region evidence |
| `VisualFeature` | visual_features | semantics, semantic_fusion | High | Feature evidence |
| `SemanticClaim`, `SemanticAttribute`, etc. | semantic_interpretation | semantic_scoring, semantic_fusion, entity_resolution | High | Semantic evidence |
| `SemanticScore`, `SemanticSupport` | semantic_scoring | semantic_fusion, semantic_mapping | High | Scoring evidence |
| `SceneContext` | semantic_interpretation | semantic_fusion, semantic_mapping | Medium | Scene-level evidence |
| `PreparedImage`, `ValidRegion`, `ExclusionRegion`, `TransformationRecord` | visual_input | region_perception, visual_features, semantics | High | Image evidence |
| `HypothesisRole`, `SemanticRegionKind`, `SemanticScoreType`, `FeatureScope` | multiple | models | High | Enums, widely used |
| `EmbeddingSpace` | visual_features | semantics, semantic_fusion, semantic_scoring | High | Embedding identity |
| `DenseFeatureMap`, `DenseFeatureSampling` | visual_features | artifact store | High | Dense sampling contract |

---

## 3. Runtime DAG Analysis

### Current Preset (CANONICAL_PRESET_V1)

```
digraph {
  source -> image_prep [label="SourceObservation"];
  image_prep -> discovery [label="PreparedImage"];
  discovery -> {normalization; grounding; {feature_dense; feature_global}};
  normalization -> {grounding; feature_regional; semantics};
  grounding -> feature_regional;
  {feature_dense; feature_global; feature_regional} -> semantics;
  semantics -> scoring;
  {discovery; normalization; grounding; feature_*; semantics; scoring} -> artifact;
}
```

### Stage Dependencies

| Stage | Input(s) | Output | Backend | Role |
|-------|----------|--------|---------|------|
| **image_prep** | SourceObservation | PreparedImage | image_preparation.py | Visual input |
| **discovery** | PreparedImage | Region2D candidates | 20+ adapters (SAM2, SAM3, Florence2) | Region proposal |
| **normalization** | Candidates | Canonical Region2D | region_normalization.py | Region curation |
| **grounding** | Region proposals + queries | Grounded boxes/points | LocateAnything | Language grounding |
| **refinement** | Box/point + image | Mask Region2D | SAM adapters | Mask generation |
| **feature_dense** | PreparedImage | DenseFeatureMap | DINO, SigLIP, CLIP | Image-wide representation |
| **feature_global** | PreparedImage | VisualFeature (global) | same | Global encoding |
| **feature_regional** | PreparedImage + Region2D | VisualFeature (regional) | same | Region pooling |
| **semantics** | PreparedImage ± Region2D ± Features | SemanticClaim + SceneContext | Qwen, Gemini, Eagle, Florence2 | VLM interpretation |
| **scoring** | SemanticClaim + VisualFeature | SemanticScore | CLIP, AlphaCLIP | Cross-modal scoring |

### Observation

The DAG is already relatively modular per domain (discovery, grounding, refinement live in separate stages; features are stage-separated by scope). The refactoring opportunity is to make these independent capabilities with their own artifact lifecycle.

---

## 4. Artifact Graph

### Current: Single PerceptionRunArtifact

```
PerceptionRunArtifact
├── outputs/
│   ├── results.jsonl                (PerceptionResult serialized)
│   ├── region-discovery-audit.jsonl (Discovery decisions)
│   ├── region-grounding.jsonl       (Grounding results)
│   ├── region-grounding-raw/        (Grounding responses)
│   ├── region-refinement.jsonl      (Refinement results)
│   ├── ineligible-regions.jsonl     (Rejected regions)
│   ├── semantic-interpretations.jsonl
│   ├── semantic-interpretation-failures.jsonl
│   ├── semantic-views/              (View payloads)
│   ├── features/
│   │   ├── feature-index.jsonl
│   │   └── *.npy                    (Payloads)
│   └── masks/
│       ├── mask-index.jsonl
│       └── *.npy
├── metrics/
│   ├── stage-timings.jsonl
│   └── feature-extraction.jsonl
└── debug/
    ├── 20-region-discovery/         (Optional)
    ├── 30-feature-extraction/       (Detailed diagnostics)
    ├── 40-semantic-interpretation/  (Interpretation diagnostics)
    └── [others]
```

**Issues:**
1. Single artifact aggregates outputs from 5 conceptually independent domains
2. No independent reuse: if you want to re-run semantic interpretation with different VLM, you must re-run discovery/features
3. PerceptionResult contains all evidence types; difficult to extend independently
4. Lineage is implicit (path-based)

### Proposed: Decomposed Artifact Graph

```
VisualInputRunArtifact? (TBD)
    └─> outputs: PreparedImage metadata

RegionPerceptionRunArtifact
    └─> outputs: regions.jsonl, audit.jsonl, masks/
        
VisualFeaturesRunArtifact
    └─> outputs: features/, feature-index.jsonl, embedding-spaces.json

SemanticInterpretationRunArtifact
    └─> outputs: interpretations.jsonl, failures.jsonl, views/
    
SemanticScoringRunArtifact
    └─> outputs: scores.jsonl

Lineage (explicit in each manifest):
- RegionPerceptionRun: visual_input?, PreparedImage
- VisualFeaturesRun: visual_input?, PreparedImage, region_perception?(optional)
- SemanticInterpretationRun: PreparedImage, region_perception, visual_features
- SemanticScoringRun: semantic_interpretation, visual_features
```

---

## 5. Configuration Flow Analysis

### Current: Implicit in runtime

```toml
# config.toml (e.g., in experiments or runtime)
[visual_perception]
discovery_backend = "sam2"
discovery_config = { ... }
grounding_backend = "locate_anything"
feature_backend = "dinov3"
semantic_backend = "qwen"
preset = "CANONICAL_PRESET_V1"
```

**Current flow:**
```
TOML → runtime.composition → adapter selection → stage DAG → execution
```

**Observation:** Config keys don't reflect the decomposed structure. Each new capability will need its own typed config.

---

## 6. Debug Structure

### Current

```
debug/
├── 30-feature-extraction/
│   ├── feature-summary.json
│   ├── feature-map-meta.json
│   ├── embedding-spaces.json
│   ├── extraction-events.jsonl
│   ├── dense/
│   │   └── [previews]
│   └── regions/
│       └── [per-region diagnostics]
├── 40-semantic-interpretation/
│   ├── <request-id>/
│   │   ├── request.json
│   │   ├── prompt.txt
│   │   ├── diagnostics.json
│   │   ├── parsed-response.json
│   │   ├── semantic-claims.json
│   │   ├── error.json [if failed]
│   │   └── raw-response.txt [if full level]
└── [no discovery/grounding debug by default]
```

**Rule:** Debug is optional, outputs are sufficient for evidence. Reader must not depend on debug.

---

## 7. Public API Surface Analysis

### Exports from __init__.py

**Count:** 180+ symbols

**Categories:**
- **Type aliases:** 10 (PerceptionRunId, RegionId, FeatureId, ClaimId, ScoreId, etc.)
- **Core contracts:** 40+ (Region2D, VisualFeature, SemanticClaim, PreparedImage, etc.)
- **Enums:** 8 (HypothesisRole, SemanticRegionKind, SemanticScoreType, FeatureScope, etc.)
- **Exceptions:** 20+ (various domain-specific errors)
- **Functions:** 40+ (pool_region_feature, normalize_regions, resolve_pipeline, etc.)
- **Runtime/Artifact:** 20+ (PerceptionRunWriter, PerceptionRunReader, StageRunner, etc.)

**Observation:** The API is already quite large and mixed. Decomposition will require careful redistribution to new capability facades.

---

## 8. Test Ownership Analysis

### Test Structure

```
tests/visual_perception/
├── test_region_*.py          (7 files)
├── test_feature_*.py         (5 files)
├── test_semantic_*.py        (8 files)
├── test_pipeline_*.py        (4 files)
├── test_artifact_*.py        (3 files)
├── backends/                 (20+ adapter tests)
├── contracts/                (type/invariant tests)
└── integration/              (end-to-end tests)
```

**Total:** ~80 test files, ~10,000+ test cases

**Observation:** Tests already reflect domain structure (regions/, features/, semantics/). Migration should be straightforward; tests become per-capability.

---

## 9. Consumer Audit

### Direct consumers of visual_perception

| Consumer | Import Pattern | Usage |
|----------|---|---|
| **runtime** | from contextmap.visual_perception import ... | Constructs stages, compose DAG, handles artifacts |
| **sensor_association** | Region2D, VisualFeature | Associates evidence to 3D space |
| **semantic_fusion** | SemanticClaim, SceneContext, SemanticScore | Accumulates evidence multi-view |
| **semantic_mapping** | SemanticClaim, SceneContext | Creates entity persistence |
| **entity_resolution** | Region2D, SemanticClaim | Resolves identities across runs |
| **evaluation** | PerceptionRunReader, PerceptionResult | Reads artifacts, computes metrics |
| **experiments** | Runtime, various contracts | Runs pipelines, analyzes results |
| **CLI/scripts** | PerceptionRunWriter, contracts | Ad-hoc execution |

**Observation:** Most consumers depend on the runtime DAG and contracts. Few depend on internal structure. Decomposition can be isolated to visual_perception + minimal runtime updates.

---

## 10. Experiment Classification

### Active (Maintained)

- **Sam3 mask scale probe** — evaluates SAM3 mask geometry
- **Correction benchmark** — tests discovery accuracy
- **Raw vs. voxel** — compares feature representations
- Others maintained in `/experiments/`

**Action:** Update imports only, preserve experiment logic.

### Historical (Frozen)

- Older benchmarks with specific commit references
- These should NOT be automatically rewritten

**Action:** Document dependency on old code path if necessary.

---

## 11. Problems Identified

### Major Boundaries to Formalize

1. **visual_input:** Does it need artifact? Or ephemeral?
   - **Question:** Is PreparedImage reused across multiple runs? How expensive is preparation?
   - **Risk:** Artifact or not changes downstream reuse significantly

2. **PerceptionResult aggregation:** Should it remain?
   - **Question:** Do consumers really need all evidence in one structure, or can they reference independent artifacts?
   - **Risk:** Keeping it adds coupling; removing it breaks many consumers

3. **Run identity vs. artifact identity:**
   - **Current:** PerceptionRun maps to single PerceptionRunArtifact
   - **Proposed:** Multiple independent artifacts per run?
   - **Risk:** Config keys, reuse keys, experiment scripts all depend on identity semantics

4. **Config structure:** How does typed config distribute?
   - **Current:** runtime handles all backend selection
   - **Target:** Each capability has own config interface
   - **Risk:** Need migration plan for existing configs/experiments

5. **Debug inventory question (noted in plan):**
   - Currently debug/ files go into file_inventory
   - Rule says debug should be optional/removable
   - **Decision pending:** Keep in inventory or remove?

### Minor Issues

- `run_artifact.py` at root is 2600+ lines; should be in artifacts/
- Some adapters could be split by capability (e.g., sam2.py has both discovery + refinement)
- Some config keys don't reflect semantic structure
- Documentation references non-existent capabilities

---

## 12. Capability Boundary Validation

### Proposed Capability: visual_input

**Problem:** Prepare raw image for downstream processing

**Owns:**
- PreparedImage
- ValidRegion, ExclusionRegion
- TransformationRecord
- Image preparation policies

**Boundary Validation:**
- ✓ Independent problem (image prep)
- ✓ Own contract (PreparedImage)
- ✓ One reason to change (image prep needs)
- ✓ Consumers: regions, features, semantics (all need PreparedImage)
- ⚠️ **Artifact question:** Needed? Only if reused across multiple runs or expensive enough to cache
- ✓ Evaluation: Pixel-level audit, transform accuracy

**Risk:** If image prep is cheap and always run fresh, artifact overhead may not be justified. Can remain logical capability without artifact.

---

### Proposed Capability: region_perception

**Problem:** Discover, ground, refine, and curate regions

**Owns:**
- Region2D
- Discovery passes, normalization, merge
- Grounding request/response
- Mask refinement
- Eligibility filtering
- Lineage and audit

**Boundary Validation:**
- ✓ Independent problem (region evidence)
- ✓ Own contract (Region2D)
- ✓ Own lifecycle (discovery → normalization → refinement → audit)
- ✓ Consumers: visual_features (pooling), semantics (region-scoped)
- ✓ Artifact pressure: YES (region audit is valued, reuse scenario likely)
- ✓ Evaluation: Region count, quality, F1 over ground truth

**Risk:** Low. This capability is already well-separated in code and tests.

---

### Proposed Capability: visual_features

**Problem:** Extract visual representations

**Owns:**
- VisualFeature
- EmbeddingSpace
- Dense/global/region feature extraction
- Feature pooling (dense → regional)
- Resolution enhancement
- Feature payload storage

**Boundary Validation:**
- ✓ Independent problem (visual representation)
- ✓ Own contract (VisualFeature)
- ✓ Own lifecycle (extraction, pooling, enhancement)
- ✓ Consumers: semantics (per-feature scoring), semantic_fusion (reuse across views)
- ✓ Artifact pressure: YES (large payloads, computation cost, reuse scenario)
- ✓ Evaluation: Embedding quality metrics, pooling accuracy

**Risk:** Moderate. Dense feature operations are complex; must preserve sampling geometry exactly.

---

### Proposed Capability: semantic_interpretation

**Problem:** Produce semantic hypotheses from visual evidence

**Owns:**
- SemanticClaim, SceneContext
- Semantic requests/policies
- Evidence views
- Prompt rendering + parsing
- Parse failures, backend failures
- Semantic audit

**Boundary Validation:**
- ✓ Independent problem (semantic inference)
- ✓ Own contract (SemanticClaim, SceneContext)
- ✓ Own lifecycle (request → prompt → VLM → parse → claim)
- ✓ Consumers: semantic_scoring (score claims), semantic_fusion (accumulate)
- ✓ Artifact pressure: YES (VLM cost, failure audit, raw response preservation)
- ✓ Evaluation: Claim accuracy, parsing correctness

**Risk:** Moderate-High. Prompt + parsing is complex; changes must be validated independently.

---

### Proposed Capability: semantic_scoring

**Problem:** Judge compatibility between semantic claims and visual features

**Owns:**
- SemanticScore
- Score type/semantics
- Scoring policy (if exists)
- Scorer backends (CLIP, AlphaCLIP)
- Scoring artifact

**Boundary Validation:**
- ✓ Independent problem (cross-modal scoring)
- ✓ Own contract (SemanticScore)
- ✓ Own lifecycle (request → score → storage)
- ⚠️ **Consumers:** Only semantic_fusion, semantic_mapping (both downstream)
- ✓ Artifact pressure: LOW (scores are small, not expensive to recompute)
- ✓ Evaluation: Score correlation with human labels

**Risk:** Low. Scoring is already fairly isolated. Artifact may not be necessary (scores could be inline with claims).

**Question:** Does semantic_scoring need its own artifact, or should scores live embedded in SemanticInterpretationRunArtifact?

---

## Summary: Boundary Assessment

| Capability | Independence | Artifact? | Risk | Notes |
|-----------|---|---|---|---|
| visual_input | High | TBD | Low | Decision needed on artifact necessity |
| region_perception | High | YES | Low | Already well-separated |
| visual_features | High | YES | Moderate | Preservation of invariants critical |
| semantic_interpretation | High | YES | Moderate-High | Prompt/parsing must be carefully managed |
| semantic_scoring | High | TBD | Low | Possible to keep inline with claims |

---

## Next Steps

1. **Validate** that the above boundary assessment holds against code reality
2. **Decide** on the 10 critical decision points (from Phase 1 plan)
3. **Design** explicit artifact graph and lineage
4. **Design** config structure per capability
5. **Design** runtime DAG changes
6. **Plan** consumer updates (minimal)
7. **Proceed** to implementation phases

---

**Phase 0 Status:** Audit complete. Ready for Phase 1 (Boundary Validation + Explicit Decisions).
