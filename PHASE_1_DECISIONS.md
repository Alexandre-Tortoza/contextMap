# Phase 1: Explicit Decisions for Decomposition

**Status:** REVISED for v1.0.0 (no legacy layers)  
**Policy:** Direct migration to final architecture. No deprecation periods, no dual artifacts, no compatibility facades.  
**Required Before:** Phase 2 implementation starts

---

## Principle: Build v1 Right, Not v1 with Debt

ContextMap2 is pre-v1.0.0. We do not maintain stable public APIs yet. Breaking changes are acceptable. We will not:
- Maintain deprecation periods
- Keep dual artifacts or parallel architectures
- Create compatibility shims or aliases
- Support outdated contracts for calendar reasons
- Defer architectural cleanup

**Policy:**
```
build new architecture
→ migrate all active consumers
→ migrate runtime/config/artifacts/tests/docs
→ validate (make check passes)
→ remove old code entirely
```

---

## Overview

The following 10 decisions define the clean-break migration. Each has a **Breaking Change** section stating what will be removed.

---

## Decision 1: Visual Input Artifact

### Question
Does `visual_input` capability produce its own artifact, or is image preparation ephemeral?

### Analysis (from Phase 0 Audit)
- **Current behavior:** PreparedImage is in-memory, not persisted independently
- **Cost:** Image preparation is fast (resize, crop, rectify typical for VR)
- **Reuse pressure:** Weak — downstream stages adapt to input image; no real scenario for re-running only downstream with same input
- **Lineage:** Upstream is SourceObservation (ingestion); downstream stages reference input observation

### Decision
**Visual input produces NO independent artifact.** `PreparedImage` is ephemeral.

- `visual_input` is a logical capability with a port (`ImagePreparation`)
- Runtime calls it synchronously within each downstream run
- No VisualInputRunArtifact
- No separate storage cost

### Validation Needed
Code review in Phase 2 to confirm PreparedImage is not persisted anywhere today and has no consumers expecting it as independent artifact.

### Breaking Changes
None — this was never an artifact anyway.

### Impact
- `visual_input` owns the PreparedImage contract but no artifact storage
- Each RegionPerceptionRun, VisualFeaturesRun, SemanticInterpretationRun calls visual_input internally
- Lineage traces back to SourceObservation + visual_input policy, not intermediate artifact
- Simpler runtime composition (5 independent artifacts instead of 6)

---

## Decision 2: PerceptionResult Replacement

### Question
`PerceptionResult` currently aggregates regions, features, claims, and scores. In the decomposed architecture, this violates ownership.

### Analysis
- **Current:** PerceptionResult contains `[regions, features, claims, scene_context, semantic_scores]`
- **Problem:** These now belong to different capabilities with different lifecycles and reuse scenarios
- **Active consumers:** runtime, sensor_association (needs Region2D + VisualFeature + SemanticClaim), evaluation, semantic_fusion, semantic_mapping
- **Use pattern:** Consumers read result.regions, result.features, result.claims independently

### Decision
**REMOVE PerceptionResult as aggregate.** Each capability produces its own result type.

- `RegionPerceptionRunArtifact` provides `RegionEvidence` (regions, audit, masks)
- `VisualFeaturesRunArtifact` provides `FeatureEvidence` (features, embeddings, diagnostics)
- `SemanticInterpretationRunArtifact` provides `SemanticEvidence` (claims, scene_context, audit)
- `SemanticScoringRunArtifact` provides `ScoringEvidence` (scores)
- Runtime reader combines these on-demand if needed

### Consumer Migration
All active consumers updated to read per-capability artifacts:
- **sensor_association:** Read RegionPerceptionRun for regions, VisualFeaturesRun for features
- **semantic_fusion:** Read SemanticInterpretationRun + SemanticScoringRun
- **semantic_mapping:** Read SemanticInterpretationRun
- **evaluation:** Read individual artifacts, combine as needed
- **Tests:** Use per-capability fixtures

### Breaking Changes
- ✗ `PerceptionResult` dataclass deleted
- ✗ All imports of PerceptionResult must be updated
- ✗ Consumer code that access `.result.regions`, `.result.features`, etc. must be refactored to read from separate artifacts
- ✗ Tests that construct PerceptionResult must use per-capability factories

### Timeline
- Phase 2-6: Extract capabilities, update all consumers in same PRs
- Phase 7+: Publish without PerceptionResult in public API
- v1.0.0: No PerceptionResult exists

### Impact
- Cleaner ownership (each capability owns its result)
- Enables independent reuse (can use region evidence with different feature/semantic backends)
- Simpler schema per artifact (no artificial bloat)
- Consumers become slightly more verbose but more explicit about what they need

---

## Decision 3: PerceptionRunArtifact Replacement

### Question
`PerceptionRunArtifact` currently bundles all evidence (regions, features, claims, scores) in one artifact. How is it replaced?

### Current Artifact Graph
```
PerceptionRunArtifact/
├── outputs/
│   ├── results.jsonl (PerceptionResult)
│   ├── region-discovery-audit.jsonl
│   ├── region-grounding.jsonl
│   ├── region-refinement.jsonl
│   ├── semantic-interpretations.jsonl
│   ├── semantic-interpretation-failures.jsonl
│   ├── features/
│   └── masks/
├── metrics/
└── debug/
```

### New Artifact Graph (v1.0.0)
```
RegionPerceptionRunArtifact/
├── outputs/regions.jsonl, audit.jsonl, masks/
SemanticInterpretationRunArtifact/
├── outputs/claims.jsonl, scene.jsonl, views/
VisualFeaturesRunArtifact/
├── outputs/features/, embeddings/
SemanticScoringRunArtifact/
├── outputs/scores.jsonl
```

With **lineage explicitly stored** in each manifest:
- RegionPerceptionRunManifest: `{ depends_on: [SourceObservation, visual_input_policy] }`
- VisualFeaturesRunManifest: `{ depends_on: [SourceObservation, visual_input_policy, (optional) RegionPerceptionRun] }`
- SemanticInterpretationRunManifest: `{ depends_on: [RegionPerceptionRun, VisualFeaturesRun, SourceObservation] }`
- SemanticScoringRunManifest: `{ depends_on: [SemanticInterpretationRun, VisualFeaturesRun] }`

### Decision
**REPLACE PerceptionRunArtifact** with per-capability independent artifacts.

- Each capability owns one artifact type
- No aggregation layer
- Lineage is explicit (metadata in each artifact manifest, not inferred from paths)
- Runtime can construct run-level manifest that references all 5 independent artifacts (logical grouping only)

### Breaking Changes
- ✗ PerceptionRunArtifact class deleted
- ✗ PerceptionRunWriter/PerceptionRunReader deleted
- ✗ Directory structure changes (no more `<run>/visual_perception/outputs/`)
- ✗ New paths: `<run>/region_perception/`, `<run>/visual_features/`, `<run>/semantic_interpretation/`, `<run>/semantic_scoring/`
- ✗ File structure changes within each artifact
- ✗ Scripts relying on PerceptionRunArtifact paths must be updated

### Consumer Migration
- **evaluation:** Updated to read per-capability artifacts, assemble results on-demand
- **sensor_association:** Reads RegionPerceptionRunArtifact + VisualFeaturesRunArtifact
- **semantic_fusion:** Reads SemanticInterpretationRunArtifact + SemanticScoringRunArtifact
- **experiments:** Updated import paths and artifact reading code
- **tests:** Updated artifact fixtures and readers

### Historical Data
- Old PerceptionRunArtifacts remain as frozen archives (no code path to read them)
- If reproducibility of historical experiments requires old artifacts, document the old code commit to checkout
- No production code maintains backward compatibility with old format

### Timeline
- Phase 2-6: Implement new artifacts, update consumers, tests, runtime
- Phase 8: Remove PerceptionRunArtifact code entirely
- v1.0.0: Only new per-capability artifacts exist

### Impact
- Cleaner artifact graph, no artificial aggregation
- Independent reuse (can reuse region evidence with different VLM)
- Explicit lineage (no path inference)
- Simpler schema per artifact
- Consumers slightly more verbose but more correct

---

## Decision 4: Public API Structure

### Question
How does the public API reorganize when visual_perception is decomposed?

### Current
```python
from contextmap.visual_perception import Region2D, VisualFeature, SemanticClaim, ...
```

### New (v1.0.0)
```python
from contextmap.region_perception import Region2D
from contextmap.visual_features import VisualFeature
from contextmap.semantic_interpretation import SemanticClaim
from contextmap.semantic_scoring import SemanticScore
```

### Decision
**NO compatibility facade. Update all imports directly.**

- `contextmap.visual_perception` does NOT re-export new capability APIs
- Each new capability has its own public `__init__.py`
- Imports changed directly at source (tests, runtime, consumers, docs)
- `contextmap.visual_perception` package is **completely removed** after decomposition
- No aliases, no shims, no gradual deprecation

### Consumers Updated
- runtime
- sensor_association
- semantic_fusion
- semantic_mapping
- evaluation
- tests
- experiments (active only)
- docs
- scripts

### Breaking Changes
- ✗ All imports from `contextmap.visual_perception` must change
- ✗ Type hints that reference `contextmap.visual_perception` types must be updated
- ✗ Old package path completely removed (not "deprecated", removed)
- ✗ No backward compatibility maintained

### Migration Strategy
1. Phase 2-6: Create new capabilities with public APIs
2. Phase 7: Update all imports in codebase (runtime, tests, downstream)
3. Phase 9: Delete visual_perception package entirely
4. v1.0.0: No contextmap.visual_perception exists

### Impact
- Clear ownership (each capability is independently importable)
- No artificial aggregation in public API
- Simpler mental model (import what you need)
- No maintenance burden of compatibility layer
- Consumers are explicit about dependencies

---

## Decision 5: Runtime Stage IDs

### Question
How are stage IDs renamed when runtime DAG reflects new capabilities?

### Current Stage IDs
- `discovery` → region_perception
- `normalization` → region_perception
- `grounding` → region_perception
- `refinement` → region_perception
- `feature_dense` → visual_features
- `feature_global` → visual_features
- `feature_regional` → visual_features
- `semantic_interpretation` → semantic_interpretation
- `semantic_scoring` → semantic_scoring

### New Stage IDs (v1.0.0)
```
visual_input
region_perception/discovery
region_perception/normalization
region_perception/grounding
region_perception/refinement
visual_features/dense
visual_features/global
visual_features/regional
semantic_interpretation
semantic_scoring
```

### Decision
**RENAME stage IDs immediately** to reflect ownership.

Format: `<capability>_<substage>` (e.g., `region_discovery`, `features_dense`, etc.)

- Reflects new architecture
- Clearer in logs, configs, and DAGs
- No backward compatibility

### Updated Everywhere
- Runtime DAG definition
- Config/preset structures
- Stage execution logging
- Artifact manifests
- Tests
- Docs
- Experiments (active only)

### Breaking Changes
- ✗ All stage ID references change
- ✗ Config keys referencing old stage IDs change
- ✗ Experiment scripts that query stage status must be updated
- ✗ Reuse keys recalculated (based on new stage IDs)

### Timeline
- Phase 7: Rename stages in runtime DAG
- Phase 7: Update all references (config, tests, experiments)
- v1.0.0: Only new stage IDs exist

### Impact
- Clearer ownership (stage ID reflects capability)
- DAG is self-documenting
- No confusion about which capability owns which stage
- Slightly more verbose but more explicit

---

## Decision 6: Config Structure Redesign

### Question
How is configuration distributed across the new capabilities?

**Current:** All config under `[visual_perception]` with backend selection handled by runtime

**Proposed:** Each capability has typed config; runtime composes them

### Option A: Flat Config (Per-Backend)
```toml
[visual_perception.discovery.sam2]
model = "sam2.1"
device = "cuda"

[visual_perception.features.dinov3]
model = "dinov3_l"
resolution = 1024
```

**Pros:**
- Direct mapping to backends
- Explicit per-backend parameters

**Cons:**
- Tightly couples config to specific backends
- Can't express multi-backend scenarios
- Not suitable for decomposition

### Option B: Hierarchical by Capability
```toml
[visual_input]
# (no config needed for ephemeral image prep)

[region_perception]
discovery = "sam2"
discovery_config = { model = "sam2.1", device = "cuda" }

[visual_features]
dense_backend = "dinov3"
dense_config = { model = "dinov3_l", resolution = 1024 }

[semantic_interpretation]
interpreter = "qwen"
prompt_policy = "v2"

[semantic_scoring]
scorer = "clip"
```

**Pros:**
- Clear ownership per capability
- Decoupled: each capability owns its policy
- Extensible

**Cons:**
- More verbose
- Runtime must know how to compose

### Option C: Capability Packages Manage Own Config
```
contextmap.region_perception.config.RegionPerceptionConfig
contextmap.visual_features.config.VisualFeaturesConfig
(etc.)

Runtime loads these and passes to composition
```

**Pros:**
- Type-safe, strongly-typed per-capability
- Self-contained (config logic within capability)
- Enables independent policy evolution

**Cons:**
- Each capability needs config infrastructure
- Runtime must know all capability config types
- More complex initially

### Decision
**REDESIGN config for v1.0.0** — typed config per capability, no parallel parsing.

Structure:
```toml
[region_perception]
discovery_backend = "sam2"
grounding_backend = "locate_anything"

[region_perception.sam2]
model = "sam2.1"
device = "cuda"

[visual_features]
dense_backend = "dinov3"

[visual_features.dinov3]
model = "dinov3_l"

[semantic_interpretation]
interpreter = "qwen"
prompt_policy = "v2"

[semantic_scoring]
scorer = "clip"
```

Each capability defines typed `config.Config` dataclass. Runtime loads TOML and composes per-capability configs.

### Breaking Changes
- ✗ Old `[visual_perception]` config structure completely replaced
- ✗ All config keys change
- ✗ Scripts/experiments using old config must be updated
- ✗ No backward-compatible parsing of old configs
- ✗ Migration guide required but no dual-parsing

### Timeline
- Phase 2-6: Each capability defines typed config
- Phase 7: Runtime loads new config structure
- Phase 8: Update all experiment configs
- v1.0.0: Only new config structure exists

### Impact
- Cleaner ownership (each capability owns its config)
- Type-safe configuration
- Simpler runtime composition
- No maintenance of legacy config parsing

---

## Decision 7: Reuse Key Recalculation

### Decision
**RECALCULATE reuse keys** — do not preserve old keys artificially.

New reuse key calculation includes:
- New capability boundaries
- New stage IDs
- New config structure
- New artifact structure
- Explicit lineage

### Breaking Changes
- ✗ Old reuse keys invalid
- ✗ Historical experiments cannot be identified by old keys
- ✗ If run needs to be reproduced, must provide explicit artifact reference

### Impact
- New architecture is properly reflected in identity
- No false positives/negatives in reuse detection
- Experiments must provide explicit run IDs if reproducibility is needed

---

## Decision 8: Experiment Script Handling

### Question
How are the ~15 experiments in `/experiments/` handled?

### Process
1. **Classify** each script:
   - Active: currently maintained, re-executable, in use
   - Historical: frozen record for reference, not maintained
   - Obsolete: no longer used, remove

2. **Active experiments:** Update imports, config, artifact paths
   - Validate correctness carefully
   - Update to new stage IDs, capability APIs, artifact readers

3. **Historical experiments:** Freeze with documentation
   - Tag with commit where they were last compatible
   - Document old artifact path if needed for reproduction
   - No production code to support them (old code branch only)

4. **Obsolete:** Delete

### Breaking Changes
- ✗ Active experiment scripts must be manually updated
- ✗ Config for active experiments must be migrated
- ✗ Old experiment commits may need old code checkout for reproduction

### Timeline
- Phase 1: Classify experiments
- Phase 2-6: Update active experiments in parallel with capability extraction
- Phase 8: Ensure all active experiments work with new structure

### Impact
- Only active experiments maintained
- Clear record of historical experiments without production code burden
- Cleaner codebase (obsolete scripts removed)

---

## Decision 9: Debug in Artifact Inventory

### Question
Should `debug/` be part of contractual file_inventory?

### Current Problem
- File inventory currently includes debug files
- But rule says debug is optional and removable
- Contradiction: if debug is in inventory, removing it breaks integrity check

### Decision
**REMOVE debug/ from file_inventory for v1.0.0 schema**.

- file_inventory covers only `outputs/` and `metrics/`
- `debug/` is parallel, independently managed
- Removing debug does not invalidate artifact
- Reader never uses inventory to locate debug files

### Breaking Changes
- ✗ Schema change: file_inventory no longer includes debug paths
- ✗ Integrity verification changes (only contractual paths verified)
- ✗ Reader must not assume debug availability

### Timeline
- Phase 8-9: Update artifact schema
- v1.0.0: New schema deployed

### Impact
- Clean separation of contractual vs. optional content
- Debug truly optional and removable
- Simpler inventory logic
- Aligns with "debug is not dependency" principle

---

## Decision 10: Visual Perception Package Removal

### Question
When is `contextmap.visual_perception` package completely removed?

### Decision
**REMOVE entirely when decomposition is complete** (Phase 9 of implementation).

No deprecation period, no compatibility facade.

- After all capabilities extracted
- After all consumers updated (runtime, tests, downstream)
- After new APIs published and verified
- Package is deleted, not marked deprecated

### Breaking Changes
- ✗ contextmap.visual_perception package completely deleted
- ✗ All imports from visual_perception must have been migrated
- ✗ No compatibility layer exists

### Timeline
- Phase 2-6: Extract capabilities, update imports as each capability is created
- Phase 7-8: Complete consumer migration
- Phase 9: Delete visual_perception package entirely
- v1.0.0: No visual_perception package exists

### Impact
- Clean codebase
- Five independent capabilities at top level
- Clear ownership
- No artificial aggregation or facade maintenance cost

---

## Summary: v1.0.0 Breaking Changes

| Decision | Action | Breaking? | Timeline |
|----------|--------|-----------|----------|
| 1. Visual Input Artifact | No artifact (ephemeral) | ✓ | Phase 2 |
| 2. PerceptionResult | Remove | ✓ | Phase 2-6 |
| 3. PerceptionRunArtifact | Replace with per-capability artifacts | ✓ | Phase 2-6 |
| 4. Public API | Direct imports to new capabilities, no facade | ✓ | Phase 7 |
| 5. Stage IDs | Rename immediately | ✓ | Phase 7 |
| 6. Config | New typed structure | ✓ | Phase 2-6 |
| 7. Reuse Keys | Recalculate | ✓ | Phase 7 |
| 8. Experiments | Update active, freeze/remove others | ✓ | Phase 1-6 |
| 9. Debug Inventory | Remove from schema | ✓ | Phase 8-9 |
| 10. visual_perception Package | Delete entirely | ✓ | Phase 9 |

---

## Collective Impact

**These 10 decisions create a clean v1.0.0 architecture:**
- No legacy code paths
- No compatibility facades
- No artificial aggregation
- No backward compatibility burden
- All 10 are breaking changes (intentionally)

**Cost:** Immediate disruption to consumers (runtime, sensor_association, semantic_fusion, evaluation, tests, experiments)

**Benefit:** Correct first-version architecture, not retrofitted onto debt

---

## Approval Required

Each decision is standalone but collectively they define the complete v1.0.0 architecture.

**I am ready to proceed with Phase 2 (Extract semantic_scoring) once you approve these 10 decisions.**

---

**Status:** Awaiting approval on all 10 decision points (v1.0.0 clean-break version).
