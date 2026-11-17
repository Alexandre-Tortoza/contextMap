# Phase 3: Region Perception Extraction — Audit

**Target Capability:** region_perception  
**Source:** visual_perception  
**Status:** Pre-extraction analysis  
**Date:** 2026-10-01

---

## Current State Analysis

### What is region_perception?

Region perception is the discovery, auditing, and normalization of candidate 2D regions in image frames.

**Current ownership:** Everything under visual_perception.regions/*

**Conceptual flow:**
```
PreparedImage
    ↓
region_discovery (find candidates)
    ↓
AuditedRegions (audit report)
    ↓
region_normalization (merge/filter)
    ↓
Region2D (accepted regions)
```

---

## Scope: What Moves to region_perception

### Contracts (from visual_perception.models)

**Core region types:**
- `RegionId`
- `Region2D` (dataclass with validated fields)
- `RegionKind` enum
- `MaskProvenance`

**Discovery output:**
- `DiscoveryOutput`
- `DiscoveryPass`, `DiscoveryPassConfig`
- `PassKind` enum
- `DiscoveryRunResult`
- `TilingConfig`
- `BorderPolicy` enum
- `RegionDiscoveryIdentity`

**Normalization:**
- `NormalizationResult`
- `MergeDecision`, `MergeKind`
- `MergeRepresentativePolicy`
- `NormalizationConfig`

**Audit:**
- `AuditedRegions`
- `RegionDiscoveryAudit`
- `BackendDiagnostics`

### Ports (from visual_perception.ports)

- `RegionDiscovery` protocol
- `AuditedRegionDiscovery` protocol
- `RegionGrounding` protocol (optional, keeps backend integration separate)
- `RegionRefinement` protocol (optional, may extract later)

### Adapters (from visual_perception/adapters/regions/)

- SAM2 region discovery
- SAM region discovery (legacy)
- Any other region discovery backends
- Grounding adapters (DINO, etc.) — **may stay in visual_perception or move to separate grounding capability**

### Artifacts

Move from visual_perception/artifacts/:
- RegionPerceptionRunManifest
- RegionPerceptionRunWriter
- RegionPerceptionRunReader

### Services/Execution

From visual_perception/execution/ or visual_perception/regions/:
- Region discovery stage execution
- Region normalization logic
- Region audit logic

### Configuration

From runtime.catalog:
- visual_perception.region_discovery → region_perception.discovery
- visual_perception.region_grounding → region_perception.grounding (or keep separate)
- visual_perception.region_refinement → region_perception.refinement (or keep separate)

---

## Scope: What Stays in visual_perception

**NOT moving (keep in visual_perception):**
- PreparedImage (input contract, owned by visual_input)
- Image evidence contracts (ValidRegion, TransformationRecord, etc.)
- Feature-related code (VisualFeature, embedding spaces, pooling)
- Semantic claims and interpretation
- PerceptionResult (aggregate that references regions, not owns them)
- Execution orchestration (PerceptionRun, stage coordination)

**Grounding/Refinement decision:**
- These are optional, specialized region processing
- Could stay as sub-stages of region_perception
- Or extract as separate capabilities later
- For Phase 3: Keep within region_perception initially (can split in Phase 4 if needed)

---

## Dependencies and Constraints

### Inbound dependencies (who uses region_perception)

- visual_perception (needs Region2D for PerceptionResult)
- sensor_association (reads Region2D for spatial grounding)
- evaluation (audits region discovery quality)
- experiments (uses region candidates for benchmarking)

### Internal dependencies

- Region discovery → PreparedImage (input)
- Region discovery → Region2D (output)
- Region audit → discovery results
- Region normalization → discovery + audit

### Outbound dependencies (what region_perception depends on)

- contextmap.ingestion (SourceObservationId, timestamps)
- contextmap.shared (Bounds, Vector, etc.)
- contextmap.visual_perception.image_evidence (PreparedImage, ValidRegion)

**Note:** image_evidence stays in visual_perception because PreparedImage is still an input contract shared with other visual processing.

---

## Known Complexities

### 1. PerceptionResult Still References Region2D

**Current:**
```python
@dataclass
class PerceptionResult:
    regions: Sequence[Region2D] = ()
    # ...
```

**After Phase 3:**
- Region2D moves to region_perception
- PerceptionResult still needs Region2D in its type signature
- region_perception exports Region2D (public contract)
- PerceptionResult imports from region_perception

**This is correct:** PerceptionResult aggregates references to regions produced by region_perception.

### 2. Region Discovery is Optional

Some pipelines may skip region discovery or use only image-global features.

- Region discovery should be optional in runtime DAG
- Fallback: empty regions list in PerceptionResult
- Already handled by Phase 2 pattern (optional stages)

### 3. Audit Data

Region audit is diagnostic, not contractual (like all debug/).

- Include in RegionPerceptionRunArtifact.debug/ (not outputs/)
- Consumers don't depend on audit
- Audit is optional, inspectable, never contractual

### 4. Refinement and Grounding

These are specialized, optional post-processing on regions.

**Decision for Phase 3:**
- Keep within region_perception as optional sub-stages
- If they grow complex, extract as separate capabilities in Phase 4+
- For now: region_perception owns discovery + grounding + refinement

---

## Phase 3 Extraction Steps (Plan)

### Step 1: Audit and plan (this document)
- ✅ Identify contracts, ports, adapters
- ✅ Identify dependencies
- ✅ Identify consumer impact

### Step 2: Create region_perception package
- Create src/contextmap/region_perception/
- Create contracts.py, ports.py, artifacts.py, adapters/
- Extract Region2D and related contracts

### Step 3: Move adapters and execution
- Move region discovery backends
- Move grounding adapters
- Move normalization and audit logic

### Step 4: Create artifacts
- RegionPerceptionRunManifest
- RegionPerceptionRunWriter
- RegionPerceptionRunReader

### Step 5: Update runtime
- Update catalog.py (new component IDs)
- Update composition.py (factories, stage definition)
- Update runtime DAG

### Step 6: Migrate consumers
- Update visual_perception (imports Region2D from region_perception)
- Update sensor_association (imports Region2D)
- Update evaluation (imports Region2D)
- Update tests

### Step 7: Remove old code
- Delete from visual_perception/regions/
- Delete old artifact classes
- Update visual_perception.__init__.py
- Update visual_perception.models (remove Region2D, RegionId, etc.)

### Step 8: Documentation
- README (same pattern as semantic_scoring)
- code-map.md
- Architecture decisions

### Step 9: Verification
- Syntax validation
- Import verification
- Ownership search
- Completion report

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| PerceptionResult still needs Region2D | Medium | Import from region_perception; clear dependency |
| Region discovery is large (SAM2) | Medium | Adapters isolated; backends lazy-loaded |
| Grounding/refinement complexity | Medium | Keep within region_perception initially |
| Audit/debug data handling | Low | Follows Phase 2 pattern (debug/ not contractual) |
| Optional stage handling | Low | Phase 2 proved pattern with semantic_scoring |
| Configuration migration | Medium | Update all active experiment configs |

---

## Expected Effort

| Task | Estimate | Complexity |
|------|----------|-----------|
| Audit and plan | 0.5h | Low |
| Create package + contracts | 1h | Low |
| Move adapters/logic | 2h | Medium |
| Create artifacts | 1h | Low |
| Update runtime | 1.5h | Medium |
| Migrate consumers | 2h | Medium |
| Remove old code | 1h | Low |
| Documentation | 2h | Low |
| Verification | 1h | Low |
| **Total** | **12h** | **Medium** |

Similar to Phase 2, slightly larger due to more adapters (SAM2, SAM, grounding).

---

## Prerequisite

Phase 2 (semantic_scoring) must be complete and committed before Phase 3 begins.

✅ **Phase 2 status:** COMPLETE (commit b58b222)  
✅ **Ready to begin Phase 3:** YES

---

## Ownership Checklist (Target)

After Phase 3:

```
Q: Who owns Region2D?
A: contextmap.region_perception.contracts

Q: Who owns region discovery adapters?
A: contextmap.region_perception.adapters

Q: Who owns region discovery execution?
A: contextmap.region_perception (execution layer)

Q: Who composes region discovery pipelines?
A: contextmap.runtime

Q: Do region discovery adapters live in visual_perception?
A: NO

Q: Does PerceptionResult reference Region2D?
A: YES (from region_perception)

Q: Are region audit reports contractual?
A: NO (debug/, not outputs/)
```

---

## Ready to Proceed

Phase 3 audit complete. Ready for implementation step 1: Create region_perception package.

Proceed with implementation? [Y/N]
