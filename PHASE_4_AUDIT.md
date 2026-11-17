# Phase 4: Visual Features Extraction — Audit

**Target Capability:** visual_features  
**Source:** visual_perception  
**Status:** Pre-extraction analysis  
**Date:** 2026-10-01

---

## Current State Analysis

### What is visual_features?

Visual feature extraction produces dense and region-scoped visual embeddings.

**Current ownership:** visual_perception.features/*, visual_perception.adapters/features/*

**Conceptual flow:**
```
PreparedImage + Region2D
    ↓
dense_feature_extraction (DINOv2, DINOv3, SigLIP2)
    ↓
region_feature_pooling (aggregate dense features per region)
    ↓
VisualFeature (evidence)
```

---

## Scope: What Moves to visual_features

### Contracts (from visual_perception.models)

**Core types:**
- `VisualFeature` (dataclass with embedding, region/frame scope, pooling metadata)
- `FeatureId` (NewType identifier)
- `FeatureScope` enum (GLOBAL, REGION, FRAME)
- `EmbeddingSpace` (model, dimension, normalization)
- `PoolingMethod` enum (AVERAGE, MAX, etc.)

### Ports (from visual_perception.ports)

- `DenseFeatureExtractor` protocol
- `RegionFeatureAggregator` protocol

### Adapters (from visual_perception/adapters/features/)

- DINOv2 dense feature extractor
- DINOv3 dense feature extractor
- SigLIP2 dense feature extractor
- CLIP region features
- AlphaCLIP region features
- Region feature pooling implementations

### Services/Execution

- Dense feature extraction pipeline
- Region feature aggregation
- Feature pooling and normalization

### Artifacts

- VisualFeaturesRunManifest
- VisualFeaturesRunWriter
- VisualFeaturesRunReader

---

## Scope: What Stays in visual_perception

**NOT moving:**
- PreparedImage (input contract, owned by visual_input)
- Region2D (now owned by region_perception)
- SemanticClaim (owned by semantic_interpretation)
- PerceptionResult (aggregate)
- Semantic interpretation logic

---

## Dependencies and Constraints

### Inbound dependencies (who uses visual_features)

- visual_perception (needs VisualFeature for PerceptionResult)
- semantic_fusion (reads features for fusion)
- semantic_mapping (uses features for entity grounding)

### Outbound dependencies (what visual_features depends on)

- contextmap.ingestion (SourceObservationId, timestamps)
- contextmap.shared (Vector embeddings)
- contextmap.region_perception (Region2D for region-scoped features)
- contextmap.visual_perception (PreparedImage input)

---

## Known Complexities

### 1. Embedding Space Compatibility

Different backends produce embeddings in different spaces. CLIP, DINOv2, SigLIP2 have different dimensions and normalization.

**Solution:** Explicit EmbeddingSpace contract with model, dimension, normalization semantics.

### 2. Region Feature Pooling

Region features must aggregate dense features per region. Requires dense feature map + region mask/bounds.

**Current code:** region_feature_pooling.py in visual_perception. Moves to visual_features.

### 3. Feature Scope

Some features are global (image-wide), some are region-scoped. Feature metadata must declare scope.

**Solution:** FeatureScope enum (GLOBAL, REGION, FRAME).

---

## Phase 4 Extraction Steps (Plan)

### Step 1: Audit (this document)
- ✓ Identify contracts, ports, adapters
- ✓ Identify dependencies

### Step 2: Create visual_features package
- Create src/contextmap/visual_features/
- Create contracts, ports, artifacts
- Extract VisualFeature, FeatureId, FeatureScope

### Step 3: Move adapters and logic
- Move dense feature extractors (DINOv2, DINOv3, SigLIP2)
- Move region feature extractors (CLIP, AlphaCLIP)
- Move pooling logic

### Step 4: Create artifacts
- VisualFeaturesRunManifest
- VisualFeaturesRunWriter
- VisualFeaturesRunReader

### Step 5: Update runtime
- Update catalog.py (new component IDs)
- Update composition.py (factories, stage definition)

### Step 6: Migrate consumers
- Update visual_perception (imports VisualFeature from visual_features)
- Update semantic_fusion (imports VisualFeature)
- Update semantic_mapping (imports VisualFeature)
- Update tests

### Step 7: Remove old code
- Delete from visual_perception/features/
- Delete from visual_perception/adapters/features/
- Update visual_perception.__init__.py

### Step 8: Documentation
- README
- code-map

### Step 9: Testing
- Contract tests
- Artifact tests
- Integration tests

---

## Effort Estimate

| Task | Estimate |
|------|----------|
| Audit and plan | 0.5h |
| Create package + contracts | 1h |
| Move adapters/logic | 2h |
| Create artifacts | 1h |
| Update runtime | 1.5h |
| Migrate consumers | 1.5h |
| Remove old code | 1h |
| Documentation | 1.5h |
| Testing | 2h |
| **Total** | **12h** |

---

## Ready to Proceed

Phase 4 audit complete. Ready for extraction.
