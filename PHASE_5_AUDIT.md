# Phase 5: Semantic Interpretation Extraction — Audit

**Target Capability:** semantic_interpretation  
**Source:** visual_perception  
**Status:** Pre-extraction analysis  
**Date:** 2026-10-01

---

## Current State Analysis

### What is semantic_interpretation?

Semantic interpretation produces semantic claims (labels, attributes) for visual regions based on VLM inference.

**Current ownership:** visual_perception.semantics/*, visual_perception.adapters/semantics/*

**Conceptual flow:**
```
PreparedImage + Region2D + VisualFeature
    ↓
semantic_claim_generation (Qwen, Gemini, Florence2, Eagle2.5)
    ↓
SemanticClaim (hypothesis with evidence)
```

---

## Scope: What Moves to semantic_interpretation

### Contracts (from visual_perception.models)

- `SemanticClaim` (dataclass with text, attributes, hypothesis role, source evidence)
- `ClaimId` (NewType identifier)
- `HypothesisRole` enum (PRIMARY, ALTERNATIVE)
- `SemanticAttribute` (key-value semantic property)
- `SemanticInferenceProvenance` (VLM model, prompt, configuration)

### Ports

- `SemanticInterpreter` protocol (interpret regions/features → claims)

### Adapters

- Qwen semantic interpreter
- Gemini semantic interpreter
- Florence2 semantic interpreter
- Eagle2.5 semantic interpreter

### Services

- Semantic claim generation
- Hypothesis role assignment
- Evidence reference tracking

### Artifacts

- SemanticInterpretationRunManifest
- SemanticInterpretationRunWriter
- SemanticInterpretationRunReader

---

## Scope: What Stays in visual_perception

**NOT moving:**
- PerceptionResult (aggregate, stays in visual_perception)
- Region2D, VisualFeature (owned by their capabilities)
- PreparedImage (owned by visual_input)

---

## Dependencies and Constraints

### Inbound

- visual_perception (needs SemanticClaim for PerceptionResult)
- semantic_fusion (reads claims for fusion)
- semantic_mapping (uses claims for entity semantics)

### Outbound

- contextmap.region_perception (Region2D, for claim scope)
- contextmap.visual_features (VisualFeature, for evidence reference)
- contextmap.visual_perception (PreparedImage for inference)

---

## Phase 5 Steps (Plan)

1. Create semantic_interpretation package
2. Extract contracts (SemanticClaim, ClaimId, etc.)
3. Move adapters (Qwen, Gemini, Florence2, Eagle2.5)
4. Implement artifacts
5. Update runtime DAG
6. Migrate consumers
7. Delete old code from visual_perception
8. Documentation
9. Testing

---

## Ready to Proceed

Phase 5 audit complete.
