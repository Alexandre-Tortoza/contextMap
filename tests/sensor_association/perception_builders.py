"""Deterministic Visual Perception evidence for the association tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, InlineMask, Region2D, RegionId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    PerceptionResult,
)
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

RUN_ID = "run-0001"
OBSERVATION_ID = SourceObservationId("frame-0001")
SEQUENCE_ID = "sequence-0001"

_REGION_BACKEND = BackendProvenance(
    backend_id="fake_region_discovery",
    capability="region_discovery",
    provider="fake",
    model="fake",
    version="0",
)


def rect_mask(width: int, height: int, x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    """A mask that is foreground on ``x0 <= x < x1`` and ``y0 <= y < y1``."""
    return InlineMask(
        np.array(
            tuple(x0 <= x < x1 and y0 <= y < y1 for y in range(height) for x in range(width)),
            dtype=bool,
        ).reshape(height, width)
    )


def make_region(
    region_id: str,
    mask: InlineMask | None,
    *,
    accepted: bool = True,
    size: tuple[int, int] | None = None,
    box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
) -> Region2D:
    """A region in the space of ``mask`` (or of ``size`` for a box-only region).

    ``box`` is ``(x, y, width, height)`` in prepared pixels; it matters to the box rule.
    """
    width, height = (mask.width, mask.height) if mask is not None else size or (640, 480)
    x, y, box_width, box_height = box
    bounding_box = BoundingBox2D(x=x, y=y, width=box_width, height=box_height)
    return Region2D(
        region_id=RegionId(region_id),
        source_observation_id=OBSERVATION_ID,
        bounding_box=bounding_box,
        provenance=_REGION_BACKEND,
        mask=mask,
        image_width=width,
        image_height=height,
        is_accepted=accepted,
        rejection_reason=None if accepted else "fixture rejection",
    )


def make_feature(
    feature_id: str, scope: FeatureScope, *, region_id: str | None = None, space: str = "dinov2:b14"
) -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        source_observation_id=OBSERVATION_ID,
        scope=scope,
        embedding_space_id=space,
        shape=(4,),
        dtype="float32",
        payload_reference=f"features/{feature_id}.npy",
        provenance=BackendProvenance(
            backend_id="fake_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0",
        ),
        region_id=None if region_id is None else RegionId(region_id),
    )


def make_claim(
    claim_id: str,
    *,
    region_id: str | None = None,
    observation_id: SourceObservationId = OBSERVATION_ID,
    semantic_run_id: str = "semantic-run-0001",
) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(claim_id),
        source_observation_id=observation_id,
        semantic_interpretation_run_id=SemanticInterpretationRunId(semantic_run_id),
        hypothesis="a hypothesis, not a label",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake_interpreter",
                capability="semantic_interpreter",
                provider="fake",
                model="fake",
                version="0",
            ),
            task_identity="task",
            prompt_template_id="prompt-v1",
            output_schema_version="schema-v1",
        ),
        region_id=None if region_id is None else RegionId(region_id),
    )


def make_result(
    regions: Sequence[Region2D],
    *,
    features: Sequence[VisualFeature] = (),
    claims: Sequence[SemanticClaim] = (),
    observation_id: SourceObservationId = OBSERVATION_ID,
) -> PerceptionResult:
    return PerceptionResult(
        source_observation_id=observation_id,
        region_artifact_id=str(RUN_ID),
        visual_evidence_artifact_id="visual-run" if features else None,
        semantic_interpretation_artifact_id=str(claims[0].semantic_interpretation_run_id)
        if claims
        else None,
        sequence_artifact_id=SEQUENCE_ID,
        regions=tuple(replace(region, source_observation_id=observation_id) for region in regions),
        features=tuple(
            replace(feature, source_observation_id=observation_id) for feature in features
        ),
        claims=tuple(claims),
    )
