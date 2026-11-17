"""Deterministic identities for visual features produced in one visual evidence run."""

from __future__ import annotations

import hashlib

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception.visual_features import FeatureId


def feature_id_for(
    *, run_id: str, source_observation_id: SourceObservationId, producer_id: str, index: int
) -> FeatureId:
    """Identify a feature by visual run, physical observation, producer, and output index.

    Args:
        run_id: Immutable visual evidence execution identity.
        source_observation_id: Physical observation processed by this execution.
        producer_id: Explicit feature extraction stage identity.
        index: Zero-based deterministic position within this stage's frame output.

    Returns:
        An identity qualified by the execution that owns the feature.

    Raises:
        ValueError: If an identity is empty or the index is negative.
    """
    if not run_id or not source_observation_id or not producer_id or index < 0:
        raise ValueError("feature identity needs non-empty producers and a non-negative index")
    producer_digest = hashlib.sha256(producer_id.encode("utf-8")).hexdigest()
    return FeatureId(
        f"{run_id}--{source_observation_id}--feature-stage-{producer_digest}-{index:04d}"
    )
