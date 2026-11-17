"""Semantic scoring evidence contracts.

Este módulo centraliza os contratos de scoring de cross-modal compatibilidade
entre claims semânticos e visual features. Um score não é automaticamente
probabilidade; a semântica é definida pelo score_type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, NewType

from contextmap.ingestion import SourceObservationId

if TYPE_CHECKING:
    from contextmap.region_perception import RegionId
    from contextmap.semantic_interpretation import (
        ClaimId,
    )
    from contextmap.shared import BackendProvenance
    from contextmap.visual_perception import (
        FeatureId,
    )

ScoreId = NewType("ScoreId", str)
"""Identity of one semantic scoring judgement in a scoring execution."""


class SemanticScoreType(Enum):
    """Numerical semantics of a SemanticScore value."""

    COSINE_SIMILARITY = "cosine_similarity"


@dataclass(frozen=True, kw_only=True)
class SemanticScore:
    """One scorer's support judgement for a claim and visual feature.

    ``value`` keeps the scale named by ``score_type``. In particular, cosine
    similarity is not remapped to ``[0, 1]`` and is never presented as a
    probability. A claim with no score remains valid and is represented by the
    absence of a ``SemanticScore`` record, not by a zero value.
    """

    score_id: ScoreId
    claim_id: ClaimId
    feature_id: FeatureId
    score_type: SemanticScoreType
    value: float
    calibrated_probability: float | None
    embedding_space_id: str
    source_observation_id: SourceObservationId
    region_id: RegionId | None
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        """Validate numerical semantics and scorer provenance."""
        for field_name, value in (
            ("score_id", str(self.score_id)),
            ("claim_id", str(self.claim_id)),
            ("feature_id", str(self.feature_id)),
            ("embedding_space_id", self.embedding_space_id),
            ("source_observation_id", str(self.source_observation_id)),
            ("scorer_id", self.provenance.backend_id),
            ("scorer_model", self.provenance.model),
            ("scorer_version", self.provenance.version),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not isfinite(self.value):
            raise ValueError("semantic score value must be finite")
        if self.score_type is SemanticScoreType.COSINE_SIMILARITY and not -1.0 <= self.value <= 1.0:
            raise ValueError("cosine similarity must be in [-1, 1]")
        if self.calibrated_probability is not None and (
            not isfinite(self.calibrated_probability)
            or not 0.0 <= self.calibrated_probability <= 1.0
        ):
            raise ValueError("calibrated_probability must be finite and in [0, 1]")
        if self.provenance.capability != "semantic_scorer":
            raise ValueError("semantic score provenance capability must be semantic_scorer")
