"""Semantic scoring capability: judge compatibility between claims and features.

Semantic Scoring is a first-level capability that scores cross-modal
compatibility between semantic claims (produced by semantic interpretation)
and visual features (produced by visual features capability).

A score is NOT automatically a probability. Scores preserve:
- Score type and semantics (e.g., cosine_similarity is [-1, 1])
- Embedding space identity
- Scorer/backend provenance
- Configuration identity
- Upstream run identity

Scores from incompatible embedding spaces must not be compared.
Absence of a score (None) is not equivalent to zero.
A score never modifies the SemanticClaim that prompted it.
"""

from contextmap.semantic_scoring.artifacts import (
    SemanticScoringLineage,
    SemanticScoringRunError,
    SemanticScoringRunId,
    SemanticScoringRunManifest,
    SemanticScoringRunReader,
    SemanticScoringRunWriter,
    decode_semantic_score,
    encode_semantic_score,
)
from contextmap.semantic_scoring.contracts import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
)
from contextmap.semantic_scoring.ports import SemanticScorer

__all__ = [
    "ScoreId",
    "SemanticScore",
    "SemanticScoreType",
    "SemanticScorer",
    "SemanticScoringLineage",
    "SemanticScoringRunError",
    "SemanticScoringRunId",
    "SemanticScoringRunManifest",
    "SemanticScoringRunReader",
    "SemanticScoringRunWriter",
    "decode_semantic_score",
    "encode_semantic_score",
]
