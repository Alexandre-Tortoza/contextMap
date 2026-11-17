"""AlphaCLIP-based semantic scorer."""

from contextmap.semantic_scoring.adapters._scoring_base import _CosineSemanticScorer
from contextmap.visual_perception import FeatureScope


class AlphaClipSemanticScorer(_CosineSemanticScorer):
    """Score region claims against AlphaCLIP features for the same frozen region."""

    _scope = FeatureScope.REGION
