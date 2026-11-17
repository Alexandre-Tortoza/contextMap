"""CLIP-based semantic scorers."""

from contextmap.semantic_scoring.adapters._scoring_base import _CosineSemanticScorer
from contextmap.visual_perception import FeatureScope


class ClipSemanticScorer(_CosineSemanticScorer):
    """Score scene claims against global CLIP features."""

    _scope = FeatureScope.GLOBAL


class ClipRegionSemanticScorer(_CosineSemanticScorer):
    """Score each region claim against the compatible region feature in the CLIP space.

    The crop feature ignores the region mask (the background inside the box enters it), so its
    scores are a distinct evidence channel from the mask-conditioned AlphaCLIP ones and never
    share their embedding space. Region features from other spaces (for example a pooled DINO
    appearance feature of the same region) are independent evidence and are ignored by this
    scorer rather than treated as an error.
    """

    _scope = FeatureScope.REGION
