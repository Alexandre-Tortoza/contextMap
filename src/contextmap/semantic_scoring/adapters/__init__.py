"""Adapters de scoring semântico: as implementações CLIP e AlphaCLIP.

As torres de texto (``clip_text``, ``alphaclip_text``) também vivem aqui, e não entre as
features visuais: elas codificam a hipótese de um claim para compará-la com a feature, o que
é scoring, e consomem a infra de template deste pacote. Enquanto estavam em
``visual_features`` fechavam um ciclo entre as duas capabilities.

Elas não são reexportadas neste ``__init__``: cada uma carrega a torre visual correspondente
para declarar o embedding space, e um reexport eager traria esse peso para todo import deste
pacote. Quem precisa delas — o composition root — importa seu módulo direto.
"""

from contextmap.semantic_scoring.adapters._scoring_base import (
    TEXT_TEMPLATE_FIELD,
    FeaturePayloadSource,
    SemanticScoringError,
    TextEmbeddingBatch,
    TextEmbeddingEncoder,
    render_text,
    validate_text_template,
)
from contextmap.semantic_scoring.adapters.alphaclip import AlphaClipSemanticScorer
from contextmap.semantic_scoring.adapters.clip import (
    ClipRegionSemanticScorer,
    ClipSemanticScorer,
)

__all__ = [
    "TEXT_TEMPLATE_FIELD",
    "AlphaClipSemanticScorer",
    "ClipRegionSemanticScorer",
    "ClipSemanticScorer",
    "FeaturePayloadSource",
    "SemanticScoringError",
    "TextEmbeddingBatch",
    "TextEmbeddingEncoder",
    "render_text",
    "validate_text_template",
]
