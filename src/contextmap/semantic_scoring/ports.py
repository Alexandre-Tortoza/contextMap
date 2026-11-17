"""Scoring backend port definition.

Este módulo define o contrato que adapters de scoring devem implementar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextmap.semantic_interpretation import (
    SemanticClaim,
)
from contextmap.semantic_scoring.contracts import SemanticScore
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    VisualFeature,
)


# NOTE: runtime_checkable é contrato, não conveniência. O runtime valida o backend com
# isinstance para recusá-lo com um erro de domínio ("semantic_scorer backend must implement
# ..."); sem o decorator, o isinstance levanta "Instance and class checks can only be used
# with @runtime_checkable protocols" e a mensagem perde o slot e o tipo recebido.
@runtime_checkable
class SemanticScorer(Protocol):
    """Backend that scores compatibility between semantic claims and visual features.

    A scorer receives batches of claims and features, computes similarity/compatibility,
    and returns SemanticScore objects. Batch processing is the primary interface because:

    - Text encoders process multiple hypotheses efficiently in a single forward pass
    - Batch matching avoids redundant text encoding calls
    - Actual implementations leverage batch operations for performance

    Different scorers may use different metrics (cosine similarity, embedding similarity, etc.).
    A claim with no compatible feature in the requested set is simply omitted from the output.

    Scoring never mutates a claim; it produces separate
    :class:`~contextmap.semantic_scoring.contracts.SemanticScore` records.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every score this backend produces. Without it a
            score could not say which model, checkpoint and configuration judged it, and
            two scorers' numbers would be indistinguishable in the artifact.
        """
        ...

    def score(
        self,
        claims: Sequence[SemanticClaim],
        features: Sequence[VisualFeature],
    ) -> Sequence[SemanticScore]:
        """Score compatibility between semantic claims and visual features in batch.

        Processes multiple claims and features together, matching each claim against
        compatible features (filtered by embedding space, scope, and spatial anchoring).

        Args:
            claims: Semantic claims to score. May be empty.
            features: Visual features to score against. May be empty.

        Returns:
            Sequence of SemanticScore objects, one per (claim, feature) pair that is
            both spatially compatible and in the scorer's embedding space. If a claim
            has no compatible feature, it does not appear in the output. Empty sequences
            are possible and valid.

        Raises:
            ValueError: If a claim or feature has invalid properties that prevent scoring,
                or if the data is structurally inconsistent (e.g., embedding space mismatch,
                normalization violation, ambiguous feature selection).
        """
        ...


# NOTE: `release()` não entra neste port. Liberar recursos de modelo é o contrato de
# ``ReleasableBackend``, que o runtime consome e adapters de várias capabilities
# implementam; exigi-lo aqui fazia um scorer puro — como o de similaridade por cosseno,
# que não carrega modelo próprio — deixar de satisfazer seu próprio port, e o estágio de
# scoring falhava com "backend must implement SemanticScorer". É a regra de ISP do
# AGENTS.md §12: o port pede o que o consumidor precisa, nada além.
