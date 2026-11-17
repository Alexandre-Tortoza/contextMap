"""A política versionada que decide quando uma hipótese pode ser exposta como primária.

`derive_ambiguity_state` diz se *algo compete* com uma hipótese, nunca *quanto suporte* ela tem.
Sem um gate, um único frame com uma única claim produz um estado não ambíguo e, com ele, uma
primária e o atributo `class` — uma observação isolada promovida a verdade persistente, o que o
AGENTS §4 proíbe.

A v1 (`unambiguous-single-hypothesis-v1`) expõe a primária sempre que o estado é não ambíguo. A v2
acrescenta um mínimo de observações físicas distintas; abaixo dele a primária é **retida** e o
estado registra o motivo em vez de simplesmente não ter primária. O default do mínimo é 1, que não
filtra nada e reproduz a v1 exatamente.

Este módulo decide a retenção; quem monta o estado é `state_mapping.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.semantic_fusion import FusedEvidence
from contextmap.semantic_mapping.invariants import require_count_of_at_least_one
from contextmap.semantic_mapping.semantic_state import (
    EntityHypothesis,
    WithheldPrimaryHypothesis,
)

PRIMARY_HYPOTHESIS_POLICY_ID = "unambiguous-single-hypothesis-v1"
"""Versioned identity of the rule that decides when a primary hypothesis is exposed."""

MIN_SUPPORT_PRIMARY_HYPOTHESIS_POLICY_ID = "unambiguous-single-hypothesis-v2"
"""Versioned identity of the v1 rule plus a minimum of independent support behind the primary."""


@dataclass(frozen=True, kw_only=True)
class MinimumSupportPrimaryPolicy:
    """Explicit configuration of ``unambiguous-single-hypothesis-v2``.

    The v2 rule is the v1 rule plus a gate: the hypothesis of an unambiguous state is exposed as
    primary only when at least ``min_supporting_physical_observations`` distinct physical
    observations support it, counted by
    :meth:`~contextmap.semantic_fusion.FusedEvidence.supporting_physical_observations`, so repeated
    inference over one frame or one claim over many geometry points never counts twice.

    Attributes:
        min_supporting_physical_observations: Distinct physical observations that must support
            the hypothesis before it is exposed. The default, ``1``, gates nothing, because every
            hypothesis has a supporting claim and therefore a physical observation, and so
            reproduces v1. A larger value is a scientific choice that needs a reference set to be
            tuned on; see
            ``src/contextmap/semantic_mapping/docs/primary-hypothesis-policy.md``.
    """

    min_supporting_physical_observations: int = 1

    def __post_init__(self) -> None:
        """Validate the minimum.

        Raises:
            ValueError: If the minimum is not an integer of at least one.
        """
        require_count_of_at_least_one(
            "min_supporting_physical_observations", self.min_supporting_physical_observations
        )


def withheld_for_lack_of_support(
    candidate: EntityHypothesis, evidence: FusedEvidence, policy: MinimumSupportPrimaryPolicy
) -> WithheldPrimaryHypothesis | None:
    """The record that withholds an unambiguous hypothesis, or ``None`` when its support suffices.

    The support is the set of distinct physical frames whose claims support the hypothesis, as
    Semantic Fusion counts it, never the number of inferences or claims.
    """
    supporting = evidence.supporting_physical_observations(candidate.hypothesis_id)
    if len(supporting) >= policy.min_supporting_physical_observations:
        return None
    return WithheldPrimaryHypothesis(
        hypothesis=candidate.ref, supporting_physical_observations=supporting
    )
