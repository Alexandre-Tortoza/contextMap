"""Como as claims de várias vistas se tornam hipóteses concorrentes sobre um suporte.

Aqui vivem as regras do baseline que decidem **o que é a mesma hipótese** e **o que cada
claim diz sobre ela**:

* a **chave de label** agrupa claims: o texto sob normalização NFKC, case-folded, com
  espaços colapsados. É uma regra puramente tipográfica -- ``"Pallet"`` e ``"pallet "``
  são uma hipótese, ``"pallet"`` e ``"wooden pallet"`` são duas. Nenhum sinônimo, plural
  ou taxonomia é assumido;
* o **stance** diz como cada claim se relaciona com cada hipótese: apoia, contradiz, é
  ambígua (alternativa da mesma interpretação) ou se abstém;
* o **fator de interação** é uma propriedade semântica que o interpretador declarou sobre
  o objeto, agregada uma amostra por observação física -- nunca uma medição, nunca suporte
  para o label;
* os **atributos observados** são o que as claims apoiadoras afirmaram, verbatim, um par
  ``(nome, valor)`` por entrada, sem nada agregado.

Nenhuma regra aqui pontua, ordena ou elege hipótese, e nada aqui lê qualidade de
observação. A numeração das hipóteses segue a chave de label, então a ordem não implica
ranking.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import median

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.evidence import EvidenceReference, SupportSignal
from contextmap.semantic_fusion.fused_evidence import (
    INTERACTION_FACTOR_AGGREGATION_POLICY_ID,
    EvidenceStance,
    FusedAttribute,
    FusedHypothesis,
    FusedInteractionFactor,
    HypothesisEvidence,
)
from contextmap.semantic_fusion.identity import EvidenceContributionId, FusedHypothesisId
from contextmap.semantic_interpretation import ClaimId, HypothesisRole, SemanticClaim


@dataclass(frozen=True)
class ViewClaim:
    """One claim of one contribution, with the signals that came with it.

    Working value of the baseline policy: it pairs a claim with the view it came from and the
    label key it was reduced to, so hypothesis formation and uncertainty reporting read the
    same facts. It is never persisted.

    Attributes:
        contribution_id: The contribution that carries the claim.
        physical_observation_id: The physical frame the view interpreted.
        claim: The claim exactly as the interpreter produced it.
        key: The label key the claim was reduced to.
        abstains: Whether the policy declared this label an abstention.
        signals: The typed scores of the claim, in canonical order.
    """

    contribution_id: EvidenceContributionId
    physical_observation_id: SourceObservationId
    claim: SemanticClaim
    key: str
    abstains: bool
    signals: tuple[SupportSignal, ...]


def label_key(text: str) -> str:
    """Reduce a hypothesis text to the key under which the baseline compares labels.

    Args:
        text: The hypothesis text as proposed.

    Returns:
        The text under NFKC normalisation, case-folded, with whitespace collapsed. Only
        typographic differences are removed; no synonym or taxonomy is applied.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


INTERACTION_FACTOR_ATTRIBUTE = "interaction_factor"


def _interaction_factor(claim: SemanticClaim) -> float | None:
    """The interaction factor the interpreter attached to a claim, if any and well-formed.

    The factor is read from the claim attribute named ``interaction_factor``. It is a semantic
    property of the object, not support for the label, so it stays optional and open-vocabulary:
    a missing, non-numeric, non-finite or non-positive value yields ``None`` — the claim simply
    carried no usable factor. Clamping it and converting it to a radius belong to the consuming
    stage, never here.
    """
    for attribute in claim.attributes:
        if attribute.name != INTERACTION_FACTOR_ATTRIBUTE:
            continue
        value = attribute.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        factor = float(value)
        return factor if isfinite(factor) and factor > 0.0 else None
    return None


def _aggregate_interaction_factor(claims: Sequence[ViewClaim]) -> FusedInteractionFactor | None:
    """The interaction factor of a fused hypothesis, one sample per physical observation.

    The factor is a property of the object the label names, so a hypothesis carries one value.
    Claims that declared a usable factor are grouped by physical observation and each observation
    contributes the median of its own claims' factors, so a frame with three claims weighs the same
    as one with a single claim. The hypothesis value is the median of those per-observation
    factors, which keeps a single stray view from deciding it. The result cites only the claims
    that declared a factor. ``None`` when no claim carried a usable factor.
    """
    declared: dict[SourceObservationId, list[tuple[ViewClaim, float]]] = {}
    for item in claims:
        factor = _interaction_factor(item.claim)
        if factor is not None:
            declared.setdefault(item.physical_observation_id, []).append((item, factor))
    if not declared:
        return None
    per_observation = {
        observation: median(factor for _, factor in items)
        for observation, items in declared.items()
    }
    samples = sorted(per_observation.values())
    return FusedInteractionFactor(
        value=median(samples),
        aggregation_policy_id=INTERACTION_FACTOR_AGGREGATION_POLICY_ID,
        physical_observation_ids=tuple(sorted(per_observation)),
        evidence=tuple(
            sorted(
                {reference_of(item) for items in declared.values() for item, _ in items},
                key=lambda ref: (ref.contribution_id, ref.claim_id or ""),
            )
        ),
        minimum=samples[0],
        maximum=samples[-1],
    )


CATEGORY_ATTRIBUTE = "category"
"""Reserved observed-attribute name under which a claim's own ``category`` is carried."""


def _attribute_text(value: str | int | float | bool | None) -> str | None:
    """The value of a claim attribute as text, or ``None`` when it states no value.

    A string is kept verbatim; any other JSON scalar is written in its JSON spelling, so
    ``True`` is ``"true"`` and ``3`` is ``"3"``. A ``null`` or blank value states nothing about
    the object, so it is not an observation (the claim itself stays referenced by the evidence).
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value)
    return text if text.strip() else None


def _observed_attributes(supporters: Sequence[ViewClaim]) -> tuple[FusedAttribute, ...]:
    """What the supporting claims stated about the object, one entry per ``(name, value)``.

    Nothing is aggregated: two values of one name are two observations, each citing its own
    claims. The interaction factor is excluded because it already has its own typed treatment on
    the hypothesis; listing it again would give one statement two representations.
    """
    cited: dict[tuple[str, str], set[tuple[EvidenceContributionId, ClaimId]]] = {}
    for item in supporters:
        stated: list[tuple[str, str | int | float | bool | None]] = [
            (CATEGORY_ATTRIBUTE, item.claim.category)
        ]
        stated.extend(
            (attribute.name, attribute.value)
            for attribute in item.claim.attributes
            if attribute.name != INTERACTION_FACTOR_ATTRIBUTE
        )
        for name, value in stated:
            text = _attribute_text(value)
            if text is not None:
                cited.setdefault((name, text), set()).add(
                    (item.contribution_id, item.claim.claim_id)
                )
    return tuple(
        FusedAttribute(
            name=name,
            value=value,
            evidence=tuple(
                EvidenceReference(contribution_id=contribution, claim_id=claim)
                for contribution, claim in sorted(cited[(name, value)])
            ),
        )
        for name, value in sorted(cited)
    )


def build_hypotheses(claims: Sequence[ViewClaim]) -> tuple[FusedHypothesis, ...]:
    """Group the claims of every view into the competing hypotheses of one support.

    Claims are grouped by label key and numbered in that order, so the position of a
    hypothesis implies no ranking. Every claim is listed under every hypothesis with its
    stance, so a consumer sees what supported, contradicted or stayed ambiguous about each
    one. Abstentions form no hypothesis.

    Args:
        claims: Every claim of every view of the support, in any order.

    Returns:
        The hypotheses, sorted by identity. Empty when no view produced a usable claim.
    """
    ordered = sorted(claims, key=lambda item: (item.contribution_id, item.claim.claim_id))
    by_key: dict[str, list[ViewClaim]] = {}
    for item in ordered:
        if not item.abstains:
            by_key.setdefault(item.key, []).append(item)
    hypotheses: list[FusedHypothesis] = []
    for number, key in enumerate(sorted(by_key), start=1):
        supporters = {item.contribution_id for item in by_key[key]}
        hypotheses.append(
            FusedHypothesis(
                hypothesis_id=FusedHypothesisId(f"hypothesis-{number:04d}"),
                label=_spelling(by_key[key]),
                evidence=tuple(
                    HypothesisEvidence(
                        contribution_id=item.contribution_id,
                        claim_id=item.claim.claim_id,
                        stance=_stance(item, key, supporters),
                        role=item.claim.role,
                        signals=item.signals,
                    )
                    for item in ordered
                ),
                interaction_factor=_aggregate_interaction_factor(by_key[key]),
                observed_attributes=_observed_attributes(by_key[key]),
            )
        )
    return tuple(hypotheses)


def _stance(
    item: ViewClaim, key: str, supporting_contributions: set[EvidenceContributionId]
) -> EvidenceStance:
    if item.abstains:
        return EvidenceStance.ABSTAINING
    if item.key == key:
        return EvidenceStance.SUPPORTING
    if (
        item.contribution_id in supporting_contributions
        or item.claim.role is HypothesisRole.ALTERNATIVE
    ):
        return EvidenceStance.AMBIGUOUS
    return EvidenceStance.CONFLICTING


def _spelling(supporters: Sequence[ViewClaim]) -> str:
    """The most common whitespace-collapsed spelling among the claims of a label key."""
    counts = Counter(" ".join(item.claim.hypothesis.split()) for item in supporters)
    return min(counts, key=lambda spelling: (-counts[spelling], spelling))


def reference_of(item: ViewClaim) -> EvidenceReference:
    """Reference the exact claim of the exact view a working value came from.

    Args:
        item: The claim as the policy saw it.

    Returns:
        A reference naming both the contribution and the claim.
    """
    return EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim.claim_id)
