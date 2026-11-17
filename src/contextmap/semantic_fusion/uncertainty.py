"""As regras que reportam o que a evidência deixou indeciso, sem resolver nada.

O baseline nunca escolhe entre hipóteses. Ele nomeia a indecisão e a evidência exata
atrás dela:

* ``CONTRADICTION`` -- observações físicas distintas apoiam hipóteses incompatíveis com
  claims primárias. Precisa de ao menos dois frames: duas execuções sobre o mesmo frame
  discordando é inferência correlacionada, não contradição do mundo;
* ``AMBIGUITY`` -- hipóteses competem sem que uma contradição entre observações explique
  (alternativas de uma interpretação, ou runs que discordam sobre um frame);
* ``NEAR_TIE`` -- as contagens de **observações físicas distintas** que apoiam as
  hipóteses líderes caem dentro da margem configurada. A comparação é só de contagens,
  nunca de scores;
* ``INSUFFICIENT_EVIDENCE`` -- não existe hipótese alguma.

Todo registro cita a regra versionada que o declarou e as contribuições e claims exatas.
Nada aqui pondera por qualidade, combina score ou descarta evidência.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from contextmap.semantic_fusion.evidence import EvidenceContribution, EvidenceReference
from contextmap.semantic_fusion.fused_evidence import (
    EvidenceStance,
    FusedHypothesis,
    UncertaintyKind,
    UncertaintyRecord,
    uncertainty_key,
)
from contextmap.semantic_fusion.hypotheses import ViewClaim, reference_of
from contextmap.semantic_interpretation import HypothesisRole


def report_uncertainty(
    hypotheses: Sequence[FusedHypothesis],
    claims: Sequence[ViewClaim],
    contributions: Sequence[EvidenceContribution],
    near_tie_margin: int,
    *,
    policy_id: str,
) -> tuple[UncertaintyRecord, ...]:
    """Report, never resolve, what the evidence leaves undecided.

    Args:
        hypotheses: The hypotheses formed over the support, in canonical order.
        claims: Every claim of every view, abstentions included.
        contributions: The contributions of the support, needed to find the claimless views.
        near_tie_margin: Largest difference in supporting physical observations that still
            counts as a tie.
        policy_id: The versioned policy that owns these rules; every ``rule_id`` is derived
            from it, so a record always names the rule that declared it.

    Returns:
        The records, in canonical order. Empty only when one hypothesis stands alone.
    """
    rule = policy_id
    if not hypotheses:
        claimless = [item for item in contributions if not item.claim_refs]
        references = [reference_of(item) for item in claims] + [
            EvidenceReference(contribution_id=item.contribution_id) for item in claimless
        ]
        return (
            UncertaintyRecord(
                kind=UncertaintyKind.INSUFFICIENT_EVIDENCE,
                hypothesis_ids=(),
                evidence=_sorted_references(references),
                rule_id=f"{rule}/insufficient-evidence",
            ),
        )

    by_reference = {(item.contribution_id, item.claim.claim_id): item for item in claims}
    supporting = {
        hypothesis.hypothesis_id: [
            by_reference[(item.contribution_id, item.claim_id)]
            for item in hypothesis.evidence
            if item.stance is EvidenceStance.SUPPORTING
        ]
        for hypothesis in hypotheses
    }
    primary = {
        hypothesis_id: [item for item in items if item.claim.role is HypothesisRole.PRIMARY]
        for hypothesis_id, items in supporting.items()
    }
    candidates = [
        hypothesis.hypothesis_id for hypothesis in hypotheses if primary[hypothesis.hypothesis_id]
    ]

    records: list[UncertaintyRecord] = []
    primary_observations = {
        item.physical_observation_id
        for hypothesis_id in candidates
        for item in primary[hypothesis_id]
    }
    if len(candidates) >= 2 and len(primary_observations) >= 2:
        records.append(
            UncertaintyRecord(
                kind=UncertaintyKind.CONTRADICTION,
                hypothesis_ids=tuple(sorted(candidates)),
                evidence=_sorted_references(
                    [
                        reference_of(item)
                        for hypothesis_id in candidates
                        for item in primary[hypothesis_id]
                    ]
                ),
                rule_id=f"{rule}/contradiction",
            )
        )
    elif len(hypotheses) >= 2:
        records.append(
            UncertaintyRecord(
                kind=UncertaintyKind.AMBIGUITY,
                hypothesis_ids=tuple(sorted(hypothesis.hypothesis_id for hypothesis in hypotheses)),
                evidence=_sorted_references(
                    [reference_of(item) for items in supporting.values() for item in items]
                ),
                rule_id=f"{rule}/ambiguity",
            )
        )

    if len(candidates) >= 2:
        counts = {
            hypothesis_id: len({item.physical_observation_id for item in supporting[hypothesis_id]})
            for hypothesis_id in candidates
        }
        leading = max(counts.values())
        tied = [
            hypothesis_id
            for hypothesis_id in candidates
            if leading - counts[hypothesis_id] <= near_tie_margin
        ]
        if len(tied) >= 2:
            records.append(
                UncertaintyRecord(
                    kind=UncertaintyKind.NEAR_TIE,
                    hypothesis_ids=tuple(sorted(tied)),
                    evidence=_sorted_references(
                        [
                            reference_of(item)
                            for hypothesis_id in tied
                            for item in supporting[hypothesis_id]
                        ]
                    ),
                    rule_id=f"{rule}/near-tie/margin-{near_tie_margin}",
                )
            )
    return tuple(sorted(records, key=uncertainty_key))


def _sorted_references(references: Iterable[EvidenceReference]) -> tuple[EvidenceReference, ...]:
    unique = {(ref.contribution_id, ref.claim_id or ""): ref for ref in references}
    return tuple(unique[key] for key in sorted(unique))
