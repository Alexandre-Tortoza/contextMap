"""Como a evidência fundida de um suporte se torna o estado semântico de uma entidade.

Semantic Fusion produz crença em formação: todas as hipóteses, a evidência de cada uma e os
conflitos entre observações, sem nenhum vencedor. Este módulo transforma isso no estado persistente
de uma entidade **sem destruir nada**: toda hipótese, com sua evidência e seus frames de apoio, e
todo registro de incerteza voltam exatamente como foram fundidos.

O que ele acrescenta são os atributos derivados sob regras versionadas — `class`, a partir da
primária, e os pares declarados pelas claims — e nada além disso. Quem decide se existe primária é
`primary_hypothesis.py`; quem escala o fator de interação é `interaction_radius.py`.
"""

from __future__ import annotations

from contextmap.semantic_fusion import (
    EvidenceReference,
    EvidenceStance,
    FusedEvidence,
)
from contextmap.semantic_mapping.primary_hypothesis import (
    MIN_SUPPORT_PRIMARY_HYPOTHESIS_POLICY_ID,
    PRIMARY_HYPOTHESIS_POLICY_ID,
    MinimumSupportPrimaryPolicy,
    withheld_for_lack_of_support,
)
from contextmap.semantic_mapping.semantic_state import (
    AmbiguityState,
    AttributeOrigin,
    EntityAttribute,
    EntityHypothesis,
    EntitySemanticState,
    EntityUncertainty,
    SemanticStateProvenance,
    derive_ambiguity_state,
    sorted_attributes,
)

SEMANTIC_STATE_MAPPING_RULE_ID = "fused-evidence-semantic-state-v1"
"""Versioned identity of the mapping from fused evidence to semantic state."""

CLASS_ATTRIBUTE_DERIVATION_ID = "primary-hypothesis-label-v1"
"""Versioned identity of the rule that derives the ``class`` attribute."""

CLAIM_ATTRIBUTE_DERIVATION_ID = "claim-attribute-v1"
"""Versioned identity of the rule that turns what claims stated into ``OBSERVED`` attributes."""


def semantic_state_from_fused_evidence(
    evidence: FusedEvidence, *, primary_support: MinimumSupportPrimaryPolicy | None = None
) -> EntitySemanticState:
    """Map the evidence fused over one support into the semantic state of an entity.

    Nothing is dropped: the hypotheses, their evidence, their distinct supporting physical
    observations and the uncertainty records are kept as fused. A primary hypothesis and the
    ``class`` attribute are added only when the state is unambiguous and, under v2, enough
    distinct physical observations support the hypothesis; otherwise the state records why it
    withheld them. What the supporting claims stated (category and attributes) becomes
    ``OBSERVED`` attributes under ``claim-attribute-v1`` whether or not a primary is exposed.

    Args:
        evidence: The evidence fused over the support the entity is materialized from.
        primary_support: The minimum support of ``unambiguous-single-hypothesis-v2``, or
            ``None`` to decide the primary hypothesis under ``unambiguous-single-hypothesis-v1``,
            which has no minimum.

    Returns:
        The semantic state, whose ambiguity state is derived from the same records it keeps.
    """
    hypotheses = tuple(
        EntityHypothesis(
            fused_evidence_id=evidence.fused_evidence_id,
            hypothesis_id=item.hypothesis_id,
            label=item.label,
            evidence=item.evidence,
            supporting_physical_observations=evidence.supporting_physical_observations(
                item.hypothesis_id
            ),
        )
        for item in evidence.hypotheses
    )
    uncertainty = tuple(
        EntityUncertainty(fused_evidence_id=evidence.fused_evidence_id, record=record)
        for record in evidence.uncertainty
    )
    ambiguity_state = derive_ambiguity_state(hypotheses, uncertainty)
    primary = hypotheses[0] if ambiguity_state is AmbiguityState.UNAMBIGUOUS else None
    withheld = None
    if primary is not None and primary_support is not None:
        withheld = withheld_for_lack_of_support(primary, evidence, primary_support)
        if withheld is not None:
            primary = None
    return EntitySemanticState(
        hypotheses=hypotheses,
        ambiguity_state=ambiguity_state,
        provenance=_provenance(primary_support),
        primary_hypothesis=None if primary is None else primary.ref,
        attributes=sorted_attributes(
            (*(() if primary is None else (_class_attribute(primary),)), *_observed(evidence))
        ),
        uncertainty=uncertainty,
        withheld_primary=withheld,
    )


def _provenance(primary_support: MinimumSupportPrimaryPolicy | None) -> SemanticStateProvenance:
    """The rules the state is derived under, with the minimum the primary policy required."""
    if primary_support is None:
        return SemanticStateProvenance(
            mapping_rule_id=SEMANTIC_STATE_MAPPING_RULE_ID,
            primary_policy_id=PRIMARY_HYPOTHESIS_POLICY_ID,
        )
    return SemanticStateProvenance(
        mapping_rule_id=SEMANTIC_STATE_MAPPING_RULE_ID,
        primary_policy_id=MIN_SUPPORT_PRIMARY_HYPOTHESIS_POLICY_ID,
        min_supporting_physical_observations=primary_support.min_supporting_physical_observations,
    )


def _observed(evidence: FusedEvidence) -> tuple[EntityAttribute, ...]:
    """One ``OBSERVED`` attribute per ``(name, value)`` the supporting claims stated.

    A value stated by claims of several hypotheses is one property of the entity, citing every one
    of those claims; different values of one name stay separate attributes.
    """
    cited: dict[tuple[str, str], set[EvidenceReference]] = {}
    for hypothesis in evidence.hypotheses:
        for item in hypothesis.observed_attributes:
            cited.setdefault((item.name, item.value), set()).update(item.evidence)
    return tuple(
        EntityAttribute(
            name=name,
            value=value,
            origin=AttributeOrigin.OBSERVED,
            derivation_id=CLAIM_ATTRIBUTE_DERIVATION_ID,
            evidence=tuple(
                sorted(
                    cited[(name, value)],
                    key=lambda ref: (ref.contribution_id, ref.claim_id or ""),
                )
            ),
        )
        for name, value in sorted(cited)
    )


def _class_attribute(primary: EntityHypothesis) -> EntityAttribute:
    """The ``class`` attribute, citing exactly the claims that support the primary hypothesis."""
    return EntityAttribute(
        name="class",
        value=primary.label,
        origin=AttributeOrigin.DERIVED,
        derivation_id=CLASS_ATTRIBUTE_DERIVATION_ID,
        evidence=tuple(
            EvidenceReference(contribution_id=item.contribution_id, claim_id=item.claim_id)
            for item in primary.evidence
            if item.stance is EvidenceStance.SUPPORTING
        ),
    )
