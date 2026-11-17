"""A regra versionada que converte o fator de interação numa distância em metros.

O fator de interação é adimensional e vem da interpretação semântica, fundido a montante sobre as
claims de uma hipótese. Um raio em metros só existe quando esse fator é escalado pelo tamanho
**medido** da entidade — e é por isso que esta regra não mora junto do mapeamento de estado: a
extensão medida pertence ao resumo geométrico, não à evidência fundida de que o estado é
construído.

O resultado continua sendo uma hipótese semântica derivada, nunca uma medição: um estado sem
primária ou sem fator não recebe atributo nenhum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from contextmap.semantic_fusion import FusedInteractionFactor
from contextmap.semantic_mapping.fingerprint import canonical_digest
from contextmap.semantic_mapping.semantic_state import (
    AttributeOrigin,
    EntityAttribute,
    EntitySemanticState,
    sorted_attributes,
)
from contextmap.shared import Vector3

INTERACTION_RADIUS_DERIVATION_ID = "interaction-radius-from-factor-v1"
"""Versioned identity of the rule that derives ``interaction_radius_m`` from factor and size."""

EXTENT_RULE_HALF_MAX_EXTENT = "half_max_extent"
"""Extent rule of ``interaction-radius-from-factor-v1``: half the largest side of the bounds."""


@dataclass(frozen=True, kw_only=True)
class InteractionRadiusPolicy:
    """Versioned rule ``interaction-radius-from-factor-v1`` that turns a factor into a radius.

    The radius is ``extent_rule(extent_m) * clamp(factor, factor_min, factor_max)`` with the extent
    rule ``half_max_extent`` (half the entity's largest measured side), so a factor of ``1.0``
    reaches about the object's own radius. It is a derived semantic hypothesis, never a measurement.

    Attributes:
        factor_min: Floor the interpreter's factor is clamped to, finite and > 0. A factor below it
            is a degenerate estimate that would collapse the interaction zone to nothing.
        factor_max: Ceiling of the clamp, finite and >= ``factor_min``. It keeps one stray claim
            from blowing the zone up across the map.

    The default range is a scientific choice to be tuned against a reference set, not a measurement.
    """

    factor_min: float = 0.1
    factor_max: float = 5.0

    def __post_init__(self) -> None:
        """Validate the clamp range.

        Raises:
            ValueError: If a bound is not finite, the floor is not positive or the ceiling is
                below the floor.
        """
        if not (isfinite(self.factor_min) and isfinite(self.factor_max)):
            raise ValueError("factor_min and factor_max must be finite")
        if self.factor_min <= 0.0:
            raise ValueError(f"factor_min must be positive, got {self.factor_min!r}")
        if self.factor_max < self.factor_min:
            raise ValueError(
                f"factor_max must not be below factor_min, got {self.factor_max!r} < "
                f"{self.factor_min!r}"
            )

    @property
    def policy_id(self) -> str:
        """The versioned identity of the rule."""
        return INTERACTION_RADIUS_DERIVATION_ID

    def fingerprint(self) -> str:
        """Hash the rule identity and its parameters, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        return canonical_digest(
            {
                "policy_id": self.policy_id,
                "extent_rule": EXTENT_RULE_HALF_MAX_EXTENT,
                "factor_min": self.factor_min,
                "factor_max": self.factor_max,
            }
        )

    @property
    def derivation_id(self) -> str:
        """The derivation identity recorded on the attribute: policy id and fingerprint."""
        return f"{self.policy_id}@{self.fingerprint()}"

    def radius_m(self, *, extent_m: Vector3, factor: float) -> float:
        """The interaction radius, in meters, for a measured extent and a declared factor."""
        clamped = min(max(factor, self.factor_min), self.factor_max)
        return 0.5 * max(extent_m) * clamped


def attach_interaction_radius(
    state: EntitySemanticState,
    *,
    extent_m: Vector3,
    interaction_factor: FusedInteractionFactor | None,
    policy: InteractionRadiusPolicy | None = None,
) -> EntitySemanticState:
    """Add the ``interaction_radius_m`` attribute when the primary hypothesis carries a factor.

    The interaction radius is a semantic property scaled onto measured geometry: the interpreter's
    dimensionless interaction factor for the primary hypothesis (fused upstream over its claims),
    times the entity's own measured size under ``policy``. The factor is a hypothesis, never a
    measurement, so a state with no primary or no factor keeps its attributes unchanged. It is
    attached here rather than in the state mapping because the measured extent belongs to the
    geometry summary, not to the fused evidence the state is built from.

    Args:
        state: The semantic state built from the fused evidence.
        extent_m: The side lengths of the entity's measured bounds, in meters.
        interaction_factor: The primary fused hypothesis's interaction factor, or ``None`` when it
            carried none. The attribute cites only the claims that declared it.
        policy: The rule that clamps the factor and scales the extent; the default is
            ``interaction-radius-from-factor-v1`` with its default range.

    Returns:
        The state, with the derived attribute appended when one applies; otherwise unchanged.
    """
    if state.primary is None or interaction_factor is None:
        return state
    chosen = InteractionRadiusPolicy() if policy is None else policy
    attribute = EntityAttribute(
        name="interaction_radius_m",
        value=repr(chosen.radius_m(extent_m=extent_m, factor=interaction_factor.value)),
        origin=AttributeOrigin.DERIVED,
        derivation_id=chosen.derivation_id,
        evidence=interaction_factor.evidence,
    )
    return replace(state, attributes=sorted_attributes((*state.attributes, attribute)))
