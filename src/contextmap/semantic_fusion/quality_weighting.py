"""O peso derivado de uma política ciente de qualidade, inspecionável por componente.

Estes contratos registram *como* a qualidade mensurável de cada vista virou um fator de
contribuição, e o suporte de cada hipótese antes e depois dele. São derivados e
opcionais: a evidência em si -- hipóteses, contribuições e incerteza -- é exatamente a
que o baseline produziria, e nada aqui a substitui.

O fator é um **peso de contribuição de fusão**. Não é confiança semântica, não é
suporte de scorer, não é probabilidade e não substitui a qualidade bruta de onde veio.
Um componente que não pôde ser medido recebe o fator neutro declarado pela política e
registra o motivo: nunca um zero silencioso.

Quem calcula esses valores é ``quality_aware``; este módulo só os define.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.identity import EvidenceContributionId, FusedHypothesisId
from contextmap.semantic_fusion.invariants import (
    require_canonical,
    require_present,
    require_unit_factor,
)


class ComponentTreatment(Enum):
    """How a quality component entered a contribution factor.

    Attributes:
        MEASURED: The component was measured and turned into a factor by its ramp.
        NEUTRAL_FALLBACK: The component could not be measured, so the policy's declared
            neutral factor was used, and the reason is recorded. Never a silent zero.
    """

    MEASURED = "measured"
    NEUTRAL_FALLBACK = "neutral_fallback"


@dataclass(frozen=True, kw_only=True)
class ComponentFactor:
    """The factor one declared quality component gave one contribution.

    Attributes:
        component: The measured quantity, named by the policy that declared it.
        treatment: Whether it was measured or replaced by the neutral factor.
        measured_value: The measured value; ``None`` for a neutral fallback.
        factor: The factor in ``[0, 1]``.
        unavailable_reason: Why the component could not be measured; ``None`` when measured.
    """

    component: str
    treatment: ComponentTreatment
    measured_value: float | None
    factor: float
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        """Validate the factor and that the treatment matches the value and the reason.

        Raises:
            ValueError: If the component is empty, the factor is outside ``[0, 1]``, a
                measured component has no finite value or has a reason, or a neutral
                fallback has a value or no reason.
        """
        require_present(self, "component")
        require_unit_factor("factor", self.factor)
        if self.treatment is ComponentTreatment.MEASURED:
            if self.measured_value is None or not math.isfinite(self.measured_value):
                raise ValueError("a measured component needs a finite measured_value")
            if self.unavailable_reason is not None:
                raise ValueError("unavailable_reason must be None for a measured component")
        else:
            if self.measured_value is not None:
                raise ValueError("measured_value must be None for a neutral fallback")
            if not (self.unavailable_reason or "").strip():
                raise ValueError("unavailable_reason is required for a neutral fallback")


@dataclass(frozen=True, kw_only=True)
class ContributionWeight:
    """The fusion factor of one contribution, with every component behind it.

    The factor is a fusion contribution weight. It is not a semantic confidence, a scorer
    support or a probability, and it never replaces the raw quality it came from.

    Attributes:
        contribution_id: The weighted contribution.
        components: The declared components and their factors, sorted by component.
        factor: The contribution factor in ``[0, 1]``, combined by the weighting's rule.
    """

    contribution_id: EvidenceContributionId
    components: tuple[ComponentFactor, ...]
    factor: float

    def __post_init__(self) -> None:
        """Validate the identity, the components and the factor.

        Raises:
            ValueError: If the identity is empty, there is no component, the components are
                not sorted and unique, or the factor is outside ``[0, 1]``.
        """
        require_present(self, "contribution_id")
        if not self.components:
            raise ValueError("a contribution weight needs at least one component")
        require_canonical("components", self.components, lambda item: (item.component,))
        require_unit_factor("factor", self.factor)


@dataclass(frozen=True, kw_only=True)
class ObservationFactor:
    """The factor of one physical observation for one hypothesis.

    Attributes:
        physical_observation_id: The physical frame.
        factor: Its factor in ``[0, 1]``, however many inference runs interpreted it.
    """

    physical_observation_id: SourceObservationId
    factor: float

    def __post_init__(self) -> None:
        """Validate the identity and the factor.

        Raises:
            ValueError: If the identity is empty or the factor is outside ``[0, 1]``.
        """
        require_present(self, "physical_observation_id")
        require_unit_factor("factor", self.factor)


@dataclass(frozen=True, kw_only=True)
class HypothesisSupport:
    """The support of one hypothesis before and after the weighting is applied.

    Attributes:
        hypothesis_id: The hypothesis.
        supporting_physical_observations: Distinct supporting physical observations: the
            support *before* weighting, the same count the baseline uses.
        observation_factors: The factor of each of those observations, sorted by observation.
        weighted_support: The sum of those factors, in units of weighted physical
            observations: the support *after* weighting. It is not a probability.
    """

    hypothesis_id: FusedHypothesisId
    supporting_physical_observations: int
    observation_factors: tuple[ObservationFactor, ...]
    weighted_support: float

    def __post_init__(self) -> None:
        """Validate the counts and that the weighted support is the sum of the factors.

        Raises:
            ValueError: If the identity is empty, the count differs from the factors listed,
                the factors are not sorted and unique, or the weighted support is not the
                sum of the factors.
        """
        require_present(self, "hypothesis_id")
        require_canonical(
            "observation_factors",
            self.observation_factors,
            lambda item: (item.physical_observation_id,),
        )
        if self.supporting_physical_observations != len(self.observation_factors):
            raise ValueError(
                f"supporting_physical_observations {self.supporting_physical_observations} "
                f"must equal the {len(self.observation_factors)} observation_factors listed"
            )
        total = math.fsum(item.factor for item in self.observation_factors)
        if not math.isclose(self.weighted_support, total, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"weighted_support {self.weighted_support!r} must equal the sum of the "
                f"observation factors, {total!r}"
            )


@dataclass(frozen=True, kw_only=True)
class QualityWeighting:
    """How a quality-aware policy weighted the evidence, for inspection and reproduction.

    The evidence itself is untouched: hypotheses, contributions and uncertainty are exactly
    what the baseline would have produced. This only adds the derived factors.

    Attributes:
        policy_id: The versioned quality-aware policy.
        combination_rule: How component factors combine into a contribution factor.
        observation_rule: How the factors of several contributions of one physical
            observation combine, so repeated inference is never counted as several views.
        definitions_version: The quality definitions the components were read under.
        contributions: The factor of every contribution, sorted by contribution.
        hypotheses: The support of every hypothesis before and after weighting, sorted.
    """

    policy_id: str
    combination_rule: str
    observation_rule: str
    definitions_version: str
    contributions: tuple[ContributionWeight, ...]
    hypotheses: tuple[HypothesisSupport, ...]

    def __post_init__(self) -> None:
        """Validate the identities and the ordering.

        Raises:
            ValueError: If a policy, rule or version is empty, or a collection is not sorted
                and unique.
        """
        require_present(
            self, "policy_id", "combination_rule", "observation_rule", "definitions_version"
        )
        require_canonical("contributions", self.contributions, lambda item: (item.contribution_id,))
        require_canonical("hypotheses", self.hypotheses, lambda item: (item.hypothesis_id,))
