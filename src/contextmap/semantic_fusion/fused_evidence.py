"""Contratos de **crença em formação**: a evidência acumulada sobre um suporte.

:class:`FusedEvidence` mantém todas as hipóteses, a evidência exata atrás de cada uma,
os conflitos entre observações físicas e o que é simplesmente desconhecido. Não elege
vencedor, não tem hipótese primária, não tem confiança combinada e não cria identidade
de objeto -- isso é Semantic Mapping.

Um label é a hipótese aberta como foi proposta; equivalência entre labels é decisão de
política, nunca suposta aqui. A incerteza é *reportada*, nunca resolvida, e sempre nomeia
as contribuições e claims exatas que a produziram.

As invariantes deste módulo são de correlação: todo grupo, hipótese, atributo, fator e
registro de incerteza tem de resolver para contribuições e claims que a evidência
realmente carrega. Um registro adulterado é recusado na construção, não confiado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.evidence import (
    ChannelProvenance,
    EvidenceChannel,
    EvidenceContribution,
    EvidenceReference,
    PhysicalObservationGroup,
    PointRepresentationRef,
    SupportSignal,
    SupportSignalKind,
)
from contextmap.semantic_fusion.identity import (
    EvidenceContributionId,
    FusedEvidenceId,
    FusedHypothesisId,
    FusionSupportId,
)
from contextmap.semantic_fusion.invariants import (
    producer_key,
    require_canonical,
    require_present,
    time_bounds_of,
)
from contextmap.semantic_fusion.quality_weighting import QualityWeighting
from contextmap.semantic_fusion.supports import FusionSupport
from contextmap.semantic_interpretation import ClaimId, HypothesisRole
from contextmap.sensor_association import SpatialObservationId
from contextmap.state_estimation import TimeBounds


class EvidenceStance(Enum):
    """How one claim relates to the hypothesis that lists it.

    Attributes:
        SUPPORTING: The claim proposes this hypothesis.
        CONFLICTING: The claim proposes an incompatible hypothesis for the same support.
        AMBIGUOUS: The claim is retained but its relation to this hypothesis is undecided.
        ABSTAINING: The claim is an abstention (``unknown``): it is neither support for the
            hypothesis nor evidence against it.
    """

    SUPPORTING = "supporting"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"
    ABSTAINING = "abstaining"


class UncertaintyKind(Enum):
    """Why a fused evidence does not settle on one interpretation.

    Attributes:
        CONTRADICTION: Distinct physical observations propose incompatible hypotheses.
        AMBIGUITY: Competing hypotheses that no contradiction between observations explains,
            for example alternatives proposed by one interpretation.
        NEAR_TIE: Hypotheses indistinguishable under the rule that declared the tie.
        INSUFFICIENT_EVIDENCE: Too little semantic evidence to support any hypothesis.
    """

    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    NEAR_TIE = "near_tie"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_COMPETITIVE_KINDS = frozenset(
    {UncertaintyKind.CONTRADICTION, UncertaintyKind.AMBIGUITY, UncertaintyKind.NEAR_TIE}
)


@dataclass(frozen=True, kw_only=True)
class UncertaintyRecord:
    """A conflict, ambiguity or lack of evidence, with the exact evidence behind it.

    Attributes:
        kind: What kind of uncertainty this is.
        hypothesis_ids: The hypotheses involved, sorted and unique.
        evidence: The contributions and claims that produced it, sorted and unique.
        rule_id: The versioned rule that declared it.
    """

    kind: UncertaintyKind
    hypothesis_ids: tuple[FusedHypothesisId, ...]
    evidence: tuple[EvidenceReference, ...]
    rule_id: str

    def __post_init__(self) -> None:
        """Validate the rule, the ordering and that competition names competitors.

        Raises:
            ValueError: If the rule is empty, a collection is not sorted and unique, or a
                contradiction, ambiguity or tie names fewer than two hypotheses.
        """
        require_present(self, "rule_id")
        require_canonical("hypothesis_ids", self.hypothesis_ids, lambda item: (item,))
        require_canonical(
            "evidence", self.evidence, lambda ref: (ref.contribution_id, ref.claim_id or "")
        )
        if self.kind in _COMPETITIVE_KINDS and len(self.hypothesis_ids) < 2:
            raise ValueError(f"a {self.kind.value} needs at least two hypotheses")


def uncertainty_key(record: UncertaintyRecord) -> tuple[str, ...]:
    """Compute the canonical ordering key of an uncertainty record.

    Args:
        record: The record to order.

    Returns:
        The key: kind, rule, the hypotheses involved and the evidence behind it, so that
        equivalent uncertainty always encodes in the same order.
    """
    return (
        record.kind.value,
        record.rule_id,
        "|".join(record.hypothesis_ids),
        "|".join(f"{ref.contribution_id}/{ref.claim_id or ''}" for ref in record.evidence),
    )


@dataclass(frozen=True, kw_only=True)
class HypothesisEvidence:
    """One claim of one contribution, as it relates to one hypothesis.

    Attributes:
        contribution_id: The contribution that carries the claim.
        claim_id: The claim, local to that contribution's perception result.
        stance: How the claim relates to the hypothesis.
        role: Whether the interpreter proposed the claim as its primary or an alternative.
        signals: Typed scores of the claim, sorted by kind and producer and unique.
    """

    contribution_id: EvidenceContributionId
    claim_id: ClaimId
    stance: EvidenceStance
    role: HypothesisRole
    signals: tuple[SupportSignal, ...] = ()

    def __post_init__(self) -> None:
        """Validate the references and the order of the signals.

        Raises:
            ValueError: If an identity is empty or the signals are not sorted and unique.
        """
        require_present(self, "contribution_id", "claim_id")
        require_canonical(
            "signals",
            self.signals,
            lambda signal: (signal.kind.value, *producer_key(signal.producer)),
        )


@dataclass(frozen=True, kw_only=True)
class FusedAttribute:
    """One property value the interpreter stated about a hypothesis, with the claims behind it.

    It is an *observation* of the claims, verbatim: the value is the interpreter's own, written as
    text, and several values of one name are several observations, never one aggregate. The name
    ``category`` is reserved for the claim's own coarse category.

    Attributes:
        name: The property, as the interpreter named it (``color``, ``material``, ``category``).
        value: Its value, as text: a string verbatim, any other JSON scalar in its JSON spelling
            (``true``, ``3``, ``2.5``).
        evidence: The claims that stated exactly this value, sorted by contribution and claim and
            unique; each names its claim.
    """

    name: str
    value: str
    evidence: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Validate the name, the value and that each piece of evidence names a claim.

        Raises:
            ValueError: If the name or value is blank, there is no evidence, a reference does not
                name its claim, or the evidence is not sorted and unique.
        """
        require_present(self, "name", "value")
        if not self.evidence:
            raise ValueError(f"attribute {self.name!r} needs the claims that stated it")
        if any(ref.claim_id is None for ref in self.evidence):
            raise ValueError(f"attribute {self.name!r} must cite the claim, not only the view")
        require_canonical(
            "evidence", self.evidence, lambda ref: (ref.contribution_id, ref.claim_id or "")
        )


INTERACTION_FACTOR_AGGREGATION_POLICY_ID = "median-of-physical-observation-medians-v1"
"""Versioned identity of the rule that aggregates declared interaction factors.

Each physical observation contributes one factor, the median of those its own claims declared;
the hypothesis factor is the median of those per-observation factors."""


@dataclass(frozen=True, kw_only=True)
class FusedInteractionFactor:
    """The interaction factor of a hypothesis, with exactly who declared it.

    It is a semantic hypothesis the interpreter stated, never a measurement. Only claims that
    declared a usable factor are cited, so a consumer can tell the contributors from the other
    supporters of the label.

    Attributes:
        value: The aggregated dimensionless factor, finite and strictly positive.
        aggregation_policy_id: Versioned rule that aggregated the declared factors.
        physical_observation_ids: The distinct physical observations that declared a factor, sorted
            and unique; its size is the number of samples the aggregate rests on.
        evidence: The supporting claims that declared a factor, sorted and unique; each names its
            claim.
        minimum: The smallest per-observation factor aggregated.
        maximum: The largest per-observation factor aggregated; with ``minimum`` it reports the
            dispersion without resolving it.
    """

    value: float
    aggregation_policy_id: str
    physical_observation_ids: tuple[SourceObservationId, ...]
    evidence: tuple[EvidenceReference, ...]
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        """Validate the value, its dispersion and its contributors.

        Raises:
            ValueError: If a number is not finite and strictly positive, the minimum, value and
                maximum are not ordered, the policy is empty, there are no contributors, the
                contributors are not sorted and unique, or a reference does not name its claim.
        """
        require_present(self, "aggregation_policy_id")
        for name in ("value", "minimum", "maximum"):
            number = getattr(self, name)
            if not (math.isfinite(number) and number > 0.0):
                raise ValueError(f"interaction_factor {name} must be finite and > 0, got {number}")
        if not self.minimum <= self.value <= self.maximum:
            raise ValueError(
                f"interaction_factor must satisfy minimum <= value <= maximum, got "
                f"{self.minimum}, {self.value}, {self.maximum}"
            )
        if not self.physical_observation_ids or not self.evidence:
            raise ValueError("interaction_factor needs the observations and claims behind it")
        require_canonical(
            "physical_observation_ids", self.physical_observation_ids, lambda item: (item,)
        )
        if any(ref.claim_id is None for ref in self.evidence):
            raise ValueError("interaction_factor must cite the claims, not only the views")
        require_canonical(
            "evidence", self.evidence, lambda ref: (ref.contribution_id, ref.claim_id or "")
        )


@dataclass(frozen=True, kw_only=True)
class FusedHypothesis:
    """A semantic candidate for a support, with every piece of evidence behind it.

    The label is the open-vocabulary text as proposed; equivalence between labels is a
    policy decision, never assumed here. There is no probability and no single
    confidence: the typed signals of each evidence item stay separate.

    Attributes:
        hypothesis_id: Identity local to its :class:`FusedEvidence`.
        label: The hypothesis text, verbatim.
        evidence: Claims that support, conflict with or are ambiguous about the hypothesis,
            sorted by contribution and claim and unique.
        interaction_factor: The dimensionless interaction factor the interpreter attached to the
            hypothesis, aggregated per physical observation over the supporting claims that
            carried one and citing exactly those; ``None`` when none did. It is a semantic
            property of the object, not support for the label: a later stage scales the entity's
            own measured size by it to obtain an interaction radius in meters.
        observed_attributes: What the supporting claims stated about the object -- their category
            and named attributes -- one entry per distinct ``(name, value)``, sorted and unique,
            each citing exactly the supporting claims that stated it. Empty when they stated
            nothing. The interaction factor has its own typed field and is not listed again.
    """

    hypothesis_id: FusedHypothesisId
    label: str
    evidence: tuple[HypothesisEvidence, ...]
    interaction_factor: FusedInteractionFactor | None = None
    observed_attributes: tuple[FusedAttribute, ...] = ()

    def __post_init__(self) -> None:
        """Validate the label, the evidence, the interaction factor and the observed attributes.

        Raises:
            ValueError: If the identity or label is empty, the evidence is not sorted and
                unique, no evidence item supports the hypothesis, the interaction factor is
                present but not finite and strictly positive, or the observed attributes are not
                sorted and unique or cite a claim that does not support the hypothesis.
        """
        require_present(self, "hypothesis_id", "label")
        require_canonical(
            "evidence", self.evidence, lambda item: (item.contribution_id, item.claim_id)
        )
        if not any(item.stance is EvidenceStance.SUPPORTING for item in self.evidence):
            raise ValueError(
                f"hypothesis {self.label!r} needs at least one supporting evidence item"
            )
        self._require_attributes_of_supporters()
        self._require_factor_of_supporters()

    def _require_factor_of_supporters(self) -> None:
        """Require that the interaction factor only cites claims supporting this hypothesis."""
        if self.interaction_factor is None:
            return
        supporters = {
            (item.contribution_id, item.claim_id)
            for item in self.evidence
            if item.stance is EvidenceStance.SUPPORTING
        }
        for ref in self.interaction_factor.evidence:
            if (ref.contribution_id, ref.claim_id) not in supporters:
                raise ValueError(
                    f"interaction_factor of hypothesis {self.label!r} cites claim "
                    f"{ref.claim_id!r}, which does not support the hypothesis"
                )

    def _require_attributes_of_supporters(self) -> None:
        """Require canonical attributes that only cite claims supporting this hypothesis."""
        require_canonical(
            "observed_attributes", self.observed_attributes, lambda item: (item.name, item.value)
        )
        supporters = {
            (item.contribution_id, item.claim_id)
            for item in self.evidence
            if item.stance is EvidenceStance.SUPPORTING
        }
        for attribute in self.observed_attributes:
            for ref in attribute.evidence:
                if (ref.contribution_id, ref.claim_id) not in supporters:
                    raise ValueError(
                        f"attribute {attribute.name!r} of hypothesis {self.label!r} cites claim "
                        f"{ref.claim_id!r}, which does not support the hypothesis"
                    )


@dataclass(frozen=True, kw_only=True)
class FusedEvidenceProvenance:
    """How evidence was grouped and fused.

    Attributes:
        grouping_policy_id: Versioned rule that grouped inference by physical observation.
        fusion_policy_id: Versioned rule that accumulated the evidence.
        configuration_fingerprint: Hash of the fusion configuration, when configurable.
        code_version: Code revision that produced the evidence, when known.
    """

    grouping_policy_id: str
    fusion_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require both policy identities.

        Raises:
            ValueError: If a policy identity is empty.
        """
        require_present(self, "grouping_policy_id", "fusion_policy_id")


@dataclass(frozen=True, kw_only=True)
class FusedEvidence:
    """The evidence accumulated over one support, with nothing collapsed.

    There is no winner, no primary hypothesis and no combined confidence: competing
    hypotheses, contradictions, ambiguity and missing evidence are all represented.
    Persistent entity identity is created by Semantic Mapping, not here.

    Attributes:
        fused_evidence_id: Identity of this fused evidence.
        fusion_support_id: The support the evidence was accumulated over.
        physical_observation_groups: One group per physical frame, sorted by frame.
        contributions: Every view's contribution, sorted by contribution and unique.
        hypotheses: The candidate interpretations, sorted by hypothesis and unique; empty
            when no view produced a claim.
        point_representation_refs: Static 3D structure of the support, listed once each.
        uncertainty: Conflicts, ambiguities and lack of evidence.
        temporal_summary: Interval spanned by the acquisition of the physical observations.
        provenance: How the evidence was grouped and fused.
        channels: The evidence channels the policy declared, sorted by channel and unique,
            each with the identities that fed it. Semantic claims and geometry support are
            always present; data of any other channel exists only if that channel is here,
            so an effect can be attributed to a channel.
        weighting: The factors a quality-aware policy derived, or ``None`` for the baseline.
    """

    fused_evidence_id: FusedEvidenceId
    fusion_support_id: FusionSupportId
    physical_observation_groups: tuple[PhysicalObservationGroup, ...]
    contributions: tuple[EvidenceContribution, ...]
    hypotheses: tuple[FusedHypothesis, ...]
    temporal_summary: TimeBounds
    provenance: FusedEvidenceProvenance
    channels: tuple[ChannelProvenance, ...]
    point_representation_refs: tuple[PointRepresentationRef, ...] = ()
    uncertainty: tuple[UncertaintyRecord, ...] = ()
    weighting: QualityWeighting | None = None

    def __post_init__(self) -> None:
        """Validate that groups, hypotheses and uncertainty agree with the contributions.

        Raises:
            ValueError: If an identity is empty, there is no contribution, a collection is
                not sorted and unique, a group disagrees with its contributions, one
                inference result is claimed by two physical observations, the temporal
                summary does not span the groups, a hypothesis or uncertainty references
                evidence that does not exist, a scorer signal has no score reference, or a
                hypothesis label is repeated.
        """
        require_present(self, "fused_evidence_id", "fusion_support_id")
        if not self.contributions:
            raise ValueError("a fused evidence needs at least one contribution")
        self._require_canonical_order()
        self._require_channels_match()
        contributions = {item.contribution_id: item for item in self.contributions}
        self._require_groups_match(contributions)
        self._require_temporal_summary()
        self._require_hypotheses_resolve(contributions)
        self._require_uncertainty_resolves(contributions)
        self._require_weighting_matches()

    @property
    def physical_observation_count(self) -> int:
        """Number of distinct physical frames, however many inferences interpreted each."""
        return len(self.physical_observation_groups)

    @property
    def inference_result_count(self) -> int:
        """Number of inference results, correlated within each physical observation."""
        return sum(group.inference_result_count for group in self.physical_observation_groups)

    def supporting_physical_observations(
        self, hypothesis_id: FusedHypothesisId
    ) -> tuple[SourceObservationId, ...]:
        """List the distinct physical frames whose claims support a hypothesis.

        Repeated inference over one frame and one claim over many geometry points each
        count once, so the result is safe to use as an independent-evidence count.

        Args:
            hypothesis_id: A hypothesis of this fused evidence.

        Returns:
            The physical observation identities, sorted.

        Raises:
            KeyError: If the hypothesis does not belong to this fused evidence.
        """
        hypothesis = next(
            (item for item in self.hypotheses if item.hypothesis_id == hypothesis_id), None
        )
        if hypothesis is None:
            raise KeyError(hypothesis_id)
        contributions = {item.contribution_id: item for item in self.contributions}
        return tuple(
            sorted(
                {
                    contributions[item.contribution_id].physical_observation_id
                    for item in hypothesis.evidence
                    if item.stance is EvidenceStance.SUPPORTING
                }
            )
        )

    def _require_weighting_matches(self) -> None:
        weighting = self.weighting
        if weighting is None:
            return
        if EvidenceChannel.OBSERVATION_QUALITY not in {item.channel for item in self.channels}:
            raise ValueError(
                "a quality weighting needs the observation_quality channel to be declared"
            )
        if [item.contribution_id for item in weighting.contributions] != [
            item.contribution_id for item in self.contributions
        ]:
            raise ValueError("the weighting must cover exactly the contributions of the evidence")
        if [item.hypothesis_id for item in weighting.hypotheses] != [
            item.hypothesis_id for item in self.hypotheses
        ]:
            raise ValueError("the weighting must cover exactly the hypotheses of the evidence")
        for support in weighting.hypotheses:
            expected = self.supporting_physical_observations(support.hypothesis_id)
            listed = tuple(item.physical_observation_id for item in support.observation_factors)
            if listed != expected:
                raise ValueError(
                    f"hypothesis {support.hypothesis_id!r} has supporting physical observations "
                    f"{list(expected)!r}, but its weighting lists {list(listed)!r}"
                )

    def _require_channels_match(self) -> None:
        active = {item.channel for item in self.channels}
        for required in (EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.GEOMETRY_SUPPORT):
            if required not in active:
                raise ValueError(
                    f"channel {required.value} must be declared: it is intrinsic to every "
                    f"fused evidence"
                )
        signals = [
            signal
            for hypothesis in self.hypotheses
            for item in hypothesis.evidence
            for signal in item.signals
        ]
        carried = {
            EvidenceChannel.SEMANTIC_SCORES: any(
                item.score_refs or item.raw_score_refs for item in self.contributions
            )
            or any(signal.kind is SupportSignalKind.SCORER_SUPPORT for signal in signals),
            EvidenceChannel.VISUAL_FEATURES: any(
                item.visual_feature_refs for item in self.contributions
            ),
            EvidenceChannel.OBSERVATION_QUALITY: any(
                item.observation_quality is not None for item in self.contributions
            ),
            EvidenceChannel.POINT_REPRESENTATION: bool(self.point_representation_refs),
        }
        for channel, has_data in carried.items():
            if has_data and channel not in active:
                raise ValueError(
                    f"data of channel {channel.value} is present, but the channel was not declared"
                )

    def _require_canonical_order(self) -> None:
        require_canonical("channels", self.channels, lambda item: (item.channel.value,))
        require_canonical("contributions", self.contributions, lambda item: (item.contribution_id,))
        require_canonical(
            "physical_observation_groups",
            self.physical_observation_groups,
            lambda group: (group.physical_observation_id,),
        )
        require_canonical("hypotheses", self.hypotheses, lambda item: (item.hypothesis_id,))
        require_canonical(
            "point_representation_refs",
            self.point_representation_refs,
            lambda ref: (ref.run_id, ref.representation_id),
        )
        require_canonical("uncertainty", self.uncertainty, uncertainty_key)

    def _require_groups_match(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        contributed: set[tuple[SpatialObservationId, str | None]] = set()
        for contribution in self.contributions:
            key = (
                contribution.spatial_observation_id,
                contribution.semantic_interpretation_artifact_id,
            )
            if key in contributed:
                raise ValueError("duplicate spatial/semantic producer contribution")
            contributed.add(key)
        groups = {
            group.physical_observation_id: group for group in self.physical_observation_groups
        }
        for observation_id in sorted(
            {item.physical_observation_id for item in contributions.values()}
        ):
            if observation_id not in groups:
                raise ValueError(
                    f"contributions of physical observation {observation_id!r} are not in any "
                    f"physical observation group"
                )
        for group in self.physical_observation_groups:
            members = [
                item
                for item in self.contributions
                if item.physical_observation_id == group.physical_observation_id
            ]
            if not members:
                raise ValueError(
                    f"physical observation group {group.physical_observation_id!r} has no "
                    f"contribution"
                )
            expected = (
                tuple(sorted({item.spatial_observation_id for item in members})),
                tuple(
                    sorted(
                        {
                            item.semantic_interpretation_artifact_id
                            for item in members
                            if item.semantic_interpretation_artifact_id is not None
                        }
                    )
                ),
                tuple(sorted({item.region_artifact_id for item in members})),
            )
            found = (
                group.spatial_observation_ids,
                group.semantic_interpretation_artifact_ids,
                group.region_artifact_ids,
            )
            for label, want, got in zip(
                ("spatial observations", "perception results", "perception runs"),
                expected,
                found,
                strict=True,
            ):
                if want != got:
                    raise ValueError(
                        f"physical observation group {group.physical_observation_id!r} lists "
                        f"{label} {list(got)!r}, but its contributions have {list(want)!r}"
                    )

    def _require_temporal_summary(self) -> None:
        expected = time_bounds_of(
            [group.acquisition_timestamp for group in self.physical_observation_groups],
            owner="the physical observations",
        )
        if self.temporal_summary != expected:
            raise ValueError(
                f"temporal_summary must span exactly the acquisition of the physical "
                f"observations, expected {expected!r}, got {self.temporal_summary!r}"
            )

    def _require_hypotheses_resolve(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        labels: dict[str, FusedHypothesisId] = {}
        for hypothesis in self.hypotheses:
            other = labels.setdefault(hypothesis.label, hypothesis.hypothesis_id)
            if other != hypothesis.hypothesis_id:
                raise ValueError(
                    f"hypothesis label {hypothesis.label!r} is used by both {other!r} and "
                    f"{hypothesis.hypothesis_id!r}"
                )
            for item in hypothesis.evidence:
                contribution = contributions.get(item.contribution_id)
                if contribution is None:
                    raise ValueError(
                        f"hypothesis {hypothesis.hypothesis_id!r} references unknown "
                        f"contribution {item.contribution_id!r}"
                    )
                if item.claim_id not in {ref.claim_id for ref in contribution.claim_refs}:
                    raise ValueError(
                        f"hypothesis {hypothesis.hypothesis_id!r} lists claim {item.claim_id!r}, "
                        f"which contribution {item.contribution_id!r} does not carry"
                    )
                scorers = {
                    producer_key(ref.scorer)
                    for ref in contribution.score_refs
                    if ref.claim_id == item.claim_id
                }
                for signal in item.signals:
                    if (
                        signal.kind is SupportSignalKind.SCORER_SUPPORT
                        and producer_key(signal.producer) not in scorers
                    ):
                        raise ValueError(
                            f"scorer signal of claim {item.claim_id!r} has no score reference "
                            f"in contribution {item.contribution_id!r}"
                        )
            self._require_factor_observations_match(hypothesis, contributions)

    @staticmethod
    def _require_factor_observations_match(
        hypothesis: FusedHypothesis,
        contributions: dict[EvidenceContributionId, EvidenceContribution],
    ) -> None:
        """Require the factor's observations to be exactly those of the claims it cites.

        The factor names who declared it twice, by claim and by physical observation; the two
        lists must tell the same story or the provenance contradicts itself.
        """
        factor = hypothesis.interaction_factor
        if factor is None:
            return
        expected = tuple(
            sorted(
                {
                    contributions[ref.contribution_id].physical_observation_id
                    for ref in factor.evidence
                }
            )
        )
        if factor.physical_observation_ids != expected:
            raise ValueError(
                f"interaction_factor of hypothesis {hypothesis.label!r} lists "
                f"physical_observation_ids {factor.physical_observation_ids!r}, but the claims it "
                f"cites come from {expected!r}"
            )

    def _require_uncertainty_resolves(
        self, contributions: dict[EvidenceContributionId, EvidenceContribution]
    ) -> None:
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        for record in self.uncertainty:
            for hypothesis_id in record.hypothesis_ids:
                if hypothesis_id not in hypothesis_ids:
                    raise ValueError(f"uncertainty names unknown hypothesis {hypothesis_id!r}")
            observations: set[SourceObservationId] = set()
            for reference in record.evidence:
                contribution = contributions.get(reference.contribution_id)
                if contribution is None:
                    raise ValueError(
                        f"uncertainty references unknown contribution {reference.contribution_id!r}"
                    )
                if reference.claim_id is not None and reference.claim_id not in {
                    ref.claim_id for ref in contribution.claim_refs
                }:
                    raise ValueError(
                        f"uncertainty references unknown claim {reference.claim_id!r} of "
                        f"contribution {reference.contribution_id!r}"
                    )
                observations.add(contribution.physical_observation_id)
            if record.kind is UncertaintyKind.CONTRADICTION and len(observations) < 2:
                raise ValueError(
                    "a contradiction needs evidence from at least two distinct physical "
                    "observations"
                )


@dataclass(frozen=True, kw_only=True)
class FusionOutcome:
    """One support and the evidence fused over it.

    It is the unit a fusion run produces and persists, and the unit Semantic Mapping
    consumes: the *where* and the *what was accumulated there*, bound together so the two
    can never drift apart.

    Attributes:
        support: The spatial support.
        evidence: The evidence fused over exactly that support.
    """

    support: FusionSupport
    evidence: FusedEvidence

    def __post_init__(self) -> None:
        """Require that the evidence belongs to the support.

        Raises:
            ValueError: If the two name different ``fusion_support_id`` values.
        """
        if self.support.fusion_support_id != self.evidence.fusion_support_id:
            raise ValueError(
                f"fusion_support_id {self.support.fusion_support_id!r} does not match the "
                f"evidence's {self.evidence.fusion_support_id!r}"
            )
