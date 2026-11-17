"""A política baseline de acumulação de evidência multi-vista.

``baseline-evidence-accumulation-v1`` transforma as observações espaciais de um
:class:`FusionSupport` em uma :class:`FusedEvidence`. É o braço de controle canônico:
transparente, determinístico e deliberadamente conservador.

Este módulo é o ponto de entrada e o dono da **configuração** da política. Ele monta uma
:class:`EvidenceContribution` por observação espacial -- com as claims, saídas de scorer,
features e, quando oferecida, a referência de qualidade -- e então delega as regras
científicas aos seus donos:

* ``hypotheses`` -- chave de label, stance, fator de interação e atributos observados;
* ``uncertainty`` -- contradição, ambiguidade, empate e evidência insuficiente;
* ``channels`` -- que identidades alimentaram cada canal declarado.

Os canais são opt-in: a política declara quais participam, e o chamador pode oferecer as
mesmas entradas a toda configuração. Dado oferecido para um canal não declarado é
ignorado; um canal declarado sem entrada alguma é erro.

O que ela deliberadamente **não** faz: ponderar por qualidade de observação (as
referências são preservadas, nunca usadas), reconhecer refinamentos como ``pallet`` e
``wooden pallet`` (isso exige uma regra explícita e versionada que o baseline não tem, então
eles competem como hipóteses separadas), eleger vencedor, inferir identidade de entidade ou
aplicar conhecimento prévio.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion.channels import channel_provenance
from contextmap.semantic_fusion.evidence import (
    EvidenceChannel,
    EvidenceContribution,
    ObservationQualityRef,
    PhysicalObservationGroup,
    PointRepresentationRef,
    RawSemanticScoreReference,
    ScoreReference,
    SemanticClaimRef,
    SupportSignal,
    SupportSignalKind,
    VisualFeatureRef,
)
from contextmap.semantic_fusion.fused_evidence import FusedEvidence, FusedEvidenceProvenance
from contextmap.semantic_fusion.grouping import PhysicalObservationGrouping
from contextmap.semantic_fusion.hypotheses import ViewClaim, build_hypotheses, label_key
from contextmap.semantic_fusion.identity import (
    EvidenceContributionId,
    evidence_contribution_id_for,
    fused_evidence_id_for,
)
from contextmap.semantic_fusion.invariants import producer_key, time_bounds_of
from contextmap.semantic_fusion.supports import FusionSupport
from contextmap.semantic_fusion.uncertainty import report_uncertainty
from contextmap.semantic_interpretation import ClaimId, SemanticClaim
from contextmap.semantic_scoring import SemanticScore
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.visual_perception import PerceptionResult, SemanticSupport

BASELINE_ACCUMULATION_POLICY_ID = "baseline-evidence-accumulation-v1"
"""Versioned identity of the baseline policy, including its label-key rule."""


@dataclass(frozen=True, kw_only=True)
class BaselineAccumulationPolicy:
    """Configuration of the baseline policy: what counts as abstention and as a near tie.

    Attributes:
        abstention_labels: Hypothesis texts that express abstention rather than a hypothesis.
            They are compared by label key, so spelling and case do not matter. Empty by
            default: no label is an abstention unless the profile says so.
        near_tie_margin: The largest difference, in distinct supporting physical
            observations, at which the leading hypotheses still count as tied. ``0`` means
            exactly equal.
        channels: The evidence channels that take part. Semantic claims are required;
            geometry support is intrinsic and always active. The default is semantic claims
            only, so a channel is used only when declared, however much data is offered.
    """

    abstention_labels: frozenset[str] = frozenset()
    near_tie_margin: int = 0
    channels: frozenset[EvidenceChannel] = frozenset({EvidenceChannel.SEMANTIC_CLAIMS})

    def __post_init__(self) -> None:
        """Validate the channels, the margin and the labels.

        Raises:
            ValueError: If semantic claims are not among the channels, the margin is
                negative or an abstention label is empty.
        """
        if EvidenceChannel.SEMANTIC_CLAIMS not in self.channels:
            raise ValueError(
                "channels must include semantic_claims: hypotheses come from semantic claims"
            )
        if self.near_tie_margin < 0:
            raise ValueError(f"near_tie_margin must not be negative, got {self.near_tie_margin}")
        if any(not label_key(label) for label in self.abstention_labels):
            raise ValueError("an abstention label must not be empty")

    @property
    def active_channels(self) -> frozenset[EvidenceChannel]:
        """The declared channels plus the intrinsic geometry support."""
        return self.channels | {EvidenceChannel.GEOMETRY_SUPPORT}

    @property
    def abstention_keys(self) -> frozenset[str]:
        """The abstention labels reduced to their label keys."""
        return frozenset(label_key(label) for label in self.abstention_labels)

    def fingerprint(self) -> str:
        """Hash the policy identity and configuration, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration; labels that
            differ only typographically give the same fingerprint.
        """
        canonical = json.dumps(
            {
                "policy_id": BASELINE_ACCUMULATION_POLICY_ID,
                "abstention_labels": sorted(self.abstention_keys),
                "near_tie_margin": self.near_tie_margin,
                "channels": sorted(channel.value for channel in self.active_channels),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def accumulate_baseline_evidence(
    support: FusionSupport,
    *,
    observations: Mapping[SpatialObservationId, SpatialObservation],
    grouping: PhysicalObservationGrouping,
    perception_results: Mapping[tuple[str, SourceObservationId], Sequence[PerceptionResult]],
    semantic_scores: Mapping[str, Sequence[SemanticSupport]] | None = None,
    raw_semantic_scores: Mapping[str, Sequence[tuple[str, SemanticScore]]] | None = None,
    observation_quality_refs: Mapping[SpatialObservationId, ObservationQualityRef] | None = None,
    point_representation_refs: Iterable[PointRepresentationRef] | None = None,
    policy: BaselineAccumulationPolicy | None = None,
    code_version: str | None = None,
) -> FusedEvidence:
    """Accumulate the evidence of one support under the baseline policy.

    Evidence channels are opt-in: the policy declares which take part, and the caller may
    offer the same inputs to every configuration. Data offered for a channel the policy did
    not declare is ignored, and a declared channel with no input at all is an error.

    Args:
        support: The support to accumulate over.
        observations: The spatial observations, by identity; every observation of the
            support must be present, and any other is ignored.
        grouping: The physical-observation grouping of the same observations, which
            supplies the acquisition time of each physical observation.
        perception_results: Explicitly selected derived views keyed by regional producer and
            physical observation. Semantic and visual producers remain independent.
        semantic_scores: Scorer outputs per perception result, needed by the
            ``semantic_scores`` channel. A score for a claim that is not attached to an
            observation of the support is ignored.
        raw_semantic_scores: Raw scorer outputs paired with their scoring-run identities.
            They preserve provenance where score identifiers are only run-local.
        observation_quality_refs: References to the measured quality of each observation,
            needed by the ``observation_quality`` channel. The baseline keeps them and never
            weights by them.
        point_representation_refs: Static 3D structure, needed by the
            ``point_representation`` channel. Only references anchored inside the support
            are kept, each once.
        policy: The channels, what counts as abstention and as a near tie; the default is
            semantic claims only, no abstention and only exact ties.
        code_version: Code revision to record in the provenance, when known.

    Returns:
        The fused evidence; it does not depend on the order of any input.

    Raises:
        ValueError: If a declared channel has no input (scores, quality references or
            structural references), an observation or perception result is missing, a claim
            is not in its result, an observation belongs to another map or has geometry outside the
            support, the grouping does not cover a physical observation or does not list an
            observation, or a scorer scored one claim twice.
    """
    chosen = BaselineAccumulationPolicy() if policy is None else policy
    active = chosen.active_channels
    if (
        EvidenceChannel.SEMANTIC_SCORES in active
        and semantic_scores is None
        and raw_semantic_scores is None
    ):
        raise ValueError("channel semantic_scores is declared but no semantic scores were provided")
    _require_input(
        active,
        EvidenceChannel.OBSERVATION_QUALITY,
        observation_quality_refs,
        "observation quality references",
    )
    _require_input(
        active,
        EvidenceChannel.POINT_REPRESENTATION,
        point_representation_refs,
        "point representation references",
    )
    scores = semantic_scores if EvidenceChannel.SEMANTIC_SCORES in active else None
    raw_scores = raw_semantic_scores if EvidenceChannel.SEMANTIC_SCORES in active else None
    quality_refs = (
        observation_quality_refs if EvidenceChannel.OBSERVATION_QUALITY in active else None
    )
    use_features = EvidenceChannel.VISUAL_FEATURES in active

    contributions: list[EvidenceContribution] = []
    claims: list[ViewClaim] = []
    for observation_id in support.spatial_observation_ids:
        observation = _observation(support, observations, observation_id)
        regional_key = (observation.region_artifact_id, observation.source_observation_id)
        selected_views = perception_results.get(regional_key)
        if selected_views is None:
            raise ValueError(f"no selected evidence for regional observation {regional_key!r}")
        by_semantic: dict[str | None, PerceptionResult] = {}
        all_features: dict[tuple[str, str], VisualFeatureRef] = {}
        for result in selected_views:
            if (
                result.region_artifact_id != observation.region_artifact_id
                or result.source_observation_id != observation.source_observation_id
            ):
                raise ValueError("selected evidence belongs to another regional observation")
            if result.sequence_artifact_id != observation.provenance.sequence_artifact_id:
                raise ValueError("selected evidence belongs to another sequence")
            producer = result.semantic_interpretation_artifact_id
            previous = by_semantic.get(producer)
            if previous is not None and (
                previous.claims != result.claims or previous.scene_context != result.scene_context
            ):
                raise ValueError("conflicting evidence for one semantic artifact")
            by_semantic.setdefault(producer, result)
            if use_features and result.visual_evidence_artifact_id is not None:
                for feature in result.features:
                    if feature.region_id not in (None, observation.region_id):
                        continue
                    reference = VisualFeatureRef(
                        visual_artifact_id=result.visual_evidence_artifact_id,
                        source_observation_id=result.source_observation_id,
                        feature_id=feature.feature_id,
                        embedding_space_id=feature.embedding_space_id,
                        scope=feature.scope,
                        region_id=feature.region_id,
                    )
                    key = (reference.visual_artifact_id, str(reference.feature_id))
                    if key in all_features and all_features[key] != reference:
                        raise ValueError("conflicting feature reference in one visual artifact")
                    all_features[key] = reference
        for producer, result in sorted(by_semantic.items(), key=lambda item: item[0] or ""):
            contribution, view_claims = _contribution(
                support,
                observation,
                result,
                () if scores is None else scores.get(producer or "", ()),
                () if raw_scores is None else raw_scores.get(producer or "", ()),
                None if quality_refs is None else quality_refs.get(observation_id),
                chosen.abstention_keys,
                use_features=use_features,
            )
            # As features compartilhadas entram uma vez por observação espacial, não por claim.
            contribution = replace(
                contribution,
                visual_feature_refs=tuple(all_features[key] for key in sorted(all_features))
                if not any(item.spatial_observation_id == observation_id for item in contributions)
                else (),
            )
            contributions.append(contribution)
            claims.extend(view_claims)

    groups = _groups(contributions, grouping)
    hypotheses = build_hypotheses(claims)
    support_geometry = set(support.geometry_support)
    structure = sorted(
        {
            ref
            for ref in (point_representation_refs or ())
            if EvidenceChannel.POINT_REPRESENTATION in active
            and ref.geometry_reference in support_geometry
        },
        key=lambda ref: (ref.run_id, ref.representation_id),
    )
    return FusedEvidence(
        fused_evidence_id=fused_evidence_id_for(fusion_support_id=support.fusion_support_id),
        fusion_support_id=support.fusion_support_id,
        physical_observation_groups=groups,
        contributions=tuple(sorted(contributions, key=lambda item: item.contribution_id)),
        hypotheses=hypotheses,
        channels=channel_provenance(active, support, claims, contributions, structure),
        point_representation_refs=tuple(structure),
        uncertainty=report_uncertainty(
            hypotheses,
            claims,
            contributions,
            chosen.near_tie_margin,
            policy_id=BASELINE_ACCUMULATION_POLICY_ID,
        ),
        temporal_summary=time_bounds_of(
            [group.acquisition_timestamp for group in groups], owner="the physical observations"
        ),
        provenance=FusedEvidenceProvenance(
            grouping_policy_id=grouping.grouping_policy_id,
            fusion_policy_id=BASELINE_ACCUMULATION_POLICY_ID,
            configuration_fingerprint=chosen.fingerprint(),
            code_version=code_version,
        ),
    )


def _observation(
    support: FusionSupport,
    observations: Mapping[SpatialObservationId, SpatialObservation],
    observation_id: SpatialObservationId,
) -> SpatialObservation:
    observation = observations.get(observation_id)
    if observation is None:
        raise ValueError(
            f"no spatial observation {observation_id!r} for support {support.fusion_support_id!r}"
        )
    if observation.provenance.geometric_map_id != support.geometric_map_id:
        raise ValueError(
            f"spatial observation {observation_id!r} references map "
            f"{observation.provenance.geometric_map_id!r}, but the support is over "
            f"{support.geometric_map_id!r}"
        )
    outside = set(observation.geometry_support) - set(support.geometry_support)
    if outside:
        raise ValueError(
            f"spatial observation {observation_id!r} has {len(outside)} geometry elements "
            f"outside support {support.fusion_support_id!r}"
        )
    return observation


def _contribution(
    support: FusionSupport,
    observation: SpatialObservation,
    result: PerceptionResult,
    result_scores: Sequence[SemanticSupport],
    raw_result_scores: Sequence[tuple[str, SemanticScore]],
    quality: ObservationQualityRef | None,
    abstention_keys: frozenset[str],
    *,
    use_features: bool,
) -> tuple[EvidenceContribution, list[ViewClaim]]:
    base = evidence_contribution_id_for(
        fusion_support_id=support.fusion_support_id,
        spatial_observation_id=observation.spatial_observation_id,
    )
    contribution_id = EvidenceContributionId(
        f"{base}--semantic-{result.semantic_interpretation_artifact_id or 'none'}"
    )
    selected = sorted(
        (claim for claim in result.claims if claim.region_id == observation.region_id),
        key=lambda claim: claim.claim_id,
    )
    claim_refs = [SemanticClaimRef(claim_id=claim.claim_id) for claim in selected]
    scores_of_claim = _scores_by_claim(observation, result_scores)

    score_refs = sorted(
        (
            ScoreReference(claim_id=claim.claim_id, scorer=score.provenance)
            for claim in selected
            for score in scores_of_claim.get(claim.claim_id, ())
        ),
        key=lambda ref: (ref.claim_id, *producer_key(ref.scorer)),
    )
    raw_score_refs = sorted(
        (
            RawSemanticScoreReference(
                semantic_scoring_run_id=scoring_run_id,
                score_id=str(score.score_id),
                claim_id=score.claim_id,
                feature_id=score.feature_id,
                score_type=score.score_type,
                value=score.value,
                embedding_space_id=score.embedding_space_id,
                scorer=score.provenance,
            )
            for scoring_run_id, score in raw_result_scores
            if score.claim_id in {claim.claim_id for claim in selected}
        ),
        key=lambda ref: (ref.claim_id, ref.semantic_scoring_run_id, ref.score_id),
    )
    contribution = EvidenceContribution(
        contribution_id=contribution_id,
        physical_observation_id=observation.source_observation_id,
        semantic_interpretation_artifact_id=result.semantic_interpretation_artifact_id,
        region_artifact_id=observation.region_artifact_id,
        spatial_observation_id=observation.spatial_observation_id,
        region_id=observation.region_id,
        geometry_support=observation.geometry_support,
        claim_refs=tuple(claim_refs),
        score_refs=tuple(score_refs),
        visual_feature_refs=(),
        observation_quality=quality,
        raw_score_refs=tuple(raw_score_refs),
    )
    view_claims = [
        ViewClaim(
            contribution_id=contribution_id,
            physical_observation_id=observation.source_observation_id,
            claim=claim,
            key=label_key(claim.hypothesis),
            abstains=label_key(claim.hypothesis) in abstention_keys,
            signals=_signals(claim, scores_of_claim.get(claim.claim_id, ())),
        )
        for claim in selected
    ]
    return contribution, view_claims


def _scores_by_claim(
    observation: SpatialObservation, result_scores: Sequence[SemanticSupport]
) -> dict[ClaimId, list[SemanticSupport]]:
    by_claim: dict[ClaimId, list[SemanticSupport]] = {}
    for score in result_scores:
        by_claim.setdefault(score.claim_id, []).append(score)
    for claim_id, claim_scores in by_claim.items():
        scorers = [producer_key(score.provenance) for score in claim_scores]
        if len(set(scorers)) != len(scorers):
            raise ValueError(
                f"more than one score from the same scorer for claim {claim_id!r} of spatial "
                f"observation {observation.spatial_observation_id!r}"
            )
    return by_claim


def _signals(claim: SemanticClaim, scores: Sequence[SemanticSupport]) -> tuple[SupportSignal, ...]:
    signals = [
        SupportSignal(
            kind=SupportSignalKind.CLAIM_CONFIDENCE,
            producer=claim.provenance.backend,
            value=claim.confidence,
        ),
        *(
            SupportSignal(
                kind=SupportSignalKind.SCORER_SUPPORT,
                producer=score.provenance,
                value=score.support_score,
            )
            for score in scores
        ),
    ]
    return tuple(sorted(signals, key=lambda item: (item.kind.value, *producer_key(item.producer))))


def _groups(
    contributions: Sequence[EvidenceContribution], grouping: PhysicalObservationGrouping
) -> tuple[PhysicalObservationGroup, ...]:
    grouped = {group.physical_observation_id: group for group in grouping.groups}
    by_frame: dict[SourceObservationId, list[EvidenceContribution]] = {}
    for contribution in contributions:
        by_frame.setdefault(contribution.physical_observation_id, []).append(contribution)
    groups: list[PhysicalObservationGroup] = []
    for frame in sorted(by_frame):
        placed = grouped.get(frame)
        if placed is None:
            raise ValueError(f"the grouping does not cover physical observation {frame!r}")
        members = by_frame[frame]
        for member in members:
            if member.spatial_observation_id not in placed.spatial_observation_ids:
                raise ValueError(
                    f"the grouping does not list spatial observation "
                    f"{member.spatial_observation_id!r} under physical observation {frame!r}"
                )
        groups.append(
            PhysicalObservationGroup(
                physical_observation_id=frame,
                acquisition_timestamp=placed.acquisition_timestamp,
                spatial_observation_ids=tuple(
                    sorted({member.spatial_observation_id for member in members})
                ),
                semantic_interpretation_artifact_ids=tuple(
                    sorted(
                        {
                            member.semantic_interpretation_artifact_id
                            for member in members
                            if member.semantic_interpretation_artifact_id is not None
                        }
                    )
                ),
                region_artifact_ids=tuple(
                    sorted({member.region_artifact_id for member in members})
                ),
            )
        )
    return tuple(groups)


def _require_input(
    active: frozenset[EvidenceChannel],
    channel: EvidenceChannel,
    provided: object | None,
    what: str,
) -> None:
    if channel in active and provided is None:
        raise ValueError(f"channel {channel.value} is declared but no {what} were provided")
