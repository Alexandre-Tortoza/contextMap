"""Reference evaluation of semantic scorer support: does the score separate right from wrong?

A :class:`~contextmap.visual_perception.SemanticScore` (#652) is one scorer's raw support for
one claim; for CLIP and AlphaCLIP a cosine similarity in ``[-1, 1]``. This evaluator asks the
question of #530: does that score separate the hypotheses the reference accepts from those it
rejects? The reference is the target; agreement with the VLM is only a diagnostic.

* **Labels.** A scored claim is *positive* when its hypothesis matches an acceptable hypothesis
  of the annotation joined to its region, *negative* when it matches a rejected one (the only
  negative truth), and *unassessed* otherwise, including every claim of a region without an
  annotation. An unassessed claim never counts as a negative.
* **Scores.** Only ``value`` is read, on its own scale. A score is never a probability,
  ``calibrated_probability`` is never read (no calibration with a held-out
  ``CalibrationSplitPair`` exists) and a claim without a score is outside every denominator.
* **Scorers.** Every metric is computed per scorer, embedding space and score type. Two of them
  are never pooled, and no combined score exists.
* **Repeats.** A repeated region is not a new sample; only ``repeat_index == 0`` counts.
* **Join.** Each :class:`ScoredRegion` carries the annotation of the reference region its run
  region was joined to. The join is an input of this evaluator, named by the context's
  ``region_join_policy_id``: the same policy must serve every compared arm.

Without a ``semantics`` family in the reference set, every support metric is not applicable,
with that reason, and never zero. See ``src/contextmap/evaluation/docs/semantic-scoring.md``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

from contextmap.evaluation._validation import require_text, require_unique
from contextmap.evaluation.annotations import (
    CASEFOLD_EXACT_POLICY,
    AnnotationFamily,
    FrameRegionAnnotation,
    LabelNormalization,
    RegionAnnotationSet,
    SemanticAnnotationSet,
)
from contextmap.evaluation.region_join import (
    FrameRegionJoin,
    RegionJoinPolicy,
    RegionJoinSummary,
    join_frame_regions,
    summarize_region_joins,
)
from contextmap.evaluation.semantic_interpretation import (
    MATCHING_POLICY,
    PERCENTILE_METHOD,
    SemanticAnnotation,
    nearest_rank_percentile,
)
from contextmap.region_perception import (
    RegionId,
    RegionPerceptionRunReader,
)
from contextmap.semantic_interpretation import (
    HypothesisRole,
    SemanticClaim,
    SemanticInterpretationRunReader,
)
from contextmap.semantic_scoring import (
    SemanticScore,
    SemanticScoreType,
    SemanticScoringRunReader,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    PerceptionResult,
)

SEMANTIC_SCORING_EVALUATOR_ID = "semantic-scoring-evaluator"
"""Evaluator identity the registry metrics of the ``semantic_scoring`` stage name."""

SEMANTIC_SCORING_EVALUATOR_VERSION = "1"
"""Version of the labeling, ranking, AUROC and AUPRC definitions implemented here."""

_SEMANTICS_SCHEMA = AnnotationFamily.SEMANTICS.schema
_MATCHING = LabelNormalization(policy_id=CASEFOLD_EXACT_POLICY)

_ScorerKey = tuple[BackendProvenance, str, SemanticScoreType]
"""Scorer, embedding space and score type: the scores one metric may put together."""


class SemanticScoringEvaluationError(ValueError):
    """Raised when scored regions cannot form one controlled support evaluation."""


class ClaimLabel(Enum):
    """What the reference says about a scored claim."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNASSESSED = "unassessed"


@dataclass(frozen=True, kw_only=True)
class SemanticScoringEvaluationPolicy:
    """The declared parameters of the support metrics; there are no hidden defaults.

    Attributes:
        policy_id: Version of the policy, cited by the report.
        minimum_class_count: The pooled AUROC and AUPRC need at least this many positive and
            this many negative scored claims per scorer and embedding space; below it they are
            not applicable.
    """

    policy_id: str
    minimum_class_count: int

    def __post_init__(self) -> None:
        """Require an identity and a positive class minimum."""
        require_text("policy_id", self.policy_id)
        if (
            isinstance(self.minimum_class_count, bool)
            or not isinstance(self.minimum_class_count, int)
            or self.minimum_class_count < 1
        ):
            raise ValueError("minimum_class_count must be an integer of at least 1")


@dataclass(frozen=True, kw_only=True)
class SemanticScoringEvaluationContext:
    """Reproducibility identities shared by every scored region of one report.

    Attributes:
        evaluation_id: Identity of this evaluation.
        reference_set_version: Version of the reference set the annotations come from.
        selection_id: The evaluated selection of the reference set.
        semantic_interpretation_artifact_id: Explicit producer of the evaluated claims.
        region_artifact_id: Selected region producer.
        visual_evidence_artifact_id: Selected feature producer.
        scoring_run_ids: The semantic scoring runs whose scores are evaluated.
        annotation_schemas: The exact annotation schemas the reference set provides.
        region_join_policy_id: The versioned policy that joined reference regions to run
            regions; ``None`` when no run region carries an annotation.
    """

    evaluation_id: str
    reference_set_version: str
    selection_id: str
    semantic_interpretation_artifact_id: str
    region_artifact_id: str
    visual_evidence_artifact_id: str
    scoring_run_ids: tuple[str, ...]
    annotation_schemas: tuple[str, ...]
    region_join_policy_id: str | None

    def __post_init__(self) -> None:
        """Require every identity, at least one scoring run and unique entries."""
        for name in (
            "evaluation_id",
            "reference_set_version",
            "selection_id",
            "semantic_interpretation_artifact_id",
            "region_artifact_id",
            "visual_evidence_artifact_id",
        ):
            require_text(name, getattr(self, name))
        if not self.scoring_run_ids:
            raise ValueError("an evaluation needs at least one scoring run")
        for run_id in self.scoring_run_ids:
            require_text("scoring run id", run_id)
        require_unique("scoring run id", self.scoring_run_ids)
        require_unique("annotation schema", self.annotation_schemas)
        if self.region_join_policy_id is not None:
            require_text("region_join_policy_id", self.region_join_policy_id)

    @property
    def has_semantics(self) -> bool:
        """Whether the reference set provides the ``semantics`` family these metrics need."""
        return _SEMANTICS_SCHEMA in self.annotation_schemas


@dataclass(frozen=True, kw_only=True)
class ScoredRegion:
    """The region claims of one run region, their scores and the annotation joined to it.

    Attributes:
        claims: Every region claim of the region, scored or not. A claim without a score stays
            in the claim count and outside every denominator.
        scores: The scores of those claims, from any scorers; at most one per claim, scorer,
            embedding space and score type.
        annotation: Expectations of the reference region joined to this run region; ``None``
            leaves every claim unassessed.
        repeat_index: ``0`` for the primary run. A repeat is not new physical evidence, so
            quality and diagnostics count only ``0``.
    """

    claims: tuple[SemanticClaim, ...]
    scores: tuple[SemanticScore, ...] = ()
    annotation: SemanticAnnotation | None = None
    repeat_index: int = 0

    def __post_init__(self) -> None:
        """Require the claims of one region and scores that belong to them.

        Raises:
            ValueError: For a scene claim, claims of different regions, observations or
                perception results, a repeated claim, a score of another claim, observation or
                perception result, two scores of one claim from one scorer and embedding space,
                or a negative repeat index.
        """
        if not self.claims:
            raise ValueError("a scored region needs at least one region claim")
        region = _claim_region(self.claims[0])
        for claim in self.claims:
            if _claim_region(claim) != region:
                raise ValueError(
                    "the claims of a scored region must describe one region of one perception "
                    f"result; claim {claim.claim_id!r} describes another"
                )
        require_unique("claim id of a scored region", (claim.claim_id for claim in self.claims))
        claims = {claim.claim_id: claim for claim in self.claims}
        seen: set[tuple[str, _ScorerKey]] = set()
        for score in self.scores:
            scored = claims.get(score.claim_id)
            if scored is None:
                raise ValueError(
                    f"score {score.score_id!r} references claim {score.claim_id!r}, which is not "
                    "a claim of the region"
                )
            if (score.source_observation_id, score.region_id) != (
                scored.source_observation_id,
                scored.region_id,
            ):
                raise ValueError(
                    f"score {score.score_id!r} comes from another observation or perception "
                    f"result than claim {scored.claim_id!r}"
                )
            key = (str(score.claim_id), _scorer_key(score))
            if key in seen:
                raise ValueError(
                    f"claim {score.claim_id!r} has more than one score from scorer "
                    f"{score.provenance.backend_id!r} in space {score.embedding_space_id!r}"
                )
            seen.add(key)
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")

    @property
    def region(self) -> tuple[str, str, str]:
        """Return ``(observation, perception result, region)`` of the claims."""
        return _claim_region(self.claims[0])


@dataclass(frozen=True, kw_only=True)
class SupportMetric:
    """One metric: a value, or the reason it is not applicable, never a zero stand-in.

    Attributes:
        value: The metric; ``None`` exactly when it is not applicable.
        sample_count: Size of the population it is, or would be, computed over.
        not_applicable_reason: Why there is no value; ``None`` exactly when there is one.
    """

    value: float | None
    sample_count: int
    not_applicable_reason: str | None = None

    def __post_init__(self) -> None:
        """Tie the value to the absence of a reason."""
        if (self.value is None) == (self.not_applicable_reason is None):
            raise ValueError("a support metric has either a value or a not-applicable reason")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative")


@dataclass(frozen=True, kw_only=True)
class ScoreDistribution:
    """Raw score values of one population, as :data:`PERCENTILE_METHOD` quantiles.

    Every quantile is ``None`` when the population is empty.
    """

    count: int
    minimum: float | None
    p25: float | None
    median: float | None
    p75: float | None
    maximum: float | None


@dataclass(frozen=True, kw_only=True)
class LabeledScore:
    """One scored claim of the primary run and its label, traceable to both records."""

    score_id: str
    claim_id: str
    source_observation_id: str
    region_id: str
    hypothesis: str
    role: HypothesisRole
    label: ClaimLabel
    value: float


@dataclass(frozen=True, kw_only=True)
class ScorerSupportQuality:
    """Reference quality of one scorer in one embedding space, never pooled with another.

    Attributes:
        scorer: The scorer's provenance, configuration fingerprint included.
        embedding_space_id: The space every score of the block lives in.
        score_type: The numerical semantics of every score of the block.
        claim_count: Region claims of the primary run.
        scored_claim_count: Of those, claims with a score of this block.
        unscored_claim_count: Claims without one; outside every denominator.
        positive_count: Scored claims matching an acceptable hypothesis.
        negative_count: Scored claims matching a rejected hypothesis.
        unassessed_count: Scored claims matching neither, or in an unannotated region.
        positive_scores: Distribution of the positive values.
        negative_scores: Distribution of the negative values.
        ranking_region_count: Regions with at least one positive and one negative scored claim.
        top1_tie_count: Eligible regions whose top score is shared by a positive and a negative.
        pairwise_tie_count: (positive, negative) pairs of one region with equal scores.
        top1_accuracy: Eligible regions whose highest-scored labeled claim is positive, over
            eligible regions; a tie at the top is never correct.
        pairwise_accuracy: Mean over eligible regions of the correctly ordered pairs, a tie
            counting one half: the per-region AUROC.
        auroc: Pooled over every region, correctly ordered pairs plus half the ties, over all
            pairs; separability by one global threshold.
        auprc: Pooled average precision of the positives ranked by score.
        samples: Every scored claim of the primary run with its label.
    """

    scorer: BackendProvenance
    embedding_space_id: str
    score_type: SemanticScoreType
    claim_count: int
    scored_claim_count: int
    unscored_claim_count: int
    positive_count: int
    negative_count: int
    unassessed_count: int
    positive_scores: ScoreDistribution
    negative_scores: ScoreDistribution
    ranking_region_count: int
    top1_tie_count: int
    pairwise_tie_count: int
    top1_accuracy: SupportMetric
    pairwise_accuracy: SupportMetric
    auroc: SupportMetric
    auprc: SupportMetric
    samples: tuple[LabeledScore, ...]


@dataclass(frozen=True, kw_only=True)
class ScorerDiagnostics:
    """Reference-free observations of one scorer in one embedding space; never correctness.

    Attributes:
        scorer: The scorer's provenance.
        embedding_space_id: The space every score lives in.
        score_type: The numerical semantics of every score.
        scores: Distribution of every scored value of the primary run, labeled or not.
        primary_agreement: Over regions with a scored VLM primary claim and another scored
            claim, the fraction where the primary has the strictly highest score. Agreement
            with the VLM is not correctness.
        primary_agreement_tie_count: Of those regions, the ones where the primary shares the
            highest score; they count in the denominator and never as agreement.
    """

    scorer: BackendProvenance
    embedding_space_id: str
    score_type: SemanticScoreType
    scores: ScoreDistribution
    primary_agreement: SupportMetric
    primary_agreement_tie_count: int


@dataclass(frozen=True, kw_only=True)
class SemanticScoringEvaluationReport:
    """Support quality and diagnostics of every scorer, with the rules that produced them.

    Attributes:
        context: The identities of the evaluated runs and reference.
        policy: The declared metric parameters.
        evaluator_id: :data:`SEMANTIC_SCORING_EVALUATOR_ID`.
        evaluator_version: :data:`SEMANTIC_SCORING_EVALUATOR_VERSION`.
        matching_policy: How a hypothesis is matched to the annotation.
        percentile_method: How the distribution quantiles are taken.
        region_count: Regions of the primary run.
        repeat_region_count: Repeated regions, left out of every count.
        claim_count: Region claims of the primary run.
        quality: One reference-quality block per scorer, embedding space and score type.
        diagnostics: One reference-free block per scorer, embedding space and score type.
        region_join: What the reference region join matched and left unmatched; ``None``
            when the caller did not hand the joins over.
    """

    context: SemanticScoringEvaluationContext
    policy: SemanticScoringEvaluationPolicy
    evaluator_id: str
    evaluator_version: str
    matching_policy: str
    percentile_method: str
    region_count: int
    repeat_region_count: int
    claim_count: int
    quality: tuple[ScorerSupportQuality, ...]
    diagnostics: tuple[ScorerDiagnostics, ...]
    region_join: RegionJoinSummary | None


def evaluate_semantic_scoring(
    *,
    context: SemanticScoringEvaluationContext,
    policy: SemanticScoringEvaluationPolicy,
    regions: Sequence[ScoredRegion],
    region_joins: Sequence[FrameRegionJoin] = (),
) -> SemanticScoringEvaluationReport:
    """Evaluate every scorer's support against the reference annotations joined to the regions.

    Args:
        context: The identities of the evaluated runs and reference.
        policy: The declared metric parameters.
        regions: The scored regions, each with the annotation joined to it.
        region_joins: The per-observation joins that attached those annotations, so the
            report counts what they matched and left unmatched (:func:`join_scored_regions`
            returns them); empty when the caller joined elsewhere.

    Raises:
        SemanticScoringEvaluationError: If a region appears twice in one repeat, a region
            carries an annotation while the context names no region join policy or the
            reference set provides no ``semantics`` family, or a join comes from another
            policy than the context names.
    """
    _check_regions(context, regions)
    region_join = _join_summary(context, region_joins)
    primary = tuple(region for region in regions if region.repeat_index == 0)
    keys = sorted(
        {_scorer_key(score) for region in primary for score in region.scores},
        key=_scorer_order,
    )
    claim_count = sum(len(region.claims) for region in primary)
    return SemanticScoringEvaluationReport(
        context=context,
        policy=policy,
        evaluator_id=SEMANTIC_SCORING_EVALUATOR_ID,
        evaluator_version=SEMANTIC_SCORING_EVALUATOR_VERSION,
        matching_policy=MATCHING_POLICY,
        percentile_method=PERCENTILE_METHOD,
        region_count=len(primary),
        repeat_region_count=len(regions) - len(primary),
        claim_count=claim_count,
        quality=tuple(_quality(key, primary, context, policy, claim_count) for key in keys),
        diagnostics=tuple(_diagnostics(key, primary) for key in keys),
        region_join=region_join,
    )


def encode_semantic_scoring_report(report: SemanticScoringEvaluationReport) -> dict[str, Any]:
    """Encode a report into JSON primitives, keeping every identity and identifier."""
    encoded = _plain(report)
    assert isinstance(encoded, dict)
    return encoded


def scorer_label(scorer: BackendProvenance) -> str:
    """Return the stratum label of a scorer: backend, model, version and configuration."""
    return "/".join(
        (scorer.backend_id, scorer.model, scorer.version, scorer.configuration_fingerprint or "-")
    )


@dataclass(frozen=True, kw_only=True)
class JoinedScoredRegions:
    """The scored regions of one perception run, joined to the reference by one policy.

    Attributes:
        regions: One scored region per run region with at least one region claim.
        joins: One join per observation of the run that has a region annotation.
        join_policy_id: The policy of every join.
        semantic_interpretation_artifact_id: The perception run whose claims were scored.
        scoring_run_ids: The scoring runs whose scores were read.
        annotation_schemas: The reference annotation schemas the regions were joined with.
    """

    regions: tuple[ScoredRegion, ...]
    joins: tuple[FrameRegionJoin, ...]
    join_policy_id: str
    semantic_interpretation_artifact_id: str
    region_artifact_id: str
    visual_evidence_artifact_id: str
    scoring_run_ids: tuple[str, ...]
    annotation_schemas: tuple[str, ...]

    def context(
        self, *, evaluation_id: str, reference_set_version: str, selection_id: str
    ) -> SemanticScoringEvaluationContext:
        """Return the evaluation context of these regions, with the evaluation's identities."""
        return SemanticScoringEvaluationContext(
            evaluation_id=evaluation_id,
            reference_set_version=reference_set_version,
            selection_id=selection_id,
            semantic_interpretation_artifact_id=self.semantic_interpretation_artifact_id,
            region_artifact_id=self.region_artifact_id,
            visual_evidence_artifact_id=self.visual_evidence_artifact_id,
            scoring_run_ids=self.scoring_run_ids,
            annotation_schemas=self.annotation_schemas,
            region_join_policy_id=self.join_policy_id,
        )


def join_scored_regions(
    *,
    regional: RegionPerceptionRunReader,
    semantic: SemanticInterpretationRunReader,
    scoring: Sequence[SemanticScoringRunReader],
    regions: RegionAnnotationSet,
    semantics: SemanticAnnotationSet | None,
    policy: RegionJoinPolicy,
) -> JoinedScoredRegions:
    """Build the scored regions of a perception run, joined to the reference by ``policy``.

    Every observation of the run with a region annotation is joined, with the run's masks
    loaded from its mask store. Each run region with region claims becomes one
    :class:`ScoredRegion` with its claims, the scores every scoring run gave them, and the
    semantic annotation of the reference region joined to it: ``None`` when the region is
    unmatched, its observation has no region annotation, the joined region has no semantic
    record, or ``semantics`` is ``None``. Scene claims are never scored and never form a
    region.

    Args:
        regional: The explicitly selected immutable regional artifact and mask source.
        semantic: The explicitly selected immutable semantic artifact and claim source.
        scoring: Its scoring runs, one per scorer.
        regions: The reference region annotation.
        semantics: The reference semantic annotation; ``None`` when the reference set has no
            ``semantics`` family.
        policy: The region join policy.

    Raises:
        SemanticScoringEvaluationError: Without a scoring run, for a scoring run of another
            perception run, an observation annotated under two samples, or a score whose
            claim is not a region claim of the run.
        RegionJoinError: If an observation cannot be joined (see :func:`join_frame_regions`).
    """
    if not scoring:
        raise SemanticScoringEvaluationError("scored regions need at least one scoring run")
    run_id = str(semantic.manifest.run_id)
    if semantic.manifest.region_artifact_id != regional.manifest.run_id:
        raise SemanticScoringEvaluationError(
            "selected regional producer contradicts semantic lineage"
        )
    visual_ids = {reader.manifest.lineage.visual_evidence_artifact_id for reader in scoring}
    if len(visual_ids) != 1:
        raise SemanticScoringEvaluationError(
            "scoring runs must share one explicitly selected visual producer"
        )
    for reader in scoring:
        lineage = reader.manifest.lineage
        if (
            lineage.semantic_interpretation_artifact_id != run_id
            or lineage.region_artifact_id != regional.manifest.run_id
        ):
            raise SemanticScoringEvaluationError(
                f"scoring run {reader.manifest.run_id!r} scored semantic producer "
                f"{lineage.semantic_interpretation_artifact_id!r}, not {run_id!r}"
            )
    scores: dict[tuple[str, str], list[SemanticScore]] = {}
    for reader in scoring:
        for score in reader.iter_scores():
            key = (str(score.source_observation_id), str(score.claim_id))
            scores.setdefault(key, []).append(score)
    frames = _frames_by_observation(regions)
    masks = regional.mask_store()
    all_regions = regional.read_regions()
    scored: list[ScoredRegion] = []
    joins: list[FrameRegionJoin] = []
    for observation in semantic.iter_observations():
        result = PerceptionResult(
            source_observation_id=observation.source_observation_id,
            sequence_artifact_id=semantic.manifest.sequence_artifact_id,
            region_artifact_id=regional.manifest.run_id,
            semantic_interpretation_artifact_id=run_id,
            claims=observation.claims,
            scene_context=observation.scene_context,
            regions=tuple(
                region
                for region in all_regions
                if region.source_observation_id == observation.source_observation_id
            ),
        )
        frame = frames.get(str(result.source_observation_id))
        join = None
        if frame is not None:
            # O leitor nunca materializa máscaras: sem carregá-las, a caixa entraria no IoU.
            run_regions = tuple(
                replace(region, mask=masks.load(result.source_observation_id, region.region_id))
                if region.mask is None and region.mask_reference is not None
                else region
                for region in result.regions
            )
            join = join_frame_regions(policy, frame, run_regions)
            joins.append(join)
        for region_id, claims in _region_claims(result).items():
            annotation = None
            reference_region = None if join is None else join.reference_region_of(region_id)
            if frame is not None and semantics is not None and reference_region is not None:
                record = semantics.find(frame.sample_id, frame.observation_id, reference_region)
                if record is not None:
                    annotation = semantics.to_semantic_annotation(record)
            scored.append(
                ScoredRegion(
                    claims=claims,
                    scores=tuple(
                        score
                        for claim in claims
                        for score in scores.pop(
                            (str(result.source_observation_id), str(claim.claim_id)), ()
                        )
                    ),
                    annotation=annotation,
                )
            )
    if scores:
        stray = sorted(claim_id for _, claim_id in scores)
        raise SemanticScoringEvaluationError(
            f"scores reference claims that are not region claims of the perception run: {stray}"
        )
    schemas: tuple[str, ...] = (AnnotationFamily.REGIONS.schema,)
    if semantics is not None:
        schemas += (_SEMANTICS_SCHEMA,)
    return JoinedScoredRegions(
        regions=tuple(scored),
        joins=tuple(joins),
        join_policy_id=policy.policy_id,
        semantic_interpretation_artifact_id=run_id,
        region_artifact_id=regional.manifest.run_id,
        visual_evidence_artifact_id=next(iter(visual_ids)),
        scoring_run_ids=tuple(str(reader.manifest.run_id) for reader in scoring),
        annotation_schemas=schemas,
    )


def _frames_by_observation(regions: RegionAnnotationSet) -> dict[str, FrameRegionAnnotation]:
    """Index the region annotation by observation; one observation has one sample."""
    frames: dict[str, FrameRegionAnnotation] = {}
    for frame in regions.frames:
        observation = str(frame.observation_id)
        if observation in frames:
            raise SemanticScoringEvaluationError(
                f"observation {observation!r} is annotated under samples "
                f"{frames[observation].sample_id!r} and {frame.sample_id!r}; the run cannot "
                "tell which one its regions belong to"
            )
        frames[observation] = frame
    return frames


def _region_claims(result: PerceptionResult) -> dict[RegionId, tuple[SemanticClaim, ...]]:
    """Group a result's region claims by region, in claim order; scene claims are left out."""
    grouped: dict[RegionId, list[SemanticClaim]] = {}
    for claim in result.claims:
        if claim.region_id is not None:
            grouped.setdefault(claim.region_id, []).append(claim)
    return {region_id: tuple(claims) for region_id, claims in grouped.items()}


def _check_regions(
    context: SemanticScoringEvaluationContext, regions: Sequence[ScoredRegion]
) -> None:
    """Refuse a region repeated within one repeat and annotations with no declared source."""
    seen: set[tuple[tuple[str, str, str], int]] = set()
    for region in regions:
        key = (region.region, region.repeat_index)
        if key in seen:
            raise SemanticScoringEvaluationError(
                f"region {region.region} appears twice in repeat {region.repeat_index}"
            )
        seen.add(key)
    if any(region.annotation is not None for region in regions):
        if context.region_join_policy_id is None:
            raise SemanticScoringEvaluationError(
                "run regions carry reference annotations, but the context names no region join "
                "policy: which reference region each run region was joined to is unaccounted for"
            )
        if not context.has_semantics:
            raise SemanticScoringEvaluationError(
                "run regions carry reference annotations, but the reference set provides no "
                f"{_SEMANTICS_SCHEMA} semantics annotations"
            )


def _join_summary(
    context: SemanticScoringEvaluationContext, joins: Sequence[FrameRegionJoin]
) -> RegionJoinSummary | None:
    """Count the joins, refusing any made under another policy than the context names."""
    if not joins:
        return None
    policies = {join.policy_id for join in joins}
    if policies != {context.region_join_policy_id}:
        raise SemanticScoringEvaluationError(
            f"the region joins come from join policy {sorted(policies)}, but the context names "
            f"{context.region_join_policy_id!r}"
        )
    return summarize_region_joins(joins)


def _quality(
    key: _ScorerKey,
    regions: Sequence[ScoredRegion],
    context: SemanticScoringEvaluationContext,
    policy: SemanticScoringEvaluationPolicy,
    claim_count: int,
) -> ScorerSupportQuality:
    """Label every score of ``key`` in the primary run and compute its support metrics."""
    samples: list[LabeledScore] = []
    by_region: list[tuple[list[float], list[float]]] = []
    for region in regions:
        region_positives: list[float] = []
        region_negatives: list[float] = []
        for claim, score in _scored_claims(region, key):
            label = _label(claim, region.annotation)
            samples.append(_labeled(claim, score, label))
            if label is ClaimLabel.POSITIVE:
                region_positives.append(score.value)
            elif label is ClaimLabel.NEGATIVE:
                region_negatives.append(score.value)
        by_region.append((region_positives, region_negatives))
    positives = [value for region_positives, _ in by_region for value in region_positives]
    negatives = [value for _, region_negatives in by_region for value in region_negatives]
    eligible = [(pos, neg) for pos, neg in by_region if pos and neg]
    ranking = _ranking(eligible)
    auroc, auprc = _pooled(positives, negatives, policy.minimum_class_count)
    top1, pairwise = ranking.top1, ranking.pairwise
    if not context.has_semantics:
        # Sem a família nada foi rotulado: o motivo é a referência, não o tamanho da população.
        missing = _not_applicable(
            0, f"the reference set provides no {_SEMANTICS_SCHEMA} annotations"
        )
        top1 = pairwise = auroc = auprc = missing
    scorer, space, score_type = key
    return ScorerSupportQuality(
        scorer=scorer,
        embedding_space_id=space,
        score_type=score_type,
        claim_count=claim_count,
        scored_claim_count=len(samples),
        unscored_claim_count=claim_count - len(samples),
        positive_count=len(positives),
        negative_count=len(negatives),
        unassessed_count=len(samples) - len(positives) - len(negatives),
        positive_scores=_distribution(positives),
        negative_scores=_distribution(negatives),
        ranking_region_count=len(eligible),
        top1_tie_count=ranking.top1_ties,
        pairwise_tie_count=ranking.pair_ties,
        top1_accuracy=top1,
        pairwise_accuracy=pairwise,
        auroc=auroc,
        auprc=auprc,
        samples=tuple(samples),
    )


@dataclass(frozen=True, kw_only=True)
class _Ranking:
    """Both ranking metrics over the eligible regions, with their declared ties."""

    top1: SupportMetric
    pairwise: SupportMetric
    top1_ties: int
    pair_ties: int


def _ranking(eligible: Sequence[tuple[list[float], list[float]]]) -> _Ranking:
    """Return the top-1 and the per-region pairwise accuracy over the eligible regions."""
    if not eligible:
        reason = "no region of the primary run has both a positive and a negative scored claim"
        return _Ranking(
            top1=_not_applicable(0, reason),
            pairwise=_not_applicable(0, reason),
            top1_ties=0,
            pair_ties=0,
        )
    # Empate no topo entre positiva e negativa não é acerto: o scorer não as separou.
    correct = sum(max(pos) > max(neg) for pos, neg in eligible)
    pairs = [(_ordered_pairs(pos, neg), len(pos) * len(neg)) for pos, neg in eligible]
    return _Ranking(
        top1=SupportMetric(value=correct / len(eligible), sample_count=len(eligible)),
        pairwise=SupportMetric(
            value=sum(credit / total for (credit, _), total in pairs) / len(eligible),
            sample_count=len(eligible),
        ),
        top1_ties=sum(max(pos) == max(neg) for pos, neg in eligible),
        pair_ties=sum(ties for (_, ties), _ in pairs),
    )


def _pooled(
    positives: Sequence[float], negatives: Sequence[float], minimum: int
) -> tuple[SupportMetric, SupportMetric]:
    """Return the pooled AUROC and AUPRC, or not applicable below the class minimum."""
    count = len(positives) + len(negatives)
    if len(positives) < minimum or len(negatives) < minimum:
        reason = (
            f"needs at least {minimum} positive and {minimum} negative scored claims; found "
            f"{len(positives)} positive and {len(negatives)} negative"
        )
        return _not_applicable(count, reason), _not_applicable(count, reason)
    credit, _ = _ordered_pairs(positives, negatives)
    auroc = credit / (len(positives) * len(negatives))
    return (
        SupportMetric(value=auroc, sample_count=count),
        SupportMetric(value=_average_precision(positives, negatives), sample_count=count),
    )


def _ordered_pairs(positives: Sequence[float], negatives: Sequence[float]) -> tuple[float, int]:
    """Return the credit of every (positive, negative) pair and the number of tied pairs.

    A pair scores one when the positive is higher and one half when both are equal, the
    Mann-Whitney convention that makes a constant scorer worth exactly chance.
    """
    ordered = sorted(negatives)
    credit = 0.0
    ties = 0
    for value in positives:
        below = bisect_left(ordered, value)
        equal = bisect_right(ordered, value) - below
        credit += below + equal / 2
        ties += equal
    return credit, ties


def _average_precision(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Return the average precision of the positives ranked by descending score.

    Each distinct score is one threshold, so tied scores enter together: the sum over
    thresholds of the recall gained times the precision at that threshold.
    """
    labeled = sorted(
        [(value, True) for value in positives] + [(value, False) for value in negatives],
        key=lambda item: -item[0],
    )
    true_positives = false_positives = 0
    recall = precision_sum = 0.0
    index = 0
    while index < len(labeled):
        threshold = labeled[index][0]
        while index < len(labeled) and labeled[index][0] == threshold:
            if labeled[index][1]:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        new_recall = true_positives / len(positives)
        precision_sum += (new_recall - recall) * true_positives / (true_positives + false_positives)
        recall = new_recall
    return precision_sum


def _diagnostics(key: _ScorerKey, regions: Sequence[ScoredRegion]) -> ScorerDiagnostics:
    """Describe every score of ``key`` in the primary run without reading the reference."""
    values: list[float] = []
    agreements = ties = eligible = 0
    for region in regions:
        scored = list(_scored_claims(region, key))
        values.extend(score.value for _, score in scored)
        primaries = [score.value for claim, score in scored if claim.role is HypothesisRole.PRIMARY]
        others = [
            score.value for claim, score in scored if claim.role is not HypothesisRole.PRIMARY
        ]
        if len(primaries) != 1 or not others:
            continue
        eligible += 1
        agreements += primaries[0] > max(others)
        ties += primaries[0] == max(others)
    scorer, space, score_type = key
    return ScorerDiagnostics(
        scorer=scorer,
        embedding_space_id=space,
        score_type=score_type,
        scores=_distribution(values),
        primary_agreement=(
            SupportMetric(value=agreements / eligible, sample_count=eligible)
            if eligible
            else _not_applicable(
                0, "no region has a scored VLM primary claim and another scored claim"
            )
        ),
        primary_agreement_tie_count=ties,
    )


def _scored_claims(
    region: ScoredRegion, key: _ScorerKey
) -> Iterable[tuple[SemanticClaim, SemanticScore]]:
    """Yield the region's claims that have a score of ``key``, in claim order."""
    scores = {score.claim_id: score for score in region.scores if _scorer_key(score) == key}
    for claim in region.claims:
        score = scores.get(claim.claim_id)
        if score is not None:
            yield claim, score


def _label(claim: SemanticClaim, annotation: SemanticAnnotation | None) -> ClaimLabel:
    """Label a claim under :data:`MATCHING_POLICY`; only a rejected hypothesis is negative."""
    if annotation is None:
        return ClaimLabel.UNASSESSED
    hypothesis = _MATCHING.key(claim.hypothesis)
    if hypothesis in {_MATCHING.key(value) for value in annotation.acceptable_hypotheses}:
        return ClaimLabel.POSITIVE
    if hypothesis in {_MATCHING.key(value) for value in annotation.rejected_hypotheses}:
        return ClaimLabel.NEGATIVE
    return ClaimLabel.UNASSESSED


def _labeled(claim: SemanticClaim, score: SemanticScore, label: ClaimLabel) -> LabeledScore:
    return LabeledScore(
        score_id=str(score.score_id),
        claim_id=str(claim.claim_id),
        source_observation_id=str(claim.source_observation_id),
        region_id=str(claim.region_id),
        hypothesis=claim.hypothesis,
        role=claim.role,
        label=label,
        value=score.value,
    )


def _distribution(values: Sequence[float]) -> ScoreDistribution:
    return ScoreDistribution(
        count=len(values),
        minimum=min(values) if values else None,
        p25=nearest_rank_percentile(values, 25),
        median=nearest_rank_percentile(values, 50),
        p75=nearest_rank_percentile(values, 75),
        maximum=max(values) if values else None,
    )


def _not_applicable(sample_count: int, reason: str) -> SupportMetric:
    return SupportMetric(value=None, sample_count=sample_count, not_applicable_reason=reason)


def _claim_region(claim: SemanticClaim) -> tuple[str, str, str]:
    """Return the region a claim describes; a scene claim has none and is refused."""
    if claim.region_id is None:
        raise ValueError(
            f"claim {claim.claim_id!r} is a scene claim; a scored region holds only region claims "
            "(scene claims are not scored, #652)"
        )
    return (
        str(claim.source_observation_id),
        str(claim.semantic_interpretation_run_id),
        str(claim.region_id),
    )


def _scorer_key(score: SemanticScore) -> _ScorerKey:
    return (score.provenance, score.embedding_space_id, score.score_type)


def _scorer_order(key: _ScorerKey) -> tuple[str, ...]:
    """Return a total order over scorer keys, so the blocks never depend on hash order."""
    scorer, space, score_type = key
    return (
        scorer.backend_id,
        scorer.capability,
        scorer.provider,
        scorer.model,
        scorer.version,
        scorer.configuration_fingerprint or "",
        space,
        score_type.value,
    )


def _plain(value: object) -> object:
    """Convert report dataclasses, enums and sequences into JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
