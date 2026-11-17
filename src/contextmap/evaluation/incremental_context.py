"""Avaliação da invariância de partição em builds de contexto incremental.

Compara estado geométrico por conteúdo e claims por métricas estatísticas pré-definidas.
Lê artifacts imutáveis sem alterar regras científicas ou resultados de execução.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Set
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from contextmap.artifact import (
    ContextMapArtifactReader,
    Severity,
    ValidationLevel,
    validate_context_map_artifact,
)
from contextmap.geometric_mapping import GeometryReference
from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.runtime import (
    CONTEXT_STAGES,
    EXECUTION_FILENAME,
    effective_maps_root,
    read_context_build,
    read_plan_document,
)
from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.semantic_interpretation import SemanticInterpretationRunReader
from contextmap.semantic_mapping import SemanticMappingRunReader

_MAX_EXAMPLES = 20
"""Items listed per side of a divergent dimension; the counts always cover every item."""

_COMPARED_DIMENSIONS = (
    "map_id",
    "physical_observations",
    "inference_results",
    "excluded_observations",
    "supports",
    "contributions",
    "entities",
    "rejections",
    "resolved_entities",
    "relations",
    "capabilities",
)
"""The layer-1 dimensions that decide equality, in report order; regions only diagnose."""

ClaimKey = tuple[str, str | None, str, str, str | None, float | None]
"""A canonical claim: frame, regional content key, hypothesis, role, category, confidence."""

_LAYER_TWO_FUSION_COUNTS = (
    "claims",
    "hypotheses",
    "evidence_stances",
    "abstaining_claims",
    "uncertainty",
    "supports_with_uncertainty",
    "scorer_signals",
)
"""The fusion counts that depend on claims: layer 2, reported side by side, never decided."""

_REQUIRED_CONTEXT_STAGES = ("visual_perception", "sensor_association")
"""The context stages every build consumes; point representation only when enabled."""

_Key = TypeVar("_Key", bound=Hashable)


@dataclasses.dataclass(frozen=True, order=True)
class GeometryKey:
    """The foundation geometry a support or an entity covers, as the set of its element ids.

    Every build of one foundation references the same geometry ids, so this key pairs supports
    and entities across builds whatever local identity each execution gave them.

    Attributes:
        geometry_ids: The ids, sorted and unique; the order they were given in is not identity.
    """

    geometry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Canonicalize the ids and refuse an empty key.

        Raises:
            ValueError: If there is no geometry id.
        """
        ids = tuple(sorted(set(self.geometry_ids)))
        if not ids:
            raise ValueError("a geometry key needs at least one geometry id")
        object.__setattr__(self, "geometry_ids", ids)

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the ids, one per line, as hexadecimal."""
        return hashlib.sha256("\n".join(self.geometry_ids).encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        """Return a short, stable rendering: a digest prefix and the number of elements."""
        return f"geometry:{self.digest[:16]}x{len(self.geometry_ids)}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerOneState:
    """What a build concluded that no semantic backend reaches, free of execution identities.

    Every collection is sorted, so two equal states are equal values.

    Attributes:
        map_id: The foundation map every geometry key refers to.
        physical_observations: Distinct physical frames in the fusion.
        inference_results: Distinct inference results in the fusion.
        excluded_observations: Spatial observations that took part in no support.
        supports: The geometry of each fusion support.
        contributions: One ``(support geometry, frame, region)`` per contribution; a frame
            inferred twice contributes twice.
        entities: The geometry of each semantic entity.
        rejections: ``(reason, count)`` of the candidates that did not become entities.
        resolved_entities: For each entity of the ContextMap, its geometry and the geometry of
            each semantic entity it was resolved from.
        relations: ``(predicate, subject geometry, object geometry, state, uncertainty kinds)``.
        capabilities: The content and relation predicates the ContextMap declares.
        regions: Diagnostic only: per frame, the regions of each inference result as
            ``(region id, x, y, width, height, kind, accepted)``. Region discovery runs before
            any claim, so it locates a divergence; it never decides equality.
    """

    map_id: str
    physical_observations: int
    inference_results: int
    excluded_observations: int
    supports: tuple[GeometryKey, ...]
    contributions: tuple[tuple[GeometryKey, str, str], ...]
    entities: tuple[GeometryKey, ...]
    rejections: tuple[tuple[str, int], ...]
    resolved_entities: tuple[tuple[GeometryKey, tuple[GeometryKey, ...]], ...]
    relations: tuple[tuple[str, GeometryKey, GeometryKey, str, tuple[str, ...]], ...]
    capabilities: tuple[str, ...]
    regions: tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerTwoState:
    """What a build concluded that depends on the claims, keyed for pairing across builds.

    Attributes:
        claims: For every frame the build's perception runs perceived, its canonical claim
            keys; a frame without a claim maps to an empty set.
        entity_states: The ambiguity state of each semantic entity, by its geometry.
        entity_labels: The set of hypothesis labels of each semantic entity, by its geometry.
        fusion: The fusion counts that depend on claims, as the fusion run recorded them.
    """

    claims: Mapping[str, frozenset[ClaimKey]]
    entity_states: Mapping[GeometryKey, str]
    entity_labels: Mapping[GeometryKey, frozenset[str]]
    fusion: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class BuildCounts:
    """The evidence accounting of one build, each count reported on its own.

    Attributes:
        physical_observations: Distinct physical frames the fusion grouped.
        inference_results: Distinct inference results the fusion grouped.
        supports: Fusion supports.
        contributions: Fusion contributions.
        excluded_observations: Spatial observations that took part in no support.
        semantic_entities: Entities the semantic mapping materialized.
        rejected_candidates: Candidates the semantic mapping could not materialize.
        resolved_entities: Entities of the ContextMap.
        relations: Relation records of the ContextMap, whatever their state.
        relations_by_state: Relation records per state.
    """

    physical_observations: int
    inference_results: int
    supports: int
    contributions: int
    excluded_observations: int
    semantic_entities: int
    rejected_candidates: int
    resolved_entities: int
    relations: int
    relations_by_state: Mapping[str, int]


@dataclasses.dataclass(frozen=True, kw_only=True)
class MapValidation:
    """The full-level validation of a ContextMapArtifact by its own validator.

    Attributes:
        status: The validator's status value (``verified`` for a map that checks out).
        errors: ``"<code>: <message>"`` of every error finding.
        warnings: The same for every warning finding.
    """

    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class BuildState:
    """Everything the comparison reads from one materialized ContextBuild.

    Attributes:
        context_build_id: The build's identity, an execution identity never compared.
        context_run_ids: The ContextRuns the build froze.
        context_map_id: The snapshot the build produced.
        counts: The evidence accounting.
        layer_one: The exact state.
        layer_two: The claim-dependent state.
        validation: The full validation of the ContextMap.
        integrity_problems: Inventory problems of the runs read, prefixed by the run; empty
            when every file is as its manifest records it.
    """

    context_build_id: str
    context_run_ids: tuple[str, ...]
    context_map_id: str
    counts: BuildCounts
    layer_one: LayerOneState
    layer_two: LayerTwoState
    validation: MapValidation
    integrity_problems: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class DimensionDifference:
    """How one layer-1 dimension of two states compares.

    Attributes:
        name: The dimension.
        equal: Whether both sides hold the same multiset of items.
        only_left: Items only the left side holds, counted with multiplicity.
        only_right: Items only the right side holds.
        left_examples: The first of those items, rendered, at most twenty.
        right_examples: The same for the right side.
    """

    name: str
    equal: bool
    only_left: int
    only_right: int
    left_examples: tuple[str, ...]
    right_examples: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerOneComparison:
    """The exact comparison of two layer-1 states.

    Attributes:
        equal: Whether every compared dimension is equal. Regions never take part.
        dimensions: Every compared dimension, in report order.
        region_frames_differing: Frames whose regions differ, to locate a divergence.
    """

    equal: bool
    dimensions: tuple[DimensionDifference, ...]
    region_frames_differing: tuple[str, ...]


class LayerOneOutcome(Enum):
    """What layer 1 says about partition invariance (H1).

    Attributes:
        HOLDS: The chunked build equals both whole builds, which equal each other.
        PARTITION_DIVERGENCE: The whole builds agree and the chunked one differs: a divergence
            caused by the partition, treated as a bug until proven otherwise.
        PRECONDITION_FAILED: The two whole builds already differ, so layer 1 is not exact under
            rerun. A finding in itself; any reclassification is a new decision, never a
            silent tolerance.
    """

    HOLDS = "holds"
    PARTITION_DIVERGENCE = "partition_divergence"
    PRECONDITION_FAILED = "precondition_failed"


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayerOneVerdict:
    """H1 and the three comparisons it rests on.

    Attributes:
        outcome: The verdict.
        precondition: ``W1`` against ``W2``.
        chunked_w1: ``C`` against ``W1``.
        chunked_w2: ``C`` against ``W2``.
    """

    outcome: LayerOneOutcome
    precondition: LayerOneComparison
    chunked_w1: LayerOneComparison
    chunked_w2: LayerOneComparison


@dataclasses.dataclass(frozen=True, kw_only=True)
class ClusterTerms:
    """The agreement between two builds, split into the clusters a bootstrap resamples.

    Attributes:
        clusters: Cluster identities, unique: frames for claims, entities for entity state.
        agreeing: Per cluster, the items both sides agree on.
        compared: Per cluster, the items compared; a cluster with nothing to compare has 0.
    """

    clusters: tuple[str, ...]
    agreeing: tuple[int, ...]
    compared: tuple[int, ...]

    def __post_init__(self) -> None:
        """Refuse terms that do not describe one count pair per unique cluster.

        Raises:
            ValueError: If the lengths differ, a cluster repeats, or a count is negative or an
                agreement exceeds what was compared.
        """
        if not len(self.clusters) == len(self.agreeing) == len(self.compared):
            raise ValueError("cluster terms need one count per cluster in each tuple")
        if len(set(self.clusters)) != len(self.clusters):
            raise ValueError("cluster identities must be unique")
        for agreeing, compared in zip(self.agreeing, self.compared, strict=True):
            if agreeing < 0 or agreeing > compared:
                raise ValueError(
                    f"agreeing items must be non-negative and cannot exceed the compared ones, "
                    f"got {agreeing} of {compared}"
                )

    @property
    def value(self) -> float | None:
        """Return the agreement over every cluster, or ``None`` when nothing was compared."""
        compared = sum(self.compared)
        return None if compared == 0 else sum(self.agreeing) / compared


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartitionEffect:
    """The partition effect on one layer-2 agreement, with its bootstrap interval.

    ``point`` is ``(agr(W1,C) + agr(W2,C)) / 2 - agr(W1,W2)``: how much the chunked build
    agrees with the whole ones beyond what two whole reruns agree with each other.

    Attributes:
        w1_c: Agreement between ``W1`` and ``C``.
        w2_c: Agreement between ``W2`` and ``C``.
        w1_w2: Agreement between the two whole builds: the rerun noise without partition.
        point: The effect, or ``None`` when an agreement has nothing to compare.
        lower: Lower bound of the percentile interval, or ``None`` when not measurable.
        upper: Upper bound of the interval.
        clusters: Clusters resampled: the union of the clusters of the three pairs.
        resamples: Bootstrap resamples drawn.
        undefined_resamples: Resamples in which some pair had nothing to compare; they are
            left out of the interval and counted here, never replaced.
        seed: Seed of the generator (``numpy.random.default_rng``).
        confidence: Coverage of the interval.
    """

    w1_c: float | None
    w2_c: float | None
    w1_w2: float | None
    point: float | None
    lower: float | None
    upper: float | None
    clusters: int
    resamples: int
    undefined_resamples: int
    seed: int
    confidence: float


class EquivalenceDecision(Enum):
    """The pre-registered decision on one layer-2 metric (H2).

    Attributes:
        REJECTS_INVARIANCE: The whole interval is below zero: the partition lowers agreement
            beyond rerun noise, a real scientific difference to investigate.
        EQUIVALENT_WITHIN_DELTA: Otherwise, when the lower bound is at least ``-delta``.
        INCONCLUSIVE: Neither; recorded as a known limitation.
        NOT_MEASURABLE: Some agreement had nothing to compare, so there is no interval.
    """

    REJECTS_INVARIANCE = "rejects_invariance"
    EQUIVALENT_WITHIN_DELTA = "equivalent_within_delta"
    INCONCLUSIVE = "inconclusive"
    NOT_MEASURABLE = "not_measurable"


def _context_map_dir(build_run: Path, workspace: Path, maps_root: Path | None) -> Path:
    """Return the ContextMapArtifact directory of a build.

    A build publishes its ``context_map`` into the maps root, isolated from the run; the run's
    execution record names its location, relative to that root. An older build that kept the
    ``context_map`` under the run is still read in place.
    """
    local = build_run / "context_map"
    if local.is_dir():
        return local
    maps_root = effective_maps_root(workspace, maps_root)
    execution = read_plan_document(build_run / EXECUTION_FILENAME)
    for stage in execution.get("stages", ()):
        if stage.get("stage_id") == "context_map":
            location = Path(str(stage["output"]["location"]))
            root = workspace if location.parts[0] == "artifacts" else maps_root
            return root / location
    return local


def read_build_state(
    build_run: Path, *, workspace: Path, maps_root: Path | None = None
) -> BuildState:
    """Read a materialized ContextBuild into its two layers, its counts and its validation.

    Only public readers are used, and every execution-local identity is replaced by content:
    supports and entities by the foundation geometry they cover, resolved entities by their
    geometry and their members', relations by the geometry of their entities, claims by their
    canonical key. The perception runs are the exact ones the build froze.

    Args:
        build_run: The journal directory of a completed build, with ``context_build.json``
            and ``execution.json`` naming the exact stage artifacts.
        workspace: The run store the build's upstream artifact locations are relative to.
        maps_root: The maps root a ``context build`` publishes its ``context_map`` into. When
            the ``context_map`` is not under ``build_run`` (the build published it to the maps
            root, the default layout), it is resolved from the build's execution record against
            this root. ``None`` means the default maps root, ``<workspace>/../maps``.

    Returns:
        The state.

    Raises:
        ValueError: If two entities of one build cover the same geometry, so the geometric
            key could not pair them across builds.
    """
    build = read_context_build(build_run, workspace=workspace)
    execution = read_plan_document(build_run / EXECUTION_FILENAME)
    outputs = {stage["stage_id"]: stage["output"] for stage in execution["stages"]}
    fusion = SemanticFusionRunReader(workspace / outputs["semantic_fusion"]["location"])
    mapping = SemanticMappingRunReader(workspace / outputs["semantic_mapping"]["location"])
    regional_runs = {
        ref.artifact_id: RegionPerceptionRunReader(workspace / ref.location)
        for ref in build.provided.get("region_perception", ())
        if ref.location is not None
    }
    region_keys = _region_content_keys(regional_runs.values())
    interpretations = {
        ref.artifact_id: SemanticInterpretationRunReader(workspace / ref.location)
        for ref in build.provided.get("semantic_interpretation", ())
        if ref.location is not None
    }
    recorded = fusion.read_record("metrics/counts.json")
    outcomes = list(fusion.iter_outcomes())
    entities = list(mapping.iter_entities())
    rejections = Counter(item.reason.value for item in mapping.rejected_candidates())
    entity_key = _unique_keys(
        (str(entity.entity_id), _key(entity.geometry.geometry_refs)) for entity in entities
    )
    map_directory = _context_map_dir(build_run, workspace, maps_root)
    with ContextMapArtifactReader.open(map_directory) as reader:
        context_map = reader.context_map()
    resolved_key = _unique_keys(
        (str(entity.entity_id), _key(entity.geometry_refs)) for entity in context_map.entities
    )
    results = [
        result for reader in interpretations.values() for result in reader.iter_observations()
    ]
    declared = context_map.metadata.capabilities
    layer_one = LayerOneState(
        map_id=str(context_map.geometry_ref.map_id),
        physical_observations=recorded["physical_observations"],
        inference_results=recorded["inference_results"],
        excluded_observations=recorded["excluded_observations"],
        supports=tuple(sorted(_key(item.support.geometry_support) for item in outcomes)),
        contributions=tuple(
            sorted(
                (
                    _key(item.support.geometry_support),
                    str(contribution.physical_observation_id),
                    _region_key(region_keys, str(contribution.region_id)),
                )
                for item in outcomes
                for contribution in item.evidence.contributions
            )
        ),
        entities=tuple(sorted(entity_key.values())),
        rejections=tuple(sorted(rejections.items())),
        resolved_entities=tuple(
            sorted(
                (
                    resolved_key[str(entity.entity_id)],
                    tuple(sorted(entity_key[str(m.entity_id)] for m in entity.member_entities)),
                )
                for entity in context_map.entities
            )
        ),
        relations=tuple(
            sorted(
                (
                    relation.predicate.value,
                    resolved_key[str(relation.subject.entity_id)],
                    resolved_key[str(relation.object.entity_id)],
                    relation.state.value,
                    tuple(sorted(kind.value for kind in relation.uncertainty_kinds)),
                )
                for relation in context_map.relations
            )
        ),
        capabilities=tuple(
            [f"content:{item.value}" for item in declared.content]
            + [f"predicate:{item.value}" for item in declared.relation_predicates]
        ),
        regions=_regions(regional_runs.values(), region_keys),
    )
    claims: dict[str, set[ClaimKey]] = {}
    for result in results:
        claims.setdefault(str(result.source_observation_id), set()).update(
            (
                str(claim.source_observation_id),
                None if claim.region_id is None else _region_key(region_keys, str(claim.region_id)),
                claim.hypothesis,
                claim.role.value,
                claim.category,
                claim.confidence,
            )
            for claim in result.claims
        )
    layer_two = LayerTwoState(
        claims={frame: frozenset(keys) for frame, keys in claims.items()},
        entity_states={
            entity_key[str(e.entity_id)]: e.semantic_state.ambiguity_state.value for e in entities
        },
        entity_labels={
            entity_key[str(e.entity_id)]: frozenset(h.label for h in e.semantic_state.hypotheses)
            for e in entities
        },
        fusion={name: recorded[name] for name in _LAYER_TWO_FUSION_COUNTS},
    )
    counts = BuildCounts(
        physical_observations=recorded["physical_observations"],
        inference_results=recorded["inference_results"],
        supports=recorded["supports"],
        contributions=recorded["contributions"],
        excluded_observations=recorded["excluded_observations"],
        semantic_entities=len(entities),
        rejected_candidates=sum(rejections.values()),
        resolved_entities=len(context_map.entities),
        relations=len(context_map.relations),
        relations_by_state=dict(
            sorted(Counter(relation.state.value for relation in context_map.relations).items())
        ),
    )
    problems = [f"semantic_fusion: {problem}" for problem in fusion.verify_integrity()]
    problems += [f"semantic_mapping: {problem}" for problem in mapping.verify_integrity()]
    for run_id, regional in sorted(regional_runs.items()):
        problems += [f"region_perception {run_id}: {p}" for p in regional.verify_integrity()]
    for run_id, semantic in sorted(interpretations.items()):
        problems += [f"semantic_interpretation {run_id}: {p}" for p in semantic.verify_integrity()]
    return BuildState(
        context_build_id=str(build.identity),
        context_run_ids=tuple(str(item) for item in build.context_run_ids),
        context_map_id=str(context_map.context_map_id),
        counts=counts,
        layer_one=layer_one,
        layer_two=layer_two,
        validation=_validate(map_directory),
        integrity_problems=tuple(problems),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class RepeatedInferenceVerdict:
    """Scenario 2: two inferences over the same frames are correlated evidence of them.

    Attributes:
        holds: The build over both runs has the frames of each and the results of both.
        physical_observations: ``(W1, W2, R)``.
        inference_results: ``(W1, W2, R)``.
    """

    holds: bool
    physical_observations: tuple[int, int, int]
    inference_results: tuple[int, int, int]


@dataclasses.dataclass(frozen=True, kw_only=True)
class DownstreamOnlyVerdict:
    """Scenario 3: a downstream policy change reuses the frozen context evidence.

    Attributes:
        holds: Both builds were given every context stage they consume and executed none, and
            their identities differ. Their outputs may differ: that is what the policy changes.
        recomputed_context_stages: ``"<build>:<stage>"`` of each context stage a build ran
            instead of being given; empty when it holds.
        identities_differ: Whether the two builds have different identities.
    """

    holds: bool
    recomputed_context_stages: tuple[str, ...]
    identities_differ: bool


def stage_dispositions(run_directory: Path) -> dict[str, str]:
    """Return how each stage of a completed run got its artifact, from ``execution.json``.

    Args:
        run_directory: A completed runtime run.

    Returns:
        By stage: ``"provided"`` for an artifact the run was given, the reuse decision
        (``"reused"`` or ``"recomputed"``) of a stage the runner decided on, and
        ``"executed"`` for a stage run without a reuse policy.

    Raises:
        PlanDocumentError: If the run has no readable execution record.
    """
    record = read_plan_document(run_directory / "execution.json")
    dispositions = {stage_id: "provided" for stage_id in record["reused"]}
    for stage in record["stages"]:
        decision = stage["decision"]
        dispositions[stage["stage_id"]] = "executed" if decision is None else decision["kind"]
    return dict(sorted(dispositions.items()))


def judge_repeated_inference(
    *, w1: BuildCounts, w2: BuildCounts, both: BuildCounts
) -> RepeatedInferenceVerdict:
    """Decide scenario 2 exactly: frames kept, inference results added.

    Args:
        w1: Counts of the build over the first run.
        w2: Counts of the build over the second run of the same frames.
        both: Counts of the build over both runs.

    Returns:
        The verdict; it holds when the three builds count the same physical observations and
        the build over both counts the inference results of each run.
    """
    physical = (w1.physical_observations, w2.physical_observations, both.physical_observations)
    results = (w1.inference_results, w2.inference_results, both.inference_results)
    return RepeatedInferenceVerdict(
        holds=len(set(physical)) == 1 and results[2] == results[0] + results[1],
        physical_observations=physical,
        inference_results=results,
    )


def judge_downstream_only(
    *,
    baseline: Mapping[str, str],
    changed: Mapping[str, str],
    baseline_build: str,
    changed_build: str,
) -> DownstreamOnlyVerdict:
    """Decide scenario 3 structurally: no context stage is recomputed by a policy change.

    Args:
        baseline: :func:`stage_dispositions` of the baseline build.
        changed: The dispositions of the build under the changed downstream policy.
        baseline_build: Identity of the baseline build.
        changed_build: Identity of the changed build.

    Returns:
        The verdict. Perception and association are always required, and any other context
        stage (point representation) is required once either build has it; a required stage
        missing from a build counts as not given, so a build without context evidence fails.
    """
    required = sorted({*_REQUIRED_CONTEXT_STAGES, *({*baseline, *changed} & set(CONTEXT_STAGES))})
    recomputed = tuple(
        f"{name}:{stage}"
        for name, dispositions in (("baseline", baseline), ("changed", changed))
        for stage in required
        if dispositions.get(stage) != "provided"
    )
    differ = baseline_build != changed_build
    return DownstreamOnlyVerdict(
        holds=not recomputed and differ,
        recomputed_context_stages=recomputed,
        identities_differ=differ,
    )


def compare_layer_one(left: LayerOneState, right: LayerOneState) -> LayerOneComparison:
    """Compare two layer-1 states exactly, dimension by dimension.

    Collections are compared as multisets; a scalar is a one-item collection, so a divergent
    scalar lists both values. Regions are compared per frame only to locate a divergence.

    Args:
        left: One state.
        right: The other state.

    Returns:
        The comparison.
    """
    dimensions = tuple(
        _difference(name, getattr(left, name), getattr(right, name))
        for name in _COMPARED_DIMENSIONS
    )
    left_regions, right_regions = dict(left.regions), dict(right.regions)
    differing = tuple(
        frame
        for frame in sorted(set(left_regions) | set(right_regions))
        if left_regions.get(frame) != right_regions.get(frame)
    )
    return LayerOneComparison(
        equal=all(item.equal for item in dimensions),
        dimensions=dimensions,
        region_frames_differing=differing,
    )


def judge_layer_one(
    *, w1: LayerOneState, w2: LayerOneState, chunked: LayerOneState
) -> LayerOneVerdict:
    """Decide H1: the chunked build's layer 1 equals both whole builds'.

    Args:
        w1: The first whole build.
        w2: The second whole build, a rerun of the first.
        chunked: The build over the chunks.

    Returns:
        The verdict. The precondition ``W1 == W2`` is checked first: without it, a difference
        of ``C`` cannot be told apart from rerun nondeterminism.
    """
    precondition = compare_layer_one(w1, w2)
    chunked_w1 = compare_layer_one(chunked, w1)
    chunked_w2 = compare_layer_one(chunked, w2)
    if not precondition.equal:
        outcome = LayerOneOutcome.PRECONDITION_FAILED
    elif chunked_w1.equal and chunked_w2.equal:
        outcome = LayerOneOutcome.HOLDS
    else:
        outcome = LayerOneOutcome.PARTITION_DIVERGENCE
    return LayerOneVerdict(
        outcome=outcome,
        precondition=precondition,
        chunked_w1=chunked_w1,
        chunked_w2=chunked_w2,
    )


def claim_agreement(
    left: Mapping[str, Set[ClaimKey]], right: Mapping[str, Set[ClaimKey]]
) -> ClusterTerms:
    """Split ``M_claim`` into per-frame terms.

    ``M_claim`` is the size of the intersection over the size of the union of the canonical
    claim keys of two builds, the definition of the v0.1.0 rerun comparison (29/90). Frames are
    the clusters: each contributes its intersection and its union, and the sums over frames give
    that Jaccard. A frame only one side perceived still counts
    its claims as disagreement.

    Args:
        left: The claim keys of each frame one build perceived, empty sets included.
        right: The same for the other build.

    Returns:
        One term per frame either side perceived, in frame order.
    """
    frames = tuple(sorted(set(left) | set(right)))
    empty: frozenset[ClaimKey] = frozenset()
    agreeing = tuple(len(left.get(f, empty) & right.get(f, empty)) for f in frames)
    compared = tuple(len(left.get(f, empty) | right.get(f, empty)) for f in frames)
    return ClusterTerms(clusters=frames, agreeing=agreeing, compared=compared)


def paired_agreement(left: Mapping[_Key, object], right: Mapping[_Key, object]) -> ClusterTerms:
    """Split an entity agreement (``M_state``, ``M_label``) into per-entity terms.

    Entities are paired by key (their geometry); only a pair is compared, and it agrees when
    both sides hold the same value. An entity without a pair is a cluster with nothing to
    compare: layer 1 accounts for it, not this agreement.

    Args:
        left: The value of each entity of one build, by key.
        right: The same for the other build.

    Returns:
        One term per key of either side, in key order, the cluster named by ``str(key)``.
    """
    keys = sorted(set(left) | set(right))  # type: ignore[type-var]
    return ClusterTerms(
        clusters=tuple(str(key) for key in keys),
        agreeing=tuple(int(k in left and k in right and left[k] == right[k]) for k in keys),
        compared=tuple(int(k in left and k in right) for k in keys),
    )


def partition_effect(
    w1_c: ClusterTerms,
    w2_c: ClusterTerms,
    w1_w2: ClusterTerms,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> PartitionEffect:
    """Estimate the partition effect ``D`` with a paired cluster bootstrap.

    Each resample draws, with replacement, as many clusters as the union of the three pairs
    holds; the same draw feeds the three pairs (paired), and each agreement is the ratio of its
    summed terms over the drawn clusters. The interval is the percentile interval of ``D``.

    Args:
        w1_c: Terms of ``W1`` against ``C``.
        w2_c: Terms of ``W2`` against ``C``.
        w1_w2: Terms of ``W1`` against ``W2``.
        resamples: Number of resamples, at least 1 (10,000 in the pre-registered protocol).
        seed: Non-negative seed of ``numpy.random.default_rng``.
        confidence: Coverage of the interval, strictly between 0 and 1.

    Returns:
        The effect and its interval.

    Raises:
        ValueError: If a bootstrap parameter is impossible.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    # Import local: importar a avaliação não carrega o NumPy; só o bootstrap precisa dele.
    import numpy as np

    pairs = (w1_c, w2_c, w1_w2)
    universe = sorted(set().union(*(terms.clusters for terms in pairs)))
    position = {cluster: index for index, cluster in enumerate(universe)}
    arrays = [_aligned(terms, position) for terms in pairs]
    values = [terms.value for terms in pairs]
    point = _effect(*values)

    draws = np.random.default_rng(seed).integers(0, len(universe), size=(resamples, len(universe)))
    agreements = [_resampled(agreeing, compared, draws) for agreeing, compared in arrays]
    statistic = 0.5 * (agreements[0] + agreements[1]) - agreements[2]
    defined = statistic[~np.isnan(statistic)]
    lower = upper = None
    if point is not None and defined.size:
        alpha = 1.0 - confidence
        low, high = np.quantile(defined, [alpha / 2, 1.0 - alpha / 2])
        lower, upper = float(low), float(high)
    return PartitionEffect(
        w1_c=values[0],
        w2_c=values[1],
        w1_w2=values[2],
        point=point,
        lower=lower,
        upper=upper,
        clusters=len(universe),
        resamples=resamples,
        undefined_resamples=int(resamples - defined.size),
        seed=seed,
        confidence=confidence,
    )


def decide_equivalence(effect: PartitionEffect, *, delta: float) -> EquivalenceDecision:
    """Apply the pre-registered decision rule to one metric's partition effect.

    The rules are tried in the order they were registered: an interval entirely below zero
    rejects invariance even when its lower bound is also within ``delta``.

    Args:
        effect: The effect and its interval.
        delta: The equivalence tolerance in absolute agreement (0.10 in the protocol), fixed
            before the run.

    Returns:
        The decision.

    Raises:
        ValueError: If ``delta`` is negative or not finite.
    """
    if not math.isfinite(delta) or delta < 0:
        raise ValueError(f"delta must be a finite, non-negative agreement, got {delta}")
    if effect.lower is None or effect.upper is None:
        return EquivalenceDecision.NOT_MEASURABLE
    if effect.upper < 0:
        return EquivalenceDecision.REJECTS_INVARIANCE
    if effect.lower >= -delta:
        return EquivalenceDecision.EQUIVALENT_WITHIN_DELTA
    return EquivalenceDecision.INCONCLUSIVE


def encode_incremental_context(value: object) -> Any:
    """Return the JSON-compatible form of a value of this module, for a report.

    Dataclasses become objects of their fields, enums their value, a geometry key its short
    rendering (as a value or as a mapping key), tuples and lists lists, and a frozenset a sorted
    list. Nothing is dropped or summarized.

    Args:
        value: A state, comparison, effect, verdict or decision of this module, or a mapping,
            sequence or scalar made of them.

    Returns:
        The JSON-compatible value.

    Raises:
        TypeError: If the value holds a type this module never produces.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, GeometryKey):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encode_incremental_context(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): encode_incremental_context(item) for key, item in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(encode_incremental_context(item) for item in value)
    if isinstance(value, tuple | list):
        return [encode_incremental_context(item) for item in value]
    raise TypeError(f"cannot encode a {type(value).__name__} object in a report")


def _key(references: Iterable[GeometryReference]) -> GeometryKey:
    """Return the geometry key of a set of references to the foundation map."""
    return GeometryKey(tuple(str(reference.geometry_id) for reference in references))


def _unique_keys(items: Iterable[tuple[str, GeometryKey]]) -> dict[str, GeometryKey]:
    """Map local ids to geometry keys, refusing a key two items share.

    Raises:
        ValueError: If two items cover the same geometry.
    """
    keys: dict[str, GeometryKey] = {}
    owner: dict[GeometryKey, str] = {}
    for local_id, key in items:
        if key in owner:
            raise ValueError(
                f"{owner[key]!r} and {local_id!r} cover the same geometry ({key}): the geometric "
                "key cannot pair them across builds"
            )
        owner[key] = local_id
        keys[local_id] = key
    return keys


def _region_key(keys: Mapping[str, str], region_id: str) -> str:
    """Resolve a producer-local region to comparable content, refusing absent evidence."""
    try:
        return keys[region_id]
    except KeyError as error:
        raise ValueError(f"regional evidence {region_id!r} is absent from selected runs") from error


def _region_content_keys(runs: Iterable[RegionPerceptionRunReader]) -> dict[str, str]:
    """Key published regions by geometry and mask bytes rather than producer identity.

    The same physical region has different IDs in independent runs. The mask index hash is
    already over the packed pixels, so comparing it needs no image allocation or payload read.
    """
    keys: dict[str, str] = {}
    for run in runs:
        regions = run.read_regions()
        mask_store = run.mask_store() if any(region.mask_reference for region in regions) else None
        for region in regions:
            if region.source_observation_id is None:
                raise ValueError("published regional evidence lacks source_observation_id")
            mask_hash = None
            if region.mask_reference is not None:
                assert mask_store is not None
                mask_hash = mask_store.entry(
                    region.source_observation_id, region.region_id
                ).content_hash
            box = region.bounding_box
            payload = (
                str(region.source_observation_id),
                box.x,
                box.y,
                box.width,
                box.height,
                region.image_width,
                region.image_height,
                region.area_pixels,
                region.region_kind,
                region.is_accepted,
                mask_hash,
            )
            digest = hashlib.sha256(
                json.dumps(payload, allow_nan=False).encode("utf-8")
            ).hexdigest()
            keys[str(region.region_id)] = f"region-content:{digest}"
    return keys


def _regions(
    runs: Iterable[RegionPerceptionRunReader], keys: Mapping[str, str]
) -> tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]:
    """Group comparable regional content by physical frame for layer-1 diagnostics."""
    by_frame: dict[str, list[tuple[Any, ...]]] = {}
    for run in runs:
        regional: dict[str, list[tuple[Any, ...]]] = {
            str(obs): [] for obs in run.manifest.source_observation_ids
        }
        for region in run.read_regions():
            if region.source_observation_id is None:
                raise ValueError("published regional evidence lacks source_observation_id")
            regional[str(region.source_observation_id)].append(
                (
                    _region_key(keys, str(region.region_id)),
                    region.bounding_box.x,
                    region.bounding_box.y,
                    region.bounding_box.width,
                    region.bounding_box.height,
                    region.region_kind,
                    region.is_accepted,
                )
            )
        for frame, records in regional.items():
            by_frame.setdefault(frame, []).append(tuple(records))
    # A ordem dos artifacts regionais selecionados não participa da comparação.
    return tuple(
        (frame, tuple(sorted(of_frame, key=repr))) for frame, of_frame in sorted(by_frame.items())
    )


def _validate(context_map: Path) -> MapValidation:
    """Validate a ContextMapArtifact at the full level and keep its findings by severity."""
    report = validate_context_map_artifact(context_map, level=ValidationLevel.FULL)
    by_severity: dict[Severity, list[str]] = {Severity.ERROR: [], Severity.WARNING: []}
    for finding in report.findings:
        by_severity[finding.severity].append(f"{finding.code}: {finding.message}")
    return MapValidation(
        status=report.status.value,
        errors=tuple(by_severity[Severity.ERROR]),
        warnings=tuple(by_severity[Severity.WARNING]),
    )


def _difference(name: str, left: object, right: object) -> DimensionDifference:
    """Compare one dimension as a multiset, a scalar being a one-item collection."""
    left_items: Counter[Any] = Counter(left if isinstance(left, tuple) else (left,))
    right_items: Counter[Any] = Counter(right if isinstance(right, tuple) else (right,))
    only_left = sorted((left_items - right_items).elements())
    only_right = sorted((right_items - left_items).elements())
    return DimensionDifference(
        name=name,
        equal=not only_left and not only_right,
        only_left=len(only_left),
        only_right=len(only_right),
        left_examples=tuple(_render(item) for item in only_left[:_MAX_EXAMPLES]),
        right_examples=tuple(_render(item) for item in only_right[:_MAX_EXAMPLES]),
    )


def _render(item: object) -> str:
    """Render an item of a dimension, a geometry key as its short digest."""
    if isinstance(item, tuple):
        return "(" + ", ".join(_render(part) for part in item) + ")"
    return str(item)


def _aligned(terms: ClusterTerms, position: Mapping[str, int]) -> tuple[Any, Any]:
    """Place a pair's terms on the shared cluster universe; a missing cluster is ``(0, 0)``."""
    import numpy as np

    agreeing = np.zeros(len(position), dtype=np.int64)
    compared = np.zeros(len(position), dtype=np.int64)
    for cluster, agree, compare in zip(terms.clusters, terms.agreeing, terms.compared, strict=True):
        agreeing[position[cluster]] = agree
        compared[position[cluster]] = compare
    return agreeing, compared


def _resampled(agreeing: Any, compared: Any, draws: Any) -> Any:
    """Return the agreement of each resample, ``NaN`` where it drew nothing to compare."""
    import numpy as np

    compared_sums = compared[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(compared_sums > 0, agreeing[draws].sum(axis=1) / compared_sums, np.nan)


def _effect(w1_c: float | None, w2_c: float | None, w1_w2: float | None) -> float | None:
    """Return ``(agr(W1,C) + agr(W2,C)) / 2 - agr(W1,W2)``, or ``None`` if one is undefined."""
    if w1_c is None or w2_c is None or w1_w2 is None:
        return None
    return 0.5 * (w1_c + w2_c) - w1_w2
