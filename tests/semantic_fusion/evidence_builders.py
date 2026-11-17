"""Scenario builders: perception evidence, scores and a support built from real observations."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

from fusion_builders import MAP_ID, make_interpreter, make_scorer, result_id
from geometry_fake import InMemoryGeometrySource
from observation_builders import make_perception_run, make_spatial_observation, timestamps

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    EvidenceReference,
    ExcludedObservation,
    FusionSupport,
    GeometryOverlapSupportPolicy,
    PhysicalObservationGrouping,
    VisualFeatureRef,
    build_fusion_supports,
    evidence_contribution_id_for,
    group_by_physical_observation,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticAttribute,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_scoring import ScoreId, SemanticScore, SemanticScoreType
from contextmap.sensor_association import (
    SpatialObservation,
    SpatialObservationId,
    spatial_observation_id_for,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import FeatureId, PerceptionResult, SemanticSupport, VisualFeature

AttributeValue = str | int | float | bool | None
"""A claim attribute value, as the interpreter emits it (a JSON scalar)."""

_REGION_BACKEND = BackendProvenance(
    backend_id="fake_region_discovery",
    capability="region_discovery",
    provider="fake",
    model="fake",
    version="0",
)


@dataclass(frozen=True)
class ClaimSpec:
    """One claim of one view, described by what fusion reads."""

    label: str
    role: HypothesisRole = HypothesisRole.PRIMARY
    confidence: float | None = None
    claim_id: str | None = None
    interaction_factor: float | None = None
    category: str | None = None
    attributes: tuple[tuple[str, AttributeValue], ...] = ()


@dataclass(frozen=True)
class View:
    """One region of one frame under one run, with the claims the interpreter made."""

    run: str
    frame: str
    claims: Sequence[ClaimSpec] = ()
    region: str = "region-0001"
    geometry: Sequence[int] = tuple(range(20))
    interpreter: BackendProvenance = field(default_factory=make_interpreter)
    features: Sequence[VisualFeatureRef] = ()


@dataclass(frozen=True)
class Scenario:
    """Everything the baseline accumulation reads, coherent by construction."""

    support: FusionSupport
    observations: dict[SpatialObservationId, SpatialObservation]
    grouping: PhysicalObservationGrouping
    results: dict[tuple[str, SourceObservationId], tuple[PerceptionResult, ...]]


def claim_id_of(view: View, index: int) -> ClaimId:
    spec = view.claims[index]
    return ClaimId(spec.claim_id or f"{view.run}--{view.frame}--claim-{index:04d}")


def make_claim(view: View, index: int) -> SemanticClaim:
    spec = view.claims[index]
    return SemanticClaim(
        claim_id=claim_id_of(view, index),
        source_observation_id=SourceObservationId(view.frame),
        semantic_interpretation_run_id=SemanticInterpretationRunId(result_id(view.run, view.frame)),
        hypothesis=spec.label,
        role=spec.role,
        provenance=SemanticInferenceProvenance(
            backend=view.interpreter,
            task_identity="task",
            prompt_template_id="prompt-v1",
            output_schema_version="schema-v1",
        ),
        confidence=spec.confidence,
        region_id=RegionId(view.region),
        category=spec.category,
        attributes=(
            *(
                ()
                if spec.interaction_factor is None
                else (SemanticAttribute(name="interaction_factor", value=spec.interaction_factor),)
            ),
            *(SemanticAttribute(name=name, value=value) for name, value in spec.attributes),
        ),
    )


def make_score(
    view: View, index: int, value: float, *, scorer: str = "clip_scorer"
) -> SemanticSupport:
    return SemanticSupport(
        claim_id=claim_id_of(view, index),
        support_score=value,
        provenance=make_scorer(scorer),
    )


def make_raw_score(
    view: View,
    index: int,
    value: float,
    *,
    scorer: str = "clip_scorer",
    feature: str = "feature-0001",
    embedding_space_id: str = "clip-vit-l14",
) -> SemanticScore:
    """A raw cosine score for one claim, the shape a CLIP region scorer emits in canonical/2."""
    return SemanticScore(
        score_id=ScoreId(f"score-{view.run}-{view.frame}-{index:04d}"),
        claim_id=claim_id_of(view, index),
        feature_id=FeatureId(feature),
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=value,
        calibrated_probability=None,
        embedding_space_id=embedding_space_id,
        source_observation_id=SourceObservationId(view.frame),
        region_id=RegionId(view.region),
        provenance=make_scorer(scorer),
    )


@dataclass(frozen=True)
class ScenarioParts:
    """Everything the accumulation reads, for any number of supports."""

    supports: tuple[FusionSupport, ...]
    excluded: tuple[ExcludedObservation, ...]
    observations: dict[SpatialObservationId, SpatialObservation]
    grouping: PhysicalObservationGrouping
    results: dict[tuple[str, SourceObservationId], tuple[PerceptionResult, ...]]


def build_scenario_parts(
    views: Sequence[View], *, geometry_points: int = 1_000, min_overlap: float = 0.5
) -> ScenarioParts:
    """Observations, results, grouping and every support that joins them."""
    observations: dict[SpatialObservationId, SpatialObservation] = {}
    claims_of_result: dict[tuple[str, SourceObservationId], list[SemanticClaim]] = {}
    regions_of_result: dict[tuple[str, SourceObservationId], set[str]] = {}
    features_of_result: dict[tuple[str, SourceObservationId], list[VisualFeatureRef]] = {}
    for view in views:
        claims = [make_claim(view, index) for index in range(len(view.claims))]
        observation = make_spatial_observation(
            view.run, view.frame, view.region, support=tuple(view.geometry)
        )
        observations[observation.spatial_observation_id] = observation
        key = (view.run, SourceObservationId(view.frame))
        claims_of_result.setdefault(key, []).extend(claims)
        regions_of_result.setdefault(key, set()).add(view.region)
        features_of_result.setdefault(key, []).extend(view.features)
    runs = sorted({view.run: view for view in views}.values(), key=lambda view: view.run)
    frames = sorted({view.frame for view in views})
    grouping = group_by_physical_observation(
        observations.values(),
        selected_region_runs=[make_perception_run(v.run, interpreter=v.interpreter) for v in runs],
        acquisition_timestamps=timestamps(frames),
    )
    grouping = dataclasses.replace(
        grouping,
        groups=tuple(
            dataclasses.replace(
                group,
                semantic_interpretation_artifact_ids=tuple(
                    result_id(run, str(group.physical_observation_id))
                    for run in group.region_artifact_ids
                ),
            )
            for group in grouping.groups
        ),
    )
    build = build_fusion_supports(
        observations.values(),
        geometry=InMemoryGeometrySource(
            MAP_ID, {i: (i * 0.1, 0.0, 0.0) for i in range(geometry_points)}
        ),
        acquisition_timestamps=timestamps(frames),
        policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=min_overlap),
    )
    results: dict[tuple[str, SourceObservationId], tuple[PerceptionResult, ...]] = {
        key: (_result(key, claims, regions_of_result[key], features_of_result[key]),)
        for key, claims in claims_of_result.items()
    }

    return ScenarioParts(
        supports=build.supports,
        excluded=build.excluded,
        observations=observations,
        grouping=grouping,
        results=results,
    )


def build_scenario(
    views: Sequence[View], *, geometry_points: int = 1_000, min_overlap: float = 0.5
) -> Scenario:
    """Observations, results, grouping and the single support that joins them."""
    parts = build_scenario_parts(views, geometry_points=geometry_points, min_overlap=min_overlap)
    assert len(parts.supports) == 1, "the scenario must be a single support"
    return Scenario(
        support=parts.supports[0],
        observations=parts.observations,
        grouping=parts.grouping,
        results=parts.results,
    )


def _result(
    key: tuple[str, SourceObservationId],
    claims: Sequence[SemanticClaim],
    regions: set[str],
    features: Sequence[VisualFeatureRef],
) -> PerceptionResult:
    run, frame = key
    feature_metadata = tuple(
        VisualFeature(
            feature_id=ref.feature_id,
            source_observation_id=frame,
            scope=ref.scope,
            embedding_space_id=ref.embedding_space_id,
            shape=(2,),
            dtype="float32",
            payload_reference=f"features/{ref.feature_id}.npy",
            provenance=_REGION_BACKEND,
            region_id=ref.region_id,
        )
        for ref in features
    )
    return PerceptionResult(
        source_observation_id=frame,
        sequence_artifact_id="sequence-0001",
        region_artifact_id=run,
        semantic_interpretation_artifact_id=result_id(run, frame),
        visual_evidence_artifact_id=features[0].visual_artifact_id if features else None,
        regions=tuple(
            Region2D(
                region_id=RegionId(region),
                bounding_box=BoundingBox2D(x=0.0, y=0.0, width=10.0, height=10.0),
                provenance=_REGION_BACKEND,
            )
            for region in sorted(regions)
        ),
        claims=tuple(claims),
        features=feature_metadata,
    )


def reference_to(scenario: Scenario, view: View, index: int | None = 0) -> EvidenceReference:
    """The reference that names one claim of one view, or the view itself when ``index`` is None."""
    return EvidenceReference(
        contribution_id=EvidenceContributionId(
            str(
                evidence_contribution_id_for(
                    fusion_support_id=scenario.support.fusion_support_id,
                    spatial_observation_id=spatial_observation_id_for(
                        association_run_id=view.run,
                        source_observation_id=SourceObservationId(view.frame),
                        region_id=RegionId(view.region),
                    ),
                )
            )
            + f"--semantic-{result_id(view.run, view.frame)}"
        ),
        claim_id=None if index is None else claim_id_of(view, index),
    )
