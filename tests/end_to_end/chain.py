"""A contract-level Solution 1 chain over the synthetic CI sequence, through the real artifacts.

Every stage runs the real capability code and persists its real artifact, then is read back through
the capability's public reader. Only the model outputs are fake: the perception evidence (masks and
semantic claims) is hand-written, because a model is exactly what CI must not depend on. Nothing
here is evidence about real data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from contextmap.artifact import (
    CONTEXT_MAP_ASSEMBLY_POLICY_ID,
    ArtifactKind,
    ContextMapArtifactReader,
    ContextMapId,
    ContextMapMetadata,
    DeclaredCapabilities,
    MapCapability,
    MapCreation,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
    UpstreamArtifact,
    artifact_digest,
    assemble_context_map_with_metrics,
    estimator_local_map_frame,
    write_context_map_with_metrics,
)
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
    ResolvedEntityReference,
    lineage_from_mapping_manifest,
    materialize_resolved_entities,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.evaluation.ci_fixtures import (
    CI_FIXTURE_ID,
    LANDMARKS,
    build_synthetic_sequence,
    project_landmark,
)
from contextmap.evaluation.cross_stage import CrossStageInputs
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryBlockSource,
    MapDebugLevel,
    MotionCorrectionPolicy,
    ScanDisposition,
    build_geometry_mapping_plan_from_artifacts,
)
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    FullSequenceSelection,
    ImageObservation,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceObservationId,
    decode_selection,
    resolve_selection_offsets,
    selection_identity,
)
from contextmap.region_perception import (
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionDiscoveryAudit,
    RegionDiscoveryIdentity,
    RegionId,
    RegionPerceptionRunManifest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    namespace_regions,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import inventory_digest
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    ExcludedObservation,
    FusionOutcome,
    FusionRunLineage,
    GeometryOverlapSupportPolicy,
    SemanticFusionRunId,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
    build_fusion_supports,
    group_by_physical_observation,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticInterpretationRunManifest,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
    SemanticObservation,
)
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    EntityReference,
    GeometrySummaryPolicy,
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunReader,
    SemanticMappingRunWriter,
    lineage_from_fusion_manifest,
    materialize_entities,
)
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationOutcome,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.sensor_association.service import AssociationFrameInput
from contextmap.shared import BackendProvenance, SourceTimestamp
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    FrameConventions,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
    SpatialRelationsRunWriter,
    decide_relations,
    generate_relation_candidates,
    lineage_from_resolution_manifest,
    resolved_entity_geometries,
)
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    LookupPolicy,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StaticRelationRequirement,
    TrajectoryId,
    TrajectoryLookup,
    calibration_identity,
    execute_state_estimation,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.visual_perception import (
    ArtifactReference,
    PerceptionResult,
    PreparedImage,
    SourceImage,
    StageOutcome,
    StageStatus,
    VisualEvidenceResult,
    VisualEvidenceRunId,
    VisualEvidenceRunWriter,
    prepare_image,
)
from contextmap.visual_perception.visual_features import VisualFeature
from contextmap.visual_perception.visual_features.adapters.dinov3 import (
    DinoV3Config,
    DinoV3DenseFeatureBackend,
    DinoV3NativeOutput,
)
from semantic_artifact_builders import write_semantic_views

SEQUENCE_NAME = CI_FIXTURE_ID
SEQUENCE_ARTIFACT_ID = SequenceArtifactId("ci-fixture")
REFERENCE_FRAME = FrameId("odom")
BODY_FRAME = FrameId("base_link")
_CAMERA_CALIBRATION = CalibrationReferenceId("front_camera-calib")
PALLET_REGION = RegionId("region-pallet")
SHELF_REGION = RegionId("region-shelf")
RUN_A = "run-a"
RUN_B = "run-b"

# Duas inferências sobre a mesma observação física (frame-0000): a run-b repete o frame com outro
# modelo e discorda sobre a palete. As demais observações vêm só da run-a.
_CLAIMS: dict[tuple[str, int, RegionId], tuple[tuple[str, HypothesisRole, float], ...]] = {
    ("run-a", 0, PALLET_REGION): (
        ("pallet", HypothesisRole.PRIMARY, 0.8),
        ("crate", HypothesisRole.ALTERNATIVE, 0.15),
    ),
    ("run-a", 0, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.7),),
    ("run-a", 1, PALLET_REGION): (("pallet", HypothesisRole.PRIMARY, 0.7),),
    ("run-a", 1, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.6),),
    ("run-a", 2, PALLET_REGION): (("pallet", HypothesisRole.PRIMARY, 0.9),),
    ("run-a", 2, SHELF_REGION): (("unknown", HypothesisRole.PRIMARY, 0.4),),
    ("run-b", 0, PALLET_REGION): (("crate", HypothesisRole.PRIMARY, 0.6),),
    ("run-b", 0, SHELF_REGION): (("shelf post", HypothesisRole.PRIMARY, 0.9),),
}
_LANDMARK_BY_REGION = {PALLET_REGION: "pallet-corner", SHELF_REGION: "shelf-post"}
_INTERPRETER = BackendProvenance(
    backend_id="canned_interpreter",
    capability="semantic_interpreter",
    provider="fixture",
    model="canned",
    version="1",
)
_REGION_BACKEND = BackendProvenance(
    backend_id="canned_region_discovery",
    capability="region_discovery",
    provider="fixture",
    model="canned",
    version="1",
)


@dataclass(frozen=True)
class SyntheticChain:
    """Every stage artifact of the synthetic chain, opened through its public reader."""

    sequence: SequenceArtifactReader
    trajectory: StateEstimationRunReader
    geometry: GeometricMapArtifactReader
    region_runs: tuple[RegionPerceptionRunManifest, ...]
    semantic_runs: tuple[SemanticInterpretationRunManifest, ...]
    semantic_observations: tuple[SemanticObservation, ...]
    perception_results: dict[str, PerceptionResult]
    associations: tuple[tuple[SensorAssociationOutcome, SensorAssociationRunReader], ...]
    fusion_outcomes: tuple[FusionOutcome, ...]
    fusion_excluded: tuple[ExcludedObservation, ...]
    fusion: SemanticFusionRunReader
    spatial_observations: dict[SpatialObservationId, SpatialObservation]
    mapping: SemanticMappingRunReader
    resolution: EntityResolutionRunReader
    relations: SpatialRelationsRunReader
    context_map_dir: Path


def _ingest(workspace: Path) -> SequenceArtifactReader:
    sequence = build_synthetic_sequence()
    directory = workspace / "ingestion"
    with SequenceArtifactWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        artifact_id=SEQUENCE_ARTIFACT_ID,
    ) as writer:
        writer.set_calibration(sequence.calibration)
        for observation in sequence.observations:
            writer.add_observation(observation)
        writer.finalize()
    return SequenceArtifactReader(directory)


def _estimate(workspace: Path, sequence: SequenceArtifactReader) -> StateEstimationRunReader:
    request = StateEstimationRequest(
        trajectory_id=TrajectoryId("ci-fixture--trajectory"),
        sequence_artifact_id=sequence.manifest.artifact_id,
        selection_id="full-sequence",
        observations=tuple(sequence.list_observations()),
        calibration=sequence.read_calibration(),
    )
    outcome = execute_state_estimation(
        ExternalPoseEstimator(
            ExternalPoseConfig(reference_frame=REFERENCE_FRAME, body_frame=BODY_FRAME)
        ),
        request,
        downstream=(
            GeometryRequirements(
                capability="geometric_mapping",
                modalities=frozenset({"lidar"}),
                static_relations=(
                    StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),
                ),
            ),
        ),
    )
    directory = workspace / "state_estimation"
    StateEstimationRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=StateEstimationRunId("state-run-0001"),
        run_index=1,
    ).finalize(outcome)
    return StateEstimationRunReader(directory)


def _map(
    workspace: Path, sequence: SequenceArtifactReader, trajectory: StateEstimationRunReader
) -> GeometricMapArtifactReader:
    plan = build_geometry_mapping_plan_from_artifacts(
        sequence=sequence,
        selection=FullSequenceSelection(),
        run=trajectory,
        pose_lookup=LookupPolicy.exact(),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
        ),
    )
    directory = workspace / "geometric_mapping"
    GeometricMapArtifactWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=GeometricMapRunId("map-run-0001"),
        run_index=1,
        debug_level=MapDebugLevel.NONE,
    ).finalize(plan=plan, aggregation=None, code_version="test")
    return GeometricMapArtifactReader(directory)


def _landmark_world(name: str) -> tuple[float, float, float]:
    return next(point for label, point in LANDMARKS if label == name)


def _mask_around_landmark(region: RegionId, frame_index: int) -> InlineMask:
    """A one-pixel mask on the analytic projection of the region's landmark.

    A single pixel keeps the two regions disjoint at 16 x 12: each contains exactly the three
    copies (one per scan) of its own landmark.
    """
    projection = project_landmark(_landmark_world(_LANDMARK_BY_REGION[region]), frame_index)
    assert projection["pixel"] is not None, "a region is only drawn for a visible landmark"
    # A associação usa o centro do pixel em coordenadas inteiras, então o pixel é o arredondado.
    u, v = (round(value) for value in projection["pixel"])
    return InlineMask(
        np.array(
            tuple(x == u and y == v for y in range(12) for x in range(16)), dtype=bool
        ).reshape(12, 16)
    )


def _perception_result(
    run: str,
    frame_index: int,
    *,
    claims_of: str | None = None,
    sequence_artifact_id: str | None = None,
) -> PerceptionResult:
    observation_id = SourceObservationId(f"frame-{frame_index:04d}")
    result_id = str(f"{run}--{observation_id}")
    regions = []
    claims = []
    for region_id in (PALLET_REGION, SHELF_REGION):
        regions.append(
            Region2D(
                region_id=region_id,
                bounding_box=BoundingBox2D(x=0.0, y=0.0, width=16.0, height=12.0),
                provenance=_REGION_BACKEND,
                mask=_mask_around_landmark(region_id, frame_index),
                image_width=16,
                image_height=12,
                is_accepted=True,
                rejection_reason=None,
                source_observation_id=observation_id,
            )
        )
        for claim_index, (label, role, confidence) in enumerate(
            _CLAIMS.get((claims_of or str(run), frame_index, region_id), ())
        ):
            claims.append(
                SemanticClaim(
                    claim_id=ClaimId(f"{result_id}--{region_id}--claim-{claim_index:04d}"),
                    source_observation_id=observation_id,
                    semantic_interpretation_run_id=SemanticInterpretationRunId(str(run)),
                    hypothesis=label,
                    role=role,
                    provenance=SemanticInferenceProvenance(
                        backend=_INTERPRETER,
                        task_identity="canned-task",
                        prompt_template_id="canned-prompt-v1",
                        output_schema_version="schema-v1",
                    ),
                    confidence=confidence,
                    region_id=region_id,
                )
            )
    return PerceptionResult(
        source_observation_id=observation_id,
        region_artifact_id=run,
        semantic_interpretation_artifact_id=run,
        sequence_artifact_id=sequence_artifact_id or str(SEQUENCE_ARTIFACT_ID),
        regions=tuple(regions),
        features=(),
        claims=tuple(claims),
    )


def _frame_input(image: ImageObservation, result: PerceptionResult) -> AssociationFrameInput:
    prepared = prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(
                uri=f"sequence://{image.observation_id}", sha256="0" * 64, media_type="image/rgb8"
            ),
            width=image.width,
            height=image.height,
        ),
        operations=(),
    )
    return AssociationFrameInput(
        observation=image,
        prepared_image=prepared,
        region_artifact_id=result.region_artifact_id,
        regions=tuple(result.regions),
        dense_maps={},
    )


def associate(
    workspace: Path,
    sequence: SequenceArtifactReader,
    trajectory: StateEstimationRunReader,
    geometry: GeometryBlockSource,
    results: dict[str, PerceptionResult],
    run: str,
    run_index: int,
) -> tuple[SensorAssociationOutcome, SensorAssociationRunReader]:
    """Associate the regions of one perception run; a frame appears once per association run.

    ``geometry`` is any block source of the map, so the same association also runs over a
    derived representation of it.
    """
    # O subconjunto de CI grava as imagens sem `calibration_id`, e a associação recusa uma imagem
    # que não nomeia a sua calibração: o teste a nomeia aqui, sem alterar o fixture versionado.
    images = {
        str(item.observation_id): replace(item, calibration_id=_CAMERA_CALIBRATION)
        for item in sequence.list_observations()
        if isinstance(item, ImageObservation)
    }
    frames = tuple(
        _frame_input(images[str(result.source_observation_id)], result)
        for result in results.values()
        if result.region_artifact_id == run
    )
    calibration = sequence.read_calibration()
    assert calibration is not None, "the synthetic sequence carries its calibration"
    request = SensorAssociationRequest(
        association_run_id=f"association-{run}",
        sequence_artifact_id=sequence.manifest.artifact_id,
        selection_id="full-sequence",
        geometry=geometry,
        trajectory=TrajectoryLookup(trajectory.trajectory()),
        pose_policy=LookupPolicy.exact(),
        calibration=calibration,
        candidate_policy=CandidateGeometryPolicy(max_range_m=None),
        occlusion_policy=OcclusionPolicy(
            cell_size_px=4, neighborhood_radius_cells=0, depth_margin_m=0.1, depth_margin_ratio=0.02
        ),
        tolerances=DiagnosticTolerances(
            max_pose_time_delta_ns=1_000_000,
            max_map_window_offset_ns=None,
            max_reprojection_p95_px=None,
            max_reprojection_invalid_rate=None,
        ),
        frames=frames,
        state_estimation_run_id=trajectory.manifest.run_id,
        code_version="test",
    )
    # A cadeia sintética tem uma associação por run de percepção, então cada uma ganha o seu
    # diretório; no runtime há um único `<run>/sensor_association/` por execução.
    directory = workspace / f"sensor_association-{run}"
    writer = SensorAssociationRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId(f"association-{run}"),
        run_index=run_index,
    )
    with writer.transaction() as sink:
        outcome = SensorAssociationService().run(request, sink=sink)
        sink.finalize(outcome)
    return outcome, SensorAssociationRunReader(directory)


def _fuse(
    workspace: Path,
    sequence: SequenceArtifactReader,
    geometry: GeometricMapArtifactReader,
    associations: tuple[tuple[SensorAssociationOutcome, SensorAssociationRunReader], ...],
    results: dict[str, PerceptionResult],
    runs: tuple[RegionPerceptionRunManifest, ...],
) -> tuple[tuple[FusionOutcome, ...], tuple[ExcludedObservation, ...], SemanticFusionRunReader]:
    # As observações vêm do artifact, como qualquer consumidor a jusante as lê: o run não
    # retém mais os frames em memória (#563).
    observations = [item for _, reader in associations for item in reader.observations()]
    timestamps = {
        item.observation_id: item.timestamp
        for item in sequence.list_observations()
        if isinstance(item, ImageObservation)
    }
    grouping = group_by_physical_observation(
        observations, selected_region_runs=runs, acquisition_timestamps=timestamps
    )
    build = build_fusion_supports(
        observations,
        geometry=geometry.geometry(),
        acquisition_timestamps=timestamps,
        policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=0.5),
        code_version="test",
    )
    by_id = {item.spatial_observation_id: item for item in observations}
    policy = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))
    outcomes = tuple(
        FusionOutcome(
            support=support,
            evidence=accumulate_baseline_evidence(
                support,
                observations=by_id,
                grouping=grouping,
                perception_results={
                    (item.region_artifact_id, item.source_observation_id): (item,)
                    for item in results.values()
                },
                policy=policy,
                code_version="test",
            ),
        )
        for support in build.supports
    )
    directory = workspace / "semantic_fusion"
    SemanticFusionRunWriter(
        output_dir=directory,
        sequence_name=SEQUENCE_NAME,
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=1,
        lineage=FusionRunLineage(
            sequence_artifact_id=str(sequence.manifest.artifact_id),
            geometric_map_id=geometry.manifest.map_id,
            association_run_ids=tuple(str(reader.manifest.run_id) for _, reader in associations),
            region_artifact_ids=tuple(run.run_id for run in runs),
            semantic_interpretation_artifact_ids=tuple(run.run_id for run in runs),
            point_representation_run_ids=(),
        ),
        code_version="test",
    ).write(outcomes, excluded=build.excluded)
    return outcomes, build.excluded, SemanticFusionRunReader(directory)


_MAPPING_GEOMETRY_SUMMARY = GeometrySummaryPolicy(
    sparse_point_threshold=3, connectivity_radius_m=0.5
)
_UP_AXIS = AxisDirection.POSITIVE_Z
_SEMANTIC_MAP_ID = SemanticMapId("semantic-map-ci")


def _materialize(
    workspace: Path, fusion: SemanticFusionRunReader, geometry: GeometricMapArtifactReader
) -> SemanticMappingRunReader:
    materialization = materialize_entities(
        fusion.iter_outcomes(),
        fusion_manifest=fusion.manifest,
        geometry=geometry.geometry(),
        semantic_map_id=_SEMANTIC_MAP_ID,
        policy=EntityMaterializationPolicy(geometry=_MAPPING_GEOMETRY_SUMMARY),
        code_version="test",
    )
    directory = workspace / "semantic_mapping"
    SemanticMappingRunWriter(
        output_dir=directory,
        sequence_name=fusion.manifest.sequence_name,
        run_id=SemanticMappingRunId("mapping-run-0001"),
        run_index=1,
        semantic_map_id=_SEMANTIC_MAP_ID,
        lineage=lineage_from_fusion_manifest(fusion.manifest),
        code_version="test",
        code_digest="sha256:" + "cd" * 32,
    ).write(materialization.entities, rejections=materialization.rejections)
    return SemanticMappingRunReader(directory)


def _resolve(workspace: Path, mapping: SemanticMappingRunReader) -> EntityResolutionRunReader:
    entities = list(mapping.iter_entities())
    candidate_sets = retrieve_candidate_sets(
        entities, CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    builder = MatchEvidenceBuilder(
        ComparisonChannels(
            geometry=GeometryComparisonPolicy(
                min_shared_support_jaccard=0.5,
                min_bounds_iou=0.5,
                min_bounds_containment=0.9,
                min_conflict_gap_m=0.5,
                min_extent_ratio=0.3,
            )
        )
    )
    resolutions = resolve_candidate_pairs(
        entities,
        candidate_sets,
        builder,
        ConservativeResolutionPolicy(
            use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
        ),
        code_version="test",
    )
    run_id = EntityResolutionRunId("resolution-run-0001")
    materialization = materialize_resolved_entities(
        entities,
        [item.decision for item in resolutions],
        resolution_run_id=run_id,
        code_version="test",
    )
    directory = workspace / "entity_resolution"
    EntityResolutionRunWriter(
        output_dir=directory,
        run_id=run_id,
        lineage=lineage_from_mapping_manifest(mapping.manifest),
        code_version="test",
    ).write(candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization)
    return EntityResolutionRunReader(directory)


def _relate(
    workspace: Path, resolution: EntityResolutionRunReader, geometry: GeometricMapArtifactReader
) -> SpatialRelationsRunReader:
    conventions = FrameConventions(
        map_frame=str(REFERENCE_FRAME), up_axis=_UP_AXIS, forward_axis=AxisDirection.POSITIVE_X
    )
    candidate_policy = CandidatePolicy(
        predicates=(RelationPredicate.NEXT_TO,), proximity_radius_m=2.0, directional_radius_m=2.0
    )
    source = geometry.geometry()
    entities = resolved_entity_geometries(
        resolution.resolved_entities(), source=source, policy=_MAPPING_GEOMETRY_SUMMARY
    )
    # O raio generoso garante ao menos um candidato geometricamente real entre a palete e o poste
    # (~1.36 m de separação): nenhum avaliador geométrico/de contato foi selecionado, então o
    # candidato fica honestamente UNRESOLVED, nunca decidido sem evidência.
    candidates = generate_relation_candidates(
        entities, policy=candidate_policy, conventions=conventions
    )
    decisions = decide_relations(candidates, [])
    directory = workspace / "spatial_relations"
    SpatialRelationsRunWriter(
        output_dir=directory,
        run_id=SpatialRelationsRunId("relations-run-0001"),
        lineage=lineage_from_resolution_manifest(resolution.manifest),
        policies=RelationsRunPolicies(
            frame_conventions=conventions,
            candidate=candidate_policy,
            geometry_summary=_MAPPING_GEOMETRY_SUMMARY,
        ),
        code_version="test",
    ).write(candidates=candidates, evidence=[], decisions=decisions)
    return SpatialRelationsRunReader(directory)


def _observation_window(clock_id: str, *, start_ns: int, end_ns: int) -> ObservationWindow:
    def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
        seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
        return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)

    return ObservationWindow(start=_timestamp(start_ns), end=_timestamp(end_ns))


def _assemble(
    workspace: Path,
    geometry: GeometricMapArtifactReader,
    resolution: EntityResolutionRunReader,
    relations: SpatialRelationsRunReader,
) -> Path:
    geometry_manifest = geometry.manifest
    bounds = geometry.geometry().geometric_map.bounds
    predicates = sorted(
        {relation.predicate for relation in relations.iter_relations()},
        key=lambda predicate: predicate.value,
    )
    metadata = ContextMapMetadata(
        creation=MapCreation(
            assembly_policy=PolicyRef(policy_id=CONTEXT_MAP_ASSEMBLY_POLICY_ID, version="2"),
            code_version="test",
            configuration_fingerprint=None,
        ),
        source_sequences=(
            SourceSequence(
                sequence_artifact_id=str(geometry_manifest.sequence_artifact_id),
                selection_id=geometry_manifest.selection_id,
            ),
        ),
        frame=estimator_local_map_frame(
            frame_id=str(geometry_manifest.map_frame), up_direction=_UP_AXIS.vector
        ),
        bounds=bounds,
        time_bounds=_observation_window(
            geometry_manifest.clock_id,
            start_ns=geometry_manifest.start_time_ns,
            end_ns=geometry_manifest.end_time_ns,
        ),
        capabilities=DeclaredCapabilities(
            content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
            relation_predicates=tuple(predicates),
        ),
    )
    sequence_dir = workspace / "ingestion"
    geometry_dir = workspace / "geometric_mapping"
    resolution_dir = workspace / "entity_resolution"
    relations_dir = workspace / "spatial_relations"
    sequence_lineage = (
        UpstreamArtifact(
            artifact_id=str(geometry_manifest.sequence_artifact_id),
            kind=ArtifactKind.SEQUENCE,
            content_identity=artifact_digest(sequence_dir),
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        ),
    )
    result = assemble_context_map_with_metrics(
        context_map_id=ContextMapId("context-map-ci-0001"),
        metadata=metadata,
        geometric_map_location=geometry_dir,
        entity_resolution_location=resolution_dir,
        spatial_relations_location=relations_dir,
        additional_lineage=sequence_lineage,
    )
    location_by_kind = {
        ArtifactKind.GEOMETRIC_MAP: geometry_dir,
        ArtifactKind.ENTITY_RESOLUTION_RUN: resolution_dir,
        ArtifactKind.SPATIAL_RELATIONS_RUN: relations_dir,
    }
    upstream_locations = {
        item.artifact_id: location_by_kind[item.kind]
        for item in result.context_map.lineage
        if item.kind.is_structural
    }
    directory = workspace / "context_map"
    write_context_map_with_metrics(
        result.context_map, output_dir=directory, upstream_locations=upstream_locations
    )
    return directory


@contextmanager
def synthetic_chain(workspace: Path) -> Iterator[SyntheticChain]:
    """Run every stage over the synthetic sequence and keep the geometry reader open."""
    sequence = _ingest(workspace)
    trajectory = _estimate(workspace, sequence)
    with _map(workspace, sequence, trajectory) as geometry:
        results = {
            str(f"{run}--frame-{index:04d}"): _perception_result(run, index)
            for run, frames in ((RUN_A, (0, 1, 2)), (RUN_B, (0,)))
            for index in frames
        }
        for key, view in tuple(results.items()):
            namespaced = namespace_regions(
                run_id=view.region_artifact_id,
                source_observation_id=view.source_observation_id,
                regions=view.regions,
            )
            results[key] = replace(
                view,
                regions=namespaced.regions,
                claims=tuple(
                    replace(
                        item, region_id=namespaced.region_id_map[cast(RegionId, item.region_id)]
                    )
                    for item in view.claims
                ),
            )
        runs_list = []
        semantic_runs_list = []
        semantic_records: list[SemanticObservation] = []
        for position, run in enumerate((RUN_A, RUN_B)):
            views = [item for item in results.values() if item.region_artifact_id == run]
            with RegionPerceptionRunWriter(workspace / f"regions-{run}") as regional_writer:
                regional_writer.write_manifest(
                    _canned_region_manifest(
                        run,
                        str(sequence.manifest.artifact_id),
                        [str(item.source_observation_id) for item in views],
                        "full-sequence",
                        position,
                    )
                )
                for view in views:
                    regional_writer.write_frame(_canned_prepared(view.source_observation_id))
                    for region in view.regions:
                        regional_writer.write_region(
                            regional_writer.externalize_region_mask(
                                view.source_observation_id, region
                            )
                        )
                runs_list.append(regional_writer.finalize())
            with SemanticInterpretationRunWriter(
                output_dir=workspace / f"semantics-{run}",
                run_id=SemanticInterpretationRunId(run),
                sequence_artifact_id=str(sequence.manifest.artifact_id),
                region_artifact_id=run,
                selection_id="full-sequence",
                configuration_digest="fixture",
                sequence_name=SEQUENCE_NAME,
                run_index=position,
            ) as semantic_writer:
                write_semantic_views(semantic_writer, views, run)
                semantic_runs_list.append(semantic_writer.finalize())
            semantic_reader = SemanticInterpretationRunReader(workspace / f"semantics-{run}")
            semantic_records.extend(semantic_reader.iter_observations())
            for record in semantic_reader.iter_observations():
                key = f"{run}--{record.source_observation_id}"
                results[key] = replace(
                    results[key], claims=record.claims, scene_context=record.scene_context
                )
        runs = tuple(runs_list)
        associations = tuple(
            associate(workspace, sequence, trajectory, geometry.geometry(), results, run, run_index)
            for run_index, run in enumerate((RUN_A, RUN_B), start=1)
        )
        fusion_outcomes, excluded, fusion = _fuse(
            workspace, sequence, geometry, associations, results, runs
        )
        mapping = _materialize(workspace, fusion, geometry)
        resolution = _resolve(workspace, mapping)
        relations = _relate(workspace, resolution, geometry)
        context_map_dir = _assemble(workspace, geometry, resolution, relations)
        yield SyntheticChain(
            sequence=sequence,
            trajectory=trajectory,
            geometry=geometry,
            region_runs=runs,
            semantic_runs=tuple(semantic_runs_list),
            semantic_observations=tuple(semantic_records),
            perception_results=results,
            associations=associations,
            fusion_outcomes=fusion_outcomes,
            fusion_excluded=excluded,
            fusion=fusion,
            spatial_observations={
                item.spatial_observation_id: item
                for _, reader in associations
                for item in reader.observations()
            },
            mapping=mapping,
            resolution=resolution,
            relations=relations,
            context_map_dir=context_map_dir,
        )


def cross_stage_inputs(chain: SyntheticChain) -> CrossStageInputs:
    """The manifests and objects the cross-stage evaluator reads, taken from the readers."""
    context_map = ContextMapArtifactReader.open(chain.context_map_dir).context_map()
    return CrossStageInputs(
        sequence=chain.sequence.manifest,
        sequence_calibration_identity=calibration_identity(chain.sequence.read_calibration()),
        trajectory=chain.trajectory.manifest,
        geometry=chain.geometry.manifest,
        geometry_source=chain.geometry.geometry(),
        region_runs=chain.region_runs,
        regions={
            (item.region_artifact_id, item.source_observation_id): tuple(item.regions)
            for item in chain.perception_results.values()
        },
        semantic_runs=chain.semantic_runs,
        semantic_observations=chain.semantic_observations,
        associations=tuple(reader.manifest for _, reader in chain.associations),
        spatial_observations=chain.spatial_observations,
        fusion=chain.fusion.manifest,
        fusion_outcomes=chain.fusion_outcomes,
        mapping=chain.mapping.manifest,
        entities={
            EntityReference(
                semantic_map_id=entity.semantic_map_id, entity_id=entity.entity_id
            ): entity
            for entity in chain.mapping.iter_entities()
        },
        resolution=chain.resolution.manifest,
        resolved_entities={
            ResolvedEntityReference(
                resolution_run_id=resolved.resolution_run_id,
                resolved_entity_id=resolved.resolved_entity_id,
            ): resolved
            for resolved in chain.resolution.resolved_entities().entities
        },
        relations=chain.relations.manifest,
        relation_records=tuple(chain.relations.iter_relations()),
        context_map=context_map,
    )


def canned_perception_results(run: str, sequence_artifact_id: str) -> list[PerceptionResult]:
    """The canned perception evidence of the first run (three frames) under another run identity."""
    return [
        _perception_result(run, index, claims_of="run-a", sequence_artifact_id=sequence_artifact_id)
        for index in range(3)
    ]


class SyntheticSequenceIngestion:
    """Test double: publishes the synthetic sequence into the stage directory it is given."""

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        sequence = build_synthetic_sequence()
        artifact_id = SequenceArtifactId(request.identity())
        with SequenceArtifactWriter(
            output_dir=request.output_dir, sequence_name=CI_FIXTURE_ID, artifact_id=artifact_id
        ) as writer:
            writer.set_calibration(sequence.calibration)
            for observation in sequence.observations:
                if isinstance(observation, ImageObservation):
                    # O subconjunto de CI grava imagens sem `calibration_id`; a associação o exige.
                    observation = replace(observation, calibration_id=_CAMERA_CALIBRATION)
                writer.add_observation(observation)
            writer.finalize()
        manifest = SequenceArtifactReader(request.output_dir).manifest
        return ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(artifact_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _FixedDinoV3Runtime:
    """Serves one fixed DINOv3 patch grid for every image: the model is what CI must not run."""

    def infer(self, image: PreparedImage) -> DinoV3NativeOutput:
        # Uma grade 3 x 4 de patches 4 x 4 cobre a imagem 16 x 12 da sequência sintética.
        rows, columns = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
        return DinoV3NativeOutput(
            array=(np.stack((rows, columns), axis=-1) + 1).astype(np.float32),
            model_input_width=16,
            model_input_height=12,
            patch_width=4,
            patch_height=4,
            register_token_count=4,
        )


def _canned_dense_feature(
    writer: VisualEvidenceRunWriter,
    run_id: VisualEvidenceRunId,
    observation_id: SourceObservationId,
) -> VisualFeature:
    """Extract one DINOv3 dense feature of a frame through the real adapter, into ``writer``.

    Only the model output is fixed: the adapter computes the sampling geometry and sends the
    payload and the contractual diagnostic to the run's writer, as the executor's sink does.
    """
    backend = DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vitb16-pretrain-lvd1689m",
            revision="d" * 40,
            input_width=16,
            input_height=12,
        ),
        run_id=run_id,
        feature_stage_id="dense_feature_extraction",
        source_artifact_id=str(run_id),
        payload_sink=writer,
        runtime=_FixedDinoV3Runtime(),
    )
    image = PreparedImage(
        source_observation_id=observation_id,
        payload_reference=f"prepared/{observation_id}.png",
        width=16,
        height=12,
        transformations=(),
    )
    return backend.extract_dense(image).dense_map.feature


def _canned_prepared(observation_id: SourceObservationId) -> PreparedImage:
    """Preparation metadata shared by all three canned producers."""
    return PreparedImage(
        source_observation_id=observation_id,
        payload_reference=f"source-observation:{observation_id}",
        width=16,
        height=12,
    )


def _canned_region_manifest(
    run: str, sequence: str, observations: Sequence[str], selection: str, index: int
) -> RegionPerceptionRunManifest:
    """Declare regional identity and exact physical observation coverage."""
    return RegionPerceptionRunManifest(
        run_id=run,
        run_index=index,
        sequence_artifact_id=sequence,
        source_observation_ids=tuple(observations),
        sequence_name=SEQUENCE_NAME,
        selection_id=selection,
        created_at="2026-01-01T00:00:00Z",
        prepared_image_id="canned-prepared-image",
        discovery_backend_id=_REGION_BACKEND.backend_id,
        discovery_model=_REGION_BACKEND.model,
        discovery_version=_REGION_BACKEND.version,
        discovery_configuration_fingerprint="fixture",
        normalization_policy_fingerprint="fixture",
    )


class CannedPerception:
    """Publish only visual features over the explicitly selected regional producer."""

    def __init__(self, *, stamp_execution_time: bool = False, dense_features: bool = False) -> None:
        """Select measured execution timing and optional deterministic dense model output."""
        self._stamp_execution_time = stamp_execution_time
        self._dense_features = dense_features

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Publish visual evidence without copying any regions, masks or semantic claims."""
        import time

        began = time.perf_counter()
        assert request.output_dir is not None and request.workspace is not None
        sequence = request.inputs["sequence"][0]
        regional = RegionPerceptionRunReader(request.directory_of(request.inputs["regions"][0]))
        sequence_reader = SequenceArtifactReader(request.directory_of(sequence))
        selection = (
            FullSequenceSelection()
            if request.observation_selection is None
            else decode_selection(dict(request.observation_selection))
        )
        selected = {
            str(item.observation_id)
            for item in sequence_reader.iter_index()
            if item.offset in set(resolve_selection_offsets(sequence_reader, selection))
        }
        run_id = VisualEvidenceRunId(request.identity())
        with VisualEvidenceRunWriter(
            output_dir=request.output_dir,
            sequence_name=CI_FIXTURE_ID,
            run_id=run_id,
            run_index=request.run_number(),
            sequence_artifact_id=sequence.artifact_id,
            region_artifact_id=regional.manifest.run_id,
            selection_id=selection_identity(SequenceArtifactId(sequence.artifact_id), selection),
            configuration_digest=request.config_digest,
        ) as writer:
            for frame in regional.iter_frames():
                if str(frame.image.source_observation_id) not in selected:
                    continue
                features = (
                    (_canned_dense_feature(writer, run_id, frame.image.source_observation_id),)
                    if self._dense_features
                    else ()
                )
                writer.add_result(
                    VisualEvidenceResult(
                        source_observation_id=frame.image.source_observation_id,
                        prepared_image=frame.image,
                        features=features,
                    )
                )
            if self._stamp_execution_time:
                writer.add_stage_outcomes(
                    (
                        StageOutcome(
                            stage_id="fixture_visual_inference",
                            status=StageStatus.SUCCEEDED,
                            output=None,
                            duration_ms=(time.perf_counter() - began) * 1000,
                        ),
                    )
                )
            manifest = writer.finalize()
        return ArtifactRef(
            stage_id="visual_perception",
            contract="VisualEvidenceArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class CannedSemanticInterpretation:
    """Publish independently auditable canned semantic requests over frozen regional IDs."""

    def __init__(self, *, stamp_execution_time: bool = False) -> None:
        self._stamp_execution_time = stamp_execution_time

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        sequence = request.inputs["sequence"][0]
        regional = RegionPerceptionRunReader(request.directory_of(request.inputs["regions"][0]))
        run_id = request.identity()
        grouped: dict[SourceObservationId, list[Region2D]] = {}
        for region in regional.read_regions():
            assert region.source_observation_id is not None
            grouped.setdefault(region.source_observation_id, []).append(region)
        sequence_reader = SequenceArtifactReader(request.directory_of(sequence))
        selection = (
            FullSequenceSelection()
            if request.observation_selection is None
            else decode_selection(dict(request.observation_selection))
        )
        selected = {
            str(item.observation_id)
            for item in sequence_reader.iter_index()
            if item.offset in set(resolve_selection_offsets(sequence_reader, selection))
        }
        views = []
        for frame in regional.iter_frames():
            if str(frame.image.source_observation_id) not in selected:
                continue
            index = int(str(frame.image.source_observation_id).rsplit("-", 1)[1])
            original = _perception_result(
                run_id, index, claims_of="run-a", sequence_artifact_id=sequence.artifact_id
            )
            regions = tuple(grouped[frame.image.source_observation_id])
            remap = dict(
                zip(
                    (PALLET_REGION, SHELF_REGION), (item.region_id for item in regions), strict=True
                )
            )
            views.append(
                replace(
                    original,
                    region_artifact_id=regional.manifest.run_id,
                    regions=regions,
                    claims=tuple(
                        replace(item, region_id=remap[cast(RegionId, item.region_id)])
                        for item in original.claims
                    ),
                )
            )
        with SemanticInterpretationRunWriter(
            output_dir=request.output_dir,
            run_id=SemanticInterpretationRunId(run_id),
            sequence_name=CI_FIXTURE_ID,
            run_index=request.run_number(),
            sequence_artifact_id=sequence.artifact_id,
            region_artifact_id=regional.manifest.run_id,
            selection_id=selection_identity(SequenceArtifactId(sequence.artifact_id), selection),
            configuration_digest=request.config_digest,
        ) as writer:
            write_semantic_views(
                writer,
                views,
                run_id,
                created_at=datetime.now(UTC).isoformat() if self._stamp_execution_time else "",
            )
            manifest = writer.finalize()
        return ArtifactRef(
            stage_id="semantic_interpretation",
            contract="SemanticInterpretationArtifact",
            artifact_id=run_id,
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class CannedRegionPerception:
    """Test double: publishes the regions of the canned perception as a region run artifact.

    ``region_perception`` has a real executor, but it runs the discovery model, and a CI
    machine has none: this stands in for the stage exactly as :class:`CannedPerception` stands
    in for ``visual_perception``, publishing the very regions the latter reports so the two
    artifacts of one execution stay consistent with each other.

    It publishes the regions' masks and one discovery audit per frame too, so what it writes
    is a complete region run: the real ``VisualPerceptionExecutor`` could replay it. The audit
    records no pass and no rejection, which is the honest statement for a canned discovery.
    """

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        (sequence,) = request.inputs["sequence"]
        reader = SequenceArtifactReader(request.directory_of(sequence))
        selection = (
            FullSequenceSelection()
            if request.observation_selection is None
            else decode_selection(dict(request.observation_selection))
        )
        offsets = set(resolve_selection_offsets(reader, selection))
        selected = {
            str(entry.observation_id) for entry in reader.iter_index() if entry.offset in offsets
        }
        run_id = request.identity()
        results = [
            result
            for result in canned_perception_results(str(run_id), sequence.artifact_id)
            if str(result.source_observation_id) in selected
        ]
        regions = [region for result in results for region in result.regions]
        observations = sorted({str(result.source_observation_id) for result in results})
        provenance = next(iter(regions)).provenance if regions else None
        with RegionPerceptionRunWriter(output_dir=request.output_dir) as writer:
            writer.write_manifest(
                RegionPerceptionRunManifest(
                    run_id=run_id,
                    run_index=request.run_number(),
                    sequence_artifact_id=sequence.artifact_id,
                    source_observation_ids=observations,
                    sequence_name=CI_FIXTURE_ID,
                    selection_id=selection_identity(
                        SequenceArtifactId(sequence.artifact_id), selection
                    ),
                    created_at="2026-01-01T00:00:00Z",
                    prepared_image_id="canned-prepared-image",
                    discovery_backend_id=provenance.backend_id if provenance else "canned",
                    discovery_model=provenance.model if provenance else "canned",
                    discovery_version=provenance.version if provenance else "1",
                    discovery_configuration_fingerprint=request.config_digest,
                    normalization_policy_fingerprint=request.config_digest,
                    region_count=len(regions),
                )
            )
            for result in results:
                observation_id = result.source_observation_id
                writer.add_region_discovery_audit(
                    RegionDiscoveryAudit(
                        source_observation_id=observation_id,
                        backend=result.regions[0].provenance,
                        passes=(),
                        diagnostics=(),
                        pass_rejections=(),
                        discovery_identity=RegionDiscoveryIdentity.effective(),
                        normalization_rejections=(),
                        merge_decisions=(),
                    )
                )
                writer.write_frame(_canned_prepared(observation_id))
                for region in namespace_regions(
                    run_id=run_id,
                    source_observation_id=observation_id,
                    regions=tuple(result.regions),
                ).regions:
                    writer.write_region(writer.externalize_region_mask(observation_id, region))
            region_manifest = writer.finalize()
        return ArtifactRef(
            stage_id="region_perception",
            contract="RegionPerceptionRunArtifact",
            artifact_id=run_id,
            content_hash=inventory_digest(region_manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


def dry_run_executors() -> dict[str, object]:
    """The executors a dry run of a real-data harness injects: perception, and nothing else.

    A CI machine has no model, so perception is what gets replaced; every other stage of the
    harness is composed by the runtime from the configuration, as on the GPU machine. It is the
    ``module:attribute`` target a harness receives as ``--dry-run-executors``. Perception spans
    two stages since the decomposition, so both are replaced.
    """
    return {
        "region_perception": CannedRegionPerception(),
        "visual_perception": CannedPerception(),
        "semantic_interpretation": CannedSemanticInterpretation(),
    }


def replicable_dry_run_executors() -> dict[str, object]:
    """The dry-run executors of a harness that repeats one configuration in another arm (#665).

    Perception is the only stage replaced, as in :func:`dry_run_executors`, and each execution
    stamps its results with its own time, as a real one does: the replica of an arm is then
    another inference execution, with its own ContextRun and its own ContextMap.
    """
    return {
        "region_perception": CannedRegionPerception(),
        "visual_perception": CannedPerception(stamp_execution_time=True),
        "semantic_interpretation": CannedSemanticInterpretation(stamp_execution_time=True),
    }


def campaign_dry_run_executors() -> dict[str, object]:
    """The dry-run executors of the #670 campaign: replicable perception with a dense channel.

    As :func:`replicable_dry_run_executors`, and each result also holds a DINOv3 dense feature,
    as the campaign's baseline perception does: its dense-channel arm samples it in the
    association it recomputes over the baseline's perception run.
    """
    return {
        "region_perception": CannedRegionPerception(),
        "visual_perception": CannedPerception(stamp_execution_time=True, dense_features=True),
        "semantic_interpretation": CannedSemanticInterpretation(stamp_execution_time=True),
    }


class _DryRunClipTextTower:
    """A CLIP text tower that encodes every text to one fixed unit vector; it loads nothing."""

    def projection_dimension(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def provide_dry_run_clip_text_tower(config: object, secrets: object) -> _DryRunClipTextTower:
    """Stand in for a machine's CLIP text tower provider: ``(config, secrets) -> runtime``.

    The canned perception holds no CLIP region feature, so a dry run's scoring run scores no
    claim; it proves the stage runs over the pinned perception run, never a similarity.
    """
    return _DryRunClipTextTower()
