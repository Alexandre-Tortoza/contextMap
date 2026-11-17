"""The synthetic Solution 1 chain driven by the runtime, through the real stage executors.

Contract evidence only (fake/contract): the sequence and the perception evidence come from test
doubles (a model is exactly what CI must not depend on), but every other stage runs its real
executor through ``run_plan`` and a ``RunJournal``, and writes only into the directory the runtime
hands it. The tests check the layout ``<workspace>/artifacts/<stage>/<run>/``, the derived identity
that makes reruns identical, reuse by reference, and recovery of an interrupted run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from chain import (
    CannedPerception,
    CannedRegionPerception,
    CannedSemanticInterpretation,
    SyntheticSequenceIngestion,
)

from contextmap.artifact import (
    ContextMapArtifactReader,
    ValidationLevel,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    EntityResolutionRunReader,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
)
from contextmap.evaluation import (
    BuildState,
    canonical_real_scenario,
    compare_layer_one,
    read_build_state,
    scenario_runtime_document,
)
from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    MotionCorrectionPolicy,
    ScanDisposition,
)
from contextmap.ingestion import (
    FrameId,
    SequenceArtifactReader,
)
from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.runtime import (
    ArtifactRef,
    ContextBranch,
    ContextBuild,
    ContextRun,
    ExecutionPlan,
    FileArtifactStore,
    PipelinePlan,
    ReusePolicy,
    RunJournal,
    SpatialFoundation,
    StageExecutionError,
    StageRequest,
    append_to_branch,
    context_scope,
    create_branch,
    plan_context_build,
    publish_context_build,
    publish_context_run,
    read_context_build,
    read_run,
    resolve_effective_config,
    resolve_plan,
    resolve_spatial_foundation,
    resume_plan,
    run_plan,
)
from contextmap.runtime.executors import (
    ContextMapExecutor,
    EntityResolutionExecutor,
    ExecutorError,
    GeometricMappingExecutor,
    SemanticFusionExecutor,
    SemanticMappingExecutor,
    SensorAssociationExecutor,
    SpatialRelationsExecutor,
    StateEstimationExecutor,
    inventory_digest,
)
from contextmap.semantic_fusion import (
    BASELINE_ACCUMULATION_POLICY_ID,
    QUALITY_AWARE_ACCUMULATION_POLICY_ID,
    BaselineAccumulationPolicy,
    ComponentTreatment,
    GeometryOverlapSupportPolicy,
    QualityAwareAccumulationPolicy,
    QualityInput,
    QualityRamp,
    SemanticFusionRunReader,
)
from contextmap.semantic_interpretation import SemanticInterpretationRunReader
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    GeometrySummaryPolicy,
    SemanticMapId,
    SemanticMappingRunReader,
)
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    QualityComponent,
    SensorAssociationRunReader,
)
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
    SpatialRelationsRunReader,
)
from contextmap.state_estimation import (
    BODY_ENDPOINT,
    GeometryRequirements,
    LookupPolicy,
    StateEstimationRunReader,
    StaticRelationRequirement,
)
from contextmap.state_estimation.backends.external_pose import (
    ExternalPoseConfig,
    ExternalPoseEstimator,
)
from contextmap.visual_perception import (
    VisualEvidenceRunReader,
)

STAGES = [
    "ingestion",
    "region_perception",
    "visual_perception",
    "semantic_interpretation",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
    "context_map",
]
PROVIDED = (
    "region_perception.discovery",
    "visual_features.dense",
    "visual_features.region",
    "semantic_interpretation.interpreter",
)


class _Failing:
    """Wraps an executor and fails once, the way a process that dies mid-stage would."""

    def __init__(self, inner: Any, stage_id: str, failures: list[str]) -> None:
        self._inner, self._stage_id, self._failures = inner, stage_id, failures

    def execute(self, request: StageRequest) -> ArtifactRef:
        if self._failures:
            raise RuntimeError(self._failures.pop())
        ref: ArtifactRef = self._inner.execute(request)
        return ref


def _executors() -> dict[str, Any]:
    estimator = ExternalPoseEstimator(
        ExternalPoseConfig(reference_frame=FrameId("odom"), body_frame=FrameId("base_link"))
    )
    # O mapa sintético está expresso no referencial odom, com +z como eixo vertical; context_map
    # reaproveita exatamente o mesmo up_axis que spatial_relations já declara abaixo, nunca um
    # valor derivado de novo.
    up_axis = AxisDirection.POSITIVE_Z
    return {
        "ingestion": SyntheticSequenceIngestion(),
        # A percepção ocupa dois estágios desde a decomposição: region_perception descobre as
        # regiões que visual_perception consome, e o dublê de cada um publica o seu artifact.
        "region_perception": CannedRegionPerception(),
        "visual_perception": CannedPerception(),
        "semantic_interpretation": CannedSemanticInterpretation(),
        "state_estimation": StateEstimationExecutor(
            estimator,
            downstream=(
                GeometryRequirements(
                    capability="geometric_mapping",
                    modalities=frozenset({"lidar"}),
                    static_relations=(
                        StaticRelationRequirement(from_endpoint=BODY_ENDPOINT, to_endpoint="lidar"),
                    ),
                ),
            ),
        ),
        "geometric_mapping": GeometricMappingExecutor(
            pose_lookup=LookupPolicy.exact(),
            motion_correction=MotionCorrectionPolicy(
                raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.WARN
            ),
            code_version="test",
        ),
        "sensor_association": SensorAssociationExecutor(
            candidates=CandidateGeometryPolicy(max_range_m=None),
            occlusion=OcclusionPolicy(
                cell_size_px=4,
                neighborhood_radius_cells=0,
                depth_margin_m=0.1,
                depth_margin_ratio=0.02,
            ),
            tolerances=DiagnosticTolerances(
                max_pose_time_delta_ns=1_000_000,
                max_map_window_offset_ns=None,
                max_reprojection_p95_px=None,
                max_reprojection_invalid_rate=None,
            ),
            pose_policy=LookupPolicy.exact(),
            code_version="test",
        ),
        "semantic_fusion": SemanticFusionExecutor(
            support_policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=0.5),
            accumulation_policy=BaselineAccumulationPolicy(
                abstention_labels=frozenset({"unknown"})
            ),
            code_version="test",
        ),
        "semantic_mapping": SemanticMappingExecutor(
            policy=EntityMaterializationPolicy(
                geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
            ),
            semantic_map_id=SemanticMapId("semantic-map-ci"),
            code_digest="sha256:" + "cd" * 32,
            code_version="test",
        ),
        "entity_resolution": EntityResolutionExecutor(
            retrieval=CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1),
            builder=MatchEvidenceBuilder(
                ComparisonChannels(
                    geometry=GeometryComparisonPolicy(
                        min_shared_support_jaccard=0.5,
                        min_bounds_iou=0.5,
                        min_bounds_containment=0.9,
                        min_conflict_gap_m=0.5,
                        min_extent_ratio=0.3,
                    )
                )
            ),
            resolution=ConservativeResolutionPolicy(
                use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
            ),
            code_version="test",
        ),
        "spatial_relations": SpatialRelationsExecutor(
            code_version="test",
            policies=RelationsRunPolicies(
                frame_conventions=FrameConventions(
                    map_frame="odom",
                    up_axis=up_axis,
                    forward_axis=AxisDirection.POSITIVE_X,
                ),
                candidate=CandidatePolicy(
                    predicates=(RelationPredicate.NEXT_TO, RelationPredicate.TOUCHING),
                    proximity_radius_m=0.6,
                    directional_radius_m=2.0,
                ),
                geometric=GeometricPredicatePolicy(
                    boundary_tolerance_m=0.02,
                    next_to_max_gap_m=0.5,
                    adjacent_penetration_m=0.05,
                    containment_slack_m=0.05,
                    directional_overlap_fraction=0.5,
                ),
                contact=ContactPredicatePolicy(
                    contact_distance_m=0.05,
                    contact_tolerance_m=0.02,
                    min_contact_points=3,
                    support_height_tolerance_m=0.05,
                    support_footprint_fraction=0.5,
                    leaning_min_tilt_deg=10.0,
                    leaning_max_tilt_deg=80.0,
                    tilt_tolerance_deg=2.0,
                    leaning_min_vertical_overlap_m=0.3,
                ),
                geometry_summary=GeometrySummaryPolicy(
                    sparse_point_threshold=3, connectivity_radius_m=0.5
                ),
            ),
        ),
        "context_map": ContextMapExecutor(up_axis=up_axis),
    }


def _scope(tmp_path: Path) -> tuple[Any, ExecutionPlan]:
    # O perfil canônico do cenário, expresso como configuração do runtime, com o dataset do run.
    document = scenario_runtime_document(canonical_real_scenario())
    document["inputs"] = {"sequence": CI_FIXTURE_ID}
    # O cenário congelado ainda não escolhe backend para estes componentes: eles só passaram a
    # existir no catálogo depois que o cenário foi congelado. A escolha é só deste teste, nunca
    # do cenário: mudar o cenário mudaria seu digest e invalidaria o relatório real registrado.
    components = document["components"]
    components.setdefault("geometric_mapping", {})["pose_lookup"] = {"backend": "lookup-policy-v1"}
    components.setdefault("geometric_mapping", {})["motion_correction"] = {
        "backend": "motion-correction-v1"
    }
    components.setdefault("sensor_association", {})["occlusion"] = {
        "backend": "conservative-depth-support-v1"
    }
    components.setdefault("sensor_association", {})["tolerances"] = {
        "backend": "diagnostic-tolerances-v1"
    }
    components.setdefault("sensor_association", {})["pose_policy"] = {"backend": "lookup-policy-v1"}
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    effective = resolve_effective_config(files=[path])
    return effective, resolve_plan(effective).scope(targets=["context_map"])


def _run(
    effective: Any,
    execution: ExecutionPlan,
    workspace: Path,
    executors: dict[str, Any],
    **options: Any,
) -> tuple[RunJournal, Any]:
    journal = RunJournal.create(workspace, effective, execution)
    record = run_plan(
        execution,
        executors,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        journal=journal,
        **options,
    )
    return journal, record


def _outputs(record: Any) -> dict[str, ArtifactRef]:
    return {stage.stage_id: stage.output for stage in record.stages}


def test_every_stage_writes_its_artifact_into_its_own_directory_of_the_run(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    journal, record = _run(effective, execution, workspace, _executors())

    run = workspace / CI_FIXTURE_ID / "run-0001"
    assert journal.directory == run
    assert read_run(run).status.value == "completed"
    assert not any(path.is_dir() for path in run.iterdir())
    assert sorted(path.name for path in (workspace / "artifacts").iterdir()) == sorted(STAGES)
    refs = _outputs(record)
    assert {stage: ref.location for stage, ref in refs.items()} == {
        stage: f"artifacts/{stage}/run-0001" for stage in STAGES
    }
    artifacts = run.parent.parent / "artifacts"
    assert SequenceArtifactReader(artifacts / "ingestion" / run.name).verify_integrity() == []
    assert (
        RegionPerceptionRunReader(artifacts / "region_perception" / run.name).verify_integrity()
        == []
    )
    assert (
        VisualEvidenceRunReader(artifacts / "visual_perception" / run.name).verify_integrity() == []
    )
    assert (
        SemanticInterpretationRunReader(
            artifacts / "semantic_interpretation" / run.name
        ).verify_integrity()
        == []
    )
    assert (
        StateEstimationRunReader(artifacts / "state_estimation" / run.name).verify_integrity() == []
    )
    with GeometricMapArtifactReader(artifacts / "geometric_mapping" / run.name) as geometry:
        assert geometry.verify_integrity() == []
    assert (
        SensorAssociationRunReader(
            run.parent.parent / "artifacts" / "sensor_association" / run.name
        ).verify_integrity()
        == []
    )
    assert (
        SemanticFusionRunReader(
            run.parent.parent / "artifacts" / "semantic_fusion" / run.name
        ).verify_integrity()
        == []
    )
    mapping = SemanticMappingRunReader(
        run.parent.parent / "artifacts" / "semantic_mapping" / run.name
    )
    assert mapping.verify_integrity() == []
    assert len(mapping.entity_ids()) == 2  # a palete e o poste, sem fusão entre suportes
    resolution = EntityResolutionRunReader(
        run.parent.parent / "artifacts" / "entity_resolution" / run.name
    )
    assert resolution.verify_integrity() == []
    relations = SpatialRelationsRunReader(
        run.parent.parent / "artifacts" / "spatial_relations" / run.name
    )
    assert relations.verify_integrity() == []
    with ContextMapArtifactReader.open(
        run.parent.parent / "artifacts" / "context_map" / run.name
    ) as context_map_reader:
        context_map = context_map_reader.context_map()
    assert len(context_map.entities) == 2  # a palete e o poste, sem fusão entre suportes


def test_no_stage_writes_outside_the_directory_the_runtime_gave_it(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _run(effective, execution, workspace, _executors())

    # O journal e os artifacts ficam separados; não existe registro científico paralelo.
    assert {path.name for path in workspace.iterdir()} == {
        CI_FIXTURE_ID,
        "artifacts",
        ".run-index.lock",
    }
    assert [path.name for path in (workspace / CI_FIXTURE_ID).iterdir()] == ["run-0001"]
    assert not any(path.name == "runs.json" for path in workspace.rglob("*"))


def test_a_rerun_of_the_same_execution_produces_the_same_identities_and_content(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _, first = _run(effective, execution, workspace, _executors())
    _, second = _run(effective, execution, workspace, _executors())

    one, two = _outputs(first), _outputs(second)
    assert set(one) == set(two) == set(STAGES)
    for stage in STAGES:
        assert one[stage].artifact_id == two[stage].artifact_id, stage
        assert one[stage].content_hash == two[stage].content_hash, stage
        assert one[stage].location != two[stage].location  # runs distintos, pastas distintas


def test_a_third_run_reuses_every_stage_by_reference_and_copies_nothing(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    store = FileArtifactStore(
        tmp_path / "index",
        verify=lambda ref: ref.location is not None and (workspace / ref.location).is_dir(),
    )
    # A ingestion fake não nomeia a fonte que lê: a política a declara para o reuso.
    policy = ReusePolicy(
        store=store, code_identity="code-1", identities={"ingestion": {"source": "bag-A"}}
    )
    _, first = _run(effective, execution, workspace, _executors(), reuse=policy)

    journal, second = _run(effective, execution, workspace, _executors(), reuse=policy)

    assert {s.decision.kind for s in second.stages if s.decision is not None} == {"reused"}
    assert not any(path.is_dir() for path in journal.directory.iterdir())  # só o diário
    assert _outputs(second) == _outputs(first)  # as mesmas referências, para a pasta do run 1
    assert all(
        ref.location is not None and ref.location.endswith("/run-0001")
        for ref in _outputs(second).values()
    )


def test_an_interrupted_run_is_resumed_from_the_completed_stages_and_leaves_no_partial_artifact(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    store = FileArtifactStore(
        tmp_path / "index",
        verify=lambda ref: ref.location is not None and (workspace / ref.location).is_dir(),
    )
    # A ingestion fake não nomeia a fonte que lê: a política a declara para o reuso.
    policy = ReusePolicy(
        store=store, code_identity="code-1", identities={"ingestion": {"source": "bag-A"}}
    )
    failing = _executors()
    failing["semantic_fusion"] = _Failing(failing["semantic_fusion"], "semantic_fusion", ["boom"])

    with pytest.raises(StageExecutionError, match="boom"):
        _run(effective, execution, workspace, failing, reuse=policy)

    interrupted = workspace / CI_FIXTURE_ID / "run-0001"
    assert read_run(interrupted).status.value == "failed"
    # Os estágios concluídos existem; o que falhou não deixou pasta, nem temporária.
    assert (
        interrupted.parent.parent / "artifacts" / "geometric_mapping" / interrupted.name
    ).is_dir()
    assert (
        interrupted.parent.parent / "artifacts" / "sensor_association" / interrupted.name
    ).is_dir()
    assert not (
        interrupted.parent.parent / "artifacts" / "semantic_fusion" / interrupted.name
    ).exists()
    assert not list(interrupted.rglob(".tmp-*"))

    journal = RunJournal.create(workspace, effective, execution)
    record = resume_plan(
        interrupted,
        execution,
        _executors(),
        reuse=policy,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        journal=journal,
    )

    refs = _outputs(record)
    resumed = "run-0002"
    assert refs["semantic_fusion"].location == f"artifacts/semantic_fusion/{resumed}"
    assert refs["semantic_mapping"].location == f"artifacts/semantic_mapping/{resumed}"
    assert refs["entity_resolution"].location == f"artifacts/entity_resolution/{resumed}"
    assert refs["spatial_relations"].location == f"artifacts/spatial_relations/{resumed}"
    for stage in STAGES[:5]:  # os concluídos foram reutilizados por referência
        assert refs[stage].location == f"artifacts/{stage}/run-0001"
    assert read_run(journal.directory).status.value == "completed"
    assert (
        SemanticFusionRunReader(
            journal.directory.parent.parent
            / "artifacts"
            / "semantic_fusion"
            / journal.directory.name
        ).verify_integrity()
        == []
    )


def test_a_stage_that_gets_several_runs_for_one_input_is_refused_not_resolved(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    executors = _executors()
    _, record = _run(effective, execution, workspace, executors)
    refs = _outputs(record)
    request = StageRequest(
        stage_id="sensor_association",
        inputs={
            "sequence": (refs["ingestion"],),
            "regions": (refs["region_perception"], refs["region_perception"]),
            "trajectory": (refs["state_estimation"],),
            "geometry": (refs["geometric_mapping"],),
        },
        components={},
        config_digest="d",
        output_dir=workspace / CI_FIXTURE_ID / "run-0009" / "sensor_association",
        workspace=workspace,
    )

    with pytest.raises(ValueError, match="exactly one run of input 'regions'"):
        executors["sensor_association"].execute(request)


def test_resolution_and_relations_are_derived_from_the_entities_without_rewriting_them(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"

    _, record = _run(effective, execution, workspace, _executors())

    run = workspace / CI_FIXTURE_ID / "run-0001"
    mapping = SemanticMappingRunReader(
        run.parent.parent / "artifacts" / "semantic_mapping" / run.name
    )
    resolution = EntityResolutionRunReader(
        run.parent.parent / "artifacts" / "entity_resolution" / run.name
    )
    relations = SpatialRelationsRunReader(
        run.parent.parent / "artifacts" / "spatial_relations" / run.name
    )
    # A resolução decide sobre as entidades materializadas e guarda a origem de cada uma; um
    # candidato sem prova de identidade fica separado (sem fusão silenciosa).
    resolved = resolution.resolved_entities()
    members = {
        member.entity_ref.entity_id for entity in resolved.entities for member in entity.members
    }
    assert members == set(mapping.entity_ids())
    lineage = resolution.manifest.lineage
    assert str(lineage.semantic_mapping_run_id) == str(mapping.manifest.run_id)
    # As relações cobrem exatamente as entidades resolvidas e nascem do run de resolução.
    assert str(relations.manifest.lineage.entity_resolution_run_id) == str(resolution.run_id)
    refs = _outputs(record)
    assert refs["entity_resolution"].artifact_id == str(resolution.run_id)
    assert refs["spatial_relations"].artifact_id == str(relations.manifest.run_id)
    assert refs["spatial_relations"].content_hash == inventory_digest(
        relations.manifest.file_inventory
    )


class TestFusionOverSeveralContextRuns:
    """a build fuses the runs of several ContextRuns as separate, correlated evidence.

    The canned perception double answers the same frames for any run, so a second perception run
    with another configuration is repeated inference over the same physical observations.
    """

    @staticmethod
    def _stage(
        workspace: Path, stage_id: str, inputs: dict[str, list[ArtifactRef]], digest: str
    ) -> ArtifactRef:
        request = StageRequest(
            stage_id=stage_id,
            inputs={name: tuple(refs) for name, refs in inputs.items()},
            components={},
            config_digest=digest,
            output_dir=workspace / CI_FIXTURE_ID / f"run-{digest[-4:]}" / stage_id,
            workspace=workspace,
        )
        ref: ArtifactRef = _executors()[stage_id].execute(request)
        return ref

    @staticmethod
    def _first(tmp_path: Path) -> tuple[Path, dict[str, ArtifactRef]]:
        effective, execution = _scope(tmp_path)
        workspace = tmp_path / "ws"
        _, record = _run(effective, execution, workspace, _executors())
        return workspace, _outputs(record)

    def _association(
        self, workspace: Path, refs: dict[str, ArtifactRef], perception: ArtifactRef, digest: str
    ) -> ArtifactRef:
        return self._stage(
            workspace,
            "sensor_association",
            {
                "sequence": [refs["ingestion"]],
                "regions": [perception],
                "trajectory": [refs["state_estimation"]],
                "geometry": [refs["geometric_mapping"]],
            },
            digest,
        )

    def _fuse(
        self,
        workspace: Path,
        refs: dict[str, ArtifactRef],
        associations: list[ArtifactRef],
        perceptions: list[ArtifactRef],
    ) -> ArtifactRef:
        return self._stage(
            workspace,
            "semantic_fusion",
            {
                "sequence": [refs["ingestion"]],
                "association": associations,
                "regions": perceptions,
                "semantic_interpretation": [
                    refs["semantic_interpretation"]
                    if ref.artifact_id == refs["region_perception"].artifact_id
                    else self._semantic_of(workspace, ref, refs["ingestion"])
                    for ref in perceptions
                ],
                "geometry": [refs["geometric_mapping"]],
            },
            "sha256:fuse",
        )

    @classmethod
    def _semantic_of(
        cls, workspace: Path, regional: ArtifactRef, sequence: ArtifactRef
    ) -> ArtifactRef:
        """Select semantic evidence independently for each regional producer."""
        from contextmap.semantic_interpretation import SemanticInterpretationRunReader

        assert regional.location is not None
        path = workspace / regional.location
        output = path.parent / "semantic_interpretation"
        if not output.exists():
            return cls._stage(
                workspace,
                "semantic_interpretation",
                {"sequence": [sequence], "regions": [regional]},
                "sha256:0002",
            )
        manifest = SemanticInterpretationRunReader(output).manifest
        return ArtifactRef(
            stage_id="semantic_interpretation",
            contract="SemanticInterpretationArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=output.relative_to(workspace).as_posix(),
        )

    @staticmethod
    def _counts(workspace: Path, fusion: ArtifactRef) -> dict[str, Any]:
        path = workspace / str(fusion.location) / "metrics" / "counts.json"
        counts: dict[str, Any] = json.loads(path.read_text("utf-8"))
        return counts

    def test_repeated_inference_stays_one_physical_observation_per_frame(
        self, tmp_path: Path
    ) -> None:
        workspace, refs = self._first(tmp_path)
        perception = self._stage(
            workspace, "region_perception", {"sequence": [refs["ingestion"]]}, "sha256:0002"
        )
        association = self._association(workspace, refs, perception, "sha256:0002")

        fused = self._fuse(
            workspace,
            refs,
            [refs["sensor_association"], association],
            [refs["region_perception"], perception],
        )

        single = self._counts(workspace, refs["semantic_fusion"])
        both = self._counts(workspace, fused)
        assert both["physical_observations"] == single["physical_observations"]
        assert both["inference_results"] == 2 * single["inference_results"]
        lineage = SemanticFusionRunReader(workspace / str(fused.location)).manifest.lineage
        assert set(lineage.association_run_ids) == {
            refs["sensor_association"].artifact_id,
            association.artifact_id,
        }
        assert set(lineage.region_artifact_ids) == {
            refs["region_perception"].artifact_id,
            perception.artifact_id,
        }

    def test_a_run_reached_through_two_context_runs_is_one_input(self, tmp_path: Path) -> None:
        workspace, refs = self._first(tmp_path)

        fused = self._fuse(
            workspace,
            refs,
            [refs["sensor_association"], refs["sensor_association"]],
            [refs["region_perception"], refs["region_perception"]],
        )

        assert self._counts(workspace, fused) == self._counts(workspace, refs["semantic_fusion"])

    def test_two_associations_of_one_perception_run_are_refused(self, tmp_path: Path) -> None:
        workspace, refs = self._first(tmp_path)
        again = self._association(workspace, refs, refs["region_perception"], "sha256:0003")

        with pytest.raises(ExecutorError, match="regional artifact"):
            self._fuse(
                workspace,
                refs,
                [refs["sensor_association"], again],
                [refs["region_perception"]],
            )

    def test_an_association_of_a_perception_run_left_out_is_refused(self, tmp_path: Path) -> None:
        workspace, refs = self._first(tmp_path)
        perception = self._stage(
            workspace, "region_perception", {"sequence": [refs["ingestion"]]}, "sha256:0002"
        )
        association = self._association(workspace, refs, perception, "sha256:0002")

        with pytest.raises(ExecutorError, match="unselected region"):
            self._fuse(workspace, refs, [association], [refs["region_perception"]])


class TestQualityAwareFusion:
    """the runtime runs the quality-aware accumulation over the association's quality.

    The chain's own fusion ran the baseline policy; the quality-aware arm reads the very same
    association, perception, geometry and sequence runs, so the baseline and quality-aware
    experiment arms differ only in the accumulation policy.
    """

    DEPTH = QualityRamp(quality_input=QualityInput.SUPPORT_DEPTH_MEDIAN_M, good=3.0, bad=6.0)
    REPROJECTION = QualityRamp(quality_input=QualityInput.REPROJECTION_MEDIAN_PX, good=1.0, bad=5.0)
    BASELINE = BaselineAccumulationPolicy(abstention_labels=frozenset({"unknown"}))

    def _policy(self, **changes: Any) -> QualityAwareAccumulationPolicy:
        fields: dict[str, Any] = {
            "definitions_version": "observation-quality-v3",
            "ramps": (self.DEPTH, self.REPROJECTION),
            "neutral_factor": 0.5,
            "baseline": self.BASELINE,
        }
        fields.update(changes)
        return QualityAwareAccumulationPolicy(**fields)

    @staticmethod
    def _first(tmp_path: Path) -> tuple[Path, dict[str, ArtifactRef]]:
        effective, execution = _scope(tmp_path)
        workspace = tmp_path / "ws"
        _, record = _run(effective, execution, workspace, _executors())
        return workspace, _outputs(record)

    @staticmethod
    def _request(workspace: Path, refs: dict[str, ArtifactRef], run: str) -> StageRequest:
        return StageRequest(
            stage_id="semantic_fusion",
            inputs={
                "sequence": (refs["ingestion"],),
                "association": (refs["sensor_association"],),
                "regions": (refs["region_perception"],),
                "visual_evidence": (refs["visual_perception"],),
                "semantic_interpretation": (refs["semantic_interpretation"],),
                "geometry": (refs["geometric_mapping"],),
            },
            components={},
            config_digest=f"sha256:{run}",
            output_dir=workspace / "artifacts" / "semantic_fusion" / run,
            workspace=workspace,
        )

    @staticmethod
    def _executor(policy: Any) -> SemanticFusionExecutor:
        return SemanticFusionExecutor(
            support_policy=GeometryOverlapSupportPolicy(min_geometry_count=1, min_overlap=0.5),
            accumulation_policy=policy,
            code_version="test",
        )

    def test_both_policies_fuse_the_same_association_and_only_the_quality_aware_one_weights(
        self, tmp_path: Path
    ) -> None:
        workspace, refs = self._first(tmp_path)
        policy = self._policy()

        aware_ref = self._executor(policy).execute(self._request(workspace, refs, "run-0101"))

        uniform = SemanticFusionRunReader(workspace / str(refs["semantic_fusion"].location))
        aware = SemanticFusionRunReader(workspace / str(aware_ref.location))
        pairs = list(zip(uniform.iter_outcomes(), aware.iter_outcomes(), strict=True))
        assert pairs
        for control, weighted in pairs:
            assert weighted.support == control.support
            assert weighted.evidence.hypotheses == control.evidence.hypotheses
            assert weighted.evidence.uncertainty == control.evidence.uncertainty
            # O braço ciente de qualidade só acrescenta a referência à qualidade da associação.
            assert [
                dataclasses.replace(item, observation_quality=None)
                for item in weighted.evidence.contributions
            ] == list(control.evidence.contributions)
            assert control.evidence.weighting is None
            assert weighted.evidence.weighting is not None
            assert weighted.evidence.weighting.definitions_version == "observation-quality-v3"
            assert control.evidence.provenance.fusion_policy_id == BASELINE_ACCUMULATION_POLICY_ID
            assert weighted.evidence.provenance.fusion_policy_id == (
                QUALITY_AWARE_ACCUMULATION_POLICY_ID
            )
        factors = {
            weight.factor
            for _, weighted in pairs
            if weighted.evidence.weighting is not None
            for weight in weighted.evidence.weighting.contributions
        }
        assert len(factors) > 1  # a profundidade medida pela associação de fato pesa
        # O manifest nomeia a acumulação que rodou, a configuração e as definições de qualidade.
        assert uniform.manifest.fusion_policy_id == BASELINE_ACCUMULATION_POLICY_ID
        assert aware.manifest.fusion_policy_id == QUALITY_AWARE_ACCUMULATION_POLICY_ID
        assert aware.manifest.fusion_configuration_fingerprint == policy.fingerprint()
        assert aware.manifest.evidence_identities["observation_quality"] == (
            "observation-quality-v3",
        )
        assert "observation_quality" not in uniform.manifest.evidence_identities
        assert aware.manifest.lineage == uniform.manifest.lineage

    def test_the_baseline_arm_keeps_its_identity_and_reads_no_quality(self, tmp_path: Path) -> None:
        """The control arm: the baseline run id is still the execution identity alone."""
        workspace, refs = self._first(tmp_path)
        request = self._request(workspace, refs, "run-0103")

        control = self._executor(self.BASELINE).execute(request)

        assert control.artifact_id == request.identity()
        reader = SemanticFusionRunReader(workspace / str(control.location))
        assert all(
            outcome.evidence.weighting is None
            and all(item.observation_quality is None for item in outcome.evidence.contributions)
            for outcome in reader.iter_outcomes()
        )
        assert reader.manifest.fusion_configuration_fingerprint == self.BASELINE.fingerprint()

    def test_a_component_the_association_did_not_measure_takes_the_neutral_factor_and_its_reason(
        self, tmp_path: Path
    ) -> None:
        workspace, refs = self._first(tmp_path)

        aware_ref = self._executor(self._policy()).execute(
            self._request(workspace, refs, "run-0104")
        )

        association = SensorAssociationRunReader(
            workspace / str(refs["sensor_association"].location)
        )
        reader = SemanticFusionRunReader(workspace / str(aware_ref.location))
        checked = 0
        for outcome in reader.iter_outcomes():
            weighting = outcome.evidence.weighting
            assert weighting is not None
            observation_of = {
                item.contribution_id: item.spatial_observation_id
                for item in outcome.evidence.contributions
            }
            for weight in weighting.contributions:
                quality = association.quality(str(observation_of[weight.contribution_id]))
                reprojection = next(
                    item for item in weight.components if item.component == "reprojection_median_px"
                )
                assert quality.reprojection is None
                assert reprojection.treatment is ComponentTreatment.NEUTRAL_FALLBACK
                assert reprojection.factor == 0.5
                assert (
                    reprojection.unavailable_reason
                    == (quality.unavailable[QualityComponent.REPROJECTION])
                )
                checked += 1
        assert checked > 0

    def test_a_policy_for_other_quality_definitions_is_refused_before_accumulating(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextmap.runtime import executors

        workspace, refs = self._first(tmp_path)

        def never(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a support was accumulated under the wrong quality definitions")

        monkeypatch.setattr(executors, "accumulate_quality_aware_evidence", never, raising=False)
        request = self._request(workspace, refs, "run-0105")

        with pytest.raises(ExecutorError, match="observation-quality-v1") as refused:
            self._executor(self._policy(definitions_version="observation-quality-v1")).execute(
                request
            )

        assert "observation-quality-v3" in str(refused.value)
        assert refs["sensor_association"].artifact_id in str(refused.value)
        assert request.output_dir is not None and not request.output_dir.exists()


def _incremental_setup(tmp_path: Path) -> tuple[Path, Any, PipelinePlan, SpatialFoundation]:
    """Run the CI chain once and validate its sequence, trajectory and map as one foundation."""
    effective, execution = _scope(tmp_path)
    workspace = tmp_path / "ws"
    _, record = _run(effective, execution, workspace, _executors())
    refs = _outputs(record)
    foundation = resolve_spatial_foundation(
        workspace,
        sequence=refs["ingestion"],
        state_estimation=refs["state_estimation"],
        geometry=refs["geometric_mapping"],
    )
    return workspace, effective, execution.plan, foundation


def _with(effective: Any, changes: dict[str, Any], name: str) -> tuple[Any, PipelinePlan]:
    """The chain's configuration with ``changes`` merged into its document."""
    document = json.loads(json.dumps(effective.config.to_document()))
    for section, values in changes.items():
        document.setdefault(section, {}).update(values)
    path = Path(effective.sources[-1].identity).parent / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    changed = resolve_effective_config(files=[path])
    return changed, resolve_plan(changed)


def _context_run(
    workspace: Path,
    effective: Any,
    foundation: SpatialFoundation,
    selection: dict[str, Any],
    *,
    components: dict[str, Any] | None = None,
) -> ContextRun:
    """Execute and publish one real ContextRun over ``selection``."""
    changes: dict[str, Any] = {"inputs": {"observation_selection": selection}}
    if components is not None:
        changes["components"] = components
    context_effective, plan = _with(
        effective, changes, f"context-{len(list(workspace.rglob('context_run.json')))}"
    )
    execution = context_scope(plan, foundation)
    journal, record = _run(context_effective, execution, workspace, _executors())
    return publish_context_run(
        journal.directory,
        workspace=workspace,
        foundation=foundation,
        execution=execution,
        record=record,
    )


def _build(
    workspace: Path,
    effective: Any,
    plan: PipelinePlan,
    branch: ContextBranch,
    *,
    revision: int | None = None,
) -> tuple[ContextBuild, Path]:
    """Freeze ``branch`` (at ``revision``), record the build and materialize its map."""
    planned = plan_context_build(workspace, branch, plan, code_identity="test", revision=revision)
    journal = RunJournal.create(workspace, effective, planned.execution)
    publish_context_build(journal.directory, workspace=workspace, build=planned.build)
    run_plan(
        planned.execution,
        _executors(),
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        journal=journal,
    )
    return planned.build, journal.directory


class TestIncrementalBuild:
    """a ContextBuild materializes its frozen ContextRuns through the real assembler.

    The foundation is the CI chain's own sequence, trajectory and map, validated as one; each
    ContextRun is a real perception and association run over a different selection; each build
    runs the real materialization stages and must leave a ContextMapArtifact that verifies.
    """

    def test_each_build_materializes_a_verified_map_and_leaves_earlier_maps_untouched(
        self, tmp_path: Path
    ) -> None:
        workspace, effective, plan, foundation = _incremental_setup(tmp_path)
        branch = create_branch(
            workspace, dataset=CI_FIXTURE_ID, name="incremental", foundation=foundation
        )
        first_frames = {"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 4}
        later_frames = {"kind": "frame_range", "start_frame_index": 4, "end_frame_index": 9}
        branch = append_to_branch(
            workspace, branch, _context_run(workspace, effective, foundation, first_frames)
        )
        first, first_run = _build(workspace, effective, plan, branch, revision=1)
        first_map = first_run.parent.parent / "artifacts" / "context_map" / first_run.name
        before = {path: path.read_bytes() for path in first_map.rglob("*") if path.is_file()}

        branch = append_to_branch(
            workspace, branch, _context_run(workspace, effective, foundation, later_frames)
        )
        second, second_run = _build(workspace, effective, plan, branch, revision=2)

        assert len(first.context_run_ids) == 1 and len(second.context_run_ids) == 2
        for directory in (
            first_map,
            second_run.parent.parent / "artifacts" / "context_map" / second_run.name,
        ):
            report = validate_context_map_artifact(directory, level=ValidationLevel.FULL)
            assert report.status is ValidationStatus.VERIFIED, report.findings
        with ContextMapArtifactReader.open(first_map) as reader:
            first_id = reader.context_map().context_map_id
        with ContextMapArtifactReader.open(
            second_run.parent.parent / "artifacts" / "context_map" / second_run.name
        ) as reader:
            assert reader.context_map().context_map_id != first_id
        after = {path: path.read_bytes() for path in first_map.rglob("*") if path.is_file()}
        assert after == before
        assert read_context_build(first_run, workspace=workspace) == first


def test_a_context_map_that_does_not_verify_fails_its_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the stage validates what it wrote; an invalid map is never a success."""
    from contextmap.artifact import ValidationReport

    def invalid(path: Path, **_options: Any) -> ValidationReport:
        report = validate_context_map_artifact(path, level=ValidationLevel.STRUCTURAL)
        return dataclasses.replace(report, status=ValidationStatus.INVALID)

    monkeypatch.setattr("contextmap.runtime.executors.validate_context_map_artifact", invalid)
    effective, execution = _scope(tmp_path)

    with pytest.raises(StageExecutionError, match="does not verify"):
        _run(effective, execution, tmp_path / "ws", _executors())


class TestPartitionInvariance:
    """the partition of the observations into ContextRuns is not science.

    Deterministic canned perception over the CI chain's three frames, every other stage real.
    Each build is read through the evaluation's own two-layer state, the same one the real-data
    arm compares (``experiments/incremental-partition-invariance-20260927/``): with a
    deterministic perception both layers must be equal, not only statistically close.
    """

    WHOLE: ClassVar[dict[str, Any]] = {
        "kind": "frame_range",
        "start_frame_index": 0,
        "end_frame_index": 9,
    }

    @staticmethod
    def _materialize(
        tmp_path: Path, name: str, selections: list[dict[str, Any]]
    ) -> tuple[BuildState, ContextBuild]:
        (tmp_path / name).mkdir()
        workspace, effective, plan, foundation = _incremental_setup(tmp_path / name)
        branch = create_branch(workspace, dataset=CI_FIXTURE_ID, name=name, foundation=foundation)
        for selection in selections:
            context = _context_run(workspace, effective, foundation, selection)
            branch = append_to_branch(workspace, branch, context)
        build, run = _build(workspace, effective, plan, branch)
        return read_build_state(run, workspace=workspace), build

    def test_chunking_the_same_observations_gives_the_same_scientific_state(
        self, tmp_path: Path
    ) -> None:
        whole_state, whole = self._materialize(tmp_path, "whole", [self.WHOLE])
        chunked_state, chunked = self._materialize(
            tmp_path,
            "chunked",
            [
                {"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 4},
                {"kind": "frame_range", "start_frame_index": 4, "end_frame_index": 9},
            ],
        )

        comparison = compare_layer_one(chunked_state.layer_one, whole_state.layer_one)
        assert comparison.equal, [item for item in comparison.dimensions if not item.equal]
        assert chunked_state.layer_one == whole_state.layer_one
        assert chunked_state.layer_two == whole_state.layer_two
        assert whole_state.counts.physical_observations == 3
        assert whole_state.layer_one.entities, "the chain must materialize entities to compare"
        assert whole_state.layer_two.claims, "the chain must carry claims to compare"
        for state in (whole_state, chunked_state):
            assert state.validation.status == ValidationStatus.VERIFIED.value
            assert state.integrity_problems == ()
        # As identidades de execução diferem, como devem.
        assert len(whole.context_run_ids) == 1 and len(chunked.context_run_ids) == 2
        assert whole.identity != chunked.identity
        assert whole_state.context_build_id != chunked_state.context_build_id

    def test_repeated_inference_adds_results_but_no_physical_observation(
        self, tmp_path: Path
    ) -> None:
        once_state, _ = self._materialize(tmp_path, "once", [self.WHOLE])
        # [0, 10) cobre os mesmos três frames que [0, 9), sob outra identidade de percepção.
        twice_state, twice = self._materialize(
            tmp_path,
            "twice",
            [self.WHOLE, {"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 10}],
        )

        once, both = once_state.counts, twice_state.counts
        assert len(twice.context_run_ids) == 2
        assert both.physical_observations == once.physical_observations == 3
        assert both.inference_results == 2 * once.inference_results
        # A mesma geometria, agora vista por duas inferências em cada contribuição.
        assert twice_state.layer_one.supports == once_state.layer_one.supports
        assert len(twice_state.layer_one.contributions) == 2 * len(
            once_state.layer_one.contributions
        )
