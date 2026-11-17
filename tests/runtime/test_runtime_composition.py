"""Tests for the composition root: configuration in, constructed implementations out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import runtime_provider_fixtures
from runtime_documents import (
    CANONICAL_PROMPT_POLICY,
    SHA_A,
    SUPPORT_POLICY,
    TIGHT_CROP_VIEW_POLICY,
    effective_from,
    selected_document,
)
from runtime_fixtures import unavailable_future_stage  # noqa: F401
from runtime_perception import (
    publish_region_run,
)
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.geometric_mapping import GeometricMapArtifactManifest
from contextmap.ingestion import SourceAdapterConfig, SourceTopicMapping
from contextmap.point_representation.backends.geometric_descriptor import GeometricDescriptorEncoder
from contextmap.region_perception.adapters.florence2 import Florence2RegionDiscovery
from contextmap.region_perception.adapters.sam3 import Sam3Config, Sam3RegionDiscovery
from contextmap.runtime import (
    ConfigProblem,
    ConfigurationError,
    PreflightReport,
    preflight,
    resolve_effective_config,
    resolve_plan,
)
from contextmap.runtime.composition import (
    ComposedRuntime,
    FeatureBuildScope,
    RuntimeProvider,
    composable_backends,
    compose,
    compose_executors,
    composed_stages,
    executor_identities,
    resolve_provider,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
    CompositionError,
    ExecutorIdentityMissingError,
    ProviderConfigurationError,
    StageUnavailableError,
)
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    EvidenceChannel,
    GeometryOverlapSupportPolicy,
    QualityAwareAccumulationPolicy,
)
from contextmap.semantic_interpretation import (
    SemanticInterpretationMode,
    SemanticModePrompt,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.semantic_interpretation.adapters.gemini import (
    GeminiSemanticInterpreter,
)
from contextmap.semantic_interpretation.adapters.qwen import (
    QwenSemanticInterpreter,
)
from contextmap.semantic_mapping import SemanticMapId
from contextmap.sensor_association import InterpolationPolicy
from contextmap.state_estimation.backends.external_pose import ExternalPoseEstimator
from contextmap.state_estimation.backends.fast_lio import FastLioEstimator
from contextmap.visual_perception import (
    VisualEvidenceRunId,
)
from contextmap.visual_perception.visual_features import FeatureScope
from contextmap.visual_perception.visual_features.adapters.clip import ClipVisualFeatureBackend
from contextmap.visual_perception.visual_features.adapters.dinov3 import DinoV3DenseFeatureBackend
from contextmap.visual_perception.visual_features.adapters.siglip2 import Siglip2FeatureBackend

REGION = "region_perception.discovery"
INTERPRETER = "semantic_interpretation.interpreter"
DENSE_FEATURES = "visual_features.dense"
REGION_FEATURES = "visual_features.region"


class _Recorder:
    """Stands in for a caller-supplied model runtime and records how it was requested."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[str, ...]]] = []

    def provider(self, name: str = "runtime") -> RuntimeProvider:
        def provide(config: Any, secrets: Any) -> object:
            self.calls.append((config, secrets.names))
            return _FakeRuntime(name)

        return provide


class _FakeRuntime:
    def __init__(self, name: str) -> None:
        self.name = name


def _providers(recorder: _Recorder) -> dict[str, RuntimeProvider]:
    return {
        REGION: recorder.provider("sam3"),
        INTERPRETER: recorder.provider("qwen"),
    }


def _compose(
    tmp_path: Path,
    *,
    document: dict[str, Any] | None = None,
    recorder: _Recorder | None = None,
    **options: Any,
) -> ComposedRuntime:
    recorder = recorder or _Recorder()
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    return compose(effective_from(tmp_path, document), providers=_providers(recorder), **options)


# Mesmas coordenadas que ``runtime_entities.InMemoryGeometrySource`` deriva para os índices
# 0..7 (``(0.1 * i, 0.2 * (i % 3), 0.05 * (i % 5))``): oito pontos bastam para dois ``make_entity``
# de quatro índices cada, e são o único jeito de um ``GeometricMapArtifact`` real concordar, ponto
# a ponto, com a geometria que ``make_entity`` já resume sozinho a partir daquele fixture.
_ALIGNED_MAP_POINTS: tuple[tuple[float, float, float], ...] = tuple(
    (0.1 * index, 0.2 * (index % 3), 0.05 * (index % 5)) for index in range(8)
)


def _write_aligned_geometric_map(output_dir: Path) -> GeometricMapArtifactManifest:
    """Persist a real, tiny ``GeometricMapArtifact`` whose points are ``make_entity``'s own support.

    Uma única captura LiDAR, uma única pose e um extrínseco identidade: o ponto do mapa fica
    numericamente igual ao ponto bruto do "scan" (``P_map = T_map_body · T_body_lidar · P_source``
    com translação zero e rotação identidade nos dois fatores), então os oito pontos persistidos
    aqui caem exatamente sobre a geometria que ``runtime_entities.make_entity`` resume por conta
    própria a partir do seu próprio ``InMemoryGeometrySource`` -- mesmos índices (via
    ``geometry_id_for``, uma função pura de ``map_id`` e índice), mesmas coordenadas. É o que
    permite ``resolved_entity_geometries`` recomputar bounds e frame idênticos aos que a resolução
    persistiu, sem precisar hackear identidades de geometria.

    O ``MapId`` final é o que ``GeometricMapArtifactWriter`` deriva (``sequence_name--run_id``);
    o chamador lê ``manifest.map_id`` e passa exatamente esse valor a ``make_entity``.
    """
    import struct

    from contextmap.geometric_mapping import (
        GeometricMapArtifactWriter,
        GeometricMapRunId,
        MotionCorrectionPolicy,
        ScanDisposition,
        build_geometry_mapping_plan,
    )
    from contextmap.ingestion import (
        CalibrationSet,
        FrameId,
        FullSequenceSelection,
        LidarObservation,
        PointFieldDataType,
        PointFieldDescriptor,
        RigidTransform,
        SensorId,
        SequenceArtifactId,
        SequenceSelectionResult,
        SourceObservationId,
        SourceProvenance,
        selection_identity,
    )
    from contextmap.shared import SourceTimestamp
    from contextmap.state_estimation import (
        EstimatorProvenance,
        LookupPolicy,
        PoseEstimate,
        PoseProvenance,
        PoseValidity,
        Trajectory,
        TrajectoryId,
        TrajectoryProvenance,
        calibration_identity,
        pose_estimate_id_for,
    )

    stamp = SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:aligned-map")
    sequence_artifact_id = SequenceArtifactId("aligned-sequence")
    trajectory_id = TrajectoryId("aligned--trajectory")
    calibration = CalibrationSet(
        entries={},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("body"),
                child_frame=FrameId("lidar"),
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )
    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        reference_frame=FrameId("map"),
        body_frame=FrameId("body"),
        poses=(
            PoseEstimate(
                estimate_id=pose_estimate_id_for(trajectory_id=trajectory_id, index=0),
                timestamp=stamp,
                parent_frame=FrameId("map"),
                child_frame=FrameId("body"),
                translation_m=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                validity=PoseValidity.VALID,
                provenance=PoseProvenance(source_observation_ids=(SourceObservationId("pose-0"),)),
            ),
        ),
        gaps=(),
        provenance=TrajectoryProvenance(
            estimator=EstimatorProvenance(backend_id="fixture_estimator", backend_version="0"),
            sequence_artifact_id=sequence_artifact_id,
            selection_id="full-sequence",
            calibration_identity=calibration_identity(calibration),
            code_version="test",
        ),
    )
    # Precisão dupla: ``InMemoryGeometrySource`` (usado por ``make_entity``) guarda coordenadas
    # float64 exatas. Empacotar em float32 arredondaria os pontos do "scan" e o mapa recomputado
    # nunca bateria, bit a bit, com os bounds que a resolução persistiu.
    fields = tuple(
        PointFieldDescriptor(
            name=name, offset_bytes=index * 8, data_type=PointFieldDataType.FLOAT64
        )
        for index, name in enumerate(("x", "y", "z"))
    )
    scan = LidarObservation(
        observation_id=SourceObservationId("scan-aligned-0000"),
        sensor_id=SensorId("velodyne"),
        frame_id=FrameId("lidar"),
        timestamp=stamp,
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/lidar"),
        point_count=len(_ALIGNED_MAP_POINTS),
        point_step_bytes=24,
        fields=fields,
        data=b"".join(struct.pack("<3d", *point) for point in _ALIGNED_MAP_POINTS),
        is_dense=True,
    )
    selection = FullSequenceSelection()
    plan = build_geometry_mapping_plan(
        sequence=SequenceSelectionResult(
            sequence_artifact_id=sequence_artifact_id,
            selection=selection,
            selection_id=selection_identity(sequence_artifact_id, selection),
            observations=(scan,),
        ),
        calibration=calibration,
        trajectory=trajectory,
        pose_lookup=LookupPolicy.exact(),
        motion_correction_policy=MotionCorrectionPolicy(
            raw=ScanDisposition.ACCEPT, unknown=ScanDisposition.ACCEPT
        ),
        motion_correction=None,
        state_estimation_run_id=None,
    )
    return GeometricMapArtifactWriter(
        output_dir=output_dir,
        sequence_name="aligned-map",
        run_id=GeometricMapRunId("run-0001"),
        run_index=1,
    ).finalize(plan=plan, aggregation=None, code_version="test")


class TestCanonicalComposition:
    def test_constructs_each_selected_implementation_from_configuration(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path)

        # Discovery e interpretação são construídos por run, sobre o rascunho dele (#661).
        assert composed.region_discovery is not None
        assert composed.semantic_interpreter is not None
        assert isinstance(composed.region_discovery.factory(tmp_path), Sam3RegionDiscovery)
        assert isinstance(composed.semantic_interpreter(tmp_path), QwenSemanticInterpreter)
        assert isinstance(composed.state_estimator, ExternalPoseEstimator)
        assert isinstance(composed.point_encoder, GeometricDescriptorEncoder)
        assert composed.support_policy == GeometryOverlapSupportPolicy(
            min_geometry_count=5, min_overlap=0.3
        )
        assert composed.accumulation_policy == BaselineAccumulationPolicy()

    def test_the_capability_config_carries_the_parameters_and_the_resource_device(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["device"] = "cuda"
        composed = _compose(tmp_path, document=document, recorder=recorder)

        config, _secrets = recorder.calls[0]
        assert isinstance(config, Sam3Config)
        assert (config.checkpoint, config.device, config.prompts) == ("sam3-x", "cuda", ("chair",))
        assert composed.region_discovery is not None
        assert composed.region_discovery.backend.configuration_fingerprint == config.digest
        assert (
            composed.region_discovery.factory(tmp_path).backend_provenance()
            == composed.region_discovery.backend
        )

    def test_a_sam3_concept_set_reaches_the_capability_config_in_order(
        self, tmp_path: Path
    ) -> None:
        """#662: the ordered phrases of ``text_prompt`` are one list of the SAM3 block."""
        fingerprints = []
        for prompts in (["chair", "table", "door"], ["table", "chair", "door"]):
            recorder = _Recorder()
            document = selected_document()
            sam3 = document["components"]["region_perception"]["discovery"]["sam3"]
            sam3["prompts"] = prompts
            composed = _compose(tmp_path, document=document, recorder=recorder)

            config, _secrets = recorder.calls[0]
            assert isinstance(config, Sam3Config)
            assert config.prompts == tuple(prompts)
            assert composed.region_discovery is not None
            assert composed.region_discovery.backend.configuration_fingerprint == config.digest
            fingerprints.append(config.digest)

        assert fingerprints[0] != fingerprints[1]

    def test_swapping_a_backend_is_a_configuration_change_only(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["region_perception"]["discovery"] = {
            "backend": "florence2",
            "florence2": {
                "checkpoint": "microsoft/Florence-2-x",
                "model_version": "1.0",
                "task": "<OD>",
            },
        }

        composed = _compose(tmp_path, document=document)

        assert composed.region_discovery is not None
        assert isinstance(composed.region_discovery.factory(tmp_path), Florence2RegionDiscovery)

    def test_quality_aware_policy_is_built_with_its_nested_values(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["semantic_fusion"]["accumulation"] = {
            "backend": "quality-aware-evidence-accumulation-v1",
            "quality-aware-evidence-accumulation-v1": {
                "definitions_version": "observation-quality-v3",
                "neutral_factor": 0.5,
                "ramps": [{"quality_input": "visible_share", "good": 1.0, "bad": 0.2}],
                "baseline": {"channels": ["semantic_claims", "visual_features"]},
            },
        }

        policy = _compose(tmp_path, document=document).accumulation_policy

        assert isinstance(policy, QualityAwareAccumulationPolicy)
        assert policy.baseline.channels == frozenset(
            {EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.VISUAL_FEATURES}
        )
        assert policy.ramps[0].good == 1.0

    def test_composes_the_external_secret_only_for_backends_that_declare_it(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["components"]["semantic_interpretation"]["interpreter"] = {
            "backend": "gemini",
            "gemini": {
                "model": "gemini-x",
                "timeout_s": 30,
                "max_retries": 2,
                "temperature": 0.0,
                "prompt_policy": dict(CANONICAL_PROMPT_POLICY),
                "view_policy": dict(TIGHT_CROP_VIEW_POLICY),
            },
        }

        composed = _compose(
            tmp_path,
            document=document,
            recorder=recorder,
            environ={"GEMINI_API_KEY": "s3cr3t"},
        )

        assert composed.semantic_interpreter is not None
        assert isinstance(composed.semantic_interpreter(tmp_path), GeminiSemanticInterpreter)
        interpreter_call = recorder.calls[1]
        assert interpreter_call[1] == ("GEMINI_API_KEY",)
        assert recorder.calls[0][1] == ()


SCENE = SemanticInterpretationMode.SCENE
REGION_MODE = SemanticInterpretationMode.REGION
_TIGHT_CROP_VIEWS = SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,))
_CANONICAL_POLICY = SemanticRequestPolicy.from_prompt_policy(
    SemanticPromptPolicy(scene="scene/v1", region="region/v1"), views=_TIGHT_CROP_VIEWS
)
_CANONICAL_PROMPTS = {
    SCENE: SemanticModePrompt(template_id="scene/v1", output_schema="semantic-response/1"),
    REGION_MODE: SemanticModePrompt(template_id="region/v1", output_schema="semantic-response/1"),
}


def _gemini_document(prompt_policy: dict[str, str]) -> dict[str, Any]:
    document = selected_document()
    document["components"]["semantic_interpretation"]["interpreter"] = {
        "backend": "gemini",
        "gemini": {
            "model": "gemini-x",
            "timeout_s": 30,
            "max_retries": 2,
            "temperature": 0.0,
            "prompt_policy": prompt_policy,
            "view_policy": dict(TIGHT_CROP_VIEW_POLICY),
        },
    }
    return document


def _florence2_document(**extra: object) -> dict[str, Any]:
    document = selected_document()
    document["components"]["semantic_interpretation"]["interpreter"] = {
        "backend": "florence2",
        "florence2": {
            "checkpoint": "florence-community/Florence-2-large",
            "revision": SHA_A,
            "task": "<REGION_TO_CATEGORY>",
            "supported_modes": ["region"],
            "precision": "float32",
            "max_new_tokens": 64,
            "temperature": 0.0,
            "view_policy": dict(TIGHT_CROP_VIEW_POLICY),
            **extra,
        },
    }
    return document


class TestSemanticPromptPolicy:
    """#542: the prompt policy is a configured input of Semantic Interpretation.

    Configuration selects it before any inference; the composed runtime hands it to the
    request builder, so ``prompt_template_id`` names the policy the backend actually renders.
    """

    def test_every_request_of_each_mode_names_the_configured_policy(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["semantic_interpretation"]["interpreter"]["qwen"][
            "prompt_policy"
        ] = {"scene": "scene/v1", "region": "region-abstention/v1"}

        composed = _compose(tmp_path, document=document)

        policy = composed.semantic_request_policy
        assert policy is not None
        assert (policy.scene, policy.region) == (
            SemanticModePrompt(template_id="scene/v1", output_schema="semantic-response/1"),
            SemanticModePrompt(
                template_id="region-abstention/v1", output_schema="semantic-response/1"
            ),
        )

    def test_qwen_and_gemini_consume_the_same_backend_neutral_policy(self, tmp_path: Path) -> None:
        policy = {"scene": "scene/v1", "region": "region-abstention/v1"}
        qwen = selected_document()
        qwen["components"]["semantic_interpretation"]["interpreter"]["qwen"]["prompt_policy"] = (
            policy
        )

        from_qwen = _compose(tmp_path, document=qwen)
        from_gemini = _compose(
            tmp_path, document=_gemini_document(policy), environ={"GEMINI_API_KEY": "k"}
        )

        assert from_qwen.semantic_request_policy == from_gemini.semantic_request_policy

    def test_the_prompt_policy_varies_without_touching_the_backend_identity(
        self, tmp_path: Path
    ) -> None:
        """A prompt ablation keeps model, checkpoint and generation settings matched."""
        canonical = _compose(tmp_path, document=selected_document())
        document = selected_document()
        document["components"]["semantic_interpretation"]["interpreter"]["qwen"][
            "prompt_policy"
        ] = {"scene": "scene/v1", "region": "region-abstention/v1"}
        alternative = _compose(tmp_path, document=document)

        assert canonical.semantic_request_policy != alternative.semantic_request_policy
        assert canonical.semantic_interpreter is not None
        assert alternative.semantic_interpreter is not None
        assert (
            canonical.semantic_interpreter(tmp_path).backend_provenance()
            == alternative.semantic_interpreter(tmp_path).backend_provenance()
        )

    def test_an_instruction_following_backend_is_never_given_a_default_policy(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        del document["components"]["semantic_interpretation"]["interpreter"]["qwen"][
            "prompt_policy"
        ]

        with pytest.raises(BackendConfigurationError, match="prompt_policy"):
            _compose(tmp_path, document=document, recorder=recorder)

        assert len(recorder.calls) == 1  # só o runtime de region discovery foi pedido

    @pytest.mark.parametrize(
        ("policy", "message"),
        [
            ({"scene": "scene/v1", "region": "region/v9"}, "unknown semantic prompt template"),
            ({"scene": "region/v1", "region": "region/v1"}, "is a region template"),
            ({"scene": "scene/v1"}, "region"),
        ],
    )
    def test_an_unrenderable_policy_is_refused_before_any_runtime_is_requested(
        self, tmp_path: Path, policy: dict[str, str], message: str
    ) -> None:
        recorder = _Recorder()

        with pytest.raises(BackendConfigurationError, match=message):
            _compose(
                tmp_path,
                document=_gemini_document(policy),
                recorder=recorder,
                environ={"GEMINI_API_KEY": "k"},
            )

        assert len(recorder.calls) == 1  # só o runtime de region discovery foi pedido

    def test_florence2_names_its_task_native_policy_only_for_the_mode_its_task_serves(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path, document=_florence2_document())

        policy = composed.semantic_request_policy
        assert policy is not None
        assert policy.scene is None
        assert policy.region == SemanticModePrompt(
            template_id="florence2-task-prompt/1:<REGION_TO_CATEGORY>",
            output_schema="semantic-response/1",
        )

    def test_florence2_reports_that_it_cannot_consume_a_free_form_policy(
        self, tmp_path: Path
    ) -> None:
        document = _florence2_document(prompt_policy=dict(CANONICAL_PROMPT_POLICY))

        with pytest.raises(BackendConfigurationError, match="task-native"):
            _compose(tmp_path, document=document)

    def test_the_composed_executor_carries_the_selected_policy(self, tmp_path: Path) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["components"]["semantic_interpretation"]["interpreter"]["qwen"][
            "prompt_policy"
        ] = {"scene": "scene/v1", "region": "region-abstention/v1"}

        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(recorder),
            module_available=lambda _name: True,
            environ={},
        )

        policy = executors["semantic_interpretation"]._semantic_request_policy  # type: ignore[attr-defined]
        assert policy.region.template_id == "region-abstention/v1"
        assert policy.scene.template_id == "scene/v1"


class TestSemanticViewPolicy:
    """#524: the evidence-view policy is configured, validated at composition, and composed."""

    @staticmethod
    def _qwen_views(view_policy: dict[str, Any] | None) -> dict[str, Any]:
        document = selected_document()
        qwen = document["components"]["semantic_interpretation"]["interpreter"]["qwen"]
        if view_policy is None:
            del qwen["view_policy"]
        else:
            qwen["view_policy"] = view_policy
        return document

    def test_the_configured_views_become_the_composed_policy(self, tmp_path: Path) -> None:
        from contextmap.semantic_interpretation import (
            SemanticContextBoundary,
            SemanticViewPolicy,
            VisualViewKind,
        )

        document = self._qwen_views(
            {
                "region_views": ["masked_subject", "tight_crop", "contextual_crop"],
                "mask_fill_rgb": [0, 0, 0],
                "context_margin_ratio": 0.5,
                "context_boundary": {"rgb": [255, 0, 0], "width_px": 2},
            }
        )

        composed = _compose(tmp_path, document=document)

        assert composed.semantic_request_policy is not None
        assert composed.semantic_request_policy.views == SemanticViewPolicy(
            region_views=(
                VisualViewKind.MASKED_SUBJECT,
                VisualViewKind.TIGHT_CROP,
                VisualViewKind.CONTEXTUAL_CROP,
            ),
            mask_fill_rgb=(0, 0, 0),
            context_margin_ratio=0.5,
            context_boundary=SemanticContextBoundary(rgb=(255, 0, 0), width_px=2),
        )

    def test_the_views_are_never_defaulted(self, tmp_path: Path) -> None:
        with pytest.raises(BackendConfigurationError, match="view_policy"):
            _compose(tmp_path, document=self._qwen_views(None))

    def test_an_ambiguous_view_policy_is_refused_before_any_runtime_is_requested(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()

        with pytest.raises(BackendConfigurationError, match="mask_fill_rgb"):
            _compose(
                tmp_path,
                document=self._qwen_views({"region_views": ["masked_subject"]}),
                recorder=recorder,
            )

        assert len(recorder.calls) == 1  # só o runtime de region discovery foi pedido

    @pytest.mark.parametrize(
        ("views", "message"),
        [
            (
                {"region_views": ["masked_subject", "tight_crop"], "mask_fill_rgb": [0, 0, 0]},
                "at most 1",
            ),
            ({"region_views": ["contextual_crop"], "context_margin_ratio": 0.5}, "contextual_crop"),
            ({"region_views": ["full_frame", "tight_crop"]}, "full_frame"),
        ],
    )
    def test_florence2_refuses_a_combination_it_cannot_consume_at_composition(
        self, tmp_path: Path, views: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(BackendConfigurationError, match=message):
            _compose(tmp_path, document=_florence2_document(view_policy=views))

    def test_florence2_accepts_its_single_region_filling_view(self, tmp_path: Path) -> None:
        document = _florence2_document(
            view_policy={"region_views": ["masked_subject"], "mask_fill_rgb": [127, 127, 127]}
        )

        composed = _compose(tmp_path, document=document)

        assert composed.semantic_request_policy is not None
        assert [kind.value for kind in composed.semantic_request_policy.views.region_views] == [
            "masked_subject"
        ]

    def test_the_composed_executor_carries_the_selected_views(self, tmp_path: Path) -> None:
        document = self._qwen_views(
            {"region_views": ["masked_subject", "tight_crop"], "mask_fill_rgb": [0, 0, 0]}
        )

        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(_Recorder()),
            module_available=lambda _name: True,
            environ={},
        )

        policy = executors["semantic_interpretation"]._semantic_request_policy.views  # type: ignore[attr-defined]
        assert [kind.value for kind in policy.region_views] == ["masked_subject", "tight_crop"]


class TestFeatureBackendsAreRunScoped:
    def _scope(self, tmp_path: Path) -> FeatureBuildScope:
        return FeatureBuildScope(
            run_id=VisualEvidenceRunId("run-1"),
            feature_stage_id="dense_feature_extraction",
            source_artifact_id="artifact-1",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )

    def test_builds_the_configured_extractors_once_a_run_scope_exists(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path)
        assert composed.dense_features is not None
        assert composed.region_features is not None

        dense = composed.dense_features(self._scope(tmp_path))
        region = composed.region_features(self._scope(tmp_path))

        assert isinstance(dense, DinoV3DenseFeatureBackend)
        assert isinstance(region, ClipVisualFeatureBackend)

    def test_the_region_features_slot_fixes_the_clip_scope(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["visual_features"]["region"]["clip"]["scope"] = "global"

        with pytest.raises(BackendConfigurationError, match="scope"):
            _compose(tmp_path, document=document)

    def _siglip2_document(self, **parameters: object) -> dict[str, Any]:
        document = selected_document()
        document["components"]["visual_features"]["dense"] = {
            "backend": "siglip2",
            "siglip2": {
                "checkpoint": "google/siglip2-so400m-patch16-512",
                "revision": "a" * 40,
                "input_size": 512,
                **parameters,
            },
        }
        return document

    def test_siglip2_is_selectable_as_the_dense_features_backend(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path, document=self._siglip2_document())
        assert composed.dense_features is not None

        dense = composed.dense_features(self._scope(tmp_path))

        assert isinstance(dense, Siglip2FeatureBackend)
        assert dense.required_scope() is FeatureScope.DENSE
        assert dense.backend_provenance().model == "google/siglip2-so400m-patch16-512"

    def test_the_dense_features_slot_fixes_the_siglip2_scope(self, tmp_path: Path) -> None:
        document = self._siglip2_document(scope="global")

        with pytest.raises(BackendConfigurationError, match="scope"):
            _compose(tmp_path, document=document)

    def test_configuration_errors_surface_at_composition_not_at_run_time(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["visual_features"]["dense"]["dinov3"]["revision"] = "main"

        with pytest.raises(BackendConfigurationError, match="revision"):
            _compose(tmp_path, document=document)


class TestOtherBackends:
    def test_fast_lio_gets_its_bundled_subprocess_runner_from_configuration(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
                "runner": {
                    "command": ["fast-lio", "{input_bag}", "{job}", "{output_dir}"],
                    "timeout_s": 60,
                },
            },
        }

        composed = _compose(tmp_path, document=document)

        assert isinstance(composed.state_estimator, FastLioEstimator)

    def test_fast_lio_without_a_runner_is_a_configuration_error(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
            },
        }

        with pytest.raises(BackendConfigurationError, match="runner"):
            _compose(tmp_path, document=document)

    def test_fast_lio_rejects_a_command_without_the_placeholders(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {
            "backend": "fast_lio",
            "fast_lio": {
                "reference_frame": "map",
                "body_frame": "base",
                "fast_lio_ref": "v1",
                "scan_period_ns": 100_000_000,
                "runner": {"command": ["fast-lio"], "timeout_s": 60},
            },
        }

        with pytest.raises(BackendConfigurationError, match="runner"):
            _compose(tmp_path, document=document)

    def test_a_learned_encoder_needs_a_support_policy_and_a_caller_runtime(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        ptv3 = {
            "variant": "ptv3-base",
            "checkpoint": "ptv3-x",
            "checkpoint_hash": "sha256:" + "0" * 64,
            "precision": "float32",
            "grid_size_m": 0.02,
            "output_dimension": 64,
            "pooling": "mean",
            "normalization": "l2",
            "min_support_points": 8,
        }
        document["components"]["point_representation"]["encoder"] = {
            "backend": "ptv3",
            "ptv3": ptv3,
        }
        with pytest.raises(BackendConfigurationError, match="support_policy"):
            _compose(tmp_path, document=document)

        ptv3["support_policy"] = SUPPORT_POLICY
        with pytest.raises(BackendRuntimeMissingError, match="PTv3Runtime"):
            _compose(tmp_path, document=document)

        recorder = _Recorder()
        providers = {**_providers(recorder), "point_representation.encoder": recorder.provider()}
        composed = compose(
            effective_from(tmp_path, document),
            providers=providers,
            module_available=lambda _name: True,
            environ={},
        )
        assert type(composed.point_encoder).__name__ == "PTv3PointEncoder"

    def test_region_discovery_forwards_its_pass_and_normalization_policies(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        sam3 = document["components"]["region_perception"]["discovery"]["sam3"]
        sam3["pass_config"] = {"include_full_frame": True, "max_candidates_per_pass": 10}
        sam3["normalization_config"] = {"minimum_area_pixels": 32}

        composed = _compose(tmp_path, document=document)

        assert composed.region_discovery is not None
        backend = composed.region_discovery.factory(tmp_path)
        assert backend._pass_config.max_candidates_per_pass == 10  # type: ignore[attr-defined]
        assert backend._normalization_config.minimum_area_pixels == 32.0  # type: ignore[attr-defined]
        # A identidade que o plano declara antes do run é a que o backend construído reporta.
        assert composed.region_discovery.discovery_identity == backend.discovery_identity()

    def test_a_scaled_discovery_pass_is_refused_before_any_run(self, tmp_path: Path) -> None:
        # O adapter só materializa passes sem escala: sem regra de reamostragem definida, um
        # tiling escalado falharia o discovery de todo frame, então a composição o recusa antes.
        document = selected_document()
        sam3 = document["components"]["region_perception"]["discovery"]["sam3"]
        sam3["pass_config"] = {
            "include_full_frame": True,
            "tiling": {"tile_width": 64, "tile_height": 64, "scale": 0.5},
        }

        with pytest.raises(BackendConfigurationError, match=r"pass_config.*scale 0\.5"):
            _compose(tmp_path, document=document)

    def test_canonical_2_refuses_tiled_sam2_before_model_loading(self, tmp_path: Path) -> None:
        """The first canonical/2 measurement is deliberately one full-frame SAM2 pass."""
        document = selected_document()
        document["pipeline"]["preset"] = "canonical/2"
        document["components"]["visual_features"]["dense"]["backend"] = "dinov2"
        discovery = document["components"]["region_perception"]["discovery"]
        discovery.clear()
        discovery.update(
            {
                "backend": "sam2",
                "sam2": {
                    "checkpoint": "sam2-x",
                    "model_version": "2.1",
                    "pass_config": {
                        "include_full_frame": True,
                        "tiling": {"tile_width": 64, "tile_height": 64},
                    },
                },
            }
        )

        with pytest.raises(BackendConfigurationError, match=r"canonical/2.*tiling"):
            _compose(tmp_path, document=document, stages=["region_perception"])

    def test_canonical_2_refuses_a_non_sam2_discovery_backend(self, tmp_path: Path) -> None:
        document = selected_document()
        document["pipeline"]["preset"] = "canonical/2"

        with pytest.raises(BackendConfigurationError, match="canonical/2 requires sam2"):
            _compose(tmp_path, document=document, stages=["region_perception"])

    def test_canonical_2_refuses_a_non_qwen_region_interpreter(self, tmp_path: Path) -> None:
        """The profile names the VLM baseline rather than accepting an accidental substitute."""
        document = selected_document()
        document["pipeline"]["preset"] = "canonical/2"
        document["components"]["visual_features"]["dense"]["backend"] = "dinov2"
        document["components"]["region_perception"]["discovery"]["backend"] = "sam2"
        document["components"]["semantic_interpretation"]["interpreter"] = {"backend": "gemini"}

        with pytest.raises(BackendConfigurationError, match="canonical/2 requires Qwen"):
            _compose(tmp_path, document=document, stages=["visual_perception"])

    def test_canonical_2_refuses_a_non_clip_scorer_composed_in_isolation(
        self, tmp_path: Path
    ) -> None:
        """A selected semantic scorer remains part of the frozen canonical profile."""
        document = selected_document()
        document["pipeline"]["preset"] = "canonical/2"
        document["pipeline"]["stages"]["semantic_scoring"] = True
        # O scorer é de semantic_scoring desde a decomposição, e o estágio está desligado por
        # padrão, então o documento base não traz o mapa da capability: ligá-lo é criá-lo.
        document["components"].setdefault("semantic_scoring", {})["scorer"] = {
            "backend": "alphaclip"
        }

        with pytest.raises(BackendConfigurationError, match="requires CLIP semantic scoring"):
            _compose(tmp_path, document=document, stages=["semantic_scoring"])

    def test_canonical_2_profile_enables_raw_semantic_score_fusion(self, tmp_path: Path) -> None:
        document = selected_document()
        base = tmp_path / "base.json"
        base.write_text(json.dumps(document), encoding="utf-8")
        layer = Path(__file__).parents[2] / "examples" / "v0.2.0" / "canonical-2-profile.toml"
        effective = resolve_effective_config(files=[base, layer])

        composed = compose(
            effective,
            stages=["semantic_fusion"],
            module_available=lambda _name: True,
            environ={},
        )

        assert composed.accumulation_policy is not None
        assert isinstance(composed.accumulation_policy, BaselineAccumulationPolicy)
        assert EvidenceChannel.SEMANTIC_SCORES in composed.accumulation_policy.active_channels
        assert EvidenceChannel.VISUAL_FEATURES in composed.accumulation_policy.active_channels
        assert effective.config.components["visual_features.dense"].backend == "dinov2"

    def test_a_mask_conditioned_extractor_needs_the_mask_source_of_its_run(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["visual_features"]["region"] = {
            "backend": "alphaclip",
            "alphaclip": {
                "model_name": "ViT-B/16",
                "base_checkpoint_path": "clip.pt",
                "alpha_checkpoint_path": "alpha.pt",
                "checkpoint_fingerprint": "sha256:" + "1" * 64,
            },
        }
        composed = _compose(tmp_path, document=document)
        assert composed.region_features is not None
        scope = FeatureBuildScope(
            run_id=VisualEvidenceRunId("run-1"),
            feature_stage_id="region_feature_extraction",
            source_artifact_id="artifact-1",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )

        with pytest.raises(BackendConfigurationError, match="mask_source"):
            composed.region_features(scope)

    def test_a_supplied_runtime_replaces_the_bundled_loader_and_its_module_checks(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        composed = compose(
            effective_from(tmp_path),
            providers={
                **_providers(recorder),
                "visual_features.dense": recorder.provider("dino"),
                "visual_features.region": recorder.provider("clip"),
            },
            module_available=lambda name: name not in {"torch", "transformers", "PIL"},
            environ={},
        )
        assert composed.dense_features is not None
        scope = FeatureBuildScope(
            run_id=VisualEvidenceRunId("run-1"),
            feature_stage_id="dense",
            source_artifact_id="artifact-1",
            payload_sink=object(),
        )

        backend = composed.dense_features(scope)

        assert backend._runtime.name == "dino"  # type: ignore[attr-defined]


class TestSourceAdapter:
    def _request(self, source_type: str) -> SourceAdapterConfig:
        return SourceAdapterConfig(
            source_type=source_type, path="bag.bag", topics=SourceTopicMapping()
        )

    def test_builds_the_configured_adapter_for_one_request(self, tmp_path: Path) -> None:
        pytest.importorskip("rosbags")
        composed = _compose(tmp_path)
        assert composed.source_adapter is not None

        adapter = composed.source_adapter(self._request("ros1_bag"))

        assert type(adapter).__name__ == "Ros1BagSourceAdapter"

    def test_refuses_a_request_for_another_adapter_family(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path)
        assert composed.source_adapter is not None

        with pytest.raises(BackendConfigurationError, match="ros2_bag"):
            composed.source_adapter(self._request("ros2_bag"))


class TestAssociationDenseSampling:
    """The optional ``sensor_association.dense_sampling`` component (#651)."""

    _COMPONENT = "sensor_association.dense_sampling"

    def _document(self, parameters: dict[str, Any]) -> dict[str, Any]:
        document = selected_document()
        document["components"]["sensor_association"]["dense_sampling"] = {
            "backend": "dense-feature-sampling-v1",
            "dense-feature-sampling-v1": parameters,
        }
        return document

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("nearest", InterpolationPolicy.NEAREST), ("bilinear", InterpolationPolicy.BILINEAR)],
    )
    def test_the_selected_interpolation_reaches_the_association(
        self, tmp_path: Path, value: str, expected: InterpolationPolicy
    ) -> None:
        composed = _compose(
            tmp_path,
            document=self._document({"interpolation": value}),
            stages=["sensor_association"],
        )

        assert composed.association_dense_sampling is expected

    def test_unselected_it_stays_none_never_a_default_interpolation(self, tmp_path: Path) -> None:
        composed = _compose(tmp_path, stages=["sensor_association"])

        assert composed.association_dense_sampling is None

    @pytest.mark.parametrize("parameters", [{}, {"interpolation": "bicubic"}])
    def test_an_interpolation_the_capability_does_not_define_is_refused(
        self, tmp_path: Path, parameters: dict[str, Any]
    ) -> None:
        with pytest.raises(BackendConfigurationError, match="interpolation") as error:
            _compose(tmp_path, document=self._document(parameters), stages=["sensor_association"])

        assert error.value.component_id == self._COMPONENT


class TestExplicitFailures:
    def test_a_backend_without_a_bundled_runtime_needs_one_from_the_caller(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BackendRuntimeMissingError) as error:
            compose(
                effective_from(tmp_path),
                providers={INTERPRETER: _Recorder().provider()},
                module_available=lambda _name: True,
                environ={},
            )

        assert REGION in str(error.value)
        assert "sam3" in str(error.value)

    def test_a_missing_optional_module_is_reported_with_its_install_hint(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(BackendUnavailableError, match="torch"):
            _compose(tmp_path, module_available=lambda name: name != "torch")

    def test_a_missing_secret_is_reported_by_name(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["semantic_interpretation"]["interpreter"] = {
            "backend": "gemini",
            "gemini": {
                "model": "gemini-x",
                "timeout_s": 30,
                "max_retries": 2,
                "temperature": 0.0,
                "prompt_policy": dict(CANONICAL_PROMPT_POLICY),
                "view_policy": dict(TIGHT_CROP_VIEW_POLICY),
            },
        }

        with pytest.raises(BackendUnavailableError, match="GEMINI_API_KEY"):
            _compose(tmp_path, document=document, environ={})

    def test_a_capability_rejection_happens_before_any_runtime_is_requested(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["components"]["region_perception"]["discovery"]["sam3"] = {
            "checkpoint": "x",
            "model_version": "3.0",
            "strategy": "text_prompt",
        }

        with pytest.raises(BackendConfigurationError, match="requires at least one phrase"):
            _compose(tmp_path, document=document, recorder=recorder)

        assert recorder.calls == []

    def test_an_unknown_parameter_names_the_valid_ones(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"]["external_pose"]["nope"] = 1

        with pytest.raises(BackendConfigurationError, match="reference_frame"):
            _compose(tmp_path, document=document)

    def test_an_incomplete_selection_is_a_configuration_error(self, tmp_path: Path) -> None:
        document = selected_document()
        document["components"]["state_estimation"]["estimator"] = {"backend": None}

        with pytest.raises(ConfigurationError, match=r"state_estimation\.estimator"):
            _compose(tmp_path, document=document)


class TestResolveProvider:
    """Unit tests for resolving a ``"module:attribute"`` string into a callable."""

    def test_resolves_a_real_importable_target(self) -> None:
        provider = resolve_provider(REGION, "runtime_provider_fixtures:load_region_discovery")

        assert provider is runtime_provider_fixtures.load_region_discovery

    @pytest.mark.parametrize(
        "target",
        ["no-colon-at-all", ":load_region_discovery", "runtime_provider_fixtures:", ":"],
    )
    def test_rejects_a_malformed_target(self, target: str) -> None:
        with pytest.raises(ProviderConfigurationError) as error:
            resolve_provider(REGION, target)

        assert error.value.component_id == REGION
        assert error.value.target == target

    def test_rejects_a_module_that_cannot_be_imported(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="could not be imported"):
            resolve_provider(REGION, "no_such_package_exists_for_contextmap:load")

    def test_rejects_an_attribute_the_module_does_not_have(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="no attribute"):
            resolve_provider(REGION, "runtime_provider_fixtures:does_not_exist")

    def test_rejects_an_attribute_that_is_not_callable(self) -> None:
        with pytest.raises(ProviderConfigurationError, match="not callable"):
            resolve_provider(REGION, "runtime_provider_fixtures:not_callable")


class TestDeclaredProviderTargets:
    """``resources.providers``: a declared target resolved lazily by the composition root.

    This is the config-driven counterpart of ``providers=``: a caller of the installed
    ``contextmap`` binary, which never passes ``providers=`` at all, still gets a real
    backend composed when the effective configuration names a ``"module:attribute"``
    target for it (#507's real, previously unmet acceptance criterion).
    """

    def test_a_declared_target_supplies_the_runtime_with_no_explicit_provider(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }

        composed = compose(
            effective_from(tmp_path, document),
            stages=["region_perception"],
            providers={INTERPRETER: recorder.provider("qwen")},  # isola o slot sob teste
            module_available=lambda _name: True,
            environ={},
        )

        assert composed.region_discovery is not None
        assert isinstance(composed.region_discovery.factory(tmp_path), Sam3RegionDiscovery)

    def test_ensure_available_skips_the_module_check_for_a_declared_target(
        self, tmp_path: Path
    ) -> None:
        """dense_features' own bundled loader needs torch/transformers/PIL -- but a
        declared provider supplies the runtime instead, so composing it must never demand
        those modules be importable (composition.py's own module docstring)."""
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            DENSE_FEATURES: "runtime_provider_fixtures:load_dense_features",
            REGION_FEATURES: "runtime_provider_fixtures:load_region_features",
        }

        composed = compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers=_providers(recorder),  # region_discovery e semantic_interpretation
            module_available=lambda _name: False,  # nada está instalado
            environ={},
        )

        assert composed.dense_features is not None
        scope = FeatureBuildScope(
            run_id=VisualEvidenceRunId("run-0001"),
            feature_stage_id="dense_feature_extraction",
            source_artifact_id="run-0001",
            payload_sink=object(),
            prepared_image_root=tmp_path,
        )
        composed.dense_features(scope)
        assert runtime_provider_fixtures.CALLS[-1][0] == "dense_features"

    def test_an_explicit_provider_wins_over_a_declared_target_and_is_reported(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        explicit = recorder.provider("explicit-sam3")
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        composed = compose(
            effective_from(tmp_path, document),
            stages=["region_perception"],
            providers={REGION: explicit, INTERPRETER: recorder.provider("qwen")},
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert composed.region_discovery is not None
        assert overrides == [REGION]
        # o provider explícito de fato construiu o backend, não o alvo declarado.
        assert recorder.calls[0][0] is not None

    def test_no_override_is_reported_when_only_a_declared_target_exists(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        compose(
            effective_from(tmp_path, document),
            stages=["visual_perception"],
            providers={INTERPRETER: recorder.provider("qwen")},
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert overrides == []

    def test_an_unresolvable_declared_target_raises_a_provider_configuration_error(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {REGION: "not-a-valid-target"}

        with pytest.raises(ProviderConfigurationError) as error:
            compose(
                effective_from(tmp_path, document),
                stages=["region_perception"],
                providers={INTERPRETER: recorder.provider("qwen")},
                module_available=lambda _name: True,
                environ={},
            )

        assert error.value.component_id == REGION

    def test_compose_executors_reports_the_override_through_its_own_callback(
        self, tmp_path: Path
    ) -> None:
        recorder = _Recorder()
        document = selected_document()
        document["resources"]["providers"] = {
            REGION: "runtime_provider_fixtures:load_region_discovery"
        }
        overrides: list[str] = []

        executors = compose_executors(
            effective_from(tmp_path, document),
            providers=_providers(recorder),
            module_available=lambda _name: True,
            environ={},
            on_provider_override=overrides.append,
        )

        assert "visual_perception" in executors
        assert overrides == [REGION]


class TestStagesAndExtensionPoints:
    def test_a_stage_that_is_not_selected_is_not_built(self, tmp_path: Path) -> None:
        document = selected_document()
        document["pipeline"]["stages"]["point_representation"] = False

        composed = _compose(tmp_path, document=document)

        assert composed.point_encoder is None
        assert "point_representation" not in composed.stages

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_stages_without_an_implemented_capability_are_listed_not_simulated(
        self, tmp_path: Path
    ) -> None:
        composed = _compose(tmp_path)

        assert set(composed.unavailable_stages) == {"scene_graph"}
        assert "milestone" in composed.unavailable_stages["scene_graph"]

    @pytest.mark.usefixtures("unavailable_future_stage")
    def test_asking_explicitly_for_an_unavailable_stage_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(StageUnavailableError, match="scene_graph"):
            _compose(tmp_path, stages=["scene_graph"])

    def test_a_subset_of_stages_builds_only_those(self, tmp_path: Path) -> None:
        recorder = _Recorder()

        composed = _compose(tmp_path, recorder=recorder, stages=["state_estimation"])

        assert composed.state_estimator is not None
        assert composed.region_discovery is None
        assert recorder.calls == []

    def test_an_unknown_stage_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="nope"):
            _compose(tmp_path, stages=["nope"])


class TestCatalogAgreement:
    def test_every_catalog_backend_has_a_builder_and_no_builder_is_undeclared(self) -> None:
        from contextmap.runtime.components import COMPONENTS

        declared = {
            (component_id, backend_id)
            for component_id, spec in COMPONENTS.items()
            for backend_id in spec.backends
        }

        assert composable_backends() == declared

    def test_every_available_stage_of_the_catalog_has_a_composer(self) -> None:
        from contextmap.runtime.components import CANONICAL_PRESET

        available = {stage.stage_id for stage in CANONICAL_PRESET.stages if stage.available}

        assert composed_stages() == available


class TestComposeExecutors:
    """Blocker #1/#2 of the PR #387 review: real ``StageExecutor``s from configuration alone.

    ``compose_executors`` is the automatic counterpart of ``compose``: it is what lets the
    installed ``contextmap`` binary run ``state_estimation``, ``geometric_mapping``,
    ``sensor_association`` and ``semantic_fusion`` with no Python caller building an
    executor by hand.
    """

    def test_composes_a_real_executor_for_every_stage_it_can_build_from_configuration(
        self, tmp_path: Path
    ) -> None:
        from contextmap.runtime.executors import (
            ContextMapExecutor,
            EntityResolutionExecutor,
            GeometricMappingExecutor,
            PointRepresentationExecutor,
            SemanticFusionExecutor,
            SemanticMappingExecutor,
            SensorAssociationExecutor,
            SpatialRelationsExecutor,
            StateEstimationExecutor,
        )

        effective = effective_from(tmp_path, selected_document())
        executors = compose_executors(
            effective,
            module_available=lambda _name: True,
            environ={},
            **executor_identities(effective, "test"),
        )

        assert set(executors) == {
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "sensor_association",
            "point_representation",
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }
        assert isinstance(executors["state_estimation"], StateEstimationExecutor)
        assert isinstance(executors["point_representation"], PointRepresentationExecutor)
        assert isinstance(executors["geometric_mapping"], GeometricMappingExecutor)
        assert isinstance(executors["sensor_association"], SensorAssociationExecutor)
        assert isinstance(executors["semantic_fusion"], SemanticFusionExecutor)
        assert isinstance(executors["semantic_mapping"], SemanticMappingExecutor)
        assert isinstance(executors["entity_resolution"], EntityResolutionExecutor)
        assert isinstance(executors["spatial_relations"], SpatialRelationsExecutor)
        assert isinstance(executors["context_map"], ContextMapExecutor)
        # O visual compõe com DINOv3/CLIP locais; SAM3 e Qwen precisam de providers próprios.
        assert "ingestion" not in executors
        assert "region_perception" not in executors
        assert "semantic_interpretation" not in executors

    def test_the_trajectory_mode_policy_is_threaded_into_state_estimation(
        self, tmp_path: Path
    ) -> None:
        """Issue #555: the ground-truth opt-in is a runtime policy, never a new
        StateEstimationRequest field -- it only decides how the executor is built."""
        from contextmap.runtime.executors import StateEstimationExecutor

        default_executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
        )
        document = selected_document()
        document["policies"] = {"trajectory_mode": "allow_ground_truth"}
        opted_in_executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
        )

        default_state_estimation = default_executors["state_estimation"]
        opted_in_state_estimation = opted_in_executors["state_estimation"]
        assert isinstance(default_state_estimation, StateEstimationExecutor)
        assert isinstance(opted_in_state_estimation, StateEstimationExecutor)
        assert default_state_estimation._allow_ground_truth_trajectory is False
        assert opted_in_state_estimation._allow_ground_truth_trajectory is True

    def test_pose_ingestion_is_never_composed_even_when_enabled(self, tmp_path: Path) -> None:
        """Issue #555: like "ingestion" itself, this stage always needs a caller-supplied
        IngestionRequest (here, a pose file path) -- never something this function can
        derive from configuration alone."""
        document = selected_document()
        document["pipeline"]["stages"]["pose_ingestion"] = True

        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
        )

        assert "pose_ingestion" not in executors

    def test_semantic_mapping_composes_only_when_map_id_and_code_digest_are_both_given(
        self, tmp_path: Path
    ) -> None:
        """Neither identity is configuration: this function never invents one (issue #177)."""
        from contextmap.runtime.executors import SemanticMappingExecutor

        failures: dict[str, CompositionError | ConfigurationError] = {}
        without_identity = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
            code_version="test",
            on_composition_failure=failures.__setitem__,
        )
        assert "semantic_mapping" not in without_identity
        failure = failures["semantic_mapping"]
        assert isinstance(failure, ExecutorIdentityMissingError)
        assert failure.missing == ("semantic_map_id", "code_digest")

        with_identity = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
            semantic_map_id=SemanticMapId("semantic-map--test"),
            code_digest="sha256:" + "cd" * 32,
            code_version="test",
        )
        assert isinstance(with_identity["semantic_mapping"], SemanticMappingExecutor)

    def test_the_quality_aware_accumulation_gets_a_real_fusion_executor(
        self, tmp_path: Path
    ) -> None:
        """#655: the selected policy runs; it is never dropped nor replaced by the baseline."""
        from contextmap.runtime.executors import SemanticFusionExecutor

        document = selected_document()
        document["components"]["semantic_fusion"]["accumulation"] = {
            "backend": "quality-aware-evidence-accumulation-v1",
            "quality-aware-evidence-accumulation-v1": {
                "definitions_version": "observation-quality-v3",
                "neutral_factor": 0.5,
                "ramps": [{"quality_input": "visible_share", "good": 1.0, "bad": 0.2}],
            },
        }

        # O run de fusão grava o código que o produziu, então só é composto com ele (#659).
        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
            code_version="commit-1",
        )

        executor = executors["semantic_fusion"]
        assert isinstance(executor, SemanticFusionExecutor)
        assert isinstance(executor._accumulation_policy, QualityAwareAccumulationPolicy)
        assert executor._accumulation_policy.neutral_factor == 0.5

    def test_without_a_code_version_the_stages_that_record_it_are_left_out_with_the_cause(
        self, tmp_path: Path
    ) -> None:
        """Issue #659: a run that records its code never records an empty one."""
        failures: dict[str, CompositionError | ConfigurationError] = {}

        executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
            on_composition_failure=failures.__setitem__,
        )

        recording = {
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
        }
        assert not recording & set(executors)
        for stage in recording:
            failure = failures[stage]
            assert isinstance(failure, ExecutorIdentityMissingError)
            assert failure.stage_id == stage and "code_version" in failure.missing
        # Os estágios cujo run não exige a identidade do código continuam compostos.
        assert {"state_estimation", "geometric_mapping", "sensor_association", "context_map"} <= (
            set(executors)
        )

    def test_an_empty_code_version_is_refused_never_recorded(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="code_version must not be empty"):
            compose_executors(
                effective_from(tmp_path, selected_document()),
                module_available=lambda _name: True,
                environ={},
                code_version=" ",
            )

    def test_the_identities_a_run_records_derive_from_its_code_identity_and_dataset(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        effective = effective_from(tmp_path, selected_document())

        identities = executor_identities(effective, "commit-1")

        assert identities == {
            "code_version": "commit-1",
            "code_digest": "sha256:" + hashlib.sha256(b"commit-1").hexdigest(),
            "semantic_map_id": "semantic-map-S1",
        }
        assert executor_identities(effective, None)["code_digest"] is None
        with pytest.raises(ValueError, match="must not be empty"):
            executor_identities(effective, "")

    def test_spatial_relations_executor_reads_geometry_summary_only_from_its_policies(
        self, tmp_path: Path
    ) -> None:
        """P2 provenance finding of the PR #540 review, second round.

        ``SpatialRelationsExecutor`` used to also accept a separate ``geometry_summary``
        keyword, independent of ``policies.geometry_summary`` -- the one persisted in the run's
        own provenance. Two independent values could diverge, making the artifact record a
        different policy than the one that actually produced its evidence. There is now only
        one place to supply it.
        """
        from contextmap.runtime.executors import SpatialRelationsExecutor

        executors = compose_executors(
            effective_from(tmp_path, selected_document()),
            module_available=lambda _name: True,
            environ={},
            code_version="test",
        )
        spatial_relations = executors["spatial_relations"]
        assert isinstance(spatial_relations, SpatialRelationsExecutor)

        # Argumento removido: a assinatura não aceita mais um segundo valor independente.
        with pytest.raises(TypeError):
            SpatialRelationsExecutor(policies=None, geometry_summary=object())  # type: ignore[call-arg,arg-type]

    def test_a_stage_whose_variation_points_are_not_selected_is_left_out_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        del document["components"]["geometric_mapping"]
        del document["components"]["sensor_association"]

        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
            code_version="test",
        )

        assert set(executors) == {
            "visual_perception",
            "state_estimation",
            "point_representation",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }

    def test_a_broken_backend_in_one_stage_never_costs_another_stage_its_executor(
        self, tmp_path: Path
    ) -> None:
        """A rejected parameter is isolated to its own stage, one at a time.

        ``sensor_association.tolerances`` rejects ``max_reprojection_invalid_rate`` above 1
        (see ``DiagnosticTolerances.__post_init__``), so composing that stage fails; the
        other stages, whose own configuration is unrelated, still compose normally.
        """
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0

        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
            code_version="test",
        )

        assert set(executors) == {
            "visual_perception",
            "state_estimation",
            "geometric_mapping",
            "point_representation",
            "semantic_fusion",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }

    def test_an_unselected_optional_channel_leaves_it_none_never_a_default_policy(
        self, tmp_path: Path
    ) -> None:
        """Every optional Entity Resolution channel and Spatial Relations predicate is absent.

        ``selected_document()`` never selects ``semantic_compatibility``, ``temporal_
        compatibility``, ``appearance``, ``representation``, ``geometric_predicate`` or
        ``contact_predicate``: the stages still compose, with the geometry-only path.
        """
        composed = _compose(tmp_path, document=selected_document())

        channels = composed.entity_comparison_channels
        assert channels is not None
        assert channels.geometry is not None
        assert channels.semantic is None
        assert channels.temporal is None
        assert channels.appearance is None
        assert channels.representation is None

        policies = composed.spatial_relations_policies
        assert policies is not None
        assert policies.frame_conventions is not None
        assert policies.candidate is not None
        assert policies.geometric is None
        assert policies.contact is None

    def test_an_incomplete_required_selection_leaves_the_stage_absent_not_fabricated(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        del document["components"]["entity_resolution"]["geometry_comparison"]

        executors = compose_executors(
            effective_from(tmp_path, document),
            module_available=lambda _name: True,
            environ={},
            code_version="test",
        )

        assert "entity_resolution" not in executors
        # spatial_relations depende de entity_resolution rio abaixo, não da sua composição:
        # a seleção incompleta de um estágio nunca contamina a composição de outro.
        assert "spatial_relations" in executors

    def test_a_composed_executor_is_the_real_thing_and_runs_against_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``StateEstimationExecutor`` is not a type-compatible stub.

        It reads a real ``SequenceArtifact`` through ``StageRequest.directory_of`` and
        writes a real ``StateEstimationRunArtifact`` where the request tells it to,
        exactly as the runtime's own DAG runner would call it.
        """
        from contextmap.ingestion import (
            ExternalPoseMeasurement,
            FrameId,
            SensorId,
            SequenceArtifactId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.shared import SourceTimestamp

        workspace = tmp_path / "ws"
        ingestion_dir = workspace / "corridor-02" / "run-0001" / "ingestion"
        with SequenceArtifactWriter(
            output_dir=ingestion_dir,
            sequence_name="corridor-02",
            artifact_id=SequenceArtifactId("sequence-0001"),
        ) as writer:
            for index in range(3):
                writer.add_observation(
                    ExternalPoseMeasurement(
                        observation_id=SourceObservationId(f"pose-{index:04d}"),
                        sensor_id=SensorId("external_pose_source"),
                        frame_id=FrameId("base"),
                        timestamp=SourceTimestamp(
                            seconds=index, nanoseconds=0, clock_id="fixture:header"
                        ),
                        provenance=SourceProvenance(
                            source_type="fixture", source_path="fixtures/poses"
                        ),
                        parent_frame=FrameId("map"),
                        translation=(float(index), 0.0, 0.0),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    )
                )
            manifest = writer.finalize()
        sequence_ref = ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(manifest.artifact_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location="corridor-02/run-0001/ingestion",
        )
        effective = effective_from(tmp_path)
        executors = compose_executors(effective, module_available=lambda _name: True, environ={})
        output_dir = workspace / "corridor-02" / "run-0001" / "state_estimation"
        request = StageRequest(
            stage_id="state_estimation",
            inputs={"sequence": (sequence_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executors["state_estimation"].execute(request)

        assert ref.contract == "StateEstimationRunArtifact"
        assert (output_dir / "manifest.json").is_file()

    def test_the_entity_resolution_executor_is_the_real_thing_and_runs_against_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``EntityResolutionExecutor`` reads a real ``SemanticMappingRunArtifact``.

        Not a fake type-compatible stand-in: ``make_entity`` builds one genuinely self-consistent
        ``Entity`` (its geometry derived through ``summarize_geometry`` over a real
        ``GeometrySource``), persisted with the capability's own ``SemanticMappingRunWriter``. The
        composed executor reads it back through ``SemanticMappingRunReader`` and writes a real
        ``EntityResolutionRunArtifact``, exactly as the runtime's own DAG runner would call it.
        """
        from runtime_entities import make_entity

        from contextmap.geometric_mapping import MapId
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.semantic_fusion import SemanticFusionRunId
        from contextmap.semantic_mapping import (
            MappingRunLineage,
            SemanticMappingRunId,
            SemanticMappingRunWriter,
        )

        workspace = tmp_path / "ws"
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        semantic_map_id = SemanticMapId("semantic-map-ci")
        entity = make_entity(semantic_map_id=semantic_map_id)
        manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=MapId("map-0001"),
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest="sha256:artifact",
                association_run_ids=("association-run-0001",),
                region_artifact_ids=("region-run-0001",),
                semantic_interpretation_artifact_ids=("semantic-run-0001",),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write([entity])
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location="corridor-02/run-0001/semantic_mapping",
        )
        effective = effective_from(tmp_path, selected_document())
        executors = compose_executors(
            effective, module_available=lambda _name: True, environ={}, code_version="test"
        )
        output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        request = StageRequest(
            stage_id="entity_resolution",
            inputs={"entities": (entities_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=output_dir,
            workspace=workspace,
        )

        ref = executors["entity_resolution"].execute(request)

        assert ref.contract == "EntityResolutionRunArtifact"
        assert (output_dir / "manifest.json").is_file()

    def test_the_spatial_relations_executor_is_the_real_thing_and_runs_against_real_artifacts(
        self, tmp_path: Path
    ) -> None:
        """The composed ``SpatialRelationsExecutor`` reads a real resolution and a real map.

        Two real upstream artifacts, not fakes: an ``EntityResolutionRunArtifact`` produced by
        running the composed ``entity_resolution`` executor itself (the same construction as
        ``test_the_entity_resolution_executor_is_the_real_thing_and_runs_against_a_real_artifact``,
        chained), and a ``GeometricMapArtifact`` written by the capability's own
        ``GeometricMapArtifactWriter`` (``_write_aligned_geometric_map``) whose eight points are,
        coordinate for coordinate, the same support ``make_entity`` already summarized for its two
        entities. That alignment is what lets ``resolved_entity_geometries`` (inside the executor)
        recompute bounds identical to what entity resolution persisted, instead of raising -- so
        candidate generation runs on genuine, map-derived positions, not on a coincidence of
        matching test doubles. The two entities are 0.1 m apart along x (well inside the
        configured ``proximity_radius_m=0.6``), so at least one real, geometry-driven candidate
        is produced; the composed executor writes a real ``SpatialRelationsRunArtifact``, exactly
        as the runtime's own DAG runner would call it.
        """
        import dataclasses

        from runtime_entities import make_entity

        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.composition import compose
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.semantic_fusion import SemanticFusionRunId
        from contextmap.semantic_mapping import (
            GEOMETRY_SUMMARY_ALGORITHM_ID,
            MappingRunLineage,
            SemanticMappingRunId,
            SemanticMappingRunWriter,
        )
        from contextmap.spatial_relations import RelationState, SpatialRelationsRunReader

        workspace = tmp_path / "ws"

        # As convenções de eixo do documento estendido declaram "odom" como o frame do mapa; o
        # fixture de geometria (tanto o real quanto o InMemoryGeometrySource de ``make_entity``)
        # está em "map". A escolha do nome do frame é um detalhe de configuração deste teste, não
        # uma invariante do domínio -- então o documento local é ajustado para concordar com a
        # geometria real, em vez de forçar a geometria a se chamar "odom".
        document = selected_document()
        document["components"]["spatial_relations"]["frame_conventions"][
            "map-frame-conventions-v1"
        ]["map_frame"] = "map"
        effective = effective_from(tmp_path, document)
        executors = compose_executors(
            effective, module_available=lambda _name: True, environ={}, code_version="test"
        )

        geometry_dir = workspace / "corridor-02" / "run-0001" / "geometric_mapping"
        geometry_manifest = _write_aligned_geometric_map(geometry_dir)
        geometry_ref = ArtifactRef(
            stage_id="geometric_mapping",
            contract="GeometricMapArtifact",
            artifact_id=str(geometry_manifest.run_id),
            content_hash=inventory_digest(geometry_manifest.file_inventory),
            location=geometry_dir.relative_to(workspace).as_posix(),
        )

        # Duas entidades reais sobre o mesmo mapa, com suportes de geometria disjuntos (índices
        # 0-3 e 4-7) e vizinhos: 0.1 m de vão no eixo x, dentro do proximity_radius_m=0.6
        # configurado, então a geração de candidatos tem um par real para medir.
        semantic_map_id = SemanticMapId("semantic-map-ci")
        entities = [
            make_entity(
                "entity--support-000001",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(0, 1, 2, 3),
            ),
            make_entity(
                "entity--support-000002",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(4, 5, 6, 7),
            ),
        ]
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        mapping_manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=geometry_manifest.map_id,
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest="sha256:artifact",
                association_run_ids=("association-run-0001",),
                region_artifact_ids=("region-run-0001",),
                semantic_interpretation_artifact_ids=("semantic-run-0001",),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write(entities)
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(mapping_manifest.run_id),
            content_hash=inventory_digest(mapping_manifest.file_inventory),
            location=mapping_dir.relative_to(workspace).as_posix(),
        )

        # Passo 1: entity_resolution real, encadeado, exatamente como o teste irmão o constrói --
        # a forma mais simples de obter um EntityResolutionRunArtifact genuíno para este teste.
        resolution_output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        resolution_request = StageRequest(
            stage_id="entity_resolution",
            inputs={"entities": (entities_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=resolution_output_dir,
            workspace=workspace,
        )
        resolution_ref = executors["entity_resolution"].execute(resolution_request)
        assert resolution_ref.contract == "EntityResolutionRunArtifact"

        # Passo 2: spatial_relations real, sobre a resolução e o mapa geométrico reais.
        relations_output_dir = workspace / "corridor-02" / "run-0001" / "spatial_relations"
        relations_request = StageRequest(
            stage_id="spatial_relations",
            inputs={"entities": (resolution_ref,), "geometry": (geometry_ref,)},
            components={},
            config_digest=effective.digest,
            output_dir=relations_output_dir,
            workspace=workspace,
        )

        ref = executors["spatial_relations"].execute(relations_request)

        assert ref.contract == "SpatialRelationsRunArtifact"
        assert (relations_output_dir / "manifest.json").is_file()

        # A geometria real e as duas entidades vizinhas produziram trabalho genuíno: pelo menos
        # um candidato de relação, ainda sem estado decidido (nenhum avaliador geométrico ou de
        # contato foi selecionado no documento estendido) -- honestamente UNRESOLVED, nunca uma
        # relação fabricada.
        reader = SpatialRelationsRunReader(relations_output_dir)
        relations = tuple(reader.iter_relations())
        assert len(relations) >= 1
        assert all(relation.state is RelationState.UNRESOLVED for relation in relations)

        # Reabre o manifesto e confere a proveniência de ``geometry_summary`` de ponta a ponta:
        # a política persistida é a mesma que a composição de fato construiu a partir do
        # documento estendido, não uma segunda instância que só coincide por acaso.
        composed = compose(
            effective,
            stages=["spatial_relations"],
            module_available=lambda _name: True,
            environ={},
        )
        assert composed.spatial_relations_policies is not None
        expected_policy = composed.spatial_relations_policies.geometry_summary
        assert reader.manifest.policies["geometry_summary"] == {
            "policy_id": GEOMETRY_SUMMARY_ALGORITHM_ID,
            "fingerprint": expected_policy.fingerprint(),
            "parameters": dataclasses.asdict(expected_policy),
        }

    def test_the_context_map_executor_is_the_real_thing_and_assembles_a_real_artifact(
        self, tmp_path: Path
    ) -> None:
        """The composed ``ContextMapExecutor`` assembles a real ``ContextMap`` (issue #177).

        Chains the same real entity_resolution -> spatial_relations construction as
        ``test_the_spatial_relations_executor_is_the_real_thing_and_runs_against_real_artifacts``,
        one hop further, feeding the composed executor the real geometric map, the real
        spatial-relations run it just produced, and a real, minimal sequence artifact whose
        identity matches the one the geometric map was actually built over (issue #438 review:
        ``ContextMapExecutor`` now opens and verifies this, so a stand-in with an unrelated
        identity is no longer accepted).
        """
        import json

        from runtime_entities import make_entity

        from contextmap.ingestion import (
            FrameId,
            ImuObservation,
            SensorId,
            SequenceArtifactWriter,
            SourceObservationId,
            SourceProvenance,
        )
        from contextmap.ingestion.sequence_provenance import SequenceProvenance
        from contextmap.runtime import ArtifactRef
        from contextmap.runtime.executors import inventory_digest
        from contextmap.runtime.pipeline import StageRequest
        from contextmap.semantic_fusion import SemanticFusionRunId
        from contextmap.semantic_mapping import (
            MappingRunLineage,
            SemanticMappingRunId,
            SemanticMappingRunWriter,
        )
        from contextmap.shared import SourceTimestamp

        workspace = tmp_path / "ws"
        document = selected_document()
        document["components"]["spatial_relations"]["frame_conventions"][
            "map-frame-conventions-v1"
        ]["map_frame"] = "map"
        effective = effective_from(tmp_path, document)
        executors = compose_executors(
            effective, module_available=lambda _name: True, environ={}, code_version="test"
        )
        assert "context_map" in executors

        geometry_dir = workspace / "corridor-02" / "run-0001" / "geometric_mapping"
        geometry_manifest = _write_aligned_geometric_map(geometry_dir)
        geometry_ref = ArtifactRef(
            stage_id="geometric_mapping",
            contract="GeometricMapArtifact",
            artifact_id=str(geometry_manifest.map_id),
            content_hash=inventory_digest(geometry_manifest.file_inventory),
            location=geometry_dir.relative_to(workspace).as_posix(),
        )

        semantic_map_id = SemanticMapId("semantic-map-ci")
        fusion_artifact_digest = "sha256:" + "ab" * 32
        entities = [
            make_entity(
                "entity--support-000001",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(0, 1, 2, 3),
                fusion_artifact_digest=fusion_artifact_digest,
            ),
            make_entity(
                "entity--support-000002",
                semantic_map_id=semantic_map_id,
                map_id=geometry_manifest.map_id,
                indexes=(4, 5, 6, 7),
                fusion_artifact_digest=fusion_artifact_digest,
            ),
        ]
        mapping_dir = workspace / "corridor-02" / "run-0001" / "semantic_mapping"
        mapping_manifest = SemanticMappingRunWriter(
            output_dir=mapping_dir,
            sequence_name="corridor-02",
            run_id=SemanticMappingRunId("run-0001--semantic-mapping"),
            run_index=1,
            semantic_map_id=semantic_map_id,
            lineage=MappingRunLineage(
                sequence_artifact_id="sequence-0001",
                geometric_map_id=geometry_manifest.map_id,
                fusion_run_id=SemanticFusionRunId("fusion-run-0001"),
                fusion_schema_version="0.1.0",
                fusion_artifact_digest=fusion_artifact_digest,
                association_run_ids=("association-run-0001",),
                region_artifact_ids=("region-run-0001",),
                semantic_interpretation_artifact_ids=("semantic-run-0001",),
                point_representation_run_ids=(),
            ),
            code_version="test",
            code_digest="sha256:" + "cd" * 32,
        ).write(entities)
        entities_ref = ArtifactRef(
            stage_id="semantic_mapping",
            contract="SemanticEntityArtifact",
            artifact_id=str(mapping_manifest.run_id),
            content_hash=inventory_digest(mapping_manifest.file_inventory),
            location=mapping_dir.relative_to(workspace).as_posix(),
        )

        resolution_output_dir = workspace / "corridor-02" / "run-0001" / "entity_resolution"
        resolution_ref = executors["entity_resolution"].execute(
            StageRequest(
                stage_id="entity_resolution",
                inputs={"entities": (entities_ref,)},
                components={},
                config_digest=effective.digest,
                output_dir=resolution_output_dir,
                workspace=workspace,
            )
        )

        relations_output_dir = workspace / "corridor-02" / "run-0001" / "spatial_relations"
        relations_ref = executors["spatial_relations"].execute(
            StageRequest(
                stage_id="spatial_relations",
                inputs={"entities": (resolution_ref,), "geometry": (geometry_ref,)},
                components={},
                config_digest=effective.digest,
                output_dir=relations_output_dir,
                workspace=workspace,
            )
        )
        # Sequence real, mínima: ContextMapExecutor agora abre este artifact e exige que sua
        # identidade seja exatamente a que o mapa geométrico foi construído sobre
        # (``geometry_manifest.sequence_artifact_id``, "aligned-sequence" -- ver
        # ``_write_aligned_geometric_map``), não um stand-in com identidade arbitrária.
        sequence_dir = workspace / "corridor-02" / "sequence"
        with SequenceArtifactWriter(
            output_dir=sequence_dir,
            sequence_name="corridor-02",
            artifact_id=geometry_manifest.sequence_artifact_id,
        ) as sequence_writer:
            sequence_writer.add_observation(
                ImuObservation(
                    observation_id=SourceObservationId("imu-0000"),
                    sensor_id=SensorId("imu"),
                    frame_id=FrameId("imu"),
                    timestamp=SourceTimestamp(
                        seconds=0, nanoseconds=0, clock_id="fixture:aligned-map"
                    ),
                    provenance=SourceProvenance(source_type="fixture", source_path="fixtures/imu"),
                )
            )
            sequence_writer.set_provenance(
                SequenceProvenance(source_type="fixture", source_path="fixtures/imu")
            )
            sequence_manifest = sequence_writer.finalize()
        sequence_ref = ArtifactRef(
            stage_id="ingestion",
            contract="SequenceArtifact",
            artifact_id=str(sequence_manifest.artifact_id),
            content_hash=inventory_digest(sequence_manifest.file_inventory),
            location=sequence_dir.relative_to(workspace).as_posix(),
        )

        context_map_output_dir = workspace / "corridor-02" / "run-0001" / "context_map"
        context_map_ref = executors["context_map"].execute(
            StageRequest(
                stage_id="context_map",
                inputs={
                    "sequence": (sequence_ref,),
                    "geometry": (geometry_ref,),
                    "entities": (resolution_ref,),
                    "relations": (relations_ref,),
                },
                components={},
                config_digest=effective.digest,
                output_dir=context_map_output_dir,
                workspace=workspace,
            )
        )

        assert context_map_ref.contract == "ContextMapArtifact"
        manifest_record = json.loads(
            (context_map_output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_record["entity_count"] == 2


class TestCompositionFailuresReachPreflight:
    """Issue #602 (RT-02): why a stage could not be composed reaches the preflight report.

    ``compose_executors`` still leaves such a stage out and never raises; the cause it used to
    swallow is now handed to its caller, and ``preflight`` reports it at the component it
    failed on instead of only saying that no executor is registered.
    """

    def _preflight(
        self,
        tmp_path: Path,
        document: dict[str, Any],
        targets: list[str],
        providers: dict[str, RuntimeProvider] | None = None,
    ) -> PreflightReport:
        effective = effective_from(tmp_path, document)
        failures: dict[str, CompositionError | ConfigurationError] = {}
        executors = compose_executors(
            effective,
            providers=providers,
            module_available=lambda _name: True,
            environ={},
            on_composition_failure=failures.__setitem__,
        )
        return preflight(
            resolve_plan(effective).scope(targets=targets),
            executors=executors,
            environ={},
            module_available=lambda _name: True,
            composition_failures=failures,
        )

    def test_a_rejected_backend_parameter_is_reported_at_its_component(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0

        report = self._preflight(tmp_path, document, ["sensor_association"])

        paths = [problem.path for problem in report.problems]
        assert "components.sensor_association.tolerances" in paths
        cause = report.problems[paths.index("components.sensor_association.tolerances")]
        assert "sensor_association" in cause.message
        assert "max_reprojection_invalid_rate" in cause.message
        assert "stages.sensor_association" not in paths
        # ingestion nunca é composto aqui e não tem causa capturada: o problema genérico fica.
        generic = ConfigProblem(path="stages.ingestion", message="no executor is registered for it")
        assert generic in report.problems

    def test_an_invalid_provider_target_is_reported_with_the_target_and_the_reason(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["resources"]["providers"] = {REGION: "no-attribute-here"}

        report = self._preflight(
            tmp_path,
            document,
            ["visual_perception"],
            providers={INTERPRETER: _Recorder().provider("qwen")},
        )

        causes = [p for p in report.problems if p.path == f"components.{REGION}"]
        assert len(causes) == 1
        assert "'no-attribute-here'" in causes[0].message
        assert "must look like 'module:attribute'" in causes[0].message
        assert "region_perception" in causes[0].message
        # O motivo real é relatado uma única vez, no componente que falhou, e não se repete como
        # problema genérico do estágio dele.
        paths = [p.path for p in report.problems]
        assert "stages.region_perception" not in paths
        # A percepção não depende mais dos backends da descoberta: ela consome as regiões como
        # entrada, então compõe o seu executor e não é bloqueada por este provider
        # (``docs/perception-dag-decomposition.md``).
        assert "stages.visual_perception" not in paths

    def test_a_candidate_reach_short_of_an_evaluator_tolerance_is_reported_at_the_candidate(
        self, tmp_path: Path
    ) -> None:
        # Issue #606: cada política é válida sozinha, mas o alcance de proximidade dos
        # candidatos (0.6 m) não cobre o next_to_max_gap_m do avaliador geométrico.
        document = selected_document()
        document["components"]["spatial_relations"]["geometric_predicate"] = {
            "backend": "bounds-geometric-predicates-v1",
            "bounds-geometric-predicates-v1": {
                "boundary_tolerance_m": 0.02,
                "next_to_max_gap_m": 0.9,
                "adjacent_penetration_m": 0.05,
                "containment_slack_m": 0.05,
                "directional_overlap_fraction": 0.5,
            },
        }

        report = self._preflight(tmp_path, document, ["spatial_relations"])

        paths = [problem.path for problem in report.problems]
        assert "components.spatial_relations.candidate" in paths
        cause = report.problems[paths.index("components.spatial_relations.candidate")]
        assert "next_to_max_gap_m=0.9 exceeds proximity_radius_m=0.6" in cause.message
        assert "stages.spatial_relations" not in paths

    def test_a_missing_model_runtime_is_reported_at_its_component(self, tmp_path: Path) -> None:
        report = self._preflight(tmp_path, selected_document(), ["visual_perception"])

        causes = [p for p in report.problems if p.path == f"components.{REGION}"]
        assert len(causes) == 1
        assert "has no bundled model loader" in causes[0].message

    def test_a_missing_code_identity_is_reported_at_the_stage_that_records_it(
        self, tmp_path: Path
    ) -> None:
        report = self._preflight(tmp_path, selected_document(), ["semantic_fusion"])

        causes = [p for p in report.problems if p.path == "stages.semantic_fusion"]
        assert len(causes) == 1
        assert "records code_version" in causes[0].message

    def test_a_cause_preflight_already_reports_is_not_repeated(self, tmp_path: Path) -> None:
        # Seleção incompleta: o próprio preflight já aponta o componente; o estágio segue
        # reportado como sem executor, sem duplicar a causa.
        document = selected_document()
        del document["components"]["entity_resolution"]["geometry_comparison"]

        report = self._preflight(tmp_path, document, ["entity_resolution"])

        paths = [problem.path for problem in report.problems]
        assert paths.count("components.entity_resolution.geometry_comparison") == 1
        assert "stages.entity_resolution" in paths

    def test_the_cause_is_handed_over_without_costing_another_stage_its_executor(
        self, tmp_path: Path
    ) -> None:
        document = selected_document()
        document["components"]["sensor_association"]["tolerances"]["diagnostic-tolerances-v1"][
            "max_reprojection_invalid_rate"
        ] = 2.0
        failures: dict[str, CompositionError | ConfigurationError] = {}

        effective = effective_from(tmp_path, document)
        executors = compose_executors(
            effective,
            providers=_providers(_Recorder()),
            module_available=lambda _name: True,
            environ={},
            on_composition_failure=failures.__setitem__,
            **executor_identities(effective, "test"),
        )

        assert set(failures) == {"sensor_association"}
        failure = failures["sensor_association"]
        assert isinstance(failure, BackendConfigurationError)
        assert failure.component_id == "sensor_association.tolerances"
        assert set(executors) == {
            "region_perception",
            "visual_perception",
            "semantic_interpretation",
            "state_estimation",
            "geometric_mapping",
            "point_representation",
            "semantic_fusion",
            "semantic_mapping",
            "entity_resolution",
            "spatial_relations",
            "context_map",
        }


def _sequence_artifact(workspace: Path, *, width: int, height: int, frames: int = 1) -> Any:
    """Publish a ``bgr8`` sequence and reference it.

    The frames are ``frame-0000``, ``frame-0001``, ... one second apart.
    """
    from contextmap.ingestion import (
        FrameId,
        ImageEncoding,
        ImageObservation,
        SensorId,
        SequenceArtifactId,
        SequenceArtifactWriter,
        SourceObservationId,
        SourceProvenance,
    )
    from contextmap.runtime import ArtifactRef
    from contextmap.runtime.executors import inventory_digest
    from contextmap.shared import SourceTimestamp

    with SequenceArtifactWriter(
        output_dir=workspace / "corridor-02" / "run-0001" / "ingestion",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    ) as writer:
        for index in range(frames):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index:04d}"),
                    sensor_id=SensorId("camera_1"),
                    frame_id=FrameId("camera_1_optical"),
                    timestamp=SourceTimestamp(
                        seconds=index, nanoseconds=0, clock_id="fixture:header"
                    ),
                    provenance=SourceProvenance(
                        source_type="fixture", source_path="fixtures/images"
                    ),
                    width=width,
                    height=height,
                    encoding=ImageEncoding.BGR8,
                    data=bytes([10, 20, 30] * (width * height)),
                )
            )
        manifest = writer.finalize()
    sequence_ref = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=str(manifest.artifact_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location="corridor-02/run-0001/ingestion",
    )
    return sequence_ref


def _region_request(workspace: Path, *, width: int, height: int, frames: int = 1) -> Any:
    """Publish a sequence and build the ``region_perception`` request over it."""
    from contextmap.runtime.pipeline import StageRequest

    sequence = _sequence_artifact(workspace, width=width, height=height, frames=frames)
    return StageRequest(
        stage_id="region_perception",
        inputs={"sequence": (sequence,)},
        components={},
        config_digest="sha256:test",
        output_dir=workspace / "corridor-02" / "run-0001" / "region_perception",
        workspace=workspace,
    )


def _perception_request(
    workspace: Path, *, width: int, height: int, frames: int = 1, discovery: Any = None
) -> Any:
    """Publish a sequence, discover its regions, and build the ``visual_perception`` request.

    The regions come from a real ``region_perception`` run over the same sequence, because
    perception replays that artifact instead of discovering anything itself.

    Args:
        workspace: The run workspace the requests resolve locations against.
        width: Image width of every frame.
        height: Image height of every frame.
        frames: How many frames the sequence holds.
        discovery: The region discovery backend of the region run; by default the one-region
            fake of ``runtime_perception``.
    """
    from contextmap.runtime.pipeline import StageRequest

    sequence_ref = _sequence_artifact(workspace, width=width, height=height, frames=frames)
    regions_ref = publish_region_run(workspace, sequence_ref, discovery=discovery)
    return StageRequest(
        stage_id="visual_perception",
        inputs={"sequence": (sequence_ref,), "regions": (regions_ref,)},
        components={},
        config_digest="sha256:test",
        output_dir=workspace / "corridor-02" / "run-0001" / "visual_perception",
        workspace=workspace,
    )


def test_composition_does_not_load_a_model_or_import_a_heavy_sdk(tmp_path: Path) -> None:
    import sys

    before = {name for name in ("torch", "transformers", "rosbags") if name in sys.modules}

    _compose(tmp_path)

    after = {name for name in ("torch", "transformers") if name in sys.modules}
    assert after <= before
