"""The optional semantic scoring stage: CLIP/AlphaCLIP scorers composed by the runtime (#652).

Scoring runs downstream of a pinned ``PerceptionRunArtifact`` (D1 = b): every scorer arm scores
exactly the claims and region features of the same perception run, which the runtime reuses
by content instead of re-running its non-deterministic VLM (#580). The text towers are fakes
supplied as model runtimes; the perception run, the scoring run, the composition and the DAG
execution are real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from runtime_documents import effective_from, selected_document

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    Region2D,
    RegionId,
    RegionPerceptionRunReader,
)
from contextmap.region_perception.artifacts.run_manifest import RegionPerceptionRunManifest
from contextmap.region_perception.artifacts.run_writer import RegionPerceptionRunWriter
from contextmap.runtime import (
    ArtifactRef,
    FileArtifactStore,
    ReusePolicy,
    RunJournal,
    StageRequest,
    resolve_plan,
    run_plan,
)
from contextmap.runtime.composition import (
    RuntimeProvider,
    compose,
    compose_executors,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
)
from contextmap.runtime.executors import inventory_digest
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SceneContext,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
)
from contextmap.semantic_scoring import SemanticScoreType, SemanticScoringRunReader
from contextmap.semantic_scoring.adapters.alphaclip_text import (
    AlphaClipTextConfig,
    alphaclip_region_scorer_provenance,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    PerceptionResult,
    PreparedImage,
    VisualEvidenceResult,
    VisualEvidenceRunId,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
)
from contextmap.visual_perception.visual_features import (
    EmbeddingSpace,
    FeatureId,
    FeatureScope,
    VisualFeature,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.visual_features.adapters.alphaclip import (
    alphaclip_embedding_space,
)
from contextmap.visual_perception.visual_features.adapters.clip import clip_embedding_space

SCORING = "semantic_scoring.scorer"
FINGERPRINT = "sha256:" + "a" * 64
ALPHACLIP: dict[str, Any] = {
    "model_name": "ViT-L/14",
    "base_checkpoint_path": "base.pt",
    "alpha_checkpoint_path": "alpha.pth",
    "checkpoint_fingerprint": FINGERPRINT,
    "text_template": "{hypothesis}",
}
ALPHACLIP_SPACE = alphaclip_embedding_space(
    model_name="ViT-L/14",
    code_version="1",
    checkpoint_fingerprint=FINGERPRINT,
    dimension=2,
    normalization="l2",
)
OBSERVATION = SourceObservationId("frame-0000")
VISUAL_RUN = VisualEvidenceRunId("perception-1")
TEXT_ROWS = {
    "chair": [0.6, 0.8],
    "table": [0.8, -0.6],
    "door": [1.0, 0.0],
    "a photo of a chair": [0.8, 0.6],
    "a photo of a table": [0.6, -0.8],
    "a photo of a door": [0.0, 1.0],
}


def _document(scorer: dict[str, Any] | None) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    if scorer is not None:
        document["pipeline"]["stages"]["semantic_scoring"] = True
        # O scoring é capability própria desde a decomposição, e o documento base não a
        # declara: quem habilita o estágio é que traz o bloco do scorer.
        document["components"].setdefault("semantic_scoring", {})["scorer"] = scorer
    return document


def _alphaclip(**changes: Any) -> dict[str, Any]:
    return {"backend": "alphaclip", "alphaclip": {**ALPHACLIP, **changes}}


class _TextTower:
    """A fake text tower: one fixed row per rendered text; it counts every use."""

    def __init__(self) -> None:
        self.uses = 0

    def projection_dimension(self) -> int:
        self.uses += 1
        return 2

    def encode(self, texts: Any) -> np.ndarray[Any, Any]:
        self.uses += 1
        return np.asarray([TEXT_ROWS[text] for text in texts], dtype=np.float32)


class _Provider:
    """Supply the fake text tower and record every configuration it was asked for."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.towers: list[_TextTower] = []

    def __call__(self, config: Any, secrets: Any) -> _TextTower:
        self.calls.append(config)
        self.towers.append(_TextTower())
        return self.towers[-1]


def _placeholder_runtime(config: Any, secrets: Any) -> object:
    return object()


def _providers(provider: RuntimeProvider | None = None) -> dict[str, RuntimeProvider]:
    """The scorer's text tower, and placeholders for the perception backends without a loader."""
    return {
        SCORING: provider or _Provider(),
        "region_perception.discovery": _placeholder_runtime,
        "semantic_interpretation.interpreter": _placeholder_runtime,
    }


def _ready(_name: str) -> bool:
    return True


# ----------------------------------------------------------------- a pinned perception run

_ANY = BackendProvenance(backend_id="fake", capability="c", provider="f", model="f", version="1")


def _claim(result_id: Any, name: str, region: str | None, role: HypothesisRole) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(f"claim-{name}"),
        source_observation_id=OBSERVATION,
        semantic_interpretation_run_id=result_id,
        hypothesis=name,
        role=role,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake_vlm",
                capability="semantic_interpreter",
                provider="f",
                model="f",
                version="1",
            ),
            task_identity="region",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=None if region is None else RegionId(region),
    )


def _perception_result(space: EmbeddingSpace, run_id: VisualEvidenceRunId) -> PerceptionResult:
    """One frame: a region with its region feature and two hypotheses, one region without."""
    result_id = SemanticInterpretationRunId("semantic-1")
    scene = _claim(result_id, "corridor", None, HypothesisRole.PRIMARY)
    return PerceptionResult(
        source_observation_id=OBSERVATION,
        region_artifact_id="regions-1",
        visual_evidence_artifact_id=str(run_id),
        semantic_interpretation_artifact_id=str(result_id),
        sequence_artifact_id="S1-sequence",
        regions=tuple(
            Region2D(
                source_observation_id=OBSERVATION,
                region_id=RegionId(region),
                bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
                provenance=replace_capability("region_discovery"),
            )
            for region in ("region-0001", "region-0002")
        ),
        features=(
            VisualFeature(
                source_observation_id=OBSERVATION,
                feature_id=FeatureId("feature-region-0001"),
                scope=FeatureScope.REGION,
                embedding_space_id=embedding_space_fingerprint(space),
                shape=(2,),
                dtype="float32",
                payload_reference="region/feature-region-0001.npy",
                provenance=replace_capability("feature_extractor"),
                region_id=RegionId("region-0001"),
                normalization="l2",
            ),
        ),
        claims=(
            _claim(result_id, "chair", "region-0001", HypothesisRole.PRIMARY),
            _claim(result_id, "table", "region-0001", HypothesisRole.ALTERNATIVE),
            _claim(result_id, "door", "region-0002", HypothesisRole.PRIMARY),
        ),
        scene_context=SceneContext(
            context_id="scene-context-1",
            source_observation_id=OBSERVATION,
            semantic_interpretation_run_id=result_id,
            provenance=scene.provenance,
            claims=(scene,),
        ),
    )


def replace_capability(capability: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id="fake", capability=capability, provider="f", model="f", version="1"
    )


def _write_perception(
    output: Path,
    *,
    space: EmbeddingSpace,
    run_id: VisualEvidenceRunId = VISUAL_RUN,
    region_id: str = "regions-1",
) -> Any:
    result = _perception_result(space, run_id)
    with VisualEvidenceRunWriter(
        output_dir=output,
        sequence_name="S1",
        run_id=run_id,
        run_index=1,
        sequence_artifact_id="S1-sequence",
        region_artifact_id=region_id,
        selection_id="sha256:selection",
        configuration_digest="sha256:test",
    ) as writer:
        writer.add_result(
            VisualEvidenceResult(source_observation_id=OBSERVATION, features=tuple(result.features))
        )
        writer.add_feature_payload(
            result.features[0], OBSERVATION, np.asarray([0.6, 0.8], dtype=np.float32)
        )
        return writer.finalize()


def _write_regions(output: Path, *, run_id: str = "regions-1") -> Any:
    with RegionPerceptionRunWriter(output) as writer:
        writer.write_manifest(
            RegionPerceptionRunManifest(
                run_id=run_id,
                run_index=1,
                sequence_name="S1",
                sequence_artifact_id="S1-sequence",
                source_observation_ids=(str(OBSERVATION),),
                selection_id="sha256:selection",
                created_at="2026-01-01T00:00:00Z",
                prepared_image_id="prepared",
                discovery_backend_id="fixture",
                discovery_model="fixture",
                discovery_version="1",
                discovery_configuration_fingerprint="sha256:fixture",
                normalization_policy_fingerprint="sha256:fixture",
            )
        )
        for region in _perception_result(ALPHACLIP_SPACE, VISUAL_RUN).regions:
            writer.write_region(region)
        writer.write_frame(
            PreparedImage(
                source_observation_id=OBSERVATION,
                payload_reference="source-observation:frame-0000",
                width=2,
                height=2,
            )
        )
        return writer.finalize()


def _write_semantics(
    output: Path, *, region_id: str = "regions-1", run_id: str = "semantic-1"
) -> Any:
    from semantic_artifact_builders import write_semantic_views

    result = _perception_result(ALPHACLIP_SPACE, VISUAL_RUN)
    with SemanticInterpretationRunWriter(
        output_dir=output,
        run_id=SemanticInterpretationRunId(run_id),
        sequence_name="S1",
        run_index=1,
        sequence_artifact_id="S1-sequence",
        region_artifact_id=region_id,
        selection_id="sha256:selection",
        configuration_digest="sha256:fixture",
    ) as writer:
        write_semantic_views(writer, [result], run_id)
        return writer.finalize()


def _fixture_ref(workspace: Path, stage: str, contract: str, manifest: Any) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage,
        contract=contract,
        artifact_id=str(manifest.run_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location=f"S1/run-0001/{stage}",
    )


def _pinned_perception(workspace: Path, *, space: EmbeddingSpace = ALPHACLIP_SPACE) -> ArtifactRef:
    location = "S1/run-0001/visual_perception"
    _write_regions(workspace / "S1/run-0001/region_perception")
    _write_semantics(workspace / "S1/run-0001/semantic_interpretation")
    manifest = _write_perception(workspace / location, space=space)
    return ArtifactRef(
        stage_id="visual_perception",
        contract="VisualEvidenceArtifact",
        artifact_id=str(manifest.run_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location=location,
    )


def _score(executor: Any, workspace: Path, perception: ArtifactRef) -> ArtifactRef:
    ref: ArtifactRef = executor.execute(
        StageRequest(
            stage_id="semantic_scoring",
            inputs={
                "regions": (
                    _fixture_ref(
                        workspace,
                        "region_perception",
                        "RegionPerceptionRunArtifact",
                        RegionPerceptionRunReader(
                            workspace / "S1/run-0001/region_perception"
                        ).manifest,
                    ),
                ),
                "visual_evidence": (perception,),
                "semantic_interpretation": (
                    _fixture_ref(
                        workspace,
                        "semantic_interpretation",
                        "SemanticInterpretationArtifact",
                        SemanticInterpretationRunReader(
                            workspace / "S1/run-0001/semantic_interpretation"
                        ).manifest,
                    ),
                ),
            },
            components={},
            config_digest="sha256:test",
            output_dir=workspace / "S1" / "run-0002" / "semantic_scoring",
            workspace=workspace,
        )
    )
    return ref


# ------------------------------------------------------------------------- composition


class TestComposition:
    def test_a_configuration_without_the_scorer_resolves_and_composes_exactly_as_today(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "off").mkdir()
        (tmp_path / "on").mkdir()
        off = effective_from(tmp_path / "off", _document(None))
        on = effective_from(tmp_path / "on", _document(_alphaclip()))

        assert off.config.pipeline.stages["semantic_scoring"] is False
        assert SCORING not in off.config.components
        assert "semantic_scoring" not in resolve_plan(off).order  # type: ignore[operator]
        # O scorer é um estágio próprio: selecioná-lo nunca muda a identidade da percepção.
        assert (
            resolve_plan(on).stage("visual_perception").config_digest
            == resolve_plan(off).stage("visual_perception").config_digest
        )
        composed = compose(off, providers=_providers(), module_available=_ready, environ={})
        assert composed.semantic_scoring is None
        assert "semantic_scoring" not in composed.stages
        failures: dict[str, Any] = {}
        executors = compose_executors(
            off,
            providers=_providers(),
            module_available=_ready,
            environ={},
            on_composition_failure=failures.__setitem__,
        )
        assert "semantic_scoring" not in executors
        # Um estágio opcional desligado não é uma falha de composição.
        assert "semantic_scoring" not in failures

    def test_selecting_the_alphaclip_scorer_composes_it_without_loading_a_model(
        self, tmp_path: Path
    ) -> None:
        provider = _Provider()
        effective = effective_from(tmp_path, _document(_alphaclip()))

        composed = compose(
            effective,
            providers=_providers(provider),
            stages=["semantic_scoring"],
            module_available=_ready,
            environ={},
        )

        plan = composed.semantic_scoring
        assert plan is not None
        assert plan.backend == alphaclip_region_scorer_provenance(
            AlphaClipTextConfig(**ALPHACLIP, device="cpu")
        )
        assert plan.text_template == "{hypothesis}"
        # A torre de texto é pedida uma vez ao chamador, e nada roda antes de o estágio rodar.
        assert provider.calls == [AlphaClipTextConfig(**ALPHACLIP, device="cpu")]
        assert [tower.uses for tower in provider.towers] == [0]

    def test_the_alphaclip_scorer_needs_a_caller_supplied_text_tower(self, tmp_path: Path) -> None:
        effective = effective_from(tmp_path, _document(_alphaclip()))

        with pytest.raises(BackendRuntimeMissingError, match="semantic_scoring"):
            compose(
                effective,
                providers={},
                stages=["semantic_scoring"],
                module_available=_ready,
                environ={},
            )

    def test_the_clip_scorer_bundles_its_loader_and_names_its_missing_modules(
        self, tmp_path: Path
    ) -> None:
        clip = {
            "backend": "clip",
            "clip": {
                "checkpoint": "openai/clip-x",
                "revision": "b" * 40,
                "text_template": "a photo of a {hypothesis}",
            },
        }
        effective = effective_from(tmp_path, _document(clip))

        composed = compose(
            effective, providers={}, stages=["semantic_scoring"], module_available=_ready
        )
        assert composed.semantic_scoring is not None
        assert composed.semantic_scoring.backend.backend_id == "clip_region_scorer"
        with pytest.raises(BackendUnavailableError, match="transformers"):
            compose(
                effective,
                providers={},
                stages=["semantic_scoring"],
                module_available=lambda name: name != "transformers",
            )

    def test_an_unusable_text_template_is_refused_at_composition(self, tmp_path: Path) -> None:
        effective = effective_from(tmp_path, _document(_alphaclip(text_template="a photo")))

        with pytest.raises(BackendConfigurationError, match="text_template"):
            compose(
                effective,
                providers=_providers(),
                stages=["semantic_scoring"],
                module_available=_ready,
                environ={},
            )


# ---------------------------------------------------------------------------- execution


class TestExecution:
    def _executor(self, tmp_path: Path, scorer: dict[str, Any]) -> Any:
        effective = effective_from(tmp_path, _document(scorer))
        executors = compose_executors(
            effective, providers=_providers(), module_available=_ready, environ={}
        )
        return executors["semantic_scoring"]

    def test_each_region_hypothesis_with_a_compatible_feature_gets_exactly_one_score(
        self, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "ws"
        perception = _pinned_perception(workspace)
        claims_before = (
            SemanticInterpretationRunReader(workspace / "S1/run-0001/semantic_interpretation")
            .list_observations()[0]
            .claims
        )

        ref = _score(self._executor(tmp_path, _alphaclip()), workspace, perception)

        assert ref.contract == "SemanticScoringRunArtifact"
        reader = SemanticScoringRunReader(workspace / str(ref.location))
        assert reader.verify_integrity() == []
        scores = reader.list_scores()
        labels = {str(claim.claim_id): claim.hypothesis for claim in claims_before}
        assert sorted(labels[str(score.claim_id)] for score in scores) == ["chair", "table"]
        by_claim = {labels[str(score.claim_id)]: score for score in scores}
        assert by_claim["chair"].value == pytest.approx(1.0)
        assert by_claim["table"].value == pytest.approx(0.0, abs=1e-6)
        assert all(score.score_type is SemanticScoreType.COSINE_SIMILARITY for score in scores)
        assert all(score.calibrated_probability is None for score in scores)
        assert {str(score.feature_id) for score in scores} == {"feature-region-0001"}
        # A região sem feature e a claim de cena ficam sem score: nenhum zero fabricado.
        counts = reader.manifest.counts
        assert counts is not None
        assert (counts.unscored_region_claim_count, counts.scene_claim_count) == (1, 1)
        assert reader.manifest.lineage.visual_evidence_artifact_id == str(VISUAL_RUN)
        assert reader.manifest.embedding_space == ALPHACLIP_SPACE
        # Nenhuma claim é alterada: o run de percepção continua byte a byte o mesmo.
        after = VisualEvidenceRunReader(workspace / str(perception.location))
        assert after.verify_integrity() == []
        assert inventory_digest(after.manifest.file_inventory) == perception.content_hash
        assert (
            SemanticInterpretationRunReader(workspace / "S1/run-0001/semantic_interpretation")
            .list_observations()[0]
            .claims
            == claims_before
        )
        assert all(claim.confidence is None for claim in claims_before)

    def test_a_scorer_scores_nothing_over_a_run_of_another_embedding_space(
        self, tmp_path: Path
    ) -> None:
        # As features da região vivem em outro espaço: são evidência de outro consumidor, não
        # entrada inválida. O scorer não pontua nada e registra todas as claims como unscored,
        # em vez de derrubar o run.
        workspace = tmp_path / "ws"
        clip_space = clip_embedding_space(
            checkpoint="openai/clip-x", revision="b" * 40, dimension=2, normalization="l2"
        )
        perception = _pinned_perception(workspace, space=clip_space)

        ref = _score(self._executor(tmp_path, _alphaclip()), workspace, perception)

        reader = SemanticScoringRunReader(workspace / str(ref.location))
        assert reader.verify_integrity() == []
        assert reader.list_scores() == []
        counts = reader.manifest.counts
        assert counts is not None
        assert (counts.unscored_region_claim_count, counts.scene_claim_count) == (3, 1)
        assert reader.manifest.embedding_space == ALPHACLIP_SPACE

    def test_the_scoring_run_identity_follows_every_setting_of_its_scorer(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        plain = self._executor(tmp_path / "a", _alphaclip())
        photo = self._executor(
            tmp_path / "b", _alphaclip(text_template="a photo of a {hypothesis}")
        )

        assert plain.configuration_identity != photo.configuration_identity


# ------------------------------------------------------------------ arms over one pinned run


class _PinnedRegionPerceptionExecutor:
    """Stands in for the region stage, which perception now depends on.

    The decomposition split perception in two, so a plan that reaches ``visual_perception``
    also reaches ``region_perception``. Scoring reads neither: it consumes the perception run,
    so this double only has to publish a well-formed region run for the plan to be runnable.
    """

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        run_id = request.identity()
        manifest = _write_regions(request.output_dir, run_id=run_id)
        return ArtifactRef(
            stage_id="region_perception",
            contract="RegionPerceptionRunArtifact",
            artifact_id=run_id,
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _PinnedPerceptionExecutor:
    """Stands in for the perception stage: publishes the same synthetic run for one request."""

    def __init__(self) -> None:
        self.executions = 0

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.executions += 1
        assert request.output_dir is not None and request.workspace is not None
        run_id = VisualEvidenceRunId(request.identity())
        manifest = _write_perception(
            request.output_dir,
            space=ALPHACLIP_SPACE,
            run_id=run_id,
            region_id=request.inputs["regions"][0].artifact_id,
        )
        return ArtifactRef(
            stage_id="visual_perception",
            contract="VisualEvidenceArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _PinnedSemanticExecutor:
    """Publish semantic evidence as its own independent reusable producer."""

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        manifest = _write_semantics(
            request.output_dir,
            region_id=request.inputs["regions"][0].artifact_id,
            run_id=request.identity(),
        )
        return ArtifactRef(
            stage_id="semantic_interpretation",
            contract="SemanticInterpretationArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


def test_scorer_arms_reuse_one_perception_run_and_publish_distinct_scoring_runs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    sequence = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id="S1-sequence",
        content_hash="sha256:" + "5" * 64,
        location="S1/run-0000/ingestion",
    )
    store = FileArtifactStore(tmp_path / "index", verify=lambda _ref: True)
    perception = _PinnedPerceptionExecutor()

    def arm(name: str, template: str) -> Any:
        (tmp_path / name).mkdir()
        effective = effective_from(tmp_path / name, _document(_alphaclip(text_template=template)))
        execution = resolve_plan(effective).scope(
            targets=["semantic_scoring"], provided={"ingestion": sequence}
        )
        scoring = compose_executors(
            effective, providers=_providers(), module_available=_ready, environ={}
        )["semantic_scoring"]
        return run_plan(
            execution,
            {
                "region_perception": _PinnedRegionPerceptionExecutor(),
                "visual_perception": perception,
                "semantic_interpretation": _PinnedSemanticExecutor(),
                "semantic_scoring": scoring,
            },
            environ={},
            module_available=_ready,
            reuse=ReusePolicy(store=store, code_identity="code-1"),
            journal=RunJournal.create(workspace, effective, execution),
        )

    first = arm("plain", "{hypothesis}")
    second = arm("photo", "a photo of a {hypothesis}")

    decisions = {stage.stage_id: stage.decision.kind for stage in second.stages}
    assert decisions["visual_perception"] == "reused"
    assert decisions["semantic_scoring"] != "reused"
    assert perception.executions == 1  # o VLM do run de percepção roda uma vez só
    outputs = [
        {stage.stage_id: stage.output for stage in record.stages} for record in (first, second)
    ]
    assert outputs[0]["visual_perception"] == outputs[1]["visual_perception"]
    scoring_a, scoring_b = (item["semantic_scoring"] for item in outputs)
    assert scoring_a.artifact_id != scoring_b.artifact_id
    assert scoring_a.content_hash != scoring_b.content_hash
    readers = [
        SemanticScoringRunReader(workspace / str(ref.location)) for ref in (scoring_a, scoring_b)
    ]
    assert {reader.manifest.lineage.visual_evidence_artifact_id for reader in readers} == {
        outputs[0]["visual_perception"].artifact_id
    }
    assert [reader.manifest.text_template for reader in readers] == [
        "{hypothesis}",
        "a photo of a {hypothesis}",
    ]
