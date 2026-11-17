"""Deterministic Visual Perception backends for the runtime executor tests.

The executor's orchestration is what these tests prove (#507), not the backends' science:
region discovery, feature extraction and semantic interpretation are fakes, while the
sequence, the prepared images, the resolved stage graph and the written run are real.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.region_perception.adapters.prepared_images import materialize_discovery_pass
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.composition import RegionDiscoveryPlan
from contextmap.runtime.executors import (
    RegionPerceptionExecutor,
    SemanticInterpretationExecutor,
    VisualPerceptionExecutor,
)
from contextmap.semantic_interpretation import (
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

CANONICAL_REQUEST_POLICY = SemanticRequestPolicy.from_prompt_policy(
    SemanticPromptPolicy(scene="scene/v1", region="region/v1"),
    views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
)
"""The canonical prompts and one tight crop per region, declared as every run must (#544)."""


def audited_without_rejections(image: Any, regions: list[Any], backend: Any) -> Any:
    """Report hand-built regions as a frame whose discovery rejected and merged nothing."""
    from contextmap.region_perception import (
        AuditedRegions,
        RegionDiscoveryAudit,
        RegionDiscoveryIdentity,
    )

    return AuditedRegions(
        regions=tuple(regions),
        audit=RegionDiscoveryAudit(
            source_observation_id=image.source_observation_id,
            backend=backend,
            passes=(),
            diagnostics=(),
            pass_rejections=(),
            discovery_identity=RegionDiscoveryIdentity.effective(),
            normalization_rejections=(),
            merge_decisions=(),
        ),
    )


class _FakeRegionDiscovery:
    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_region_discovery",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def discover(self, image: PreparedImage) -> list[Region2D]:
        return [
            Region2D(
                region_id=RegionId("region-0000"),
                bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
                provenance=self.backend_provenance(),
                source_observation_id=image.source_observation_id,
            )
        ]

    def discovery_identity(self) -> Any:
        from contextmap.region_perception import RegionDiscoveryIdentity

        return RegionDiscoveryIdentity.effective()

    def discover_audited(self, image: PreparedImage) -> Any:
        return audited_without_rejections(image, self.discover(image), self.backend_provenance())


class _FakeFeatureExtractor:
    def __init__(self, scope: FeatureScope, run_id: str = "fixture-run") -> None:
        self._scope = scope
        self._run_id = run_id

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id=f"fake_{self._scope.value}_feature_extractor",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return self._scope

    def extract(
        self, image: PreparedImage, regions: Sequence[Region2D] = ()
    ) -> Sequence[VisualFeature]:
        if self._scope is FeatureScope.DENSE:
            return [
                VisualFeature(
                    feature_id=FeatureId(f"{self._run_id}--{image.source_observation_id}--dense"),
                    source_observation_id=image.source_observation_id,
                    scope=FeatureScope.DENSE,
                    embedding_space_id="fake-dense-space",
                    shape=(2,),
                    dtype="float32",
                    payload_reference=f"{image.source_observation_id}-dense.npy",
                    provenance=self.backend_provenance(),
                )
            ]
        return [
            VisualFeature(
                feature_id=FeatureId(
                    f"{self._run_id}--{image.source_observation_id}--feature-{region.region_id}"
                ),
                source_observation_id=image.source_observation_id,
                scope=FeatureScope.REGION,
                embedding_space_id="fake-region-space",
                shape=(2,),
                dtype="float32",
                payload_reference=f"{image.source_observation_id}-{region.region_id}.npy",
                provenance=self.backend_provenance(),
                region_id=region.region_id,
            )
            for region in regions
        ]


class _FakeSemanticInterpreter:
    """Implements the real ``interpret()`` port, like every real backend does today.

    The executor bridges this to its own legacy ``interpret_scene``/``interpret_regions``
    dispatch (see ``_LegacySemanticInterpreterBridge``); a fake standing in for a real
    backend must match what a real backend implements.
    """

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_semantic_interpreter",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def capabilities(self) -> SemanticInterpreterCapabilities:
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(SemanticInterpretationMode),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=False,
        )

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        import json

        from contextmap.semantic_interpretation import (
            SEMANTIC_PROMPT_TEMPLATES,
            SemanticBackendDiagnostics,
            SemanticConfidencePolicy,
            SemanticInferenceProvenance,
            SemanticInterpretationExecution,
            parse_semantic_response,
            render_semantic_prompt,
        )

        template = SEMANTIC_PROMPT_TEMPLATES[request.prompt_template_id]
        rendered = render_semantic_prompt(
            request, template, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
        )
        raw_response = json.dumps({"abstained": True, "claims": [], "scene_context": None})
        provenance = SemanticInferenceProvenance(
            backend=self.backend_provenance(),
            task_identity=f"fake-{request.mode.value}",
            prompt_template_id=request.prompt_template_id,
            output_schema_version=request.requested_output_schema,
        )
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=raw_response,
            parsed=parse_semantic_response(
                raw_response,
                request,
                provenance,
                confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
            ),
            diagnostics=SemanticBackendDiagnostics(latency_ms=0.0),
            effective_configuration={"backend": "fake"},
        )


def served_discovery(discovery: Any) -> RegionDiscoveryPlan:
    """Serve one already built region discovery backend to every run of an executor."""
    return RegionDiscoveryPlan(
        factory=lambda _root: discovery,
        backend=discovery.backend_provenance(),
        discovery_identity=discovery.discovery_identity(),
    )


def run_scoped_discovery(build: Callable[[Callable[[Any], object]], Any]) -> RegionDiscoveryPlan:
    """Plan a real discovery adapter as the composition root does: one per run (#661).

    Args:
        build: Builds the adapter from its pass-image source. Each run's adapter cuts its passes
            from that run's prepared images (``materialize_discovery_pass``); the one built to
            read the declared identity never materializes anything.
    """

    def unused(discovery_input: Any) -> object:
        raise AssertionError("the adapter that declares the identity never discovers")

    declared = build(unused)
    return RegionDiscoveryPlan(
        factory=lambda root: build(partial(materialize_discovery_pass, root)),
        backend=declared.backend_provenance(),
        discovery_identity=declared.discovery_identity(),
    )


def stub_perception_executor(*, dense_features: Any = None) -> VisualPerceptionExecutor:
    """A real executor over deterministic fake backends: two features per replayed region.

    The discovery is not here. It belongs to the ``region_perception`` stage, and perception
    replays the regions that stage published through its ``regions`` input — see
    :func:`publish_region_run`, which produces exactly that input.

    Args:
        dense_features: The ``FeatureFactory`` of the dense stage; by default a fake whose dense
            feature records no sampling geometry.
    """
    return VisualPerceptionExecutor(
        dense_features=dense_features
        or (lambda scope: _FakeFeatureExtractor(FeatureScope.DENSE, scope.run_id)),  # type: ignore[arg-type]
        region_features=lambda scope: _FakeFeatureExtractor(FeatureScope.REGION, scope.run_id),  # type: ignore[arg-type]
    )


def publish_region_run(
    workspace: Path,
    sequence: ArtifactRef,
    *,
    discovery: Any = None,
    output_dir: Path | None = None,
    image_masks: Mapping[str, Any] | None = None,
    observation_selection: dict[str, Any] | None = None,
    config_digest: str = "sha256:regions",
    code_version: str | None = None,
) -> ArtifactRef:
    """Run the ``region_perception`` stage for real and reference the artifact it published.

    Perception consumes discovered regions as an input instead of discovering them, so a test
    that exercises ``VisualPerceptionExecutor`` needs a published
    ``RegionPerceptionRunArtifact`` to point the ``regions`` input at. This runs the real
    ``RegionPerceptionExecutor`` over the same sequence rather than hand-writing the artifact:
    the regions, their mask payloads and their discovery audits then come from the producer
    that owns them, which is what the replay reads back.

    Args:
        workspace: The run workspace the stage requests resolve locations against.
        sequence: The published sequence artifact to discover regions in.
        discovery: The region discovery backend, or an already planned one
            (:class:`RegionDiscoveryPlan`, as the composition root resolves it); by default the
            one-region fake.
        output_dir: Where the region run is written; by default a ``region_perception``
            directory beside the sequence artifact.
        image_masks: Static per-camera masks, which must match the preparation perception
            applies to the same frames.
        observation_selection: Restricts the frames discovery covers. ``None`` covers the whole
            sequence, which is what a perception selection of any subset needs.
        config_digest: The stage's declared configuration digest.
        code_version: The code revision recorded in the run manifest.

    Returns:
        The reference to use as perception's ``regions`` input.
    """
    if discovery is None:
        discovery = _FakeRegionDiscovery()
    plan = discovery if isinstance(discovery, RegionDiscoveryPlan) else served_discovery(discovery)
    executor = RegionPerceptionExecutor(
        region_discovery=plan, image_masks=image_masks, code_version=code_version
    )
    if output_dir is None:
        assert sequence.location is not None, "the sequence artifact must name its location"
        output_dir = (workspace / sequence.location).parent / "region_perception"
    return executor.execute(
        StageRequest(
            stage_id="region_perception",
            inputs={"sequence": (sequence,)},
            components={},
            config_digest=config_digest,
            output_dir=output_dir,
            workspace=workspace,
            observation_selection=observation_selection,
        )
    )


def stub_semantic_executor() -> SemanticInterpretationExecutor:
    """Build the independent semantic producer over the deterministic request backend."""
    return SemanticInterpretationExecutor(
        semantic_interpreter=lambda _root: _FakeSemanticInterpreter(),
        semantic_request_policy=CANONICAL_REQUEST_POLICY,
    )  # type: ignore[arg-type,return-value]


def publish_selected_evidence(
    workspace: Path,
    *,
    sequence_id: str,
    selection: dict[str, Any],
    run_index: int = 1,
    selections: Mapping[str, dict[str, Any]] | None = None,
    sequences: Mapping[str, str] | None = None,
    regional_dependencies: Mapping[str, str] | None = None,
) -> dict[str, tuple[ArtifactRef, ...]]:
    """Publish three empty producer artifacts for selection and lineage validation tests.

    These tests exercise immutable manifests, independently of image/model computation.
    Per-producer overrides construct deliberately incompatible selections before publication.
    """
    from contextmap.ingestion import SequenceArtifactId, decode_selection, selection_identity
    from contextmap.region_perception import RegionPerceptionRunManifest, RegionPerceptionRunWriter
    from contextmap.runtime.executors import inventory_digest
    from contextmap.semantic_interpretation import (
        SemanticInterpretationRunId,
        SemanticInterpretationRunWriter,
    )
    from contextmap.visual_perception import VisualEvidenceRunWriter

    selections = selections or {}
    sequences = sequences or {}
    regional_dependencies = regional_dependencies or {}
    references: dict[str, tuple[ArtifactRef, ...]] = {}
    for stage, contract in (
        ("region_perception", "RegionPerceptionRunArtifact"),
        ("visual_perception", "VisualEvidenceArtifact"),
        ("semantic_interpretation", "SemanticInterpretationArtifact"),
    ):
        producer_id = f"{stage}-{run_index}"
        upstream = sequences.get(stage, sequence_id)
        selection_id = selection_identity(
            SequenceArtifactId(upstream), decode_selection(selections.get(stage, selection))
        )
        location = f"S1/run-{run_index:04d}/{stage}"
        output = workspace / location
        regional_id = regional_dependencies.get(stage, f"region_perception-{run_index}")
        if stage == "region_perception":
            with RegionPerceptionRunWriter(output) as regional_writer:
                regional_writer.write_manifest(
                    RegionPerceptionRunManifest(
                        run_id=producer_id,
                        run_index=run_index,
                        sequence_artifact_id=upstream,
                        source_observation_ids=(),
                        sequence_name="S1",
                        selection_id=selection_id,
                        created_at="2026-01-01T00:00:00+00:00",
                        prepared_image_id="fixture/preparation/1",
                        discovery_backend_id="fake",
                        discovery_model="fake",
                        discovery_version="1",
                        discovery_configuration_fingerprint="sha256:test",
                        normalization_policy_fingerprint="sha256:test",
                    )
                )
                inventory = regional_writer.finalize().file_inventory
        elif stage == "visual_perception":
            with VisualEvidenceRunWriter(
                output_dir=output,
                run_id=producer_id,
                sequence_name="S1",
                run_index=run_index,
                sequence_artifact_id=upstream,
                selection_id=selection_id,
                region_artifact_id=regional_id,
                configuration_digest="sha256:test",
            ) as visual_writer:
                inventory = visual_writer.finalize().file_inventory
        else:
            with SemanticInterpretationRunWriter(
                output_dir=output,
                run_id=SemanticInterpretationRunId(producer_id),
                sequence_name="S1",
                run_index=run_index,
                sequence_artifact_id=upstream,
                selection_id=selection_id,
                region_artifact_id=regional_id,
                configuration_digest="sha256:test",
            ) as semantic_writer:
                inventory = semantic_writer.finalize().file_inventory
        references[stage] = (
            ArtifactRef(
                stage_id=stage,
                contract=contract,
                artifact_id=producer_id,
                content_hash=inventory_digest(inventory),
                location=location,
            ),
        )
    return references
