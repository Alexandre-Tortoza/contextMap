"""Region grounding (#569) and its SAM2 refinement (#568), composed and executed from config.

Grounding and refinement are optional producers inside the ``region_perception`` stage. Its queries
are a reserved parameter group (``query_set``) that becomes one explicit request per image
and query; they never reach the backend configuration or its fingerprint.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_perception import audited_without_rejections
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.region_perception import (
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingQueryPolicy,
    GroundingTask,
    Region2D,
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
)
from contextmap.region_perception.adapters.locateanything import (
    CATEGORY_DETECTION_POLICY,
    POINTING_POLICY,
    LocateAnythingConfig,
    LocateAnythingGeneration,
    LocateAnythingRegionGrounding,
    TransformersLocateAnythingRuntime,
)
from contextmap.runtime.composition import (
    RegionGroundingPlan,
    RegionRefinementPlan,
    RuntimeProvider,
    compose,
    compose_executors,
)
from contextmap.runtime.errors import BackendConfigurationError, BackendUnavailableError
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    PreparedImage,
)

GROUNDING = "region_perception.grounding"
REFINEMENT = "region_perception.refinement"

REGION = "region_perception.discovery"

INTERPRETER = "semantic_interpretation.interpreter"

REVISION = "c" * 40

CATEGORIES = {
    "task": "category_detection",
    "policy_id": CATEGORY_DETECTION_POLICY,
    "geometry": "box",
    "categories": ["chair", "table"],
}

POINTING = {
    "task": "phrase_grounding",
    "policy_id": POINTING_POLICY,
    "geometry": "point",
    "text": "the door handle",
}


def _document(*queries: dict[str, Any], **parameters: Any) -> dict[str, Any]:
    document = selected_document()
    block: dict[str, Any] = {
        "model": "nvidia/LocateAnything-3B",
        "revision": REVISION,
        "dtype": "bfloat16",
        "generation_mode": "hybrid",
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "text_attention": "sdpa",
        "vision_attention": "sdpa",
        "query_set": {"queries": list(queries or (CATEGORIES,))},
    }
    block.update(parameters)
    document["components"]["region_perception"]["grounding"] = {
        "backend": "locateanything",
        "locateanything": block,
    }
    return document


class _FakeLocateAnythingRuntime:
    def generate(self, *, image: Any, prompt: str, config: Any) -> LocateAnythingGeneration:
        return LocateAnythingGeneration(text="<|im_end|>")


def _providers(**extra: RuntimeProvider) -> dict[str, RuntimeProvider]:
    providers: dict[str, RuntimeProvider] = {
        REGION: lambda _config, _secrets: object(),
        INTERPRETER: lambda _config, _secrets: object(),
    }
    providers.update(extra)
    return providers


def _compose(tmp_path: Path, document: dict[str, Any], **options: Any) -> Any:
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    options.setdefault("providers", _providers())
    # Discovery, grounding e refinement passaram a ser compostos pelo stage
    # region_perception; pedir só visual_perception deixaria os três planos em None.
    return compose(
        effective_from(tmp_path, document),
        stages=["region_perception", "visual_perception"],
        **options,
    )


def test_region_grounding_is_optional_and_absent_unless_selected(tmp_path: Path) -> None:
    composed = _compose(tmp_path, selected_document())

    assert composed.region_grounding is None
    assert composed.region_discovery is not None


def test_locateanything_is_selectable_with_explicit_queries(tmp_path: Path) -> None:
    composed = _compose(tmp_path, _document(CATEGORIES, POINTING))

    plan = composed.region_grounding
    assert isinstance(plan, RegionGroundingPlan)
    assert plan.queries == (
        GroundingQuery(
            task=GroundingTask.CATEGORY_DETECTION,
            policy_id=CATEGORY_DETECTION_POLICY,
            geometry=GroundingGeometry.BOX,
            categories=("chair", "table"),
        ),
        GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id=POINTING_POLICY,
            geometry=GroundingGeometry.POINT,
            text="the door handle",
        ),
    )


def test_the_bundled_runtime_is_built_per_run_and_loads_nothing_at_composition(
    tmp_path: Path,
) -> None:
    plan = _compose(tmp_path, _document()).region_grounding

    backend = plan.factory(tmp_path)

    assert isinstance(backend, LocateAnythingRegionGrounding)
    assert isinstance(backend._runtime, TransformersLocateAnythingRuntime)  # type: ignore[attr-defined]
    assert backend._runtime._loaded is None  # type: ignore[attr-defined]


def test_the_resolved_device_reaches_the_backend_configuration(tmp_path: Path) -> None:
    backend = _compose(tmp_path, _document()).region_grounding.factory(tmp_path)

    assert backend._config.device == "cpu"  # type: ignore[attr-defined]


def test_a_provider_runtime_replaces_the_bundled_loader(tmp_path: Path) -> None:
    calls: list[Any] = []

    def provide(config: Any, secrets: Any) -> object:
        calls.append(config)
        return _FakeLocateAnythingRuntime()

    composed = _compose(tmp_path, _document(), providers=_providers(**{GROUNDING: provide}))
    backend = composed.region_grounding.factory(tmp_path)

    assert isinstance(backend._runtime, _FakeLocateAnythingRuntime)  # type: ignore[attr-defined]
    assert isinstance(calls[0], LocateAnythingConfig)


def test_the_query_set_is_required(tmp_path: Path) -> None:
    document = _document()
    del document["components"]["region_perception"]["grounding"]["locateanything"]["query_set"]

    with pytest.raises(BackendConfigurationError, match="query_set"):
        _compose(tmp_path, document)


@pytest.mark.parametrize(
    "query",
    [
        {**CATEGORIES, "geometry": "point"},
        {**CATEGORIES, "policy_id": "locateanything.category-detection/9"},
        {**CATEGORIES, "categories": ["chair</c>table"]},
        {**CATEGORIES, "categories": []},
    ],
    ids=["unsupported-geometry", "unknown-policy", "separator", "no-category"],
)
def test_an_unservable_query_fails_at_composition_before_any_model(
    tmp_path: Path, query: dict[str, Any]
) -> None:
    with pytest.raises(BackendConfigurationError, match="query_set"):
        _compose(tmp_path, _document(query))


def test_a_repeated_query_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigurationError, match="unique"):
        _compose(tmp_path, _document(CATEGORIES, CATEGORIES))


def test_invalid_inference_settings_fail_at_composition(tmp_path: Path) -> None:
    with pytest.raises(BackendConfigurationError, match="revision"):
        _compose(tmp_path, _document(revision="main"))
    with pytest.raises(BackendConfigurationError, match="text_attention"):
        _compose(tmp_path, _document(text_attention="la_flash"))


def test_a_missing_sdk_is_reported_before_any_model(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableError, match="torch"):
        _compose(tmp_path, _document(), module_available=lambda name: name != "torch")


def test_queries_change_the_run_identity_but_not_the_backend_fingerprint(
    tmp_path: Path,
) -> None:
    first_document = _document(CATEGORIES)
    second_document = _document({**CATEGORIES, "categories": ["table", "chair"]})

    first = effective_from(tmp_path, first_document)
    first_backend = _compose(tmp_path, first_document).region_grounding.factory(tmp_path)
    second = effective_from(tmp_path, second_document)
    second_backend = _compose(tmp_path, second_document).region_grounding.factory(tmp_path)

    assert first.digest != second.digest
    assert (
        first_backend.backend_provenance().configuration_fingerprint
        == second_backend.backend_provenance().configuration_fingerprint
    )


def test_compose_executors_wires_grounding_into_region_perception(tmp_path: Path) -> None:
    from contextmap.runtime.executors import RegionPerceptionExecutor

    executors = compose_executors(
        effective_from(tmp_path, _document()),
        providers=_providers(),
        module_available=lambda _name: True,
        environ={},
    )

    executor = executors["region_perception"]
    assert isinstance(executor, RegionPerceptionExecutor)
    assert isinstance(executor._region_grounding, RegionGroundingPlan)  # type: ignore[attr-defined]


class _FakeGrounding:
    """Answers every query with one box over the whole image, recording each request.

    ``failing_on`` names an observation whose every request raises, as a backend whose
    runtime fails on one image would.
    """

    def __init__(
        self,
        *,
        fingerprint: str = "sha256:fake-grounding",
        failing_on: str | None = None,
    ) -> None:
        self.requests: list[RegionGroundingRequest] = []
        self._fingerprint = fingerprint
        self._failing_on = failing_on

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint=self._fingerprint,
        )

    def capabilities(self) -> RegionGroundingCapabilities:
        return RegionGroundingCapabilities(
            query_policies=(
                GroundingQueryPolicy(
                    policy_id="fake.phrase/1",
                    task=GroundingTask.PHRASE_GROUNDING,
                    geometry=GroundingGeometry.BOX,
                ),
            )
        )

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        self.requests.append(request)
        if str(request.source_observation_id) == self._failing_on:
            raise RuntimeError(f"grounding runtime failed on {self._failing_on}")
        return RegionGroundingExecution(
            request=request,
            provenance=self.backend_provenance(),
            rendered_prompt=f"find {request.query.text}",
            raw_response="<box><0><0><1000><1000></box>",
            outputs=(
                GroundingOutput(
                    output_index=0,
                    native_text="<box><0><0><1000><1000></box>",
                    label=request.query.text,
                    box=BoundingBox2D(
                        x=0.0, y=0.0, width=request.image.width, height=request.image.height
                    ),
                ),
            ),
            diagnostics=GroundingDiagnostics(latency_ms=1.0),
            effective_configuration=MappingProxyType({"backend": "fake"}),
        )


def _refined_document(**sam2: Any) -> dict[str, Any]:
    document = _document()
    document["components"]["region_perception"]["refinement"] = {
        "backend": "sam2",
        "sam2": {"checkpoint": "sam2.1_hiera_large", "model_version": "2.1", **sam2},
    }
    return document


class _FakePromptRuntime:
    """Answers every prompt with a full-image mask, like a SAM2 that sees one object."""

    def predict_prompts(self, **kwargs: Any) -> tuple[Any, ...]:
        import numpy as np

        from contextmap.region_perception import InlineMask
        from contextmap.region_perception.adapters.sam2 import Sam2PromptedMask

        mask = InlineMask(np.ones((kwargs["height"], kwargs["width"]), dtype=bool))
        return tuple(Sam2PromptedMask(mask=mask, predicted_iou=0.9) for _ in kwargs["prompts"])


def _refinement_providers(calls: list[Any] | None = None) -> dict[str, RuntimeProvider]:
    def provide(config: Any, secrets: Any) -> object:
        (calls if calls is not None else []).append(config)
        return _FakePromptRuntime()

    return _providers(**{REFINEMENT: provide})


def test_refinement_is_optional_and_absent_unless_selected(tmp_path: Path) -> None:
    assert _compose(tmp_path, _document()).region_refinement is None


def test_sam2_refinement_is_selectable_with_a_provided_prompt_runtime(tmp_path: Path) -> None:
    from contextmap.region_perception.adapters.sam2 import (
        Sam2PromptRefinement,
        Sam2RefinementConfig,
    )

    calls: list[Any] = []
    composed = _compose(tmp_path, _refined_document(), providers=_refinement_providers(calls))

    refiner = composed.region_refinement.factory(tmp_path)

    assert isinstance(refiner, Sam2PromptRefinement)
    assert isinstance(calls[0], Sam2RefinementConfig)
    assert calls[0].device == "cpu"
    assert len(calls) == 1  # o provider é pedido uma vez na composição, não por run


def test_refinement_needs_a_provided_sam2_runtime(tmp_path: Path) -> None:
    from contextmap.runtime.errors import BackendRuntimeMissingError

    with pytest.raises(BackendRuntimeMissingError):
        _compose(tmp_path, _refined_document())


def test_refinement_without_grounding_is_an_explicit_configuration_error(tmp_path: Path) -> None:
    from contextmap.runtime import ConfigurationError

    document = _refined_document()
    del document["components"]["region_perception"]["grounding"]

    with pytest.raises(ConfigurationError, match=re.escape("region_perception.grounding")):
        _compose(tmp_path, document, providers=_refinement_providers())


def test_grounding_and_refinement_are_ablated_independently(tmp_path: Path) -> None:
    grounding_only = _compose(tmp_path, _document())
    refined = _compose(tmp_path, _refined_document(), providers=_refinement_providers())
    other_refiner = _compose(
        tmp_path, _refined_document(mask_threshold=0.5), providers=_refinement_providers()
    )

    def grounding_fingerprint(composed: Any) -> Any:
        return composed.region_grounding.factory(tmp_path).backend_provenance()

    assert grounding_fingerprint(grounding_only) == grounding_fingerprint(refined)
    assert grounding_fingerprint(refined) == grounding_fingerprint(other_refiner)
    assert (
        refined.region_refinement.factory(tmp_path).backend_provenance().configuration_fingerprint
        != other_refiner.region_refinement.factory(tmp_path)
        .backend_provenance()
        .configuration_fingerprint
    )


def test_compose_executors_wires_refinement_into_region_perception(tmp_path: Path) -> None:
    executors = compose_executors(
        effective_from(tmp_path, _refined_document()),
        providers=_refinement_providers(),
        module_available=lambda _name: True,
        environ={},
    )

    assert executors["region_perception"]._region_refinement is not None  # type: ignore[attr-defined]


class _TwoRegionsDiscovery:
    """Discovers ``region-0000`` and ``region-0001``, one pixel each, on every 2x2 frame."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="fake",
            model="f",
            version="0",
        )

    def discovery_identity(self) -> Any:
        from contextmap.region_perception import RegionDiscoveryIdentity

        return RegionDiscoveryIdentity.effective()

    def discover(self, image: PreparedImage) -> list[Region2D]:
        from contextmap.region_perception import RegionId

        return [
            Region2D(
                region_id=RegionId(f"region-{index:04d}"),
                bounding_box=BoundingBox2D(x=index, y=index, width=1, height=1),
                provenance=self.backend_provenance(),
                source_observation_id=image.source_observation_id,
            )
            for index in range(2)
        ]

    def discover_audited(self, image: PreparedImage) -> Any:
        return audited_without_rejections(image, self.discover(image), self.backend_provenance())


def _refinement_plan() -> RegionRefinementPlan:
    """SAM2 refinement over a fake prompt runtime that returns a full-image mask per prompt."""
    from contextmap.region_perception.adapters.sam2 import (
        Sam2PromptRefinement,
        Sam2RefinementConfig,
    )

    config = Sam2RefinementConfig(checkpoint="sam2.1_hiera_large", model_version="2.1")
    return RegionRefinementPlan(
        factory=lambda root: Sam2PromptRefinement(
            config=config, runtime=_FakePromptRuntime(), prepared_image_root=root
        ),
        backend=Sam2PromptRefinement.provenance_for(config),
    )


class _RefinerFailingOn:
    """Delegates to a SAM2 refiner, except on one observation, where the refiner raises."""

    def __init__(self, refiner: Any, failing_on: str) -> None:
        self._refiner = refiner
        self._failing_on = failing_on

    def backend_provenance(self) -> BackendProvenance:
        return self._refiner.backend_provenance()  # type: ignore[no-any-return]

    def capabilities(self) -> Any:
        return self._refiner.capabilities()

    def refine(self, request: Any) -> Any:
        if str(request.source_observation_id) == self._failing_on:
            raise RuntimeError(f"refinement runtime failed on {self._failing_on}")
        return self._refiner.refine(request)


def _run_regions(
    tmp_path: Path,
    *,
    grounding: Any = None,
    discovery: Any = None,
    refinement: RegionRefinementPlan | None = None,
    with_grounding: bool = True,
    declared_grounding_backend: BackendProvenance | None = None,
) -> tuple[Any, list[Path]]:
    """Execute regional discovery, queries and optional refinement over two physical frames."""
    from runtime_perception import served_discovery

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
    from contextmap.region_perception import RegionPerceptionRunReader
    from contextmap.runtime import ArtifactRef, StageRequest
    from contextmap.runtime.executors import RegionPerceptionExecutor, inventory_digest
    from contextmap.shared import SourceTimestamp

    backend = grounding or _FakeGrounding()
    roots: list[Path] = []

    def factory(root: Path) -> Any:
        roots.append(root)
        return backend

    queries = tuple(
        GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            policy_id="fake.phrase/1",
            geometry=GroundingGeometry.BOX,
            text=text,
        )
        for text in ("the red chair", "the table")
    )
    executor = RegionPerceptionExecutor(
        region_discovery=served_discovery(discovery or _TwoRegionsDiscovery()),
        region_grounding=RegionGroundingPlan(
            queries=queries,
            factory=factory,
            backend=declared_grounding_backend or backend.backend_provenance(),
        )
        if with_grounding
        else None,
        region_refinement=refinement,
    )
    workspace = tmp_path / "ws"
    ingestion = workspace / "corridor-02/run-0001/ingestion"
    with SequenceArtifactWriter(
        output_dir=ingestion,
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    ) as writer:
        for index in range(2):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index:04d}"),
                    sensor_id=SensorId("camera"),
                    frame_id=FrameId("camera-optical"),
                    timestamp=SourceTimestamp(seconds=index, nanoseconds=0, clock_id="fixture"),
                    provenance=SourceProvenance(source_type="fixture", source_path="fixtures"),
                    width=2,
                    height=2,
                    encoding=ImageEncoding.RGB8,
                    data=bytes([index, 20, 30] * 4),
                )
            )
        manifest = writer.finalize()
    sequence = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=str(manifest.artifact_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location="corridor-02/run-0001/ingestion",
    )
    output = workspace / "corridor-02/run-0001/region_perception"
    executor.execute(
        StageRequest(
            stage_id="region_perception",
            inputs={"sequence": (sequence,)},
            components={},
            config_digest="sha256:test",
            output_dir=output,
            workspace=workspace,
        )
    )
    return RegionPerceptionRunReader(output), roots


@pytest.mark.usefixtures("fake_pillow")
def test_each_query_is_persisted_under_its_regional_producer(tmp_path: Path) -> None:
    reader, roots = _run_regions(tmp_path)
    assert reader.verify_integrity() == []
    assert len(roots) == 1
    assert all(not root.exists() for root in roots)
    executions = reader.read_grounding()
    assert len(executions) == 4
    assert {(str(e.request.source_observation_id), e.request.query.text) for e in executions} == {
        (f"frame-{index:04d}", text)
        for index in range(2)
        for text in ("the red chair", "the table")
    }
    assert all(e.request.region_perception_run_id == reader.manifest.run_id for e in executions)
    assert len(reader.read_regions()) == 8
    assert len({r.region_id for r in reader.read_regions()}) == 8
    assert len(reader.read_frames()) == 2


@pytest.mark.usefixtures("fake_pillow")
def test_refinement_preserves_grounded_boxes_and_publishes_lazy_masks(tmp_path: Path) -> None:
    reader, _ = _run_regions(tmp_path, refinement=_refinement_plan())
    assert reader.verify_integrity() == []
    assert len(reader.read_refinement()) == 2
    regions = reader.read_regions()
    boxes = [r for r in regions if r.provenance.capability == "region_grounding"]
    refined = [r for r in regions if r.provenance.capability == "region_refinement"]
    assert len(boxes) == len(refined) == 4
    assert all(r.mask is None and r.mask_reference is None for r in boxes)
    assert all(r.mask is None and r.mask_reference is not None for r in refined)
    for region in refined:
        assert reader.mask_store().load(region.source_observation_id, region.region_id).area == 4


@pytest.mark.usefixtures("fake_pillow")
def test_grounding_failure_is_a_failed_regional_frame_with_discovery_preserved(
    tmp_path: Path,
) -> None:
    reader, _ = _run_regions(tmp_path, grounding=_FakeGrounding(failing_on="frame-0001"))
    assert reader.verify_integrity() == []
    frames = reader.read_frames()
    assert [f.outcome for f in frames] == ["succeeded", "failed"]
    assert "grounding runtime failed on frame-0001" in frames[1].failure_detail
    assert len(reader.read_grounding()) == 2
    failed = [r for r in reader.read_regions() if str(r.source_observation_id) == "frame-0001"]
    assert len(failed) == 2
    assert all(r.provenance.capability == "region_discovery" for r in failed)


@pytest.mark.usefixtures("fake_pillow")
def test_refinement_failure_preserves_discovery_and_grounding_of_its_frame(tmp_path: Path) -> None:
    from dataclasses import replace

    plan = _refinement_plan()
    failing = replace(
        plan, factory=lambda root: _RefinerFailingOn(plan.factory(root), "frame-0001")
    )
    reader, _ = _run_regions(tmp_path, refinement=failing)
    assert reader.verify_integrity() == []
    assert len(reader.read_grounding()) == 4
    assert len(reader.read_refinement()) == 1
    failed = reader.read_frames()[1]
    assert failed.outcome == "failed"
    assert "refinement runtime failed on frame-0001" in failed.failure_detail
    assert sorted(
        r.provenance.capability
        for r in reader.read_regions()
        if str(r.source_observation_id) == "frame-0001"
    ) == ["region_discovery", "region_discovery", "region_grounding", "region_grounding"]


@pytest.mark.usefixtures("fake_pillow")
def test_no_selected_grounding_keeps_only_namespaced_discovery(tmp_path: Path) -> None:
    reader, roots = _run_regions(tmp_path, with_grounding=False)
    assert roots == []
    assert reader.read_grounding() == ()
    assert reader.read_refinement() == ()
    assert len(reader.read_regions()) == 4
    assert all(r.provenance.capability == "region_discovery" for r in reader.read_regions())
    assert all(f.outcome == "succeeded" for f in reader.read_frames())


def _regional_identity(tmp_path: Path, document: dict[str, Any]) -> str:
    executors = compose_executors(
        effective_from(tmp_path, document),
        providers=_refinement_providers(),
        module_available=lambda _name: True,
        environ={},
    )
    from contextmap.runtime.executors import RegionPerceptionExecutor

    executor = executors["region_perception"]
    assert isinstance(executor, RegionPerceptionExecutor)
    return executor.configuration_identity


def test_effective_grounding_defaults_and_parser_version_change_regional_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config_defaults import replace_default

    from contextmap.region_perception.adapters import locateanything

    baseline = _regional_identity(tmp_path, _document())
    with monkeypatch.context() as scoped:
        replace_default(scoped, LocateAnythingConfig, "top_p", 0.8)
        assert _regional_identity(tmp_path, _document()) != baseline
    with monkeypatch.context() as scoped:
        scoped.setattr(locateanything, "PARSER_VERSION", "locateanything-response/0")
        assert _regional_identity(tmp_path, _document()) != baseline


def test_refinement_defaults_and_acceptance_policy_change_regional_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config_defaults import replace_default

    from contextmap.region_perception.adapters.sam2 import Sam2RefinementConfig
    from contextmap.runtime import executors as module

    baseline = _regional_identity(tmp_path, _refined_document())
    with monkeypatch.context() as scoped:
        replace_default(scoped, Sam2RefinementConfig, "mask_threshold", 0.5)
        assert _regional_identity(tmp_path, _refined_document()) != baseline
    with monkeypatch.context() as scoped:
        scoped.setattr(module, "REFINEMENT_ACCEPTANCE_POLICY", "refinement-acceptance/0")
        assert _regional_identity(tmp_path, _refined_document()) != baseline


def test_unselected_regional_defaults_do_not_change_its_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from config_defaults import replace_default

    from contextmap.region_perception.adapters.sam2 import Sam2RefinementConfig

    baseline = _regional_identity(tmp_path, selected_document())
    replace_default(monkeypatch, LocateAnythingConfig, "top_p", 0.8)
    replace_default(monkeypatch, Sam2RefinementConfig, "mask_threshold", 0.5)
    assert _regional_identity(tmp_path, selected_document()) == baseline
    assert _regional_identity(tmp_path, _document()) != baseline
    assert _regional_identity(tmp_path, _refined_document()) != _regional_identity(
        tmp_path, _document()
    )


@pytest.mark.usefixtures("fake_pillow")
@pytest.mark.parametrize("producer", ["grounding", "refinement"])
def test_a_backend_different_from_its_composed_identity_is_rejected(
    tmp_path: Path, producer: str
) -> None:
    from dataclasses import replace

    from contextmap.runtime.executors import ExecutorError

    if producer == "grounding":
        backend = _FakeGrounding()
        options: dict[str, Any] = {
            "grounding": backend,
            "declared_grounding_backend": replace(backend.backend_provenance(), model="other"),
        }
    else:
        plan = _refinement_plan()
        options = {"refinement": replace(plan, backend=replace(plan.backend, model="other"))}
    with pytest.raises(ExecutorError, match="provenance differs from its composed plan"):
        _run_regions(tmp_path, **options)
    assert not (tmp_path / "ws/corridor-02/run-0001/region_perception/manifest.json").exists()
