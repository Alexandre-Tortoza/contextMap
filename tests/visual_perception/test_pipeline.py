from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import pytest
from config_defaults import replace_default
from fakes import (
    FakeDenseFeatureExtractor,
    FakeRegionDiscovery,
    FakeRegionFeatureExtractor,
    FakeSemanticInterpreter,
)

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, DiscoveryPassConfig, NormalizationConfig
from contextmap.region_perception.adapters.sam2 import Sam2Config, Sam2RegionDiscovery
from contextmap.region_perception.discovery import region_discovery as discovery_module
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.execution.pipeline_presets import CANONICAL_PRESET_V1
from contextmap.visual_perception.execution.pipeline_resolution import (
    KNOWN_CAPABILITIES,
    PipelineConfigError,
    PipelinePreset,
    StageBackendFactory,
    StageSpec,
    decode_pipeline_preset,
    encode_pipeline_preset,
    resolve_pipeline,
)
from contextmap.visual_perception.execution.stage_execution import execute_stage_graph
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_features import FeatureScope


def _prepared_image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-0124"),
        payload_reference="__DEBUG__/frame-0124/prepared.jpg",
        width=640,
        height=480,
    )


def _canonical_backend_factories(
    *, region_discovery: object | None = None, call_log: list[str] | None = None
) -> dict[str, StageBackendFactory]:
    def _record(stage_id: str, factory: Callable[[], object]) -> StageBackendFactory:
        def wrapped(_parameters: Mapping[str, object]) -> object:
            if call_log is not None:
                call_log.append(stage_id)
            return factory()

        return wrapped

    return {
        "region_discovery": _record(
            "region_discovery", lambda: region_discovery or FakeRegionDiscovery()
        ),
        "dense_feature_extraction": _record("dense_feature_extraction", FakeDenseFeatureExtractor),
        "region_feature_extraction": _record(
            "region_feature_extraction", FakeRegionFeatureExtractor
        ),
        "scene_interpretation": _record("scene_interpretation", FakeSemanticInterpreter),
        "region_interpretation": _record("region_interpretation", FakeSemanticInterpreter),
    }


def test_canonical_preset_resolves_and_produces_full_evidence() -> None:
    image = _prepared_image()
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    stages = resolved.build_stage_graph({"image_preparation": image})
    outcomes = execute_stage_graph(stages)

    outputs = {item.stage_id: item.output for item in outcomes}
    assert len(cast(Any, outputs["region_discovery"])) == 1
    assert len(cast(Any, outputs["dense_feature_extraction"])) == 1
    assert len(cast(Any, outputs["region_feature_extraction"])) == 1
    assert len(cast(Any, outputs["region_interpretation"])) == 1
    assert outputs["scene_interpretation"] is not None


def test_heavy_backends_are_not_constructed_until_validation_succeeds() -> None:
    call_log: list[str] = []
    invalid_preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(
                stage_id="region_discovery",
                capability="region_discovery",
                inputs={"image": "does_not_exist"},
                backend_id="region_discovery/canonical",
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match="unknown stage"):
        resolve_pipeline(
            invalid_preset, backend_factories=_canonical_backend_factories(call_log=call_log)
        )

    assert call_log == []


def test_validate_rejects_wrong_input_name_before_constructing_backends() -> None:
    call_log: list[str] = []
    preset = PipelinePreset(
        preset_id="broken/inputs",
        stages=(
            StageSpec(stage_id="image_preparation", capability="image_preparation"),
            StageSpec(
                stage_id="region_discovery",
                capability="region_discovery",
                inputs={"wrong": "image_preparation"},
                backend_id="region_discovery/canonical",
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match=r"required input.*image"):
        resolve_pipeline(preset, backend_factories=_canonical_backend_factories(call_log=call_log))

    assert call_log == []


def test_validate_rejects_incompatible_input_producer() -> None:
    preset = PipelinePreset(
        preset_id="broken/input-contract",
        stages=(
            StageSpec(stage_id="image_preparation", capability="image_preparation"),
            StageSpec(
                stage_id="first_discovery",
                capability="region_discovery",
                inputs={"image": "image_preparation"},
                backend_id="region_discovery/first",
            ),
            StageSpec(
                stage_id="second_discovery",
                capability="region_discovery",
                inputs={"image": "first_discovery"},
                backend_id="region_discovery/second",
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match=r"input 'image'.*image_preparation"):
        resolve_pipeline(preset, backend_factories={})


def test_region_feature_scope_requires_regions_before_constructing_backends() -> None:
    call_log: list[str] = []
    preset = PipelinePreset(
        preset_id="broken/region-feature-inputs",
        stages=(
            StageSpec(stage_id="image_preparation", capability="image_preparation"),
            StageSpec(
                stage_id="region_feature_extraction",
                capability="feature_extractor",
                inputs={"image": "image_preparation"},
                backend_id="region_feature_extractor/canonical",
                feature_scope=FeatureScope.REGION,
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match=r"required input.*regions"):
        resolve_pipeline(preset, backend_factories=_canonical_backend_factories(call_log=call_log))

    assert call_log == []


def test_backends_are_resolved_once_and_reused_across_many_observations() -> None:
    call_log: list[str] = []
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories(call_log=call_log)
    )
    assert call_log.count("region_discovery") == 1

    for observation_id in ("frame-0001", "frame-0002", "frame-0003"):
        image = PreparedImage(
            source_observation_id=SourceObservationId(observation_id),
            payload_reference=f"__DEBUG__/{observation_id}/prepared.jpg",
            width=640,
            height=480,
        )
        stages = resolved.build_stage_graph({"image_preparation": image})
        execute_stage_graph(stages)

    # The backend factory ran exactly once at resolve time, not once per
    # observation: build_stage_graph only ever reuses the already
    # resolved backend instance.
    assert call_log.count("region_discovery") == 1


def test_validate_rejects_duplicate_stage_id() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(stage_id="a", capability="image_preparation"),
            StageSpec(stage_id="a", capability="image_preparation"),
        ),
    )
    with pytest.raises(PipelineConfigError, match="duplicate stage_id"):
        resolve_pipeline(preset, backend_factories={})


def test_validate_rejects_cycle() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(
            StageSpec(
                stage_id="a",
                capability="region_discovery",
                inputs={"image": "b"},
                backend_id="fake",
            ),
            StageSpec(
                stage_id="b",
                capability="region_discovery",
                inputs={"image": "a"},
                backend_id="fake",
            ),
        ),
    )
    factories: dict[str, StageBackendFactory] = {
        "a": lambda _p: object(),
        "b": lambda _p: object(),
    }
    with pytest.raises(PipelineConfigError, match="cycle"):
        resolve_pipeline(preset, backend_factories=factories)


def test_validate_rejects_unknown_capability() -> None:
    preset = PipelinePreset(
        preset_id="broken/1",
        stages=(StageSpec(stage_id="a", capability="not_a_real_capability", backend_id="fake"),),
    )
    factories: dict[str, StageBackendFactory] = {"a": lambda _p: object()}
    with pytest.raises(PipelineConfigError, match="no capability adapter"):
        resolve_pipeline(preset, backend_factories=factories)


def test_optional_stage_can_be_inserted_without_touching_downstream_capability_code() -> None:
    """A non-canonical preset can route region_feature_extraction's 'regions'
    input through an extra optional stage instead of region_discovery
    directly — without editing this module's capability adapters at all.
    """

    def _passthrough_regions(image: PreparedImage) -> Sequence:
        return FakeRegionDiscovery(boxes=(BoundingBox2D(x=1, y=1, width=5, height=5),)).discover(
            image
        )

    class _FakeRegionRefinement:
        def backend_provenance(self) -> BackendProvenance:
            return BackendProvenance(
                backend_id="fake_region_refinement",
                capability="region_discovery",
                provider="fake",
                model="fake",
                version="0.1",
            )

        def discover(self, image: PreparedImage) -> Sequence:
            return _passthrough_regions(image)

    preset = PipelinePreset(
        preset_id="experimental/region-refinement",
        description="Inserts an optional region-refinement stage before feature extraction.",
        stages=(
            StageSpec(stage_id="image_preparation", capability="image_preparation"),
            StageSpec(
                stage_id="region_discovery",
                capability="region_discovery",
                inputs={"image": "image_preparation"},
                backend_id="region_discovery/canonical",
            ),
            StageSpec(
                stage_id="region_refinement",
                capability="region_discovery",
                inputs={"image": "image_preparation"},
                backend_id="region_refinement/experimental",
                optional=True,
            ),
            StageSpec(
                stage_id="region_feature_extraction",
                capability="feature_extractor",
                inputs={"image": "image_preparation", "regions": "region_refinement"},
                backend_id="region_feature_extractor/canonical",
                feature_scope=FeatureScope.REGION,
            ),
        ),
    )

    resolved = resolve_pipeline(
        preset,
        backend_factories={
            "region_discovery": lambda _p: FakeRegionDiscovery(),
            "region_refinement": lambda _p: _FakeRegionRefinement(),
            "region_feature_extraction": lambda _p: FakeRegionFeatureExtractor(),
        },
    )
    stages = resolved.build_stage_graph({"image_preparation": _prepared_image()})
    outcomes = execute_stage_graph(stages)
    by_id = {outcome.stage_id: outcome for outcome in outcomes}

    region_features = by_id["region_feature_extraction"].output
    assert by_id["region_refinement"].output is not None
    assert isinstance(region_features, Sequence)
    assert len(region_features) == 1


def test_configuration_digest_is_stable_and_changes_with_backend_identity() -> None:
    resolved_a = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    resolved_b = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    assert resolved_a.configuration_digest() == resolved_b.configuration_digest()

    class _DifferentRegionDiscovery(FakeRegionDiscovery):
        def backend_provenance(self) -> BackendProvenance:
            return replace(super().backend_provenance(), backend_id="a_different_backend")

    resolved_c = resolve_pipeline(
        CANONICAL_PRESET_V1,
        backend_factories=_canonical_backend_factories(
            region_discovery=_DifferentRegionDiscovery()
        ),
    )
    assert resolved_c.configuration_digest() != resolved_a.configuration_digest()


def _discovery_digest(
    pass_config: DiscoveryPassConfig | None = None,
    normalization_config: NormalizationConfig | None = None,
) -> str:
    """Digest the canonical preset over a real discovery adapter; its SDK runtime never runs."""
    discovery = Sam2RegionDiscovery(
        config=Sam2Config(checkpoint="facebook/sam2-hiera-large", model_version="2.1"),
        runtime=object(),  # type: ignore[arg-type]
        pass_images=object(),  # type: ignore[arg-type]
        pass_config=pass_config,
        normalization_config=normalization_config,
    )
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1,
        backend_factories=_canonical_backend_factories(region_discovery=discovery),
    )
    return resolved.configuration_digest()


def test_configuration_digest_covers_the_effective_discovery_configuration() -> None:
    baseline = _discovery_digest()

    assert _discovery_digest() == baseline
    assert _discovery_digest(DiscoveryPassConfig(), NormalizationConfig()) == baseline
    assert _discovery_digest(DiscoveryPassConfig(max_candidates_per_pass=3)) != baseline
    assert _discovery_digest(normalization_config=NormalizationConfig(maximum_regions=3)) != (
        baseline
    )


def test_configuration_digest_changes_with_a_discovery_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _discovery_digest()

    replace_default(monkeypatch, NormalizationConfig, "duplicate_iou_threshold", 0.5)

    assert _discovery_digest() != baseline


def test_configuration_digest_changes_with_the_mask_remap_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _discovery_digest()

    monkeypatch.setattr(discovery_module, "MASK_REMAP_POLICY_ID", "nearest_pixel_corner_v0")

    assert _discovery_digest() != baseline


def test_pipeline_preset_encode_decode_round_trip() -> None:
    decoded = decode_pipeline_preset(encode_pipeline_preset(CANONICAL_PRESET_V1))
    assert decoded == CANONICAL_PRESET_V1


def test_known_capabilities_matches_canonical_preset_backend_stages() -> None:
    used = {
        stage.capability for stage in CANONICAL_PRESET_V1.stages if stage.backend_id is not None
    }
    assert used <= KNOWN_CAPABILITIES


# Entradas exigidas por capability e o rótulo do estágio-fonte que as produz; o valor semeado
# não importa, porque o tipo do backend é conferido antes de qualquer entrada.
_CAPABILITY_INPUT_SOURCES: dict[str, dict[str, str]] = {
    "region_discovery": {"image": "image_preparation"},
    "region_grounding": {"image": "image_preparation"},
    "region_refinement": {"image": "image_preparation", "groundings": "region_grounding"},
    "feature_extractor": {"image": "image_preparation"},
    "feature_resolution_enhancement": {"dense_map": "dense_feature_map_source"},
    "scene_interpretation": {"image": "image_preparation"},
    "region_interpretation": {"image": "image_preparation", "regions": "region_discovery"},
    "semantic_interpreter": {"request": "semantic_request"},
    "semantic_scorer": {"claims": "semantic_claims", "features": "visual_features"},
    "dense_region_pooling": {
        "image": "image_preparation",
        "regions": "region_discovery",
        "dense_features": "feature_extractor",
    },
}


@pytest.mark.parametrize("capability", sorted(KNOWN_CAPABILITIES))
def test_a_backend_of_the_wrong_type_is_a_semantic_error_not_an_assertion(
    capability: str,
) -> None:
    """VP-13: ``assert isinstance`` vanishes under ``python -O`` and says nothing when it fires."""
    sources = _CAPABILITY_INPUT_SOURCES[capability]
    preset = PipelinePreset(
        preset_id="wrong-backend/1",
        stages=(
            *(
                StageSpec(stage_id=f"{name}_source", capability=label)
                for name, label in sources.items()
            ),
            StageSpec(
                stage_id="stage",
                capability=capability,
                inputs={name: f"{name}_source" for name in sources},
                backend_id="not-a-backend",
                feature_scope=FeatureScope.DENSE if capability == "feature_extractor" else None,
            ),
        ),
    )
    resolved = resolve_pipeline(preset, backend_factories={"stage": lambda _p: object()})
    seeds = {f"{name}_source": None for name in sources}
    stage = next(
        definition
        for definition in resolved.build_stage_graph(seeds)
        if definition.stage_id == "stage"
    )

    with pytest.raises(TypeError, match=rf"^{capability} backend must implement .+, got object$"):
        stage.run(seeds)


def test_a_stage_input_of_the_wrong_type_is_a_semantic_error_not_an_assertion() -> None:
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1, backend_factories=_canonical_backend_factories()
    )
    stages = resolved.build_stage_graph({"image_preparation": _prepared_image()})
    stage = next(definition for definition in stages if definition.stage_id == "region_discovery")

    with pytest.raises(
        TypeError, match=r"^region_discovery input 'image' must be PreparedImage, got str$"
    ):
        stage.run({"image_preparation": "frame-0124.jpg"})


def test_a_backend_without_provenance_is_named_by_its_stage() -> None:
    resolved = resolve_pipeline(
        CANONICAL_PRESET_V1,
        backend_factories=_canonical_backend_factories(region_discovery=object()),
    )

    with pytest.raises(
        TypeError, match=r"^backend of stage 'region_discovery' does not report backend_provenance"
    ):
        resolved.configuration_digest()
