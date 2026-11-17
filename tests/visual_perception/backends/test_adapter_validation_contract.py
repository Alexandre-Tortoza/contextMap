"""Validation contract shared by the adapters that implement the same Visual Perception port.

Implementations of one port must reject the same invalid execution settings and the same invalid
native values, whatever SDK sits behind them (AGENTS §12, LSP; issue #617, VPB-10).
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    AuditedRegionDiscovery,
    BoundingBox,
    DiscoveryInput,
    DiscoveryPass,
    DiscoveryPassConfig,
    NormalizationConfig,
    PassKind,
    RegionDiscoveryIdentity,
)
from contextmap.region_perception.adapters import florence2, sam2, sam3
from contextmap.semantic_interpretation.adapters import (
    eagle2_5,
    florence2_semantic,
    qwen,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
)
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_features import FeatureScope
from contextmap.visual_perception.visual_features.adapters import (
    alphaclip,
    clip,
    dinov2,
    dinov3,
    siglip2,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _refuse_load(*args: object, **kwargs: object) -> object:
    raise OSError("weights must not load when the configured device cannot be used")


class _UnloadableTransformers:
    """Every checkpoint load fails, so a device error has to come before the weights."""

    def __getattr__(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(from_pretrained=_refuse_load)


def _torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        float32="torch.float32",
        float16="torch.float16",
        bfloat16="torch.bfloat16",
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def _install_sdks(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, *, cuda: bool, mps: bool
) -> None:
    sdks = {
        "torch": _torch(cuda=cuda, mps=mps),
        "transformers": _UnloadableTransformers(),
        "alpha_clip": SimpleNamespace(load=_refuse_load),
        "PIL.Image": SimpleNamespace(),
    }
    monkeypatch.setattr(module, "importlib", SimpleNamespace(import_module=sdks.__getitem__))


LoadingRuntime = tuple[ModuleType, Callable[[], None], type[Exception]]


def _local_weight_runtime(adapter: str, device: str, root: Path) -> LoadingRuntime:
    """Return the module, the weight-loading step, and the device error of one runtime."""
    if adapter == "dinov2":
        dinov2_runtime = dinov2.HuggingFaceDinoV2Runtime(
            config=dinov2.DinoV2Config(
                checkpoint="facebook/dinov2-base", revision=REVISION, device=device
            ),
            prepared_image_root=root,
        )
        return dinov2, dinov2_runtime._ensure_loaded, dinov2.DinoV2DeviceError
    if adapter == "dinov3":
        dinov3_runtime = dinov3.HuggingFaceDinoV3Runtime(
            config=dinov3.DinoV3Config(
                checkpoint="facebook/dinov3-vits16", revision=REVISION, device=device
            ),
            prepared_image_root=root,
        )
        return dinov3, dinov3_runtime._ensure_loaded, dinov3.DinoV3DeviceError
    if adapter == "clip":
        clip_runtime = clip.HuggingFaceClipRuntime(
            config=clip.ClipConfig(
                checkpoint="openai/clip-vit-large-patch14",
                revision=REVISION,
                scope=FeatureScope.REGION,
                device=device,
            ),
            prepared_image_root=root,
        )
        return clip, clip_runtime._ensure_loaded, clip.ClipDeviceError
    if adapter == "alphaclip":
        alphaclip_runtime = alphaclip.OfficialAlphaClipRuntime(
            config=alphaclip.AlphaClipConfig(
                model_name="ViT-L/14",
                base_checkpoint_path="base.pt",
                alpha_checkpoint_path="alpha.pth",
                checkpoint_fingerprint="sha256:abc",
                device=device,
            ),
            prepared_image_root=root,
            checkpoint_root=root,
        )
        return alphaclip, alphaclip_runtime._ensure_loaded, alphaclip.AlphaClipDeviceError
    if adapter == "qwen":
        qwen_runtime = qwen.HuggingFaceQwenRuntime(
            config=qwen.QwenSemanticConfig(
                model="Qwen/Qwen3-VL-4B-Instruct",
                revision=REVISION,
                device=device,
                precision="float32",
                max_new_tokens=8,
                temperature=0.0,
            ),
        )
        return qwen, qwen_runtime.load, qwen.QwenDeviceError
    if adapter == "eagle2_5":
        eagle_runtime = eagle2_5.HuggingFaceEagleRuntime(
            config=eagle2_5.EagleSemanticConfig(
                model="nvidia/Eagle2.5-8B",
                revision=REVISION,
                device=device,
                precision="float32",
                max_new_tokens=8,
                temperature=0.0,
                max_dynamic_tiles=4,
            ),
        )
        return eagle2_5, eagle_runtime.load, eagle2_5.EagleDeviceError
    if adapter == "siglip2":
        siglip2_runtime = siglip2.HuggingFaceSiglip2Runtime(
            config=siglip2.Siglip2Config(
                checkpoint="google/siglip2-base-patch16-224",
                revision=REVISION,
                scope=FeatureScope.DENSE,
                input_size=224,
                device=device,
            ),
            prepared_image_root=root,
        )
        return siglip2, siglip2_runtime._ensure_loaded, siglip2.Siglip2DeviceError
    florence2_runtime = florence2_semantic.HuggingFaceFlorence2SemanticRuntime(
        config=florence2_semantic.Florence2SemanticConfig(
            checkpoint="florence-community/Florence-2-large",
            revision=REVISION,
            task="<CAPTION>",
            supported_modes=frozenset({SemanticInterpretationMode.SCENE}),
            device=device,
            precision="float32",
            max_new_tokens=8,
            temperature=0.0,
        ),
    )
    return florence2_semantic, florence2_runtime.load, florence2_semantic.Florence2DeviceError


LOCAL_WEIGHT_RUNTIMES = (
    "dinov2",
    "dinov3",
    "clip",
    "alphaclip",
    "siglip2",
    "qwen",
    "florence2_semantic",
    "eagle2_5",
)
SEMANTIC_RUNTIMES = ("qwen", "florence2_semantic", "eagle2_5")


@pytest.mark.parametrize("adapter", LOCAL_WEIGHT_RUNTIMES)
def test_an_unavailable_mps_device_is_rejected_before_the_weights_load(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, load, device_error = _local_weight_runtime(adapter, "mps", tmp_path)
    _install_sdks(monkeypatch, module, cuda=True, mps=False)

    with pytest.raises(device_error, match="MPS device is unavailable"):
        load()


@pytest.mark.parametrize("adapter", SEMANTIC_RUNTIMES)
def test_semantic_runtimes_accept_only_cpu_cuda_or_mps_device_types(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, load, device_error = _local_weight_runtime(adapter, "tpu", tmp_path)
    _install_sdks(monkeypatch, module, cuda=True, mps=True)

    with pytest.raises(device_error, match="device type must be one of: cpu, cuda, mps"):
        load()


@pytest.mark.parametrize("adapter", SEMANTIC_RUNTIMES)
def test_semantic_runtimes_keep_accepting_an_indexed_device(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, load, device_error = _local_weight_runtime(adapter, "cuda:1", tmp_path)
    _install_sdks(monkeypatch, module, cuda=True, mps=False)

    # A validação de device passa: a falha é a carga recusada pelo transformers fake.
    with pytest.raises(Exception, match="weights must not load") as caught:
        load()

    assert not isinstance(caught.value, device_error)


def _sam2_score(score: object) -> object:
    record = {
        "segmentation": [[True]],
        "bbox": [0.0, 0.0, 0.0, 0.0],
        "area": 1,
        "predicted_iou": score,
        "stability_score": 0.9,
    }
    return sam2._parse_automatic_mask_record(record, index=0, width=1, height=1)


def _sam3_score(score: object) -> object:
    output = {"boxes": [[0.0, 0.0, 1.0, 1.0]], "scores": [score], "masks": [[[0.9]]]}
    return sam3._parse_image_processor_output(
        output, width=1, height=1, mask_threshold=0.5, prompt="floor", prompt_index=0
    )


def _florence2_score(score: object) -> object:
    result = {"bboxes": [[0.0, 0.0, 1.0, 1.0]], "scores": [score]}
    return florence2._parse_task_result(result, width=1, height=1)


DISCOVERY_PARSERS: dict[str, Callable[[object], Any]] = {
    "sam2": _sam2_score,
    "sam3": _sam3_score,
    "florence2": _florence2_score,
}


@pytest.mark.parametrize("adapter", sorted(DISCOVERY_PARSERS))
@pytest.mark.parametrize("score", [True, False])
def test_discovery_parsers_reject_a_boolean_native_score(adapter: str, score: bool) -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        DISCOVERY_PARSERS[adapter](score)


@pytest.mark.parametrize("adapter", sorted(DISCOVERY_PARSERS))
def test_discovery_parsers_accept_an_integer_native_score(adapter: str) -> None:
    DISCOVERY_PARSERS[adapter](1)


def _discovery_adapter(
    adapter: str, *, runtime: Any = None, pass_images: Any = None, **configs: Any
) -> Any:
    """Build one Region Discovery adapter; unless given, its SDK runtime never runs here."""
    parts = {
        "runtime": object() if runtime is None else runtime,
        "pass_images": object() if pass_images is None else pass_images,
        **configs,
    }
    if adapter == "sam2":
        return sam2.Sam2RegionDiscovery(
            config=sam2.Sam2Config(checkpoint="facebook/sam2-hiera-large", model_version="2.1"),
            **parts,
        )
    if adapter == "sam3":
        return sam3.Sam3RegionDiscovery(
            config=sam3.Sam3Config(checkpoint="facebook/sam3", model_version="3.0"), **parts
        )
    return florence2.Florence2RegionDiscovery(
        config=florence2.Florence2Config(
            checkpoint="microsoft/Florence-2-large",
            model_version="1.0",
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            prompt="visible regions",
        ),
        **parts,
    )


class _UnreachableRuntime:
    """A discovery runtime whose inference must never be reached."""

    def __init__(self) -> None:
        self.images: list[object] = []

    def predict(self, discovery_input: Any, image: object, config: Any) -> Any:
        self.images.append(image)
        raise AssertionError("the runtime was reached with an image the pass does not have")


def _full_frame_input(width: int = 4, height: int = 3) -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-0001"),
            payload_reference="frame-0001.png",
            width=width,
            height=height,
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=width, y_max=height),
        ),
        perception_run_id="run-0001",
        perception_result_id="run-0001--frame-0001",
    )


@pytest.mark.parametrize("adapter", ["sam2", "sam3", "florence2"])
def test_discovery_adapters_refuse_a_materialized_image_the_pass_does_not_have(
    adapter: str,
) -> None:
    """#661: the model runtime only ever receives the pass's own, dimension-checked pixels."""
    runtime = _UnreachableRuntime()
    backend = _discovery_adapter(
        adapter,
        runtime=runtime,
        pass_images=lambda discovery_input: np.zeros((3, 5, 3), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="materialized discovery image dimensions"):
        backend.discover_candidates(_full_frame_input())
    assert runtime.images == []


@pytest.mark.parametrize("adapter", ["sam2", "sam3", "florence2"])
def test_discovery_adapters_declare_their_provenance_before_they_are_built(adapter: str) -> None:
    """#661: the composition root names a run's discovery before the run's images exist."""
    backend = _discovery_adapter(adapter)
    adapter_type = type(backend)

    assert adapter_type.provenance_for(backend._config) == backend.backend_provenance()


@pytest.mark.parametrize("adapter", ["sam2", "sam3", "florence2"])
def test_discovery_adapters_report_the_effective_discovery_they_apply(adapter: str) -> None:
    """#636: the identity names the configurations the adapter applies, defaults resolved."""
    passes = DiscoveryPassConfig(max_candidates_per_pass=3)
    normalization = NormalizationConfig(maximum_regions=5)

    configured = _discovery_adapter(adapter, pass_config=passes, normalization_config=normalization)

    assert isinstance(configured, AuditedRegionDiscovery)
    assert configured.discovery_identity() == RegionDiscoveryIdentity.effective(
        passes, normalization
    )
    assert _discovery_adapter(adapter).discovery_identity() == RegionDiscoveryIdentity.effective()
