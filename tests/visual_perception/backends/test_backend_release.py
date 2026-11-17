"""Backends release their model on request, so a stage-major run fits a small GPU (#689).

A backend forwards ``release()`` to its runtime. A lazily loading runtime drops its weights and
loads again on the next call; a runtime handed an already loaded model drops it and refuses
further calls, instead of failing deep inside the SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from contextmap.model_runtime import model_release
from contextmap.region_perception.adapters.sam2 import (
    Sam2AutomaticMaskRuntime,
    Sam2Config,
    Sam2RegionDiscovery,
)
from contextmap.region_perception.adapters.sam3 import (
    Sam3Config,
    Sam3ImageProcessorRuntime,
    Sam3RegionDiscovery,
)
from contextmap.semantic_interpretation.adapters.qwen import (
    HuggingFaceQwenRuntime,
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)
from contextmap.visual_perception import ReleasableBackend, VisualEvidenceRunId
from contextmap.visual_perception.visual_features import FeatureScope
from contextmap.visual_perception.visual_features.adapters.clip import (
    ClipConfig,
    ClipVisualFeatureBackend,
    HuggingFaceClipRuntime,
)
from contextmap.visual_perception.visual_features.adapters.dinov3 import (
    DinoV3Config,
    DinoV3DenseFeatureBackend,
    HuggingFaceDinoV3Runtime,
)

_REVISION = "c" * 40
_RUN = VisualEvidenceRunId("run-0001")


class _Runtime:
    """A runtime that only records being released."""

    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


def _sam2(runtime: Any) -> Any:
    return Sam2RegionDiscovery(
        config=Sam2Config(checkpoint="facebook/sam2.1-hiera-large", model_version="2.1"),
        runtime=runtime,
        pass_images=object(),  # type: ignore[arg-type]
    )


def _sam3(runtime: Any) -> Any:
    return Sam3RegionDiscovery(
        config=Sam3Config(checkpoint="facebook/sam3", model_version="3.0"),
        runtime=runtime,
        pass_images=object(),  # type: ignore[arg-type]
    )


def _dinov3(runtime: Any) -> Any:
    return DinoV3DenseFeatureBackend(
        config=DinoV3Config(
            checkpoint="facebook/dinov3-vitb16-pretrain-lvd1689m", revision=_REVISION
        ),
        run_id=_RUN,
        feature_stage_id="dense_feature_extraction",
        source_artifact_id=str(_RUN),
        payload_sink=object(),  # type: ignore[arg-type]
        runtime=runtime,
    )


def _clip(runtime: Any) -> Any:
    return ClipVisualFeatureBackend(
        config=ClipConfig(
            checkpoint="openai/clip-vit-large-patch14",
            revision=_REVISION,
            scope=FeatureScope.REGION,
        ),
        run_id=_RUN,
        feature_stage_id="region_feature_extraction",
        payload_sink=object(),  # type: ignore[arg-type]
        runtime=runtime,
    )


def _qwen(runtime: Any) -> Any:
    return QwenSemanticInterpreter(
        config=QwenSemanticConfig(
            model="Qwen/Qwen3-VL-8B-Instruct",
            revision=_REVISION,
            device="cpu",
            precision="float32",
            max_new_tokens=8,
            temperature=0.0,
        ),
        runtime=runtime,
        view_root=Path("."),
    )


@pytest.mark.parametrize("build", [_sam2, _sam3, _dinov3, _clip, _qwen])
def test_a_backend_forwards_release_to_its_runtime(build: Any) -> None:
    runtime = _Runtime()
    backend = build(runtime)

    assert isinstance(backend, ReleasableBackend)
    backend.release()

    assert runtime.released == 1


@pytest.mark.parametrize("build", [_sam2, _sam3, _dinov3, _clip, _qwen])
def test_a_runtime_that_cannot_release_is_left_alone(build: Any) -> None:
    build(object()).release()


@pytest.mark.parametrize(
    "runtime",
    [
        lambda: HuggingFaceDinoV3Runtime(
            config=DinoV3Config(
                checkpoint="facebook/dinov3-vitb16-pretrain-lvd1689m", revision=_REVISION
            ),
            prepared_image_root=Path("."),
        ),
        lambda: HuggingFaceClipRuntime(
            config=ClipConfig(
                checkpoint="openai/clip-vit-large-patch14",
                revision=_REVISION,
                scope=FeatureScope.REGION,
            ),
            prepared_image_root=Path("."),
        ),
        lambda: HuggingFaceQwenRuntime(
            config=QwenSemanticConfig(
                model="Qwen/Qwen3-VL-8B-Instruct",
                revision=_REVISION,
                device="cpu",
                precision="float32",
                max_new_tokens=8,
                temperature=0.0,
            )
        ),
    ],
)
def test_a_lazy_runtime_drops_its_weights_and_frees_cuda_memory(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    freed: list[bool] = []
    monkeypatch.setattr(model_release, "free_cuda_memory", lambda: freed.append(True))
    built = runtime()
    built._model, built._processor = object(), object()

    built.release()

    assert built._model is None
    assert built._processor is None
    assert freed == [True]


class _NoMaskGenerator:
    """An automatic-mask generator that proposes nothing; it is never called after release."""

    def generate(self, image: object) -> list[dict[str, object]]:
        return []


def test_a_loaded_sam2_runtime_refuses_inference_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_release, "free_cuda_memory", lambda: None)
    runtime = Sam2AutomaticMaskRuntime(
        model=object(),
        mask_generator=_NoMaskGenerator(),
        config_digest="sha256:x",
    )

    runtime.release()

    with pytest.raises(RuntimeError, match="released"):
        runtime.predict(object(), object(), object())  # type: ignore[arg-type]


def test_a_loaded_sam3_runtime_refuses_inference_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_release, "free_cuda_memory", lambda: None)
    runtime = Sam3ImageProcessorRuntime(processor=object())  # type: ignore[arg-type]

    runtime.release()

    with pytest.raises(RuntimeError, match="released"):
        runtime.predict(object(), object(), object())  # type: ignore[arg-type]
