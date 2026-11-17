"""The CLIP text tower projects claim hypotheses into the joint space of the region features.

The model is never loaded here: the adapter runs over a fake text runtime, and the bundled
Transformers runtime over fake SDK modules, so the loading and batching code paths are real
while the weights are not.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_scoring.adapters import (
    ClipRegionSemanticScorer,
)
from contextmap.semantic_scoring.adapters import clip_text as clip_text_module
from contextmap.semantic_scoring.adapters.clip_text import (
    ClipTextConfig,
    ClipTextDependencyError,
    ClipTextEncoder,
    ClipTextInferenceError,
    HuggingFaceClipTextRuntime,
    clip_region_scorer_provenance,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage, VisualEvidenceRunId
from contextmap.visual_perception.visual_features import (
    FeatureId,
    FeatureScope,
    VisualFeature,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.visual_features.adapters.clip import (
    ClipConfig,
    ClipNativeOutput,
    ClipView,
    ClipVisualFeatureBackend,
)

CHECKPOINT = "openai/clip-vit-base-patch32"
REVISION = "c" * 40
TEMPLATE = "a photo of a {hypothesis}"


class FakeTextRuntime:
    """Return one fixed row per text, in the order asked, and record every call."""

    def __init__(self, rows: dict[str, list[float]], *, dtype: str = "float32") -> None:
        self._rows = rows
        self._dtype = dtype
        self.calls: list[tuple[str, ...]] = []

    def projection_dimension(self) -> int:
        return len(next(iter(self._rows.values())))

    def encode(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        self.calls.append(tuple(texts))
        return np.asarray([self._rows[text] for text in texts], dtype=self._dtype)


def _config(**changes: Any) -> ClipTextConfig:
    values: dict[str, Any] = {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "text_template": TEMPLATE,
    }
    values.update(changes)
    return ClipTextConfig(**values)


class _Sink:
    def __init__(self) -> None:
        self.arrays: dict[FeatureId, np.ndarray[Any, Any]] = {}

    def add_feature_payload(
        self, feature: VisualFeature, source_observation_id: SourceObservationId, array: Any
    ) -> None:
        self.arrays[feature.feature_id] = array.copy()

    def load(
        self, source_observation_id: SourceObservationId, feature_id: FeatureId
    ) -> np.ndarray[Any, Any]:
        return self.arrays[feature_id]


class _FakeClipImageRuntime:
    def encode(self, image: PreparedImage, views: Sequence[ClipView]) -> ClipNativeOutput:
        return ClipNativeOutput(
            array=np.asarray([[3.0, 4.0, 0.0]] * len(views), dtype=np.float32),
            elapsed_seconds=0.0,
        )


def _region_feature(sink: _Sink, *, revision: str = REVISION) -> VisualFeature:
    """Extract the CLIP crop feature of one region, exactly as the perception stage does."""
    backend = ClipVisualFeatureBackend(
        config=ClipConfig(checkpoint=CHECKPOINT, revision=revision, scope=FeatureScope.REGION),
        run_id=VisualEvidenceRunId("run-0001"),
        feature_stage_id="region_feature_extraction",
        payload_sink=sink,
        runtime=_FakeClipImageRuntime(),
    )
    image = PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="prepared/frame-0001.png",
        width=10,
        height=10,
    )
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=1, y=1, width=4, height=4),
        provenance=BackendProvenance(
            backend_id="fake", capability="region_discovery", provider="f", model="f", version="1"
        ),
    )
    (feature,) = backend.extract(image, (region,))
    return feature


def _claim(result_id: SemanticInterpretationRunId, hypothesis: str) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId(f"claim-{hypothesis}"),
        source_observation_id=SourceObservationId("frame-0001"),
        semantic_interpretation_run_id=result_id,
        hypothesis=hypothesis,
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake",
                capability="semantic_interpreter",
                provider="f",
                model="f",
                version="1",
            ),
            task_identity="region",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=RegionId("region-0001"),
    )


def test_hypotheses_land_in_the_exact_space_of_the_clip_region_features() -> None:
    sink = _Sink()
    feature = _region_feature(sink)
    runtime = FakeTextRuntime(
        {"a photo of a chair": [3.0, 4.0, 0.0], "a photo of a door": [0.0, 0.0, 2.0]}
    )
    encoder = ClipTextEncoder(config=_config(), runtime=runtime)

    batch = encoder.encode(("chair", "door"))

    assert embedding_space_fingerprint(encoder.embedding_space()) == feature.embedding_space_id
    assert batch.embedding_space_id == feature.embedding_space_id
    assert batch.normalization == "l2"
    np.testing.assert_allclose(batch.values, [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], rtol=1e-6)
    # O template é aplicado a cada hipótese, na ordem pedida, antes da torre de texto.
    assert runtime.calls == [("a photo of a chair", "a photo of a door")]


def test_the_region_scorer_compares_text_and_region_crop_in_one_space() -> None:
    sink = _Sink()
    feature = _region_feature(sink)
    encoder = ClipTextEncoder(
        config=_config(), runtime=FakeTextRuntime({"a photo of a chair": [3.0, 4.0, 0.0]})
    )
    scorer = ClipRegionSemanticScorer(
        provenance=clip_region_scorer_provenance(_config()),
        embedding_space_id=embedding_space_fingerprint(encoder.embedding_space()),
        payloads=sink,
        text_encoder=encoder,
    )
    result_id = SemanticInterpretationRunId("result-0001")

    (score,) = scorer.score((_claim(result_id, "chair"),), (feature,))

    assert score.value == pytest.approx(1.0)
    assert score.embedding_space_id == feature.embedding_space_id


def test_a_feature_of_another_revision_is_ignored_never_scored_across_spaces() -> None:
    sink = _Sink()
    feature = _region_feature(sink)
    other = _config(revision="d" * 40)
    encoder = ClipTextEncoder(
        config=other, runtime=FakeTextRuntime({"a photo of a chair": [3.0, 4.0, 0.0]})
    )
    assert embedding_space_fingerprint(encoder.embedding_space()) != feature.embedding_space_id
    scorer = ClipRegionSemanticScorer(
        provenance=clip_region_scorer_provenance(other),
        embedding_space_id=embedding_space_fingerprint(encoder.embedding_space()),
        payloads=sink,
        text_encoder=encoder,
    )

    # A feature de outro espaço não é candidata deste scorer: fica sem score, nunca pontuada
    # entre espaços.
    assert (
        scorer.score((_claim(SemanticInterpretationRunId("result-0001"), "chair"),), (feature,))
        == ()
    )


def test_the_scorer_identity_names_the_text_tower_and_every_setting_that_changes_a_value() -> None:
    provenance = clip_region_scorer_provenance(_config())

    assert provenance.capability == "semantic_scorer"
    assert (provenance.model, provenance.version) == (CHECKPOINT, REVISION)
    for changed in (
        _config(text_template="{hypothesis}"),
        _config(precision="float16", device="cuda"),
        _config(max_batch_size=4),
    ):
        fingerprint = clip_region_scorer_provenance(changed).configuration_fingerprint
        assert fingerprint != provenance.configuration_fingerprint


@pytest.mark.parametrize(
    "template",
    ["a photo", "{hypothesis} and {hypothesis}", "{label}", "{hypothesis!r}", "{hypothesis", ""],
)
def test_the_text_template_holds_the_hypothesis_exactly_once(template: str) -> None:
    with pytest.raises(ValueError, match="text_template"):
        _config(text_template=template)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", ""),
        ("revision", "main"),
        ("device", "tpu"),
        ("precision", "bfloat16"),
        ("max_batch_size", 0),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_a_runtime_that_breaks_the_declared_output_is_refused() -> None:
    rows = {"a photo of a chair": [3.0, 4.0, 0.0]}
    wrong_dtype = ClipTextEncoder(config=_config(), runtime=FakeTextRuntime(rows, dtype="float64"))
    with pytest.raises(ClipTextInferenceError, match="dtype"):
        wrong_dtype.encode(("chair",))

    class _Short(FakeTextRuntime):
        def encode(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
            return super().encode(texts)[:0]

    with pytest.raises(ClipTextInferenceError, match="embeddings for 1"):
        ClipTextEncoder(config=_config(), runtime=_Short(rows)).encode(("chair",))
    zero = ClipTextEncoder(
        config=_config(), runtime=FakeTextRuntime({"a photo of a chair": [0.0, 0.0, 0.0]})
    )
    with pytest.raises(ClipTextInferenceError, match="zero-norm"):
        zero.encode(("chair",))


# ------------------------------------------------------------- bundled Transformers runtime


class _SdkTensor:
    """NumPy-backed stand-in for the torch tensors the runtime moves and detaches."""

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def to(self, *args: object, **kwargs: object) -> _SdkTensor:
        return self

    def detach(self) -> _SdkTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.array


class _Tokenizer:
    """Tokenize each text as its word lengths, padded to the longest text of the batch."""

    def __init__(self) -> None:
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, _SdkTensor]:
        self.kwargs.append(kwargs)
        tokens = [[len(word) for word in text.split()] for text in texts]
        width = max(len(row) for row in tokens)
        padded = np.array([row + [0] * (width - len(row)) for row in tokens], dtype=np.int64)
        return {"input_ids": _SdkTensor(padded), "attention_mask": _SdkTensor(padded > 0)}


class _TextModel:
    """Embed a text as (number of tokens, sum of token ids): every row identifies its text."""

    def __init__(self, *, max_position_embeddings: int = 77) -> None:
        self.config = SimpleNamespace(
            projection_dim=2, max_position_embeddings=max_position_embeddings
        )
        self.batch_sizes: list[int] = []
        self.loaded_with: dict[str, object] = {}

    def to(self, device: str) -> _TextModel:
        return self

    def eval(self) -> _TextModel:
        return self

    def __call__(self, *, input_ids: _SdkTensor, attention_mask: _SdkTensor) -> SimpleNamespace:
        self.batch_sizes.append(len(input_ids.array))
        rows = np.stack(
            [attention_mask.array.sum(axis=1), input_ids.array.sum(axis=1)], axis=1
        ).astype(np.float32)
        return SimpleNamespace(text_embeds=_SdkTensor(rows))


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, model: _TextModel) -> _Tokenizer:
    tokenizer = _Tokenizer()

    def load_model(*args: object, **kwargs: object) -> _TextModel:
        model.loaded_with = {"args": args, **kwargs}
        return model

    sdks = {
        "torch": SimpleNamespace(
            float32="torch.float32",
            float16="torch.float16",
            inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            CLIPTextModelWithProjection=SimpleNamespace(from_pretrained=load_model),
        ),
    }
    monkeypatch.setattr(
        clip_text_module, "importlib", SimpleNamespace(import_module=sdks.__getitem__)
    )
    return tokenizer


def test_the_bundled_runtime_loads_only_the_text_tower_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TextModel()
    tokenizer = _install_fake_sdk(monkeypatch, model)
    runtime = HuggingFaceClipTextRuntime(config=_config(max_batch_size=2))

    assert runtime.projection_dimension() == 2
    array = runtime.encode(("a b", "abc", "a bb ccc", "dd"))

    assert model.loaded_with["args"] == (CHECKPOINT,)
    assert model.loaded_with["revision"] == REVISION
    assert model.loaded_with["local_files_only"] is True
    assert model.batch_sizes == [2, 2]
    np.testing.assert_array_equal(array, [[2, 2], [1, 3], [3, 6], [1, 2]])
    # Nenhuma hipótese é truncada em silêncio: o tokenizer é chamado sem truncamento.
    assert all(call["truncation"] is False for call in tokenizer.kwargs)


def test_a_text_longer_than_the_tower_accepts_is_refused_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch, _TextModel(max_position_embeddings=2))
    runtime = HuggingFaceClipTextRuntime(config=_config())

    with pytest.raises(ClipTextInferenceError, match="3 tokens"):
        runtime.encode(("a bb ccc",))


def test_missing_sdk_dependencies_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        clip_text_module, "importlib", SimpleNamespace(import_module=missing_dependency)
    )
    runtime = HuggingFaceClipTextRuntime(config=_config())

    with pytest.raises(ClipTextDependencyError, match="torch and transformers"):
        runtime.encode(("chair",))
