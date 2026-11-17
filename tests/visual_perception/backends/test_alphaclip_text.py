"""The AlphaCLIP scorer projects hypotheses with the base CLIP text tower into the AlphaCLIP space.

No weights are loaded: the adapter runs over a fake text runtime and the official-package runtime
over fake ``torch``/``alpha_clip`` modules, with real checkpoint files whose fingerprint is
verified exactly as the region extractor verifies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.model_runtime import combined_checkpoint_fingerprint
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_scoring.adapters import (
    AlphaClipSemanticScorer,
)
from contextmap.semantic_scoring.adapters import alphaclip_text as alphaclip_text_module
from contextmap.semantic_scoring.adapters.alphaclip_text import (
    AlphaClipTextConfig,
    AlphaClipTextDependencyError,
    AlphaClipTextEncoder,
    AlphaClipTextInferenceError,
    AlphaClipTextModelLoadError,
    OfficialAlphaClipTextRuntime,
    alphaclip_region_scorer_provenance,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage, VisualEvidenceRunId
from contextmap.visual_perception.visual_features import (
    FeatureId,
    VisualFeature,
    embedding_space_fingerprint,
)
from contextmap.visual_perception.visual_features.adapters.alphaclip import (
    AlphaClipConfig,
    AlphaClipNativeOutput,
    AlphaClipRegionFeatureBackend,
    AlphaClipRequest,
)

FINGERPRINT = "sha256:" + "a" * 64


class FakeTextRuntime:
    def __init__(self, rows: dict[str, list[float]]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, ...]] = []

    def projection_dimension(self) -> int:
        return len(next(iter(self._rows.values())))

    def encode(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        self.calls.append(tuple(texts))
        return np.asarray([self._rows[text] for text in texts], dtype=np.float32)


def _config(**changes: Any) -> AlphaClipTextConfig:
    values: dict[str, Any] = {
        "model_name": "ViT-L/14",
        "base_checkpoint_path": "base.pt",
        "alpha_checkpoint_path": "alpha.pth",
        "checkpoint_fingerprint": FINGERPRINT,
        "text_template": "{hypothesis}",
    }
    values.update(changes)
    return AlphaClipTextConfig(**values)


class _Sink:
    def __init__(self) -> None:
        self.arrays: dict[FeatureId, np.ndarray[Any, Any]] = {}

    def add_feature_payload(
        self, feature: VisualFeature, source_observation_id: SourceObservationId, array: Any
    ) -> None:
        self.arrays[feature.feature_id] = array.copy()

    def add_ineligible_region(self, record: object) -> None:
        raise AssertionError(f"no region of this test is box-only: {record!r}")

    def load(
        self, source_observation_id: SourceObservationId, feature_id: FeatureId
    ) -> np.ndarray[Any, Any]:
        return self.arrays[feature_id]


class _FakeImageRuntime:
    def encode(
        self, image: PreparedImage, requests: Sequence[AlphaClipRequest]
    ) -> AlphaClipNativeOutput:
        return AlphaClipNativeOutput(
            array=np.asarray([[0.0, 5.0]] * len(requests), dtype=np.float32),
            elapsed_seconds=0.0,
            peak_memory_bytes=None,
        )


class _Masks:
    def load_box_local_mask(self, region: Region2D) -> np.ndarray[Any, Any]:
        return np.ones((2, 2), dtype=np.bool_)


def _region_feature(sink: _Sink, *, code_version: str = "1") -> VisualFeature:
    backend = AlphaClipRegionFeatureBackend(
        config=AlphaClipConfig(
            model_name="ViT-L/14",
            base_checkpoint_path="base.pt",
            alpha_checkpoint_path="alpha.pth",
            checkpoint_fingerprint=FINGERPRINT,
            code_version=code_version,
        ),
        run_id=VisualEvidenceRunId("run-0001"),
        feature_stage_id="region_feature_extraction",
        mask_source=_Masks(),
        payload_sink=sink,
        runtime=_FakeImageRuntime(),
    )
    region = Region2D(
        region_id=RegionId("region-0001"),
        bounding_box=BoundingBox2D(x=1, y=1, width=2, height=2),
        mask_reference="masks/region-0001.npy",
        provenance=BackendProvenance(
            backend_id="fake", capability="region_discovery", provider="f", model="f", version="1"
        ),
    )
    image = PreparedImage(
        source_observation_id=SourceObservationId("frame-0001"),
        payload_reference="prepared/frame-0001.png",
        width=6,
        height=4,
    )
    (feature,) = backend.extract(image, (region,))
    return feature


def test_hypotheses_land_in_the_space_of_the_alphaclip_region_features() -> None:
    sink = _Sink()
    feature = _region_feature(sink)
    runtime = FakeTextRuntime({"chair": [0.0, 2.0], "door": [3.0, 4.0]})
    encoder = AlphaClipTextEncoder(config=_config(), runtime=runtime)

    batch = encoder.encode(("chair", "door"))

    assert embedding_space_fingerprint(encoder.embedding_space()) == feature.embedding_space_id
    assert batch.embedding_space_id == feature.embedding_space_id
    np.testing.assert_allclose(batch.values, [[0.0, 1.0], [0.6, 0.8]], rtol=1e-6)
    scorer = AlphaClipSemanticScorer(
        provenance=alphaclip_region_scorer_provenance(_config()),
        embedding_space_id=batch.embedding_space_id,
        payloads=sink,
        text_encoder=encoder,
    )
    assert scorer.backend_provenance().capability == "semantic_scorer"


@pytest.mark.parametrize(
    "changes",
    [{"checkpoint_fingerprint": "sha256:" + "b" * 64}, {"visual_code_version": "2"}],
)
def test_a_feature_of_another_alphaclip_space_is_ignored_never_scored_across(
    changes: dict[str, str],
) -> None:
    sink = _Sink()
    feature = _region_feature(sink)
    other = _config(**changes)
    encoder = AlphaClipTextEncoder(config=other, runtime=FakeTextRuntime({"chair": [0.0, 2.0]}))
    scorer = AlphaClipSemanticScorer(
        provenance=alphaclip_region_scorer_provenance(other),
        embedding_space_id=embedding_space_fingerprint(encoder.embedding_space()),
        payloads=sink,
        text_encoder=encoder,
    )
    claim = SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        source_observation_id=SourceObservationId("frame-0001"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("result-0001"),
        hypothesis="chair",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="f",
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

    # O scorer vive no espaço do seu próprio text tower; a feature de outro espaço é evidência
    # irrelevante para ele, então fica sem score, nunca pontuada entre espaços.
    assert scorer.score((claim,), (feature,)) == ()


def test_the_scorer_identity_names_the_weights_and_every_setting_that_changes_a_value() -> None:
    provenance = alphaclip_region_scorer_provenance(_config())

    assert (provenance.model, provenance.version) == ("ViT-L/14", FINGERPRINT)
    assert provenance.provider == "SunzeY/AlphaCLIP"
    for changed in (
        _config(text_template="a photo of a {hypothesis}"),
        _config(visual_code_version="2"),
        _config(precision="float16", device="cuda"),
    ):
        assert (
            alphaclip_region_scorer_provenance(changed).configuration_fingerprint
            != provenance.configuration_fingerprint
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text_template", "chair"),
        ("model_name", ""),
        ("base_checkpoint_path", "/abs/base.pt"),
        ("alpha_checkpoint_path", "../alpha.pth"),
        ("checkpoint_fingerprint", ""),
        ("visual_code_version", ""),
        ("device", "tpu"),
        ("max_batch_size", 0),
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


# ---------------------------------------------------------------- official alpha_clip runtime


class _Tensor:
    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array
        self.shape = array.shape

    def to(self, *args: object, **kwargs: object) -> _Tensor:
        return self

    def detach(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.array


class _AlphaClipModel:
    """Embed each text as (length, 1): every row identifies its text."""

    def __init__(self) -> None:
        self.text_projection = _Tensor(np.zeros((8, 2), dtype=np.float32))
        self.batch_sizes: list[int] = []

    def float(self) -> _AlphaClipModel:
        return self

    def eval(self) -> _AlphaClipModel:
        return self

    def encode_text(self, tokens: _Tensor) -> _Tensor:
        self.batch_sizes.append(len(tokens.array))
        rows = np.stack([tokens.array[:, 0], np.ones(len(tokens.array))], axis=1)
        return _Tensor(rows.astype(np.float32))


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, model: _AlphaClipModel) -> dict[str, Any]:
    loaded: dict[str, Any] = {}

    def load(*args: object, **kwargs: object) -> tuple[_AlphaClipModel, None]:
        loaded.update({"args": args, **kwargs})
        return model, None

    def tokenize(texts: list[str], *, truncate: bool) -> _Tensor:
        loaded.setdefault("truncate", []).append(truncate)
        if any(len(text) > 20 for text in texts):
            raise RuntimeError("Input is too long for context length 77")
        return _Tensor(np.array([[len(text)] for text in texts], dtype=np.int64))

    sdks = {
        "torch": SimpleNamespace(
            inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
        "alpha_clip": SimpleNamespace(load=load, tokenize=tokenize),
    }
    monkeypatch.setattr(
        alphaclip_text_module, "importlib", SimpleNamespace(import_module=sdks.__getitem__)
    )
    return loaded


def _checkpoints(tmp_path: Path) -> str:
    (tmp_path / "base.pt").write_bytes(b"base checkpoint")
    (tmp_path / "alpha.pth").write_bytes(b"alpha checkpoint")
    return combined_checkpoint_fingerprint(tmp_path / "base.pt", tmp_path / "alpha.pth")


def test_the_official_runtime_loads_the_verified_checkpoints_and_encodes_in_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _AlphaClipModel()
    loaded = _install_fake_sdk(monkeypatch, model)
    config = _config(checkpoint_fingerprint=_checkpoints(tmp_path), max_batch_size=2)
    runtime = OfficialAlphaClipTextRuntime(config=config, checkpoint_root=tmp_path)

    assert runtime.projection_dimension() == 2
    array = runtime.encode(("a", "bb", "ccc"))

    assert loaded["args"] == (str((tmp_path / "base.pt").resolve()),)
    assert loaded["alpha_vision_ckpt_pth"] == str((tmp_path / "alpha.pth").resolve())
    assert model.batch_sizes == [2, 1]
    np.testing.assert_array_equal(array, [[1, 1], [2, 1], [3, 1]])
    assert loaded["truncate"] == [False, False]


def test_checkpoints_that_do_not_match_the_declared_fingerprint_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_sdk(monkeypatch, _AlphaClipModel())
    _checkpoints(tmp_path)
    runtime = OfficialAlphaClipTextRuntime(config=_config(), checkpoint_root=tmp_path)

    with pytest.raises(AlphaClipTextModelLoadError, match="fingerprint mismatch"):
        runtime.encode(("chair",))


def test_a_text_longer_than_the_tower_accepts_is_refused_not_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_sdk(monkeypatch, _AlphaClipModel())
    config = _config(checkpoint_fingerprint=_checkpoints(tmp_path))
    runtime = OfficialAlphaClipTextRuntime(config=config, checkpoint_root=tmp_path)

    with pytest.raises(AlphaClipTextInferenceError, match="never truncated"):
        runtime.encode(("a hypothesis far too long",))


def test_missing_sdk_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr(
        alphaclip_text_module, "importlib", SimpleNamespace(import_module=missing_dependency)
    )
    runtime = OfficialAlphaClipTextRuntime(config=_config(), checkpoint_root=tmp_path)

    with pytest.raises(AlphaClipTextDependencyError, match="torch and alpha_clip"):
        runtime.encode(("chair",))
