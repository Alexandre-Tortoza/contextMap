"""The #610 probe: statistics and fixtures of SAM3's ``masks_logits`` and ``masks`` tensors.

The probe runs on a machine with the SAM3 checkpoint (``experiments/sam3-mask-scale-20260927/``).
What it computes from the tensors, and the fixture it writes for the contract test in
``tests/visual_perception/adapters/test_sam3.py``, is deterministic and tested here with
synthetic tensors, so the measurement cannot be spoiled by a harness bug.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from contextmap.evaluation import canonical_real_scenario
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
from contextmap.region_perception.adapters.sam3 import Sam3Config, Sam3Strategy
from contextmap.shared import SourceTimestamp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "experiments" / "sam3-mask-scale-20260927" / "scripts"


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    yield importlib.import_module("sam3_mask_scale")


LOGITS = np.array([-3.0, -0.1, 0.25, 0.7, 4.0], dtype=np.float32)
PROBABILITIES = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)


class TestTensorStatistics:
    def test_probabilities_stay_inside_the_unit_interval(self, probe: ModuleType) -> None:
        statistics = probe.TensorStatistics()
        statistics.update(PROBABILITIES, dtype="torch.float32")

        record = statistics.to_record()

        assert record["dtype"] == ["torch.float32"]
        assert record["count"] == 4
        assert (record["min"], record["max"]) == (0.0, 1.0)
        assert (record["below_zero"], record["above_one"]) == (0, 0)
        assert record["outside_unit_interval_fraction"] == 0.0
        assert record["binary"] is False

    def test_logits_are_counted_on_both_sides_of_the_unit_interval(self, probe: ModuleType) -> None:
        statistics = probe.TensorStatistics()
        statistics.update(LOGITS, dtype="torch.float32")

        record = statistics.to_record()

        assert (record["below_zero"], record["above_one"]) == (2, 1)
        assert record["outside_unit_interval_fraction"] == pytest.approx(3 / 5)
        counts = {(bin_["from"], bin_["to"]): bin_["count"] for bin_ in record["histogram"]}
        assert counts[(-5.0, -2.0)] == 1
        assert counts[(-1.0, 0.0)] == 1
        assert counts[(0.25, 0.5)] == 1
        assert counts[(0.5, 0.75)] == 1
        assert counts[(2.0, 5.0)] == 1
        assert sum(counts.values()) == 5
        # O primeiro e o último bin não têm limite: nenhum valor fica fora do histograma.
        assert record["histogram"][0]["from"] is None
        assert record["histogram"][-1]["to"] is None

    def test_a_boolean_mask_is_binary(self, probe: ModuleType) -> None:
        statistics = probe.TensorStatistics()
        statistics.update(np.array([[True, False], [False, False]]), dtype="torch.bool")

        record = statistics.to_record()

        assert record["binary"] is True
        assert (record["min"], record["max"]) == (0.0, 1.0)

    def test_updates_accumulate_and_non_finite_values_are_counted_apart(
        self, probe: ModuleType
    ) -> None:
        statistics = probe.TensorStatistics()
        statistics.update(PROBABILITIES, dtype="torch.float32")
        statistics.update(np.array([np.nan, 7.0]), dtype="torch.bfloat16")

        record = statistics.to_record()

        assert record["count"] == 6
        assert record["nonfinite"] == 1
        assert record["max"] == 7.0
        assert record["dtype"] == ["torch.bfloat16", "torch.float32"]

    def test_the_scale_is_read_from_the_values(self, probe: ModuleType) -> None:
        logits, probabilities, nothing = (probe.TensorStatistics() for _ in range(3))
        logits.update(LOGITS, dtype="torch.float32")
        probabilities.update(PROBABILITIES, dtype="torch.float32")

        assert probe.observed_scale(logits) == "logit"
        assert probe.observed_scale(probabilities) == "probability"
        assert probe.observed_scale(nothing) == "undetermined"


class TestBinarization:
    def test_the_current_rule_and_the_sigmoid_rule_are_compared_with_the_sdk_mask(
        self, probe: ModuleType
    ) -> None:
        comparison = probe.binarization_comparison(LOGITS, LOGITS > 0, threshold=0.5)

        assert comparison == {
            "pixels": 5,
            "current_rule_foreground": 2,
            "sigmoid_rule_foreground": 3,
            "rules_disagree": 1,
            "masks_binary": True,
            "masks_foreground": 3,
            "masks_equal_current_rule": pytest.approx(4 / 5),
            "masks_equal_sigmoid_rule": 1.0,
        }

    def test_without_a_binary_sdk_mask_only_the_rules_are_compared(self, probe: ModuleType) -> None:
        comparison = probe.binarization_comparison(LOGITS, LOGITS * 0.5, threshold=0.5)

        assert comparison["masks_binary"] is False
        assert comparison["masks_equal_current_rule"] is None
        assert comparison["masks_equal_sigmoid_rule"] is None


class TestSampleWindow:
    def test_the_window_holds_a_pixel_where_the_two_readings_disagree(
        self, probe: ModuleType
    ) -> None:
        logits = np.full((8, 8), -6.0)
        logits[5, 6] = 0.3  # a única posição em que x >= 0.5 e sigmoid(x) >= 0.5 divergem
        logits[1:3, 1:3] = 5.0

        x0, y0, x1, y1 = probe.sample_window(logits, threshold=0.5, size=4, scale="logit")

        assert (x1 - x0, y1 - y0) == (4, 4)
        assert x0 <= 6 < x1 and y0 <= 5 < y1
        assert min(x0, y0) >= 0 and max(x1, y1) <= 8

    def test_probabilities_are_sampled_across_the_mask_boundary(self, probe: ModuleType) -> None:
        # Sob a escala de probabilidade, o 0 do fundo não é uma "divergência" entre leituras.
        probabilities = np.zeros((6, 10))
        probabilities[2:4, 4:7] = 0.9

        x0, y0, x1, y1 = probe.sample_window(
            probabilities, threshold=0.5, size=4, scale="probability"
        )
        window = probabilities[y0:y1, x0:x1]

        assert (window >= 0.5).any() and (window < 0.5).any()

    def test_a_mask_without_boundary_has_no_window(self, probe: ModuleType) -> None:
        for values in (np.zeros((4, 4)), np.ones((4, 4))):
            assert probe.sample_window(values, threshold=0.5, size=4, scale="probability") is None


def _config(**changes: Any) -> Sam3Config:
    values: dict[str, Any] = {
        "checkpoint": "facebook/sam3",
        "model_version": "rev-under-test",
        "device": "cpu",
        "precision": "float32",
        "strategy": Sam3Strategy.TEXT_PROMPT,
        "prompts": ("floor",),
        "score_threshold": 0.5,
    }
    values.update(changes)
    return Sam3Config(**values)


class TestFixture:
    def test_a_sample_keeps_the_exact_values_of_its_window(self, probe: ModuleType) -> None:
        logits = np.full((4, 4), np.float32(0.3), dtype=np.float32).astype(np.float64)
        logits[0, 0] = -2.0

        sample = probe.fixture_sample(
            frame="frame-1",
            prompt="floor",
            proposal_index=3,
            box=(0.0, 0.0, 2.0, 2.0),
            score=0.9,
            logits=logits,
            masks=logits > 0,
            window=(0, 0, 2, 2),
            logits_dtype="torch.float32",
            masks_dtype="torch.bool",
        )
        restored = json.loads(json.dumps(sample))

        # O valor float32 0.3 sobrevive exato ao JSON: nenhum arredondamento muda um corte.
        stored = float(np.float32(0.3))
        assert restored["masks_logits"] == [[-2.0, stored], [stored, stored]]
        assert restored["masks"] == [[False, True], [True, True]]
        assert (restored["prompt"], restored["box"], restored["score"]) == (
            "floor",
            [0.0, 0.0, 2.0, 2.0],
            0.9,
        )
        assert (restored["width"], restored["height"]) == (2, 2)
        assert restored["window"] == [0, 0, 2, 2]
        assert restored["full_min"] == -2.0

    def test_the_document_names_what_the_contract_test_reads(self, probe: ModuleType) -> None:
        document = probe.fixture_document(
            samples=[{"prompt": "floor"}],
            scale="logit",
            config=_config(),
            identity={"sam3": "1.0"},
        )

        assert document["observed_scale"] == "logit"
        assert document["mask_threshold"] == 0.5
        assert (document["checkpoint"], document["model_version"]) == (
            "facebook/sam3",
            "rev-under-test",
        )
        assert document["config_digests"] == {"floor": _config().digest}
        assert document["samples"] == [{"prompt": "floor"}]
        assert document["identity"] == {"sam3": "1.0"}
        assert "rule" in document


def test_the_declared_frames_are_frozen_frames_of_the_canonical_sample(
    probe: ModuleType,
) -> None:
    frozen = [
        image.observation_id
        for image in canonical_real_scenario().subject.selection.selected_images
    ]

    assert list(probe.DEFAULT_FRAMES) == [frozen[0], frozen[10], frozen[-1]]


class TestImages:
    @pytest.mark.parametrize(
        ("encoding", "data", "expected"),
        [
            (ImageEncoding.RGB8, bytes([1, 2, 3, 4, 5, 6]), [[[1, 2, 3], [4, 5, 6]]]),
            (ImageEncoding.BGR8, bytes([1, 2, 3, 4, 5, 6]), [[[3, 2, 1], [6, 5, 4]]]),
            (ImageEncoding.MONO8, bytes([7, 9]), [[[7, 7, 7], [9, 9, 9]]]),
        ],
    )
    def test_an_observation_becomes_an_rgb_array(
        self, probe: ModuleType, encoding: ImageEncoding, data: bytes, expected: list[Any]
    ) -> None:
        array = probe.rgb_array(_image("frame-1", encoding=encoding, data=data, width=2))

        assert array.dtype == np.uint8
        assert array.tolist() == expected


class TestCheckpointIdentity:
    def test_the_revision_and_blob_of_each_cached_file_are_recorded(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        repository = tmp_path / "models--facebook--sam3"
        revision = "d" * 40
        (repository / "refs").mkdir(parents=True)
        (repository / "refs" / "main").write_text(revision, encoding="utf-8")
        (repository / "blobs").mkdir()
        (repository / "blobs" / ("e" * 64)).write_bytes(b"weights")
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "sam3.pt").symlink_to(Path("..") / ".." / "blobs" / ("e" * 64))

        identity = probe.checkpoint_identity("facebook/sam3", cache=tmp_path)

        assert identity == {
            "repo_id": "facebook/sam3",
            "revision": revision,
            "files": {"sam3.pt": "e" * 64},
        }

    def test_a_checkpoint_outside_the_cache_is_recorded_as_unknown(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        assert probe.checkpoint_identity("facebook/sam3", cache=tmp_path) == {
            "repo_id": "facebook/sam3",
            "revision": None,
            "files": {},
        }


def _image(
    frame: str,
    *,
    encoding: ImageEncoding = ImageEncoding.RGB8,
    data: bytes | None = None,
    width: int = 8,
    height: int = 1,
) -> ImageObservation:
    channels = 1 if encoding is ImageEncoding.MONO8 else 3
    return ImageObservation(
        observation_id=SourceObservationId(frame),
        sensor_id=SensorId("camera_1"),
        frame_id=FrameId("camera_1_optical"),
        timestamp=SourceTimestamp(seconds=1, nanoseconds=0, clock_id="fixture:header"),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/images"),
        width=width,
        height=height,
        encoding=encoding,
        data=bytes(width * height * channels) if data is None else data,
    )


class _FakeProcessor:
    """Answers every prompt with two proposals whose masks are logits of a known shape."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(
            parameters=lambda: [SimpleNamespace(device="cpu", dtype="torch.float32")]
        )
        self.device = "cpu"
        self.prompts: list[str] = []

    def set_image(self, image: object) -> object:
        assert getattr(image, "shape", None) == (6, 8, 3)
        return {"image": True}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        return state

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self.prompts.append(prompt)
        logits = np.full((2, 1, 6, 8), -5.0, dtype=np.float32)
        logits[0, 0, 1:4, 2:6] = 3.0
        logits[0, 0, 4, 2:6] = 0.2  # entre 0 e 0,5: as duas leituras divergem aqui
        logits[1, 0, 0:2, 0:2] = 1.5
        return {
            "masks_logits": logits,
            "masks": logits > 0,
            "boxes": np.array([[2.0, 1.0, 6.0, 5.0], [0.0, 0.0, 2.0, 2.0]], dtype=np.float32),
            "scores": np.array([0.9, 0.6], dtype=np.float32),
        }


def _sequence(directory: Path, frames: tuple[str, ...]) -> Path:
    with SequenceArtifactWriter(
        output_dir=directory,
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-under-test"),
    ) as writer:
        for frame in frames:
            writer.add_observation(_image(frame, width=8, height=6))
        writer.finalize()
    return directory


class TestProbe:
    def test_one_probe_measures_every_frame_and_prompt_and_samples_the_fixture(
        self, probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # O runtime oficial sempre entra em torch.inference_mode(); o torch é opcional aqui.
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext())
        )
        sequence = _sequence(tmp_path / "sequence", ("frame-1", "frame-2"))
        processor = _FakeProcessor()

        measurement, fixture = probe.run_probe(
            sequence=sequence,
            frames=("frame-1", "frame-2"),
            prompts=("floor", "door"),
            config=_config(),
            processor=processor,
            to_image=lambda array: array,
            samples=3,
            window=4,
        )

        assert processor.prompts == ["floor", "door", "floor", "door"]
        assert [(item["frame"], item["prompt"]) for item in measurement["calls"]] == [
            ("frame-1", "floor"),
            ("frame-1", "door"),
            ("frame-2", "floor"),
            ("frame-2", "door"),
        ]
        call = measurement["calls"][0]
        assert call["proposals"] == 2
        assert call["masks_logits"]["shape"] == [[2, 1, 6, 8]]
        assert call["masks"]["binary"] is True
        # A binarização do runtime atual é a regra x >= 0,5 aplicada ao tensor consumido.
        assert call["runtime_foreground"] == call["binarization"]["current_rule_foreground"]
        assert measurement["observed_scale"] == "logit"
        assert measurement["total"]["masks_logits"]["count"] == 4 * 2 * 6 * 8
        assert fixture["observed_scale"] == "logit"
        assert len(fixture["samples"]) == 3
        for sample in fixture["samples"]:
            x0, y0, x1, y1 = sample["window"]
            assert (x1 - x0, y1 - y0) == (sample["width"], sample["height"])
            assert len(sample["masks_logits"]) == sample["height"]
        # A primeira amostra cobre a faixa em que as duas leituras divergem.
        first = np.asarray(fixture["samples"][0]["masks_logits"])
        assert ((first >= 0) & (first < 0.5)).any()
        # Os dois documentos são JSON puro: nenhum escalar do NumPy sobra neles.
        assert json.loads(json.dumps(measurement)) == measurement
        assert json.loads(json.dumps(fixture)) == fixture

    def test_a_probe_without_any_proposal_writes_no_fixture(
        self, probe: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(inference_mode=lambda: nullcontext())
        )

        class Silent(_FakeProcessor):
            def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
                empty = np.zeros((0, 1, 6, 8), dtype=np.float32)
                return {"masks_logits": empty, "masks": empty > 0, "boxes": [], "scores": []}

        measurement, fixture = probe.run_probe(
            sequence=_sequence(tmp_path / "sequence", ("frame-1",)),
            frames=("frame-1",),
            prompts=("floor",),
            config=_config(),
            processor=Silent(),
            to_image=lambda array: array,
            samples=3,
            window=4,
        )

        assert measurement["observed_scale"] == "undetermined"
        assert fixture is None

    def test_the_outputs_are_never_overwritten(self, probe: ModuleType, tmp_path: Path) -> None:
        target = tmp_path / "measurement.json"
        probe.write_json(target, {"first": True})

        with pytest.raises(FileExistsError):
            probe.write_json(target, {"second": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {"first": True}
