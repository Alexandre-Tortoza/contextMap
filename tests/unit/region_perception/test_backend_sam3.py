from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fakes import FakeLoadedModel
from mask_cases import BACKEND_SEEDS, GOLDEN, digest, sam3_candidates

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    AuditedRegionDiscovery,
    BoundingBox,
    InlineMask,
    Region2D,
    RegionDiscovery,
    RejectionReason,
)
from contextmap.region_perception.adapters import sam3 as sam3_module
from contextmap.region_perception.adapters.sam3 import (
    Sam3Config,
    Sam3ImageProcessorRuntime,
    Sam3NativeOutput,
    Sam3NativeProposal,
    Sam3RegionDiscovery,
    Sam3Strategy,
)
from contextmap.region_perception.discovery.region_discovery import (
    DiscoveryInput,
    DiscoveryPass,
    PassKind,
)
from contextmap.region_perception.discovery.region_normalization import normalize_regions
from contextmap.shared import ArtifactReference
from contextmap.visual_perception import PreparedImage


@dataclass(frozen=True)
class MaterializedImage:
    size: tuple[int, int]
    token: tuple[str, str]


class InferenceModeTorch(ModuleType):
    """Fake torch whose inference-mode switch the fake SDK reads, as it would read the real one."""

    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self) -> None:
        super().__init__("torch")
        self._inference_mode = False
        self.autocasts: list[dict[str, object]] = []

    def is_inference_mode_enabled(self) -> bool:
        return self._inference_mode

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        previous, self._inference_mode = self._inference_mode, True
        try:
            yield
        finally:
            self._inference_mode = previous

    def autocast(self, **kwargs: object) -> AbstractContextManager[None]:
        self.autocasts.append(kwargs)
        return nullcontext()


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> InferenceModeTorch:
    """The official runtime always enters ``torch.inference_mode()``; torch itself is optional."""
    torch = InferenceModeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


class OnePhraseProcessor:
    """Base of the official-processor stand-ins queried with one phrase, which is never reset."""

    def reset_all_prompts(self, state: object) -> None:
        raise AssertionError("a run with a single phrase never resets the prompts")


def _materialized_image(discovery_input: DiscoveryInput) -> MaterializedImage:
    discovery_pass = discovery_input.discovery_pass
    return MaterializedImage(
        size=(discovery_pass.input_width, discovery_pass.input_height),
        token=("image", discovery_pass.pass_id),
    )


def _input() -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-8"),
            payload_reference="outputs/frame-8.png",
            payload_artifact=ArtifactReference(
                uri="outputs/frame-8.png",
                sha256=sha256(b"frame-8").hexdigest(),
                media_type="image/png",
            ),
            width=5,
            height=4,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=5, y_max=4),
        ),
        perception_run_id="run-sam3",
        perception_result_id="result-sam3",
    )


class FakeSam3Runtime:
    def predict(
        self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
    ) -> Sam3NativeOutput:
        return Sam3NativeOutput(
            proposals=(
                Sam3NativeProposal(
                    proposal_id="proposal-3",
                    box=(1.0, 1.0, 4.0, 3.0),
                    mask=_mask_5x4(x_range=range(1, 4), y_range=range(1, 3)),
                    score_name="mask_score",
                    score=0.93,
                    query_id="query-1",
                    prompt="all movable items",
                    metadata=(("decoder_pass", 2),),
                ),
            ),
            warnings=("runtime used deterministic precision",),
            metadata=(("peak_memory_mb", 512.0),),
        )


def test_sam3_preserves_strategy_query_and_native_score_semantics() -> None:
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        device="cuda:0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=("all movable items",),
        score_threshold=0.8,
        mask_threshold=0.5,
        strategy_settings=(("max_queries", 8),),
    )
    backend = Sam3RegionDiscovery(
        config=config, runtime=FakeSam3Runtime(), pass_images=_materialized_image
    )

    output = backend.discover_candidates(_input())

    candidate = output.candidates[0]
    assert candidate.score is not None
    assert candidate.score.name == "mask_score"
    assert "not calibrated" in candidate.score.semantics
    assert candidate.provenance.backend_id == "sam3"
    assert candidate.provenance.query == "text_prompt:all movable items:query-1"
    assert candidate.provenance.config_digest == config.digest
    assert dict(candidate.native_metadata)["decoder_pass"] == 2
    assert output.diagnostics.warnings == ("runtime used deterministic precision",)
    assert dict(output.diagnostics.metadata)["strategy"] == "text_prompt"
    assert dict(output.diagnostics.metadata)["peak_memory_mb"] == 512.0
    json.dumps(candidate.to_dict())
    assert isinstance(backend, RegionDiscovery)
    regions = backend.discover(_input().prepared_image)
    assert all(isinstance(region, Region2D) for region in regions)
    assert regions[0].provenance == backend.backend_provenance()


def test_sam3_reports_the_audit_of_the_regions_it_discovers() -> None:
    """#611: the regions of the public port come with the passes, rejections and merges."""
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=("all movable items",),
        score_threshold=0.8,
    )
    backend = Sam3RegionDiscovery(
        config=config, runtime=FakeSam3Runtime(), pass_images=_materialized_image
    )
    image = _input().prepared_image

    audited = backend.discover_audited(image)

    assert isinstance(backend, AuditedRegionDiscovery)
    assert audited.regions == backend.discover(image)
    assert audited.audit.source_observation_id == image.source_observation_id
    assert audited.audit.backend == backend.backend_provenance()
    assert [discovery_pass.pass_id for discovery_pass in audited.audit.passes] == ["full-frame"]


def test_sam3_strategy_configuration_is_explicit() -> None:
    with pytest.raises(ValueError, match="requires at least one phrase"):
        Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.TEXT_PROMPT)
    with pytest.raises(ValueError, match="does not consume text prompts"):
        Sam3Config(
            checkpoint="sam3",
            model_version="3.0",
            strategy=Sam3Strategy.AUTOMATIC,
            prompts=("hidden architectural default",),
        )
    with pytest.raises(ValueError, match="score_threshold"):
        Sam3Config(checkpoint="sam3", model_version="3.0", score_threshold=-0.1)


# --- #662: um conjunto de conceitos num único run -----------------------------------------


def _concept_set_config(*prompts: str) -> Sam3Config:
    return Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=prompts,
        score_threshold=0.5,
    )


class ConceptSetRuntime:
    """Answers each configured phrase with one proposal of its own, as the official runtime.

    Each phrase finds its own mask, or ``mask`` for every phrase when one is given.
    """

    def __init__(self, mask: InlineMask | None = None) -> None:
        self._mask = mask

    def predict(
        self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
    ) -> Sam3NativeOutput:
        return Sam3NativeOutput(
            proposals=tuple(
                Sam3NativeProposal(
                    proposal_id=f"sam3-{index:06d}-000000",
                    box=(float(index), 1.0, float(index + 2), 3.0),
                    mask=(
                        _mask_5x4(x_range=range(index, index + 2), y_range=range(1, 3))
                        if self._mask is None
                        else self._mask
                    ),
                    score_name="concept_score",
                    score=0.9,
                    query_id=f"text-prompt-{index:06d}-000000",
                    prompt=phrase,
                )
                for index, phrase in enumerate(config.prompts)
            )
        )


def test_sam3_keeps_the_phrase_of_each_proposal_of_a_concept_set() -> None:
    config = _concept_set_config("chair", "table")
    backend = Sam3RegionDiscovery(
        config=config, runtime=ConceptSetRuntime(), pass_images=_materialized_image
    )

    output = backend.discover_candidates(_input())

    assert [candidate.provenance.query for candidate in output.candidates] == [
        "text_prompt:chair:text-prompt-000000-000000",
        "text_prompt:table:text-prompt-000001-000000",
    ]
    assert [dict(candidate.native_metadata)["text_prompt"] for candidate in output.candidates] == [
        "chair",
        "table",
    ]
    # A frase é hipótese de consulta, não rótulo: não vira texto nativo nem claim.
    assert all(candidate.native_text is None for candidate in output.candidates)
    assert {candidate.provenance.config_digest for candidate in output.candidates} == {
        config.digest
    }
    metadata = dict(output.diagnostics.metadata)
    assert (metadata["prompt_000000"], metadata["prompt_000001"]) == ("chair", "table")


def test_the_same_mask_from_two_phrases_is_one_region_that_keeps_both_phrases() -> None:
    """#662: the documented outcome of a duplicate across phrases, from unchanged normalization."""
    mask = _mask_5x4(x_range=range(1, 4), y_range=range(1, 3))
    backend = Sam3RegionDiscovery(
        config=_concept_set_config("table", "desk"),
        runtime=ConceptSetRuntime(mask),
        pass_images=_materialized_image,
    )

    audited = backend.discover_audited(_input().prepared_image)

    (region,) = audited.regions
    first, second = "full-frame/sam3-000000-000000", "full-frame/sam3-000001-000000"
    assert region.contributor_candidate_ids == (first, second)
    assert [item.query for item in region.discovery_provenance] == [
        "text_prompt:table:text-prompt-000000-000000",
        "text_prompt:desk:text-prompt-000001-000000",
    ]
    # Mesma área: o representante é o menor candidate_id, a frase que vem antes em prompts.
    (decision,) = audited.audit.merge_decisions
    assert (decision.representative_candidate_id, decision.merged_candidate_id) == (first, second)
    assert [item.candidate_id for item in audited.audit.normalization_rejections] == [second]


@pytest.mark.parametrize(
    ("prompts", "message"),
    [
        ((), "requires at least one phrase"),
        (("chair", "table", "chair"), "must be unique"),
        (("chair", ""), "must not be blank"),
        (("chair", "  "), "must not be blank"),
    ],
)
def test_sam3_text_prompt_requires_an_ordered_set_of_distinct_phrases(
    prompts: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _concept_set_config(*prompts)


def test_sam3_digest_covers_every_phrase_in_order() -> None:
    concept_set = _concept_set_config("chair", "table")

    assert concept_set.digest != _concept_set_config("table", "chair").digest
    assert concept_set.digest != _concept_set_config("chair", "door").digest
    assert concept_set.digest == _concept_set_config("chair", "table").digest


@pytest.mark.parametrize("phrase", [None, "sofa"])
def test_sam3_refuses_a_proposal_for_a_phrase_the_configuration_did_not_ask(
    phrase: str | None,
) -> None:
    proposal = Sam3NativeProposal(
        proposal_id="sam3-000000-000000",
        box=(1.0, 1.0, 3.0, 3.0),
        mask=_mask_5x4(x_range=range(1, 3), y_range=range(1, 3)),
        score_name="concept_score",
        score=0.9,
        query_id="text-prompt-000000-000000",
        prompt=phrase,
    )
    backend = Sam3RegionDiscovery(
        config=_concept_set_config("chair", "table"),
        runtime=ProposalRuntime(proposal),
        pass_images=_materialized_image,
    )

    with pytest.raises(ValueError, match="configured phrases"):
        backend.discover_candidates(_input())


class ConceptSetProcessor:
    """Official-processor stand-in with the SDK's state semantics (``Sam3Processor``).

    ``set_text_prompt`` writes its results into the state it receives and returns that same
    state, and ``reset_all_prompts`` deletes the prompt and the results from it: a runtime that
    read a phrase's output after the next phrase or reset would read the wrong masks.
    """

    model = FakeLoadedModel()
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_image(self, image: object) -> object:
        self.calls.append(("image", image))
        return {"backbone_out": {"image": "encoded"}}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        self.calls.append(("threshold", threshold))
        return state

    def reset_all_prompts(self, state: object) -> None:
        self.calls.append(("reset", None))
        if not isinstance(state, dict):
            raise TypeError("test state must be a dictionary")
        state["backbone_out"].pop("language_features", None)
        for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
            state.pop(key, None)

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self.calls.append(("prompt", prompt))
        if not isinstance(state, dict):
            raise TypeError("test state must be a dictionary")
        # Cada frase acende colunas próprias: "chair" a coluna 0; "table" as colunas 2 e 4.
        columns = {"chair": (0,), "table": (2, 4)}[prompt]
        masks = np.zeros((len(columns), 1, 4, 5))
        for index, column in enumerate(columns):
            masks[index, 0, :, column] = 0.9
        state["backbone_out"]["language_features"] = prompt
        state.update(
            boxes=[[float(column), 0.0, float(column + 1), 4.0] for column in columns],
            scores=[0.8] * len(columns),
            masks_logits=masks,
        )
        return state


def test_official_sam3_runtime_encodes_the_pass_once_and_queries_every_phrase_in_order() -> None:
    processor = ConceptSetProcessor()
    runtime = Sam3ImageProcessorRuntime(processor=processor)

    output = runtime.predict(
        _input(), _materialized_image(_input()), _concept_set_config("chair", "table")
    )

    assert processor.calls == [
        ("image", MaterializedImage((5, 4), ("image", "full-frame"))),
        ("threshold", 0.5),
        ("prompt", "chair"),
        ("reset", None),
        ("prompt", "table"),
    ]
    assert [
        (proposal.proposal_id, proposal.query_id, proposal.prompt) for proposal in output.proposals
    ] == [
        ("sam3-000000-000000", "text-prompt-000000-000000", "chair"),
        ("sam3-000001-000000", "text-prompt-000001-000000", "table"),
        ("sam3-000001-000001", "text-prompt-000001-000001", "table"),
    ]
    lit_columns = [
        np.flatnonzero(proposal.mask.as_array().any(axis=0)).tolist()
        for proposal in output.proposals
    ]
    assert lit_columns == [[0], [2], [4]]


def test_sam3_does_not_fall_back_when_configured_runtime_fails() -> None:
    class FailingRuntime:
        def predict(
            self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
        ) -> Sam3NativeOutput:
            raise RuntimeError("configured strategy is unavailable")

    backend = Sam3RegionDiscovery(
        config=Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.AUTOMATIC),
        runtime=FailingRuntime(),
        pass_images=_materialized_image,
    )

    with pytest.raises(RuntimeError, match="configured strategy is unavailable"):
        backend.discover_candidates(_input())


def test_official_sam3_text_processor_output_is_detached_and_thresholded() -> None:
    class NativeArray:
        def __init__(self, value: object) -> None:
            self._value = value

        def tolist(self) -> object:
            return self._value

    class ImageProcessor(OnePhraseProcessor):
        model = FakeLoadedModel()
        device = "cpu"

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def set_image(self, image: object) -> object:
            self.calls.append(("image", image))
            return {"image_state": "encoded"}

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            self.calls.append(("threshold", threshold))
            return state

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            self.calls.append(("prompt", prompt))
            if not isinstance(state, dict):
                raise TypeError("test state must be a dictionary")
            return {
                **state,
                "boxes": NativeArray([[1.0, 1.0, 4.0, 3.0]]),
                "scores": NativeArray([0.92]),
                "masks_logits": NativeArray(
                    [
                        [
                            [0.2, 0.8, 0.8, 0.2, 0.1],
                            [0.2, 0.8, 0.8, 0.2, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.1, 0.1],
                        ]
                    ]
                ),
            }

    processor = ImageProcessor()
    runtime = Sam3ImageProcessorRuntime(processor=processor)
    config = Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=("movable item",),
        score_threshold=0.7,
        mask_threshold=0.5,
    )

    output = runtime.predict(_input(), _materialized_image(_input()), config)

    assert processor.calls == [
        ("image", MaterializedImage((5, 4), ("image", "full-frame"))),
        ("threshold", 0.7),
        ("prompt", "movable item"),
    ]
    assert output.proposals[0].box == (1.0, 1.0, 4.0, 3.0)
    assert output.proposals[0].mask.as_array()[0, 1:3].tolist() == [True, True]
    assert output.proposals[0].score == 0.92
    assert output.proposals[0].query_id == "text-prompt-000000-000000"
    assert output.proposals[0].prompt == "movable item"


def test_official_sam3_runtime_rejects_an_unimplemented_strategy_without_fallback() -> None:
    class UnusedProcessor(OnePhraseProcessor):
        model = FakeLoadedModel()
        device = "cpu"

        def set_image(self, image: object) -> object:
            raise AssertionError("unsupported strategy must fail before inference")

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            raise AssertionError("unsupported strategy must fail before inference")

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            raise AssertionError("unsupported strategy must fail before inference")

    runtime = Sam3ImageProcessorRuntime(processor=UnusedProcessor())
    config = Sam3Config(checkpoint="sam3", model_version="3.0", strategy=Sam3Strategy.AUTOMATIC)

    with pytest.raises(ValueError, match="supports only text_prompt"):
        runtime.predict(_input(), _materialized_image(_input()), config)


def _config() -> Sam3Config:
    return Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=("floor",),
        score_threshold=0.5,
    )


def _proposal(box: tuple[float, float, float, float], mask: InlineMask) -> Sam3NativeProposal:
    return Sam3NativeProposal(
        proposal_id="proposal-1",
        box=box,
        mask=mask,
        score_name="concept_score",
        score=0.9,
        query_id="text-prompt-000000-000000",
        prompt="floor",
    )


class ProposalRuntime:
    def __init__(self, *proposals: Sam3NativeProposal) -> None:
        self._proposals = proposals

    def predict(
        self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
    ) -> Sam3NativeOutput:
        return Sam3NativeOutput(proposals=self._proposals)


def _mask_5x4(*, x_range: range, y_range: range) -> InlineMask:
    return InlineMask(
        np.array([[x in x_range and y in y_range for x in range(5)] for y in range(4)])
    )


def test_sam3_derives_the_candidate_box_from_the_mask_and_keeps_the_native_box() -> None:
    # A caixa nativa vem de um head independente e aqui não contém a máscara.
    proposal = _proposal((1.0, 1.0, 4.0, 3.0), _mask_5x4(x_range=range(0, 5), y_range=range(0, 2)))
    backend = Sam3RegionDiscovery(
        config=_config(), runtime=ProposalRuntime(proposal), pass_images=_materialized_image
    )

    candidate = backend.discover_candidates(_input()).candidates[0]

    assert candidate.bounding_box == BoundingBox(x_min=0, y_min=0, x_max=5, y_max=2)
    metadata = dict(candidate.native_metadata)
    assert (
        metadata["native_box_x_min"],
        metadata["native_box_y_min"],
        metadata["native_box_x_max"],
        metadata["native_box_y_max"],
    ) == (1.0, 1.0, 4.0, 3.0)
    assert metadata["native_box_contains_mask"] is False
    assert len(backend.discover(_input().prepared_image)) == 1


def test_sam3_native_box_outside_the_image_does_not_abort_discovery() -> None:
    proposal = _proposal((-0.4, 0.0, 5.6, 4.2), _mask_5x4(x_range=range(0, 5), y_range=range(2, 4)))
    backend = Sam3RegionDiscovery(
        config=_config(), runtime=ProposalRuntime(proposal), pass_images=_materialized_image
    )

    candidate = backend.discover_candidates(_input()).candidates[0]

    assert candidate.bounding_box == BoundingBox(x_min=0, y_min=2, x_max=5, y_max=4)
    assert dict(candidate.native_metadata)["native_box_x_min"] == -0.4
    assert dict(candidate.native_metadata)["native_box_contains_mask"] is True
    regions = backend.discover(_input().prepared_image)
    assert len(regions) == 1
    assert regions[0].area_pixels == 10


def test_sam3_empty_masks_are_rejected_explicitly_by_normalization() -> None:
    empty = InlineMask(np.zeros((4, 5), dtype=bool))
    inside = _proposal((1.0, 1.0, 3.0, 3.0), empty)
    outside = Sam3NativeProposal(
        proposal_id="proposal-2",
        box=(-5.0, -5.0, -1.0, -1.0),
        mask=empty,
        score_name="concept_score",
        score=0.8,
        query_id="text-prompt-000000-000001",
        prompt="floor",
    )
    backend = Sam3RegionDiscovery(
        config=_config(), runtime=ProposalRuntime(inside, outside), pass_images=_materialized_image
    )

    output = backend.discover_candidates(_input())
    result = normalize_regions(
        output.candidates, _input().prepared_image, backend.backend_provenance()
    )

    assert len(output.candidates) == 2
    assert result.regions == ()
    assert {item.reason for item in result.rejected} == {RejectionReason.INVALID_GEOMETRY}


class RecordingProcessor:
    """Official-processor stand-in that records call order into a shared event list."""

    def __init__(self, events: list[str], *, model: object | None = None) -> None:
        self._events = events
        self.model = model or FakeLoadedModel("cuda:0")
        self.device = "cuda:0"

    def set_image(self, image: object) -> object:
        self._events.append("image")
        return {}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        self._events.append("threshold")
        return state

    def reset_all_prompts(self, state: object) -> None:
        self._events.append("reset")

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self._events.append("prompt")
        return {"boxes": [], "scores": [], "masks": []}


class RecordingContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> None:
        self._events.append("enter")

    def __exit__(self, *exc_info: object) -> None:
        self._events.append("exit")


def _text_prompt_config(precision: str) -> Sam3Config:
    return Sam3Config(
        checkpoint="facebook/sam3",
        model_version="3.0",
        device="cuda:0",
        precision=precision,
        strategy=Sam3Strategy.TEXT_PROMPT,
        prompts=("floor",),
    )


def test_sam3_precision_must_be_a_supported_inference_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        Sam3Config(checkpoint="sam3", model_version="3.0", precision="int8")
    for precision in ("float32", "float16", "bfloat16"):
        assert (
            Sam3Config(checkpoint="sam3", model_version="3.0", precision=precision).precision
            == precision
        )


def test_official_sam3_runtime_runs_the_sdk_inside_the_configured_inference_context() -> None:
    events: list[str] = []
    received: list[Sam3Config] = []

    def autocast(config: Sam3Config) -> RecordingContext:
        received.append(config)
        return RecordingContext(events)

    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events),
        autocast=autocast,
    )
    config = _text_prompt_config("bfloat16")

    runtime.predict(_input(), _materialized_image(_input()), config)

    assert received == [config]
    assert events == ["enter", "image", "threshold", "prompt", "exit"]


def test_official_sam3_runtime_float32_runs_without_autocast(
    fake_torch: InferenceModeTorch,
) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(processor=RecordingProcessor(events))

    runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config("float32"))

    assert events == ["image", "threshold", "prompt"]
    assert fake_torch.autocasts == []


def test_official_sam3_runtime_default_context_is_torch_autocast(
    fake_torch: InferenceModeTorch,
) -> None:
    runtime = Sam3ImageProcessorRuntime(processor=RecordingProcessor([]))

    runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config("bfloat16"))

    assert fake_torch.autocasts == [{"device_type": "cuda", "dtype": "torch.bfloat16"}]


def test_official_sam3_runtime_without_torch_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(sam3_module, "import_module", unavailable)
    runtime = Sam3ImageProcessorRuntime(processor=RecordingProcessor([]))

    with pytest.raises(RuntimeError, match="requires torch inference_mode"):
        runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config("float32"))


class InferenceModeRecordingProcessor(OnePhraseProcessor):
    """Official-processor stand-in that records whether each SDK call ran in inference mode."""

    def __init__(self, torch: InferenceModeTorch) -> None:
        self._torch = torch
        self.model = FakeLoadedModel("cuda:0")
        self.device = "cuda:0"
        self.inference_mode: list[bool] = []

    def set_image(self, image: object) -> object:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return {}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return state

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        self.inference_mode.append(self._torch.is_inference_mode_enabled())
        return {"boxes": [], "scores": [], "masks": []}


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_official_sam3_runtime_runs_the_sdk_in_torch_inference_mode(
    precision: str, fake_torch: InferenceModeTorch
) -> None:
    processor = InferenceModeRecordingProcessor(fake_torch)
    runtime = Sam3ImageProcessorRuntime(processor=processor)

    runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config(precision))

    assert processor.inference_mode == [True, True, True]
    assert not fake_torch.is_inference_mode_enabled()


def test_sam3_configuration_requires_an_explicit_model_version() -> None:
    with pytest.raises(TypeError, match="model_version"):
        Sam3Config(checkpoint="facebook/sam3")  # type: ignore[call-arg]


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_official_sam3_runtime_rejects_a_model_on_another_device(device: str) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events, model=FakeLoadedModel(device, "float32")),
    )

    with pytest.raises(ValueError, match="device 'cuda:0'"):
        runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config("bfloat16"))

    assert events == []


@pytest.mark.parametrize("processor_device", ["cpu", "cuda:1"])
def test_official_sam3_runtime_rejects_a_processor_that_sends_images_elsewhere(
    processor_device: str,
) -> None:
    # #646: o processor oficial move cada imagem para o próprio device (default "cuda"), que
    # nada conferia contra a configuração, mesmo com o modelo no lugar certo.
    events: list[str] = []
    processor = RecordingProcessor(events)
    processor.device = processor_device
    runtime = Sam3ImageProcessorRuntime(processor=processor)

    with pytest.raises(ValueError, match=f"processor .*{processor_device}.*device 'cuda:0'"):
        runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config("bfloat16"))

    assert events == []


@pytest.mark.parametrize(
    ("weights", "precision"),
    [("float16", "bfloat16"), ("float16", "float32"), ("bfloat16", "float32")],
)
def test_official_sam3_runtime_rejects_weights_that_autocast_does_not_realize(
    weights: str, precision: str
) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events, model=FakeLoadedModel("cuda:0", weights)),
    )

    with pytest.raises(ValueError, match=f"precision '{precision}'"):
        runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config(precision))

    assert events == []


@pytest.mark.parametrize(
    ("weights", "precision"),
    [
        ("float32", "float32"),
        # Configuração que funciona com o SDK oficial (#338): pesos float32 sob autocast bfloat16.
        ("float32", "bfloat16"),
        ("bfloat16", "bfloat16"),
        ("float32", "float16"),
        ("float16", "float16"),
    ],
)
def test_official_sam3_runtime_accepts_float32_or_autocast_dtype_weights(
    weights: str, precision: str
) -> None:
    events: list[str] = []
    runtime = Sam3ImageProcessorRuntime(
        processor=RecordingProcessor(events, model=FakeLoadedModel("cuda:0", weights)),
    )

    runtime.predict(_input(), _materialized_image(_input()), _text_prompt_config(precision))

    assert events == ["image", "threshold", "prompt"]


@pytest.mark.parametrize("seed", BACKEND_SEEDS)
def test_sam3_mask_conversion_matches_the_recorded_behaviour(seed: int) -> None:
    # #593: do resultado nativo do SDK ao RegionCandidate, registrado antes da vetorização.
    assert digest(sam3_candidates(seed)) == GOLDEN["sam3"][str(seed)]


def test_sam3_tensor_masks_are_detached_to_the_cpu_in_double_precision() -> None:
    # #593: o tensor do SDK (GPU, bfloat16) vira um array de uma vez, nunca uma lista por pixel.
    calls: list[object] = []

    class FakeTensor:
        def __init__(self, array: np.ndarray[Any, Any]) -> None:
            self._array = array

        def detach(self) -> FakeTensor:
            calls.append("detach")
            return self

        def to(self, device: str) -> FakeTensor:
            calls.append(("to", device))
            return self

        def double(self) -> FakeTensor:
            calls.append("double")
            return FakeTensor(self._array.astype(np.float64))

        def numpy(self) -> np.ndarray[Any, Any]:
            calls.append("numpy")
            return self._array

        def tolist(self) -> object:
            raise AssertionError("a tensor mask must not become one Python object per pixel")

    logits = np.full((1, 1, 4, 5), 0.2, dtype=np.float32)
    logits[0, 0, 1, 1:4] = 0.9

    class Processor(OnePhraseProcessor):
        model = FakeLoadedModel()
        device = "cpu"

        def set_image(self, image: object) -> object:
            return {}

        def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
            return state

        def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
            return {
                "boxes": [[1.0, 1.0, 4.0, 2.0]],
                "scores": [0.9],
                "masks_logits": FakeTensor(logits),
            }

    runtime = Sam3ImageProcessorRuntime(processor=Processor())

    output = runtime.predict(_input(), _materialized_image(_input()), _config())

    assert calls == ["detach", ("to", "cpu"), "double", "numpy"]
    assert output.proposals[0].mask.as_array()[1].tolist() == [False, True, True, True, False]


# --- #610: a escala real de masks_logits -------------------------------------------------

SAM3_MASK_SCALE_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "sam3_mask_scale" / "samples.json"
)
"""Windows of real SAM3 ``masks_logits`` recorded by ``experiments/sam3-mask-scale-20260927``."""


class RecordedOutputProcessor(OnePhraseProcessor):
    """Replays one recorded ``set_text_prompt`` output through the official runtime path."""

    model = FakeLoadedModel()
    device = "cpu"

    def __init__(self, sample: Mapping[str, Any]) -> None:
        self._sample = sample

    def set_image(self, image: object) -> object:
        return {}

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        return state

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        return {
            "boxes": [self._sample["box"]],
            "scores": [self._sample["score"]],
            "masks_logits": [[self._sample["masks_logits"]]],
        }


def _input_of(*, width: int, height: int) -> DiscoveryInput:
    return DiscoveryInput(
        prepared_image=PreparedImage(
            source_observation_id=SourceObservationId("frame-recorded"),
            payload_reference="outputs/frame-recorded.png",
            payload_artifact=ArtifactReference(
                uri="outputs/frame-recorded.png",
                sha256=sha256(b"frame-recorded").hexdigest(),
                media_type="image/png",
            ),
            width=width,
            height=height,
            transformations=(),
        ),
        discovery_pass=DiscoveryPass(
            pass_id="full-frame",
            kind=PassKind.FULL_FRAME,
            window=BoundingBox(x_min=0, y_min=0, x_max=width, y_max=height),
        ),
        perception_run_id="run-sam3-recorded",
        perception_result_id="result-sam3-recorded",
    )


def _pin_the_recorded_scale(fixture: Mapping[str, Any]) -> None:
    """Require the recorded windows to show the scale the probe measured on the whole tensors."""
    scale = fixture["observed_scale"]
    samples = fixture["samples"]
    assert scale in {"probability", "logit"}
    assert samples
    # A escala vem do tensor inteiro de cada proposta, não só da janela gravada.
    inside = all(sample["full_min"] >= 0.0 and sample["full_max"] <= 1.0 for sample in samples)
    assert inside == (scale == "probability")
    if scale == "logit":
        # Sem pixel em [0, limiar) as duas leituras coincidiriam e a janela não fixaria nada.
        threshold = fixture["mask_threshold"]
        assert any(
            ((values >= 0) & (values < threshold)).any()
            for values in (np.asarray(sample["masks_logits"]) for sample in samples)
        )


def _runtime_and_measured_masks(
    fixture: Mapping[str, Any],
) -> list[tuple[list[list[bool]], list[list[bool]]]]:
    """Binarize every recorded window through the runtime and by the measured scale.

    Under a probability scale a pixel is foreground when ``x >= mask_threshold``, the runtime's
    direct comparison. Under a logit scale it is foreground when
    ``sigmoid(x) >= mask_threshold``, which the direct comparison contradicts wherever
    ``0 <= x < mask_threshold``.
    """
    threshold = fixture["mask_threshold"]
    pairs = []
    for sample in fixture["samples"]:
        values = np.asarray(sample["masks_logits"], dtype=np.float64)
        if fixture["observed_scale"] == "logit":
            values = 0.5 * (1.0 + np.tanh(0.5 * values))
        config = Sam3Config(
            checkpoint=fixture["checkpoint"],
            model_version=fixture["model_version"],
            strategy=Sam3Strategy.TEXT_PROMPT,
            prompts=(sample["prompt"],),
            mask_threshold=threshold,
        )
        runtime = Sam3ImageProcessorRuntime(processor=RecordedOutputProcessor(sample))
        discovery_input = _input_of(width=sample["width"], height=sample["height"])
        output = runtime.predict(discovery_input, _materialized_image(discovery_input), config)
        (proposal,) = output.proposals
        pairs.append((proposal.mask.as_array().tolist(), (values >= threshold).tolist()))
    return pairs


@pytest.mark.skipif(
    not SAM3_MASK_SCALE_FIXTURE.is_file(),
    reason=(
        "#610: real SAM3 masks_logits have not been recorded yet. Run "
        "experiments/sam3-mask-scale-20260927/scripts/sam3_mask_scale.py on a machine with the "
        "SAM3 checkpoint (see experiments/sam3-mask-scale-20260927/README.md) and commit the "
        "samples.json it writes as tests/fixtures/sam3_mask_scale/samples.json"
    ),
)
def test_the_runtime_thresholds_masks_logits_on_the_scale_measured_from_the_real_sdk() -> None:
    fixture = json.loads(SAM3_MASK_SCALE_FIXTURE.read_text(encoding="utf-8"))

    _pin_the_recorded_scale(fixture)
    for by_runtime, by_measured_scale in _runtime_and_measured_masks(fixture):
        assert by_runtime == by_measured_scale


def _synthetic_recording(scale: str, window: list[list[float]]) -> dict[str, Any]:
    """A fixture in the probe's format, for what the contract test does before real data."""
    flat = [value for row in window for value in row]
    return {
        "observed_scale": scale,
        "checkpoint": "facebook/sam3",
        "model_version": "synthetic",
        "mask_threshold": 0.5,
        "samples": [
            {
                "prompt": "floor",
                "box": [0.0, 0.0, 2.0, 2.0],
                "score": 0.9,
                "width": len(window[0]),
                "height": len(window),
                "masks_logits": window,
                "full_min": min(flat),
                "full_max": max(flat),
            }
        ],
    }


def test_the_contract_test_passes_on_probabilities_as_the_runtime_reads_them_today() -> None:
    recording = _synthetic_recording("probability", [[0.1, 0.6], [0.4, 0.9]])

    _pin_the_recorded_scale(recording)
    assert _runtime_and_measured_masks(recording) == [
        ([[False, True], [False, True]], [[False, True], [False, True]])
    ]


def test_the_contract_test_fails_against_the_current_comparison_if_logits_are_measured() -> None:
    # Se a medição real mostrar logits, o teste acima fica vermelho contra sam3.py hoje: 0.2 é
    # fundo pela comparação direta com 0,5 e primeiro plano por sigmoid(0.2) >= 0,5.
    recording = _synthetic_recording("logit", [[-3.0, 0.2], [0.6, 4.0]])

    _pin_the_recorded_scale(recording)
    assert _runtime_and_measured_masks(recording) == [
        ([[False, False], [True, True]], [[False, True], [True, True]])
    ]
