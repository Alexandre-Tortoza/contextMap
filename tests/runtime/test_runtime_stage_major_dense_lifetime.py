"""Stage-major perception must not keep every frame's dense payload alive until the end.

The deferred payload sink caches each dense array so mask pooling can reuse it without a second
inference. In a stage-major run ``record()`` only happens after every stage ran for every frame,
so the cache has to be released per frame, right after the last stage that reads it.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from runtime_perception import (
    _FakeFeatureExtractor,
    publish_region_run,
)
from runtime_pillow import fake_pillow  # noqa: F401
from test_independent_perception_executors import _sequence

from contextmap.runtime import StageRequest
from contextmap.runtime import executors as executors_module
from contextmap.runtime.executors import (
    ExecutorError,
    PerceptionSchedule,
    VisualPerceptionExecutor,
    _DeferredFeaturePayloadSink,
)
from contextmap.visual_perception.visual_features import FeatureScope


class _Payload(np.ndarray):  # type: ignore[type-arg]
    """ndarray subclass: a plain ndarray cannot be weakly referenced."""


class _Probe:
    """Records, at each event, which frames still hold a live dense payload."""

    def __init__(self) -> None:
        self.alive: dict[str, weakref.ref[Any]] = {}
        self.events: list[tuple[str, str, frozenset[str]]] = []

    def __call__(self, event: str, observation_id: object) -> None:
        gc.collect()
        live = frozenset(key for key, ref in self.alive.items() if ref() is not None)
        self.events.append((event, str(observation_id), live))

    def frames(self, event: str) -> list[str]:
        return [o for e, o, _ in self.events if e == event]

    def alive_when(self, event: str, observation_id: str) -> frozenset[str]:
        return next(live for e, o, live in self.events if (e, o) == (event, observation_id))

    def leaked(self) -> list[str]:
        gc.collect()
        return [key for key, ref in self.alive.items() if ref() is not None]


class _DenseThatQueuesPayload(_FakeFeatureExtractor):
    """Fake dense backend that queues one payload per frame through the sink it was given."""

    def __init__(self, sink: Any, probe: _Probe) -> None:
        super().__init__(FeatureScope.DENSE)
        self._sink, self._probe = sink, probe

    def extract(self, image: Any, regions: Any = ()) -> Any:
        self._probe("dense_extract", image.source_observation_id)
        features = list(super().extract(image, regions))
        array = np.zeros(2, dtype=np.float32).view(_Payload)
        self._probe.alive[str(image.source_observation_id)] = weakref.ref(array)
        self._sink.add_feature_payload(features[0], image.source_observation_id, array)
        return features


def _run(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pooling: bool,
    pool: Callable[..., Any] | None = None,
) -> _Probe:
    probe = _Probe()
    original = executors_module._append_mask_pooled_dense_features

    def observed_pooling(**kwargs: Any) -> Any:
        probe("pooling", kwargs["result"].source_observation_id)
        return (pool or original)(**kwargs)

    monkeypatch.setattr(executors_module, "_append_mask_pooled_dense_features", observed_pooling)
    executor = VisualPerceptionExecutor(
        dense_features=lambda scope: _DenseThatQueuesPayload(scope.payload_sink, probe),  # type: ignore[arg-type,return-value]
        region_features=lambda _scope: _FakeFeatureExtractor(FeatureScope.REGION),  # type: ignore[arg-type,return-value]
        perception_schedule=PerceptionSchedule.STAGE_MAJOR,
        dense_region_pooling_enabled=pooling,
    )
    sequence = _sequence(workspace, frames=3)
    executor.execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={
                "sequence": (sequence,),
                "regions": (publish_region_run(workspace, sequence),),
            },
            components={},
            config_digest="sha256:test",
            output_dir=workspace / "S1" / "run-0002" / "visual_perception",
            workspace=workspace,
            observation_selection=None,
        )
    )
    return probe


def test_the_sink_discards_only_the_named_observation() -> None:
    sink = _DeferredFeaturePayloadSink()
    sink._current_arrays[("obs-a", "f")] = object()
    sink._current_arrays[("obs-b", "f")] = object()

    sink.discard_for("obs-a")

    assert list(sink._current_arrays) == [("obs-b", "f")]


@pytest.mark.usefixtures("fake_pillow")
class TestDensePayloadLifetime:
    def test_without_pooling_a_payload_dies_before_the_next_frame_extracts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe = _run(tmp_path, monkeypatch, pooling=False)

        frames = probe.frames("dense_extract")
        assert len(frames) == 3
        # Quando o frame i+1 começa a extrair, o payload do frame i já não pode estar vivo.
        for previous, following in pairwise(frames):
            assert previous not in probe.alive_when("dense_extract", following)
        assert probe.leaked() == []

    def test_with_pooling_a_payload_dies_after_its_own_pooling_while_later_frames_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe = _run(tmp_path, monkeypatch, pooling=True)

        frames = probe.frames("pooling")
        assert len(frames) == 3
        for previous, following in pairwise(frames):
            assert previous not in probe.alive_when("pooling", following)
        assert probe.leaked() == []

    def test_a_failing_pooling_still_releases_every_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(**_kwargs: Any) -> Any:
            raise ExecutorError("pooling exploded")

        probe = _run(tmp_path, monkeypatch, pooling=True, pool=failing)

        frames = probe.frames("pooling")
        assert len(frames) == 3
        for previous, following in pairwise(frames):
            assert previous not in probe.alive_when("pooling", following)
        assert probe.leaked() == []
