"""``execute_stage_graphs_stage_major`` tells the caller when each frame finishes each stage."""

from __future__ import annotations

from collections.abc import Mapping

from contextmap.visual_perception import (
    StageDefinition,
    StageOutcome,
    StageStatus,
    execute_stage_graphs_stage_major,
)


def _graph(fail_second: bool) -> list[StageDefinition]:
    def second(_context: Mapping[str, object]) -> object:
        if fail_second:
            raise RuntimeError("boom")
        return "b"

    return [
        StageDefinition(stage_id="first", capability="x", run=lambda _c: "a"),
        StageDefinition(
            stage_id="second", capability="x", run=second, depends_on=frozenset({"first"})
        ),
    ]


def test_the_hook_runs_after_each_stage_of_each_frame_with_its_outcome() -> None:
    calls: list[tuple[int, str, StageStatus]] = []

    def hook(frame: int, stage_id: str, outcome: StageOutcome) -> None:
        calls.append((frame, stage_id, outcome.status))

    execute_stage_graphs_stage_major([_graph(False), _graph(True)], after_frame_stage=hook)

    assert calls == [
        (0, "first", StageStatus.SUCCEEDED),
        (1, "first", StageStatus.SUCCEEDED),
        (0, "second", StageStatus.SUCCEEDED),
        (1, "second", StageStatus.FAILED),
    ]
