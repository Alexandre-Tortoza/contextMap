from collections.abc import Mapping

import pytest

from contextmap.visual_perception import (
    StageDefinition,
    StageGraphError,
    StageStatus,
    execute_stage_graph,
)
from contextmap.visual_perception.execution.stage_execution import STAGE_TRACEBACK_MAX_CHARS


def test_independent_stages_run_in_deterministic_alphabetical_order() -> None:
    order: list[str] = []

    stages = [
        StageDefinition(stage_id="b", capability="fake", run=lambda ctx: order.append("b")),
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: order.append("a")),
    ]

    execute_stage_graph(stages)

    assert order == ["a", "b"]


def test_dependent_stage_runs_after_its_dependency() -> None:
    order: list[str] = []

    stages = [
        StageDefinition(
            stage_id="downstream",
            capability="fake",
            depends_on=frozenset({"upstream"}),
            run=lambda ctx: order.append("downstream"),
        ),
        StageDefinition(
            stage_id="upstream", capability="fake", run=lambda ctx: order.append("upstream")
        ),
    ]

    execute_stage_graph(stages)

    assert order == ["upstream", "downstream"]


def test_downstream_stage_receives_upstream_output() -> None:
    def upstream(context: Mapping[str, object]) -> int:
        return 42

    def downstream(context: Mapping[str, object]) -> int:
        return context["upstream"] * 2  # type: ignore[operator]

    stages = [
        StageDefinition(stage_id="upstream", capability="fake", run=upstream),
        StageDefinition(
            stage_id="downstream",
            capability="fake",
            depends_on=frozenset({"upstream"}),
            run=downstream,
        ),
    ]

    outcomes = execute_stage_graph(stages)

    by_id = {o.stage_id: o for o in outcomes}
    assert by_id["downstream"].output == 84


def test_failing_stage_does_not_crash_the_run() -> None:
    def failing(context: Mapping[str, object]) -> None:
        raise RuntimeError("backend exploded")

    outcomes = execute_stage_graph(
        [StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing)]
    )

    assert outcomes[0].status is StageStatus.FAILED
    assert "backend exploded" in (outcomes[0].error or "")


def test_a_failed_outcome_keeps_the_exception_type_and_its_traceback() -> None:
    """VP-11: ``error=str(error)`` was all a failure kept, and ``str(ValueError())`` is ``""``."""

    def failing(context: Mapping[str, object]) -> None:
        raise ValueError()

    outcome = execute_stage_graph(
        [StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing)]
    )[0]

    assert outcome.status is StageStatus.FAILED
    assert outcome.error_type == "ValueError"
    assert outcome.error_traceback is not None
    assert outcome.error_traceback.startswith("Traceback (most recent call last):")
    assert "in failing" in outcome.error_traceback
    assert outcome.error_traceback.endswith("ValueError\n")


def test_a_long_failure_traceback_keeps_only_its_innermost_part() -> None:
    # Duas funções alternadas: o traceback não colapsa linhas repetidas e passa do limite.
    def ping(depth: int) -> None:
        if depth == 0:
            raise RuntimeError("innermost failure")
        pong(depth - 1)

    def pong(depth: int) -> None:
        ping(depth - 1)

    def failing(context: Mapping[str, object]) -> None:
        ping(200)

    outcome = execute_stage_graph(
        [StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing)]
    )[0]

    assert outcome.error_traceback is not None
    assert outcome.error_traceback.startswith("[traceback truncated: ")
    assert len(outcome.error_traceback) <= STAGE_TRACEBACK_MAX_CHARS + 64
    assert outcome.error_traceback.endswith("RuntimeError: innermost failure\n")


def test_only_a_failed_outcome_carries_an_exception_type_and_traceback() -> None:
    def failing(context: Mapping[str, object]) -> None:
        raise RuntimeError("upstream exploded")

    outcomes = execute_stage_graph(
        [
            StageDefinition(stage_id="upstream", capability="fake", run=failing),
            StageDefinition(
                stage_id="downstream",
                capability="fake",
                depends_on=frozenset({"upstream"}),
                run=lambda ctx: None,
            ),
            StageDefinition(stage_id="independent", capability="fake", run=lambda ctx: None),
        ]
    )
    by_id = {outcome.stage_id: outcome for outcome in outcomes}

    assert by_id["upstream"].error_type == "RuntimeError"
    for stage_id in ("downstream", "independent"):
        assert by_id[stage_id].error_type is None
        assert by_id[stage_id].error_traceback is None


def test_dependent_of_failing_stage_is_skipped_but_independent_branch_still_runs() -> None:
    def failing(context: Mapping[str, object]) -> None:
        raise RuntimeError("region discovery exploded")

    def dependent(context: Mapping[str, object]) -> str:
        return "region features"

    def independent(context: Mapping[str, object]) -> str:
        return "dense features"

    stages = [
        StageDefinition(stage_id="region_discovery", capability="region_discovery", run=failing),
        StageDefinition(
            stage_id="region_feature_extraction",
            capability="feature_extractor",
            depends_on=frozenset({"region_discovery"}),
            run=dependent,
        ),
        StageDefinition(
            stage_id="dense_feature_extraction", capability="feature_extractor", run=independent
        ),
    ]

    outcomes = execute_stage_graph(stages)
    by_id = {o.stage_id: o for o in outcomes}

    assert by_id["region_discovery"].status is StageStatus.FAILED
    assert by_id["region_feature_extraction"].status is StageStatus.SKIPPED
    assert by_id["dense_feature_extraction"].status is StageStatus.SUCCEEDED
    assert by_id["dense_feature_extraction"].output == "dense features"


def test_duplicate_stage_id_is_rejected() -> None:
    stages = [
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: None),
        StageDefinition(stage_id="a", capability="fake", run=lambda ctx: None),
    ]

    with pytest.raises(StageGraphError, match="duplicate"):
        execute_stage_graph(stages)


def test_dependency_on_unknown_stage_is_rejected() -> None:
    stages = [
        StageDefinition(
            stage_id="a",
            capability="fake",
            depends_on=frozenset({"does-not-exist"}),
            run=lambda ctx: None,
        ),
    ]

    with pytest.raises(StageGraphError, match="unknown"):
        execute_stage_graph(stages)


def test_cycle_is_rejected() -> None:
    stages = [
        StageDefinition(
            stage_id="a", capability="fake", depends_on=frozenset({"b"}), run=lambda ctx: None
        ),
        StageDefinition(
            stage_id="b", capability="fake", depends_on=frozenset({"a"}), run=lambda ctx: None
        ),
    ]

    with pytest.raises(StageGraphError, match="cycle"):
        execute_stage_graph(stages)
