"""Execução determinística dos estágios da Percepção Visual.

Este módulo executa um grafo de estágios já resolvido e isola falhas entre
branches independentes. Ele não constrói backends, não seleciona configuração
e não importa SDKs de modelo; a montagem do resultado permanece temporariamente
no mesmo módulo até sua próxima extração estrutural.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

STAGE_TRACEBACK_MAX_CHARS = 4000
"""Characters of a failed stage's traceback kept in its :class:`StageOutcome`.

A run records one outcome per stage per frame, so a failure repeated over thousands of frames must
not grow the persisted metrics without bound. The innermost part is kept: it holds the frame that
raised and the exception itself.
"""

StageRunner = Callable[[Mapping[str, object]], object]
"""A stage's execution logic.

Receives a mapping of already-completed upstream stage_id -> that
stage's output, and returns this stage's own output. The orchestration
service treats this value opaquely; only the caller assembling the
canonical pipeline needs to know its real type at each stage_id.
"""


class StageGraphError(Exception):
    """Raised when a stage graph is structurally invalid."""


class StageStatus(Enum):
    """The outcome of attempting to execute one stage."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, kw_only=True)
class StageDefinition:
    """One node in a resolved perception stage graph.

    Attributes:
        stage_id: Unique identity of this stage within the graph.
        capability: Capability port this stage exercises, e.g.
            ``"region_discovery"``. Used for diagnostics/provenance only
            — the executor calls ``run``, never a capability-specific
            method.
        run: This stage's execution logic.
        depends_on: ``stage_id``s that must succeed before this stage
            runs.
    """

    stage_id: str
    capability: str
    run: StageRunner
    depends_on: frozenset[str] = frozenset()


@dataclass(frozen=True, kw_only=True)
class StageOutcome:
    """The result of attempting to execute one stage.

    Attributes:
        stage_id: Which stage this outcome is for.
        status: Whether the stage succeeded, failed, or was skipped
            because an upstream dependency did not succeed.
        output: The stage's produced value. ``None`` unless ``status`` is
            ``SUCCEEDED``.
        error: Human-readable explanation. Set when ``status`` is
            ``FAILED`` (the raised exception's message, which may be empty)
            or ``SKIPPED`` (which dependency did not succeed).
        duration_ms: Wall-clock duration of this stage's own execution.
            ``0.0`` for a skipped stage.
        error_type: The raised exception's type name. Set only when
            ``status`` is ``FAILED``.
        error_traceback: The raised exception's formatted traceback, keeping
            only its last :data:`STAGE_TRACEBACK_MAX_CHARS` characters behind a
            ``[traceback truncated: ...]`` line when it is longer. Set only when
            ``status`` is ``FAILED``.
    """

    stage_id: str
    status: StageStatus
    output: object | None = None
    error: str | None = None
    duration_ms: float = 0.0
    error_type: str | None = None
    error_traceback: str | None = None


def execute_stage_graph(stages: Sequence[StageDefinition]) -> Sequence[StageOutcome]:
    """Execute a resolved stage graph, isolating failures per independent branch.

    Args:
        stages: The stages to execute. Order does not need to already be
            dependency-sorted; this function computes a deterministic
            topological order itself.

    Returns:
        One :class:`StageOutcome` per stage, in the deterministic
        execution order this function computed.

    Raises:
        StageGraphError: If ``stages`` has a duplicate ``stage_id``, a
            dependency on an unknown ``stage_id``, or a cycle.
    """
    ordered = _topological_order(stages)
    outcomes: dict[str, StageOutcome] = {}
    context: dict[str, object] = {}
    for stage in ordered:
        outcomes[stage.stage_id] = _run_stage(stage, outcomes, context)
    return tuple(outcomes[stage.stage_id] for stage in ordered)


def execute_stage_graphs_stage_major(
    graphs: Sequence[Sequence[StageDefinition]],
    *,
    after_stage: Callable[[str], None] | None = None,
    after_frame_stage: Callable[[int, str, StageOutcome], None] | None = None,
) -> tuple[Sequence[StageOutcome], ...]:
    """Execute several frames' graphs one stage at a time: a stage runs for every frame first.

    Each frame keeps its own context and outcomes, so a failure is isolated exactly as in
    :func:`execute_stage_graph`; only the interleaving changes. ``after_stage`` is called once
    a stage has run for every frame, which is where a caller releases a model whose stages are
    all done before the next stage loads its own (#689).

    Args:
        graphs: One stage graph per frame; every graph must hold the same stages.
        after_stage: Called with each ``stage_id`` after it ran for every frame.
        after_frame_stage: Called with the frame index, the ``stage_id`` and the outcome right
            after that stage ran (or failed, or was skipped) for that one frame, before the
            stage starts on the next frame. It lets a caller free per-frame data as soon as
            the last stage that reads it is done, instead of keeping it for every frame until
            the whole run ends. It must not raise: a hook error is not a stage failure.

    Returns:
        One outcome sequence per graph, in the order :func:`execute_stage_graph` returns it.

    Raises:
        StageGraphError: If a graph is invalid or the graphs do not hold the same stages.
    """
    orders = [_topological_order(graph) for graph in graphs]
    if not orders:
        return ()
    stage_ids = [stage.stage_id for stage in orders[0]]
    if any([stage.stage_id for stage in order] != stage_ids for order in orders[1:]):
        raise StageGraphError(
            "stage-major execution needs every frame's graph to hold the same stages"
        )
    outcomes: list[dict[str, StageOutcome]] = [{} for _ in orders]
    contexts: list[dict[str, object]] = [{} for _ in orders]
    for position, stage_id in enumerate(stage_ids):
        for frame, order in enumerate(orders):
            outcomes[frame][stage_id] = _run_stage(
                order[position], outcomes[frame], contexts[frame]
            )
            if after_frame_stage is not None:
                after_frame_stage(frame, stage_id, outcomes[frame][stage_id])
        if after_stage is not None:
            after_stage(stage_id)
    return tuple(
        tuple(frame_outcomes[stage_id] for stage_id in stage_ids) for frame_outcomes in outcomes
    )


def _run_stage(
    stage: StageDefinition, outcomes: Mapping[str, StageOutcome], context: dict[str, object]
) -> StageOutcome:
    """Run one stage of one frame, or skip it when a dependency did not succeed."""
    unmet = sorted(
        dep for dep in stage.depends_on if outcomes[dep].status is not StageStatus.SUCCEEDED
    )
    if unmet:
        return StageOutcome(
            stage_id=stage.stage_id,
            status=StageStatus.SKIPPED,
            error=f"upstream stage(s) did not succeed: {unmet}",
        )
    start = time.monotonic()
    try:
        output = stage.run(context)
    except Exception as error:
        # A stage's own failure must never crash the run or sibling
        # branches — only stages that depend on this one are skipped.
        return StageOutcome(
            stage_id=stage.stage_id,
            status=StageStatus.FAILED,
            error=str(error),
            duration_ms=(time.monotonic() - start) * 1000,
            error_type=type(error).__name__,
            error_traceback=_truncated_traceback(error),
        )
    context[stage.stage_id] = output
    return StageOutcome(
        stage_id=stage.stage_id,
        status=StageStatus.SUCCEEDED,
        output=output,
        duration_ms=(time.monotonic() - start) * 1000,
    )


def _truncated_traceback(error: BaseException) -> str:
    """Format ``error``'s traceback, keeping its innermost :data:`STAGE_TRACEBACK_MAX_CHARS`."""
    formatted = "".join(traceback.format_exception(error))
    if len(formatted) <= STAGE_TRACEBACK_MAX_CHARS:
        return formatted
    dropped = len(formatted) - STAGE_TRACEBACK_MAX_CHARS
    return (
        f"[traceback truncated: {dropped} leading characters dropped]\n"
        f"{formatted[-STAGE_TRACEBACK_MAX_CHARS:]}"
    )


def _topological_order(stages: Sequence[StageDefinition]) -> list[StageDefinition]:
    by_id = {stage.stage_id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise StageGraphError("duplicate stage_id in graph")

    for stage in stages:
        unknown = stage.depends_on - by_id.keys()
        if unknown:
            raise StageGraphError(
                f"stage {stage.stage_id!r} depends on unknown stages: {sorted(unknown)}"
            )

    resolved: list[StageDefinition] = []
    resolved_ids: set[str] = set()
    remaining = {stage.stage_id: stage for stage in stages}
    while remaining:
        ready = sorted(
            (stage for stage in remaining.values() if stage.depends_on <= resolved_ids),
            key=lambda stage: stage.stage_id,
        )
        if not ready:
            raise StageGraphError(f"cycle detected among stages: {sorted(remaining.keys())}")
        for stage in ready:
            resolved.append(stage)
            resolved_ids.add(stage.stage_id)
            del remaining[stage.stage_id]

    return resolved
