"""Orquestração do preflight de geometria.

Este módulo só encadeia os passos: resumir as observações, avaliar a calibração estática, os
requisitos de relações e os clocks, e montar o relatório. Os detalhes de cada passo ficam em
``facts``, ``calibration``, ``readiness`` e ``clocks``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from contextmap.ingestion import CalibrationSet, FrameId, SourceObservation
from contextmap.state_estimation.preflight.calibration import (
    StaticCalibrationAssessment,
    assess_static_calibration,
    calibration_findings,
    calibration_identity,
    unused_calibration_warnings,
)
from contextmap.state_estimation.preflight.clocks import evaluate_clock_requirements
from contextmap.state_estimation.preflight.facts import ObservationFacts, gather_observation_facts
from contextmap.state_estimation.preflight.models import (
    CONVENTIONS,
    DownstreamReadiness,
    FrameGraphSummary,
    GeometryPreflightReport,
    PreflightFinding,
    PreflightTolerances,
    TransformCheck,
)
from contextmap.state_estimation.preflight.readiness import evaluate_requirements
from contextmap.state_estimation.requirements import GeometryRequirements


def run_geometry_preflight(
    *,
    observations: Iterable[SourceObservation],
    calibration: CalibrationSet | None,
    execution: GeometryRequirements,
    downstream: Sequence[GeometryRequirements] = (),
    tolerances: PreflightTolerances | None = None,
) -> GeometryPreflightReport:
    """Check that the frames, extrinsics and clocks a run needs are trustworthy.

    Only the requirements of ``execution`` can block. ``downstream``
    capabilities are evaluated the same way but reported as readiness, so a
    LiDAR-inertial run is not blocked for missing camera intrinsics.

    Args:
        observations: The selected canonical observations.
        calibration: The sequence's canonical calibration, or ``None``.
        execution: What the capability about to run needs; it must declare
            ``reference_frame`` and ``body_frame``.
        downstream: What later capabilities will need.
        tolerances: Numeric tolerances; defaults are documented on
            :class:`PreflightTolerances`.

    Returns:
        The report; ``status`` is ``BLOCKED`` when the run must not start.

    Raises:
        ValueError: If ``execution`` does not declare its dynamic frames.
    """
    if execution.reference_frame is None or execution.body_frame is None:
        raise ValueError("the executing capability must declare its reference_frame and body_frame")
    limits = tolerances if tolerances is not None else PreflightTolerances()
    body_frame = execution.body_frame

    facts = gather_observation_facts(observations)
    static = assess_static_calibration(calibration, limits)
    transform_checks = list(static.checks)

    findings, relation_checks, touched = evaluate_requirements(
        execution, facts=facts, calibration=calibration, graph=static.graph, body_frame=body_frame
    )
    transform_checks.extend(relation_checks)
    blockers = [*findings, *calibration_findings(static, touched)]
    warnings = unused_calibration_warnings(static, touched)

    clock_checks, clock_blockers = evaluate_clock_requirements(execution, facts)
    blockers.extend(clock_blockers)

    readiness: list[DownstreamReadiness] = []
    for requirements in downstream:
        capability_readiness, downstream_checks = _assess_downstream_readiness(
            requirements,
            facts=facts,
            calibration=calibration,
            static=static,
            body_frame=body_frame,
        )
        transform_checks.extend(downstream_checks)
        readiness.append(capability_readiness)
        if not capability_readiness.ready:
            warnings.append(
                PreflightFinding(
                    code="downstream_not_ready",
                    message=f"{requirements.capability} lacks "
                    f"{sorted({finding.code for finding in capability_readiness.missing})}",
                )
            )

    return GeometryPreflightReport(
        required_inputs=execution.modalities,
        available_inputs=facts.modalities,
        frame_graph=FrameGraphSummary(
            reference_frame=execution.reference_frame,
            body_frame=body_frame,
            static_frames=tuple(sorted(static.graph.frames, key=str)),
            static_edges=static.graph.edges,
            component_count=len(
                {frozenset(static.graph.component_of(frame)) for frame in static.graph.frames}
            ),
        ),
        calibration_identity=calibration_identity(calibration),
        transform_checks=tuple(transform_checks),
        clock_checks=tuple(clock_checks),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        downstream=tuple(readiness),
        conventions=dict(CONVENTIONS),
    )


def _assess_downstream_readiness(
    requirements: GeometryRequirements,
    *,
    facts: ObservationFacts,
    calibration: CalibrationSet | None,
    static: StaticCalibrationAssessment,
    body_frame: FrameId,
) -> tuple[DownstreamReadiness, list[TransformCheck]]:
    """Evaluate a later capability's prerequisites without letting them block the run."""
    missing, checks, touched = evaluate_requirements(
        requirements,
        facts=facts,
        calibration=calibration,
        graph=static.graph,
        body_frame=body_frame,
    )
    missing = [*missing, *calibration_findings(static, touched)]
    readiness = DownstreamReadiness(
        capability=requirements.capability, ready=not missing, missing=tuple(missing)
    )
    return readiness, checks
