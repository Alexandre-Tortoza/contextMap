"""API pública da capability State Estimation.

State Estimation possui a pose dinâmica do rig: onde o frame do corpo está no frame do mapa ao
longo do tempo. Publica contratos independentes de backend (``PoseEstimate``, ``Trajectory``)
para que Geometric Mapping e Sensor Association nunca dependam de mensagens ROS, de tipos
nativos de um estimador ou do formato de pose de um dataset.

Esta fachada é a única superfície cross-capability. Os backends concretos (``backends/``) não
são reexportados: só o composition root em ``runtime`` os importa. Visão geral e mapa de
arquivos: ``src/contextmap/state_estimation/docs/README.md``.
"""

from contextmap.state_estimation.frame_graph import (
    FrameGraphError,
    LoopInconsistency,
    ResolvedTransform,
    StaticFrameGraph,
)
from contextmap.state_estimation.lookup import (
    ClockDomainMismatchError,
    LookupMode,
    LookupOutcome,
    LookupPolicy,
    LookupRejection,
    PoseLookupResult,
    RejectedLookup,
    ResolvedPose,
    TemporalAlignmentSummary,
    TrajectoryLookup,
    summarize_lookups,
)
from contextmap.state_estimation.models import (
    EstimatorProvenance,
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
    TimeBounds,
    Trajectory,
    TrajectoryGap,
    TrajectoryId,
    TrajectoryProvenance,
    TrajectoryQualitySummary,
    pose_estimate_id_for,
)
from contextmap.state_estimation.motion import (
    DistributionSummary,
    MotionDelta,
    MotionSummary,
    motion_deltas,
    summarize_distribution,
    summarize_motion,
)
from contextmap.state_estimation.ports import (
    DiagnosticSeverity,
    EstimationDiagnostic,
    MissingEstimatorInputError,
    StateEstimationError,
    StateEstimationRequest,
    StateEstimationResult,
    StateEstimator,
)
from contextmap.state_estimation.preflight import (
    ClockCheck,
    DownstreamReadiness,
    FrameGraphSummary,
    GeometryPreflightReport,
    PreflightFinding,
    PreflightStatus,
    PreflightTolerances,
    TransformCheck,
    calibration_identity,
    run_geometry_preflight,
)
from contextmap.state_estimation.requirements import (
    BODY_ENDPOINT,
    GeometryRequirements,
    StaticRelationRequirement,
)
from contextmap.state_estimation.run_artifact import (
    IncompleteRunArtifactError,
    RunArtifactError,
    StateEstimationDebugLevel,
    StateEstimationRunId,
    StateEstimationRunManifest,
    StateEstimationRunReader,
    StateEstimationRunWriter,
)
from contextmap.state_estimation.service import (
    GeometryPreflightError,
    StateEstimationOutcome,
    execute_state_estimation,
)

__all__ = [
    "BODY_ENDPOINT",
    "ClockCheck",
    "ClockDomainMismatchError",
    "DiagnosticSeverity",
    "DistributionSummary",
    "DownstreamReadiness",
    "EstimationDiagnostic",
    "EstimatorProvenance",
    "FrameGraphError",
    "FrameGraphSummary",
    "GeometryPreflightError",
    "GeometryPreflightReport",
    "GeometryRequirements",
    "IncompleteRunArtifactError",
    "LookupMode",
    "LookupOutcome",
    "LookupPolicy",
    "LookupRejection",
    "LoopInconsistency",
    "MissingEstimatorInputError",
    "MotionDelta",
    "MotionSummary",
    "PoseEstimate",
    "PoseEstimateId",
    "PoseLookupResult",
    "PoseProvenance",
    "PoseValidity",
    "PreflightFinding",
    "PreflightStatus",
    "PreflightTolerances",
    "RejectedLookup",
    "ResolvedPose",
    "ResolvedTransform",
    "RunArtifactError",
    "StateEstimationDebugLevel",
    "StateEstimationError",
    "StateEstimationOutcome",
    "StateEstimationRequest",
    "StateEstimationResult",
    "StateEstimationRunId",
    "StateEstimationRunManifest",
    "StateEstimationRunReader",
    "StateEstimationRunWriter",
    "StateEstimator",
    "StaticFrameGraph",
    "StaticRelationRequirement",
    "TemporalAlignmentSummary",
    "TimeBounds",
    "Trajectory",
    "TrajectoryGap",
    "TrajectoryId",
    "TrajectoryLookup",
    "TrajectoryProvenance",
    "TrajectoryQualitySummary",
    "TransformCheck",
    "calibration_identity",
    "execute_state_estimation",
    "motion_deltas",
    "pose_estimate_id_for",
    "run_geometry_preflight",
    "summarize_distribution",
    "summarize_lookups",
    "summarize_motion",
]
