"""Contracts for geometric mapping: sources, policy, and plan.

Separates concerns of data availability, processing decisions, and validated execution plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from contextmap.geometric_mapping.deskew import PoseInterpolationDeskewPolicy
from contextmap.geometric_mapping.motion_correction import (
    MotionCorrectionPolicy,
    MotionCorrectionRecord,
)
from contextmap.ingestion import FrameId, SequenceArtifactId, SequenceSelection, SourceObservationId
from contextmap.state_estimation import LookupPolicy, StateEstimationRunId, TrajectoryId


@dataclass(frozen=True, kw_only=True)
class GeometryMappingSources:
    """Available data and evidence for geometry mapping.

    Represents the concrete data artifacts and observations available for one mapping run.
    These sources are fixed once the run begins; they do not change during processing.

    Attributes:
        sequence_artifact_id: The canonical sequence the scans come from.
        selection: Selection specification over that sequence.
        selection_id: Deterministic identity of the selection.
        trajectory_id: State estimation trajectory providing poses.
        state_estimation_run_id: State estimation run identity (if available).
        map_frame: Target frame for the map (trajectory's reference frame).
        body_frame: Rig's body frame.
        calibration_identity: Identity of the extrinsic calibration used.
        motion_correction: Evidence of motion correction per scan (if available).
    """

    sequence_artifact_id: SequenceArtifactId
    selection: SequenceSelection
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    map_frame: FrameId
    body_frame: FrameId
    calibration_identity: str | None
    motion_correction: Mapping[SourceObservationId, MotionCorrectionRecord] | None = None


@dataclass(frozen=True, kw_only=True)
class GeometryMappingPolicy:
    """Processing decisions that govern geometry mapping.

    Represents the algorithmic and operational choices made for one mapping run.
    Two runs with the same sources but different policies produce different geometries.

    Attributes:
        pose_lookup: Policy for resolving poses from the trajectory.
        motion_correction_policy: Policy for motion correction verification and application.
        deskew: Policy for per-point motion correction (if selected).
    """

    pose_lookup: LookupPolicy
    motion_correction_policy: MotionCorrectionPolicy
    deskew: PoseInterpolationDeskewPolicy | None = None
