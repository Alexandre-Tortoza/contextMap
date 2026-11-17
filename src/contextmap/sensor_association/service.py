"""Execução streaming de Sensor Association.

Valida lineage e entradas, coordena os passos científicos e entrega cada frame ao sink.
O serviço não possui persistência nem retém arrays dos frames concluídos.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from contextmap.geometric_mapping import GeometricMap, GeometryBlockSource
from contextmap.ingestion import (
    CalibrationSet,
    ImageObservation,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.region_perception import Region2D
from contextmap.sensor_association.dense_sampling import (
    DenseFeatureSamples,
    InterpolationPolicy,
    sample_dense_features,
)
from contextmap.sensor_association.diagnostics import (
    DiagnosticTolerances,
    FrameDiagnostics,
    TrustedCorrespondences,
    diagnose_frame,
    reprojection_statistics,
)
from contextmap.sensor_association.errors import AssociationInputError
from contextmap.sensor_association.membership import (
    MEMBERSHIP_POLICY_ID,
    BoxDepthComponentPolicy,
    FrameMembership,
    RegionMaskLoader,
    associate_regions,
    build_spatial_observations,
    configured_membership_policy_id,
)
from contextmap.sensor_association.models import SpatialObservation
from contextmap.sensor_association.projection_chain.candidate_geometry import (
    CandidateGeometryPolicy,
)
from contextmap.sensor_association.projection_chain.frame_projection import (
    FrameProjector,
    RejectedProjection,
)
from contextmap.sensor_association.projection_chain.visibility import (
    OcclusionPolicy,
    VisibilityResolution,
    resolve_visibility,
)
from contextmap.sensor_association.quality import ObservationQuality
from contextmap.sensor_association.quality_derivation import derive_observation_quality
from contextmap.sensor_association.run_definitions import fingerprint_definitions
from contextmap.state_estimation import (
    LookupPolicy,
    StateEstimationRunId,
    TrajectoryId,
    TrajectoryLookup,
    calibration_identity,
)
from contextmap.visual_perception import (
    DenseFeatureMap,
    PreparedImage,
    VisualEvidenceResult,
)


@dataclass(frozen=True, kw_only=True)
class DenseChannel:
    """One declared dense-feature evidence channel of a run.

    Attributes:
        channel_id: Identity of the channel within the run, e.g. ``"dinov2-native"`` or
            ``"dinov2-enhanced"``. Channels are never combined.
        interpolation: How the channel's feature map is read at a pixel.
    """

    channel_id: str
    interpolation: InterpolationPolicy

    def __post_init__(self) -> None:
        """Require a channel identity.

        Raises:
            ValueError: If ``channel_id`` is empty.
        """
        if not self.channel_id:
            raise ValueError("channel_id must not be empty")


@dataclass(frozen=True, kw_only=True)
class AssociationFrameInput:
    """Everything one camera frame contributes to a run.

    Attributes:
        observation: The RGB observation.
        prepared_image: The image Visual Perception prepared from it.
        region_artifact_id: The selected regional producer.
        regions: Its regions for this physical observation.
        visual_artifact_id: Optional dense-feature producer.
        visual_evidence: Optional independently selected visual evidence.
        dense_maps: The dense feature map of each declared channel, by ``channel_id``.
        correspondences: Trusted reference correspondences for this frame, when they exist.
    """

    observation: ImageObservation
    prepared_image: PreparedImage
    region_artifact_id: str
    regions: tuple[Region2D, ...]
    visual_artifact_id: str | None = None
    visual_evidence: VisualEvidenceResult | None = None
    dense_maps: Mapping[str, DenseFeatureMap] = field(default_factory=dict)
    correspondences: TrustedCorrespondences | None = None


@dataclass(frozen=True, kw_only=True)
class SensorAssociationRequest:
    """The inputs of one association run.

    Attributes:
        association_run_id: Identity of the execution producing spatial observations.
        sequence_artifact_id: The canonical sequence the run belongs to.
        selection_id: Deterministic identity of the sequence selection.
        geometry: The block read boundary of the persistent map.
        trajectory: The pose lookup of the selected state-estimation trajectory.
        pose_policy: Which pose lookups are acceptable for an image timestamp.
        calibration: The canonical calibration, the one the map and the trajectory used.
        candidate_policy: Which map geometry each frame evaluates, before projection.
        occlusion_policy: The visibility parameters.
        tolerances: The diagnostic tolerances.
        frames: The camera frames to associate, in order. It is consumed **once**, so a
            generator is welcome and is what keeps the run's input side bounded too: a caller
            that yields one frame at a time never holds every image payload at once.
        dense_channels: The declared dense-feature evidence channels.
        state_estimation_run_id: The persisted state-estimation run the trajectory came from.
        code_version: Code revision that produces the run, when known.
        mask_loader: Resolves a region's mask when a frame's regional artifact was
            reopened from a persisted run and no longer carries it inline, for
            example that run's ``RegionPerceptionRunReader.mask_store()``. ``None`` keeps a
            region without an inline mask and no resolvable reference as skipped.
        box_policy: Enables the box rule of membership for regions without mask
            (``box-depth-component-v1``). ``None`` keeps them skipped (``mask-membership-v1``).
    """

    association_run_id: str
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    geometry: GeometryBlockSource
    trajectory: TrajectoryLookup
    pose_policy: LookupPolicy
    calibration: CalibrationSet
    candidate_policy: CandidateGeometryPolicy
    occlusion_policy: OcclusionPolicy
    tolerances: DiagnosticTolerances
    frames: Iterable[AssociationFrameInput]
    dense_channels: tuple[DenseChannel, ...] = ()
    state_estimation_run_id: StateEstimationRunId | None = None
    code_version: str | None = None
    mask_loader: RegionMaskLoader | None = None
    box_policy: BoxDepthComponentPolicy | None = None


@dataclass(frozen=True, kw_only=True, eq=False)
class FrameAssociation:
    """The association of one camera frame.

    Attributes:
        source_observation_id: The camera frame.
        resolution: The visibility of every map point.
        membership: Which region masks contain the visible points.
        observations: One spatial observation per evaluated region.
        qualities: The quality of each observation, in the same order.
        diagnostics: The frame's diagnostic report.
        dense_samples: The dense feature sampling of each declared channel, by ``channel_id``.
    """

    source_observation_id: SourceObservationId
    resolution: VisibilityResolution
    membership: FrameMembership
    observations: tuple[SpatialObservation, ...]
    qualities: tuple[ObservationQuality, ...]
    diagnostics: FrameDiagnostics
    dense_samples: Mapping[str, DenseFeatureSamples]


class FrameSink(Protocol):
    """Capability port: consume each frame the service completes, one at a time.

    The sink is what makes bounded-memory execution possible: the service hands a frame
    over and then drops its own reference, so whatever the sink does not keep is
    collectable before the next frame is projected. A sink that accumulates every frame is
    a valid choice for a small run, and the choice is the caller's, not the service's.
    """

    def accept(self, frame: FrameAssociation) -> None:
        """Take one completed frame.

        Raising aborts the run; a sink that persists must leave nothing publishable behind.
        """
        ...


@dataclass(frozen=True, kw_only=True, eq=False)
class SensorAssociationOutcome:
    """The result of an association run, ready to be persisted.

    Attributes:
        geometric_map: The map's identity, frame and provenance.
        sequence_artifact_id: The canonical sequence.
        selection_id: The sequence selection.
        trajectory_id: The trajectory that placed the camera.
        state_estimation_run_id: The state-estimation run it came from, when there is one.
        calibration_identity: Hash of the calibration used.
        association_run_id: The execution that produced the spatial evidence.
        region_artifact_ids: The selected regional artifacts, sorted.
        visual_artifact_ids: Optional selected dense-feature artifacts, sorted.
        candidate_policy: The candidate rule applied to every frame.
        occlusion_policy: The visibility parameters applied.
        box_policy: The box rule of membership, when it was enabled.
        pose_policy: The pose lookup policy applied.
        tolerances: The diagnostic tolerances applied.
        dense_channels: The declared dense-feature channels.
        configuration_fingerprint: Hash of the effective configuration of the run.
        code_version: Code revision that produced the run, when known.
        frame_count: Frames that were associated and handed to the sink, in request order.
            The frames themselves are not here: they were released as they completed.
        rejected: The frames whose pose the lookup policy rejected.
    """

    association_run_id: str
    geometric_map: GeometricMap
    sequence_artifact_id: SequenceArtifactId
    selection_id: str
    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    calibration_identity: str
    region_artifact_ids: tuple[str, ...]
    visual_artifact_ids: tuple[str, ...]
    candidate_policy: CandidateGeometryPolicy
    occlusion_policy: OcclusionPolicy
    pose_policy: LookupPolicy
    tolerances: DiagnosticTolerances
    dense_channels: tuple[DenseChannel, ...]
    configuration_fingerprint: str
    code_version: str | None
    frame_count: int
    rejected: tuple[RejectedProjection, ...]
    box_policy: BoxDepthComponentPolicy | None = None

    @property
    def membership_policy_id(self) -> str:
        """The membership policy the run was configured with (not the per-region rule)."""
        return configured_membership_policy_id(self.box_policy)


class SensorAssociationService:
    """Runs Sensor Association over the frames of a sequence selection."""

    def run(
        self, request: SensorAssociationRequest, *, sink: FrameSink
    ) -> SensorAssociationOutcome:
        """Associate every frame of a request, streaming each one to the sink.

        A frame is projected, resolved, associated, measured, diagnosed, sampled, handed to
        the sink and then dropped, so no frame's arrays outlive the frame after it.

        Args:
            request: The inputs of the run.
            sink: Where each completed frame goes. Whatever it does not keep is released.

        Returns:
            The run's identity, policies, aggregates and the frames whose pose was
            rejected. The frames are not returned; the sink received them.

        Raises:
            AssociationInputError: If the map cannot be read in blocks, upstream sequences
                disagree with the request, a region belongs to another observation,
                channel or frame identities repeat, a declared dense map is missing, or
                any step finds incompatible inputs. Frame lineage is checked before pose
                lookup, including frames whose poses would be rejected.
        """
        channel_ids = _validate_request(request)
        identity = calibration_identity(request.calibration)
        if identity is None:
            raise AssociationInputError("the calibration set has no identity")
        fingerprint = _configuration_fingerprint(request)
        projector = FrameProjector(
            geometry=request.geometry,
            candidate_policy=request.candidate_policy,
            trajectory=request.trajectory,
            pose_policy=request.pose_policy,
            calibration=request.calibration,
            state_estimation_run_id=request.state_estimation_run_id,
        )
        frame_count = 0
        rejected: list[RejectedProjection] = []
        seen_observations: set[SourceObservationId] = set()
        regional_runs: set[str] = set()
        visual_runs: set[str] = set()
        for frame_input in request.frames:
            _validate_frame(frame_input, channel_ids, seen_observations)
            regional_runs.add(frame_input.region_artifact_id)
            if channel_ids and frame_input.visual_artifact_id is not None:
                visual_runs.add(frame_input.visual_artifact_id)
            projection = projector.project(frame_input.observation, frame_input.prepared_image)
            if isinstance(projection, RejectedProjection):
                rejected.append(projection)
                continue
            resolution = resolve_visibility(projection, request.occlusion_policy)
            membership = associate_regions(
                resolution,
                frame_input.regions,
                region_artifact_id=frame_input.region_artifact_id,
                sequence_artifact_id=request.sequence_artifact_id,
                mask_loader=request.mask_loader,
                box_policy=request.box_policy,
            )
            observations = build_spatial_observations(
                membership,
                association_run_id=request.association_run_id,
                configuration_fingerprint=fingerprint,
                code_version=request.code_version,
            )
            statistics = (
                None
                if frame_input.correspondences is None
                else reprojection_statistics(projection, frame_input.correspondences)
            )
            dense_samples = {
                channel.channel_id: sample_dense_features(
                    resolution,
                    _visual_evidence(frame_input),
                    frame_input.dense_maps[channel.channel_id],
                    interpolation=channel.interpolation,
                )
                for channel in request.dense_channels
            }
            sink.accept(
                FrameAssociation(
                    source_observation_id=projection.source_observation_id,
                    resolution=resolution,
                    membership=membership,
                    observations=observations,
                    qualities=derive_observation_quality(
                        membership, observations, reprojection=statistics
                    ),
                    diagnostics=diagnose_frame(
                        resolution,
                        tolerances=request.tolerances,
                        membership=membership,
                        correspondences=frame_input.correspondences,
                        dense_samples=tuple(dense_samples.values()),
                    ),
                    dense_samples=dense_samples,
                )
            )
            frame_count += 1
            # Solta as referências locais antes do próximo frame: o que o sink não guardou
            # é coletável agora, e é isso que mantém o pico de memória por frame.
            del projection, resolution, membership, observations, dense_samples, statistics
        geometric_map = request.geometry.geometric_map
        return SensorAssociationOutcome(
            association_run_id=request.association_run_id,
            geometric_map=geometric_map,
            sequence_artifact_id=request.sequence_artifact_id,
            selection_id=request.selection_id,
            trajectory_id=request.trajectory.trajectory.trajectory_id,
            state_estimation_run_id=request.state_estimation_run_id,
            calibration_identity=identity,
            region_artifact_ids=tuple(sorted(regional_runs)),
            visual_artifact_ids=tuple(sorted(visual_runs)),
            candidate_policy=request.candidate_policy,
            occlusion_policy=request.occlusion_policy,
            pose_policy=request.pose_policy,
            tolerances=request.tolerances,
            dense_channels=request.dense_channels,
            configuration_fingerprint=fingerprint,
            code_version=request.code_version,
            frame_count=frame_count,
            rejected=tuple(rejected),
            box_policy=request.box_policy,
        )


def _validate_request(request: SensorAssociationRequest) -> tuple[str, ...]:
    """Check what can be checked before the first frame, and return the declared channels.

    Raises:
        AssociationInputError: If the map cannot be read in blocks, the declared sequence
            differs from an upstream sequence, or channel identities repeat.
    """
    if not request.association_run_id:
        raise AssociationInputError("association_run_id must identify the producing execution")
    if not isinstance(request.geometry, GeometryBlockSource):
        raise AssociationInputError(
            "sensor association reads the map in vectorized blocks: "
            f"{type(request.geometry).__name__} does not implement GeometryBlockSource"
        )
    for owner, sequence_id in (
        ("map", request.geometry.geometric_map.provenance.sequence_artifact_id),
        ("trajectory", request.trajectory.trajectory.provenance.sequence_artifact_id),
    ):
        if sequence_id != request.sequence_artifact_id:
            raise AssociationInputError(
                f"the request declares sequence {request.sequence_artifact_id!r} "
                f"but the {owner} comes from sequence {sequence_id!r}"
            )
    channel_ids = tuple(channel.channel_id for channel in request.dense_channels)
    if len(set(channel_ids)) != len(channel_ids):
        raise AssociationInputError(
            f"dense channel identities must be unique, got {list(channel_ids)}"
        )
    return channel_ids


def _validate_frame(
    frame: AssociationFrameInput,
    channel_ids: tuple[str, ...],
    seen: set[SourceObservationId],
) -> None:
    """Validate one frame as it arrives, so the run never has to hold them all.

    Raises:
        AssociationInputError: If a frame repeats an observation, lacks a selected producer,
            or a declared dense channel is missing.
    """
    observation_id = frame.observation.observation_id
    if observation_id in seen:
        raise AssociationInputError("a frame cannot appear twice in a run")
    seen.add(observation_id)
    if not frame.region_artifact_id:
        raise AssociationInputError("frame requires selected region_artifact_id")
    if any(region.source_observation_id not in (None, observation_id) for region in frame.regions):
        raise AssociationInputError("region source observation does not match frame")
    if channel_ids and (frame.visual_evidence is None or not frame.visual_artifact_id):
        raise AssociationInputError("dense channels require explicit visual artifact and evidence")
    if (
        frame.visual_evidence is not None
        and frame.visual_evidence.source_observation_id != observation_id
    ):
        raise AssociationInputError("visual evidence belongs to another physical observation")
    missing = [channel for channel in channel_ids if channel not in frame.dense_maps]
    if missing:
        raise AssociationInputError(
            f"frame {observation_id!r} has no dense map for the declared channel(s) {missing}"
        )
    for channel in channel_ids:
        if frame.dense_maps[channel].source_artifact_id != frame.visual_artifact_id:
            raise AssociationInputError("dense map belongs to an unselected visual artifact")


def _configuration_fingerprint(request: SensorAssociationRequest) -> str:
    """Hash effective policies and definition versions using persisted key spellings."""
    pose = request.pose_policy
    tolerances = request.tolerances
    payload: dict[str, Any] = {
        "candidates": request.candidate_policy.to_record(),
        "occlusion": request.occlusion_policy.to_record(),
        "pose_policy": {
            "mode": pose.mode.value,
            "max_time_delta_ns": pose.max_time_delta_ns,
            "max_interpolation_gap_ns": pose.max_interpolation_gap_ns,
        },
        "tolerances": {
            "max_pose_time_delta_ns": tolerances.max_pose_time_delta_ns,
            "max_map_window_offset_ns": tolerances.max_map_window_offset_ns,
            "max_reprojection_p95_px": tolerances.max_reprojection_p95_px,
            "max_reprojection_invalid_rate": tolerances.max_reprojection_invalid_rate,
        },
        "dense_channels": [
            {"channel_id": c.channel_id, "interpolation": c.interpolation.value}
            for c in request.dense_channels
        ],
        "policies": {
            "membership": (
                MEMBERSHIP_POLICY_ID
                if request.box_policy is None
                else request.box_policy.to_record()
            ),
            **fingerprint_definitions(),
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _visual_evidence(frame: AssociationFrameInput) -> VisualEvidenceResult:
    """Return the preflight-validated visual channel input without a fallback."""
    if frame.visual_evidence is None:
        raise AssociationInputError("dense channel requires visual evidence")
    return frame.visual_evidence
