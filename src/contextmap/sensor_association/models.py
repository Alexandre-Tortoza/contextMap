"""Contratos canônicos da evidência espacial de Sensor Association.

Preserva referências, identidade, pose, calibração e provenance com validação local.
Não possui projeção, persistência ou interpretação semântica.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NewType

from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SequenceArtifactId,
    SourceObservationId,
)
from contextmap.region_perception import (
    RegionId,
)
from contextmap.state_estimation import (
    LookupOutcome,
    PoseEstimateId,
    StateEstimationRunId,
    TrajectoryId,
)

SpatialObservationId = NewType("SpatialObservationId", str)
"""Identity of one region's spatial observation."""

PixelCoordinate = tuple[float, float]
"""``(u, v)`` in pixels, continuous, with the pixel *center* at the integer."""


def spatial_observation_id_for(
    *, association_run_id: str, source_observation_id: SourceObservationId, region_id: RegionId
) -> SpatialObservationId:
    """Identify one regional association within its own execution and physical frame."""
    if not association_run_id or not str(source_observation_id) or not str(region_id):
        raise ValueError(
            "spatial observation requires association run, observation and region identities"
        )
    return SpatialObservationId(
        f"spatial--{association_run_id}--{source_observation_id}--{region_id}"
    )


class DepthMetric(Enum):
    """How the depth of a point is measured for occlusion and diagnostics.

    Attributes:
        OPTICAL_AXIS: The ``z`` coordinate in the camera optical frame. Natural for a
            perspective camera, where a fronto-parallel surface has constant depth.
        RAY_RANGE: The distance from the optical center along the viewing ray. Used
            when the field of view can exceed a hemisphere, so ``z`` can be non-positive.
    """

    OPTICAL_AXIS = "optical_axis"
    RAY_RANGE = "ray_range"


class VisibilityState(Enum):
    """Why a candidate geometry element is, or is not, evidence for a region.

    Attributes:
        BEHIND_CAMERA: The camera model cannot project the point.
        OUTSIDE_IMAGE: It projects outside the prepared image.
        OUTSIDE_VALID_SUPPORT: It lands inside the prepared image but outside the
            valid support or inside an exclusion region.
        OCCLUDED: A nearer surface hides it along the same ray.
        VISIBLE_UNASSIGNED: It is visible but falls in no region's mask.
        ASSOCIATED: It is visible and inside the region's mask.
    """

    BEHIND_CAMERA = "behind_camera"
    OUTSIDE_IMAGE = "outside_image"
    OUTSIDE_VALID_SUPPORT = "outside_valid_support"
    OCCLUDED = "occluded"
    VISIBLE_UNASSIGNED = "visible_unassigned"
    ASSOCIATED = "associated"


class FootprintKind(Enum):
    """What delimits a region's footprint in the prepared image.

    Attributes:
        MASK: The region's frozen mask; membership is the mask rule.
        BOX: The region's bounding box, for a region with no mask; the support is the 3D
            component the box rule selects inside it, never the whole box.
    """

    MASK = "mask"
    BOX = "box"


_PIXEL_REQUIRED = frozenset(
    {
        VisibilityState.OUTSIDE_VALID_SUPPORT,
        VisibilityState.OCCLUDED,
        VisibilityState.VISIBLE_UNASSIGNED,
        VisibilityState.ASSOCIATED,
    }
)


def _require_finite_pair(name: str, pixel: PixelCoordinate | None) -> None:
    """Reject a pixel coordinate unless both components are finite."""
    if pixel is not None and not all(math.isfinite(value) for value in pixel):
        raise ValueError(f"{name} must be finite, got {pixel!r}")


def _require_depth(name: str, value: float, *, allow_negative: bool = False) -> None:
    """Reject a provided depth unless it is finite and positive."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0 and not allow_negative:
        raise ValueError(f"{name} must not be negative, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class PointCorrespondence:
    """The 2D outcome of one map element for one camera frame.

    This is the lower-level record behind the region-level summary: it says where
    one geometry element landed and what became of it. It exists for diagnostics
    and reprojection checks; downstream stages consume
    :class:`SpatialObservation`.

    Attributes:
        geometry: The map element.
        visibility: What became of it.
        camera_depth_m: Depth of the point under the frame's :class:`DepthMetric`, in meters.
        raw_pixel: Pixel in the raw image, when the model produced one.
        prepared_pixel: Pixel in the prepared image, when the point reached it.
        support_depth_m: Depth of the nearest surface supported around that pixel, in
            meters, under the same metric; the occluder's depth for an ``OCCLUDED`` point.
    """

    geometry: GeometryReference
    visibility: VisibilityState
    camera_depth_m: float
    raw_pixel: PixelCoordinate | None
    prepared_pixel: PixelCoordinate | None
    support_depth_m: float | None

    def __post_init__(self) -> None:
        """Validate the record against its visibility state.

        Raises:
            ValueError: If a value is not finite, a depth is negative (except for a
                point behind the camera), a pixel is missing for a point that reached
                the prepared image, a pixel or support is present for a point the model
                cannot project, or an occluded point is not hidden by a strictly nearer
                surface.
        """
        # Só um ponto atrás da câmera pode ter profundidade óptica negativa.
        _require_depth(
            "camera_depth_m",
            self.camera_depth_m,
            allow_negative=self.visibility is VisibilityState.BEHIND_CAMERA,
        )
        if self.support_depth_m is not None:
            _require_depth("support_depth_m", self.support_depth_m)
        _require_finite_pair("raw_pixel", self.raw_pixel)
        _require_finite_pair("prepared_pixel", self.prepared_pixel)
        state = self.visibility
        if state in _PIXEL_REQUIRED and self.prepared_pixel is None:
            raise ValueError(f"a point in state {state.value} must carry its prepared_pixel")
        if state is VisibilityState.BEHIND_CAMERA and (
            self.raw_pixel is not None
            or self.prepared_pixel is not None
            or self.support_depth_m is not None
        ):
            raise ValueError(
                "a behind_camera point has no raw_pixel, prepared_pixel or support_depth_m"
            )
        if state is VisibilityState.OUTSIDE_IMAGE and self.support_depth_m is not None:
            raise ValueError("an outside_image point has no support_depth_m")
        if state is VisibilityState.OCCLUDED and (
            self.support_depth_m is None or self.support_depth_m >= self.camera_depth_m
        ):
            raise ValueError(
                "an occluded point needs a support_depth_m nearer than its camera_depth_m, "
                f"got {self.support_depth_m!r} and {self.camera_depth_m!r}"
            )


@dataclass(frozen=True, kw_only=True)
class VisibilityDiagnostics:
    """How many candidate elements ended in each visibility state.

    Only positive counts are kept, in :class:`VisibilityState` declaration order,
    so two equal diagnostics always encode to the same record.

    Attributes:
        counts: Number of elements per state.
    """

    counts: Mapping[VisibilityState, int]

    def __post_init__(self) -> None:
        """Validate and canonicalize the counts.

        Raises:
            ValueError: If a count is negative.
        """
        for state, count in self.counts.items():
            if count < 0:
                raise ValueError(f"visibility count for {state.value} must not be negative")
        canonical = {
            state: self.counts[state] for state in VisibilityState if self.counts.get(state, 0) > 0
        }
        object.__setattr__(self, "counts", canonical)

    @property
    def total(self) -> int:
        """Number of elements accounted for across every state."""
        return sum(self.counts.values())

    def count(self, state: VisibilityState) -> int:
        """Return the number of elements in ``state``."""
        return self.counts.get(state, 0)


@dataclass(frozen=True, kw_only=True)
class ProjectionSummary:
    """Frame-level facts about the projection a region's evidence came from.

    Attributes:
        camera_model_kind: Camera model used, e.g. ``"pinhole"`` or ``"mei"``.
        depth_metric: How depth is measured in this frame's diagnostics.
        image_transform_id: Identity of the raw-to-prepared image transform the
            pixels are expressed after.
        prepared_image_size: ``(width, height)`` of the prepared image, in pixels.
        considered_count: Geometry elements evaluated for this frame.
        visible_count: Of those, elements that were in front of the camera,
            inside the valid prepared image and not occluded.
    """

    camera_model_kind: str
    depth_metric: DepthMetric
    image_transform_id: str
    prepared_image_size: tuple[int, int]
    considered_count: int
    visible_count: int

    def __post_init__(self) -> None:
        """Validate the summary.

        Raises:
            ValueError: If an identity is empty, the image size is not positive,
                a count is negative, or ``visible_count`` exceeds
                ``considered_count``.
        """
        if not self.camera_model_kind:
            raise ValueError("camera_model_kind must not be empty")
        if not self.image_transform_id:
            raise ValueError("image_transform_id must not be empty")
        if any(side < 1 for side in self.prepared_image_size):
            raise ValueError(
                f"prepared_image_size must be positive, got {self.prepared_image_size!r}"
            )
        if self.considered_count < 0 or self.visible_count < 0:
            raise ValueError("projection counts must not be negative")
        if self.visible_count > self.considered_count:
            raise ValueError(
                f"visible_count {self.visible_count} must not exceed considered_count "
                f"{self.considered_count}"
            )


@dataclass(frozen=True, kw_only=True)
class CalibrationRef:
    """Which calibration and camera turned geometry into pixels.

    Attributes:
        calibration_identity: Hash of the calibration set used.
        camera_calibration_id: The camera entry within that set.
        camera_model_kind: Camera model of that entry.
        camera_frame: The camera optical frame the projection was done in.
    """

    calibration_identity: str
    camera_calibration_id: CalibrationReferenceId
    camera_model_kind: str
    camera_frame: FrameId

    def __post_init__(self) -> None:
        """Require every identity.

        Raises:
            ValueError: If an identity is empty.
        """
        for name in (
            "calibration_identity",
            "camera_calibration_id",
            "camera_model_kind",
            "camera_frame",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class PoseRef:
    """Which pose placed the camera in the map frame for this observation.

    Mirrors the evidence of a state-estimation lookup without embedding the pose.

    Attributes:
        trajectory_id: The trajectory the pose was read from.
        state_estimation_run_id: The persisted run it came from, when there is one.
        source_estimate_ids: The estimated pose, or the two neighbors of an
            interpolation.
        lookup_outcome: How the pose was resolved.
        time_delta_ns: Distance in nanoseconds from the frame timestamp to the
            nearest contributing pose.
        interpolation_fraction: Position in ``[0, 1]`` between the two neighbors
            for an interpolated pose; ``None`` otherwise.
    """

    trajectory_id: TrajectoryId
    state_estimation_run_id: StateEstimationRunId | None
    source_estimate_ids: tuple[PoseEstimateId, ...]
    lookup_outcome: LookupOutcome
    time_delta_ns: int
    interpolation_fraction: float | None

    def __post_init__(self) -> None:
        """Validate that the reference is coherent with how the pose was resolved.

        Raises:
            ValueError: If the trajectory or estimates are missing, the time delta
                is negative, or the estimates and fraction do not match the
                lookup outcome.
        """
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if not self.source_estimate_ids:
            raise ValueError("source_estimate_ids must name at least one pose estimate")
        if self.time_delta_ns < 0:
            raise ValueError(f"time_delta_ns must not be negative, got {self.time_delta_ns}")
        if self.lookup_outcome is LookupOutcome.INTERPOLATED:
            if len(self.source_estimate_ids) != 2:
                raise ValueError("an interpolated pose reference needs exactly two estimates")
            fraction = self.interpolation_fraction
            if fraction is None or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"an interpolated pose reference needs an interpolation_fraction in "
                    f"[0, 1], got {fraction!r}"
                )
        else:
            if len(self.source_estimate_ids) != 1:
                raise ValueError("a direct pose reference names exactly one source estimate")
            if self.interpolation_fraction is not None:
                raise ValueError("a direct pose reference has no interpolation_fraction")


@dataclass(frozen=True, kw_only=True)
class AssociationProvenance:
    """Run-level traceability of a spatial observation.

    Attributes:
        geometric_map_id: The immutable map the geometry references belong to.
        association_run_id: The execution producing this observation.
        region_artifact_id: The regional producer owning the region.
        sequence_artifact_id: The canonical sequence both were built from.
        visibility_policy_id: Versioned rule that decides visibility and occlusion.
        membership_policy_id: Versioned rule applied to this observation's region (mask or
            box rule), not the run's configured policy.
        configuration_fingerprint: Hash of the association configuration, when
            there is something configurable.
        code_version: Code revision that produced the observation, when known.
    """

    geometric_map_id: MapId
    association_run_id: str
    region_artifact_id: str
    sequence_artifact_id: SequenceArtifactId
    visibility_policy_id: str
    membership_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require every identity and policy.

        Raises:
            ValueError: If an identity or policy is empty.
        """
        for name in (
            "geometric_map_id",
            "association_run_id",
            "region_artifact_id",
            "sequence_artifact_id",
            "visibility_policy_id",
            "membership_policy_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, kw_only=True)
class SpatialObservation:
    """The map geometry one region of one image sees.

    A spatial observation is evidence produced in one execution. Regions may
    overlap, so the same geometry can support several observations; nothing here
    decides which region owns it. It carries references to the geometry, the
    selected region, never copies of visual features or semantic claims.

    Attributes:
        spatial_observation_id: Identity of this observation.
        source_observation_id: The camera frame the region was observed in.
        region_artifact_id: The regional artifact that owns the region.
        region_id: The observed region.
        geometry_support: References to the visible geometry inside the region,
            sorted by ``geometry_id`` and unique, all from one map.
        projection_summary: Frame-level projection facts.
        calibration_ref: The calibration and camera used.
        pose_ref: The pose used.
        visibility: Why candidate elements were or were not taken.
        provenance: Run-level traceability.
    """

    spatial_observation_id: SpatialObservationId
    source_observation_id: SourceObservationId
    region_artifact_id: str
    region_id: RegionId
    geometry_support: tuple[GeometryReference, ...]
    projection_summary: ProjectionSummary
    calibration_ref: CalibrationRef
    pose_ref: PoseRef
    visibility: VisibilityDiagnostics
    provenance: AssociationProvenance

    def __post_init__(self) -> None:
        """Validate identities, support and cross-field consistency.

        Raises:
            ValueError: If an identity is empty; the support spans more than one
                map, is not the provenance map, or is not sorted and unique; the
                associated count differs from the support size; the diagnostics
                exceed what the projection considered or kept visible; or a
                region-scoped feature belongs to another region.
        """
        for name in (
            "spatial_observation_id",
            "source_observation_id",
            "region_artifact_id",
            "region_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        maps = {reference.map_id for reference in self.geometry_support}
        if maps - {self.provenance.geometric_map_id}:
            raise ValueError(
                f"geometry_support must reference only the provenance map "
                f"{self.provenance.geometric_map_id!r}, got maps {sorted(maps)!r}"
            )
        identities = [reference.geometry_id for reference in self.geometry_support]
        if any(left >= right for left, right in pairwise(identities)):
            raise ValueError("geometry_support must be sorted by geometry_id and unique")
        associated = self.visibility.count(VisibilityState.ASSOCIATED)
        if associated != len(self.geometry_support):
            raise ValueError(
                f"the associated count {associated} must equal the geometry_support size "
                f"{len(self.geometry_support)}"
            )
        summary = self.projection_summary
        if self.visibility.total > summary.considered_count:
            raise ValueError(
                f"visibility counts {self.visibility.total} exceed the "
                f"considered_count {summary.considered_count}"
            )
        if len(self.geometry_support) > summary.visible_count:
            raise ValueError(
                f"geometry_support size {len(self.geometry_support)} exceeds the "
                f"visible_count {summary.visible_count}"
            )
        if self.region_artifact_id != self.provenance.region_artifact_id:
            raise ValueError("region_artifact_id must match association provenance")
