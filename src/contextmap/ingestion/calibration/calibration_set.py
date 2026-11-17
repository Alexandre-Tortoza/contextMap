"""Contratos canônicos de calibração, câmera e transforms estáticos.

O módulo define modelos pinhole, fisheye e MEI sem conversão implícita com perda, além de
CalibrationEntry, RigidTransform, CalibrationSet e metadata de refinamento. Relações dinâmicas no
tempo não são calibração e devem permanecer no fluxo de observações/estado.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from contextmap.ingestion.canonical_observations import CalibrationReferenceId, FrameId, SensorId

SCHEMA_VERSION = "0.1.0"
"""Calibration contract schema version."""

REFINEMENT_DOCUMENT_SCHEMA = "calibration-refinement/1"
"""Schema name for a calibration refinement document."""

CAMERA_OPTICAL_ROTATION_CONVENTION = (
    "p_camera = Rz(roll) @ Rx(pitch) @ Ry(yaw) @ p_camera_base; "
    "optical axes x right, y down, z forward"
)
"""Rotation convention used by camera calibration refinements."""

MAX_ROTATION_CORRECTION_DEG = 10.0
"""Largest accepted absolute correction angle in degrees."""

_QUATERNION_NORM_TOLERANCE = 1e-3


class DistortionModel(Enum):
    """Lens distortion model for :class:`PinholeCameraModel`.

    Coefficient order follows the OpenCV convention for each model.
    """

    NONE = "none"
    PLUMB_BOB = "plumb_bob"
    RATIONAL_POLYNOMIAL = "rational_polynomial"


_PLUMB_BOB_COEFFICIENT_COUNTS: Mapping[DistortionModel, int] = {
    DistortionModel.NONE: 0,
    DistortionModel.PLUMB_BOB: 5,
    DistortionModel.RATIONAL_POLYNOMIAL: 8,
}


class CalibrationError(Exception):
    """Raised when a calibration set fails validation."""


@dataclass(frozen=True, kw_only=True)
class PinholeCameraModel:
    """Standard pinhole camera intrinsics, with optional radial/tangential distortion.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.
        distortion_model: Distortion model used by ``distortion_coefficients``.
        distortion_coefficients: Coefficients in the order OpenCV documents
            for ``distortion_model`` (empty for
            :attr:`DistortionModel.NONE`).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: DistortionModel = DistortionModel.NONE
    distortion_coefficients: tuple[float, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FisheyeCameraModel:
    """Fisheye camera intrinsics using the equidistant (Kannala-Brandt) model.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.
        distortion_coefficients: Equidistant model coefficients
            ``(k1, k2, k3, k4)``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coefficients: tuple[float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class MeiCameraModel:
    """Unified omnidirectional camera intrinsics (Geyer-Daniilidis / Mei).

    The model first projects a point through a unit sphere whose center is
    shifted by ``xi`` along the optical axis, then applies radial and
    tangential distortion on the normalized plane, then the generalized
    projection matrix. This is the parameterization used by CamOdoCal and by
    calibrations declared with ``model_type: MEI``.

    Attributes:
        width: Image width in pixels this calibration applies to.
        height: Image height in pixels this calibration applies to.
        fx: Generalized focal length along x (``gamma1``), in pixels.
        fy: Generalized focal length along y (``gamma2``), in pixels.
        cx: Principal point x (``u0``), in pixels.
        cy: Principal point y (``v0``), in pixels.
        xi: Mirror parameter, dimensionless and not negative. ``0`` degenerates
            to a distorted pinhole; values above ``1`` describe a field of view
            wider than a hemisphere.
        distortion_coefficients: ``(k1, k2, p1, p2)``: two radial and two
            tangential coefficients, applied on the normalized plane.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    distortion_coefficients: tuple[float, float, float, float]


CameraModel = PinholeCameraModel | FisheyeCameraModel | MeiCameraModel
"""Any camera model kind; no lossy conversion exists between them."""


def camera_model_kind(model: CameraModel) -> str:
    """Return ``"pinhole"``, ``"fisheye"`` or ``"mei"`` for a camera model.

    Args:
        model: A pinhole, fisheye or unified omnidirectional camera model.

    Returns:
        ``"pinhole"``, ``"fisheye"`` or ``"mei"``.

    Raises:
        TypeError: If ``model`` is not a recognized camera model type.
    """
    if isinstance(model, PinholeCameraModel):
        return "pinhole"
    if isinstance(model, FisheyeCameraModel):
        return "fisheye"
    if isinstance(model, MeiCameraModel):
        return "mei"
    raise TypeError(f"unsupported camera model type: {type(model)!r}")


@dataclass(frozen=True, kw_only=True)
class RigidTransform:
    """A static rigid-body transform between two coordinate frames.

    Follows the ``T_parent_child`` convention: this transform takes
    coordinates expressed in ``child_frame`` into ``parent_frame``, i.e.
    ``p_parent = rotate(rotation, p_child) + translation``. ``rotation`` is
    a unit quaternion ordered ``(x, y, z, w)``.

    A ``RigidTransform`` is only valid for a relationship that does not
    change during the sequence (e.g. a camera rigidly mounted on a robot
    body). A frame whose pose changes over time (e.g. the robot body in the
    world) is not a calibration transform; it is reported per-timestamp as
    an :class:`~contextmap.ingestion.canonical_observations.ExternalPoseMeasurement`.

    Attributes:
        parent_frame: Frame this transform expresses coordinates in.
        child_frame: Frame whose coordinates are being transformed.
        translation: ``(x, y, z)`` in meters.
        rotation: Unit quaternion ``(x, y, z, w)`` rotating ``child_frame``'s
            axes into ``parent_frame``'s axes.
    """

    parent_frame: FrameId
    child_frame: FrameId
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class CalibrationProvenance:
    """Traceability metadata for one calibration entry.

    Attributes:
        source_type: Origin of the calibration, e.g. ``"ros1_bag"``,
            ``"ros2_bag"``, ``"dataset"``, ``"manual"``.
        source_path: Path or identity of the calibration source.
        original_values: Source-native calibration values preserved as
            primitives before normalization (e.g. the raw distortion array
            or frame name as given by the source). Empty when the source
            already matched the canonical representation exactly.
        conversions_applied: Human-readable notes of any normalization
            performed, e.g. ``"reordered quaternion from wxyz to xyzw"``.
            An empty sequence asserts no conversion was necessary.
    """

    source_type: str
    source_path: str
    original_values: Mapping[str, object] = field(default_factory=dict)
    conversions_applied: Sequence[str] = ()


@dataclass(frozen=True, kw_only=True)
class CalibrationEntry:
    """One sensor's canonical, inspectable calibration.

    Attributes:
        calibration_id: Identity referenced by
            ``SourceObservation.calibration_id``.
        sensor_id: Sensor this calibration applies to.
        frame_id: Coordinate frame ``camera_model`` (when present) is
            expressed in, e.g. the camera's optical frame.
        camera_model: Intrinsics/distortion, when ``sensor_id`` is a
            camera. ``None`` for non-camera sensors (e.g. LiDAR, which has
            no intrinsic camera model but may still have a
            ``CalibrationEntry`` for its frame identity/provenance).
        provenance: Traceability back to the calibration source.
        content_hash: ``"sha256:<hex digest>"`` of this entry's canonical
            values, for change detection.
    """

    calibration_id: CalibrationReferenceId
    sensor_id: SensorId
    frame_id: FrameId
    camera_model: CameraModel | None
    provenance: CalibrationProvenance
    content_hash: str


@dataclass(frozen=True, kw_only=True)
class RefinementValidation:
    """The evidence that a refinement improves the calibration.

    It documents the refinement and never enters any calibration identity, the same way
    :class:`CalibrationProvenance` stays out of :attr:`CalibrationEntry.content_hash`.

    Attributes:
        fit_ids: Ids of the observations or groups the correction was estimated on.
        held_out_ids: Ids kept out of the estimation and used only to validate it.
        metrics: Named measurements; each name carries its own semantics and unit (e.g.
            ``held_out_photometric_inconsistency_base``), and no two are assumed comparable.
    """

    fit_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class CalibrationRefinement:
    """A versioned rotation correction of one camera extrinsic, with how it was estimated.

    The refinement targets the static transform ``T_parent_child`` whose child is a calibrated
    camera optical frame. The correction rotates the base optical coordinates,
    ``p_camera = Rcorr @ p_camera_base``, with ``Rcorr = Rz(roll) @ Rx(pitch) @ Ry(yaw)``
    (:data:`CAMERA_OPTICAL_ROTATION_CONVENTION`); the optical centre does not move. It is
    applied once, by :func:`apply_calibration_refinement`, to the transform whose digest is
    ``base_transform_digest``.

    Attributes:
        refinement_id: Stable, unique name of this refinement.
        parent_frame: Parent frame of the target transform (the body).
        child_frame: Child frame of the target transform (the camera optical frame).
        base_transform_digest: :func:`rigid_transform_digest` of the transform the
            correction was estimated on.
        yaw_deg: Rotation about the optical ``y`` axis, in degrees; positive moves the image
            of a point to the right (``+u``).
        pitch_deg: Rotation about the optical ``x`` axis, in degrees; positive moves the image
            of a point up (``-v``).
        roll_deg: Rotation about the optical ``z`` axis, in degrees.
        convention: Exactly :data:`CAMERA_OPTICAL_ROTATION_CONVENTION`.
        method: What the estimation measured, e.g. multi-frame photometric consistency.
        estimator: Script or module that estimated the correction.
        code_identity: Identity of the estimator code that produced it.
        pose_source_kind: The kind of trajectory the estimation used (e.g. ground truth).
        pose_source_digest: Digest of that trajectory's source.
        source_files: Name to ``"sha256:<hex>"`` of each calibration source file the base
            transform came from.
        validation: Fit/held-out evidence; outside every identity.
    """

    refinement_id: str
    parent_frame: FrameId
    child_frame: FrameId
    base_transform_digest: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    convention: str
    method: str
    estimator: str
    code_identity: str
    pose_source_kind: str
    pose_source_digest: str
    source_files: Mapping[str, str]
    validation: RefinementValidation

    def identity_record(self) -> dict[str, object]:
        """Return what of this refinement a calibration identity covers.

        The target, the base it was estimated on and the correction change which calibration
        a run uses; method, estimator, pose source, source files and validation only document
        it, so they stay out.

        Returns:
            A JSON-serializable mapping.
        """
        return {
            "refinement_id": self.refinement_id,
            "parent_frame": str(self.parent_frame),
            "child_frame": str(self.child_frame),
            "base_transform_digest": self.base_transform_digest,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "convention": self.convention,
        }


@dataclass(frozen=True, kw_only=True)
class CalibrationSet:
    """The full, inspectable calibration and coordinate-frame inventory of a sequence.

    Attributes:
        entries: One :class:`CalibrationEntry` per calibrated sensor, keyed
            by its ``calibration_id``.
        static_transforms: Rigid transforms between sensor/body frames that
            do not change during the sequence. Dynamic transforms belong in
            the temporal observation stream, not here.
        refinements: The refinements already applied to ``static_transforms``, in the
            order they were applied. Empty for a calibration taken as the source gave it.
        schema_version: Calibration contract schema version.
    """

    entries: Mapping[CalibrationReferenceId, CalibrationEntry]
    static_transforms: Sequence[RigidTransform]
    refinements: tuple[CalibrationRefinement, ...] = ()
    schema_version: str = SCHEMA_VERSION
