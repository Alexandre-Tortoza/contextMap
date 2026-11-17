"""Correção de movimento por interpolação de pose durante um scan.

A política declara referência temporal e unidades; a evidência preserva inputs e identidade.
Os cálculos carregam NumPy sob demanda, sem antecipar a dependência na importação.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmap.geometric_mapping.motion_correction import (
    MotionCorrectionEvidence,
    MotionCorrectionRecord,
    MotionCorrectionState,
)
from contextmap.ingestion import LidarObservation, PointFieldDataType
from contextmap.shared import (
    Quaternion,
    SourceTimestamp,
    Vector3,
    compose_rigid,
    invert_rigid,
    quaternion_to_rotation_matrix,
)
from contextmap.state_estimation import (
    LookupMode,
    LookupPolicy,
    PoseEstimate,
    RejectedLookup,
    ResolvedPose,
    ResolvedTransform,
    TrajectoryLookup,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from contextmap.geometric_mapping.inputs import PointCloudLayout

POSE_INTERPOLATION_DESKEW_V1 = "pose-interpolation-deskew-v1"
"""Identity of the policy; also the producer named in the correction evidence."""

_NANOSECONDS_PER_SECOND = 1_000_000_000
_FLOAT_CODES = {PointFieldDataType.FLOAT32: "<f4", PointFieldDataType.FLOAT64: "<f8"}


class DeskewError(ValueError):
    """Raised when a scan cannot be deskewed without guessing."""


class PointTimeReference(Enum):
    """Instant the per-point times are offsets from; the corrected points are expressed at it.

    Attributes:
        SCAN_TIMESTAMP: The scan's own timestamp. When the source stamps a scan at the start
            of its acquisition, the offsets run from zero to the scan's duration.
    """

    SCAN_TIMESTAMP = "scan_timestamp"


class PointTimeUnit(Enum):
    """Unit of the per-point time field, which a point record does not declare.

    Attributes:
        SECONDS: Seconds.
    """

    SECONDS = "seconds"


_NANOSECONDS_PER_UNIT = {PointTimeUnit.SECONDS: float(_NANOSECONDS_PER_SECOND)}


@dataclass(frozen=True, kw_only=True)
class PoseInterpolationDeskewPolicy:
    """The explicit rule of ``pose-interpolation-deskew-v1``.

    No attribute has a default: the name and unit of the time field belong to a sensor's
    driver and are never inherited from one dataset.

    Attributes:
        time_field: Name of the per-point time field in the point record.
        time_unit: Unit of that field.
        reference: What the per-point times are offsets from.
        time_bin_ns: Width of a time bin, in nanoseconds. Points are grouped by the nearest
            multiple of their instant (ties to even) and one pose lookup serves a whole bin,
            at that multiple clamped to the earliest and latest instant of the bin's points.
    """

    time_field: str
    time_unit: PointTimeUnit
    reference: PointTimeReference
    time_bin_ns: int

    def __post_init__(self) -> None:
        """Validate the field name and the bin width.

        Raises:
            ValueError: If ``time_field`` is empty or ``time_bin_ns`` is not positive.
        """
        if not self.time_field:
            raise ValueError("time_field must name the per-point time field")
        if self.time_bin_ns <= 0:
            raise ValueError(f"time_bin_ns must be positive, got {self.time_bin_ns}")

    @property
    def policy_id(self) -> str:
        """Identity of the rule, versioned with its semantics."""
        return POSE_INTERPOLATION_DESKEW_V1


def encode_deskew_policy(policy: PoseInterpolationDeskewPolicy) -> dict[str, Any]:
    """Encode a deskew policy as JSON primitives.

    Args:
        policy: The policy.

    Returns:
        Its identity and every parameter that changes the corrected coordinates.
    """
    return {
        "policy_id": policy.policy_id,
        "time_field": policy.time_field,
        "time_unit": policy.time_unit.value,
        "reference": policy.reference.value,
        "time_bin_ns": policy.time_bin_ns,
    }


def deskew_configuration_fingerprint(
    policy: PoseInterpolationDeskewPolicy, pose_lookup: LookupPolicy
) -> str:
    """Hash the configuration of a correction: the deskew policy and the pose lookup it used.

    Args:
        policy: The deskew policy.
        pose_lookup: The policy every pose of the correction was resolved under.

    Returns:
        ``"sha256:<hex>"``, equal for equal configurations and different when any
        parameter of either policy changes.
    """
    # Import tardio: serialization depende de transformation, que depende de inputs, que usa
    # este módulo; no momento da chamada todos já estão carregados.
    from contextmap.geometric_mapping.serialization import encode_lookup_policy

    payload = json.dumps(
        {"deskew": encode_deskew_policy(policy), "pose_lookup": encode_lookup_policy(pose_lookup)},
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def require_interpolated_lookup(pose_lookup: LookupPolicy) -> None:
    """Refuse a pose lookup that cannot give a pose at every point instant.

    Args:
        pose_lookup: The policy the deskew would resolve its poses under.

    Raises:
        DeskewError: If the policy is not ``INTERPOLATED``: an exact lookup rejects almost
            every instant and a nearest one would move the points in steps.
    """
    if pose_lookup.mode is not LookupMode.INTERPOLATED:
        raise DeskewError(
            f"{POSE_INTERPOLATION_DESKEW_V1} needs an interpolated pose lookup, got "
            f"{pose_lookup.mode.value!r}"
        )


@dataclass(frozen=True, kw_only=True)
class DeskewedScan:
    """A scan re-expressed at its timestamp, with the record that declares it corrected.

    Attributes:
        observation: The same physical observation (identity, frame, timestamp, layout and
            every other field), with its coordinates corrected.
        record: ``CORRECTED`` record whose evidence names the policy, the trajectory, the
            hash of ``observation.data`` and the configuration fingerprint.
    """

    observation: LidarObservation
    record: MotionCorrectionRecord


def deskew_scan(
    observation: LidarObservation,
    *,
    layout: PointCloudLayout,
    reference_pose: ResolvedPose,
    static_transform: ResolvedTransform | None,
    lookup: TrajectoryLookup,
    policy: PoseInterpolationDeskewPolicy,
) -> DeskewedScan:
    """Re-express every point of a scan in the sensor frame at the scan's timestamp.

    Args:
        observation: The scan, with a per-point time field.
        layout: Where the coordinates are in each point record.
        reference_pose: ``T_map_body`` resolved at the scan's timestamp; its policy is the
            one every other pose of the correction is resolved under.
        static_transform: ``T_body_source``; ``None`` when the scan is in the body frame.
        lookup: The trajectory the reference pose was resolved from.
        policy: The deskew rule.

    Returns:
        The corrected scan and its ``CORRECTED`` record.

    Raises:
        DeskewError: If the pose lookup is not interpolated, the reference pose or the static
            transform does not belong to the scan, the time field is missing or not one
            floating-point element, a point with finite coordinates has no finite time, or a
            point instant has no acceptable pose.
    """
    import numpy as np

    require_interpolated_lookup(reference_pose.policy)
    _require_reference(observation, reference_pose, static_transform)
    count = observation.point_count
    buffer = bytearray(observation.data)
    coordinates = np.frombuffer(buffer, dtype=_coordinate_record(layout), count=count)
    columns = [coordinates[name].astype(np.float64) for name in ("x", "y", "z")]
    finite = np.isfinite(columns[0]) & np.isfinite(columns[1]) & np.isfinite(columns[2])
    offsets_ns = _point_offsets_ns(observation, policy)
    unknown = finite & ~np.isfinite(offsets_ns)
    if unknown.any():
        raise DeskewError(
            f"point {int(np.flatnonzero(unknown)[0])} of scan {observation.observation_id!r} has "
            f"finite coordinates but its {policy.time_field!r} is not finite, so its instant is "
            "unknown"
        )

    positions = np.flatnonzero(finite)
    offsets = offsets_ns[positions]
    used, of_point = np.unique(
        np.rint(offsets / policy.time_bin_ns).astype(np.int64), return_inverse=True
    )
    # O instante de um bin é o seu múltiplo, trazido para dentro dos instantes dos seus pontos:
    # arredondar o último ponto para cima consultaria uma pose depois do fim do scan, onde
    # nenhum ponto foi medido (medido em dados reais: dentro de um gap da trajetória).
    earliest = np.full(len(used), np.inf)
    latest = np.full(len(used), -np.inf)
    np.minimum.at(earliest, of_point, offsets)
    np.maximum.at(latest, of_point, offsets)
    instants = np.rint(np.clip(used * float(policy.time_bin_ns), earliest, latest))
    rotations = np.empty((len(used), 3, 3), dtype=np.float64)
    translations = np.empty((len(used), 3), dtype=np.float64)
    moving = np.empty(len(used), dtype=bool)
    for slot, offset_ns in enumerate(instants.astype(np.int64).tolist()):
        pose = reference_pose.pose
        if offset_ns != 0:
            pose = _pose_at(observation, reference_pose, lookup, offset_ns, policy)
        translation, rotation = _relative_motion(reference_pose.pose, pose, static_transform)
        rotations[slot] = quaternion_to_rotation_matrix(rotation)
        translations[slot] = translation
        moving[slot] = (pose.translation_m, pose.orientation) != (
            reference_pose.pose.translation_m,
            reference_pose.pose.orientation,
        )

    # Só os pontos cuja pose difere da de referência são reescritos: os demais guardam os bytes.
    moved = moving[of_point]
    _write_corrected(
        coordinates,
        columns,
        positions[moved],
        rotations[of_point[moved]],
        translations[of_point[moved]],
    )
    corrected = dataclasses.replace(observation, data=bytes(buffer))
    return DeskewedScan(
        observation=corrected,
        record=MotionCorrectionRecord(
            observation_id=observation.observation_id,
            state=MotionCorrectionState.CORRECTED,
            per_point_timing_available=True,
            evidence=MotionCorrectionEvidence(
                producer=policy.policy_id,
                trajectory_id=lookup.trajectory.trajectory_id,
                payload_hash=f"sha256:{hashlib.sha256(corrected.data).hexdigest()}",
                configuration_fingerprint=deskew_configuration_fingerprint(
                    policy, reference_pose.policy
                ),
            ),
        ),
    )


def _require_reference(
    observation: LidarObservation,
    reference_pose: ResolvedPose,
    static_transform: ResolvedTransform | None,
) -> None:
    """Require the reference pose at the scan's timestamp and a chain that ends in its frame."""
    if reference_pose.query_timestamp != observation.timestamp:
        raise DeskewError(
            f"the reference pose of scan {observation.observation_id!r} was resolved at "
            f"{reference_pose.query_timestamp} but the scan's timestamp is {observation.timestamp}"
        )
    body = reference_pose.pose.child_frame
    parent, child = (
        (body, body)
        if static_transform is None
        else (static_transform.parent_frame, static_transform.child_frame)
    )
    if parent != body or child != observation.frame_id:
        raise DeskewError(
            f"scan {observation.observation_id!r} is in frame {observation.frame_id!r} but its "
            f"chain goes from the body {body!r} to {child!r}"
        )


def _coordinate_record(layout: PointCloudLayout) -> np.dtype[Any]:
    import numpy as np

    code = _FLOAT_CODES[layout.scalar]
    return np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [code, code, code],
            "offsets": [layout.x_offset_bytes, layout.y_offset_bytes, layout.z_offset_bytes],
            "itemsize": layout.point_step_bytes,
        }
    )


def _point_offsets_ns(
    observation: LidarObservation, policy: PoseInterpolationDeskewPolicy
) -> NDArray[np.float64]:
    """Each point's offset from the scan's timestamp, in nanoseconds, as the policy reads it."""
    import numpy as np

    field = next((item for item in observation.fields if item.name == policy.time_field), None)
    if field is None:
        raise DeskewError(
            f"scan {observation.observation_id!r} has no {policy.time_field!r} field, so the "
            "instants of its points are unknown"
        )
    if field.data_type not in _FLOAT_CODES or field.count != 1:
        raise DeskewError(
            f"the {policy.time_field!r} field of scan {observation.observation_id!r} must be one "
            f"float32 or float64 element, got {field.count} x {field.data_type.value}"
        )
    record = np.dtype(
        {
            "names": ["time"],
            "formats": [_FLOAT_CODES[field.data_type]],
            "offsets": [field.offset_bytes],
            "itemsize": observation.point_step_bytes,
        }
    )
    times = np.frombuffer(observation.data, dtype=record, count=observation.point_count)["time"]
    offsets: NDArray[np.float64] = (
        times.astype(np.float64) * _NANOSECONDS_PER_UNIT[policy.time_unit]
    )
    return offsets


def _pose_at(
    observation: LidarObservation,
    reference_pose: ResolvedPose,
    lookup: TrajectoryLookup,
    offset_ns: int,
    policy: PoseInterpolationDeskewPolicy,
) -> PoseEstimate:
    """The body pose ``offset_ns`` after the scan's timestamp, under the reference's policy."""
    stamp = observation.timestamp
    seconds, nanoseconds = divmod(stamp.total_nanoseconds() + offset_ns, _NANOSECONDS_PER_SECOND)
    result = lookup.pose_at(
        SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=stamp.clock_id),
        policy=reference_pose.policy,
    )
    if isinstance(result, RejectedLookup):
        raise DeskewError(
            f"points of scan {observation.observation_id!r} measured {offset_ns / 1e6:.3f} ms "
            f"after its timestamp (time bin {policy.time_bin_ns} ns) have no acceptable pose: "
            f"{result.rejection.value}: {result.detail}"
        )
    return result.pose


def _relative_motion(
    reference: PoseEstimate, pose: PoseEstimate, static_transform: ResolvedTransform | None
) -> tuple[Vector3, Quaternion]:
    """``T_body_source^-1 · T_map_body(t_ref)^-1 · T_map_body(t) · T_body_source``."""
    to_reference = invert_rigid(translation=reference.translation_m, rotation=reference.orientation)
    translation, rotation = compose_rigid(
        outer_translation=to_reference[0],
        outer_rotation=to_reference[1],
        inner_translation=pose.translation_m,
        inner_rotation=pose.orientation,
    )
    if static_transform is None:
        return translation, rotation
    to_body = invert_rigid(
        translation=static_transform.translation, rotation=static_transform.rotation
    )
    translation, rotation = compose_rigid(
        outer_translation=to_body[0],
        outer_rotation=to_body[1],
        inner_translation=translation,
        inner_rotation=rotation,
    )
    return compose_rigid(
        outer_translation=translation,
        outer_rotation=rotation,
        inner_translation=static_transform.translation,
        inner_rotation=static_transform.rotation,
    )


def _write_corrected(
    coordinates: NDArray[Any],
    columns: list[NDArray[np.float64]],
    positions: NDArray[np.intp],
    rotations: NDArray[np.float64],
    translations: NDArray[np.float64],
) -> None:
    """Overwrite the coordinates at ``positions`` with the relative motion of their bin.

    The arithmetic is float64 with a fixed evaluation order (no BLAS), as in the mapping's
    transformation, and the result is stored in the payload's own scalar type.
    """
    x, y, z = (column[positions] for column in columns)
    for axis, name in enumerate(("x", "y", "z")):
        coordinates[name][positions] = (
            rotations[:, axis, 0] * x
            + rotations[:, axis, 1] * y
            + rotations[:, axis, 2] * z
            + translations[:, axis]
        )
