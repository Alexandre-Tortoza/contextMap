"""Conversão de metadata ROS de câmera para calibração canônica de ingestion.

As funções extraem modelos de câmera a partir de CameraInfo, registram normalizações realizadas,
mesclam calibração descoberta com a fornecida por configuração e mantêm referências por sensor.
Projeção geométrica e calibração automática não pertencem a este módulo.
"""

from __future__ import annotations

from typing import Any

from contextmap.ingestion.calibration import (
    CalibrationEntry,
    CalibrationError,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    PinholeCameraModel,
)
from contextmap.ingestion.canonical_observations import CalibrationReferenceId, SensorId

PINHOLE_DISTORTION_MODEL_MAP: dict[str, DistortionModel] = {
    "plumb_bob": DistortionModel.PLUMB_BOB,
    "rational_polynomial": DistortionModel.RATIONAL_POLYNOMIAL,
}


def merge_calibration(
    provided: CalibrationSet | None,
    discovered_entries: tuple[CalibrationEntry, ...],
) -> CalibrationSet | None:
    """Merge configured calibration with source-discovered entries by sensor identity."""
    entries = dict(provided.entries) if provided is not None else {}
    by_sensor: dict[SensorId, CalibrationEntry] = {}
    for entry in entries.values():
        previous = by_sensor.get(entry.sensor_id)
        if previous is not None:
            raise CalibrationError(
                f"multiple calibration entries for sensor_id={entry.sensor_id!r}"
            )
        by_sensor[entry.sensor_id] = entry

    for entry in discovered_entries:
        previous = by_sensor.get(entry.sensor_id)
        if previous is not None:
            if previous.content_hash != entry.content_hash:
                raise CalibrationError(f"conflicting calibration for sensor_id={entry.sensor_id!r}")
            continue
        entries[entry.calibration_id] = entry
        by_sensor[entry.sensor_id] = entry

    if not entries and provided is None:
        return None
    # Os refinamentos viajam com os transforms que corrigiram: sem eles, o transform corrigido
    # chegaria à sequência sem dizer qual correção o produziu (nem entraria na identidade).
    return CalibrationSet(
        entries=entries,
        static_transforms=provided.static_transforms if provided is not None else (),
        refinements=provided.refinements if provided is not None else (),
    )


def calibration_ids_by_sensor(
    calibration: CalibrationSet | None,
) -> dict[SensorId, CalibrationReferenceId]:
    """Build an unambiguous sensor-to-calibration lookup."""
    if calibration is None:
        return {}
    result: dict[SensorId, CalibrationReferenceId] = {}
    for entry in calibration.entries.values():
        if entry.sensor_id in result:
            raise CalibrationError(
                f"multiple calibration entries for sensor_id={entry.sensor_id!r}"
            )
        result[entry.sensor_id] = entry.calibration_id
    return result


def build_camera_model(
    *,
    width: int,
    height: int,
    k_matrix: Any,
    distortion_model_name: str,
    distortion_coefficients: Any,
) -> tuple[CameraModel, tuple[str, ...]]:
    """Build a canonical camera model from primitive ``CameraInfo`` values.

    Callers extract ``K``/``D`` (ROS 1) or ``k``/``d`` (ROS 2) themselves,
    since that field-name casing is the one real difference between the
    two ROS versions' ``sensor_msgs/CameraInfo``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        k_matrix: The 9-element row-major intrinsic matrix.
        distortion_model_name: ``CameraInfo.distortion_model``.
        distortion_coefficients: The distortion coefficient array.

    Returns:
        The camera model and the notes of every normalization applied to the
        source values, for
        :attr:`~contextmap.ingestion.calibration.CalibrationProvenance.conversions_applied`
        (empty when the values mapped as given). The model is a
        :class:`~contextmap.ingestion.calibration.FisheyeCameraModel` when
        ``distortion_model_name == "equidistant"``, whose ``D`` is truncated
        or zero-filled to its four coefficients (with a note), otherwise a
        :class:`~contextmap.ingestion.calibration.PinholeCameraModel`
        (falling back to :attr:`DistortionModel.NONE` for an unrecognized
        model name, with a note naming it, and leaving validation to catch
        a resulting coefficient-count inconsistency rather than raising
        here).
    """
    fx = float(k_matrix[0])
    fy = float(k_matrix[4])
    cx = float(k_matrix[2])
    cy = float(k_matrix[5])

    if distortion_model_name == "equidistant":
        source = [float(value) for value in distortion_coefficients]
        conversions: tuple[str, ...] = ()
        if len(source) > 4:
            conversions = (
                f"equidistant takes 4 distortion coefficients (k1, k2, k3, k4), got "
                f"{len(source)}: dropped {len(source) - 4} extra coefficient(s) {source[4:]}",
            )
        elif len(source) < 4:
            conversions = (
                f"equidistant takes 4 distortion coefficients (k1, k2, k3, k4), got "
                f"{len(source)}: zero-filled the missing {4 - len(source)}",
            )
        k1, k2, k3, k4 = (source + [0.0] * 4)[:4]
        return FisheyeCameraModel(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_coefficients=(k1, k2, k3, k4),
        ), conversions

    coefficients = tuple(float(value) for value in distortion_coefficients)
    distortion_model = PINHOLE_DISTORTION_MODEL_MAP.get(distortion_model_name)
    pinhole_conversions: tuple[str, ...] = ()
    if distortion_model is None:
        # Com D=[], o fallback seria indistinguível de uma câmera sem distorção: fica registrado.
        distortion_model = DistortionModel.NONE
        pinhole_conversions = (
            f"unrecognized distortion_model {distortion_model_name!r}: fell back to none "
            f"with the {len(coefficients)} coefficient(s) of D as given",
        )
    return PinholeCameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
    ), pinhole_conversions
