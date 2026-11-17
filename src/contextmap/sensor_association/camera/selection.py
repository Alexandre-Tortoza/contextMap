"""Escolha do modelo de câmera a partir da calibração canônica.

O modelo vem **apenas** da entrada de calibração de Ingestion: nenhum modelo é substituído por
outro, nenhum parâmetro é inferido do dataset e nenhum arquivo de origem é lido aqui. Uma
calibração inválida é recusada antes de virar uma projeção, porque projetar com ela produziria
pixels plausíveis e errados.
"""

from __future__ import annotations

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationError,
    CalibrationSet,
    FisheyeCameraModel,
    PinholeCameraModel,
    validate_calibration_set,
)
from contextmap.sensor_association.camera.fisheye import FisheyeProjection
from contextmap.sensor_association.camera.mei import MeiProjection
from contextmap.sensor_association.camera.pinhole import PinholeProjection
from contextmap.sensor_association.camera.projection import CameraProjection


def camera_projection_for(entry: CalibrationEntry) -> CameraProjection:
    """Build the projection that the canonical calibration declares.

    Args:
        entry: A calibration entry of a camera.

    Returns:
        A pinhole, fisheye or MEI projection according to the entry's camera model.

    Raises:
        ValueError: If the entry has no camera model.
        CalibrationError: If the calibration is invalid (non-finite or non-positive
            parameters, wrong coefficient counts, negative ``xi``, ...).
    """
    model = entry.camera_model
    if model is None:
        raise ValueError(
            f"calibration {entry.calibration_id!r} has no camera model to project with"
        )
    problems = validate_calibration_set(
        CalibrationSet(entries={entry.calibration_id: entry}, static_transforms=())
    )
    if problems:
        raise CalibrationError(f"cannot project with an invalid calibration: {problems}")
    if isinstance(model, PinholeCameraModel):
        return PinholeProjection(entry, model)
    if isinstance(model, FisheyeCameraModel):
        return FisheyeProjection(entry, model)
    return MeiProjection(entry, model)
