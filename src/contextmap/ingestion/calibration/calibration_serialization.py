"""Serialização estrita dos contratos canônicos de calibração.

Este módulo converte CalibrationSet e documentos de CalibrationRefinement para registros JSON e
de volta, recusando schemas, campos ou formas desconhecidas quando o contrato exige exatidão.
Persistência aqui preserva significado e provenance; não executa descoberta de calibração.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextmap.ingestion.calibration.calibration_set import (
    REFINEMENT_DOCUMENT_SCHEMA,
    CalibrationEntry,
    CalibrationError,
    CalibrationProvenance,
    CalibrationRefinement,
    CalibrationSet,
    CameraModel,
    DistortionModel,
    FisheyeCameraModel,
    MeiCameraModel,
    PinholeCameraModel,
    RefinementValidation,
    RigidTransform,
    camera_model_kind,
)
from contextmap.ingestion.calibration.calibration_set_validation import (
    validate_calibration_refinement,
)
from contextmap.ingestion.canonical_observations import CalibrationReferenceId, FrameId, SensorId


def _encode_camera_model(model: CameraModel | None) -> dict[str, Any] | None:
    """Encode one optional canonical camera model.

    Args:
        model: Camera model to persist, or None for non-camera calibration entries.

    Returns:
        JSON-compatible camera record, or None.
    """
    if model is None:
        return None
    record: dict[str, Any] = {
        "kind": camera_model_kind(model),
        "width": model.width,
        "height": model.height,
        "fx": model.fx,
        "fy": model.fy,
        "cx": model.cx,
        "cy": model.cy,
        "distortion_coefficients": list(model.distortion_coefficients),
    }
    if isinstance(model, PinholeCameraModel):
        record["distortion_model"] = model.distortion_model.value
    if isinstance(model, MeiCameraModel):
        record["xi"] = model.xi
    return record


def _decode_camera_model(record: dict[str, Any] | None) -> CameraModel | None:
    """Decode one optional camera-model record into its exact canonical variant.

    Args:
        record: Persisted camera-model record, or None.

    Returns:
        Pinhole, fisheye or MEI model matching the stored kind, or None.

    Raises:
        ValueError: If a persisted enum value is invalid.
        KeyError: If required fields are absent.
    """
    if record is None:
        return None
    if record["kind"] == "pinhole":
        return PinholeCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            distortion_model=DistortionModel(record["distortion_model"]),
            distortion_coefficients=tuple(record["distortion_coefficients"]),
        )
    if record["kind"] == "fisheye":
        x, y, z, w = record["distortion_coefficients"]
        return FisheyeCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            distortion_coefficients=(x, y, z, w),
        )
    if record["kind"] == "mei":
        k1, k2, p1, p2 = record["distortion_coefficients"]
        return MeiCameraModel(
            width=record["width"],
            height=record["height"],
            fx=record["fx"],
            fy=record["fy"],
            cx=record["cx"],
            cy=record["cy"],
            xi=record["xi"],
            distortion_coefficients=(k1, k2, p1, p2),
        )
    raise CalibrationError(f"unknown camera model kind: {record['kind']!r}")


def encode_calibration_set(calibration_set: CalibrationSet) -> dict[str, Any]:
    """Encode a calibration set into a JSON-serializable dict.

    Args:
        calibration_set: The calibration set to encode.

    Returns:
        A dict suitable for ``json.dumps``; ``refinements`` is present only when the set has
        one, so an unrefined calibration encodes exactly as before refinements existed.
    """
    record: dict[str, Any] = {
        "schema_version": calibration_set.schema_version,
        "entries": [
            {
                "calibration_id": str(entry.calibration_id),
                "sensor_id": str(entry.sensor_id),
                "frame_id": str(entry.frame_id),
                "camera_model": _encode_camera_model(entry.camera_model),
                "provenance": {
                    "source_type": entry.provenance.source_type,
                    "source_path": entry.provenance.source_path,
                    "original_values": dict(entry.provenance.original_values),
                    "conversions_applied": list(entry.provenance.conversions_applied),
                },
                "content_hash": entry.content_hash,
            }
            for entry in calibration_set.entries.values()
        ],
        "static_transforms": [
            _encode_transform(transform) for transform in calibration_set.static_transforms
        ],
    }
    if calibration_set.refinements:
        record["refinements"] = [
            _encode_refinement(refinement) for refinement in calibration_set.refinements
        ]
    return record


def decode_calibration_set(record: dict[str, Any]) -> CalibrationSet:
    """Decode a calibration set from a dict produced by :func:`encode_calibration_set`.

    Args:
        record: A dict as produced by :func:`encode_calibration_set`. A record without
            ``refinements`` (every calibration written before refinements existed) decodes
            to a set with no refinement.

    Returns:
        The decoded calibration set.

    Raises:
        CalibrationError: If a refinement record is missing a field or has an unknown one.
    """
    entries = {
        CalibrationReferenceId(item["calibration_id"]): CalibrationEntry(
            calibration_id=CalibrationReferenceId(item["calibration_id"]),
            sensor_id=SensorId(item["sensor_id"]),
            frame_id=FrameId(item["frame_id"]),
            camera_model=_decode_camera_model(item["camera_model"]),
            provenance=CalibrationProvenance(
                source_type=item["provenance"]["source_type"],
                source_path=item["provenance"]["source_path"],
                original_values=dict(item["provenance"]["original_values"]),
                conversions_applied=list(item["provenance"]["conversions_applied"]),
            ),
            content_hash=item["content_hash"],
        )
        for item in record["entries"]
    }
    static_transforms = tuple(
        RigidTransform(
            parent_frame=FrameId(item["parent_frame"]),
            child_frame=FrameId(item["child_frame"]),
            translation=tuple(item["translation"]),
            rotation=tuple(item["rotation"]),
        )
        for item in record["static_transforms"]
    )
    return CalibrationSet(
        entries=entries,
        static_transforms=static_transforms,
        refinements=tuple(_decode_refinement(item) for item in record.get("refinements", ())),
        schema_version=record["schema_version"],
    )


def encode_calibration_refinement_document(refinement: CalibrationRefinement) -> dict[str, Any]:
    """Encode a refinement as a standalone ``calibration-refinement/1`` document.

    Args:
        refinement: The refinement to encode.

    Returns:
        A dict suitable for ``json.dumps``.
    """
    return {"schema": REFINEMENT_DOCUMENT_SCHEMA, **_encode_refinement(refinement)}


def decode_calibration_refinement_document(record: Mapping[str, Any]) -> CalibrationRefinement:
    """Decode and check a ``calibration-refinement/1`` document.

    Args:
        record: The parsed JSON document.

    Returns:
        The refinement it declares.

    Raises:
        CalibrationError: If the schema is not :data:`REFINEMENT_DOCUMENT_SCHEMA`, a field is
            missing, unknown or of the wrong type, or the refinement is malformed (see
            :func:`apply_calibration_refinement`).
    """
    if record.get("schema") != REFINEMENT_DOCUMENT_SCHEMA:
        raise CalibrationError(
            f"expected a {REFINEMENT_DOCUMENT_SCHEMA!r} document, got schema "
            f"{record.get('schema')!r}"
        )
    refinement = _decode_refinement(
        {key: value for key, value in record.items() if key != "schema"}
    )
    problems = validate_calibration_refinement(refinement)
    if problems:
        raise CalibrationError(f"invalid calibration refinement: {problems}")
    return refinement


def load_calibration_refinement(path: str | Path) -> CalibrationRefinement:
    """Read a ``calibration-refinement/1`` JSON document from disk.

    Args:
        path: The document's path.

    Returns:
        The refinement it declares.

    Raises:
        CalibrationError: If the document is not valid (see
            :func:`decode_calibration_refinement_document`).
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not JSON.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise CalibrationError(f"{path}: a calibration refinement document is a JSON object")
    try:
        return decode_calibration_refinement_document(record)
    except CalibrationError as error:
        raise CalibrationError(f"{path}: {error}") from error


def _encode_transform(transform: RigidTransform) -> dict[str, Any]:
    """Encode one static rigid transform as primitive JSON-compatible values."""
    return {
        "parent_frame": str(transform.parent_frame),
        "child_frame": str(transform.child_frame),
        "translation": list(transform.translation),
        "rotation": list(transform.rotation),
    }


def _encode_refinement(refinement: CalibrationRefinement) -> dict[str, Any]:
    """Encode a calibration refinement together with its provenance and validation evidence."""
    return {
        "refinement_id": refinement.refinement_id,
        "target": {
            "parent_frame": str(refinement.parent_frame),
            "child_frame": str(refinement.child_frame),
            "base_transform_digest": refinement.base_transform_digest,
        },
        "rotation_correction": {
            "yaw_deg": refinement.yaw_deg,
            "pitch_deg": refinement.pitch_deg,
            "roll_deg": refinement.roll_deg,
            "convention": refinement.convention,
        },
        "method": refinement.method,
        "estimator": {"module": refinement.estimator, "code_identity": refinement.code_identity},
        "pose_source": {
            "kind": refinement.pose_source_kind,
            "digest": refinement.pose_source_digest,
        },
        "source_files": dict(refinement.source_files),
        "validation": {
            "fit_ids": list(refinement.validation.fit_ids),
            "held_out_ids": list(refinement.validation.held_out_ids),
            "metrics": dict(refinement.validation.metrics),
        },
    }


def _fields(record: object, names: tuple[str, ...], where: str) -> tuple[Any, ...]:
    """Return ``record``'s values for exactly ``names``; a missing or unknown key is an error.

    Uma chave desconhecida (p.ex. um erro de digitação) seria ignorada em silêncio; aqui ela
    recusa o registro.
    """
    if not isinstance(record, Mapping):
        raise CalibrationError(f"{where} must be a JSON object, got {type(record).__name__}")
    missing = [name for name in names if name not in record]
    unknown = sorted(set(record) - set(names))
    if missing or unknown:
        raise CalibrationError(f"{where}: missing field(s) {missing}, unknown field(s) {unknown}")
    return tuple(record[name] for name in names)


def _text(value: object, where: str) -> str:
    """Require and return a string field from a refinement document."""
    if not isinstance(value, str):
        raise CalibrationError(f"{where} must be a string, got {value!r}")
    return value


def _texts(value: object, where: str) -> tuple[str, ...]:
    """Require and return a JSON array containing only string fields."""
    if not isinstance(value, list):
        raise CalibrationError(f"{where} must be a JSON array of strings, got {value!r}")
    return tuple(_text(item, where) for item in value)


def _number(value: object, where: str) -> float:
    """Require a numeric JSON field and normalize it to float without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalibrationError(f"{where} must be a number, got {value!r}")
    return float(value)


def _decode_refinement(record: Mapping[str, Any]) -> CalibrationRefinement:
    """Decode and structurally validate one refinement record.

    Args:
        record: Primitive mapping containing exactly the refinement contract fields.

    Returns:
        Canonical calibration refinement.

    Raises:
        CalibrationError: If the record shape, field types or refinement rules are invalid.
    """
    (
        refinement_id,
        target,
        correction,
        method,
        estimator,
        pose_source,
        source_files,
        validation,
    ) = _fields(
        record,
        (
            "refinement_id",
            "target",
            "rotation_correction",
            "method",
            "estimator",
            "pose_source",
            "source_files",
            "validation",
        ),
        "refinement",
    )
    name = _text(refinement_id, "refinement.refinement_id")
    parent, child, base_digest = _fields(
        target, ("parent_frame", "child_frame", "base_transform_digest"), f"{name}.target"
    )
    yaw, pitch, roll, convention = _fields(
        correction,
        ("yaw_deg", "pitch_deg", "roll_deg", "convention"),
        f"{name}.rotation_correction",
    )
    module, code_identity = _fields(estimator, ("module", "code_identity"), f"{name}.estimator")
    pose_kind, pose_digest = _fields(pose_source, ("kind", "digest"), f"{name}.pose_source")
    fit_ids, held_out_ids, metrics = _fields(
        validation, ("fit_ids", "held_out_ids", "metrics"), f"{name}.validation"
    )
    if not isinstance(source_files, Mapping):
        raise CalibrationError(f"{name}.source_files must be a JSON object")
    if not isinstance(metrics, Mapping):
        raise CalibrationError(f"{name}.validation.metrics must be a JSON object")
    return CalibrationRefinement(
        refinement_id=name,
        parent_frame=FrameId(_text(parent, f"{name}.target.parent_frame")),
        child_frame=FrameId(_text(child, f"{name}.target.child_frame")),
        base_transform_digest=_text(base_digest, f"{name}.target.base_transform_digest"),
        yaw_deg=_number(yaw, f"{name}.rotation_correction.yaw_deg"),
        pitch_deg=_number(pitch, f"{name}.rotation_correction.pitch_deg"),
        roll_deg=_number(roll, f"{name}.rotation_correction.roll_deg"),
        convention=_text(convention, f"{name}.rotation_correction.convention"),
        method=_text(method, f"{name}.method"),
        estimator=_text(module, f"{name}.estimator.module"),
        code_identity=_text(code_identity, f"{name}.estimator.code_identity"),
        pose_source_kind=_text(pose_kind, f"{name}.pose_source.kind"),
        pose_source_digest=_text(pose_digest, f"{name}.pose_source.digest"),
        source_files={
            _text(key, f"{name}.source_files"): _text(value, f"{name}.source_files.{key}")
            for key, value in source_files.items()
        },
        validation=RefinementValidation(
            fit_ids=_texts(fit_ids, f"{name}.validation.fit_ids"),
            held_out_ids=_texts(held_out_ids, f"{name}.validation.held_out_ids"),
            metrics={
                _text(key, f"{name}.validation.metrics"): _number(
                    value, f"{name}.validation.metrics.{key}"
                )
                for key, value in metrics.items()
            },
        ),
    )
