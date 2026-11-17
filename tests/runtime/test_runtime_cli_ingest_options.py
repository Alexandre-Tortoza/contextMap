"""``contextmap ingest`` accepts what an ``IngestionRequest`` already models.

A recorded source that needs a temporal window, a clock correction, an external calibration or
adapter-specific settings (a TUM pose file) could only be ingested through the Python API. Each of
those becomes an explicit flag that fills the matching request field, and the published sequence
records exactly what was requested; nothing is inferred when a flag is absent.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import selected_document
from runtime_ingestion import factory as fake_factory

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationSet,
    ConstantOffsetCorrection,
    FrameId,
    RigidTransform,
    SensorId,
    TimestampPolicy,
)
from contextmap.ingestion.calibration import compute_content_hash, encode_calibration_set
from contextmap.ingestion.canonical_observations import CalibrationReferenceId
from contextmap.ingestion.source_timestamp_normalization import encode_timestamp_policy
from contextmap.runtime import compose, resolve_effective_config
from contextmap.runtime.cli import main
from contextmap.shared import SourceTimestamp


def _cli(*argv: str, **options: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    options.setdefault("module_available", lambda _name: True)
    options.setdefault("environ", {})
    code = main(list(argv), stdout=out, stderr=err, **options)
    return code, out.getvalue(), err.getvalue()


def _config(tmp_path: Path, backend: str = "ros1_bag") -> Path:
    document = selected_document()
    document["components"]["ingestion"]["source_adapter"] = {"backend": backend}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _args(tmp_path: Path, *extra: str) -> list[str]:
    source = tmp_path / "recording.bag"
    source.write_bytes(b"raw-source-bytes")
    return [
        "ingest",
        "-c",
        str(_config(tmp_path)),
        "--source",
        str(source),
        "--sequence-name",
        "corridor-02",
        "--topic",
        "rgb=/camera",
        "--topic",
        "imu=/imu",
        "--clock-id",
        "fake:header",
        "--sync-reference",
        "image",
        "--sync-tolerance-ns",
        "100000000",
        "--workspace",
        str(tmp_path / "workspace"),
        *extra,
    ]


def _provenance(tmp_path: Path) -> dict[str, Any]:
    artifact = tmp_path / "workspace" / "artifacts" / "ingestion-corridor-02"
    document: dict[str, Any] = json.loads(
        (artifact / "provenance" / "provenance.json").read_text(encoding="utf-8")
    )
    return document


def _policy() -> TimestampPolicy:
    return TimestampPolicy(
        event_clock="header_stamp",
        window_clock="recording_time",
        correction=ConstantOffsetCorrection.from_anchors(
            source_time=SourceTimestamp(seconds=100, nanoseconds=0, clock_id="fake:header"),
            reference_time=SourceTimestamp(seconds=160, nanoseconds=0, clock_id="fake:recording"),
        ),
    )


def _calibration() -> CalibrationSet:
    entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("front_camera-calib"),
        sensor_id=SensorId("front_camera"),
        frame_id=FrameId("front_camera_optical"),
        camera_model=None,
        provenance=CalibrationProvenance(source_type="manual", source_path="calibration.json"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("front_camera"),
            frame_id=FrameId("front_camera_optical"),
            camera_model=None,
        ),
    )
    imu = CalibrationEntry(
        calibration_id=CalibrationReferenceId("imu0-calib"),
        sensor_id=SensorId("imu0"),
        frame_id=FrameId("imu_link"),
        camera_model=None,
        provenance=CalibrationProvenance(source_type="manual", source_path="calibration.json"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("imu0"), frame_id=FrameId("imu_link"), camera_model=None
        ),
    )
    return CalibrationSet(
        entries={entry.calibration_id: entry, imu.calibration_id: imu},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("imu_link"),
                child_frame=FrameId("front_camera_optical"),
                translation=(0.1, 0.0, 0.2),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )


def test_a_timestamp_policy_file_is_the_request_s_policy(tmp_path: Path) -> None:
    policy_file = tmp_path / "timestamp-policy.json"
    policy_file.write_text(json.dumps(encode_timestamp_policy(_policy())), encoding="utf-8")

    code, out, err = _cli(
        *_args(tmp_path, "--timestamp-policy", str(policy_file)), adapter_factory=fake_factory()
    )

    assert code == 0, out + err
    recorded = _provenance(tmp_path)["ingestion_config"]["timestamp_policy"]
    assert recorded == encode_timestamp_policy(_policy())


def test_a_window_is_the_request_s_window(tmp_path: Path) -> None:
    built: list[Any] = []

    code, out, err = _cli(
        *_args(
            tmp_path,
            "--window-clock",
            "fake:recording",
            "--window-start",
            "0.5",
            "--window-end",
            "1.75",
        ),
        adapter_factory=fake_factory(built=built),
    )

    assert code == 0, out + err
    assert _provenance(tmp_path)["ingestion_config"]["window"] == {
        "clock_id": "fake:recording",
        "start_seconds": 0.5,
        "end_seconds": 1.75,
    }
    assert built[0].config.window is not None


@pytest.mark.parametrize(
    "partial",
    [("--window-start", "0.5"), ("--window-clock", "c", "--window-end", "2.0")],
)
def test_a_partial_window_is_a_usage_error(tmp_path: Path, partial: tuple[str, ...]) -> None:
    code, out, err = _cli(*_args(tmp_path, *partial), adapter_factory=fake_factory())

    assert code == 2
    assert "--window-clock" in out + err


def test_a_calibration_file_is_merged_into_the_published_sequence(tmp_path: Path) -> None:
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(json.dumps(encode_calibration_set(_calibration())), "utf-8")
    built: list[Any] = []

    code, out, err = _cli(
        *_args(tmp_path, "--calibration", str(calibration_file)),
        adapter_factory=fake_factory(built=built),
    )

    assert code == 0, out + err
    assert encode_calibration_set(built[0].config.calibration) == encode_calibration_set(
        _calibration()
    )
    artifact = tmp_path / "workspace" / "artifacts" / "ingestion-corridor-02"
    published = json.loads((artifact / "calibration" / "calibration.json").read_text())
    assert (
        published["static_transforms"]
        == encode_calibration_set(_calibration())["static_transforms"]
    )


@pytest.mark.parametrize("content", ["not json", json.dumps({"entries": []})])
def test_an_unreadable_calibration_file_is_a_usage_error(tmp_path: Path, content: str) -> None:
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(content, encoding="utf-8")

    code, out, err = _cli(
        *_args(tmp_path, "--calibration", str(calibration_file)), adapter_factory=fake_factory()
    )

    assert code == 2
    assert "--calibration" in out + err
    assert not (tmp_path / "workspace").exists()


def test_extra_settings_reach_the_adapter_verbatim(tmp_path: Path) -> None:
    built: list[Any] = []

    code, out, err = _cli(
        *_args(tmp_path, "--extra", "format=tum", "--extra", "pose_role=ground_truth"),
        adapter_factory=fake_factory(built=built),
    )

    assert code == 0, out + err
    assert dict(built[0].config.extra) == {"format": "tum", "pose_role": "ground_truth"}


def test_without_the_new_flags_the_request_is_unchanged(tmp_path: Path) -> None:
    built: list[Any] = []

    code, out, err = _cli(*_args(tmp_path), adapter_factory=fake_factory(built=built))

    assert code == 0, out + err
    config = built[0].config
    assert (config.window, config.calibration, dict(config.extra)) == (None, None, {})
    assert _provenance(tmp_path)["ingestion_config"]["window"] is None


def test_the_pose_file_adapter_is_composable_for_ingest(tmp_path: Path) -> None:
    from contextmap.ingestion import SourceAdapterConfig, SourceTopicMapping
    from contextmap.ingestion.adapters.pose_file_source_adapter import PoseFileSourceAdapter

    composed = compose(
        resolve_effective_config(files=[_config(tmp_path, backend="pose_file")]),
        stages=["ingestion"],
        module_available=lambda _name: True,
        environ={},
    )
    assert composed.source_adapter is not None

    adapter = composed.source_adapter(
        SourceAdapterConfig(
            source_type="pose_file",
            path=str(tmp_path / "gt.txt"),
            topics=SourceTopicMapping(pose="gt.txt"),
            timestamp_clock_id="fake:header",
            extra={
                "format": "tum",
                "parent_frame": "map",
                "body_frame": "base_link",
                "pose_role": "ground_truth",
            },
        )
    )

    assert isinstance(adapter, PoseFileSourceAdapter)
