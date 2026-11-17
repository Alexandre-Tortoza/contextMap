"""The #669 command: a reproducible window of corridor-02 for the ablation campaign.

The command lives under ``experiments/corridor-02-window-ingestion-20260927/scripts/`` (it is the
composition root of one experiment, not package code), but what it publishes is the input of the
whole campaign (#670): the window it resolves, the calibration and clock correction it applies,
the two sequences and the ``window.json`` record must be right, or every arm downstream inherits
the error. Everything here runs over a tiny synthetic ROS 1 bag and a synthetic "canonical"
sequence artifact shaped like the real one (its provenance written before #618, with the offset in
float seconds), never over the real dataset.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ConstantOffsetCorrection,
    ExternalPoseMeasurement,
    FrameId,
    MeiCameraModel,
    RigidTransform,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceProvenance,
    decode_timestamp_policy,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.shared import SourceTimestamp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "experiments" / "corridor-02-window-ingestion-20260927" / "scripts"

_TS = get_typestore(Stores.ROS1_NOETIC)
_MS = 1_000_000
CLOCK = "corridor-02-header"

RECORDING_EPOCH_NS = 1_724_621_400_000_000_000
"""Recording time of the bag's first message (the one on ``/rosout``)."""
HEADER_EPOCH_NS = 1_646_000_050_000_000_000
"""The sensors' header clock, years behind the recording clock, as on the real corridor-02."""
CAMERA_LATENCY_MS = 20
LIDAR_LATENCY_MS = 30
IMU_LATENCY_MS = 5

ANCHOR_SOURCE = SourceTimestamp(seconds=1_646_000_050, nanoseconds=500_000_000, clock_id=CLOCK)
"""The header stamp of the first image (t = 500 ms)..."""
ANCHOR_REFERENCE = SourceTimestamp(
    seconds=1_724_621_400, nanoseconds=520_000_000, clock_id="corridor-02-recording-time"
)
"""...and its recording time: the camera message is recorded 20 ms after its header stamp."""
OFFSET_NS = ANCHOR_REFERENCE.total_nanoseconds() - ANCHOR_SOURCE.total_nanoseconds()

WINDOW_KEYS = {
    "schema",
    "bag",
    "bag_first_message_recording_ns",
    "start_offset_s",
    "end_offset_s",
    "padding_s",
    "clock_id",
    "start_ns",
    "end_ns",
    "start_recording_ns",
    "end_recording_ns",
    "start_raw_header_ns",
    "end_raw_header_ns",
    "sequence_artifact_id",
    "sequence_location",
    "pose_sequence_artifact_id",
    "pose_sequence_location",
    "canonical_sequence_artifact_id",
    "offset_correction",
    "quality",
}
"""The #670 contract plus the recording and raw header instants and the canonical artifact."""


@pytest.fixture
def command(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    yield importlib.import_module("ingest_window")


# --- fixtures: a tiny bag, a TUM pose file and the canonical sequence ---------------------------


def _stamp(total_ns: int) -> Any:
    seconds, nanoseconds = divmod(total_ns, 1_000_000_000)
    return _TS.types["builtin_interfaces/msg/Time"](sec=seconds, nanosec=nanoseconds)


def _header(total_ns: int, frame_id: str) -> Any:
    return _TS.types["std_msgs/msg/Header"](seq=0, stamp=_stamp(total_ns), frame_id=frame_id)


def _image(header_ns: int) -> Any:
    return _TS.types["sensor_msgs/msg/Image"](
        header=_header(header_ns, "camera_1"),
        height=1,
        width=1,
        encoding="mono8",
        is_bigendian=0,
        step=1,
        data=np.array([7], dtype=np.uint8),
    )


def _scan(header_ns: int) -> Any:
    field = _TS.types["sensor_msgs/msg/PointField"]
    return _TS.types["sensor_msgs/msg/PointCloud2"](
        header=_header(header_ns, "velodyne"),
        height=1,
        width=1,
        fields=[
            field(name="x", offset=0, datatype=7, count=1),
            field(name="y", offset=4, datatype=7, count=1),
            field(name="z", offset=8, datatype=7, count=1),
        ],
        is_bigendian=False,
        point_step=12,
        row_step=12,
        data=np.zeros(12, dtype=np.uint8),
        is_dense=True,
    )


def _imu(header_ns: int) -> Any:
    types = _TS.types
    return types["sensor_msgs/msg/Imu"](
        header=_header(header_ns, "epson"),
        orientation=types["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9, dtype=np.float64),
        angular_velocity=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.01),
        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
        linear_acceleration=types["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=9.81),
        linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
    )


def write_bag(path: Path, *, lidar_hole_ms: frozenset[int] = frozenset()) -> Path:
    """Write a 5 s bag: camera and LiDAR at 10 Hz, IMU at 20 Hz, a log line first.

    Times are milliseconds after the first message. The ``/rosout`` line is the bag's first
    message and no configured topic: offsets must count from it, not from the first sensor.
    """
    types = _TS.types
    messages: list[tuple[int, str, Any]] = [(0, "/rosout", types["std_msgs/msg/String"](data="up"))]
    messages += [
        (t + CAMERA_LATENCY_MS, "/camera_1/image_raw", _image(HEADER_EPOCH_NS + t * _MS))
        for t in range(500, 5000, 100)
    ]
    messages += [
        (t + LIDAR_LATENCY_MS, "/velodyne_points", _scan(HEADER_EPOCH_NS + t * _MS))
        for t in range(550, 5000, 100)
        if t not in lidar_hole_ms
    ]
    messages += [
        (t + IMU_LATENCY_MS, "/imu/data", _imu(HEADER_EPOCH_NS + t * _MS))
        for t in range(520, 5000, 50)
    ]
    msgtypes = {
        "/rosout": "std_msgs/msg/String",
        "/camera_1/image_raw": "sensor_msgs/msg/Image",
        "/velodyne_points": "sensor_msgs/msg/PointCloud2",
        "/imu/data": "sensor_msgs/msg/Imu",
    }
    with Writer(path) as writer:
        connections = {
            topic: writer.add_connection(topic, types[msgtype].__msgtype__, typestore=_TS)
            for topic, msgtype in msgtypes.items()
        }
        for recorded_ms, topic, message in sorted(messages, key=lambda item: item[0]):
            connection = connections[topic]
            raw = _TS.serialize_ros1(message, connection.msgtype)
            writer.write(connection, RECORDING_EPOCH_NS + recorded_ms * _MS, raw)
    return path


def write_pose_file(path: Path, *, hole_ms: frozenset[int] = frozenset()) -> Path:
    """Write a TUM trajectory at 10 Hz on the (uncorrected) header clock."""
    lines = []
    for t in range(0, 6000, 100):
        if t in hole_ms:
            continue
        seconds, nanoseconds = divmod(HEADER_EPOCH_NS + t * _MS, 1_000_000_000)
        lines.append(f"{seconds}.{nanoseconds:09d} {t / 1000} 0 0 0 0 0 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def canonical_calibration() -> CalibrationSet:
    camera = SensorId("camera_1_image_raw")
    frame = FrameId("camera_1")
    model = MeiCameraModel(
        width=1,
        height=1,
        fx=761.679,
        fy=678.768,
        cx=0.5,
        cy=0.5,
        xi=1.563,
        distortion_coefficients=(-0.0708, 0.3095, 0.00046, 0.000157),
    )
    calibration_id = CalibrationReferenceId("cal-camera_1_image_raw")
    entry = CalibrationEntry(
        calibration_id=calibration_id,
        sensor_id=camera,
        frame_id=frame,
        camera_model=model,
        provenance=CalibrationProvenance(
            source_type="dataset", source_path="datasets/corridor-02/corridor-02-Intrinsics.yaml"
        ),
        content_hash=compute_content_hash(sensor_id=camera, frame_id=frame, camera_model=model),
    )
    return CalibrationSet(
        entries={calibration_id: entry},
        static_transforms=(
            RigidTransform(
                parent_frame=FrameId("epson"),
                child_frame=FrameId("velodyne"),
                translation=(0.1, 0.0, 0.2),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            RigidTransform(
                parent_frame=FrameId("epson"),
                child_frame=frame,
                translation=(0.2, 0.0, 0.1),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )


def canonical_ingestion_config(*, correction: dict[str, Any] | None = None) -> dict[str, Any]:
    """The provenance ``ingestion_config`` as the canonical run wrote it (before #618)."""
    if correction is None:
        correction = {
            "type": "constant_offset",
            "offset_seconds": OFFSET_NS / 1e9,
            "anchor_source_time": ANCHOR_SOURCE.to_record(),
            "anchor_reference_time": ANCHOR_REFERENCE.to_record(),
        }
    return {
        "topics": {
            "rgb": "/camera_1/image_raw",
            "camera_info": None,
            "lidar": "/velodyne_points",
            "imu": "/imu/data",
            "pose": None,
        },
        "required_topics": ["imu", "lidar", "rgb"],
        "timestamp_clock_id": CLOCK,
        "timestamp_policy": {
            "event_clock": "header_stamp",
            "window_clock": "recording_time",
            "correction": correction,
        },
        "window": {
            "clock_id": "ros1_bag:/somewhere/corridor-02.bag:recording_time",
            "start_seconds": 1724621400.5,
            "end_seconds": 1724621404.5,
        },
        "synchronization": {"reference_modality": "image", "tolerance_nanoseconds": 50 * _MS},
        "validation": {"allow_duplicate_timestamps": True, "on_problems": "warn"},
        "hash_source": True,
        "request_identity": "sha256:canonical-request",
        "config_identity": None,
    }


def write_canonical(
    directory: Path,
    *,
    calibration: CalibrationSet | None = None,
    ingestion_config: dict[str, Any] | None = None,
) -> Path:
    """Write a canonical-shaped sequence: calibration and provenance, no observation needed."""
    with SequenceArtifactWriter(
        output_dir=directory,
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("canonical-0001"),
    ) as writer:
        if calibration is not None:
            writer.set_calibration(calibration)
        writer.set_provenance(
            SequenceProvenance(
                source_type="ros1_bag",
                source_path="/somewhere/corridor-02.bag",
                ingestion_config=ingestion_config or canonical_ingestion_config(),
            )
        )
        writer.finalize()
    return directory


class Dataset:
    """Paths of one synthetic dataset and the arguments of the command over it."""

    def __init__(
        self,
        root: Path,
        *,
        lidar_hole_ms: frozenset[int] = frozenset(),
        pose_hole_ms: frozenset[int] = frozenset(),
    ) -> None:
        self.bag = write_bag(root / "corridor-02.bag", lidar_hole_ms=lidar_hole_ms)
        self.poses = write_pose_file(root / "corridor-02-gt.txt", hole_ms=pose_hole_ms)
        self.canonical = write_canonical(root / "canonical", calibration=canonical_calibration())
        self.outputs = root / "outputs"

    def arguments(self, *window: str) -> list[str]:
        return [
            "--bag",
            str(self.bag),
            "--pose-file",
            str(self.poses),
            "--canonical-sequence",
            str(self.canonical),
            "--outputs-root",
            str(self.outputs),
            *window,
        ]


WINDOW_2_TO_3_S = ("--start-offset-s", "2.0", "--end-offset-s", "3.0", "--padding-s", "0.5")


def _record(outputs: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((outputs / "window.json").read_text(encoding="utf-8"))
    return record


def _timestamps(sequence: Path, modality: str) -> list[int]:
    return [
        entry.timestamp.total_nanoseconds()
        for entry in SequenceArtifactReader(sequence).iter_index()
        if entry.modality == modality
    ]


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return Dataset(tmp_path)


@pytest.fixture
def ingested(command: ModuleType, dataset: Dataset) -> Dataset:
    assert command.main(dataset.arguments(*WINDOW_2_TO_3_S)) == 0
    return dataset


# --- the window ---------------------------------------------------------------------------------


def test_offsets_count_from_the_first_message_of_the_bag_whatever_its_topic(
    ingested: Dataset,
) -> None:
    record = _record(ingested.outputs)

    # A primeira mensagem do bag é o /rosout em t = 0, não a primeira imagem (t = 520 ms).
    assert record["bag_first_message_recording_ns"] == RECORDING_EPOCH_NS
    assert record["start_offset_s"] == 2.0
    assert record["end_offset_s"] == 3.0
    assert record["padding_s"] == 0.5
    assert record["start_recording_ns"] == RECORDING_EPOCH_NS + 2000 * _MS
    assert record["end_recording_ns"] == RECORDING_EPOCH_NS + 3000 * _MS


def test_the_window_is_resolved_on_the_clock_of_the_ingested_observations(
    ingested: Dataset,
) -> None:
    record = _record(ingested.outputs)
    sequence = ingested.outputs / record["sequence_location"]

    # O instante de gravação R corresponde ao header bruto R - offset, que a correção leva de volta
    # a R: no relógio corrigido, a janela tem os mesmos números que no de gravação.
    assert record["clock_id"] == CLOCK
    assert record["start_raw_header_ns"] == HEADER_EPOCH_NS + 1980 * _MS
    assert record["end_raw_header_ns"] == HEADER_EPOCH_NS + 2980 * _MS
    assert record["start_ns"] == RECORDING_EPOCH_NS + 2000 * _MS
    assert record["end_ns"] == RECORDING_EPOCH_NS + 3000 * _MS
    in_window = [
        stamp
        for stamp in _timestamps(sequence, "image")
        if record["start_ns"] <= stamp < record["end_ns"]
    ]
    # Imagens de header t em [1980, 2980) ms: t = 2000, 2100, ..., 2900.
    expected = [HEADER_EPOCH_NS + t * _MS + OFFSET_NS for t in range(2000, 3000, 100)]
    assert in_window == expected


def test_only_the_padded_window_of_the_bag_is_read(ingested: Dataset) -> None:
    record = _record(ingested.outputs)
    sequence = ingested.outputs / record["sequence_location"]

    # Janela lida, em tempo de gravação: [1500, 3500) ms; a imagem de header t é gravada em t + 20.
    images = _timestamps(sequence, "image")
    assert images == [HEADER_EPOCH_NS + t * _MS + OFFSET_NS for t in range(1500, 3500, 100)]
    assert len(_timestamps(sequence, "lidar")) == 20
    assert len(_timestamps(sequence, "imu")) == 40


# --- canonical calibration and clock correction -------------------------------------------------


def test_the_canonical_calibration_is_applied_to_both_sequences(ingested: Dataset) -> None:
    record = _record(ingested.outputs)
    canonical = SequenceArtifactReader(ingested.canonical).read_calibration()

    for location in (record["sequence_location"], record["pose_sequence_location"]):
        assert SequenceArtifactReader(ingested.outputs / location).read_calibration() == canonical
    assert record["canonical_sequence_artifact_id"] == "canonical-0001"


def test_the_offset_is_derived_from_the_canonical_anchors_and_applied_to_both_sequences(
    ingested: Dataset,
) -> None:
    record = _record(ingested.outputs)
    expected = ConstantOffsetCorrection.from_anchors(
        source_time=ANCHOR_SOURCE, reference_time=ANCHOR_REFERENCE
    )

    assert record["offset_correction"] == {
        "offset_ns": OFFSET_NS,
        "anchors": {
            "source_time": ANCHOR_SOURCE.to_record(),
            "reference_time": ANCHOR_REFERENCE.to_record(),
        },
    }
    for location in (record["sequence_location"], record["pose_sequence_location"]):
        provenance = SequenceArtifactReader(ingested.outputs / location).read_provenance()
        assert provenance is not None
        config: dict[str, Any] = dict(provenance.ingestion_config)
        assert decode_timestamp_policy(config["timestamp_policy"]).correction == expected
        assert config["timestamp_clock_id"] == CLOCK
    # A pose de header t (arquivo TUM, relógio bruto) é publicada em t + offset.
    poses = _timestamps(ingested.outputs / record["pose_sequence_location"], "external_pose")
    assert poses[0] == HEADER_EPOCH_NS + OFFSET_NS


def test_the_canonical_topics_synchronization_and_validation_are_reused(ingested: Dataset) -> None:
    record = _record(ingested.outputs)
    provenance = SequenceArtifactReader(
        ingested.outputs / record["sequence_location"]
    ).read_provenance()
    canonical = canonical_ingestion_config()

    assert provenance is not None
    config = provenance.ingestion_config
    for key in ("topics", "required_topics", "synchronization", "validation", "hash_source"):
        assert config[key] == canonical[key], key


def test_the_pose_sequence_is_the_whole_ground_truth_file_declared_as_ground_truth(
    ingested: Dataset,
) -> None:
    record = _record(ingested.outputs)
    reader = SequenceArtifactReader(ingested.outputs / record["pose_sequence_location"])
    poses = [o for o in reader.list_observations() if isinstance(o, ExternalPoseMeasurement)]

    # O adapter de arquivo de pose nunca aceita janela: a sequência auxiliar é o arquivo inteiro.
    assert len(poses) == reader.manifest.observation_counts["external_pose"] == 60
    assert {pose.provenance.raw_metadata["pose_role"] for pose in poses} == {"ground_truth"}
    assert {(str(pose.frame_id), str(pose.parent_frame)) for pose in poses} == {("epson", "map")}


# --- publication and the window record ----------------------------------------------------------


def test_both_sequences_are_published_where_the_record_says(ingested: Dataset) -> None:
    record = _record(ingested.outputs)

    for id_key, location_key in (
        ("sequence_artifact_id", "sequence_location"),
        ("pose_sequence_artifact_id", "pose_sequence_location"),
    ):
        reader = SequenceArtifactReader(ingested.outputs / record[location_key])
        assert reader.verify_integrity() == []
        assert reader.manifest.artifact_id == record[id_key]


def test_the_window_record_follows_the_campaign_contract(ingested: Dataset) -> None:
    text = (ingested.outputs / "window.json").read_text(encoding="utf-8")
    record = json.loads(text)

    assert set(record) == WINDOW_KEYS
    assert record["schema"] == "contextmap.corridor-window/v1"
    assert record["bag"] == "corridor-02.bag"
    for key in ("bag_first_message_recording_ns", "start_ns", "end_ns"):
        assert isinstance(record[key], int), key
    for key in ("start_offset_s", "end_offset_s", "padding_s"):
        assert isinstance(record[key], float), key
    assert set(record["quality"]) == {
        "max_pose_gap_ms",
        "max_lidar_gap_ms",
        "pose_gap_threshold_ms",
        "lidar_gap_threshold_ms",
        "within_thresholds",
    }
    # Nenhum caminho absoluto da máquina: só o nome do bag e locais relativos à raiz de saídas.
    assert str(ingested.outputs.parent) not in text
    for key in ("sequence_location", "pose_sequence_location"):
        assert not Path(record[key]).is_absolute()


# --- window quality -----------------------------------------------------------------------------


def test_a_clean_window_is_within_the_canonical_thresholds(ingested: Dataset) -> None:
    quality = _record(ingested.outputs)["quality"]

    assert quality == {
        "max_pose_gap_ms": pytest.approx(100.0),
        "max_lidar_gap_ms": pytest.approx(100.0),
        "pose_gap_threshold_ms": 605,
        "lidar_gap_threshold_ms": 250,
        "within_thresholds": True,
    }


def test_a_lidar_hole_above_the_threshold_is_reported_without_changing_the_window(
    command: ModuleType, tmp_path: Path
) -> None:
    # Sem as varreduras de header 2350 e 2450 ms: 2250 -> 2550 é um buraco de 300 ms na janela.
    dataset = Dataset(tmp_path, lidar_hole_ms=frozenset({2350, 2450}))

    assert command.main(dataset.arguments(*WINDOW_2_TO_3_S)) == 0

    record = _record(dataset.outputs)
    assert record["quality"]["max_lidar_gap_ms"] == pytest.approx(300.0)
    assert record["quality"]["max_pose_gap_ms"] == pytest.approx(100.0)
    assert record["quality"]["within_thresholds"] is False
    assert record["start_ns"] == RECORDING_EPOCH_NS + 2000 * _MS
    assert record["end_ns"] == RECORDING_EPOCH_NS + 3000 * _MS


def test_a_pose_hole_above_the_threshold_is_reported(command: ModuleType, tmp_path: Path) -> None:
    # Sem as poses de 2200 a 2800 ms: 2100 -> 2900 é um buraco de 800 ms na janela.
    dataset = Dataset(tmp_path, pose_hole_ms=frozenset(range(2200, 2900, 100)))

    assert command.main(dataset.arguments(*WINDOW_2_TO_3_S)) == 0

    quality = _record(dataset.outputs)["quality"]
    assert quality["max_pose_gap_ms"] == pytest.approx(800.0)
    assert quality["within_thresholds"] is False


class TestMaxGap:
    def test_only_consecutive_samples_inside_the_window_count(self, command: ModuleType) -> None:
        stamps = [0, 100 * _MS, 250 * _MS, 1000 * _MS]

        assert command.max_gap_ms(stamps, start_ns=50 * _MS, end_ns=900 * _MS) == 150.0

    def test_the_order_of_the_samples_does_not_matter(self, command: ModuleType) -> None:
        stamps = [250 * _MS, 0, 400 * _MS, 100 * _MS]

        assert command.max_gap_ms(stamps, start_ns=0, end_ns=1000 * _MS) == 150.0

    def test_fewer_than_two_samples_in_the_window_have_no_gap(self, command: ModuleType) -> None:
        stamps = [0, 100 * _MS, 900 * _MS]

        assert command.max_gap_ms(stamps, start_ns=50 * _MS, end_ns=800 * _MS) is None


# --- refusals -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "padding"),
    [
        ("2.0", "2.0", "0.5"),  # vazia
        ("3.0", "2.0", "0.5"),  # invertida
        ("-1.0", "2.0", "0.5"),  # antes da primeira mensagem
        ("2.0", "3.0", "-0.1"),  # folga negativa
        ("nan", "3.0", "0.5"),  # não finita
    ],
)
def test_an_empty_inverted_or_invalid_window_is_refused_before_anything_is_written(
    command: ModuleType, dataset: Dataset, start: str, end: str, padding: str
) -> None:
    arguments = dataset.arguments(
        "--start-offset-s", start, "--end-offset-s", end, "--padding-s", padding
    )

    assert command.main(arguments) == 2
    assert not dataset.outputs.exists()


def test_the_window_spec_rejects_an_empty_window(command: ModuleType) -> None:
    with pytest.raises(ValueError, match="end_offset_s"):
        command.WindowSpec(start_offset_s=150.0, end_offset_s=150.0, padding_s=5.0)
    assert math.isclose(
        command.WindowSpec(start_offset_s=150.0, end_offset_s=155.0, padding_s=5.0).padding_s, 5.0
    )


def test_an_existing_window_record_is_never_overwritten(
    command: ModuleType, dataset: Dataset
) -> None:
    dataset.outputs.mkdir()
    (dataset.outputs / "window.json").write_text("{}", encoding="utf-8")

    assert command.main(dataset.arguments(*WINDOW_2_TO_3_S)) == 2
    assert (dataset.outputs / "window.json").read_text(encoding="utf-8") == "{}"
    assert not (dataset.outputs / "sequences").exists()


@pytest.mark.parametrize("defect", ["no calibration", "no anchors"])
def test_a_canonical_sequence_without_calibration_or_anchors_is_refused(
    command: ModuleType, tmp_path: Path, defect: str
) -> None:
    dataset = Dataset(tmp_path)
    canonical = tmp_path / "defective-canonical"
    if defect == "no calibration":
        write_canonical(canonical)
    else:
        write_canonical(
            canonical,
            calibration=canonical_calibration(),
            ingestion_config=canonical_ingestion_config(correction={"type": "none"}),
        )
    dataset.canonical = canonical

    assert command.main(dataset.arguments(*WINDOW_2_TO_3_S)) == 2
    assert not dataset.outputs.exists()


def test_the_campaign_window_is_the_default(command: ModuleType, dataset: Dataset) -> None:
    arguments = command.build_parser().parse_args(dataset.arguments())

    assert (arguments.start_offset_s, arguments.end_offset_s) == (150.0, 155.0)
    assert arguments.padding_s == 5.0
