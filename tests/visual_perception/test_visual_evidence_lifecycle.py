"""Publication, lazy reads and bounded payload memory for visual evidence."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    StageDefinition,
    StageOutcome,
    StageStatus,
    execute_stage_graph,
)
from contextmap.visual_perception import visual_evidence_artifact as run_artifact_module
from contextmap.visual_perception.visual_evidence_artifact import (
    VisualEvidenceResult,
    VisualEvidenceRunError,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
)
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="feature_extractor", provider="fake", model="fake", version="1"
)


def _run_dir(tmp_path: Path, run_index: int = 1) -> Path:
    return tmp_path / f"run-{run_index:04d}"


def _result(
    observation_id: str, run_id: str, *, features: tuple[VisualFeature, ...] = ()
) -> VisualEvidenceResult:
    return VisualEvidenceResult(
        source_observation_id=SourceObservationId(observation_id), features=features
    )


def _write_run(
    tmp_path: Path, *, run_index: int = 1, sequence_name: str = "corridor-02"
) -> VisualEvidenceRunWriter:
    return VisualEvidenceRunWriter(
        output_dir=_run_dir(tmp_path, run_index),
        run_id=f"run-{run_index:04d}",
        run_index=run_index,
        sequence_name=sequence_name,
        sequence_artifact_id="corridor-02-a1b2c3",
        region_artifact_id="region-run",
        selection_id="sha256:aaaa",
        configuration_digest="sha256:test",
    )


def _current_rss_bytes() -> int:
    """RSS corrente do processo, não o pico (`ru_maxrss` é monotônico e esconderia a liberação)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    pytest.skip("VmRSS unavailable on this platform")


def test_result_lookup_by_source_observation_id(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_result(_result("frame-0002", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = VisualEvidenceRunReader(run_dir)

    result = reader.result(SourceObservationId("frame-0002"))
    assert result.source_observation_id == "frame-0002"


def test_result_lookup_raises_for_unknown_observation(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = VisualEvidenceRunReader(run_dir)

    with pytest.raises(VisualEvidenceRunError, match="no result"):
        reader.result(SourceObservationId("does-not-exist"))


def test_writer_rejects_duplicate_source_observation(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))

    with pytest.raises(VisualEvidenceRunError, match="source_observation_id"):
        writer.add_result(_result("frame-0001", "run-0001"))


def test_stage_outcomes_are_persisted_as_metrics(tmp_path: Path) -> None:
    outcomes = execute_stage_graph(
        [
            StageDefinition(
                stage_id="region_discovery", capability="region_discovery", run=lambda ctx: ()
            )
        ]
    )

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_stage_outcomes(outcomes)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    metrics_lines = (
        (run_dir / "metrics" / "stage-timings.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(metrics_lines) == 1
    record = json.loads(metrics_lines[0])
    assert record["stage_id"] == "region_discovery"
    assert record["status"] == StageStatus.SUCCEEDED.value


def test_a_failed_stage_is_persisted_with_its_exception_type_and_traceback(
    tmp_path: Path,
) -> None:
    """VP-11: the metrics kept only ``str(error)``, empty for ``ValueError()``."""

    def failing(context: object) -> None:
        raise ValueError()

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_stage_outcomes(
        execute_stage_graph(
            [
                StageDefinition(
                    stage_id="region_discovery", capability="region_discovery", run=failing
                )
            ]
        )
    )
    writer.finalize()

    metrics_path = _run_dir(tmp_path) / "metrics" / "stage-timings.jsonl"
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert record["status"] == StageStatus.FAILED.value
    assert record["error"] == ""
    assert record["error_type"] == "ValueError"
    assert record["error_traceback"].endswith("ValueError\n")


def test_the_run_appears_exactly_at_output_dir_and_nothing_else_is_created(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))

    writer.finalize()

    assert list(tmp_path.iterdir()) == [_run_dir(tmp_path)]
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "runs.json").exists()


def test_run_id_and_run_index_are_recorded_as_supplied(tmp_path: Path) -> None:
    writer = _write_run(tmp_path, run_index=7)
    writer.add_result(_result("frame-0001", "run-0007"))

    manifest = writer.finalize()

    reopened = VisualEvidenceRunReader(_run_dir(tmp_path, 7)).manifest
    assert (manifest.run_id, manifest.run_index) == ("run-0007", 7)
    assert (reopened.run_id, reopened.run_index) == ("run-0007", 7)


def test_finalize_refuses_a_second_run_at_the_same_directory_without_altering_the_first(
    tmp_path: Path,
) -> None:
    first = _write_run(tmp_path)
    first.add_result(_result("frame-0001", "run-0001"))
    first.finalize()
    run_dir = _run_dir(tmp_path)
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    second = _write_run(tmp_path)
    second.add_result(_result("frame-0002", "run-0001"))

    with pytest.raises(VisualEvidenceRunError, match="already exists"):
        second.finalize()

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert list(tmp_path.iterdir()) == [run_dir]  # e o run recusado não deixou nada ao redor
    assert VisualEvidenceRunReader(run_dir).verify_integrity() == []


def test_opening_a_directory_without_a_manifest_fails(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()

    with pytest.raises(VisualEvidenceRunError):
        VisualEvidenceRunReader(empty_dir)


def test_verify_integrity_detects_a_missing_output_file(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    (run_dir / "outputs" / "results.jsonl").unlink()

    reader = VisualEvidenceRunReader(run_dir)
    assert any("missing file" in problem for problem in reader.verify_integrity())


def test_the_inventory_is_checked_without_reading_whole_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VP-07: finalize() and verify_integrity() read every inventoried file whole to hash it.

    A dense feature or a mask is a ``.npy`` payload that can be large, so the check must
    stream it in chunks like ``FeatureStoreReader.load`` does since #518.
    """
    feature = _dense_feature()
    read_whole: list[Path] = []
    real_read_bytes = Path.read_bytes

    def spying_read_bytes(path: Path) -> bytes:
        read_whole.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", spying_read_bytes)
    writer = _write_run(tmp_path)
    writer.add_result(replace(_result("frame-0001", "run-0001"), features=(feature,)))
    writer.add_feature_payload(
        feature, SourceObservationId("frame-0001"), np.zeros((2, 2), dtype="float32")
    )
    writer.finalize()
    reader = VisualEvidenceRunReader(_run_dir(tmp_path))
    assert reader.verify_integrity() == []

    assert [path for path in read_whole if path.suffix == ".npy"] == []


def _dense_feature(feature_id: str = "feature-dense-0000") -> VisualFeature:
    return VisualFeature(
        feature_id=FeatureId(feature_id),
        source_observation_id=SourceObservationId("frame-0001"),
        scope=FeatureScope.DENSE,
        embedding_space_id="fake-dense-space",
        shape=(2, 2),
        dtype="float32",
        payload_reference=f"frame-0001/{feature_id}.npy",
        provenance=_PROVENANCE,
    )


def test_feature_payload_is_persisted_and_lazily_loadable(tmp_path: Path) -> None:
    feature = _dense_feature()
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(feature, SourceObservationId("frame-0001"), array)
    writer.finalize()

    run_dir = _run_dir(tmp_path)
    reader = VisualEvidenceRunReader(run_dir)
    assert reader.verify_integrity() == []

    store = reader.feature_store()
    observation_id = SourceObservationId("frame-0001")
    assert store.feature_keys() == ((observation_id, feature.feature_id),)
    # Metadata is readable without loading the array.
    entry = store.entry(observation_id, feature.feature_id)
    assert entry.shape == (2, 2)

    loaded = store.load(observation_id, feature.feature_id)
    np.testing.assert_array_equal(loaded, array)


def test_finalize_rejects_payload_absent_from_result(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    with pytest.raises(
        VisualEvidenceRunError, match="does not resolve to exactly one result feature"
    ):
        writer.finalize()


def test_payload_rejects_wrong_observation_before_staging(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    with pytest.raises(VisualEvidenceRunError, match="source_observation_id"):
        writer.add_feature_payload(
            feature, SourceObservationId("frame-0002"), np.zeros((2, 2), dtype="float32")
        )
    assert not list(tmp_path.iterdir())


def test_finalize_rejects_payload_metadata_that_disagrees_with_result(tmp_path: Path) -> None:
    feature = _dense_feature()
    contradictory_feature = replace(feature, normalization="l2")
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(
        contradictory_feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    with pytest.raises(VisualEvidenceRunError, match="normalization"):
        writer.finalize()


def test_a_run_with_no_feature_payloads_has_an_empty_feature_store(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001"))
    writer.finalize()

    reader = VisualEvidenceRunReader(_run_dir(tmp_path))
    assert reader.feature_store().feature_keys() == ()


def test_iter_results_streams_instead_of_materializing_every_result(tmp_path: Path) -> None:
    """A reader must be able to walk a run without holding all of it (#517)."""
    writer = _write_run(tmp_path)
    for index in range(5):
        writer.add_result(_result(f"frame-{index:04d}", "run-0001"))
    writer.finalize()

    reader = VisualEvidenceRunReader(_run_dir(tmp_path))
    streamed = reader.iter_results()
    assert not isinstance(streamed, list), "iter_results() must be lazy, not a built list"
    assert [r.source_observation_id for r in streamed] == [
        r.source_observation_id for r in reader.list_results()
    ]


def test_result_lookup_stops_at_the_match_instead_of_decoding_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """result() used to call list_results(), so one lookup cost the entire run (#517)."""
    writer = _write_run(tmp_path)
    for index in range(20):
        writer.add_result(_result(f"frame-{index:04d}", "run-0001"))
    writer.finalize()

    reader = VisualEvidenceRunReader(_run_dir(tmp_path))
    decoded = 0
    real_decode = run_artifact_module._decode_result

    def counting_decode(payload: dict[str, object]) -> VisualEvidenceResult:
        nonlocal decoded
        decoded += 1
        return real_decode(payload)

    monkeypatch.setattr(run_artifact_module, "_decode_result", counting_decode)
    found = reader.result(SourceObservationId("frame-0000"))

    assert found.source_observation_id == SourceObservationId("frame-0000")
    assert decoded == 1, f"looking up the first result decoded {decoded} of 20 results"


def _staging_dir(tmp_path: Path, run_index: int = 1) -> Path:
    staging = [p for p in tmp_path.iterdir() if p.name.startswith(f".tmp-run-{run_index:04d}-")]
    assert len(staging) == 1, f"expected exactly one staging directory, found {staging}"
    return staging[0]


def test_feature_arrays_reach_disk_when_added_not_at_finalize(tmp_path: Path) -> None:
    """Dense DINOv2 payloads are megabytes per frame; they must not queue up in RAM."""
    feature = _dense_feature()
    writer = _write_run(tmp_path)
    writer.add_result(_result("frame-0001", "run-0001", features=(feature,)))
    writer.add_feature_payload(
        feature,
        SourceObservationId("frame-0001"),
        np.zeros((2, 2), dtype="float32"),
    )

    payloads = list((_staging_dir(tmp_path) / "outputs" / "features").rglob("*.npy"))
    assert payloads, "add_feature_payload() must persist the array instead of buffering it"


def test_writer_memory_does_not_grow_with_the_number_of_feature_payloads(
    tmp_path: Path,
) -> None:
    """A real run queues one dense array per frame; 1500 frames must not cost 1500 arrays."""
    frames = 30
    rows = cols = 768  # 2.25 MB per array, the order of a real DINOv2 dense payload
    retained_bytes_if_buffered = frames * rows * cols * 4

    writer = _write_run(tmp_path)
    before = _current_rss_bytes()
    for index in range(frames):
        feature = replace(
            _dense_feature(f"feature-dense-{index:04d}"),
            source_observation_id=SourceObservationId(f"frame-{index:04d}"),
            shape=(rows, cols),
            payload_reference=f"frame-{index:04d}/feature-dense-{index:04d}.npy",
        )
        writer.add_feature_payload(
            feature,
            SourceObservationId(f"frame-{index:04d}"),
            # np.zeros() would not fault its pages in, so it would not show up in RSS at all.
            np.full((rows, cols), float(index), dtype="float32"),
        )
    growth = _current_rss_bytes() - before

    assert growth < retained_bytes_if_buffered // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {frames} dense payloads; "
        f"buffering them all would cost ~{retained_bytes_if_buffered / 2**20:.0f} MB"
    )


def test_writer_memory_does_not_grow_with_retained_stage_outputs(tmp_path: Path) -> None:
    """The real driver hands the region_discovery outcome straight to add_stage_outcomes().

    Its ``output`` is the same tuple of mask-bearing Region2D that add_result() receives, so
    retaining the StageOutcome retains the pixels a second time — and stage timings only ever
    serialize stage_id/status/duration_ms/error, never the output.
    """
    frames = 40
    retained_bytes_if_buffered = frames * 640 * 480 * 8

    writer = _write_run(tmp_path)
    before = _current_rss_bytes()
    for _ in range(frames):
        writer.add_stage_outcomes(
            (
                StageOutcome(
                    stage_id="region_discovery",
                    status=StageStatus.SUCCEEDED,
                    output=np.ones((640, 480, 8), dtype="float64"),
                    duration_ms=1.0,
                ),
            )
        )
    growth = _current_rss_bytes() - before

    assert growth < retained_bytes_if_buffered // 4, (
        f"writer grew {growth / 2**20:.1f} MB over {frames} stage outcomes; "
        f"retaining their outputs would cost ~{retained_bytes_if_buffered / 2**20:.0f} MB"
    )
