import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.visual_features import (
    FEATURE_INDEX_SCHEMA_VERSION,
    FeatureId,
    FeaturePayloadIntegrityError,
    FeatureScope,
    FeatureStoreError,
    FeatureStoreReader,
    FeatureStoreWriter,
    VisualFeature,
    write_feature_index,
)

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="feature_extractor", provider="fake", model="fake", version="0.1"
)


def _dense_feature(feature_id: str = "feature-dense-0000") -> VisualFeature:
    return VisualFeature(
        source_observation_id=SourceObservationId("frame-0124"),
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.DENSE,
        embedding_space_id="fake-dense-space",
        shape=(4, 4, 8),
        dtype="float32",
        payload_reference=f"frame-0124/{feature_id}.npy",
        provenance=_PROVENANCE,
    )


def _region_feature(feature_id: str = "feature-region-0000") -> VisualFeature:
    return VisualFeature(
        source_observation_id=SourceObservationId("frame-0124"),
        feature_id=FeatureId(feature_id),
        scope=FeatureScope.REGION,
        embedding_space_id="fake-region-space",
        shape=(768,),
        dtype="float64",
        payload_reference=f"frame-0124/{feature_id}.npy",
        provenance=_PROVENANCE,
        region_id=RegionId("region-0001"),
    )


def test_written_payload_round_trips_exactly(tmp_path: Path) -> None:
    feature = _dense_feature()
    array = np.arange(4 * 4 * 8, dtype="float32").reshape(4, 4, 8)

    writer = FeatureStoreWriter(tmp_path)
    entry = writer.write(feature, SourceObservationId("frame-0124"), array)
    write_feature_index(tmp_path, writer.entries())

    header = json.loads((tmp_path / "feature-index.jsonl").read_text().splitlines()[0])
    assert header == {
        "record_type": "feature_index",
        "schema_version": FEATURE_INDEX_SCHEMA_VERSION,
    }

    reader = FeatureStoreReader.open(tmp_path)
    observation_id = SourceObservationId("frame-0124")
    assert reader.feature_keys() == ((observation_id, feature.feature_id),)
    assert reader.entry(observation_id, feature.feature_id) == entry

    loaded = reader.load(observation_id, feature.feature_id)
    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == array.dtype


def test_multiple_dtypes_and_shapes_round_trip(tmp_path: Path) -> None:
    dense = _dense_feature()
    region = _region_feature()
    dense_array = np.random.default_rng(0).random((4, 4, 8)).astype("float32")
    region_array = np.random.default_rng(1).random(768).astype("float64")

    writer = FeatureStoreWriter(tmp_path)
    writer.write(dense, SourceObservationId("frame-0124"), dense_array)
    writer.write(region, SourceObservationId("frame-0124"), region_array)
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    observation_id = SourceObservationId("frame-0124")
    np.testing.assert_array_equal(reader.load(observation_id, dense.feature_id), dense_array)
    np.testing.assert_array_equal(reader.load(observation_id, region.feature_id), region_array)


def test_opening_a_root_without_an_index_is_empty_not_an_error(tmp_path: Path) -> None:
    reader = FeatureStoreReader.open(tmp_path)
    assert reader.feature_keys() == ()
    with pytest.raises(FeatureStoreError, match="no persisted payload"):
        reader.entry(SourceObservationId("frame-0124"), FeatureId("does-not-exist"))


def test_same_feature_id_in_different_observations_is_unambiguous(tmp_path: Path) -> None:
    feature_id = FeatureId("feature-dense-0000")
    first = _dense_feature(str(feature_id))
    second = replace(
        first,
        source_observation_id=SourceObservationId("frame-0125"),
        payload_reference="frame-0125/feature-dense-0000.npy",
    )
    first_observation = SourceObservationId("frame-0124")
    second_observation = SourceObservationId("frame-0125")
    first_array = np.zeros((4, 4, 8), dtype="float32")
    second_array = np.ones((4, 4, 8), dtype="float32")

    writer = FeatureStoreWriter(tmp_path)
    writer.write(first, first_observation, first_array)
    writer.write(second, second_observation, second_array)
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    assert reader.feature_keys() == (
        (first_observation, feature_id),
        (second_observation, feature_id),
    )
    np.testing.assert_array_equal(reader.load(first_observation, feature_id), first_array)
    np.testing.assert_array_equal(reader.load(second_observation, feature_id), second_array)


def test_writer_rejects_duplicate_feature_key_and_payload_reference(tmp_path: Path) -> None:
    feature = _dense_feature()
    array = np.zeros((4, 4, 8), dtype="float32")
    writer = FeatureStoreWriter(tmp_path)
    writer.write(feature, SourceObservationId("frame-0124"), array)

    with pytest.raises(FeatureStoreError, match="duplicate feature payload key"):
        writer.write(
            replace(feature, payload_reference="frame-0124/duplicate.npy"),
            SourceObservationId("frame-0124"),
            array,
        )

    with pytest.raises(FeatureStoreError, match="duplicate payload_reference"):
        writer.write(
            replace(feature, source_observation_id=SourceObservationId("frame-0125")),
            SourceObservationId("frame-0125"),
            array,
        )


def test_open_rejects_unsupported_feature_index_schema(tmp_path: Path) -> None:
    index_path = tmp_path / "feature-index.jsonl"
    index_path.write_text(
        json.dumps({"record_type": "feature_index", "schema_version": "999.0.0"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FeatureStoreError, match="unsupported feature index schema_version"):
        FeatureStoreReader.open(tmp_path)


def test_open_distinguishes_a_corrupt_entry_from_an_unsupported_schema(tmp_path: Path) -> None:
    index_path = tmp_path / "feature-index.jsonl"
    index_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "record_type": "feature_index",
                        "schema_version": FEATURE_INDEX_SCHEMA_VERSION,
                    }
                ),
                json.dumps({"feature_id": "missing-required-fields"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FeatureStoreError, match="invalid feature index entry at line 2"):
        FeatureStoreReader.open(tmp_path)


def test_write_rejects_shape_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    wrong_shape_array = np.zeros((2, 2), dtype="float32")

    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="shape"):
        writer.write(feature, SourceObservationId("frame-0124"), wrong_shape_array)


def test_write_rejects_dtype_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    wrong_dtype_array = np.zeros((4, 4, 8), dtype="float64")

    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="dtype"):
        writer.write(feature, SourceObservationId("frame-0124"), wrong_dtype_array)


def test_write_rejects_payload_reference_escaping_the_store_root(tmp_path: Path) -> None:
    escaping_feature = VisualFeature(
        source_observation_id=SourceObservationId("frame-0124"),
        feature_id=FeatureId("feature-evil"),
        scope=FeatureScope.GLOBAL,
        embedding_space_id="fake-space",
        shape=(4,),
        dtype="float32",
        payload_reference="../outside.npy",
        provenance=_PROVENANCE,
    )
    array = np.zeros(4, dtype="float32")
    writer = FeatureStoreWriter(tmp_path)
    with pytest.raises(FeatureStoreError, match="boundary"):
        writer.write(escaping_feature, SourceObservationId("frame-0124"), array)


def test_load_detects_missing_payload_file(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    writer.write(feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32"))
    write_feature_index(tmp_path, writer.entries())

    (tmp_path / feature.payload_reference).unlink()

    reader = FeatureStoreReader.open(tmp_path)
    with pytest.raises(FeatureStoreError, match="missing payload"):
        reader.load(SourceObservationId("frame-0124"), feature.feature_id)


def test_load_detects_content_hash_mismatch(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    writer.write(feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32"))
    write_feature_index(tmp_path, writer.entries())

    payload_path = tmp_path / feature.payload_reference
    tampered = np.ones((4, 4, 8), dtype="float32")
    np.save(payload_path, tampered)

    reader = FeatureStoreReader.open(tmp_path)
    with pytest.raises(FeaturePayloadIntegrityError, match="content hash mismatch"):
        reader.load(SourceObservationId("frame-0124"), feature.feature_id)


def test_load_detects_corrupt_payload(tmp_path: Path) -> None:
    feature = _dense_feature()
    writer = FeatureStoreWriter(tmp_path)
    entry = writer.write(
        feature, SourceObservationId("frame-0124"), np.zeros((4, 4, 8), dtype="float32")
    )

    payload_path = tmp_path / feature.payload_reference
    corrupted = b"not a valid npy file"
    payload_path.write_bytes(corrupted)
    import hashlib

    from contextmap.visual_perception.visual_features.feature_payload_store import (
        FeaturePayloadEntry,
    )

    corrupted_entry = FeaturePayloadEntry(
        feature_id=entry.feature_id,
        source_observation_id=entry.source_observation_id,
        scope=entry.scope,
        embedding_space_id=entry.embedding_space_id,
        shape=entry.shape,
        dtype=entry.dtype,
        normalization=entry.normalization,
        payload_reference=entry.payload_reference,
        content_hash=f"sha256:{hashlib.sha256(corrupted).hexdigest()}",
        size_bytes=len(corrupted),
        provenance=entry.provenance,
    )

    reader = FeatureStoreReader(tmp_path, (corrupted_entry,))
    with pytest.raises(FeaturePayloadIntegrityError, match="unsupported or corrupt"):
        reader.load(SourceObservationId("frame-0124"), feature.feature_id)


def _current_rss_bytes() -> int:
    """RSS corrente do processo; ru_maxrss seria pico monotônico e esconderia a liberação."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    pytest.skip("VmRSS unavailable on this platform")


def test_load_does_not_hold_the_serialized_bytes_and_the_array_at_once(tmp_path: Path) -> None:
    """load() read the whole .npy into bytes, hashed it, then decoded from those bytes (#518).

    For a real dense payload that made peak memory about twice the array itself.
    """
    rows, cols = 2048, 2048  # 16 MB float32
    payload_bytes = rows * cols * 4
    feature = replace(
        _dense_feature(),
        shape=(rows, cols),
        payload_reference="frame-0124/feature-dense-0000.npy",
    )
    writer = FeatureStoreWriter(tmp_path)
    writer.write(
        feature,
        SourceObservationId("frame-0124"),
        np.full((rows, cols), 3.5, dtype="float32"),
    )
    write_feature_index(tmp_path, writer.entries())

    reader = FeatureStoreReader.open(tmp_path)
    before = _current_rss_bytes()
    loaded = reader.load(SourceObservationId("frame-0124"), FeatureId("feature-dense-0000"))
    growth = _current_rss_bytes() - before

    assert loaded.shape == (rows, cols)
    assert growth < payload_bytes * 3 // 2, (
        f"load() grew RSS by {growth / 2**20:.1f} MB for a {payload_bytes / 2**20:.0f} MB payload; "
        "the serialized bytes and the decoded array must not coexist"
    )


# Digests registrados antes da #645: a convergência para o hash em blocos compartilhado não pode
# mudar nenhum digest gravado nem a verificação na leitura. O payload grande ocupa mais de três
# blocos de 1 MiB, para exercitar as fronteiras de bloco.
_GOLDEN_PAYLOAD_DIGESTS = {
    (4, 4, 8): "sha256:8f6340bf5f033ad0fa4caaf41efc6f6ace02edc27772f0b0990277743cbc0d6c",
    (96, 96, 96): "sha256:c5a748748241bf9e82b9c0c7c9d93392e6f438d2e4f2986a16c698187ed084b4",
}


@pytest.mark.parametrize("shape", sorted(_GOLDEN_PAYLOAD_DIGESTS))
def test_the_payload_digest_is_the_one_the_run_inventory_computes(
    tmp_path: Path, shape: tuple[int, int, int]
) -> None:
    """#645: the feature store and the run inventory hash one payload file identically."""
    from contextmap.shared import FileEntry, check_file_inventory

    feature = replace(_dense_feature(f"feature-dense-{shape[0]}"), shape=shape)
    array = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    writer = FeatureStoreWriter(tmp_path)
    entry = writer.write(feature, SourceObservationId("frame-0124"), array)
    write_feature_index(tmp_path, writer.entries())
    payload = tmp_path / entry.payload_reference

    assert entry.content_hash == _GOLDEN_PAYLOAD_DIGESTS[shape]
    inventory = FileEntry(
        path=entry.payload_reference,
        size_bytes=payload.stat().st_size,
        content_hash=entry.content_hash,
    )
    assert check_file_inventory(tmp_path, [inventory]) == []
    # A leitura recalcula o digest do arquivo e o confere com o do índice.
    loaded = FeatureStoreReader.open(tmp_path).load(
        SourceObservationId("frame-0124"), feature.feature_id
    )
    np.testing.assert_array_equal(loaded, array)


def test_store_rejects_payload_assigned_to_a_different_physical_observation(tmp_path: Path) -> None:
    feature = _dense_feature()
    with pytest.raises(FeatureStoreError, match="source_observation_id"):
        FeatureStoreWriter(tmp_path).write(
            feature,
            SourceObservationId("frame-other"),
            np.zeros(feature.shape, dtype=feature.dtype),
        )
