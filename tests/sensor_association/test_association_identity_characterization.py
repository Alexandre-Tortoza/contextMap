"""Characterization of the identities a Sensor Association run persists.

Reorganizing the capability must not move a single byte of what a run writes. The values
pinned here are the ones a reader or a downstream stage can observe forever: the
``configuration_fingerprint`` that goes into every persisted ``SpatialObservation``, the
artifact schema version, the definition versions recorded in the manifest, and the content
hash of every contractual file of a known run.

These are deliberately **literal** values, not recomputed expectations: a test that derives
the expectation from the code under test cannot notice the code changing. A failure here is
either a real schema change, which belongs in a new ``schema_version`` and a new definitions
version, or an accident.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_builders import ENHANCED, NATIVE, CollectingSink, make_request, run_collecting

from contextmap.sensor_association import (
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunManifest,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.sensor_association.artifacts import SCHEMA_VERSION

SEQUENCE_NAME = "corridor-fixture"

# Fingerprints da configuração efetiva do run de referência de `run_builders`. Eles entram na
# proveniência de cada SpatialObservation gravada, então mudá-los invalida a comparação entre
# execuções já persistidas.
FINGERPRINT_WITHOUT_CHANNELS = (
    "sha256:7ea8e96d2f30b07f4deeb61e1f1a76c046023305c188d556a64c7c1668eef28e"
)
FINGERPRINT_WITH_CHANNELS = (
    "sha256:0615e7eb70d32ce1e34e877225340dbb6375903165f75f9dbebd42becbd3aa9f"
)

DEFINITION_VERSIONS = {
    "coverage": "region-support-metrics-v2",
    "dense_sampling": "dense-feature-sampling-v1",
    "diagnostics": "association-diagnostics-v3",
    "quality": "observation-quality-v3",
}

# Hash de conteúdo de cada arquivo contratual do run de referência, como o manifest o
# inventaria. Cobre o layout colunar, a ordem dos registros e a codificação JSON de ponta a
# ponta; o manifest fica de fora porque carrega `created_at`.
CONTRACTUAL_FILE_HASHES = {
    "metrics/frame-diagnostics.jsonl": (
        "sha256:767e9b735406858f41a1b1b8f0a629f573d7509859c27bbd503f029a5928fba8"
    ),
    "metrics/summary.json": (
        "sha256:c13faed55ecf5ac5b45dac178bbf60ffe6ee1c00f50e2b3c16e607f36af30d0a"
    ),
    "outputs/dense-feature-associations.jsonl": (
        "sha256:852dae751eabbc730248fe70ef69a239cf7a2c76d0e37743a5598201806d5a45"
    ),
    "outputs/dense-feature-cells.bin": (
        "sha256:08da553ac35e147ece565b8075589491899aeb7a4e9b135d89efe832541250dc"
    ),
    "outputs/geometry-support.u32": (
        "sha256:0d131a2b1c5607e17f96b7cf812a5a9c71365aa91162500fbe6ae4dcd60c9400"
    ),
    "outputs/observation-index.jsonl": (
        "sha256:654faa7de0dd9ad238ae0cf3785740865c26786ac93037764e66d29d991c1900"
    ),
    "outputs/observation-quality.jsonl": (
        "sha256:7cf1365c79ec82108a4716e97a73feae287d040dd910799cb23e5839d148bb68"
    ),
    "outputs/projection-records.jsonl": (
        "sha256:9c947ab1263222f8cb17220fe0fe7fdbc21a470237f4f112d05cd7b949b63643"
    ),
    "outputs/spatial-observations.jsonl": (
        "sha256:8e9fb05fc2a98266f0432290cfa619627161e845461dbd95400b564fa8f3ecba"
    ),
    "outputs/visibility-records.jsonl": (
        "sha256:2b05c38887190c20fe0bbb8603031c1dccfa52db9d0c51883ef12c52e1c2b395"
    ),
}


def _write_reference_run(root: Path) -> tuple[SensorAssociationRunManifest, Path]:
    """Persist the reference run of ``run_builders`` with both dense channels declared."""
    run_dir = root / "run-0001"
    writer = SensorAssociationRunWriter(
        output_dir=run_dir,
        sequence_name=SEQUENCE_NAME,
        run_id=SensorAssociationRunId("association-run-0001"),
        run_index=1,
        debug_level=SensorAssociationDebugLevel.NONE,
    )
    with writer.transaction() as run:
        outcome = SensorAssociationService().run(
            make_request(channels=(NATIVE, ENHANCED)), sink=CollectingSink(run)
        )
        manifest = run.finalize(outcome)
    return manifest, run_dir


def test_the_configuration_fingerprint_of_the_reference_run_is_pinned() -> None:
    assert run_collecting(make_request())[0].configuration_fingerprint == (
        FINGERPRINT_WITHOUT_CHANNELS
    )
    assert run_collecting(make_request(channels=(NATIVE, ENHANCED)))[
        0
    ].configuration_fingerprint == (FINGERPRINT_WITH_CHANNELS)


def test_the_observations_carry_the_run_fingerprint_into_their_provenance() -> None:
    outcome, frames = run_collecting(make_request(channels=(NATIVE, ENHANCED)))
    observations = [observation for frame in frames for observation in frame.observations]
    assert observations
    for observation in observations:
        assert observation.provenance.configuration_fingerprint == FINGERPRINT_WITH_CHANNELS
        assert observation.provenance.configuration_fingerprint == outcome.configuration_fingerprint


def test_the_artifact_schema_and_definition_versions_are_pinned(tmp_path: Path) -> None:
    manifest, run_dir = _write_reference_run(tmp_path)

    assert manifest.schema_version == SCHEMA_VERSION == "0.3.0"
    assert dict(manifest.definitions) == DEFINITION_VERSIONS
    # O manifest gravado precisa dizer o mesmo que o contrato decodificado.
    raw = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert raw["definitions"] == DEFINITION_VERSIONS
    assert raw["configuration_fingerprint"] == FINGERPRINT_WITH_CHANNELS


def test_every_contractual_file_of_the_reference_run_keeps_its_bytes(tmp_path: Path) -> None:
    manifest, _ = _write_reference_run(tmp_path)

    assert {entry.path: entry.content_hash for entry in manifest.file_inventory} == (
        CONTRACTUAL_FILE_HASHES
    )
