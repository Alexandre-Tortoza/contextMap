"""The persisted semantic spatial support round-trips and stays inventoried.

Contract evidence only: a support built over a real voxel aggregation is written and read
back, and every cell -- its space-derived identity, its entity support and its geometry
summary -- returns intact, without any dependency on a raw point index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from map_builders import MAP_ID
from voxel_builders import revisit_map, unit_policy

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import GeometryReference, aggregate_geometry, geometry_id_for
from contextmap.spatial_support import (
    EntityCellInput,
    EntityVoxelSupportPolicy,
    SemanticSpatialSupport,
    SpatialSupportArtifactError,
    SpatialSupportArtifactReader,
    SpatialSupportArtifactWriter,
    SpatialSupportManifest,
    SpatialSupportRunId,
    build_semantic_spatial_support,
)

REVISIT = (
    ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
    ((0.40, 0.40, 0.40),),
    ((2.60, 0.20, 0.80),),
)
RUN_ID = SpatialSupportRunId("support-run-0001")
LINEAGE = {"source": {"context_map": {"context_map_id": "context-map-0001"}}}


def _support() -> SemanticSpatialSupport:
    raw = revisit_map(REVISIT)
    aggregation = aggregate_geometry(raw, unit_policy(cell_m=1.0, frame="map"))
    count = raw.geometric_map.point_count
    references = tuple(
        GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))
        for index in range(count)
    )
    return build_semantic_spatial_support(
        entities=[
            EntityCellInput(entity_id=ContextEntityId("entity--alpha"), geometry_refs=references)
        ],
        aggregation=aggregation,
        raw_geometry=raw,
    )


def _write(directory: Path) -> SpatialSupportManifest:
    return SpatialSupportArtifactWriter(output_dir=directory, run_id=RUN_ID).finalize(
        support=_support(),
        source_context_map_id="context-map-0001",
        policy=EntityVoxelSupportPolicy(),
        lineage=LINEAGE,
        code_version="test",
    )


def test_reading_the_artifact_back_reproduces_the_support_exactly(tmp_path: Path) -> None:
    expected = _support()
    _write(tmp_path / "support")

    read = SpatialSupportArtifactReader(tmp_path / "support").support()

    assert read == expected


def test_a_finalized_support_verifies_and_records_its_counts(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "support")
    reader = SpatialSupportArtifactReader(tmp_path / "support")

    assert reader.verify_integrity() == []
    assert manifest.cell_count == len(_support().cells)
    assert manifest.entity_count == 1
    assert manifest.total_support_weight == revisit_map(REVISIT).geometric_map.point_count


def test_an_existing_support_directory_is_never_replaced(tmp_path: Path) -> None:
    _write(tmp_path / "support")

    with pytest.raises(SpatialSupportArtifactError, match="already exists"):
        _write(tmp_path / "support")


def test_an_unsupported_schema_version_is_refused_on_open(tmp_path: Path) -> None:
    _write(tmp_path / "support")
    manifest_path = tmp_path / "support" / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = "9.9.9"
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SpatialSupportArtifactError, match="unsupported"):
        SpatialSupportArtifactReader(tmp_path / "support")


def test_a_tampered_cell_payload_is_reported_by_the_inventory(tmp_path: Path) -> None:
    _write(tmp_path / "support")
    cells = tmp_path / "support" / "outputs" / "cells.jsonl"

    # O inventário guarda tamanho e SHA-256 de cada arquivo contratual: uma linha a mais
    # não pode passar como íntegra, mesmo sendo JSON válido.
    with cells.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    problems = SpatialSupportArtifactReader(tmp_path / "support").verify_integrity()

    assert problems
    assert any("cells.jsonl" in problem for problem in problems)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_map_id", "another-map"),
        ("frame_id", "odom"),
        ("grid_fingerprint", "sha256:changed"),
        ("aggregation_rule", "another-rule"),
        ("cell_count", 999),
        ("entity_count", 999),
        ("total_support_weight", 999),
    ],
)
def test_manifest_only_tampering_is_reported(
    tmp_path: Path, field: str, replacement: str | int
) -> None:
    _write(tmp_path / "support")
    path = tmp_path / "support" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    path.write_text(json.dumps(manifest), encoding="utf-8")

    problems = SpatialSupportArtifactReader(tmp_path / "support").verify_integrity()

    assert any(field in problem for problem in problems)
