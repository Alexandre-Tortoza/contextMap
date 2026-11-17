"""The identity of the join rule, as the run records it.

The policy is what tells two runs apart when the rule itself changes: the grid varies and
travels in its own fingerprint, but what the join *computes* -- one unit of support per raw
point of an entity that falls in the voxel -- is this policy's version.
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
    SCHEMA_VERSION,
    EntityCellInput,
    EntityVoxelSupportPolicy,
    SemanticSpatialSupport,
    SpatialSupportArtifactError,
    SpatialSupportArtifactReader,
    SpatialSupportArtifactWriter,
    SpatialSupportRunId,
    build_semantic_spatial_support,
)

REVISIT = (
    ((0.25, 0.25, 0.25), (0.75, 0.50, 0.50), (0.50, 0.75, 0.25), (1.50, 0.50, 0.50)),
    ((0.40, 0.40, 0.40),),
    ((2.60, 0.20, 0.80),),
)
RUN_ID = SpatialSupportRunId("support-run-0001")


def _support() -> SemanticSpatialSupport:
    raw = revisit_map(REVISIT)
    aggregation = aggregate_geometry(raw, unit_policy(cell_m=1.0, frame="map"))
    references = tuple(
        GeometryReference(map_id=MAP_ID, geometry_id=geometry_id_for(map_id=MAP_ID, index=index))
        for index in range(raw.geometric_map.point_count)
    )
    return build_semantic_spatial_support(
        entities=[
            EntityCellInput(entity_id=ContextEntityId("entity--alpha"), geometry_refs=references)
        ],
        aggregation=aggregation,
        raw_geometry=raw,
    )


def _write(directory: Path) -> None:
    SpatialSupportArtifactWriter(output_dir=directory, run_id=RUN_ID).finalize(
        support=_support(),
        source_context_map_id="context-map-0001",
        policy=EntityVoxelSupportPolicy(),
        lineage={"source": {"context_map": {"context_map_id": "context-map-0001"}}},
        code_version="test",
    )


def test_the_policy_fingerprint_is_a_pure_function_of_what_the_join_computes() -> None:
    first = EntityVoxelSupportPolicy()
    again = EntityVoxelSupportPolicy()

    assert first.fingerprint() == again.fingerprint()
    assert first.fingerprint().startswith("sha256:")
    record = first.to_record()
    # O peso é contagem de medições, nunca probabilidade: a semântica entra na identidade,
    # para que mudá-la obrigue uma versão nova em vez de passar silenciosamente.
    assert record["weight_semantics"] == "measurement-count"
    assert record["policy_id"] == "entity-voxel-support"


def test_a_finalized_run_records_the_policy_identity(tmp_path: Path) -> None:
    _write(tmp_path / "support")

    config = json.loads((tmp_path / "support" / "config.json").read_text(encoding="utf-8"))
    manifest = SpatialSupportArtifactReader(tmp_path / "support").manifest

    assert config["policy"] == EntityVoxelSupportPolicy().to_record()
    assert config["policy_fingerprint"] == EntityVoxelSupportPolicy().fingerprint()
    assert manifest.policy_fingerprint == EntityVoxelSupportPolicy().fingerprint()


def test_every_contractual_json_document_ends_with_a_newline(tmp_path: Path) -> None:
    _write(tmp_path / "support")

    for relative in ("config.json", "lineage.json", "environment.json", "metrics/support.json"):
        payload = (tmp_path / "support" / relative).read_bytes()
        assert payload.endswith(b"\n"), relative


def test_a_run_of_the_previous_schema_is_refused(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == "0.2.0"
    _write(tmp_path / "support")
    manifest_path = tmp_path / "support" / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["schema_version"] = "0.1.0"
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    # Durante v0.x não há camada de migração: um run do schema anterior é recusado, e o
    # caminho suportado é derivar outro run.
    with pytest.raises(SpatialSupportArtifactError, match="unsupported"):
        SpatialSupportArtifactReader(tmp_path / "support")
