"""Derive the semantic spatial support of a real synthetic ContextMap over a voxel grid.

Boundary evidence: the join reads three finalized artifacts -- the ContextMap, its raw
geometric map and a voxel aggregation of that map -- and produces a support in which every
entity of the map is anchored to the voxels its points fall in, with a cell identity that
does not depend on any raw point index. Nothing here is evidence about real data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from chain import synthetic_chain
from voxel_builders import write_raw_artifact

from contextmap.artifact import ContextMapArtifactReader, artifact_digest
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    InterScanVoxelPolicy,
    VoxelAggregationArtifactReader,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    VoxelGridSpec,
)
from contextmap.spatial_support import (
    SpatialSupportArtifactError,
    SpatialSupportArtifactReader,
    SpatialSupportRunId,
    derive_semantic_spatial_support_artifact,
)

FOREIGN_SCANS = (((0.25, 0.25, 0.25), (1.50, 0.50, 0.50)), ((0.40, 0.40, 0.40),))


def _copy_raw_with_distinct_content(source: Path, target: Path) -> None:
    """Copy a valid raw artifact while changing its contractual content, not its map id."""
    shutil.copytree(source, target)
    environment_path = target / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["test_variant"] = "same-map-different-content"
    payload = (json.dumps(environment, sort_keys=True) + "\n").encode("utf-8")
    environment_path.write_bytes(payload)
    manifest_path = target / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in record["file_inventory"] if item["path"] == "environment.json")
    entry["size_bytes"] = len(payload)
    entry["content_hash"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")


def test_the_support_anchors_every_context_map_entity_to_voxels(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_dir = tmp_path / "chain" / "geometric_mapping"
        frame = chain.geometry.geometry().geometric_map.frame_id
        voxel_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=raw_dir,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )

        output = tmp_path / "support"
        manifest = derive_semantic_spatial_support_artifact(
            context_map_dir=chain.context_map_dir,
            voxel_aggregation_dir=voxel_dir,
            raw_geometry_dir=raw_dir,
            output_dir=output,
            run_id=SpatialSupportRunId("support-ci-0001"),
            code_version="test",
        )

        reader = SpatialSupportArtifactReader(output)
        assert reader.verify_integrity() == []
        support = reader.support()
        assert support.cells, "a ContextMap with entities must support at least one cell"

        with ContextMapArtifactReader.open(chain.context_map_dir) as context_map:
            known = set(context_map.entity_ids())
        placed = {str(item.entity_id) for cell in support.cells for item in cell.entity_support}
        assert placed <= known
        assert placed, "every supported cell names entities the ContextMap holds"
        assert manifest.entity_count == len(placed)
        lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))["source"]
        with GeometricMapArtifactReader(raw_dir) as raw:
            assert lineage["geometric_map"]["content_identity"] == artifact_digest(raw_dir)
            assert raw.verify_integrity() == []
        assert lineage["voxel_aggregation"]["content_identity"] == artifact_digest(voxel_dir)
        assert VoxelAggregationArtifactReader(voxel_dir).verify_integrity() == []
        with ContextMapArtifactReader.open(chain.context_map_dir) as context_map:
            source_context = lineage["context_map"]["content_identity"]
            assert source_context == context_map.manifest.content_identity


def test_three_artifacts_that_do_not_share_one_raw_map_are_refused(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_dir = tmp_path / "chain" / "geometric_mapping"
        frame = chain.geometry.geometry().geometric_map.frame_id
        voxel_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=raw_dir,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )

        # Um mapa bruto alheio: a derivação precisa recusar antes de ler geometria, porque
        # juntar entidades de um mapa à grade de outro produziria suporte sem significado.
        foreign_dir = write_raw_artifact(tmp_path / "foreign", FOREIGN_SCANS)

        with pytest.raises(SpatialSupportArtifactError, match="do not share one raw map"):
            derive_semantic_spatial_support_artifact(
                context_map_dir=chain.context_map_dir,
                voxel_aggregation_dir=voxel_dir,
                raw_geometry_dir=foreign_dir,
                output_dir=tmp_path / "support-refused",
                run_id=SpatialSupportRunId("support-ci-0002"),
                code_version="test",
            )

        assert not (tmp_path / "support-refused").exists()


def test_another_raw_artifact_with_the_same_map_id_is_refused(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_dir = tmp_path / "chain" / "geometric_mapping"
        frame = chain.geometry.geometry().geometric_map.frame_id
        voxel_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=raw_dir,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )
        other_raw = tmp_path / "other-raw"
        _copy_raw_with_distinct_content(raw_dir, other_raw)
        with GeometricMapArtifactReader(other_raw) as other:
            assert other.manifest.map_id == chain.geometry.manifest.map_id
            assert other.verify_integrity() == []

        with pytest.raises(SpatialSupportArtifactError, match=r"voxel aggregation.*raw"):
            derive_semantic_spatial_support_artifact(
                context_map_dir=chain.context_map_dir,
                voxel_aggregation_dir=voxel_dir,
                raw_geometry_dir=other_raw,
                output_dir=tmp_path / "support-refused",
                run_id=SpatialSupportRunId("support-ci-0003"),
                code_version="test",
            )


def test_context_map_and_matching_aggregation_from_distinct_raw_content_are_refused(
    tmp_path: Path,
) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_a = tmp_path / "chain" / "geometric_mapping"
        raw_b = tmp_path / "raw-b"
        _copy_raw_with_distinct_content(raw_a, raw_b)
        with GeometricMapArtifactReader(raw_b) as other:
            frame = other.geometry().geometric_map.frame_id
            assert other.manifest.map_id == chain.geometry.manifest.map_id
            assert artifact_digest(raw_b) != artifact_digest(raw_a)

        voxel_b = tmp_path / "voxel-b"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_b, run_id=VoxelAggregationRunId("voxel-b-0.05m")
        ).finalize(
            source_dir=raw_b,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )

        # Os ids de mapa coincidem e a agregação é válida para B, mas o ContextMap foi
        # construído sobre os bytes de A e suas referências não podem ser reinterpretadas.
        with pytest.raises(SpatialSupportArtifactError, match=r"geometric_map.*differs"):
            derive_semantic_spatial_support_artifact(
                context_map_dir=chain.context_map_dir,
                voxel_aggregation_dir=voxel_b,
                raw_geometry_dir=raw_b,
                output_dir=tmp_path / "support-refused",
                run_id=SpatialSupportRunId("support-ci-0004"),
                code_version="test",
            )

        assert not (tmp_path / "support-refused").exists()


def test_same_size_context_map_payload_tampering_is_refused(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_dir = tmp_path / "chain" / "geometric_mapping"
        frame = chain.geometry.geometry().geometric_map.frame_id
        voxel_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=raw_dir,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )
        entities_path = chain.context_map_dir / "entities/entities.jsonl"
        original = entities_path.read_bytes()
        offset = next(index for index, byte in enumerate(original) if byte not in b'\r\n {}[],:"')
        replacement = b"x" if original[offset : offset + 1] != b"x" else b"y"
        entities_path.write_bytes(original[:offset] + replacement + original[offset + 1 :])
        assert entities_path.stat().st_size == len(original)

        with pytest.raises(SpatialSupportArtifactError, match=r"hash|integrity|match"):
            derive_semantic_spatial_support_artifact(
                context_map_dir=chain.context_map_dir,
                voxel_aggregation_dir=voxel_dir,
                raw_geometry_dir=raw_dir,
                output_dir=tmp_path / "support-refused",
                run_id=SpatialSupportRunId("support-ci-0005"),
                code_version="test",
            )

        assert not (tmp_path / "support-refused").exists()
