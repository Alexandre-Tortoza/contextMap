"""Regression: the semantic stages never change the geometry they annotate.

Permanent form of test E of ``experiments/semantic-geometry-investigation-20260930`` (gate C10):
the geometric map built with and without every semantic stage is byte-identical, and no
semantic artifact carries point coordinates of its own, only references into the map plus
summaries that recompute from it.

Contract evidence on the synthetic CI chain (``chain.py``): every stage runs its real code and
writes its real artifact; only the perception output is canned. Nothing here is evidence about
real data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# O harness não expõe um caminho só-geometria público; estes helpers são exatamente os três
# primeiros estágios que `synthetic_chain` executa, então as entradas dos dois lados são as
# mesmas por construção.
from chain import _estimate, _ingest, _map, synthetic_chain

from contextmap.geometric_mapping import GeometricMapArtifactReader, geometry_index_of

# Os únicos campos com trincas de coordenadas que um artefato semântico pode ter: resumos
# derivados do suporte (centroide, caixa, extensão) e a convenção de eixo do mapa.
DERIVED_COORDINATE_FIELDS = frozenset(
    {"centroid_m", "minimum_m", "maximum_m", "extent_m", "up_direction"}
)


def _geometry_only(workspace: Path) -> Path:
    """Run ingestion, state estimation and geometric mapping only: semantics off."""
    sequence = _ingest(workspace)
    trajectory = _estimate(workspace, sequence)
    with _map(workspace, sequence, trajectory):
        pass
    return workspace / "geometric_mapping"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_artifact_dirs(workspace: Path) -> list[Path]:
    """Every artifact the chain writes downstream of the geometric map."""
    return [
        *sorted(workspace.glob("sensor_association-*")),
        *(
            workspace / name
            for name in (
                "semantic_fusion",
                "semantic_mapping",
                "entity_resolution",
                "spatial_relations",
                "context_map",
            )
        ),
    ]


def _json_documents(directory: Path) -> Iterator[tuple[Path, Any]]:
    for path in sorted(directory.rglob("*.json*")):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield path, json.loads(line)
        else:
            yield path, json.loads(text)


def _is_coordinate(value: Any) -> bool:
    """A 3-vector of numbers with at least one float: how JSON spells an XYZ."""
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in value)
        and any(isinstance(item, float) for item in value)
    )


def _coordinate_fields(node: Any, key: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(field, value)`` for every coordinate-like value, lists of them included."""
    if isinstance(node, dict):
        if {"x", "y", "z"} <= node.keys():
            yield key or "<xyz object>", node
        for child_key, child in node.items():
            yield from _coordinate_fields(child, child_key)
    elif isinstance(node, list):
        if _is_coordinate(node):
            yield key, node
        elif node and all(_is_coordinate(item) for item in node):
            yield f"{key}[]", node
        else:
            for item in node:
                yield from _coordinate_fields(item, key)


def test_the_geometric_map_is_byte_identical_with_and_without_the_semantic_stages(
    tmp_path: Path,
) -> None:
    off = _geometry_only(tmp_path / "semantics-off")

    with synthetic_chain(tmp_path / "semantics-on") as chain:
        on = tmp_path / "semantics-on" / "geometric_mapping"
        # Os estágios semânticos rodaram depois do mapa e não reescreveram nada nele.
        assert chain.geometry.verify_integrity() == []
        with GeometricMapArtifactReader(off) as reference:
            assert chain.geometry.manifest.map_id == reference.manifest.map_id
            assert chain.geometry.manifest.run_id == reference.manifest.run_id
            assert chain.geometry.manifest.point_count == reference.manifest.point_count
            assert chain.geometry.manifest.file_inventory == reference.manifest.file_inventory

    assert (on / "outputs/geometry.bin").read_bytes() == (off / "outputs/geometry.bin").read_bytes()
    assert {path.name: _digest(path) for path in sorted((on / "outputs").iterdir())} == {
        path.name: _digest(path) for path in sorted((off / "outputs").iterdir())
    }


def test_semantic_artifacts_refer_to_the_map_only_by_identity_and_index(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path) as chain:
        map_id = chain.geometry.manifest.map_id
        point_count = chain.geometry.manifest.point_count

        reference = json.loads(
            (chain.context_map_dir / "geometry/geometry-reference.json").read_text()
        )
        assert reference == {"map_id": map_id, "point_count": point_count}

        entities = [row["record"] for _, row in _json_documents(chain.context_map_dir / "entities")]
        assert entities
        for entity in entities:
            for ref in entity["geometry_refs"]:
                assert ref["map_id"] == map_id
                assert (
                    0
                    <= geometry_index_of(map_id=map_id, geometry_id=ref["geometry_id"])
                    < (point_count)
                )

        for association in sorted(tmp_path.glob("sensor_association-*")):
            support = np.fromfile(association / "outputs/geometry-support.u32", dtype="<u4")
            assert support.size > 0
            assert int(support.max()) < point_count


def test_no_semantic_artifact_stores_point_coordinates_of_its_own(tmp_path: Path) -> None:
    with synthetic_chain(tmp_path) as chain:
        coordinate_fields = set()
        for directory in _semantic_artifact_dirs(tmp_path):
            assert directory.is_dir(), directory
            # Payloads binários a jusante do mapa são só índices u32 no mapa, nunca XYZ.
            binaries = (
                {
                    path.suffix
                    for path in (directory / "outputs").rglob("*")
                    if path.is_file() and not path.name.endswith((".json", ".jsonl"))
                }
                if (directory / "outputs").is_dir()
                else set()
            )
            assert binaries <= {".u32"}, (directory.name, binaries)
            for path, document in _json_documents(directory):
                for field, _ in _coordinate_fields(document):
                    assert not field.endswith("[]"), f"point list {field!r} in {path}"
                    coordinate_fields.add(field)
        assert coordinate_fields <= DERIVED_COORDINATE_FIELDS, coordinate_fields

        # Os resumos derivados recomputam exatamente a partir dos pontos referenciados no mapa.
        geometry = chain.geometry.geometry()
        summaries = [
            (entity.geometry.geometry_refs, entity.geometry.centroid_m, entity.geometry.bounds)
            for entity in chain.mapping.iter_entities()
        ] + [
            (outcome.support.geometry_support, outcome.support.centroid_m, outcome.support.bounds)
            for outcome in chain.fusion_outcomes
        ]
        assert summaries
        for refs, centroid, bounds in summaries:
            points = np.array([geometry.get(ref).coordinates_m for ref in refs])
            np.testing.assert_allclose(centroid, points.mean(axis=0), rtol=0, atol=1e-12)
            assert tuple(bounds.minimum_m) == tuple(points.min(axis=0))
            assert tuple(bounds.maximum_m) == tuple(points.max(axis=0))
