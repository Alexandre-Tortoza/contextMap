"""The installed CLI runs Point Representation as an opt-in derived stage (issue #663).

Over the CI chain, the stage is enabled with the deterministic descriptor or with PTv3 through a
fake ``PTv3Runtime`` provider, and composed by the CLI itself from the configuration. It publishes
the ``PointRepresentationRunArtifact`` of the run, over the map and the association of the DAG,
and nothing downstream consumes it: the fusion keeps its inputs, the map keeps its lineage (#657).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from chain import (
    CannedPerception,
    CannedRegionPerception,
    CannedSemanticInterpretation,
    SyntheticSequenceIngestion,
    dry_run_executors,
)
from incremental_harness import foundation_run, write_config
from pointrep_ptv3_fakes import FakePTv3Runtime

from contextmap.evaluation import read_build_state
from contextmap.point_representation import (
    OBSERVATION_SUPPORT_ANCHOR_POLICY_ID,
    PointRepresentationRunReader,
)
from contextmap.runtime.cli import main

CODE_IDENTITY = "ci-code-663"
FRAMES = json.dumps({"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 10})
SUPPORT_POLICY = {
    "support_type": "neighborhood",
    "method": "radius",
    "radius_m": 0.5,
    "k": None,
    "max_neighbors": None,
    "preparation": {"centering": "center", "scale_normalization": "none"},
}
PTV3 = {
    "variant": "ptv3-base",
    "checkpoint": "ptv3-ci",
    "checkpoint_hash": "sha256:" + "ab" * 32,
    "device": "cpu",
    "precision": "float32",
    "grid_size_m": 0.05,
    "output_dimension": 8,
    "pooling": "center",
    "normalization": "none",
    "min_support_points": 1,
    "support_policy": SUPPORT_POLICY,
}
ENCODERS = {
    "geometric_descriptor": {"support_policy": SUPPORT_POLICY},
    "ptv3": PTV3,
}


def _providers(encoder: str) -> dict[str, Any]:
    """A fake PTv3 runtime for the ``ptv3`` encoder; the descriptor needs no provider."""
    if encoder != "ptv3":
        return {}
    return {"point_representation.encoder": lambda _config, _secrets: FakePTv3Runtime()}


def _config(tmp_path: Path, encoder: str) -> Path:
    """The CI chain's configuration with the stage enabled under ``encoder``."""
    path = write_config(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pipeline"]["stages"]["point_representation"] = True
    document["components"]["point_representation"] = {
        "center_selection": {"backend": OBSERVATION_SUPPORT_ANCHOR_POLICY_ID},
        "encoder": {"backend": encoder, encoder: ENCODERS[encoder]},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _cli(argv: list[str], **options: Any) -> tuple[int, dict[str, Any], str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdout=out, stderr=err, **options)
    text = out.getvalue()
    document = json.loads(text) if text.strip().startswith("{") else {}
    return code, document, text + err.getvalue()


def _manifest(directory: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((directory / "manifest.json").read_text("utf-8"))
    return document


def _stages(execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["stage_id"]: stage for stage in execution["stages"]}


def _run(tmp_path: Path, encoder: str) -> tuple[int, dict[str, Any], str]:
    return _cli(
        [
            "run",
            "--code-identity",
            CODE_IDENTITY,
            "-c",
            str(_config(tmp_path, encoder)),
            "--workspace",
            str(tmp_path / "ws"),
            "--set",
            f"inputs.observation_selection={FRAMES}",
            "--json",
        ],
        executors={
            "ingestion": SyntheticSequenceIngestion(),
            # Os três produtores de percepção publicam artifacts independentes.
            "region_perception": CannedRegionPerception(),
            "visual_perception": CannedPerception(),
            "semantic_interpretation": CannedSemanticInterpretation(),
        },
        providers=_providers(encoder),
        module_available=lambda _name: True,
    )


@pytest.mark.parametrize("encoder", sorted(ENCODERS))
def test_a_run_publishes_the_representation_run_over_the_map_and_association_of_the_dag(
    tmp_path: Path, encoder: str
) -> None:
    code, run, text = _run(tmp_path, encoder)

    assert code == 0, text
    stages = _stages(run["execution"])
    assert "context_map" in stages
    representation = stages["point_representation"]
    assert representation["output"]["contract"] == "PointRepresentationRunArtifact"
    run_directory = Path(run["run_directory"])
    reader = PointRepresentationRunReader(
        run_directory.parent.parent / "artifacts" / "point_representation" / run_directory.name
    )
    assert reader.verify_integrity() == []
    manifest = reader.manifest
    assert manifest.encoder.backend_id == encoder
    assert manifest.center_selection_policy_id == OBSERVATION_SUPPORT_ANCHOR_POLICY_ID
    assert manifest.code_version == CODE_IDENTITY
    assert manifest.representation_count > 0
    association = _manifest(
        run_directory.parent.parent / "artifacts" / "sensor_association" / run_directory.name
    )
    assert manifest.association_context_id == association["run_id"]
    assert str(manifest.geometric_map_id) == association["geometric_map_id"]
    assert representation["output"]["artifact_id"] == manifest.run_id
    assert representation["inputs"].keys() == {"geometry", "association"}
    # O mapa é referenciado, nunca copiado: nenhum arquivo do mapa entra no run.
    inventory = {entry.path for entry in manifest.file_inventory}
    assert not {"outputs/geometry.bin", "outputs/source-index.jsonl"} & inventory


def test_the_fusion_neither_receives_nor_cites_the_representation_run(tmp_path: Path) -> None:
    code, run, text = _run(tmp_path, "geometric_descriptor")

    assert code == 0, text
    fusion = _stages(run["execution"])["semantic_fusion"]
    assert "representation" not in fusion["inputs"]
    lineage = _manifest(
        tmp_path / "ws" / "artifacts" / "semantic_fusion" / Path(run["run_directory"]).name
    )["lineage"]
    assert lineage["point_representation_run_ids"] == []


def test_ptv3_without_a_provider_blocks_the_run_at_its_component(tmp_path: Path) -> None:
    code, _, text = _cli(
        [
            "run",
            "--code-identity",
            CODE_IDENTITY,
            "-c",
            str(_config(tmp_path, "ptv3")),
            "--workspace",
            str(tmp_path / "ws"),
        ],
        executors={
            "ingestion": SyntheticSequenceIngestion(),
            # Os três produtores de percepção publicam artifacts independentes.
            "region_perception": CannedRegionPerception(),
            "visual_perception": CannedPerception(),
            "semantic_interpretation": CannedSemanticInterpretation(),
        },
        module_available=lambda _name: True,
    )

    assert code == 1
    assert "components.point_representation.encoder" in text and "PTv3Runtime" in text
    assert not list((tmp_path / "ws").glob("*/run-*/ingestion"))


def test_a_context_run_records_the_representation_and_the_build_materializes_the_map(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "geometric_descriptor")
    workspace = tmp_path / "ws"
    foundation = foundation_run(workspace, config)
    common = ["-c", str(config), "--workspace", str(workspace), "--json"]
    code, _, text = _cli(
        ["context", "branch", "create", "b", "--from-run", str(foundation), *common]
    )
    assert code == 0, text
    code, context_run, text = _cli(
        [
            "context",
            "run",
            "--branch",
            "b",
            "--code-identity",
            CODE_IDENTITY,
            "--set",
            f"inputs.observation_selection={FRAMES}",
            *common,
        ],
        executors=dry_run_executors(),
        module_available=lambda _name: True,
    )
    assert code == 0, text
    recorded = {
        artifact["stage_id"]: artifact for artifact in context_run["context_run"]["artifacts"]
    }
    assert recorded["point_representation"]["contract"] == "PointRepresentationRunArtifact"
    assert context_run["stages"]["point_representation"] == "produced"

    code, build, text = _cli(
        ["context", "build", "--branch", "b", "--code-identity", CODE_IDENTITY, *common]
    )

    assert code == 0, text
    # A fusão não consome a representação: o build não a fornece nem a recomputa.
    assert "point_representation" not in build["context_build"]["provided"]
    build_run = Path(build["run_directory"])
    assert not (
        build_run.parent.parent / "artifacts" / "point_representation" / build_run.name
    ).exists()
    state = read_build_state(build_run, workspace=workspace)
    assert state.validation.status == "verified"
    assert state.integrity_problems == ()
