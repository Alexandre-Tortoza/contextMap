"""The installed CLI materializes a ContextMap on its own (issue #659).

Over the CI chain, only what no configuration can compose on a CI machine is injected: the
ingestion of the foundation (a request no configuration carries) and perception (a model). Every
stage from semantic fusion to ``context_map`` is composed by the CLI itself, from the
configuration and the code identity it was given, exactly as the installed binary composes them.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chain import (
    CannedPerception,
    CannedRegionPerception,
    CannedSemanticInterpretation,
    SyntheticSequenceIngestion,
    dry_run_executors,
)
from incremental_harness import foundation_run, write_config

from contextmap.evaluation import read_build_state
from contextmap.runtime.cli import main

CODE_IDENTITY = "ci-code-659"
FRAMES = json.dumps({"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 10})
BUILD_STAGES = ("semantic_fusion", "semantic_mapping", "entity_resolution", "spatial_relations")


def _cli(argv: list[str], **options: Any) -> tuple[int, dict[str, Any], str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdout=out, stderr=err, **options)
    text = out.getvalue()
    document = json.loads(text) if text.strip().startswith("{") else {}
    return code, document, text + err.getvalue()


def _manifest(directory: Path) -> Mapping[str, Any]:
    document: dict[str, Any] = json.loads((directory / "manifest.json").read_text("utf-8"))
    return document


def test_context_build_materializes_a_verified_map_with_no_injected_executor(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)
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

    code, build, text = _cli(
        ["context", "build", "--branch", "b", "--code-identity", CODE_IDENTITY, *common]
    )

    assert code == 0, text
    build_run = Path(build["run_directory"])
    state = read_build_state(build_run, workspace=workspace)
    assert state.validation.status == "verified"
    assert state.integrity_problems == ()
    for stage in BUILD_STAGES:
        assert (
            _manifest(build_run.parent.parent / "artifacts" / stage / build_run.name)[
                "code_version"
            ]
            == CODE_IDENTITY
        ), stage
    mapping = _manifest(build_run.parent.parent / "artifacts" / "semantic_mapping" / build_run.name)
    assert mapping["code_digest"] == (
        "sha256:" + hashlib.sha256(CODE_IDENTITY.encode("utf-8")).hexdigest()
    )
    association = (
        workspace / "artifacts" / "sensor_association" / Path(context_run["run_directory"]).name
    )
    assert _manifest(association)["code_version"] == CODE_IDENTITY


def test_run_materializes_a_verified_map_composing_every_stage_after_perception(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)

    code, run, text = _cli(
        [
            "run",
            "--code-identity",
            CODE_IDENTITY,
            "-c",
            str(config),
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
        module_available=lambda _name: True,
    )

    assert code == 0, text
    stages = {stage["stage_id"] for stage in run["execution"]["stages"]}
    assert {*BUILD_STAGES, "context_map"} <= stages
    run_directory = Path(run["run_directory"])
    for stage in BUILD_STAGES:
        assert (
            _manifest(run_directory.parent.parent / "artifacts" / stage / run_directory.name)[
                "code_version"
            ]
            == CODE_IDENTITY
        ), stage


def test_a_run_reaching_fusion_without_a_code_identity_is_blocked_before_any_stage(
    tmp_path: Path,
) -> None:
    config = write_config(tmp_path)

    code, _, text = _cli(
        [
            "run",
            "-c",
            str(config),
            "--workspace",
            str(tmp_path / "ws"),
            "--set",
            f"inputs.observation_selection={FRAMES}",
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
    assert "the plan cannot run" in text
    for stage in BUILD_STAGES:
        assert f"stages.{stage}" in text and "code identity" in text, stage
    # Nada rodou: o bloqueio é do preflight, antes do primeiro estágio.
    assert not list((tmp_path / "ws").glob("*/run-*/ingestion"))
