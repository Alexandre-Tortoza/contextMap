"""Dry run of the #504 incremental-workflow harness over the CI chain, end to end.

The harness under ``experiments/incremental-context-workflow-20260927/`` runs phases A to F of
the milestone acceptance execution on the GPU machine. Here it runs the same way, over the
synthetic foundation, with perception replaced by the canned double and every other stage
composed from configuration. What this proves is the path and the checks, not real behavior.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from incremental_harness import (
    DRY_RUN_EXECUTORS,
    REPOSITORY,
    checkout_copy,
    foundation_run,
    run_script,
    write_config,
)

from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.runtime import open_branch

EXPERIMENT = "incremental-context-workflow-20260927"
SCRIPTS = REPOSITORY / "experiments" / EXPERIMENT / "scripts"
WORKFLOW = Path("experiments", EXPERIMENT, "scripts", "run_workflow.py")
"""The workflow's script, relative to a checkout."""
PHASES = ("A", "B", "C", "D", "E", "F")


def _driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_workflow", SCRIPTS / "run_workflow.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Uma dataclass resolve o próprio módulo em sys.modules durante a execução dele.
    sys.modules["run_workflow"] = module
    spec.loader.exec_module(module)
    return module


def _workflow(
    tmp_path: Path,
    *extra: str,
    script: Path = SCRIPTS / "run_workflow.py",
    code_identity: str | None = "dry-run",
) -> tuple[int, Path, str]:
    config = write_config(tmp_path)
    workspace = tmp_path / "ws"
    foundation = foundation_run(workspace, config)
    identity = [] if code_identity is None else ["--code-identity", code_identity]
    result = run_script(
        script,
        "--workspace",
        str(workspace),
        "--foundation-run",
        str(foundation),
        "-c",
        str(config),
        "--first-image",
        "0",
        "--s1-images",
        "1",
        "--s2-images",
        "2",
        *identity,
        "--outputs-root",
        str(tmp_path / "out"),
        *extra,
    )
    harness = tmp_path / "out" / EXPERIMENT / "run-0001" / "harness"
    return result.returncode, harness, result.stdout + result.stderr


def _read(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def test_the_whole_workflow_runs_phase_by_phase_and_proves_each_invariant(
    tmp_path: Path,
) -> None:
    code, harness, output = _workflow(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS)

    assert code == 0, output
    report = _read(harness / "report.json")
    manifest = _read(harness / "workflow-manifest.json")
    assert report["status"] == "complete" and report["failure"] is None
    assert report["evidence_class"] == manifest["evidence_class"] == "dry_run"
    s1, s2 = manifest["selections"]["S1"], manifest["selections"]["S2"]
    assert len(s1["observation_ids"]) == 1 and len(s2["observation_ids"]) == 2
    assert not set(s1["observation_ids"]) & set(s2["observation_ids"])
    assert manifest["foundation"]["identity"]
    assert manifest["environment"]["contextmap"]
    assert list(report["phases"]) == list(PHASES)
    for phase in ("A", "B", "C", "D", "E"):
        step = report["phases"][phase]["step"]
        assert step["returncode"] == 0 and step["wall_time_seconds"] > 0
        assert step["peak_rss_mb"] > 0
    # D: a fundação é dada, nunca recalculada, e a percepção roda de novo só sobre S2.
    stages = report["phases"]["D"]["step"]["stages"]
    assert stages["state_estimation"] == stages["geometric_mapping"] == "provided"
    assert stages["visual_perception"] == stages["sensor_association"] == "executed"
    for phase in ("C", "E"):
        build = report["phases"][phase]["build"]
        assert build["validation"]["status"] == "verified"
        assert build["integrity_problems"] == []
    assert report["phases"]["C"]["build"]["counts"]["physical_observations"] == 1
    assert report["phases"]["E"]["build"]["counts"]["physical_observations"] == 3
    invariants = report["invariants"]
    assert invariants == {name: True for name in invariants}, invariants
    assert set(invariants) >= {
        "same_foundation",
        "same_map_id",
        "foundation_not_recomputed",
        "cr1_immutable",
        "cb1_immutable",
        "m1_unchanged_and_valid",
        "cb2_consumes_cr1_and_cr2",
        "fusion_counts_explained",
        "m2_lineage_closes",
        "m2_validates",
    }
    assert report["phases"]["F"]["changed_files"] == []
    assert report["cost_avoided"]["foundation_stages"] == ["geometric_mapping", "state_estimation"]
    for name in ("report.json", "workflow-manifest.json"):
        assert str(tmp_path) not in (harness / name).read_text(encoding="utf-8"), name


def test_absent_perception_providers_stop_the_workflow_in_phase_b(tmp_path: Path) -> None:
    # Um run real exige um checkout limpo: roda da própria cópia, qualquer que seja a árvore
    # de quem roda os testes, e grava o commit dela.
    code, harness, _ = _workflow(
        tmp_path, script=checkout_copy(tmp_path, EXPERIMENT) / WORKFLOW, code_identity=None
    )

    assert code != 0
    report = _read(harness / "report.json")
    assert report["status"] == "failed"
    assert report["failure"]["phase"] == "B"
    assert "resources.providers" in report["failure"]["hint"]
    assert list(report["phases"]) == ["A"]
    branch = report["phases"]["A"]["branch"]
    assert open_branch(tmp_path / "ws", dataset=CI_FIXTURE_ID, name=branch).revision == 0


def test_a_real_run_from_a_dirty_checkout_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    checkout = checkout_copy(tmp_path, EXPERIMENT)
    # A mudança não commitada é código que roda sem que o commit registrado o identifique.
    with (checkout / WORKFLOW).open("a", encoding="utf-8") as file:
        file.write("# a local change\n")

    code, harness, output = _workflow(tmp_path, script=checkout / WORKFLOW, code_identity=None)

    assert code == 2, output
    assert "dirty" in output and "commit or stash" in output
    assert not harness.exists() and not harness.parent.exists()


def test_windows_that_do_not_fit_are_refused_before_the_run_exists(tmp_path: Path) -> None:
    code, harness, output = _workflow(tmp_path, "--s2-images", "50")

    assert code == 2
    assert "do not fit" in output
    assert not harness.exists()


def test_a_changed_file_breaks_the_snapshot_of_an_immutable_run(tmp_path: Path) -> None:
    driver = _driver()
    run = tmp_path / "run"
    (run / "stage").mkdir(parents=True)
    (run / "stage" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "execution.json").write_text("{}", encoding="utf-8")
    before = driver.snapshot(run)

    (run / "stage" / "manifest.json").write_text('{"x": 1}', encoding="utf-8")
    (run / "late.json").write_text("{}", encoding="utf-8")

    assert driver.changed_files(before, driver.snapshot(run)) == [
        "late.json",
        "stage/manifest.json",
    ]
    assert driver.changed_files(before, before) == []


def test_s1_and_s2_are_consecutive_disjoint_windows_of_images() -> None:
    driver = _driver()
    images = [(index * 2, f"camera-{index:03d}") for index in range(6)]

    s1, s2 = driver.consecutive_windows(images, first=1, sizes=(2, 3))

    assert s1 == ("camera-001", "camera-002")
    assert s2 == ("camera-003", "camera-004", "camera-005")
