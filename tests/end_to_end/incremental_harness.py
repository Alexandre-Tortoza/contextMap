"""Shared setup of the runs of the incremental real-data harnesses (#501, #504, #665, #670).

A dry run executes a harness exactly as on the GPU machine, over the CI chain: the foundation is
a completed runtime run of the synthetic sequence, every stage but perception is composed by the
runtime from a fully parameterized configuration, and perception is the one stage replaced, by
the canned double of ``chain.dry_run_executors``. Nothing here is evidence about real data.

A real run must run exactly the commit it records, so a test that executes one runs the harness
from a copy of it in a git repository of its own (:func:`checkout_copy`), never from the tree
running the suite, whose state the test does not control.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chain import SyntheticSequenceIngestion

from contextmap.evaluation import canonical_real_scenario, scenario_runtime_document
from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.runtime.cli import main

REPOSITORY = Path(__file__).resolve().parents[2]
DRY_RUN_EXECUTORS = "chain:dry_run_executors"
PARTITION_SCRIPTS = Path("experiments", "incremental-partition-invariance-20260927", "scripts")
"""The #501 steps every harness imports, relative to a checkout."""

# Os valores de política da cadeia de CI (`test_runtime_chain.py`), agora como configuração:
# no dry run só a percepção é substituída, e o runtime compõe o resto como no run real.
_PARAMETERS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("state_estimation", "estimator", "external_pose"): {
        "reference_frame": "odom",
        "body_frame": "base_link",
    },
    ("geometric_mapping", "pose_lookup", "lookup-policy-v1"): {"mode": "exact"},
    ("geometric_mapping", "motion_correction", "motion-correction-v1"): {
        "raw": "accept",
        "unknown": "warn",
    },
    ("sensor_association", "occlusion", "conservative-depth-support-v1"): {
        "cell_size_px": 4,
        "neighborhood_radius_cells": 0,
        "depth_margin_m": 0.1,
        "depth_margin_ratio": 0.02,
    },
    ("sensor_association", "tolerances", "diagnostic-tolerances-v1"): {
        "max_pose_time_delta_ns": 1_000_000,
        "max_map_window_offset_ns": None,
        "max_reprojection_p95_px": None,
        "max_reprojection_invalid_rate": None,
    },
    ("sensor_association", "pose_policy", "lookup-policy-v1"): {"mode": "exact"},
    ("semantic_fusion", "support", "geometry-jaccard-support-v1"): {
        "min_geometry_count": 1,
        "min_overlap": 0.5,
    },
    ("semantic_fusion", "accumulation", "baseline-evidence-accumulation-v1"): {
        "abstention_labels": ["unknown"],
    },
    ("semantic_mapping", "geometry_summary", "entity-geometry-summary-v1"): {
        "sparse_point_threshold": 3,
        "connectivity_radius_m": 0.5,
    },
    ("entity_resolution", "retrieval", "entity-candidate-retrieval-v1"): {
        "centroid_radius_m": 20.0,
        "bounds_margin_m": 0.1,
    },
    ("entity_resolution", "resolution", "conservative-staged-resolution-v1"): {
        "use_channels": ["geometry"],
        "min_supporting_channels": 1,
    },
    ("entity_resolution", "geometry_comparison", "entity-geometry-comparison-v1"): {
        "min_shared_support_jaccard": 0.5,
        "min_bounds_iou": 0.5,
        "min_bounds_containment": 0.9,
        "min_conflict_gap_m": 0.5,
        "min_extent_ratio": 0.3,
    },
    ("spatial_relations", "frame_conventions", "map-frame-conventions-v1"): {
        "map_frame": "odom",
        "up_axis": "+z",
        "forward_axis": "+x",
    },
    ("spatial_relations", "candidate", "bounds-neighborhood-candidates-v1"): {
        "predicates": ["next_to", "touching"],
        "proximity_radius_m": 0.6,
        "directional_radius_m": 2.0,
    },
    ("spatial_relations", "geometry_summary", "entity-geometry-summary-v1"): {
        "sparse_point_threshold": 3,
        "connectivity_radius_m": 0.5,
    },
}


def write_config(directory: Path) -> Path:
    """Write the canonical profile with the CI chain's parameters; perception has none."""
    document = scenario_runtime_document(canonical_real_scenario())
    document["inputs"] = {"sequence": CI_FIXTURE_ID}
    for (capability, slot, backend), values in _PARAMETERS.items():
        document["components"][capability][slot][backend] = values
    path = directory / "dry-run.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def machine_values(value: Any) -> Any:
    """Fill each template placeholder with a value of the type its parameter needs.

    A placeholder names what the GPU machine fills (``REPLACE...``): a 40-hex revision, a
    precision, a token budget or a text such as a provider target or a model version.
    """
    if isinstance(value, dict):
        return {key: machine_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [machine_values(item) for item in value]
    if not isinstance(value, str) or "REPLACE" not in value:
        return value
    if "40-HEX" in value:
        return "0" * 40
    if "PRECISION" in value:
        return "float16"
    if "TOKEN-BUDGET" in value:
        return 256
    return "dry-run"


def foundation_run(workspace: Path, config: Path) -> Path:
    """Run the foundation stages through the CLI and return the completed run directory."""
    code = main(
        [
            "run",
            "--stage",
            "geometric_mapping",
            "-c",
            str(config),
            "--workspace",
            str(workspace),
            "--json",
        ],
        executors={"ingestion": SyntheticSequenceIngestion()},
        module_available=lambda _name: True,
        environ={},
    )
    assert code == 0, "the dry-run foundation must complete"
    return workspace / CI_FIXTURE_ID / "run-0001"


def run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a harness script as the GPU machine would, with the CI doubles importable."""
    environment = dict(os.environ)
    paths = [
        str(REPOSITORY / "src"),
        str(REPOSITORY / "tests" / "end_to_end"),
        str(REPOSITORY / "tests"),
    ]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def run_git(checkout: Path, *arguments: str) -> str:
    """Run git in a checkout and return what it printed, stripped; a failure raises."""
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def commit_all(checkout: Path, message: str) -> str:
    """Commit everything in a checkout and return the full SHA of the new commit."""
    run_git(checkout, "add", "--all")
    # Identidade e assinatura explícitas: o runner da CI não tem identidade git.
    run_git(
        checkout,
        "-c",
        "user.email=harness@example.invalid",
        "-c",
        "user.name=Harness",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "--allow-empty",
        "--message",
        message,
    )
    return run_git(checkout, "rev-parse", "HEAD")


def checkout_copy(
    tmp_path: Path, experiment: str, *, git: str = "committed", imports: Sequence[str] = ()
) -> Path:
    """Copy a harness bundle and the #501 scripts it imports into a checkout of their own.

    A harness takes its checkout from where its script lives, so the copy is the checkout whose
    commit and tree a run records, whatever the state of the repository running the tests.

    Args:
        tmp_path: The test's directory; the checkout is its ``checkout/``.
        experiment: The bundle under ``experiments/`` whose script runs.
        git: ``"committed"`` (a repository with everything committed), ``"initialized"`` (a
            repository without any commit) or ``"none"`` (no repository at all).
        imports: Other bundles under ``experiments/`` the harness imports or runs, copied too.

    Returns:
        The checkout; each script is at the same path relative to it as in the repository.
    """
    checkout = tmp_path / "checkout"
    bundles = (Path("experiments", experiment), *(Path("experiments", item) for item in imports))
    for bundle in (*bundles, PARTITION_SCRIPTS):
        # O bundle do #501 já contém os próprios scripts.
        shutil.copytree(
            REPOSITORY / bundle,
            checkout / bundle,
            ignore=shutil.ignore_patterns("__pycache__"),
            dirs_exist_ok=True,
        )
    # Os processos do harness gravam bytecode ao lado dos scripts; no checkout real o
    # .gitignore o deixa fora do estado da árvore.
    (checkout / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    if git == "none":
        return checkout
    run_git(checkout, "init", "--quiet")
    if git == "committed":
        commit_all(checkout, "harness")
    return checkout
