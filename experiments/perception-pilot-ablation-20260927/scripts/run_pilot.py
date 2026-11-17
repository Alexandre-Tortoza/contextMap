"""Perception pilot of issue #665: seven arms, each its own context branch and ContextMap.

One command runs the protocol registered in the issue (see the README) over one spatial
foundation. Before anything is created it checks the frozen selection against the foundation
and every arm's configuration (resolution, template placeholders, only declared differences)
and, for a real run, that it runs from a clean checkout of the commit it records
(``steps.resolve_code_identity``), then freezes ``protocol.json``. Arm by arm, it then creates
the arm's branch from the foundation, runs ``context run`` over the selection and ``context
build``, each one command of the ``contextmap`` CLI in a fresh, measured process, and reads the
metrics back from the published artifacts through their public readers.

``report.json`` holds every arm's steps and metrics and the baseline-replica label agreement;
``maps.json`` locates each arm's ContextMap, perception run and point representation for the
viewer, relative to the workspace. The run stops at the first step that fails and still writes
a ``failed`` report; only a ``complete`` report whose criterion passes, with exit status 0, is
the pilot's evidence. Nothing is overwritten: an existing run directory is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# Os passos são os do experimento #501, compartilhados como no #504: o mesmo comando da CLI
# num processo novo e medido, e a mesma leitura do build materializado.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "incremental-partition-invariance-20260927"
        / "scripts"
    ),
)

from pilot_arms import (
    ARMS,
    BASELINE,
    COMPARED_SECTIONS,
    REPLICA,
    VARIABLES,
    PilotArm,
    arm_layers,
    check_controlled,
    refuse_placeholders,
    resolve_arm,
)
from pilot_metrics import REPLICA_JOIN, TOP_LABELS, perception_metrics, replica_agreement
from pilot_selection import FrozenSelection, check_against_foundation, read_frozen_selection
from steps import (
    ContextCli,
    Portable,
    StepFailed,
    build_record,
    configuration_record,
    git_identity,
    image_positions,
    map_location,
    resolve_code_identity,
    step_record,
    unexpected_failure,
    write_once,
)

from contextmap.evaluation import DriverRun, ExperimentDriverError, read_build_state
from contextmap.runtime import (
    ConfigurationError,
    SpatialFoundation,
    SpatialFoundationError,
    foundation_of_run,
)

EXPERIMENT = "perception-pilot-ablation-20260927"
SCHEMA = "contextmap.experiment.perception-pilot-ablation/v2"
MAPS_SCHEMA = "contextmap.experiment.perception-pilot-ablation.maps/v2"
CHECKOUT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = (
    CHECKOUT
    / "src"
    / "contextmap"
    / "evaluation"
    / "docs"
    / "scenarios"
    / "solution-1-canonical-1.0.5-corridor-02-sample.json"
)
"""The frozen 20-image sample of corridor-02, scenario 1.0.5."""


def main(argv: list[str] | None = None) -> int:
    """Run the pilot; 0 only when every arm published a verified map, 2 for a refused run."""
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    base = args.base_config.resolve()
    shared = tuple(path.resolve() for path in args.all_arms_config or ())
    evidence_class = "real" if args.dry_run_executors is None else "dry_run"
    # Tudo que pode recusar a execução é conferido antes de criar o diretório do run.
    try:
        run = DriverRun.for_driver(
            Path(__file__),
            experiment=EXPERIMENT,
            run_number=args.run_number,
            outputs_root=None if args.outputs_root is None else args.outputs_root.resolve(),
        )
        foundation = foundation_of_run(workspace, args.foundation_run.resolve())
        selection = read_frozen_selection(args.selection_file.resolve())
        assert foundation.sequence.location is not None
        check_against_foundation(
            selection,
            sequence_artifact_id=foundation.sequence.artifact_id,
            image_ids=[
                image for _, image in image_positions(workspace / foundation.sequence.location)
            ],
        )
        layers = arm_layers(args.arms_dir.resolve())
        effectives = {slug: resolve_arm(base, shared, layer) for slug, layer in layers.items()}
        refuse_placeholders(effectives)
        differences = check_controlled(effectives)
        harness = run.output_dir("harness")
        protocol = _protocol(
            run,
            foundation=foundation,
            selection=selection,
            selection_file=args.selection_file.resolve(),
            configs={arm.slug: (base, *shared, layers[arm.slug]) for arm in ARMS},
            differences=differences,
        )
        code_identity = resolve_code_identity(
            protocol["code"],
            checkout=CHECKOUT,
            evidence_class=evidence_class,
            requested=args.code_identity,
        )
        harness.mkdir(parents=True)
    except FileExistsError:
        print(f"error: {harness} already exists: a run is never overwritten", file=sys.stderr)
        return 2
    except (ExperimentDriverError, SpatialFoundationError, ConfigurationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    protocol.update(evidence_class=evidence_class, code_identity=code_identity)
    write_once(harness / "protocol.json", protocol)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": evidence_class,
        "protocol_sha256": hashlib.sha256((harness / "protocol.json").read_bytes()).hexdigest(),
        "status": "running",
        "failure": None,
        "arms": {
            arm.slug: {"status": "not_run", "branch": _branch(run, arm), "steps": {}}
            for arm in ARMS
        },
        "replica_agreement": None,
    }
    maps: dict[str, Any] = {"schema": MAPS_SCHEMA, "paths_relative_to": "workspace", "arms": {}}
    portable = Portable(workspace=workspace, outputs=run.outputs_root, checkout=CHECKOUT)
    clis = {
        arm.slug: ContextCli(
            workspace=workspace,
            configs=(base, *shared, layers[arm.slug]),
            overrides=(),
            steps_dir=harness / "steps" / arm.slug,
            code_identity=code_identity,
            dry_run_executors=args.dry_run_executors,
        )
        for arm in ARMS
    }
    try:
        _execute(
            clis,
            workspace=workspace,
            foundation_run=args.foundation_run.resolve(),
            selection=selection,
            report=report,
            maps=maps,
        )
        report["status"] = "complete"
    except StepFailed as failure:
        arm = _failed_arm(report)
        assert arm is not None, "a command only runs inside an arm"
        report.update(
            status="failed",
            failure={"arm": arm, **portable.failure(failure, harness / "steps" / arm)},
        )
    except Exception as error:
        # Um erro fora de um comando (ao ler ou medir) também deixa o relatório, e sobe.
        failed = {"arm": _failed_arm(report), **unexpected_failure(error)}
        report.update(status="failed", failure=portable.document(failed))
        _finish(harness, report, maps)
        raise
    _finish(harness, report, maps)
    print(f"{report['status']}: {harness / 'report.json'}")
    return 0 if report["criterion"]["passes"] else 1


def _execute(
    clis: dict[str, ContextCli],
    *,
    workspace: Path,
    foundation_run: Path,
    selection: FrozenSelection,
    report: dict[str, Any],
    maps: dict[str, Any],
) -> None:
    """Run every arm in order, fill ``report`` and ``maps`` as it goes, then compare replicas.

    Raises:
        StepFailed: At the first command that does not complete; later arms stay ``not_run``.
    """
    semantic_runs: dict[str, Path] = {}
    regional_runs: dict[str, Path] = {}
    for arm in ARMS:
        entry = report["arms"][arm.slug]
        # Um braço fica "failed" até o último passo dele terminar: a exceção que interrompe o
        # run deixa esse estado no relatório.
        entry["status"] = "failed"
        cli, branch, steps = clis[arm.slug], entry["branch"], entry["steps"]
        steps["branch"] = cli.branch(branch, foundation_run=foundation_run).record
        context = cli.context_run(
            "context-run", branch=branch, selection=selection.observation_selection()
        )
        steps["context_run"] = step_record(context, workspace)
        located = {
            artifact["stage_id"]: artifact["location"]
            for artifact in context.document["context_run"]["artifacts"]
        }
        build = cli.build("build", branch=branch, revision=1)
        steps["build"] = step_record(build, workspace)
        build_run = Path(build.document["run_directory"])
        entry["build"] = build_record(
            read_build_state(build_run, workspace=workspace), build_run, workspace
        )
        semantic_runs[arm.slug] = workspace / located["semantic_interpretation"]
        regional_runs[arm.slug] = workspace / located["region_perception"]
        entry["perception"] = perception_metrics(
            semantic_runs[arm.slug], region_run=regional_runs[arm.slug]
        )
        maps["arms"][arm.slug] = {
            "branch": branch,
            "context_map_id": entry["build"]["context_map_id"],
            "paths": {
                "context_map": map_location(build),
                "region_artifact": located["region_perception"],
                "visual_evidence_artifact": located["visual_perception"],
                "semantic_interpretation_artifact": located["semantic_interpretation"],
                "point_representation_run": located.get("point_representation"),
            },
        }
        entry["status"] = "complete"
    report["replica_agreement"] = replica_agreement(
        semantic_runs[BASELINE],
        semantic_runs[REPLICA],
        REPLICA_JOIN,
        baseline_regions=regional_runs[BASELINE],
        replica_regions=regional_runs[REPLICA],
    )


def _finish(harness: Path, report: dict[str, Any], maps: dict[str, Any]) -> None:
    """Judge the pilot's criterion and write the report and the maps, never over another run."""
    complete = [entry for entry in report["arms"].values() if entry["status"] == "complete"]
    verified = [
        entry
        for entry in complete
        if entry["build"]["validation"]["status"] == "verified"
        and not entry["build"]["integrity_problems"]
    ]
    identities = {entry["build"]["context_map_id"] for entry in complete}
    # O piloto passa quando os sete braços terminam com artifacts verificados e mapas publicados,
    # um mapa por braço, e a rodada chega ao fim (a concordância da réplica incluída).
    report["criterion"] = {
        "arms": len(ARMS),
        "complete_arms": len(complete),
        "verified_maps": len(verified),
        "distinct_context_maps": len(identities),
        "passes": report["status"] == "complete"
        and len(complete) == len(verified) == len(identities) == len(ARMS),
    }
    write_once(harness / "report.json", report)
    write_once(harness / "maps.json", maps)


def _failed_arm(report: dict[str, Any]) -> str | None:
    """Return the arm the run stopped in, or ``None`` when it stopped after every arm."""
    return next(
        (slug for slug, entry in report["arms"].items() if entry["status"] == "failed"), None
    )


def _branch(run: DriverRun, arm: PilotArm) -> str:
    """Name an arm's branch after the run, so another round never meets this one's branches."""
    return f"pilot-{run.run_id}-{arm.slug}"


def _protocol(
    run: DriverRun,
    *,
    foundation: SpatialFoundation,
    selection: FrozenSelection,
    selection_file: Path,
    configs: dict[str, tuple[Path, ...]],
    differences: dict[str, Any],
) -> dict[str, Any]:
    """Return the record frozen before any step runs: what the protocol fixes for this run."""
    return {
        "schema": SCHEMA,
        "issue": 665,
        "run": run.run_id,
        "code": git_identity(CHECKOUT),
        "foundation": foundation.to_document(),
        "selection": selection.to_document(_portable_name(selection_file)),
        "variables": [variable.to_record() for variable in VARIABLES],
        "arms": [
            {
                "arm": arm.slug,
                "role": arm.role,
                "branch": _branch(run, arm),
                "assignments": dict(arm.assignments),
                **configuration_record(configs[arm.slug], ()),
                "differences_from_baseline": [item.to_record() for item in differences[arm.slug]],
            }
            for arm in ARMS
        ],
        "configuration_check": {
            "method": "contextmap.evaluation.arm_differences over every pair of arms, field by "
            "field, on the resolved configuration",
            "compared": ["components", *COMPARED_SECTIONS],
            "excluded": {"resources": "deployment: model providers, device and workspace"},
        },
        "replica_join": {
            "policy_id": REPLICA_JOIN.policy_id,
            "min_iou": REPLICA_JOIN.min_iou,
            "baseline": BASELINE,
            "replica": REPLICA,
            "regions": "accepted regions; masks loaded from each run's mask store",
            "label": "the set of a region's primary region-claim hypotheses, exact text",
            "agreement": "agreeing / compared, over joined pairs labeled on both sides",
        },
        "region_labels": {
            "top": TOP_LABELS,
            "label": "the set of a region's primary region-claim hypotheses, exact text",
        },
    }


def _portable_name(path: Path) -> str:
    """Name a file relative to the checkout when it is inside it, otherwise by its name only."""
    try:
        return path.relative_to(CHECKOUT).as_posix()
    except ValueError:
        return path.name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True, help="the runtime workspace")
    parser.add_argument(
        "--foundation-run",
        type=Path,
        required=True,
        help="completed run whose sequence, trajectory and map are the one foundation",
    )
    parser.add_argument(
        "--base-config", type=Path, required=True, help="the baseline configuration, filled"
    )
    parser.add_argument(
        "--arms-dir",
        type=Path,
        required=True,
        help="directory with one filled layer per arm, <arm>.json",
    )
    parser.add_argument(
        "--all-arms-config",
        type=Path,
        action="append",
        metavar="PATH",
        help="a layer applied to every arm, after the base and before the arm's own",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=DEFAULT_SELECTION,
        help="scenario snapshot whose selected images every arm perceives "
        "(default: the frozen corridor-02 sample of scenario 1.0.5)",
    )
    parser.add_argument(
        "--code-identity",
        help="the code every step records (default: git HEAD); a real run records the full SHA "
        "and refuses any value that is not the checked-out commit",
    )
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, help="default: the checkout's outputs/")
    parser.add_argument(
        "--dry-run-executors",
        metavar="MODULE:ATTRIBUTE",
        help="CI only: executors that replace perception; the report is marked dry_run",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
