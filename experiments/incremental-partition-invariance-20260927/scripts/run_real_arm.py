"""Real-data arm of issue #501: partition invariance and repeated-inference correlation.

One command runs the pre-registered protocol end to end over one spatial foundation (see the
README): it freezes the protocol record, runs the ContextRuns and builds ``W1``, ``W2``, ``C``,
``R`` and the scenario-3 build through the ``contextmap context`` CLI, one measured process per
step, and writes ``report.json`` with both layers, scenarios 2 and 3 and every count on its own.

The run stops at the first step that fails and still writes a ``failed`` report; only a
``complete`` report with exit status 0 is evidence for #501. Nothing is overwritten: an existing
run directory is refused. A real run is refused, before anything exists, unless it runs from a
clean checkout of the commit it records (``steps.resolve_code_identity``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from steps import (
    ContextCli,
    StepFailed,
    build_record,
    configuration_record,
    explicit,
    git_identity,
    image_positions,
    plan_partition,
    resolve_code_identity,
    step_record,
    unexpected_failure,
    write_once,
)

from contextmap.evaluation import (
    BuildState,
    DriverRun,
    ExperimentDriverError,
    claim_agreement,
    decide_equivalence,
    encode_incremental_context,
    judge_downstream_only,
    judge_layer_one,
    judge_repeated_inference,
    paired_agreement,
    partition_effect,
    read_build_state,
)
from contextmap.runtime import ConfigurationError, SpatialFoundationError, foundation_of_run

EXPERIMENT = "incremental-partition-invariance-20260927"
SCHEMA = "contextmap.experiment.incremental-partition-invariance/v1"
DEFAULT_SEED = 20260926
"""The bootstrap seed, fixed with the protocol (pre-registered on 2026-09-26)."""
DEFAULT_SCENARIO3 = (
    "components.spatial_relations.candidate.bounds-neighborhood-candidates-v1."
    "proximity_radius_m=0.3"
)
"""The downstream-only change of scenario 3: a spatial-relations candidate radius."""
CHECKOUT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    """Run the arm; 0 only for a complete run, 1 for a failed one, 2 for a refused one."""
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    configs = tuple(path.resolve() for path in args.config)
    overrides = tuple(args.set or ())
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
        assert foundation.sequence.location is not None
        partition = plan_partition(
            image_positions(workspace / foundation.sequence.location),
            first=args.first_image,
            count=args.images,
            chunks=args.chunks,
        )
        code = git_identity(CHECKOUT)
        protocol = _protocol(args, run.run_id, configs, overrides, foundation, partition, code)
        code_identity = resolve_code_identity(
            code, checkout=CHECKOUT, evidence_class=evidence_class, requested=args.code_identity
        )
        harness = run.output_dir("harness")
        harness.mkdir(parents=True)
    except FileExistsError:
        print(f"error: {harness} already exists: a run is never overwritten", file=sys.stderr)
        return 2
    except (ExperimentDriverError, SpatialFoundationError, ConfigurationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    protocol.update(evidence_class=evidence_class, code_identity=code_identity)
    write_once(harness / "protocol.json", protocol)

    cli = ContextCli(
        workspace=workspace,
        configs=configs,
        overrides=overrides,
        steps_dir=harness / "steps",
        code_identity=code_identity,
        dry_run_executors=args.dry_run_executors,
    )
    branches = {kind: f"partition-{run.run_id}-{kind}" for kind in ("whole", "chunked")}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": evidence_class,
        "protocol_sha256": hashlib.sha256((harness / "protocol.json").read_bytes()).hexdigest(),
        "branches": branches,
        "steps": {},
        "builds": {},
        "status": "running",
        "failure": None,
    }
    try:
        _execute(cli, args, workspace, partition, branches, report)
        report["status"] = "complete"
    except StepFailed as failure:
        report.update(status="failed", failure=failure.to_document())
    except Exception as error:
        # Um erro fora de um comando (ao ler ou comparar) também deixa o relatório, e sobe.
        report.update(status="failed", failure=unexpected_failure(error))
        write_once(harness / "report.json", report)
        raise
    write_once(harness / "report.json", report)
    print(f"{report['status']}: {harness / 'report.json'}")
    return 0 if report["status"] == "complete" else 1


def _execute(
    cli: ContextCli,
    args: argparse.Namespace,
    workspace: Path,
    partition: Any,
    branches: dict[str, str],
    report: dict[str, Any],
) -> None:
    """Run every step in protocol order and compare the builds; fill ``report`` as it goes."""
    for name in branches.values():
        cli.branch(name, foundation_run=args.foundation_run.resolve())
    ids: dict[str, str] = {}

    def context_run(name: str, branch: str, selection: dict[str, Any]) -> None:
        step = cli.context_run(name, branch=branches[branch], selection=selection)
        ids[name] = step.document["context_run"]["context_run_id"]
        report["steps"][name] = step_record(step, workspace)

    context_run("W1", "whole", explicit(partition.whole))
    context_run("W2", "whole", partition.rerun)
    for index, chunk in enumerate(partition.chunks, start=1):
        context_run(f"C{index}", "chunked", explicit(chunk))

    chunked_revision = len(partition.chunks)
    builds = {
        "W1": dict(branch=branches["whole"], revision=1),
        "W2": dict(branch=branches["whole"], revision=2, members=[ids["W2"]]),
        "C": dict(branch=branches["chunked"], revision=chunked_revision),
        "R": dict(branch=branches["whole"], revision=2),
        "C_policy": dict(
            branch=branches["chunked"], revision=chunked_revision, overrides=[args.scenario3_set]
        ),
    }
    states: dict[str, BuildState] = {}
    for name, options in builds.items():
        step = cli.build(f"build-{name}", **options)  # type: ignore[arg-type]
        report["steps"][f"build-{name}"] = step_record(step, workspace)
        build_run = Path(step.document["run_directory"])
        states[name] = read_build_state(build_run, workspace=workspace)
        report["builds"][name] = build_record(states[name], build_run, workspace)
    report.update(_compare(states, report, args))


def _compare(
    states: dict[str, BuildState], report: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Apply the pre-registered criteria: H1, H2 per metric, scenarios 2 and 3."""
    w1, w2, chunked = states["W1"], states["W2"], states["C"]
    layer_two: dict[str, Any] = {}
    metrics = {
        "M_claim": lambda a, b: claim_agreement(a.layer_two.claims, b.layer_two.claims),
        "M_state": lambda a, b: paired_agreement(
            a.layer_two.entity_states, b.layer_two.entity_states
        ),
        "M_label": lambda a, b: paired_agreement(
            a.layer_two.entity_labels, b.layer_two.entity_labels
        ),
    }
    for metric, terms_of in metrics.items():
        pairs = {
            "W1_C": terms_of(w1, chunked),
            "W2_C": terms_of(w2, chunked),
            "W1_W2": terms_of(w1, w2),
        }
        effect = partition_effect(
            pairs["W1_C"],
            pairs["W2_C"],
            pairs["W1_W2"],
            resamples=args.resamples,
            seed=args.seed,
            confidence=args.confidence,
        )
        layer_two[metric] = {
            "pairs": {
                name: {
                    "value": terms.value,
                    "agreeing": sum(terms.agreeing),
                    "compared": sum(terms.compared),
                    "clusters": len(terms.clusters),
                }
                for name, terms in pairs.items()
            },
            "effect": encode_incremental_context(effect),
            "decision": decide_equivalence(effect, delta=args.delta).value,
        }
    baseline = report["steps"]["build-C"]["stages"]
    changed = report["steps"]["build-C_policy"]["stages"]
    downstream = judge_downstream_only(
        baseline=baseline,
        changed=changed,
        baseline_build=chunked.context_build_id,
        changed_build=states["C_policy"].context_build_id,
    )
    return {
        "layer_one": encode_incremental_context(
            judge_layer_one(w1=w1.layer_one, w2=w2.layer_one, chunked=chunked.layer_one)
        ),
        "layer_two": layer_two,
        "scenario_2": encode_incremental_context(
            judge_repeated_inference(w1=w1.counts, w2=w2.counts, both=states["R"].counts)
        ),
        "scenario_3": {
            **encode_incremental_context(downstream),
            "override": args.scenario3_set,
            "outputs_differ": states["C_policy"].layer_one != chunked.layer_one,
        },
    }


def _protocol(
    args: argparse.Namespace,
    run_id: str,
    configs: tuple[Path, ...],
    overrides: tuple[str, ...],
    foundation: Any,
    partition: Any,
    code: dict[str, Any],
) -> dict[str, Any]:
    """Return the record frozen before any step runs: what the protocol fixes for this run."""
    return {
        "schema": SCHEMA,
        "issue": 501,
        "run": run_id,
        "code": code,
        **configuration_record(configs, overrides),
        "foundation": foundation.to_document(),
        "selection": {
            "first_image": args.first_image,
            "images": args.images,
            "chunk_count": args.chunks,
            "whole": explicit(partition.whole),
            "rerun": partition.rerun,
            "chunks": [list(chunk) for chunk in partition.chunks],
        },
        "bootstrap": {
            "method": "paired cluster bootstrap, percentile interval",
            "generator": "numpy.random.default_rng",
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence": args.confidence,
        },
        "delta": args.delta,
        "scenario_3_override": args.scenario3_set,
    }


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
        "-c", "--config", type=Path, action="append", required=True, help="configuration file"
    )
    parser.add_argument("--set", action="append", metavar="PATH=VALUE", help="override")
    parser.add_argument(
        "--first-image", type=int, required=True, help="s: index of the first image selected"
    )
    parser.add_argument("--images", type=int, default=100, help="consecutive images selected")
    parser.add_argument("--chunks", type=int, default=4, help="ContextRuns of C")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--scenario3-set", default=DEFAULT_SCENARIO3, metavar="PATH=VALUE")
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
