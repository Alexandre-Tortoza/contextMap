"""Execução de aceitação do workflow incremental pela CLI.

Verifica lineage, reuso e imutabilidade sem modificar a fundação nem artifacts publicados."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Os passos usam a mesma CLI e composição que o harness de invariância de partição.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "incremental-partition-invariance-20260927"
        / "scripts"
    ),
)

from steps import (
    ContextCli,
    StepFailed,
    build_record,
    configuration_record,
    explicit,
    git_identity,
    image_positions,
    relative,
    resolve_code_identity,
    step_record,
    unexpected_failure,
    write_once,
)

from contextmap.artifact import ContextMapArtifactReader
from contextmap.evaluation import (
    BuildState,
    DriverRun,
    ExperimentDriverError,
    read_build_state,
)
from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.runtime import (
    ConfigurationError,
    SpatialFoundationError,
    capture_environment,
    foundation_of_run,
    read_context_build,
    read_context_run,
    read_run,
    resolve_effective_config,
)
from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.shared import hash_file

EXPERIMENT = "incremental-context-workflow-20260927"
SCHEMA = "contextmap.experiment.incremental-context-workflow/v1"
FOUNDATION_STAGES = ("geometric_mapping", "state_estimation")
"""The stages a spatial foundation fixes, which no increment may rerun."""
CHECKOUT = Path(__file__).resolve().parents[3]


class PhaseFailed(Exception):
    """A phase could not run; the workflow stops there."""

    def __init__(self, phase: str, failure: StepFailed) -> None:
        """Keep the phase and the step failure that stopped it."""
        super().__init__(f"phase {phase}: {failure}")
        self.phase = phase
        self.failure = failure


def main(argv: list[str] | None = None) -> int:
    """Run the workflow; 0 only when complete and accepted, 1 otherwise, 2 when refused."""
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    foundation_run = args.foundation_run.resolve()
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
        foundation = foundation_of_run(workspace, foundation_run)
        assert foundation.sequence.location is not None
        s1, s2 = consecutive_windows(
            image_positions(workspace / foundation.sequence.location),
            first=args.first_image,
            sizes=(args.s1_images, args.s2_images),
        )
        effective = resolve_effective_config(files=list(configs), overrides=list(overrides))
        configuration = configuration_record(configs, overrides)
        code = git_identity(CHECKOUT)
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
    manifest = {
        "schema": SCHEMA,
        "issue": 504,
        "run": run.run_id,
        "evidence_class": evidence_class,
        "code": code,
        "code_identity": code_identity,
        "dataset": effective.config.inputs.sequence,
        "sequence": foundation.sequence.to_document(),
        "foundation": foundation.to_document(),
        "foundation_run": relative(foundation_run, workspace),
        "selections": {"first_image": args.first_image, "S1": explicit(s1), "S2": explicit(s2)},
        **configuration,
        "environment": {**capture_environment(effective.config), "gpus": _gpus()},
    }
    write_once(harness / "workflow-manifest.json", manifest)

    cli = ContextCli(
        workspace=workspace,
        configs=configs,
        overrides=overrides,
        steps_dir=harness / "steps",
        code_identity=code_identity,
        dry_run_executors=args.dry_run_executors,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": evidence_class,
        "manifest_sha256": hashlib.sha256(
            (harness / "workflow-manifest.json").read_bytes()
        ).hexdigest(),
        "phases": {},
        "invariants": {},
        "cost_avoided": None,
        "status": "running",
        "accepted": False,
        "failure": None,
    }
    try:
        _run_phases(cli, workspace, foundation_run, f"workflow-{run.run_id}", (s1, s2), report)
        report["status"] = "complete"
        report["accepted"] = all(report["invariants"].values())
    except PhaseFailed as stopped:
        report["status"] = "failed"
        report["failure"] = {"phase": stopped.phase, **stopped.failure.to_document()}
    except Exception as error:
        # Um erro fora de um comando (ao ler ou conferir) também deixa o relatório, e sobe.
        report.update(status="failed", failure=unexpected_failure(error))
        write_once(harness / "report.json", report)
        raise
    write_once(harness / "report.json", report)
    print(f"{report['status']}, accepted={report['accepted']}: {harness / 'report.json'}")
    return 0 if report["status"] == "complete" and report["accepted"] else 1


def _run_phases(
    cli: ContextCli,
    workspace: Path,
    foundation_run: Path,
    branch: str,
    selections: tuple[Sequence[str], Sequence[str]],
    report: dict[str, Any],
) -> None:
    """Run phases A to F in order, filling ``report`` as each one completes."""
    phases = report["phases"]
    s1, s2 = selections

    # A: a fundação é selecionada, nunca reconstruída; o custo evitado vem do próprio run dela.
    step = _phase("A", lambda: cli.branch(branch, foundation_run=foundation_run))
    foundation = step.document["foundation"]
    with GeometricMapArtifactReader(workspace / foundation["geometry"]["location"]) as geometry:
        map_id = str(geometry.manifest.map_id)
    phases["A"] = {
        "step": step.record,
        "branch": branch,
        "foundation_id": foundation["identity"],
        "map_id": map_id,
        **_foundation_cost(foundation_run),
    }
    report["cost_avoided"] = phases["A"].pop("cost_avoided")

    # B: CR1 sobre S1.
    step = _phase("B", lambda: cli.context_run("B-CR1", branch=branch, selection=explicit(s1)))
    cr1_dir = Path(step.document["run_directory"])
    phases["B"] = _context_phase(step, workspace)

    # C: CB1 sobre [CR1], materializando M1; o estado de CR1, CB1 e M1 é fotografado aqui.
    step = _phase("C", lambda: cli.build("C-CB1", branch=branch, revision=1))
    cb1_dir = Path(step.document["run_directory"])
    m1 = read_build_state(cb1_dir, workspace=workspace)
    phases["C"] = _build_phase(step, m1, cb1_dir, workspace)
    before = {"CR1": snapshot(cr1_dir), "CB1": snapshot(cb1_dir)}
    records = {
        "CR1": read_context_run(cr1_dir, workspace=workspace).to_document(),
        "CB1": read_context_build(cb1_dir, workspace=workspace).to_document(),
    }
    # O mapa é publicado no maps root, fora do run do build; a CLI informa onde.
    m1_map = Path(step.document["map"]["location"])
    m1_identity = _content_identity(m1_map)

    # D: CR2 sobre S2, depois e disjunta de S1, contra a mesma fundação.
    step = _phase("D", lambda: cli.context_run("D-CR2", branch=branch, selection=explicit(s2)))
    cr2_dir = Path(step.document["run_directory"])
    phases["D"] = _context_phase(step, workspace)

    # E: CB2 sobre [CR1, CR2], materializando M2.
    step = _phase("E", lambda: cli.build("E-CB2", branch=branch, revision=2))
    cb2_dir = Path(step.document["run_directory"])
    m2 = read_build_state(cb2_dir, workspace=workspace)
    phases["E"] = _build_phase(step, m2, cb2_dir, workspace)

    # F: CR1, CB1 e M1 relidos depois de M2 existir.
    start = time.perf_counter()
    changed = [
        f"{name}:{path}"
        for name, directory in (("CR1", cr1_dir), ("CB1", cb1_dir))
        for path in changed_files(before[name], snapshot(directory))
    ]
    reread = {
        "CR1": read_context_run(cr1_dir, workspace=workspace).to_document(),
        "CB1": read_context_build(cb1_dir, workspace=workspace).to_document(),
    }
    m1_again = read_build_state(cb1_dir, workspace=workspace)
    phases["F"] = {
        "wall_time_seconds": round(time.perf_counter() - start, 3),
        "changed_files": changed,
        "records_equal": reread == records,
        "m1_validation": m1_again.validation.status,
    }
    cr1 = read_context_run(cr1_dir, workspace=workspace)
    cr2 = read_context_run(cr2_dir, workspace=workspace)
    report["invariants"] = _invariants(
        report,
        cr=(cr1, cr2),
        builds=(m1, m2, m1_again),
        cb_dirs=(cb1_dir, cb2_dir),
        m1_map=m1_map,
        m1_identity=m1_identity,
        selections=(s1, s2),
        changed=changed,
    )


def _invariants(
    report: dict[str, Any],
    *,
    cr: tuple[Any, Any],
    builds: tuple[BuildState, BuildState, BuildState],
    cb_dirs: tuple[Path, Path],
    m1_map: Path,
    m1_identity: str,
    selections: tuple[Sequence[str], Sequence[str]],
    changed: list[str],
) -> dict[str, bool]:
    """Check each workflow invariant independently."""
    phases = report["phases"]
    cr1, cr2 = cr
    m1, m2, m1_again = builds
    cb1_dir, cb2_dir = cb_dirs
    cb1_foundation = phases["C"]["build"]["foundation_id"]
    cb2_foundation = phases["E"]["build"]["foundation_id"]
    frames = {frame for _, frame, _ in m2.layer_one.contributions}
    selected = set(selections[0]) | set(selections[1])
    lineage = SemanticFusionRunReader(
        cb2_dir.parent.parent / "artifacts" / "semantic_fusion" / cb2_dir.name
    ).manifest.lineage
    perception = {
        stage: {
            artifact.ref.artifact_id
            for run in (cr1, cr2)
            for artifact in run.artifacts
            if artifact.ref.stage_id == stage
        }
        for stage in ("region_perception", "visual_perception", "semantic_interpretation")
    }
    association = {
        artifact.ref.artifact_id
        for run in (cr1, cr2)
        for artifact in run.artifacts
        if artifact.ref.stage_id == "sensor_association"
    }
    sf1 = phases["A"]["foundation_id"]
    return {
        "same_foundation": {
            cr1.foundation.identity,
            cr2.foundation.identity,
            cb1_foundation,
            cb2_foundation,
        }
        == {sf1},
        # Nenhuma troca escondida do mapa: M1 e M2 citam o mapa da fundação selecionada.
        "same_map_id": m1.layer_one.map_id == m2.layer_one.map_id == phases["A"]["map_id"],
        # Uma ContextRun recebe a fundação inteira; um build só consome dela o mapa e a
        # sequência. Em nenhuma fase um estágio da fundação é executado.
        "foundation_not_recomputed": all(
            phases[phase]["step"]["stages"].get(stage) == "provided"
            for phase in ("B", "D")
            for stage in FOUNDATION_STAGES
        )
        and all(
            phases[phase]["step"]["stages"].get(stage, "provided") == "provided"
            for phase in ("C", "E")
            for stage in FOUNDATION_STAGES
        ),
        "cr1_immutable": not [item for item in changed if item.startswith("CR1:")]
        and phases["F"]["records_equal"],
        "cb1_immutable": not [item for item in changed if item.startswith("CB1:")]
        and phases["F"]["records_equal"],
        "m1_unchanged_and_valid": m1_again.validation.status == "verified"
        and m1_again.context_map_id == m1.context_map_id
        and _content_identity(m1_map) == m1_identity,
        "cb2_consumes_cr1_and_cr2": set(m2.context_run_ids) == {cr1.identity, cr2.identity},
        "fusion_counts_explained": frames <= selected
        and m2.counts.physical_observations <= len(selected)
        and m2.counts.inference_results == m2.counts.physical_observations,
        "m2_lineage_closes": set(lineage.region_artifact_ids)
        == perception["region_perception"]
        and set(lineage.visual_artifact_ids) == perception["visual_perception"]
        and set(lineage.semantic_interpretation_artifact_ids)
        == perception["semantic_interpretation"]
        and set(lineage.association_run_ids) == association,
        "m2_validates": m2.validation.status == "verified" and not m2.integrity_problems,
    }


def snapshot(directory: Path) -> dict[str, str]:
    """Return the SHA-256 of every file under a directory, by path relative to it."""
    return {
        path.relative_to(directory).as_posix(): hash_file(path)[1]
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Return every path added, removed or changed between two snapshots, sorted."""
    return sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )


def consecutive_windows(
    images: Sequence[tuple[int, str]], *, first: int, sizes: Sequence[int]
) -> tuple[tuple[str, ...], ...]:
    """Split consecutive images from ``first`` into disjoint windows of the given sizes.

    Raises:
        ValueError: If a size is not positive or the windows do not fit the sequence.
    """
    if first < 0 or any(size < 1 for size in sizes):
        raise ValueError(f"the first image and every size must be positive, got {first}, {sizes}")
    if first + sum(sizes) > len(images):
        raise ValueError(
            f"windows of {list(sizes)} images from {first} do not fit {len(images)} images"
        )
    ids = [observation_id for _, observation_id in images]
    windows: list[tuple[str, ...]] = []
    for size in sizes:
        windows.append(tuple(ids[first : first + size]))
        first += size
    return tuple(windows)


def _phase(name: str, step: Any) -> Any:
    """Run one phase's step, turning its failure into the phase's."""
    try:
        return step()
    except StepFailed as failure:
        raise PhaseFailed(name, failure) from None


def _context_phase(step: Any, workspace: Path) -> dict[str, Any]:
    """Record a context-run phase: its cost, reuse decisions and the ContextRun it appended."""
    record = step_record(step, workspace)
    context_run = step.document["context_run"]
    return {
        "step": record,
        "context_run_id": context_run["context_run_id"],
        "artifacts": context_run["artifacts"],
        "model_load_sets": int(record["stages"].get("visual_perception") == "executed"),
    }


def _build_phase(step: Any, state: BuildState, build_run: Path, workspace: Path) -> dict[str, Any]:
    """Record a build phase: its cost, reuse decisions and the materialized ContextMap."""
    record = step_record(step, workspace)
    build = read_context_build(build_run, workspace=workspace)
    return {
        "step": record,
        "build": {
            **build_record(state, build_run, workspace),
            "foundation_id": build.foundation.identity,
        },
        "model_load_sets": int(record["stages"].get("visual_perception") == "executed"),
    }


def _foundation_cost(foundation_run: Path) -> dict[str, Any]:
    """Read what building the foundation cost, from the foundation run's own journal."""
    summary = read_run(foundation_run)
    elapsed = {
        str(event.stage_id): float(event.data["elapsed_s"])
        for event in summary.events
        if event.kind == "stage_completed" and event.stage_id in FOUNDATION_STAGES
    }
    return {
        "foundation_stage_seconds": elapsed,
        "cost_avoided": {
            "foundation_stages": sorted(elapsed),
            "seconds_per_increment": round(sum(elapsed.values()), 3) if elapsed else None,
            "source": "elapsed_s of the foundation run's stage_completed events; a stage the "
            "foundation run was given instead of executing has no cost recorded here",
        },
    }


def _content_identity(context_map: Path) -> str:
    """Return the content identity a ContextMapArtifact's manifest records."""
    with ContextMapArtifactReader.open(context_map) as reader:
        return str(reader.manifest.content_identity)


def _gpus() -> list[str] | None:
    """Describe the GPUs of this machine through ``nvidia-smi``, or ``None`` without one."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True, help="the runtime workspace")
    parser.add_argument(
        "--foundation-run",
        type=Path,
        required=True,
        help="phase A: completed run whose sequence, trajectory and map are SF1",
    )
    parser.add_argument(
        "-c", "--config", type=Path, action="append", required=True, help="configuration file"
    )
    parser.add_argument("--set", action="append", metavar="PATH=VALUE", help="override")
    parser.add_argument(
        "--first-image", type=int, required=True, help="index of the first image of S1"
    )
    parser.add_argument("--s1-images", type=int, default=50, help="consecutive images of S1")
    parser.add_argument(
        "--s2-images", type=int, default=50, help="consecutive images of S2, right after S1"
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
