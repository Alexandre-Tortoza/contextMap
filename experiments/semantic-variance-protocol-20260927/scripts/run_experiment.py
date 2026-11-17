"""Run the #580 variance protocol end to end on the GPU machine: arms A and B, then the report.

    A   the canonical Visual Perception configuration, as is, repeated n >= 3 times
    B   A plus the determinism flags (fixed seeds, torch.use_deterministic_algorithms(True),
        CUBLAS_WORKSPACE_CONFIG=:4096:8), repeated n >= 3 times

Steps, each refusing to go on after a failure:

1. preflight, without loading any model: the configuration resolves, selects scenario 1.0.4's
   backends, validates every parameter through the capability's own configuration, and every
   declared ``resources.providers`` target imports; the source sequence is the artifact the
   frozen selection is pinned to;
2. the frozen 20 frames of scenario 1.0.4 are copied, unchanged, into a derived
   ``SequenceArtifact`` inside the experiment's workspace (the stage processes every image of its
   input, so its input is exactly the selection);
3. repetitions alternate between the arms (A1, B1, A2, B2, ...), each as its own measured
   process (``repetition.py``), so a drift of the machine over time does not land on one arm;
4. the report pairs every two repetitions of each arm (``semantic_variance.py``).

Everything is retained under ``outputs/semantic-variance-protocol-20260927/run-NNNN/`` of this
checkout (``DriverRun``); an existing run directory is never reused or overwritten.

Usage:
    python run_experiment.py --source-sequence DIR --parameters FILE
        [--repetitions 3] [--run-number 1] [--outputs-root DIR] [--seed 0] [--preflight-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import repetition as repetition_script
import semantic_variance

from contextmap.evaluation import (
    DriverRun,
    canonical_real_scenario,
    measure_child_process,
    scenario_runtime_document,
)
from contextmap.ingestion import (
    ExplicitIdsSelection,
    ImageObservation,
    SequenceArtifactId,
    SequenceArtifactManifest,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SequenceProvenance,
    SequenceSelectionError,
    SourceObservationId,
    compute_configuration_hash,
    current_code_version,
    resolve_selection,
)
from contextmap.runtime import (
    ConfigurationError,
    ProviderConfigurationError,
    compose_executors,
    resolve_effective_config,
    resolve_provider,
)
from contextmap.runtime.executors import inventory_digest

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
REPETITION = HERE / "repetition.py"
EXPERIMENT = semantic_variance.EXPERIMENT
SCENARIO_SNAPSHOT = (
    REPOSITORY
    / "src"
    / "contextmap"
    / "evaluation"
    / "docs"
    / "scenarios"
    / "solution-1-canonical-1.0.4-corridor-02-sample.json"
)
"""The frozen scenario the issue names: its selection and its backends are this experiment's."""

ARMS = repetition_script.ARMS
STAGE = repetition_script.STAGE
REGION_STAGE = repetition_script.REGION_STAGE
SNAPSHOT_STAGE = "visual_perception"
"""The stage the 1.0.4 snapshot recorded every perception component under."""
RENAMED_COMPONENTS = {
    "visual_perception.region_discovery": "region_perception.discovery",
    "visual_perception.dense_features": "visual_features.dense",
    "visual_perception.region_features": "visual_features.region",
    "visual_perception.semantic_interpretation": "semantic_interpretation.interpreter",
}
"""Como cada componente do snapshot 1.0.4 se chama na configuração de hoje.

A extração da capacidade de regiões renomeou os pontos de variação da percepção e tirou a
descoberta para o estágio `region_perception`. Os backends selecionados são os mesmos, então o
experimento continua pinado na 1.0.4: o preflight compara os mesmos backends sob os nomes novos,
em vez de aceitar os nomes atuais e deixar de conferir a seleção congelada.
"""
MIN_REPETITIONS = 3
SEQUENCE_LOCATION = "inputs/sequence"
"""Where the derived sequence lives, relative to the experiment's workspace."""


class HarnessError(Exception):
    """Raised when the experiment cannot be set up as the protocol declares it."""


@dataclass(frozen=True, kw_only=True)
class FrozenSelection:
    """The frames a frozen scenario selected, and what pins them.

    Attributes:
        scenario_version: Version of the scenario the selection comes from.
        scenario_digest: Digest of that scenario snapshot.
        sequence_artifact_id: The only sequence artifact the frames may be read from.
        clock_id: Clock domain of the frozen timestamps.
        selection_identity: The scenario's identity of its source selection.
        frames: ``(observation_id, timestamp_ns)`` of every selected image, in scenario order.
    """

    scenario_version: str
    scenario_digest: str
    sequence_artifact_id: str
    clock_id: str
    selection_identity: str
    frames: tuple[tuple[str, int], ...]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible form recorded with the derived sequence and the report."""
        return {
            "scenario_version": self.scenario_version,
            "scenario_digest": self.scenario_digest,
            "sequence_artifact_id": self.sequence_artifact_id,
            "clock_id": self.clock_id,
            "selection_identity": self.selection_identity,
            "frames": [[frame, timestamp_ns] for frame, timestamp_ns in self.frames],
        }


def _snapshot(snapshot: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(snapshot.read_text(encoding="utf-8"))
    return document


def frozen_selection(snapshot: Path = SCENARIO_SNAPSHOT) -> FrozenSelection:
    """Read the selected images of the frozen scenario snapshot."""
    document = _snapshot(snapshot)
    subject = document["subject"]
    selection = subject["selection"]
    return FrozenSelection(
        scenario_version=document["version"],
        scenario_digest=document["digest"],
        sequence_artifact_id=subject["sequence_artifact_id"],
        clock_id=selection["clock_id"],
        selection_identity=selection["selection_identity"],
        frames=tuple(
            (image["observation_id"], image["timestamp_ns"])
            for image in selection["selected_images"]
        ),
    )


def scenario_backends(snapshot: Path = SCENARIO_SNAPSHOT) -> dict[str, str]:
    """Return the backend the frozen scenario selects for each perception component.

    The component ids are the current ones (:data:`RENAMED_COMPONENTS`); the backends are the
    frozen scenario's.
    """
    (stage,) = (
        item for item in _snapshot(snapshot)["stages"] if item["stage_id"] == SNAPSHOT_STAGE
    )
    return {
        RENAMED_COMPONENTS.get(component["component_id"], component["component_id"]): component[
            "backend"
        ]
        for component in stage["components"]
    }


def scenario_document() -> dict[str, Any]:
    """Return the runtime document that selects the canonical scenario's backends.

    It comes from the current scenario, which kept 1.0.4's stages and backends; the preflight
    checks the resolved selection against the 1.0.4 snapshot itself.
    """
    return scenario_runtime_document(canonical_real_scenario())


def derive_frozen_sequence(
    source: Path, output: Path, selection: FrozenSelection
) -> SequenceArtifactManifest:
    """Copy the frozen frames, unchanged, into a new sequence artifact with explicit lineage.

    The identity of the new artifact is derived from the source artifact and the frames only,
    so deriving twice gives the same identity. Calibration is carried over; the provenance names
    the source artifact, the selection and the frozen scenario it comes from.

    Args:
        source: The sequence artifact the frozen selection is pinned to.
        output: The directory of the derived artifact; it must not exist.
        selection: The frozen selection.

    Returns:
        The manifest of the derived artifact.

    Raises:
        HarnessError: If ``output`` exists, ``source`` is another artifact or does not verify,
            a frame is missing or is not an image, or a frame is at another instant.
    """
    if output.exists():
        raise HarnessError(f"{output} already exists: a derived sequence is never overwritten")
    reader = SequenceArtifactReader(source)
    parent = reader.manifest
    if str(parent.artifact_id) != selection.sequence_artifact_id:
        raise HarnessError(
            f"{source} is sequence artifact {str(parent.artifact_id)!r}; scenario "
            f"{selection.scenario_version} froze its frames in {selection.sequence_artifact_id!r}"
        )
    problems = reader.verify_integrity()
    if problems:
        raise HarnessError(f"{source} fails its integrity check: {problems}")
    frames = dict(selection.frames)
    explicit = ExplicitIdsSelection(
        observation_ids=frozenset(SourceObservationId(frame) for frame in frames)
    )
    try:
        resolved = resolve_selection(reader, explicit)
    except SequenceSelectionError as error:
        raise HarnessError(f"{source} does not hold every frozen frame: {error}") from error
    configuration = {
        "parent_sequence_artifact_id": str(parent.artifact_id),
        "selection_id": resolved.selection_id,
        "observation_ids": sorted(frames),
        "frozen_selection": selection.to_record(),
    }
    parent_provenance = reader.read_provenance()
    with SequenceArtifactWriter(
        output_dir=output,
        sequence_name=parent.sequence_name,
        artifact_id=SequenceArtifactId(
            hashlib.sha256(resolved.selection_id.encode("utf-8")).hexdigest()[:32]
        ),
    ) as writer:
        writer.set_provenance(
            SequenceProvenance(
                source_type="sequence_artifact",
                source_path=str(parent.artifact_id),
                source_content_hash=inventory_digest(parent.file_inventory),
                ingestion_config=configuration,
                configuration_hash=compute_configuration_hash(configuration),
                adapter_type=None,
                code_version=current_code_version(),
                calibration_source_hash=(
                    None if parent_provenance is None else parent_provenance.calibration_source_hash
                ),
                warnings=(
                    "image subset of an existing sequence artifact: every observation is copied "
                    "unchanged, none is re-ingested",
                ),
            )
        )
        calibration = reader.read_calibration()
        if calibration is not None:
            writer.set_calibration(calibration)
        for observation in resolved.observations:
            frame = str(observation.observation_id)
            if not isinstance(observation, ImageObservation):
                raise HarnessError(f"frozen frame {frame!r} is not an image observation")
            timestamp = observation.timestamp
            if (
                timestamp.clock_id != selection.clock_id
                or timestamp.total_nanoseconds() != frames[frame]
            ):
                raise HarnessError(
                    f"frozen frame {frame!r} has timestamp {timestamp.total_nanoseconds()} "
                    f"({timestamp.clock_id}); scenario {selection.scenario_version} froze "
                    f"{frames[frame]} ({selection.clock_id})"
                )
            writer.add_observation(observation)
        return writer.finalize()


def arm_command(arm: str, command: Sequence[str]) -> list[str]:
    """Prefix a repetition's command with its arm's environment.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA starts, so it goes on the command line;
    arm A removes it, in case the shell exported it.

    Raises:
        HarnessError: If the arm is not one of the protocol's.
    """
    variable = repetition_script.CUBLAS_VARIABLE
    if arm == repetition_script.DETERMINISTIC_ARM:
        return ["env", f"{variable}={repetition_script.CUBLAS_WORKSPACE_CONFIG}", *command]
    if arm in ARMS:
        return ["env", "-u", variable, *command]
    raise HarnessError(f"unknown arm {arm!r}: the protocol has arms {ARMS}")


def schedule(arms: Sequence[str], repetitions: int) -> list[tuple[str, int]]:
    """Alternate the arms repetition by repetition: A1, B1, A2, B2, ..."""
    return [(arm, index) for index in range(1, repetitions + 1) for arm in arms]


def _unloaded_runtime(config: object, secrets: object) -> object:
    """Stand in for a model runtime during preflight: composing must not load any model."""
    return object()


def preflight(
    config_files: Sequence[Path],
    *,
    dataset: str,
    module_available: Callable[[str], bool] | None = None,
) -> list[str]:
    """Check everything that can fail before a model is loaded, and say what did.

    The stage is composed exactly as the runtime composes it, except that every declared
    provider is resolved (its module imported) but not called: a placeholder stands in for the
    model runtime it would load.

    Args:
        config_files: The scenario's backend selection and the maintainer's parameter layer.
        dataset: ``inputs.sequence``.
        module_available: Predicate telling whether an optional module is installed.

    Returns:
        Every problem found; empty when the repetitions can start.
    """
    try:
        effective = resolve_effective_config(
            files=config_files, overrides=[f"inputs.sequence={dataset}"]
        )
    except ConfigurationError as error:
        return [f"configuration: {problem}" for problem in error.problems]
    problems: list[str] = []
    components = effective.config.components
    expected = scenario_backends()
    for component_id, backend in expected.items():
        selected = components[component_id].backend if component_id in components else None
        if selected != backend:
            problems.append(
                f"{component_id}: scenario 1.0.4 selects {backend!r}, the configuration "
                f"selects {selected!r}"
            )
    # Os providers são declarados por componente: a percepção ocupa dois estágios, então o
    # filtro é o próprio conjunto de componentes que o cenário congelado seleciona.
    declared = {
        component_id: target
        for component_id, target in effective.config.resources.providers.items()
        if component_id in expected
    }
    for component_id, target in sorted(declared.items()):
        try:
            resolve_provider(component_id, target)
        except ProviderConfigurationError as error:
            problems.append(str(error))
    failures: dict[str, Exception] = {}
    executors = compose_executors(
        effective,
        providers=dict.fromkeys(declared, _unloaded_runtime),
        module_available=module_available,
        on_composition_failure=failures.__setitem__,
    )
    # A percepção ocupa dois estágios executados, compostos de forma independente: a repetição
    # roda a descoberta e o run medido, então os dois precisam compor antes de pedir a GPU.
    problems.extend(
        f"{stage_id} cannot be composed: {failures.get(stage_id)}"
        for stage_id in (REGION_STAGE, STAGE)
        if stage_id not in executors
    )
    return problems


def _commit() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=REPOSITORY, capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=1, sort_keys=True) + "\n")


def _write_config(directory: Path, parameters: Path) -> list[Path]:
    """Write the two configuration layers every repetition resolves, in precedence order."""
    directory.mkdir(parents=True, exist_ok=True)
    scenario = directory / "scenario-runtime.json"
    scenario.write_text(json.dumps(scenario_document(), indent=1) + "\n", encoding="utf-8")
    layer = directory / f"parameters{parameters.suffix}"
    layer.write_bytes(parameters.read_bytes())
    return [scenario, layer]


def _repetition(
    *,
    arm: str,
    index: int,
    root: Path,
    workspace: Path,
    config_files: Sequence[Path],
    dataset: str,
    code_identity: str,
    seed: int,
) -> tuple[dict[str, Any] | None, int]:
    """Run one repetition as its own measured process; return its record and exit status."""
    label = f"{arm}-{index:02d}"
    record_path = root / "repetitions" / f"{label}.json"
    command = [
        sys.executable,
        str(REPETITION),
        "--arm",
        arm,
        "--repetition",
        str(index),
        "--workspace",
        str(workspace),
        *(argument for path in config_files for argument in ("--config", str(path))),
        "--sequence-location",
        SEQUENCE_LOCATION,
        "--dataset",
        dataset,
        "--code-identity",
        code_identity,
        "--seed",
        str(seed),
        "--record",
        str(record_path),
    ]
    print(f"--- {label}", flush=True)
    measurement = measure_child_process(arm_command(arm, command))
    # O log é diagnóstico humano: nunca é lido por nada a jusante.
    log = root / "repetitions" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(measurement.stdout + "\n--- stderr\n" + measurement.stderr, encoding="utf-8")
    print(f"    exit {measurement.returncode} in {measurement.wall_time_seconds:.0f} s", flush=True)
    if not record_path.is_file():
        return None, measurement.returncode
    record: dict[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
    record["label"] = label
    record["measurement"] = measurement.to_record()
    return record, measurement.returncode


def repetition_failures(label: str, record: Mapping[str, Any] | None, returncode: int) -> list[str]:
    """Say why a repetition is not evidence; empty when it is.

    A repetition counts only when its process exited cleanly, left a record, and the record says
    the runtime completed the stage and names the perception artifact it produced.
    """
    if record is None:
        return [f"{label}: exited with status {returncode} and no record"]
    if (
        returncode != 0
        or record.get("status") != "completed"
        or record.get("semantic_interpretation") is None
    ):
        return [
            f"{label}: run {record.get('run')} {record.get('status')} (exit status "
            f"{returncode}): {record.get('failure')} {record.get('blocked_problems')}"
        ]
    return []


def _analysis(
    records: Sequence[Mapping[str, Any]], *, workspace: Path, workspace_location: str
) -> dict[str, Any]:
    """Pair the repetitions of each arm; locations are recorded relative to the outputs root."""
    arms: dict[str, list[semantic_variance.RunContent]] = {}
    for record in records:
        location = record["semantic_interpretation"]["location"]
        arms.setdefault(record["arm"], []).append(
            semantic_variance.load_run(
                record["label"],
                workspace / location,
                region_run_dir=workspace / record["regions"]["location"],
                location=f"{workspace_location}/{location}",
            )
        )
    return semantic_variance.build_report(
        {arm: semantic_variance.compare_arm(arm, runs) for arm, runs in arms.items()}
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the whole protocol; exit non-zero unless every repetition and the report completed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-sequence",
        type=Path,
        required=True,
        help="the SequenceArtifact scenario 1.0.4 pinned (720a486de8d44c16a9d3d2ff9fa7b1a4)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        required=True,
        help="runtime configuration layer with the canonical Visual Perception parameters and "
        "its resources.providers targets",
    )
    parser.add_argument("--repetitions", type=int, default=MIN_REPETITIONS)
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args(argv)
    if options.repetitions < MIN_REPETITIONS:
        parser.error(f"the protocol needs at least {MIN_REPETITIONS} repetitions per arm")

    run = DriverRun.for_driver(
        Path(__file__),
        experiment=EXPERIMENT,
        run_number=options.run_number,
        outputs_root=options.outputs_root,
    )
    root = run.outputs_root / run.experiment / run.run_id
    if root.exists():
        raise HarnessError(f"{root} already exists: a run of the experiment is never reused")
    selection = frozen_selection()
    source = SequenceArtifactReader(options.source_sequence).manifest
    dataset = source.sequence_name
    with tempfile.TemporaryDirectory() as scratch:
        problems = preflight(_write_config(Path(scratch), options.parameters), dataset=dataset)
    if str(source.artifact_id) != selection.sequence_artifact_id:
        problems.append(
            f"--source-sequence is {str(source.artifact_id)!r}; scenario "
            f"{selection.scenario_version} froze its frames in {selection.sequence_artifact_id!r}"
        )
    if problems:
        print("preflight failed:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print("preflight: ok", flush=True)
    if options.preflight_only:
        return 0

    workspace = root / "workspace"
    config_files = _write_config(root / "config", options.parameters)
    code = _commit()
    code_identity = code["commit"] + ("-dirty" if code["dirty"] else "")
    try:
        derived = derive_frozen_sequence(
            options.source_sequence, workspace / SEQUENCE_LOCATION, selection
        )
    except HarnessError as error:
        failed = {"experiment": EXPERIMENT, "code": code, "status": "failed"}
        _write_json(root / "experiment.json", {**failed, "failures": [f"sequence: {error}"]})
        print(f"FAILED: sequence: {error}", file=sys.stderr)
        return 1
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for arm, index in schedule(ARMS, options.repetitions):
        record, returncode = _repetition(
            arm=arm,
            index=index,
            root=root,
            workspace=workspace,
            config_files=config_files,
            dataset=dataset,
            code_identity=code_identity,
            seed=options.seed,
        )
        if record is not None:
            records.append(record)
        failures.extend(repetition_failures(f"{arm}-{index:02d}", record, returncode))
        if failures:
            # Fail-fast: nenhuma repetição roda depois de uma inválida, e o registro diz isso.
            break

    workspace_location = f"{run.experiment}/{run.run_id}/workspace"
    provenance = {
        "experiment": EXPERIMENT,
        "issue": semantic_variance.ISSUE,
        "code": code,
        "selection": selection.to_record(),
        "sequence": {
            "source_artifact_id": str(source.artifact_id),
            "artifact_id": str(derived.artifact_id),
            "content_hash": inventory_digest(derived.file_inventory),
            "location": f"{workspace_location}/{SEQUENCE_LOCATION}",
        },
        "configuration": {
            "scenario_runtime_sha256": _sha256(config_files[0]),
            "parameters_sha256": _sha256(config_files[1]),
        },
        "seed": options.seed,
        "repetitions": records,
    }
    if not failures:
        try:
            report = _analysis(records, workspace=workspace, workspace_location=workspace_location)
        except semantic_variance.VarianceProtocolError as error:
            failures.append(f"analysis: {error}")
        else:
            report["provenance"] = provenance
            semantic_variance.write_report(report, root / "report")
    status = "failed" if failures else "complete"
    _write_json(root / "experiment.json", {**provenance, "status": status, "failures": failures})
    print(f"experiment: {root / 'experiment.json'} ({status})")
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
