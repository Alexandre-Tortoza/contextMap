"""Tests for the workspace layout.

The journal of a run lives in ``<workspace>/<dataset>/<run-id>/``; the artifact of each stage it
executes lives in ``<workspace>/artifacts/<stage>/<run-id>/``. The run number is unique in the
whole workspace, so runs of different datasets never share an artifact directory. The runtime
only decides *where* a stage writes; the executor's writer creates and finalizes that directory,
so an executor never computes a path of its own.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from runtime_documents import effective_from, selected_document
from runtime_worlds import World

from contextmap.runtime import (
    ArtifactRef,
    EffectiveConfig,
    ExecutionPlan,
    ReusePolicy,
    RunJournal,
    StageRequest,
    resolve_plan,
    run_plan,
)

DATASET = "corridor-02"
STAGES = [
    "ingestion",
    # Cada produtor de evidência recebe seu diretório e identidade próprios.
    "region_perception",
    "visual_perception",
    "semantic_interpretation",
    "state_estimation",
    "geometric_mapping",
    "sensor_association",
    "semantic_fusion",
]
PROVIDED = (
    "region_perception.discovery",
    "visual_features.dense",
    "visual_features.region",
    "semantic_interpretation.interpreter",
)


def _document(sequence: str | None = DATASET) -> dict[str, Any]:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False
    document["inputs"] = {"sequence": sequence}
    return document


def _scope(tmp_path: Path, sequence: str | None = DATASET) -> tuple[EffectiveConfig, ExecutionPlan]:
    effective = effective_from(tmp_path, _document(sequence))
    return effective, resolve_plan(effective).scope(targets=["semantic_fusion"])


class _Recording:
    """An executor that records the directory it was given and writes nothing itself."""

    def __init__(self, world: World, stage_id: str, contract: str, *, source: bool) -> None:
        self._inner = world.executor(stage_id, contract, source=source)
        self.directories: dict[str, Path | None] = {}
        if source:
            # Estágio sem entradas: repassa a identidade da fonte que o executor interno nomeia.
            self.source_identity: str = self._inner.source_identity

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.directories[request.stage_id] = request.output_dir
        ref: ArtifactRef = self._inner.execute(request)
        return ref


def _executors(execution: ExecutionPlan, world: World) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded: dict[str, Path | None] = {}
    executors: dict[str, Any] = {}
    for stage in execution.plan.stages:
        recording = _Recording(world, stage.stage_id, stage.output or "", source=not stage.inputs)
        recording.directories = recorded
        executors[stage.stage_id] = recording
    return executors, recorded


def _run(execution: ExecutionPlan, executors: dict[str, Any], **options: Any) -> None:
    run_plan(
        execution,
        executors,
        environ={},
        module_available=lambda _name: True,
        provided_runtimes=PROVIDED,
        **options,
    )


def test_a_run_lives_under_the_dataset_of_its_configuration(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path)

    first = RunJournal.create(tmp_path / "ws", effective, execution)
    second = RunJournal.create(tmp_path / "ws", effective, execution)

    assert first.directory == tmp_path / "ws" / DATASET / "run-0001"
    assert second.directory == tmp_path / "ws" / DATASET / "run-0002"
    assert (first.directory / "status.json").is_file()  # o diário fica na raiz do run
    assert (first.directory / "effective_config.json").is_file()


def test_runs_of_different_datasets_share_the_workspace_counter_but_not_a_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "other").mkdir()
    effective, execution = _scope(tmp_path)
    other, other_execution = _scope(tmp_path / "other", "corridor-03")

    one = RunJournal.create(tmp_path / "ws", effective, execution)
    two = RunJournal.create(tmp_path / "ws", other, other_execution)
    three = RunJournal.create(tmp_path / "ws", effective, execution)

    assert one.directory == tmp_path / "ws" / DATASET / "run-0001"
    assert two.directory == tmp_path / "ws" / "corridor-03" / "run-0002"
    assert three.directory == tmp_path / "ws" / DATASET / "run-0003"


def test_concurrent_allocations_in_different_datasets_never_share_a_run_number(
    tmp_path: Path,
) -> None:
    datasets = ["corridor-02", "corridor-03", "corridor-04", "corridor-05"]
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # A largada: cada filho só aloca quando o arquivo aparece, para que as varreduras do
    # contador se sobreponham. Sem o lock do workspace, dois datasets veem o mesmo máximo.
    gate = tmp_path / "gate"
    reports: dict[int, Path] = {}
    for index, dataset in enumerate(datasets):
        root = tmp_path / f"root-{index}"
        root.mkdir()
        effective, execution = _scope(root, dataset)
        report = tmp_path / f"allocated-{index}"
        pid = os.fork()
        if pid == 0:  # processo filho: aloca e sai sem passar pelo teardown do pytest
            code = 1
            try:
                while not gate.exists():
                    time.sleep(0.001)
                report.write_text(
                    RunJournal.create(workspace, effective, execution).run_id, encoding="ascii"
                )
                code = 0
            finally:
                os._exit(code)
        reports[pid] = report

    gate.touch()

    for pid, report in reports.items():
        assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0, f"{report} was not allocated"
    allocated = [report.read_text(encoding="ascii") for report in reports.values()]
    assert sorted(allocated) == ["run-0001", "run-0002", "run-0003", "run-0004"]
    assert sorted(path.name for path in workspace.glob("*/run-*")) == sorted(allocated)


def test_runs_of_different_datasets_write_their_stages_to_distinct_artifact_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "other").mkdir()
    effective, execution = _scope(tmp_path)
    other, other_execution = _scope(tmp_path / "other", "corridor-03")
    one = RunJournal.create(tmp_path / "ws", effective, execution)
    two = RunJournal.create(tmp_path / "ws", other, other_execution)
    one_executors, one_recorded = _executors(execution, World())
    two_executors, two_recorded = _executors(other_execution, World())

    _run(execution, one_executors, journal=one)
    _run(other_execution, two_executors, journal=two)

    artifacts = tmp_path / "ws" / "artifacts"
    assert one_recorded == {stage: artifacts / stage / one.run_id for stage in STAGES}
    assert two_recorded == {stage: artifacts / stage / two.run_id for stage in STAGES}


def test_the_artifacts_directory_is_not_a_dataset(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path, "artifacts")

    with pytest.raises(ValueError, match="artifacts"):
        RunJournal.create(tmp_path / "ws", effective, execution)

    assert not (tmp_path / "ws").exists()


def test_a_run_without_a_dataset_is_refused_before_anything_is_created(tmp_path: Path) -> None:
    effective, execution = _scope(tmp_path, None)

    with pytest.raises(ValueError, match=r"inputs\.sequence"):
        RunJournal.create(tmp_path / "ws", effective, execution)

    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("dataset", ["../escape", "a/b", "..", "."])
def test_a_dataset_that_is_not_one_path_component_is_refused(tmp_path: Path, dataset: str) -> None:
    effective, execution = _scope(tmp_path, dataset)

    with pytest.raises(ValueError, match="dataset"):
        RunJournal.create(tmp_path / "ws", effective, execution)

    assert not (tmp_path / "ws").exists()


def test_each_stage_receives_its_own_artifact_directory_and_none_is_precreated(
    tmp_path: Path,
) -> None:
    effective, execution = _scope(tmp_path)
    journal = RunJournal.create(tmp_path / "ws", effective, execution)
    executors, recorded = _executors(execution, World())

    _run(execution, executors, journal=journal)

    artifacts = tmp_path / "ws" / "artifacts"
    assert recorded == {stage: artifacts / stage / journal.run_id for stage in STAGES}
    # Quem cria e finaliza a pasta da etapa é o writer do executor, de forma atômica.
    assert not any((artifacts / stage / journal.run_id).exists() for stage in STAGES)
    assert not any((journal.directory / stage).exists() for stage in STAGES)


def test_a_run_without_a_journal_has_no_output_directory(tmp_path: Path) -> None:
    _, execution = _scope(tmp_path)
    executors, recorded = _executors(execution, World())

    _run(execution, executors)

    assert recorded == dict.fromkeys(STAGES)


def test_a_reused_stage_is_referenced_and_gets_no_request_or_directory(tmp_path: Path) -> None:
    world = World()
    effective, execution = _scope(tmp_path)
    policy = ReusePolicy(store=world.store(tmp_path / "index"), code_identity="code-1")
    first_journal = RunJournal.create(tmp_path / "ws", effective, execution)
    first_executors, _ = _executors(execution, world)
    _run(execution, first_executors, journal=first_journal, reuse=policy)

    second_journal = RunJournal.create(tmp_path / "ws", effective, execution)
    second_executors, recorded = _executors(execution, world)
    _run(execution, second_executors, journal=second_journal, reuse=policy)

    assert recorded == {}  # tudo reutilizado: nenhum executor foi chamado
    artifacts = tmp_path / "ws" / "artifacts"
    assert not any((artifacts / stage / second_journal.run_id).exists() for stage in STAGES)


def test_the_cli_refuses_a_real_run_without_a_dataset_and_creates_nothing(tmp_path: Path) -> None:
    from runtime_worlds import run_cli, world_executors

    config = tmp_path / "experiment.json"
    config.write_text(json.dumps(_document(None)), encoding="utf-8")

    code, _, err = run_cli(
        "run",
        "-c",
        str(config),
        "--stage",
        "ingestion",
        "--workspace",
        str(tmp_path / "ws"),
        executors=world_executors(World()),
    )

    assert code == 2
    assert "inputs.sequence" in err
    assert not (tmp_path / "ws").exists()


def _request(digest: str = "d1", **inputs: str | None) -> StageRequest:
    refs = {
        name: (ArtifactRef(stage_id="s", contract="c", artifact_id=name, content_hash=value),)
        for name, value in inputs.items()
    }
    return StageRequest(
        stage_id="geometric_mapping", inputs=refs, components={}, config_digest=digest
    )


def test_the_identity_of_an_execution_repeats_and_follows_its_configuration_and_inputs() -> None:
    base = _request(sequence="sha256:a", trajectory="sha256:b")

    assert base.identity() == _request(sequence="sha256:a", trajectory="sha256:b").identity()
    assert len(base.identity()) == 32
    assert base.identity() != _request("d2", sequence="sha256:a", trajectory="sha256:b").identity()
    assert base.identity() != _request(sequence="sha256:x", trajectory="sha256:b").identity()


def test_an_input_without_a_content_hash_cannot_take_part_in_an_identity() -> None:
    with pytest.raises(ValueError, match="no content hash"):
        _request(sequence=None).identity()


def test_the_run_number_is_read_from_the_artifact_directory_and_is_zero_otherwise(
    tmp_path: Path,
) -> None:
    numbered = StageRequest(
        stage_id="s",
        inputs={},
        components={},
        config_digest="d",
        output_dir=tmp_path / "ws" / "artifacts" / "s" / "run-0007",
    )
    # Um driver de experimento dispõe os seus runs como run-NNNN/<braço>, não pelo DAG.
    arm = StageRequest(
        stage_id="s",
        inputs={},
        components={},
        config_digest="d",
        output_dir=tmp_path / "campaign" / "run-0007" / "eagle",
    )
    named = StageRequest(
        stage_id="s",
        inputs={},
        components={},
        config_digest="d",
        output_dir=tmp_path / "maps" / "corridor-map" / "000007",
    )

    assert numbered.run_number() == 7
    assert arm.run_number() == 7
    assert named.run_number() == 0
    assert _request().run_number() == 0
