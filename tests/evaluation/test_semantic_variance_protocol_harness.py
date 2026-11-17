"""The #580 harness: frozen frames, arm isolation and one repetition through the runtime.

What runs on the GPU machine is the maintainer's model runtime; everything around it (which
frames, which flags in which arm, which configuration, what gets recorded) is deterministic and
is tested here, so a harness bug never costs a GPU run.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contextmap.evaluation import canonical_real_scenario
from contextmap.ingestion import (
    CalibrationEntry,
    CalibrationProvenance,
    CalibrationReferenceId,
    CalibrationSet,
    ExplicitIdsSelection,
    FrameId,
    ImageEncoding,
    ImageObservation,
    PinholeCameraModel,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactReader,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
    selection_identity,
)
from contextmap.ingestion.calibration import compute_content_hash
from contextmap.region_perception import (
    BoundingBox2D,
    Region2D,
    RegionId,
    RegionPerceptionRunManifest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
)
from contextmap.runtime import ArtifactRef, Runtime, StageRequest
from contextmap.runtime.executors import inventory_digest
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticInterpretationRunWriter,
    SemanticObservation,
)
from contextmap.shared import BackendProvenance, SourceTimestamp
from contextmap.visual_perception import PerceptionResult, PreparedImage
from semantic_artifact_builders import write_semantic_views

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "experiments" / "semantic-variance-protocol-20260927" / "scripts"
CLOCK = "fixture:header"
SHA = "c" * 40


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    yield importlib.import_module("run_experiment")


@pytest.fixture
def repetition(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    yield importlib.import_module("repetition")


def _calibration() -> CalibrationSet:
    model = PinholeCameraModel(width=2, height=1, fx=1.0, fy=1.0, cx=1.0, cy=0.5)
    entry = CalibrationEntry(
        calibration_id=CalibrationReferenceId("camera_1-calib"),
        sensor_id=SensorId("camera_1"),
        frame_id=FrameId("camera_1_optical"),
        camera_model=model,
        provenance=CalibrationProvenance(source_type="dataset", source_path="fixtures/calib.yaml"),
        content_hash=compute_content_hash(
            sensor_id=SensorId("camera_1"), frame_id=FrameId("camera_1_optical"), camera_model=model
        ),
    )
    return CalibrationSet(entries={entry.calibration_id: entry}, static_transforms=())


def _image(observation_id: str, timestamp_ns: int, value: int) -> ImageObservation:
    return ImageObservation(
        observation_id=SourceObservationId(observation_id),
        sensor_id=SensorId("camera_1"),
        frame_id=FrameId("camera_1_optical"),
        timestamp=SourceTimestamp(
            seconds=timestamp_ns // 1_000_000_000,
            nanoseconds=timestamp_ns % 1_000_000_000,
            clock_id=CLOCK,
        ),
        provenance=SourceProvenance(source_type="fixture", source_path="fixtures/images"),
        width=2,
        height=1,
        encoding=ImageEncoding.RGB8,
        data=bytes([value] * 6),
    )


FRAMES = (("f-1", 1_000_000_001), ("f-2", 2_000_000_002), ("f-3", 3_000_000_003))


def _source(directory: Path, artifact_id: str = "source") -> Path:
    with SequenceArtifactWriter(
        output_dir=directory,
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId(artifact_id),
    ) as writer:
        for index, (observation_id, timestamp_ns) in enumerate(FRAMES):
            writer.add_observation(_image(observation_id, timestamp_ns, index))
        writer.set_calibration(_calibration())
        writer.finalize()
    return directory


def _selection(harness: ModuleType, **changes: Any) -> Any:
    values: dict[str, Any] = {
        "scenario_version": "test",
        "scenario_digest": "sha256:scenario",
        "sequence_artifact_id": "source",
        "clock_id": CLOCK,
        "selection_identity": "sha256:selection",
        "frames": (FRAMES[0], FRAMES[2]),
    }
    values.update(changes)
    return harness.FrozenSelection(**values)


class TestFrozenSelection:
    def test_the_frames_are_the_twenty_scenario_1_0_4_froze(self, harness: ModuleType) -> None:
        selection = harness.frozen_selection()

        assert selection.scenario_version == "1.0.4"
        assert selection.sequence_artifact_id == "720a486de8d44c16a9d3d2ff9fa7b1a4"
        assert selection.clock_id == "corridor-02-header"
        assert len(selection.frames) == 20
        # A 1.0.5 e a 1.0.6 mantiveram a mesma amostra; se um dia divergir, este experimento
        # fica na 1.0.4.
        current = canonical_real_scenario().subject.selection.selected_images
        assert selection.frames == tuple(
            (image.observation_id, image.timestamp_ns) for image in current
        )


class TestDerivedSequence:
    def test_only_the_frozen_images_are_kept_with_their_bytes_and_lineage(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        source = _source(tmp_path / "source")
        selection = _selection(harness)

        manifest = harness.derive_frozen_sequence(source, tmp_path / "derived", selection)

        reader = SequenceArtifactReader(tmp_path / "derived")
        assert reader.verify_integrity() == []
        observations = {str(item.observation_id): item for item in reader.list_observations()}
        assert set(observations) == {"f-1", "f-3"}
        original = SequenceArtifactReader(source)
        for observation_id in ("f-1", "f-3"):
            kept = observations[observation_id]
            assert isinstance(kept, ImageObservation)
            assert kept == original.get_observation(SourceObservationId(observation_id))
        calibration = reader.read_calibration()
        assert calibration is not None
        assert set(calibration.entries) == {CalibrationReferenceId("camera_1-calib")}
        provenance = reader.read_provenance()
        assert provenance is not None
        assert provenance.source_type == "sequence_artifact"
        assert provenance.source_path == "source"
        assert provenance.ingestion_config["selection_id"] == selection_identity(
            SequenceArtifactId("source"),
            ExplicitIdsSelection(
                observation_ids=frozenset({SourceObservationId("f-1"), SourceObservationId("f-3")})
            ),
        )
        assert provenance.ingestion_config["frozen_selection"] == _selection(harness).to_record()
        assert manifest.sequence_name == "corridor-02"

    def test_the_derived_identity_depends_only_on_the_source_and_the_frames(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        source = _source(tmp_path / "source")
        selection = _selection(harness)

        first = harness.derive_frozen_sequence(source, tmp_path / "one", selection)
        second = harness.derive_frozen_sequence(source, tmp_path / "two", selection)

        assert first.artifact_id == second.artifact_id != "source"

    def test_a_source_other_than_the_pinned_artifact_is_refused(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        source = _source(tmp_path / "source", artifact_id="another")

        with pytest.raises(harness.HarnessError, match="another"):
            harness.derive_frozen_sequence(source, tmp_path / "derived", _selection(harness))
        assert not (tmp_path / "derived").exists()

    def test_a_missing_frame_is_refused(self, harness: ModuleType, tmp_path: Path) -> None:
        source = _source(tmp_path / "source")
        selection = _selection(harness, frames=(FRAMES[0], ("f-9", 9)))

        with pytest.raises(harness.HarnessError, match="f-9"):
            harness.derive_frozen_sequence(source, tmp_path / "derived", selection)

    def test_a_frame_at_another_instant_is_refused(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        source = _source(tmp_path / "source")
        selection = _selection(harness, frames=(("f-1", FRAMES[0][1] + 1),))

        with pytest.raises(harness.HarnessError, match="timestamp"):
            harness.derive_frozen_sequence(source, tmp_path / "derived", selection)


class TestArmCommand:
    def test_only_arm_b_runs_with_the_cublas_workspace_configuration(
        self, harness: ModuleType
    ) -> None:
        command = ["python", "repetition.py"]

        assert harness.arm_command("B", command) == [
            "env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "python",
            "repetition.py",
        ]
        # O braço A remove a variável mesmo que o shell do mantenedor a tenha exportado.
        assert harness.arm_command("A", command) == [
            "env",
            "-u",
            "CUBLAS_WORKSPACE_CONFIG",
            "python",
            "repetition.py",
        ]

    def test_an_unknown_arm_is_refused(self, harness: ModuleType) -> None:
        with pytest.raises(harness.HarnessError, match="C"):
            harness.arm_command("C", ["python"])

    def test_repetitions_alternate_between_the_arms(self, harness: ModuleType) -> None:
        assert harness.schedule(("A", "B"), 3) == [
            ("A", 1),
            ("B", 1),
            ("A", 2),
            ("B", 2),
            ("A", 3),
            ("B", 3),
        ]


PROVIDER_MODULE = '''
"""A declared provider for the harness tests: it returns a runtime nobody calls."""


def load(config, secrets):
    return object()
'''


def _parameters(**region_discovery: Any) -> dict[str, Any]:
    region = {
        "backend": "sam2",
        "sam2": {"checkpoint": "sam2.1-hiera-tiny", "model_version": "2.1"},
    }
    region.update(region_discovery)
    return {
        "resources": {
            "device": "cpu",
            "providers": {
                "region_perception.discovery": "variance_test_provider:load",
                "semantic_interpretation.interpreter": "variance_test_provider:load",
            },
        },
        "components": {
            "region_perception": {"discovery": region},
            "visual_features": {
                "dense": {
                    "backend": "dinov2",
                    "dinov2": {"checkpoint": "facebook/dinov2-base", "revision": SHA},
                },
                "region": {
                    "backend": "clip",
                    "clip": {"checkpoint": "openai/clip-vit-large-patch14", "revision": SHA},
                },
            },
            "semantic_interpretation": {
                "interpreter": {
                    "backend": "qwen",
                    "qwen": {
                        "model": "Qwen/Qwen3-VL-4B-Instruct",
                        "precision": "float32",
                        "max_new_tokens": 256,
                        "temperature": 0.0,
                        "revision": SHA,
                        "prompt_policy": {"scene": "scene/v1", "region": "region/v1"},
                        "view_policy": {"region_views": ["tight_crop"]},
                    },
                },
            },
        },
    }


@pytest.fixture
def provider_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "variance_test_provider.py").write_text(PROVIDER_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(modules))


def _config_files(harness: ModuleType, directory: Path, parameters: dict[str, Any]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    scenario = directory / "scenario-runtime.json"
    scenario.write_text(json.dumps(harness.scenario_document()), encoding="utf-8")
    layer = directory / "parameters.json"
    layer.write_text(json.dumps(parameters), encoding="utf-8")
    return [scenario, layer]


def _preflight(harness: ModuleType, tmp_path: Path, parameters: dict[str, Any]) -> list[str]:
    problems: list[str] = harness.preflight(
        _config_files(harness, tmp_path / "config", parameters),
        dataset="corridor-02",
        module_available=lambda _name: True,
    )
    return problems


@pytest.mark.usefixtures("provider_module")
class TestPreflight:
    def test_the_canonical_backends_with_valid_parameters_pass(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        assert _preflight(harness, tmp_path, _parameters()) == []

    def test_a_backend_other_than_the_scenario_one_is_a_problem(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        parameters = _parameters(
            backend="sam3",
            sam3={
                "checkpoint": "sam3",
                "model_version": "3",
                "strategy": "text_prompt",
                "prompts": ["x"],
            },
        )

        problems = _preflight(harness, tmp_path, parameters)

        assert any("sam2" in problem and "sam3" in problem for problem in problems)

    def test_a_backend_without_a_provider_is_a_problem(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        parameters = _parameters()
        del parameters["resources"]["providers"]["region_perception.discovery"]

        problems = _preflight(harness, tmp_path, parameters)

        # A descoberta compõe no estágio de regiões, que a repetição também roda: o problema é
        # daquele estágio, com a causa real no componente dele.
        (problem,) = problems
        assert problem.startswith("region_perception cannot be composed:")
        assert "region_perception.discovery" in problem

    def test_an_unresolvable_provider_is_a_problem(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        parameters = _parameters()
        parameters["resources"]["providers"]["region_perception.discovery"] = "nowhere:load"

        problems = _preflight(harness, tmp_path, parameters)

        assert any("nowhere" in problem for problem in problems)

    def test_a_parameter_its_backend_rejects_is_a_problem(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        parameters = _parameters()
        parameters["components"]["semantic_interpretation"]["interpreter"]["qwen"][
            "max_new_tokens"
        ] = 0

        problems = _preflight(harness, tmp_path, parameters)

        assert any("max_new_tokens" in problem for problem in problems)

    def test_a_layer_without_the_semantic_request_policy_is_a_problem(
        self, harness: ModuleType, tmp_path: Path
    ) -> None:
        """Since the perception ablation the request policy has no default (#542, #524)."""
        parameters = _parameters()
        qwen = parameters["components"]["semantic_interpretation"]["interpreter"]["qwen"]
        del qwen["prompt_policy"], qwen["view_policy"]

        problems = _preflight(harness, tmp_path, parameters)

        assert any("prompt_policy" in problem and "view_policy" in problem for problem in problems)


class _FakeTorch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.cuda = SimpleNamespace(
            manual_seed_all=lambda seed: self.calls.append(("cuda.manual_seed_all", seed))
        )
        self._deterministic = False

    def manual_seed(self, seed: int) -> None:
        self.calls.append(("manual_seed", seed))

    def use_deterministic_algorithms(self, mode: bool) -> None:
        self.calls.append(("use_deterministic_algorithms", mode))
        self._deterministic = mode

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic


class _Seeded:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def seed(self, value: int) -> None:
        self.seeds.append(value)


class TestDeterminism:
    def test_arm_b_fixes_every_seed_and_the_deterministic_algorithms(
        self, repetition: ModuleType
    ) -> None:
        torch, python_random, numpy_random = _FakeTorch(), _Seeded(), _Seeded()

        applied = repetition.apply_determinism(
            "B",
            seed=7,
            environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            torch=torch,
            python_random=python_random,
            numpy_random=numpy_random,
        )

        assert torch.calls == [
            ("manual_seed", 7),
            ("cuda.manual_seed_all", 7),
            ("use_deterministic_algorithms", True),
        ]
        assert python_random.seeds == numpy_random.seeds == [7]
        assert applied == {
            "seed": 7,
            "use_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        }

    def test_arm_b_refuses_to_start_without_the_cublas_workspace_configuration(
        self, repetition: ModuleType
    ) -> None:
        with pytest.raises(repetition.RepetitionError, match="CUBLAS_WORKSPACE_CONFIG"):
            repetition.apply_determinism(
                "B",
                seed=0,
                environ={},
                torch=_FakeTorch(),
                python_random=_Seeded(),
                numpy_random=_Seeded(),
            )

    def test_arm_a_touches_nothing(self, repetition: ModuleType) -> None:
        torch, python_random, numpy_random = _FakeTorch(), _Seeded(), _Seeded()

        applied = repetition.apply_determinism(
            "A",
            seed=0,
            environ={},
            torch=torch,
            python_random=python_random,
            numpy_random=numpy_random,
        )

        assert torch.calls == python_random.seeds == numpy_random.seeds == []
        assert applied == {
            "seed": None,
            "use_deterministic_algorithms": False,
            "cublas_workspace_config": None,
        }

    def test_arm_a_refuses_a_leaked_cublas_workspace_configuration(
        self, repetition: ModuleType
    ) -> None:
        with pytest.raises(repetition.RepetitionError, match="arm A"):
            repetition.apply_determinism(
                "A",
                seed=0,
                environ={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                torch=None,
                python_random=_Seeded(),
                numpy_random=_Seeded(),
            )


class _FakeRegionPerception:
    """Stands in for the region stage the measured stage now depends on.

    ``region_perception`` is an executed stage upstream of ``visual_perception``, so every
    repetition runs it too. The perception double does not read the region run, so this only
    has to publish a well-formed region artifact for the plan to be runnable.
    """

    def execute(self, request: StageRequest) -> ArtifactRef:
        assert request.output_dir is not None and request.workspace is not None
        run_id = request.identity()
        (sequence,) = request.inputs["sequence"]
        reader = SequenceArtifactReader(request.directory_of(sequence))
        selected = tuple(entry.observation_id for entry in reader.iter_index())
        writer = RegionPerceptionRunWriter(output_dir=request.output_dir)
        writer.write_manifest(
            RegionPerceptionRunManifest(
                run_id=run_id,
                run_index=request.run_number(),
                sequence_name="corridor-02",
                selection_id="selection",
                sequence_artifact_id=str(reader.manifest.artifact_id),
                source_observation_ids=selected,
                created_at="2026-09-27T00:00:00+00:00",
                prepared_image_id="fake-prepared-image",
                discovery_backend_id="fake",
                discovery_model="fake",
                discovery_version="1",
                discovery_configuration_fingerprint=request.config_digest,
                normalization_policy_fingerprint=request.config_digest,
            )
        )
        for obs in selected:
            writer.write_frame(
                PreparedImage(
                    source_observation_id=SourceObservationId(obs),
                    payload_reference="source.png",
                    width=1,
                    height=1,
                )
            )
            writer.write_region(
                Region2D(
                    region_id=RegionId(f"{run_id}--{obs}--region-0000"),
                    source_observation_id=SourceObservationId(obs),
                    bounding_box=BoundingBox2D(x=0, y=0, width=1, height=1),
                    provenance=_backend("region_discovery"),
                )
            )
        manifest = writer.finalize()
        return ArtifactRef(
            stage_id=request.stage_id,
            contract="RegionPerceptionRunArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


class _FakePerception:
    """Stands in for the composed VisualPerceptionExecutor: writes an empty, valid run."""

    def __init__(self) -> None:
        self.requests: list[StageRequest] = []

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.requests.append(request)
        assert request.output_dir is not None and request.workspace is not None
        (sequence,) = request.inputs["sequence"]
        (regional,) = request.inputs["regions"]
        reader = SequenceArtifactReader(request.directory_of(sequence))
        writer = SemanticInterpretationRunWriter(
            output_dir=request.output_dir,
            sequence_name="corridor-02",
            run_id=SemanticInterpretationRunId(request.identity()),
            run_index=request.run_number(),
            sequence_artifact_id=str(reader.manifest.artifact_id),
            selection_id="sha256:selection",
            region_artifact_id=regional.artifact_id,
            configuration_digest="sha256:pipeline",
        )
        for entry in reader.iter_index():
            writer.add_observation(
                SemanticObservation(
                    source_observation_id=SourceObservationId(entry.observation_id),
                    semantic_interpretation_run_id=SemanticInterpretationRunId(request.identity()),
                )
            )
        manifest = writer.finalize()
        return ArtifactRef(
            stage_id=request.stage_id,
            contract="SemanticInterpretationArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


@pytest.mark.usefixtures("provider_module")
class TestRepetition:
    def _run(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        executor: _FakePerception,
    ) -> dict[str, Any]:
        workspace = tmp_path / "workspace"
        harness.derive_frozen_sequence(
            _source(tmp_path / "source"), workspace / "inputs" / "sequence", _selection(harness)
        )
        runtime = Runtime(
            workspace=workspace,
            executors={
                "region_perception": _FakeRegionPerception(),
                "semantic_interpretation": executor,
            },
            module_available=lambda _name: True,
        )
        files: Sequence[Path] = _config_files(harness, tmp_path / "config", _parameters())
        record: dict[str, Any] = repetition.run_repetition(
            runtime=runtime,
            config_files=files,
            sequence_location="inputs/sequence",
            dataset="corridor-02",
            code_identity="commit-under-test",
        )
        return record

    def test_one_repetition_runs_the_visual_perception_stage_on_the_frozen_frames(
        self, harness: ModuleType, repetition: ModuleType, tmp_path: Path
    ) -> None:
        executor = _FakePerception()

        record = self._run(harness, repetition, tmp_path, executor)

        (request,) = executor.requests
        (sequence,) = request.inputs["sequence"]
        assert sequence.location == "inputs/sequence"
        assert record["status"] == "completed"
        assert record["run"] == "corridor-02/run-0001"
        assert (
            record["semantic_interpretation"]["location"]
            == "artifacts/semantic_interpretation/run-0001"
        )
        assert record["semantic_interpretation"]["content_hash"].startswith("sha256:")
        # A percepção consome o run de regiões da mesma repetição: o registro nomeia os dois,
        # senão não se saberia sobre quais regiões as claims comparadas foram feitas.
        (regions,) = request.inputs["regions"]
        assert regions.location == "artifacts/region_perception/run-0001"
        assert record["regions"]["location"] == regions.location
        assert record["regions"]["content_hash"].startswith("sha256:")
        assert record["config_digest"].startswith("sha256:")
        assert record["code_identity"] == "commit-under-test"
        assert record["failure"] is None
        # Nenhum caminho absoluto da máquina entra no registro que volta para o repositório.
        assert str(tmp_path) not in json.dumps(record)

    def test_a_second_repetition_is_a_new_run_never_a_reuse(
        self, harness: ModuleType, repetition: ModuleType, tmp_path: Path
    ) -> None:
        executor = _FakePerception()
        self._run(harness, repetition, tmp_path, executor)
        runtime = Runtime(
            workspace=tmp_path / "workspace",
            executors={
                "region_perception": _FakeRegionPerception(),
                "semantic_interpretation": executor,
            },
            module_available=lambda _name: True,
        )

        record = repetition.run_repetition(
            runtime=runtime,
            config_files=[
                tmp_path / "config" / "scenario-runtime.json",
                tmp_path / "config" / "parameters.json",
            ],
            sequence_location="inputs/sequence",
            dataset="corridor-02",
            code_identity="commit-under-test",
        )

        assert record["run"] == "corridor-02/run-0002"
        assert len(executor.requests) == 2

    def test_the_real_child_process_always_leaves_a_record_of_its_arm(
        self, harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # O filho é um processo novo: o módulo do provider e o `src/` chegam pelo PYTHONPATH.
        search_path = [str(tmp_path / "modules"), str(REPOSITORY_ROOT / "src")]
        monkeypatch.setenv(
            "PYTHONPATH", os.pathsep.join([*search_path, os.environ.get("PYTHONPATH", "")])
        )
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        workspace = tmp_path / "workspace"
        harness.derive_frozen_sequence(
            _source(tmp_path / "source"), workspace / "inputs" / "sequence", _selection(harness)
        )
        files = _config_files(harness, tmp_path / "config", _parameters())

        record, returncode = harness._repetition(
            arm="A",
            index=1,
            root=tmp_path / "root",
            workspace=workspace,
            config_files=files,
            dataset="corridor-02",
            code_identity="commit-under-test",
            seed=0,
        )

        # O runtime do provider de teste não infere nada: o run não conclui, mas fica registrado.
        assert returncode == 1
        assert record is not None
        assert record["status"] in {"blocked", "failed"}
        assert (record["arm"], record["repetition"], record["label"]) == ("A", 1, "A-01")
        # O braço A roda sem a variável, mesmo com ela exportada no shell de quem chamou.
        assert record["environment_variables"]["CUBLAS_WORKSPACE_CONFIG"] is None
        assert record["determinism"]["use_deterministic_algorithms"] is False
        assert record["measurement"]["returncode"] == 1
        assert "contextmap" in record["packages"]
        assert (tmp_path / "root" / "repetitions" / "A-01.log").is_file()
        assert harness.repetition_failures("A-01", record, returncode) != []


def _backend(capability: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=f"fake_{capability}",
        capability=capability,
        provider="fake",
        model="fake",
        version="0.1",
    )


class _ClaimingPerception(_FakePerception):
    """Writes one region and one claim per frame of its input: every repetition agrees."""

    def _results(self, request: StageRequest, run_id: str) -> list[PerceptionResult]:
        (sequence,) = request.inputs["sequence"]
        reader = SequenceArtifactReader(request.directory_of(sequence))
        regional = RegionPerceptionRunReader(request.directory_of(request.inputs["regions"][0]))
        all_regions = regional.read_regions()
        results = []
        for entry in reader.iter_index():
            region = next(
                item
                for item in all_regions
                if str(item.source_observation_id) == entry.observation_id
            )
            frame = SourceObservationId(entry.observation_id)
            result_id = SemanticInterpretationRunId(run_id)
            results.append(
                PerceptionResult(
                    source_observation_id=frame,
                    region_artifact_id=request.inputs["regions"][0].artifact_id,
                    semantic_interpretation_artifact_id=run_id,
                    sequence_artifact_id=str(reader.manifest.artifact_id),
                    regions=(region,),
                    claims=(
                        SemanticClaim(
                            claim_id=ClaimId("claim-0000"),
                            source_observation_id=frame,
                            semantic_interpretation_run_id=result_id,
                            hypothesis="chair",
                            role=HypothesisRole.PRIMARY,
                            provenance=SemanticInferenceProvenance(
                                backend=_backend("semantic_interpreter"),
                                task_identity="qwen-region-interpretation",
                                prompt_template_id="region/v1",
                                output_schema_version="semantic-response/1",
                            ),
                            region_id=region.region_id,
                        ),
                    ),
                )
            )
        return results

    fail_next = False

    def execute(self, request: StageRequest) -> ArtifactRef:
        self.requests.append(request)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        assert request.output_dir is not None and request.workspace is not None
        run_id = request.identity()
        results = self._results(request, run_id)
        writer = SemanticInterpretationRunWriter(
            output_dir=request.output_dir,
            sequence_name="corridor-02",
            run_id=SemanticInterpretationRunId(run_id),
            run_index=request.run_number(),
            sequence_artifact_id=results[0].sequence_artifact_id,
            selection_id="sha256:selection",
            region_artifact_id=request.inputs["regions"][0].artifact_id,
            configuration_digest="sha256:pipeline",
        )
        write_semantic_views(writer, results, run_id)
        manifest = writer.finalize()
        return ArtifactRef(
            stage_id=request.stage_id,
            contract="SemanticInterpretationArtifact",
            artifact_id=str(manifest.run_id),
            content_hash=inventory_digest(manifest.file_inventory),
            location=request.output_dir.relative_to(request.workspace).as_posix(),
        )


@pytest.mark.usefixtures("provider_module")
class TestExperiment:
    """The orchestrator end to end, with each repetition run in-process by a fake stage."""

    def _main(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *extra: str,
        fail_at: str | None = None,
        vanish_at: str | None = None,
        parameters: dict[str, Any] | None = None,
        selection: Any = None,
    ) -> tuple[int, Path]:
        source = _source(tmp_path / "source")
        frozen = _selection(harness) if selection is None else selection
        monkeypatch.setattr(harness, "frozen_selection", lambda: frozen)
        checked = harness.preflight
        monkeypatch.setattr(
            harness,
            "preflight",
            lambda files, *, dataset: checked(
                files, dataset=dataset, module_available=lambda _name: True
            ),
        )
        executor = _ClaimingPerception()

        def in_process(
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
            label = f"{arm}-{index:02d}"
            if label == vanish_at:
                return None, 137
            executor.fail_next = label == fail_at
            record: dict[str, Any] = repetition.run_repetition(
                runtime=Runtime(
                    workspace=workspace,
                    executors={
                        "region_perception": _FakeRegionPerception(),
                        "semantic_interpretation": executor,
                    },
                    module_available=lambda _name: True,
                ),
                config_files=config_files,
                sequence_location=harness.SEQUENCE_LOCATION,
                dataset=dataset,
                code_identity=code_identity,
            )
            record.update(arm=arm, repetition=index, label=label)
            # Como o processo filho: sai com 1 quando o run não concluiu.
            return record, 0 if record["status"] == "completed" else 1

        monkeypatch.setattr(harness, "_repetition", in_process)
        layer = tmp_path / "parameters.json"
        layer.write_text(json.dumps(parameters or _parameters()), encoding="utf-8")
        code: int = harness.main(
            [
                "--source-sequence",
                str(source),
                "--parameters",
                str(layer),
                "--outputs-root",
                str(tmp_path / "outputs"),
                *extra,
            ]
        )
        return code, tmp_path / "outputs" / harness.EXPERIMENT / "run-0001"

    def test_a_complete_experiment_reports_both_arms_without_machine_paths(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        code, root = self._main(harness, repetition, tmp_path, monkeypatch)

        assert code == 0
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        assert experiment["status"] == "complete"
        assert [record["label"] for record in experiment["repetitions"]] == [
            "A-01",
            "B-01",
            "A-02",
            "B-02",
            "A-03",
            "B-03",
        ]
        report_text = (root / "report" / "report.json").read_text(encoding="utf-8")
        report = json.loads(report_text)
        assert {name: len(arm["runs"]) for name, arm in report["arms"].items()} == {"A": 3, "B": 3}
        assert report["arms"]["A"]["spreads"]["tuple"]["min"] == 1.0
        assert report["hypotheses"]["H1"]["verdict"] == "corroborated"
        derived = SequenceArtifactReader(root / "workspace" / "inputs" / "sequence").manifest
        assert report["provenance"]["sequence"]["artifact_id"] == str(derived.artifact_id)
        assert report["arms"]["B"]["runs"][0]["location"] == (
            f"{harness.EXPERIMENT}/run-0001/workspace/artifacts/semantic_interpretation/run-0002"
        )
        for text in (report_text, (root / "experiment.json").read_text(encoding="utf-8")):
            assert str(tmp_path) not in text

    def test_a_failed_run_stops_the_experiment_before_any_report(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        code, root = self._main(harness, repetition, tmp_path, monkeypatch, fail_at="B-01")

        assert code == 1
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        assert experiment["status"] == "failed"
        (failure,) = experiment["failures"]
        assert failure.startswith("B-01: run corridor-02/run-0002 failed (exit status 1)")
        assert "illegal memory access" in failure
        assert [record["label"] for record in experiment["repetitions"]] == ["A-01", "B-01"]
        assert not (root / "report").exists()

    def test_a_repetition_that_left_no_record_stops_the_experiment(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        code, root = self._main(harness, repetition, tmp_path, monkeypatch, vanish_at="A-02")

        assert code == 1
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        assert experiment["failures"] == ["A-02: exited with status 137 and no record"]
        assert [record["label"] for record in experiment["repetitions"]] == ["A-01", "B-01"]
        assert not (root / "report").exists()

    def test_a_source_without_every_frozen_frame_fails_before_any_repetition(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        selection = _selection(harness, frames=(FRAMES[0], ("f-9", 9)))
        code, root = self._main(harness, repetition, tmp_path, monkeypatch, selection=selection)

        assert code == 1
        experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        assert experiment["status"] == "failed"
        (failure,) = experiment["failures"]
        assert failure.startswith("sequence: ") and "f-9" in failure
        assert not (root / "repetitions").exists()

    def test_a_preflight_only_run_writes_nothing(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        code, root = self._main(harness, repetition, tmp_path, monkeypatch, "--preflight-only")

        assert code == 0
        assert not root.exists()

    def test_a_failed_preflight_writes_nothing(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parameters = _parameters()
        del parameters["resources"]["providers"]["semantic_interpretation.interpreter"]

        code, root = self._main(harness, repetition, tmp_path, monkeypatch, parameters=parameters)

        assert code == 1
        assert not root.exists()

    def test_an_earlier_run_of_the_experiment_is_never_reused(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "outputs" / harness.EXPERIMENT / "run-0001").mkdir(parents=True)

        with pytest.raises(harness.HarnessError, match="never reused"):
            self._main(harness, repetition, tmp_path, monkeypatch)

    def test_fewer_than_three_repetitions_are_refused(
        self,
        harness: ModuleType,
        repetition: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pytest.raises(SystemExit):
            self._main(harness, repetition, tmp_path, monkeypatch, "--repetitions", "2")
