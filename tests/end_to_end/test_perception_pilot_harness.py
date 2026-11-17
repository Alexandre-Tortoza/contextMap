"""Dry run of the #665 perception pilot over the CI chain, end to end, and its refusals.

The harness under ``experiments/perception-pilot-ablation-20260927/`` is the command the GPU
machine runs: seven arms, each its own branch and its own ContextMap over one spatial
foundation. Here it runs the same way, in fresh processes through the ``contextmap context``
CLI, over the synthetic foundation, with perception replaced by the canned double (stamped with
its execution time, as a real execution is, so the replica is its own evidence) and every other
stage composed from configuration. What this proves is the path, the refusals and the
configuration checks, never anything about the arms' science.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from incremental_harness import (
    REPOSITORY,
    checkout_copy,
    commit_all,
    foundation_run,
    run_git,
    run_script,
    write_config,
)

from contextmap.evaluation import RegionJoinPolicy
from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.ingestion import SourceObservationId
from contextmap.point_representation import OBSERVATION_SUPPORT_ANCHOR_POLICY_ID
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.runtime import (
    check_selection,
    compose_executors,
    open_branch,
    resolve_effective_config,
    resolve_semantic_request_policy,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.semantic_mapping import SemanticMapId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PerceptionResult

EXPERIMENT = "perception-pilot-ablation-20260927"
BUNDLE = REPOSITORY / "experiments" / EXPERIMENT
SCRIPTS = BUNDLE / "scripts"
PILOT = Path("experiments", EXPERIMENT, "scripts", "run_pilot.py")
"""The pilot script, relative to a checkout."""
CONFIG = BUNDLE / "config"
BASE_TEMPLATE = CONFIG / "base.template.json"
POINT_TEMPLATE = CONFIG / "point-representation.template.json"
ARMS = (
    "sam2-region-v1",
    "sam2-region-abstention-v1",
    "sam2-region-scene-context-v1",
    "sam3-region-v1",
    "sam3-region-abstention-v1",
    "sam3-region-scene-context-v1",
    "sam2-region-v1-replica",
)
CI_SELECTION = (
    REPOSITORY
    / "src"
    / "contextmap"
    / "evaluation"
    / "docs"
    / "scenarios"
    / "solution-1-canonical-1.0.5-ci-synthetic-subset.json"
)
# O dublê da percepção carimba cada execução com a hora, como a percepção real: sem isso a
# réplica reproduziria o baseline byte a byte e seria a mesma ContextRun.
DRY_RUN_EXECUTORS = "chain:replicable_dry_run_executors"


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Uma dataclass resolve o próprio módulo em sys.modules durante a execução dele.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _machine_values(value: Any) -> Any:
    """Fill each placeholder with a value of the type its parameter needs, as the machine does."""
    if isinstance(value, dict):
        return {key: _machine_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_machine_values(item) for item in value]
    if not isinstance(value, str) or "REPLACE" not in value:
        return value
    if "40-HEX" in value:
        return "0" * 40
    if "PRECISION" in value:
        return "float16"
    if "TOKEN-BUDGET" in value:
        return 256
    return "dry-run"


def _filled_arms(tmp_path: Path) -> Path:
    arms = tmp_path / "arms"
    arms.mkdir()
    for layer in (CONFIG / "arms").glob("*.json"):
        (arms / layer.name).write_text(json.dumps(_machine_values(_read(layer))), encoding="utf-8")
    return arms


def _pilot(
    tmp_path: Path,
    *extra: str,
    arms: Path | None = None,
    selection: Path | None = CI_SELECTION,
    script: Path = SCRIPTS / "run_pilot.py",
    code_identity: str | None = "dry-run",
) -> tuple[int, Path, str]:
    config = write_config(tmp_path)
    workspace = tmp_path / "ws"
    foundation = foundation_run(workspace, config)
    arguments = [
        "--workspace",
        str(workspace),
        "--foundation-run",
        str(foundation),
        "--base-config",
        str(config),
        "--arms-dir",
        str(arms or _filled_arms(tmp_path)),
        "--outputs-root",
        str(tmp_path / "out"),
    ]
    if code_identity is not None:
        arguments += ["--code-identity", code_identity]
    if selection is not None:
        arguments += ["--selection-file", str(selection)]
    result = run_script(script, *arguments, *extra)
    harness = tmp_path / "out" / EXPERIMENT / "run-0001" / "harness"
    return result.returncode, harness, result.stdout + result.stderr


def _branch(arm: str) -> str:
    return f"pilot-run-0001-{arm}"


# --- the dry run -------------------------------------------------------------------------


def test_every_arm_runs_through_the_cli_and_publishes_its_own_context_map(tmp_path: Path) -> None:
    code, harness, output = _pilot(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS)

    assert code == 0, output
    protocol = _read(harness / "protocol.json")
    report = _read(harness / "report.json")
    maps = _read(harness / "maps.json")
    assert protocol["evidence_class"] == report["evidence_class"] == "dry_run"
    assert report["status"] == "complete" and report["failure"] is None
    assert [arm["arm"] for arm in protocol["arms"]] == list(ARMS)
    assert protocol["selection"]["observation_ids"] == ["frame-0000", "frame-0001", "frame-0002"]
    assert protocol["replica_join"]["policy_id"] == "pilot-replica-join/v1"
    assert protocol["replica_join"]["min_iou"] == 0.5
    # Cada braço difere do baseline só nas variáveis que ele atribui de outro jeito.
    by_arm = {arm["arm"]: arm for arm in protocol["arms"]}
    assert by_arm["sam2-region-v1-replica"]["differences_from_baseline"] == []
    for arm in protocol["arms"]:
        for difference in arm["differences_from_baseline"]:
            assert difference["declared_by"], difference

    workspace = tmp_path / "ws"
    assert set(report["arms"]) == set(maps["arms"]) == set(ARMS)
    for arm in ARMS:
        entry = report["arms"][arm]
        assert entry["status"] == "complete"
        assert open_branch(workspace, dataset=CI_FIXTURE_ID, name=_branch(arm)).revision == 1
        for step in ("branch", "context_run", "build"):
            assert entry["steps"][step]["returncode"] == 0
            assert entry["steps"][step]["wall_time_seconds"] > 0
            assert entry["steps"][step]["peak_rss_mb"] > 0
        assert entry["steps"]["context_run"]["stages"]["visual_perception"] == "executed"
        assert entry["build"]["validation"]["status"] == "verified"
        assert entry["build"]["integrity_problems"] == []
        perception = entry["perception"]
        assert perception["frames"] == 3
        assert perception["regions_per_frame"] == {"min": 2, "median": 2.0, "max": 2, "total": 6}
        assert perception["semantic_requests"]["attempted"] == 6
        assert perception["backend_call_failures"] == {}
        assert perception["region_labels"]["top"][0] == ["pallet", 3]
        located = maps["arms"][arm]
        assert located["branch"] == _branch(arm)
        assert located["context_map_id"] == entry["build"]["context_map_id"]
        # O mapa é publicado no maps root (irmão do run store), não sob o workspace.
        assert not Path(located["paths"]["context_map"]).is_absolute()
        assert (workspace.parent / "maps" / located["paths"]["context_map"]).is_dir()
        for kind in (
            "region_artifact",
            "visual_evidence_artifact",
            "semantic_interpretation_artifact",
        ):
            assert not Path(located["paths"][kind]).is_absolute()
            assert (workspace / located["paths"][kind]).is_dir()
        assert located["paths"]["point_representation_run"] is None
    # Sete branches, sete mapas: a réplica é outra inferência, então outro mapa.
    assert len({maps["arms"][arm]["context_map_id"] for arm in ARMS}) == len(ARMS)
    assert report["criterion"] == {
        "arms": 7,
        "complete_arms": 7,
        "verified_maps": 7,
        "distinct_context_maps": 7,
        "passes": True,
    }
    agreement = report["replica_agreement"]
    assert agreement["policy_id"] == "pilot-replica-join/v1"
    assert agreement["matched"] == 6 and agreement["compared"] == 6
    assert agreement["agreement"] == 1.0
    # O que volta para o Git não pode citar um caminho da máquina.
    for name in ("protocol.json", "report.json", "maps.json"):
        assert str(tmp_path) not in (harness / name).read_text(encoding="utf-8"), name


def test_the_ptv3_layer_gives_every_arm_its_point_representation_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The layer the real run applies to every arm, with the PTv3 runtime a provider loads.

    The provider is the deterministic double of the PTv3 tests, imported by the pilot's fresh
    processes the way a machine's loaders module is.
    """
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY / "tests" / "point_representation"))
    layer = _machine_values(_read(POINT_TEMPLATE))
    layer["resources"]["providers"]["point_representation.encoder"] = (
        "pointrep_ptv3_fakes:provide_fake_ptv3"
    )
    shared = tmp_path / "point-representation.json"
    shared.write_text(json.dumps(layer), encoding="utf-8")

    code, harness, output = _pilot(
        tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, "--all-arms-config", str(shared)
    )

    assert code == 0, output
    report = _read(harness / "report.json")
    maps = _read(harness / "maps.json")
    assert report["criterion"]["passes"] is True
    workspace = tmp_path / "ws"
    runs = set()
    for arm in ARMS:
        stages = report["arms"][arm]["steps"]["context_run"]["stages"]
        assert stages["point_representation"] == "executed", arm
        located = maps["arms"][arm]["paths"]["point_representation_run"]
        assert located is not None and not Path(located).is_absolute(), arm
        assert (workspace / located).is_dir(), arm
        runs.add(located)
    # Cada braço representa as observações da própria associação: um run por braço.
    assert len(runs) == len(ARMS)


# --- refusals ----------------------------------------------------------------------------


def test_absent_perception_providers_stop_the_pilot_before_any_context_run_exists(
    tmp_path: Path,
) -> None:
    # Um run real exige um checkout limpo: roda da própria cópia, qualquer que seja a árvore
    # de quem roda os testes, e grava o commit dela.
    code, harness, _ = _pilot(
        tmp_path, script=checkout_copy(tmp_path, EXPERIMENT) / PILOT, code_identity=None
    )

    assert code == 1
    report = _read(harness / "report.json")
    assert report["status"] == "failed"
    assert report["failure"]["arm"] == ARMS[0]
    assert report["failure"]["step"] == "context-run"
    assert "visual_perception" in report["failure"]["message"]
    assert "resources.providers" in report["failure"]["hint"]
    assert report["arms"][ARMS[0]]["status"] == "failed"
    assert {report["arms"][arm]["status"] for arm in ARMS[1:]} == {"not_run"}
    # Nada foi fingido: nenhuma ContextRun entrou na branch criada.
    workspace = tmp_path / "ws"
    assert open_branch(workspace, dataset=CI_FIXTURE_ID, name=_branch(ARMS[0])).revision == 0
    assert _read(harness / "maps.json")["arms"] == {}
    assert report["criterion"]["passes"] is False
    for name in ("protocol.json", "report.json", "maps.json"):
        assert str(tmp_path) not in (harness / name).read_text(encoding="utf-8"), name


def test_a_real_run_from_a_dirty_checkout_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    checkout = checkout_copy(tmp_path, EXPERIMENT)
    # A mudança não commitada é código que roda sem que o commit registrado o identifique.
    with (checkout / "experiments" / EXPERIMENT / "scripts" / "pilot_arms.py").open("a") as file:
        file.write("# a local change\n")

    code, harness, output = _pilot(tmp_path, script=checkout / PILOT, code_identity=None)

    assert code == 2, output
    assert "dirty" in output and "commit or stash" in output
    assert not harness.exists() and not harness.parent.exists()


@pytest.mark.parametrize("git", ["none", "initialized"])
def test_a_real_run_without_a_commit_is_refused_before_anything_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: str
) -> None:
    # Sem o teto, o git subiria até um repositório que contivesse o diretório do teste.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    code, harness, output = _pilot(
        tmp_path, script=checkout_copy(tmp_path, EXPERIMENT, git=git) / PILOT, code_identity=None
    )

    assert code == 2, output
    assert "no commit" in output and "commit or stash" in output
    assert not harness.exists() and not harness.parent.exists()


@pytest.mark.parametrize("named", ["omitted", "abbreviated"])
def test_a_real_run_from_a_clean_checkout_records_the_full_sha_of_its_commit(
    tmp_path: Path, named: str
) -> None:
    checkout = checkout_copy(tmp_path, EXPERIMENT)
    head = run_git(checkout, "rev-parse", "HEAD")

    code, harness, output = _pilot(
        tmp_path,
        script=checkout / PILOT,
        code_identity=None if named == "omitted" else head[:12],
    )

    # Sem providers a rodada para depois, no primeiro `context run`; aqui só importa que o
    # checkout limpo não é recusado e que o código gravado é o commit inteiro.
    assert "commit or stash" not in output
    assert code != 2, output
    assert _read(harness / "protocol.json")["code_identity"] == head


def test_a_real_run_naming_another_commit_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    checkout = checkout_copy(tmp_path, EXPERIMENT)
    earlier = run_git(checkout, "rev-parse", "HEAD")
    # A árvore continua limpa; só o commit citado não é o que roda.
    commit_all(checkout, "later")

    code, harness, output = _pilot(tmp_path, script=checkout / PILOT, code_identity=earlier)

    assert code == 2, output
    assert earlier in output and "--code-identity" in output
    assert not harness.exists() and not harness.parent.exists()


def test_a_selection_outside_the_foundation_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    snapshot = _read(CI_SELECTION)
    snapshot["subject"]["selection"]["selected_images"].append(
        {"observation_id": "frame-0099", "timestamp_ns": 99, "nearest_lidar_id": None}
    )
    selection = tmp_path / "outside.json"
    selection.write_text(json.dumps(snapshot), encoding="utf-8")

    code, harness, output = _pilot(
        tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, selection=selection
    )

    assert code == 2
    assert "frame-0099" in output
    # Nada foi criado: corrigir o argumento e rodar de novo usa o mesmo número de run.
    assert not harness.exists()


def test_the_default_selection_is_the_frozen_corridor_sample(tmp_path: Path) -> None:
    code, harness, output = _pilot(
        tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, selection=None
    )

    # A amostra congelada do corridor-02 foi escolhida sobre a sequência ingerida do corridor-02,
    # não sobre a fundação sintética.
    assert code == 2
    assert "corridor-02-sample" in output and "720a486de8d44c16a9d3d2ff9fa7b1a4" in output
    assert not harness.exists()


def test_a_selection_pinned_to_another_sequence_is_refused(tmp_path: Path) -> None:
    snapshot = _read(CI_SELECTION)
    snapshot["subject"]["sequence_artifact_id"] = "another-sequence"
    selection = tmp_path / "pinned.json"
    selection.write_text(json.dumps(snapshot), encoding="utf-8")

    code, harness, output = _pilot(
        tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, selection=selection
    )

    assert code == 2
    assert "another-sequence" in output
    assert not harness.exists()


def test_a_second_round_never_overwrites_the_first(tmp_path: Path) -> None:
    (tmp_path / "out" / EXPERIMENT / "run-0001" / "harness").mkdir(parents=True)

    code, _, output = _pilot(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS)

    assert code == 2
    assert "already exists" in output


def test_a_template_placeholder_left_in_an_arm_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    code, harness, output = _pilot(
        tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, arms=CONFIG / "arms"
    )

    assert code == 2
    assert "sam3-region-v1" in output and "REPLACE" in output
    assert not harness.exists()


def test_an_arm_that_varies_an_undeclared_field_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    arms = _filled_arms(tmp_path)
    layer = arms / "sam2-region-abstention-v1.json"
    document = _read(layer)
    document["components"]["semantic_interpretation"]["interpreter"]["qwen"]["temperature"] = 0.7
    layer.write_text(json.dumps(document), encoding="utf-8")

    code, harness, output = _pilot(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS, arms=arms)

    assert code == 2
    assert "sam2-region-abstention-v1" in output and "qwen.temperature" in output
    assert not harness.exists()


# --- the configuration layers ------------------------------------------------------------


def _resolved_templates() -> dict[str, Any]:
    arms = _module("pilot_arms")
    return {
        arm: arms.resolve_arm(BASE_TEMPLATE, (), CONFIG / "arms" / f"{arm}.json") for arm in ARMS
    }


def test_every_arm_layer_resolves_and_differs_from_the_baseline_only_in_its_variable() -> None:
    arms = _module("pilot_arms")

    differences = arms.check_controlled(_resolved_templates())

    assert differences["sam2-region-v1"] == ()
    assert differences["sam2-region-v1-replica"] == ()
    declared = {
        arm: {name for item in found for name in item.declared_by}
        for arm, found in differences.items()
    }
    assert declared["sam2-region-abstention-v1"] == {"region_prompt"}
    assert declared["sam2-region-scene-context-v1"] == {"region_prompt"}
    assert declared["sam3-region-v1"] == {"region_discovery"}
    assert declared["sam3-region-abstention-v1"] == {"region_discovery", "region_prompt"}
    assert declared["sam3-region-scene-context-v1"] == {"region_discovery", "region_prompt"}
    assert all(item.declared for found in differences.values() for item in found)


def test_an_arm_layer_that_does_not_change_its_variable_is_refused() -> None:
    arms = _module("pilot_arms")
    resolved = _resolved_templates()
    # Um braço SAM3 cuja camada esqueceu o SAM3 não é o braço que diz ser.
    resolved["sam3-region-v1"] = resolved["sam2-region-v1"]

    with pytest.raises(ValueError, match=r"sam3-region-v1.*region_discovery"):
        arms.check_controlled(resolved)


def test_each_arm_resolves_the_semantic_request_policy_its_prompt_names() -> None:
    views = SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,))
    expected = {
        "region/v1": SemanticPromptPolicy(scene="scene/v1", region="region/v1"),
        "region-abstention/v1": SemanticPromptPolicy(
            scene="scene/v1", region="region-abstention/v1"
        ),
        "region-scene-context/v1": SemanticPromptPolicy(
            scene="scene/v1", region="region-scene-context/v1", region_scene_context=True
        ),
    }
    arms = {arm.slug: arm for arm in _module("pilot_arms").ARMS}

    for slug, effective in _resolved_templates().items():
        policy = resolve_semantic_request_policy(effective)

        assert policy == SemanticRequestPolicy.from_prompt_policy(
            expected[arms[slug].region_prompt], views=views
        ), slug


def test_the_base_template_composes_every_stage_but_the_perception_it_leaves_open() -> None:
    effective = resolve_effective_config(files=[BASE_TEMPLATE])
    failures: dict[str, Exception] = {}

    composed = compose_executors(
        effective,
        semantic_map_id=SemanticMapId("semantic-map-template"),
        code_digest="sha256:" + "0" * 64,
        code_version="template",
        on_composition_failure=failures.__setitem__,
        environ={},
        module_available=lambda _name: True,
    )

    assert check_selection(effective.config) == ()
    assert set(composed) == {
        "state_estimation",
        "geometric_mapping",
        "sensor_association",
        "semantic_fusion",
        "semantic_mapping",
        "entity_resolution",
        "spatial_relations",
        "context_map",
    }
    # A percepção fica para a máquina com GPU: um placeholder, nunca um valor inventado.
    # Desde a decomposição são três produtores com dependências próprias:
    # discovery regional, checkpoint visual e provider de interpretação semântica.
    assert set(failures) == {"region_perception", "visual_perception", "semantic_interpretation"}


def test_the_base_template_declares_the_protocol_baseline() -> None:
    components = resolve_effective_config(files=[BASE_TEMPLATE]).config.components

    # Cada slot mora na capability que o possui desde a decomposição, por isso o id
    # completo: um prefixo único não serve mais para os quatro.
    selected = {
        component_id: (components[component_id].backend, name, value)
        for component_id, name, value in (
            ("region_perception.discovery", "checkpoint", "facebook/sam2.1-hiera-tiny"),
            ("visual_features.dense", "checkpoint", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            ("visual_features.region", "checkpoint", "openai/clip-vit-large-patch14"),
            ("semantic_interpretation.interpreter", "model", "Qwen/Qwen3-VL-4B-Instruct"),
        )
        if components[component_id].parameters[name] == value
    }
    qwen = components["semantic_interpretation.interpreter"].parameters

    assert [backend for backend, _, _ in selected.values()] == ["sam2", "dinov3", "clip", "qwen"]
    assert (qwen["quantization"], qwen["temperature"]) == ("4bit", 0.0)


def _fake_runtime(config: Any, secrets: Any) -> object:
    """Stand in for a model runtime a provider would load; composition never calls it."""
    return object()


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_composes_its_perception_once_the_machine_values_are_filled(
    arm: str, tmp_path: Path
) -> None:
    layers = []
    for source in (BASE_TEMPLATE, CONFIG / "arms" / f"{arm}.json"):
        layer = tmp_path / source.name
        layer.write_text(json.dumps(_machine_values(_read(source))), encoding="utf-8")
        layers.append(layer)
    failures: dict[str, Exception] = {}

    composed = compose_executors(
        resolve_effective_config(files=layers),
        providers={
            "region_perception.discovery": _fake_runtime,
            "semantic_interpretation.interpreter": _fake_runtime,
        },
        semantic_map_id=SemanticMapId("semantic-map-template"),
        code_digest="sha256:" + "0" * 64,
        code_version="template",
        on_composition_failure=failures.__setitem__,
        environ={},
        module_available=lambda _name: True,
    )

    assert failures == {}, failures
    assert "visual_perception" in composed


def test_the_point_representation_layer_turns_ptv3_on_for_every_arm() -> None:
    from contextmap.point_representation.backends.ptv3 import PTv3Config

    arms = _module("pilot_arms")
    for arm in ARMS:
        effective = arms.resolve_arm(
            BASE_TEMPLATE, (POINT_TEMPLATE,), CONFIG / "arms" / f"{arm}.json"
        )
        config = effective.config

        assert config.pipeline.stages["point_representation"] is True, arm
        assert config.components["point_representation.encoder"].backend == "ptv3", arm
        assert (
            config.components["point_representation.center_selection"].backend
            == OBSERVATION_SUPPORT_ANCHOR_POLICY_ID
        ), arm
    ptv3 = dict(_read(POINT_TEMPLATE)["components"]["point_representation"]["encoder"]["ptv3"])
    ptv3.pop("support_policy")

    encoder = PTv3Config(**ptv3)

    assert (encoder.variant, encoder.grid_size_m, encoder.output_dimension) == (
        "ptv3m1-base",
        0.05,
        64,
    )


# --- the metrics -------------------------------------------------------------------------

_BACKEND = BackendProvenance(
    backend_id="fixture", capability="region_discovery", provider="fixture", model="m", version="1"
)
_INTERPRETER = SemanticInferenceProvenance(
    backend=BackendProvenance(
        backend_id="fixture",
        capability="semantic_interpreter",
        provider="fixture",
        model="m",
        version="1",
    ),
    task_identity="region",
    prompt_template_id="region/v1",
    output_schema_version="semantic-response/1",
)


def _frame(run: str, regions: dict[str, tuple[int, int, str | None]]) -> PerceptionResult:
    """One 16 x 12 frame: each region a 4 x 4 box at ``(x, y)`` with an optional primary label."""
    observation = SourceObservationId("frame-0000")
    result_id = f"{run}--frame-0000"
    claims = [
        SemanticClaim(
            claim_id=ClaimId(f"{result_id}--{region}--claim"),
            source_observation_id=observation,
            semantic_interpretation_run_id=SemanticInterpretationRunId(run),
            hypothesis=label,
            role=HypothesisRole.PRIMARY,
            provenance=_INTERPRETER,
            region_id=RegionId(region),
        )
        for region, (_, _, label) in regions.items()
        if label is not None
    ]
    return PerceptionResult(
        source_observation_id=observation,
        region_artifact_id=f"{run}-regional",
        semantic_interpretation_artifact_id=run,
        sequence_artifact_id="sequence",
        regions=tuple(
            Region2D(
                region_id=RegionId(region),
                bounding_box=BoundingBox2D(x=x, y=y, width=4, height=4),
                provenance=_BACKEND,
                source_observation_id=observation,
                image_width=16,
                image_height=12,
            )
            for region, (x, y, _) in regions.items()
        ),
        claims=tuple(claims),
    )


def test_the_replica_label_agreement_compares_only_regions_both_runs_labeled() -> None:
    metrics = _module("pilot_metrics")
    baseline = _frame("baseline", {"b1": (0, 0, "door"), "b2": (8, 0, "wall"), "b3": (0, 8, None)})
    replica = _frame(
        "replica",
        {"r1": (0, 0, "door"), "r2": (8, 0, "floor"), "r3": (0, 8, "chair"), "r4": (12, 8, "box")},
    )

    agreement = metrics.label_agreement(
        {"frame-0000": baseline},
        {"frame-0000": replica},
        RegionJoinPolicy(policy_id="test-join/v1", min_iou=0.5),
    )

    assert agreement == {
        "policy_id": "test-join/v1",
        "min_iou": 0.5,
        "frames": 1,
        "frames_in_one_run_only": 0,
        "frames_without_regions": 0,
        "matched": 3,
        "unmatched_baseline": 0,
        "unmatched_replica": 1,
        "compared": 2,
        "agreeing": 1,
        "agreement": 0.5,
        "labeled_on_one_side": 1,
        "unlabeled_on_both": 0,
    }


def test_the_replica_agreement_is_undefined_when_no_pair_is_labeled_on_both_sides() -> None:
    metrics = _module("pilot_metrics")
    baseline = _frame("baseline", {"b1": (0, 0, None)})
    replica = _frame("replica", {"r1": (0, 0, None)})

    agreement = metrics.label_agreement(
        {"frame-0000": baseline},
        {"frame-0000": replica},
        RegionJoinPolicy(policy_id="test-join/v1", min_iou=0.5),
    )

    # 0/0 não é concordância perfeita.
    assert agreement["compared"] == 0 and agreement["agreement"] is None
    assert agreement["unlabeled_on_both"] == 1
