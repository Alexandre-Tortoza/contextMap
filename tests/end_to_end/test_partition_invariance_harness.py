"""Dry run of the #501 real-data harness over the CI chain, end to end.

The harness under ``experiments/incremental-partition-invariance-20260927/`` is the command the
GPU machine runs. Here it runs the same way, in fresh processes through the ``contextmap
context`` CLI, over the synthetic foundation, with perception replaced by the canned double and
every other stage composed from configuration. The canned perception is deterministic, so both
layers must come out equal; what this proves is the path, not the science.
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
    DRY_RUN_EXECUTORS,
    PARTITION_SCRIPTS,
    REPOSITORY,
    checkout_copy,
    foundation_run,
    run_script,
    write_config,
)

from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.runtime import (
    check_selection,
    compose_executors,
    open_branch,
    resolve_effective_config,
    resolve_semantic_request_policy,
)
from contextmap.semantic_interpretation import (
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.semantic_mapping import SemanticMapId

EXPERIMENT = "incremental-partition-invariance-20260927"
SCRIPTS = REPOSITORY / PARTITION_SCRIPTS
ARM = PARTITION_SCRIPTS / "run_real_arm.py"
"""The arm's script, relative to a checkout."""


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Uma dataclass resolve o próprio módulo em sys.modules durante a execução dele.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _arm(
    tmp_path: Path,
    *extra: str,
    script: Path = SCRIPTS / "run_real_arm.py",
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
        "--images",
        "2",
        "--chunks",
        "2",
        "--resamples",
        "200",
        *identity,
        "--outputs-root",
        str(tmp_path / "out"),
        *extra,
    )
    harness = tmp_path / "out" / EXPERIMENT / "run-0001"
    return result.returncode, harness / "harness", result.stdout + result.stderr


def _read(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def test_the_whole_arm_runs_through_the_cli_and_reports_every_criterion(tmp_path: Path) -> None:
    code, harness, output = _arm(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS)

    assert code == 0, output
    report = _read(harness / "report.json")
    protocol = _read(harness / "protocol.json")
    assert report["status"] == "complete" and report["failure"] is None
    assert report["evidence_class"] == "dry_run"
    assert protocol["evidence_class"] == "dry_run"
    assert protocol["selection"]["chunks"] == [
        protocol["selection"]["whole"]["observation_ids"][:1],
        protocol["selection"]["whole"]["observation_ids"][1:],
    ]
    assert protocol["bootstrap"] == {
        "method": "paired cluster bootstrap, percentile interval",
        "generator": "numpy.random.default_rng",
        "resamples": 200,
        "seed": 20260926,
        "confidence": 0.95,
    }
    assert set(report["builds"]) == {"W1", "W2", "C", "R", "C_policy"}
    for build in report["builds"].values():
        assert build["validation"]["status"] == "verified"
        assert build["integrity_problems"] == []
    # A percepção enlatada é determinística: as duas camadas saem iguais.
    assert report["layer_one"]["outcome"] == "holds"
    for metric in ("M_claim", "M_state", "M_label"):
        assert report["layer_two"][metric]["effect"]["point"] == 0.0
        assert report["layer_two"][metric]["decision"] == "equivalent_within_delta"
    assert report["scenario_2"]["holds"]
    assert (
        report["scenario_2"]["inference_results"][2]
        == 2 * report["scenario_2"]["inference_results"][0]
    )
    assert report["scenario_3"]["holds"]
    for name in ("W1", "W2", "C1", "C2", "build-W1", "build-C", "build-C_policy"):
        step = report["steps"][name]
        assert step["returncode"] == 0 and step["wall_time_seconds"] > 0
        assert step["peak_rss_mb"] > 0
    assert report["steps"]["W1"]["stages"]["visual_perception"] == "executed"
    assert report["steps"]["build-C"]["stages"]["visual_perception"] == "provided"
    # O que volta para o Git não pode citar um caminho da máquina.
    for name in ("report.json", "protocol.json"):
        assert str(tmp_path) not in (harness / name).read_text(encoding="utf-8"), name


def test_absent_perception_providers_stop_the_arm_before_any_context_run_exists(
    tmp_path: Path,
) -> None:
    # Um run real exige um checkout limpo: roda da própria cópia, qualquer que seja a árvore
    # de quem roda os testes, e grava o commit dela.
    code, harness, _ = _arm(
        tmp_path, script=checkout_copy(tmp_path, EXPERIMENT) / ARM, code_identity=None
    )

    assert code != 0
    report = _read(harness / "report.json")
    assert report["status"] == "failed"
    assert report["failure"]["step"] == "W1"
    assert "visual_perception" in report["failure"]["message"]
    assert "resources.providers" in report["failure"]["hint"]
    # Nada foi fingido: nenhuma ContextRun entrou em branch alguma.
    workspace = tmp_path / "ws"
    for branch in report["branches"].values():
        assert open_branch(workspace, dataset=CI_FIXTURE_ID, name=branch).revision == 0
    assert "builds" not in report or report["builds"] == {}


def test_a_real_run_from_a_dirty_checkout_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    checkout = checkout_copy(tmp_path, EXPERIMENT)
    # A mudança não commitada é código que roda sem que o commit registrado o identifique.
    with (checkout / PARTITION_SCRIPTS / "steps.py").open("a", encoding="utf-8") as file:
        file.write("# a local change\n")

    code, harness, output = _arm(tmp_path, script=checkout / ARM, code_identity=None)

    assert code == 2, output
    assert "dirty" in output and "commit or stash" in output
    assert not harness.exists() and not harness.parent.exists()


def test_a_selection_that_does_not_fit_is_refused_before_the_run_exists(tmp_path: Path) -> None:
    code, harness, output = _arm(tmp_path, "--first-image", "99")

    assert code == 2
    assert "do not fit" in output
    # Nada foi criado: corrigir o argumento e rodar de novo usa o mesmo número de run.
    assert not harness.exists()


def test_a_second_run_never_overwrites_the_first(tmp_path: Path) -> None:
    (tmp_path / "out" / "incremental-partition-invariance-20260927" / "run-0001" / "harness").mkdir(
        parents=True
    )

    code, _, output = _arm(tmp_path, "--dry-run-executors", DRY_RUN_EXECUTORS)

    assert code == 2
    assert "already exists" in output


def test_the_corridor_template_composes_every_stage_but_the_perception_it_leaves_open() -> None:
    template = SCRIPTS.parent / "config" / "corridor-02.template.json"
    effective = resolve_effective_config(files=[template])
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
    # A percepção fica para a máquina com GPU: um placeholder, nunca um valor inventado. Ela
    # ocupa dois estágios desde a decomposição, e a percepção depende do de região, por isso
    # os dois ficam sem executor: o motivo real é do de região, e o da percepção nomeia a
    # dependência.
    assert set(failures) == {"region_perception", "visual_perception", "semantic_interpretation"}


def test_the_corridor_template_declares_the_semantic_request_policy_of_the_v0_1_0_run() -> None:
    """The request policy has no default since the perception ablation (#542, #524).

    It is not a machine value, so the template declares it: the policy that reproduces the
    request of the v0.1.0 perception run, the prompts before #542 and the tight crop alone.
    """
    template = SCRIPTS.parent / "config" / "corridor-02.template.json"

    policy = resolve_semantic_request_policy(resolve_effective_config(files=[template]))

    assert policy == SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(scene="scene/v1", region="region/v1"),
        views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
    )


def test_the_corridor_template_quantizes_qwen_with_a_value_the_backend_accepts() -> None:
    from contextmap.semantic_interpretation.adapters.qwen import (
        QwenSemanticConfig,
    )

    template = json.loads((SCRIPTS.parent / "config" / "corridor-02.template.json").read_text())
    qwen = template["components"]["semantic_interpretation"]["interpreter"]["qwen"]
    # prompt_policy e view_policy são grupos reservados da política de request (#542, #524), que
    # a composition root separa antes de montar a configuração do backend; não são parâmetros dela.
    parameters = {
        name: value for name, value in qwen.items() if name not in {"prompt_policy", "view_policy"}
    }
    # Os placeholders são do usuário; só a quantização gravada no template é verificada aqui.
    filled = {**parameters, "precision": "float16", "max_new_tokens": 256}

    config = QwenSemanticConfig(**filled)

    assert config.quantization == "4bit"


# --- selection planning ------------------------------------------------------------------


def _images(count: int, *, every: int = 3) -> list[tuple[int, str]]:
    # Uma imagem a cada `every` posições do índice canônico, como imagens entre scans e IMU.
    return [(index * every + 1, f"camera-{index:06d}") for index in range(count)]


def test_the_whole_selection_is_consecutive_images_partitioned_into_equal_chunks() -> None:
    steps = _module("steps")

    plan = steps.plan_partition(_images(10), first=2, count=4, chunks=2)

    assert plan.whole == ("camera-000002", "camera-000003", "camera-000004", "camera-000005")
    assert plan.chunks == (
        ("camera-000002", "camera-000003"),
        ("camera-000004", "camera-000005"),
    )
    # A rerun cobre as mesmas imagens por uma faixa do índice: outra identidade de inferência.
    assert plan.rerun == {"kind": "frame_range", "start_frame_index": 7, "end_frame_index": 17}


@pytest.mark.parametrize(
    ("first", "count", "chunks", "message"),
    [
        (8, 4, 2, "only 10 images"),
        (0, 5, 2, "equal chunks"),
        (-1, 4, 2, "first"),
        (0, 0, 1, "count"),
    ],
)
def test_an_impossible_partition_is_refused(
    first: int, count: int, chunks: int, message: str
) -> None:
    steps = _module("steps")

    with pytest.raises(ValueError, match=message):
        steps.plan_partition(_images(10), first=first, count=count, chunks=chunks)
