"""Dry run of the #670 capability ablation campaign over the CI chain, end to end, and its refusals.

The harness under ``experiments/perception-ablation-campaign-20260927/`` is the command the GPU
machine runs: fourteen arms over one spatial foundation and one window of camera images, each
published to the viewer index as soon as it finishes. Here it runs the same way, in fresh
processes through the ``contextmap`` CLI, over the synthetic foundation and a synthetic window of
its sequence, with perception replaced by the canned double (stamped with its execution time and
holding a DINOv3 dense feature, as the baseline's perception does) and every other stage composed
from configuration. What this proves is the path, the reuse of the baseline's perception by the
reuse arms, the index, the refusals and the configuration checks, never anything about the arms'
science.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from incremental_harness import (
    REPOSITORY,
    checkout_copy,
    foundation_run,
    machine_values,
    run_git,
    run_script,
    write_config,
)

from contextmap.evaluation.ci_fixtures import CI_FIXTURE_ID
from contextmap.runtime import (
    compose,
    compose_executors,
    create_branch,
    foundation_of_run,
    open_branch,
    resolve_effective_config,
    resolve_semantic_request_policy,
)
from contextmap.semantic_fusion import QualityAwareAccumulationPolicy
from contextmap.semantic_interpretation import (
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.semantic_mapping import SemanticMapId
from contextmap.sensor_association import InterpolationPolicy

EXPERIMENT = "perception-ablation-campaign-20260927"
PILOT = "perception-pilot-ablation-20260927"
VOXEL = "geometric-aggregation-raw-vs-voxel-20260926"
BUNDLE = REPOSITORY / "experiments" / EXPERIMENT
SCRIPTS = BUNDLE / "scripts"
CAMPAIGN = Path("experiments", EXPERIMENT, "scripts", "run_campaign.py")
"""The campaign script, relative to a checkout."""
CONFIG = BUNDLE / "config"
ARMS_DIR = CONFIG / "arms"
BASE_TEMPLATE = CONFIG / "base.template.json"
POINT_TEMPLATE = CONFIG / "point-representation.template.json"
PILOT_CONFIG = REPOSITORY / "experiments" / PILOT / "config"
IMPORT_PATHS = (
    SCRIPTS,
    REPOSITORY / "experiments" / PILOT / "scripts",
    REPOSITORY / "experiments" / "incremental-partition-invariance-20260927" / "scripts",
)
ARMS = (
    "sam2-region-v1",
    "fusion-quality-aware",
    "association-dinov3-dense",
    "scoring-clip",
    "sam2-region-abstention-v1",
    "sam2-region-scene-context-v1",
    "sam3-region-v1",
    "sam3-region-abstention-v1",
    "sam3-region-scene-context-v1",
    "interpreter-florence2-region-category",
    "qwen-view-contextual-crop",
    "qwen-view-masked-subject",
    "sam2-region-v1-replica",
    # O Eagle 2.5 (8B em bf16, sem quantização) é o braço que mais pode falhar na máquina real:
    # por último, a falha dele não custa nenhum outro braço.
    "interpreter-eagle2-5",
)
EAGLE = "interpreter-eagle2-5"
BASELINE = "sam2-region-v1"
REPLICA = "sam2-region-v1-replica"
REUSE_ARMS = ("fusion-quality-aware", "association-dinov3-dense", "scoring-clip")
PERCEPTION_ARMS = tuple(arm for arm in ARMS if arm not in REUSE_ARMS)
PILOT_ARMS = (
    "sam2-region-v1",
    "sam2-region-abstention-v1",
    "sam2-region-scene-context-v1",
    "sam3-region-v1",
    "sam3-region-abstention-v1",
    "sam3-region-scene-context-v1",
    "sam2-region-v1-replica",
)
# O dublê da percepção carimba cada execução com a hora e grava uma feature densa DINOv3 por
# frame, como a percepção do baseline, que o braço do canal denso amostra.
DRY_RUN_EXECUTORS = "chain:campaign_dry_run_executors"
TEXT_TOWER = "chain:provide_dry_run_clip_text_tower"
PTV3 = "pointrep_ptv3_fakes:provide_fake_ptv3"
SCORING = "semantic_scoring.scorer"
# A base da CI não traz a política de request semântico do baseline nem o checkpoint do CLIP das
# features de região (a base do protocolo traz): esta camada comum os põe em todo braço, como a
# base real preenchida, para as camadas compararem igual e o scorer ver o espaço das features.
CLIP_CHECKPOINT = "openai/clip-vit-large-patch14"
BASELINE_REQUESTS = {
    "components": {
        "visual_features": {
            "region": {"clip": {"checkpoint": CLIP_CHECKPOINT, "revision": "0" * 40}},
        },
        "semantic_interpretation": {
            "interpreter": {
                "qwen": {
                    "prompt_policy": {"scene": "scene/v1", "region": "region/v1"},
                    "view_policy": {"region_views": ["tight_crop"]},
                }
            },
        },
    }
}


def _module(name: str) -> ModuleType:
    """Import a script module of the campaign, of the pilot or of #501, as the harness does."""
    for path in IMPORT_PATHS:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = next(path / f"{name}.py" for path in IMPORT_PATHS if (path / f"{name}.py").is_file())
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Uma dataclass resolve o próprio módulo em sys.modules durante a execução dele.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _filled_arms(tmp_path: Path, *, text_tower: str | None = TEXT_TOWER) -> Path:
    """Fill every arm layer as the machine does; the scorer gets the dry run's text tower."""
    arms = tmp_path / "arms"
    for layer in ARMS_DIR.glob("*.json"):
        document = machine_values(_read(layer))
        if layer.stem == "scoring-clip" and text_tower is not None:
            document.setdefault("resources", {}).setdefault("providers", {})[SCORING] = text_tower
        _write(arms / layer.name, document)
    return arms


def _shared_layers(tmp_path: Path, *, ptv3: bool) -> list[Path]:
    layers = [_write(tmp_path / "baseline-requests.json", BASELINE_REQUESTS)]
    if ptv3:
        layer = machine_values(_read(POINT_TEMPLATE))
        layer["resources"]["providers"]["point_representation.encoder"] = PTV3
        layers.append(_write(tmp_path / "point-representation.json", layer))
    return layers


def _window(tmp_path: Path, foundation_sequence: str, **changes: Any) -> Path:
    """A window over the whole synthetic sequence (three images, one per second).

    It has every field of the #669 record as written, the ones added after the contract
    included, which the campaign keeps and records without using.
    """
    anchor = {"seconds": 0, "nanoseconds": 0, "clock_id": "fixture:header"}
    document: dict[str, Any] = {
        "schema": "contextmap.corridor-window/v1",
        "bag": "contextmap-ci-subset.bag",
        "bag_first_message_recording_ns": 0,
        "start_offset_s": 0.0,
        "end_offset_s": 3.0,
        "padding_s": 0.0,
        "clock_id": "fixture:header",
        "start_ns": 0,
        "end_ns": 3_000_000_000,
        "start_recording_ns": 0,
        "end_recording_ns": 3_000_000_000,
        "start_raw_header_ns": 0,
        "end_raw_header_ns": 3_000_000_000,
        "sequence_artifact_id": foundation_sequence,
        "canonical_sequence_artifact_id": foundation_sequence,
        "sequence_location": f"{CI_FIXTURE_ID}/run-0001/ingestion",
        "pose_sequence_artifact_id": "not-used-by-the-ci-foundation",
        "pose_sequence_location": "not-used",
        "offset_correction": {
            "offset_ns": 0,
            "anchors": {"source_time": anchor, "reference_time": anchor},
        },
        "quality": {
            "max_pose_gap_ms": 0.0,
            "max_lidar_gap_ms": 0.0,
            "pose_gap_threshold_ms": 605,
            "lidar_gap_threshold_ms": 250,
            "within_thresholds": True,
        },
    }
    for key, value in changes.items():
        if key == "quality":
            document["quality"].update(value)
        else:
            document[key] = value
    return _write(tmp_path / "window.json", document)


class _Campaign:
    """One prepared round of the campaign over a fresh CI foundation."""

    def __init__(self, tmp_path: Path, *, ptv3: bool = False) -> None:
        self.tmp_path = tmp_path
        self.config = write_config(tmp_path)
        self.workspace = tmp_path / "ws"
        self.foundation = foundation_run(self.workspace, self.config)
        self.sequence_artifact_id = foundation_of_run(
            self.workspace, self.foundation
        ).sequence.artifact_id
        self.shared = _shared_layers(tmp_path, ptv3=ptv3)
        self.harness = tmp_path / "out" / EXPERIMENT / "run-0001" / "harness"

    def arguments(
        self,
        *extra: str,
        arms: Path | None = None,
        window: Path | None = None,
        code_identity: str | None = "dry-run",
        dry_run: bool = True,
    ) -> list[str]:
        arguments = [
            "--workspace",
            str(self.workspace),
            "--foundation-run",
            str(self.foundation),
            "--window-file",
            str(window or _window(self.tmp_path, self.sequence_artifact_id)),
            "--base-config",
            str(self.config),
            "--arms-dir",
            str(arms or _filled_arms(self.tmp_path)),
            "--outputs-root",
            str(self.tmp_path / "out"),
        ]
        for layer in self.shared:
            arguments += ["--all-arms-config", str(layer)]
        if code_identity is not None:
            arguments += ["--code-identity", code_identity]
        if dry_run:
            arguments += ["--dry-run-executors", DRY_RUN_EXECUTORS]
        return [*arguments, *extra]

    def run(
        self, *extra: str, script: Path = SCRIPTS / "run_campaign.py", **options: Any
    ) -> tuple[int, str]:
        result = run_script(script, *self.arguments(*extra, **options))
        return result.returncode, result.stdout + result.stderr


def _branch(arm: str) -> str:
    return f"campaign-run-0001-{arm}"


# --- the dry run of every arm -------------------------------------------------------------


class _IndexReader:
    """Reads the viewer index over and over while the campaign runs, as the viewer would."""

    def __init__(self, index: Path) -> None:
        self.index = index
        self.arm_counts: list[int] = []
        self.invalid: list[str] = []

    def read(self) -> None:
        try:
            text = self.index.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        try:
            document = json.loads(text)
            self.arm_counts.append(len(document["arms"]))
        except (ValueError, KeyError, TypeError) as error:
            self.invalid.append(f"{error!r}: {text[:200]!r}")


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Run the whole campaign once, with PTv3 on, reading its index while it runs."""
    patch = pytest.MonkeyPatch()
    patch.setenv("PYTHONPATH", str(REPOSITORY / "tests" / "point_representation"))
    tmp_path = tmp_path_factory.mktemp("campaign")
    campaign = _Campaign(tmp_path, ptv3=True)
    reader = _IndexReader(campaign.harness / "maps.json")
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome["code"], outcome["output"] = campaign.run()

    thread = threading.Thread(target=run)
    thread.start()
    while thread.is_alive():
        reader.read()
        time.sleep(0.02)
    thread.join()
    reader.read()
    yield {
        **outcome,
        "campaign": campaign,
        "reader": reader,
        "protocol": _read(campaign.harness / "protocol.json"),
        "report": _read(campaign.harness / "report.json"),
        "maps": _read(campaign.harness / "maps.json"),
    }
    patch.undo()


def test_every_arm_completes_and_the_report_passes(dry_run: dict[str, Any]) -> None:
    assert dry_run["code"] == 0, dry_run["output"]
    report, protocol = dry_run["report"], dry_run["protocol"]
    workspace = dry_run["campaign"].workspace

    assert protocol["evidence_class"] == report["evidence_class"] == "dry_run"
    assert report["status"] == "complete" and report["failure"] is None
    assert [arm["arm"] for arm in protocol["arms"]] == list(ARMS)
    # O relatório grava as chaves em ordem; a ordem de execução é a do protocolo e do índice.
    assert set(report["arms"]) == set(ARMS)
    for arm in ARMS:
        entry = report["arms"][arm]
        assert entry["status"] == "complete", arm
        for step in entry["steps"].values():
            assert step["returncode"] == 0, arm
            assert step["wall_time_seconds"] > 0 and step["peak_rss_mb"] > 0, arm
    for arm in PERCEPTION_ARMS:
        entry = report["arms"][arm]
        assert set(entry["steps"]) == {"branch", "context_run", "build"}, arm
        assert open_branch(workspace, dataset=CI_FIXTURE_ID, name=_branch(arm)).revision == 1
        assert entry["steps"]["context_run"]["stages"]["visual_perception"] == "executed"
        assert entry["steps"]["context_run"]["stages"]["point_representation"] == "executed"
        assert entry["build"]["validation"]["status"] == "verified", arm
        assert entry["build"]["integrity_problems"] == [], arm
        assert entry["perception"]["frames"] == 1, arm
        assert entry["perception"]["regions_per_frame"]["total"] == 2, arm
    assert report["criterion"] == {
        "arms": 14,
        "complete_arms": 14,
        "verified_maps": 13,
        "distinct_context_maps": 13,
        "reuse_holds": 3,
        "passes": True,
    }
    agreement = report["replica_agreement"]
    assert agreement["policy_id"] == "pilot-replica-join/v1"
    assert agreement["matched"] == 2 and agreement["agreement"] == 1.0
    # O que volta para o Git não pode citar um caminho da máquina.
    harness = dry_run["campaign"].harness
    for name in ("protocol.json", "report.json", "maps.json"):
        assert str(dry_run["campaign"].tmp_path) not in (harness / name).read_text(), name


def test_the_reuse_arms_consume_the_baseline_perception_and_never_rerun_it(
    dry_run: dict[str, Any],
) -> None:
    report, maps = dry_run["report"], dry_run["maps"]
    baseline = report["arms"][BASELINE]
    for arm in REUSE_ARMS:
        reuse = report["arms"][arm]["reuse"]
        assert reuse["source_arm"] == BASELINE, arm
        assert reuse["holds"] is True, arm
        for stage, key, path in (
            ("region_perception", "regions", "region_artifact"),
            ("visual_perception", "visual_evidence", "visual_evidence_artifact"),
            (
                "semantic_interpretation",
                "semantic_interpretation",
                "semantic_interpretation_artifact",
            ),
        ):
            producer = baseline["artifacts"][stage]
            assert reuse[key]["source"] == producer, arm
            assert reuse[key]["consumed"] == producer, arm
            assert reuse["dispositions"][stage] == "provided", arm
            assert maps["arms"][arm]["paths"][path] == producer["location"], arm

    quality = report["arms"]["fusion-quality-aware"]
    assert set(quality["steps"]) == {"build"}
    assert quality["reuse"]["dispositions"]["sensor_association"] == "provided"
    association = baseline["artifacts"]["sensor_association"]
    assert quality["reuse"]["association"]["consumed"] == association
    assert quality["build"]["context_run_ids"] == baseline["build"]["context_run_ids"]
    assert quality["build"]["context_map_id"] != baseline["build"]["context_map_id"]
    assert maps["arms"]["fusion-quality-aware"]["branch"] == _branch(BASELINE)

    dense = report["arms"]["association-dinov3-dense"]
    assert set(dense["steps"]) == {"association_run"}
    assert dense["reuse"]["dispositions"]["sensor_association"] == "executed"
    assert dense["reuse"]["dispositions"]["point_representation"] == "executed"
    assert dense["materialization"]["validation"]["status"] == "verified"
    assert maps["arms"]["association-dinov3-dense"]["branch"] is None
    assert (
        maps["arms"]["association-dinov3-dense"]["paths"]["association_run"]
        != maps["arms"][BASELINE]["paths"]["association_run"]
    )

    scoring = report["arms"]["scoring-clip"]
    assert set(scoring["steps"]) == {"semantic_scoring"}
    assert scoring["reuse"]["dispositions"]["semantic_scoring"] == "executed"
    assert scoring["scoring"]["integrity_problems"] == []
    assert scoring["scoring"]["text_template"] == "a photo of a {hypothesis}."
    # O scoring não entra no mapa (nenhuma política de fusão consome SemanticScore, #180).
    baseline_map = maps["arms"][BASELINE]["context_map_id"]
    assert maps["arms"]["scoring-clip"]["context_map_id"] == baseline_map


def test_the_index_grows_after_each_arm_and_every_read_of_it_is_valid(
    dry_run: dict[str, Any],
) -> None:
    reader, maps = dry_run["reader"], dry_run["maps"]
    harness = dry_run["campaign"].harness
    workspace = dry_run["campaign"].workspace

    assert reader.invalid == []
    assert reader.arm_counts == sorted(reader.arm_counts)
    # O leitor viu o índice crescer, braço a braço, enquanto a campanha rodava.
    assert len(set(reader.arm_counts)) > 2
    assert reader.arm_counts[-1] == len(ARMS)
    assert list(maps["arms"]) == list(ARMS)
    assert sorted(path.stem for path in (harness / "maps").glob("*.json")) == sorted(ARMS)
    assert [path.name for path in harness.rglob(".tmp-*")] == []
    for arm in ARMS:
        entry = maps["arms"][arm]
        assert _read(harness / "maps" / f"{arm}.json") == entry
        for key, location in entry["paths"].items():
            if location is None:
                continue
            assert not Path(location).is_absolute(), (arm, key)
            # O mapa de um build fica no maps root, irmão do run store; o de um run, no workspace.
            roots = (workspace, workspace.parent / "maps") if key == "context_map" else (workspace,)
            assert any((root / location).is_dir() for root in roots), (arm, key)
        if arm in PERCEPTION_ARMS:
            assert entry["branch"] == _branch(arm)
            assert entry["paths"]["point_representation_run"] is not None
    context_maps = {maps["arms"][arm]["context_map_id"] for arm in ARMS if arm != "scoring-clip"}
    assert len(context_maps) == len(ARMS) - 1


def test_the_voxel_layers_are_derived_once_from_the_foundation_map_before_the_arms(
    dry_run: dict[str, Any],
) -> None:
    report, maps, protocol = dry_run["report"], dry_run["maps"], dry_run["protocol"]
    harness = dry_run["campaign"].harness

    voxel = report["voxel_layers"]
    assert voxel["status"] == "complete"
    assert voxel["step"]["returncode"] == 0
    assert voxel["step"]["wall_time_seconds"] > 0 and voxel["step"]["peak_rss_mb"] > 0
    assert protocol["voxel_layers"]["cells_m"] == [0.05, 0.10]
    assert maps["voxel_layers"]["paths_relative_to"] == "harness"
    layers = maps["voxel_layers"]["layers"]
    assert sorted(layers) == ["0.05", "0.1"]
    for location in layers.values():
        assert not Path(location).is_absolute()
        assert (harness / location / "manifest.json").is_file()
    # A camada voxel é da geometria da fundação, a mesma de todo braço.
    foundation = protocol["foundation"]["geometry"]["artifact_id"]
    assert voxel["source_map_run_id"] == foundation


def test_the_protocol_records_the_window_selection_and_every_parameter_origin(
    dry_run: dict[str, Any],
) -> None:
    protocol = dry_run["protocol"]

    selection = protocol["selection"]
    assert selection["image_stride"] == 6
    assert selection["window_images"] == 3
    assert selection["observation_ids"] == ["frame-0000"]
    assert selection["observation_selection"] == {
        "kind": "explicit_ids",
        "observation_ids": ["frame-0000"],
    }
    window = protocol["window"]
    assert window["document"]["clock_id"] == "fixture:header"
    # Os campos que o #669 acrescentou ao contrato são aceitos e gravados como vieram.
    sequence = dry_run["campaign"].sequence_artifact_id
    assert window["document"]["canonical_sequence_artifact_id"] == sequence
    assert window["document"]["end_recording_ns"] == 3_000_000_000
    assert window["document"]["offset_correction"]["anchors"]["source_time"]["seconds"] == 0
    assert window["accept_window_gaps"] is False
    assert len(window["sha256"]) == 64
    parameters = {item["name"]: item for item in protocol["parameters"]}
    assert parameters["context_margin_ratio"]["value"] == 0.5
    assert parameters["mask_fill_rgb"]["value"] == [0, 0, 0]
    assert parameters["quality_neutral_factor"]["value"] == 0.5
    assert parameters["quality_definitions_version"]["value"] == "observation-quality-v3"
    assert parameters["dense_interpolation"]["value"] == "bilinear"
    assert parameters["eagle_max_dynamic_tiles"]["value"] == 6
    assert parameters["florence2_task"]["value"] == "<REGION_TO_CATEGORY>"
    assert parameters["clip_scoring_template"]["value"] == "a photo of a {hypothesis}."
    assert "s04_fusion.py" in parameters["quality_ramps"]["origin"]
    assert all(item["origin"] for item in protocol["parameters"])
    by_arm = {arm["arm"]: arm for arm in protocol["arms"]}
    assert by_arm[REPLICA]["differences_from_baseline"] == []
    for arm in protocol["arms"]:
        for difference in arm["differences_from_baseline"]:
            assert difference["declared_by"], difference


# --- a failure mid-campaign --------------------------------------------------------------


def test_a_failed_arm_stops_the_campaign_with_a_valid_index_of_the_arms_before_it(
    tmp_path: Path,
) -> None:
    campaign = _Campaign(tmp_path)
    arguments = campaign.arguments(dry_run=False)
    # Sem a feature densa no dublê, o braço do canal denso (o terceiro) falha na associação.
    arguments += ["--dry-run-executors", "chain:replicable_dry_run_executors"]

    result = run_script(SCRIPTS / "run_campaign.py", *arguments)

    assert result.returncode == 1, result.stdout + result.stderr
    report = _read(campaign.harness / "report.json")
    maps = _read(campaign.harness / "maps.json")
    assert report["status"] == "failed"
    assert report["failure"]["arm"] == "association-dinov3-dense"
    assert "dense" in report["failure"]["message"]
    assert list(maps["arms"]) == ["sam2-region-v1", "fusion-quality-aware"]
    assert {report["arms"][arm]["status"] for arm in ARMS[3:]} == {"not_run"}
    assert report["criterion"]["passes"] is False
    for name in ("protocol.json", "report.json", "maps.json"):
        assert str(tmp_path) not in (campaign.harness / name).read_text(), name


def test_an_eagle_failure_costs_only_the_eagle_arm(tmp_path: Path) -> None:
    campaign = _Campaign(tmp_path)
    # Uma branch com o nome da do Eagle já existe: o primeiro passo dele falha, como o último.
    create_branch(
        campaign.workspace,
        dataset=CI_FIXTURE_ID,
        name=_branch(EAGLE),
        foundation=foundation_of_run(campaign.workspace, campaign.foundation),
    )

    code, output = campaign.run()

    assert code == 1, output
    report = _read(campaign.harness / "report.json")
    maps = _read(campaign.harness / "maps.json")
    assert report["failure"]["arm"] == EAGLE
    assert report["arms"][EAGLE]["status"] == "failed"
    # Todo outro braço terminou e está no índice; a concordância da réplica não se perde.
    assert list(maps["arms"]) == list(ARMS[:-1])
    assert {report["arms"][arm]["status"] for arm in ARMS[:-1]} == {"complete"}
    assert report["replica_agreement"] is not None
    assert report["replica_agreement"]["compared"] == 2


# --- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"sequence_artifact_id": "another-sequence"}, "another-sequence"),
        ({"start_ns": 5_000_000_000, "end_ns": 6_000_000_000}, "no camera image"),
        ({"clock_id": "another-clock"}, "another-clock"),
        ({"schema": "contextmap.corridor-window/v0"}, "contextmap.corridor-window/v1"),
    ],
    ids=["another-sequence", "no-image", "another-clock", "another-schema"],
)
def test_a_window_the_foundation_cannot_serve_is_refused_before_anything_exists(
    tmp_path: Path, changes: dict[str, Any], expected: str
) -> None:
    campaign = _Campaign(tmp_path)
    window = _window(tmp_path, campaign.sequence_artifact_id, **changes)

    code, output = campaign.run(window=window)

    assert code == 2, output
    assert expected in output
    assert not campaign.harness.parent.exists()


@pytest.mark.parametrize(
    "gaps",
    [
        {"max_pose_gap_ms": 900.0, "within_thresholds": False},
        # Com menos de duas amostras o maior intervalo não é medido (null), e a janela não passa.
        {"max_pose_gap_ms": None, "max_lidar_gap_ms": None, "within_thresholds": False},
    ],
    ids=["pose-gap-over-threshold", "gaps-not-measured"],
)
def test_a_window_outside_the_gap_thresholds_is_refused_unless_explicitly_accepted(
    tmp_path: Path, gaps: dict[str, Any]
) -> None:
    campaign = _Campaign(tmp_path)
    window = _window(tmp_path, campaign.sequence_artifact_id, quality=gaps)

    code, output = campaign.run(window=window)

    assert code == 2, output
    assert "--accept-window-gaps" in output
    assert not campaign.harness.parent.exists()


def test_accepted_window_gaps_are_recorded_in_the_protocol(tmp_path: Path) -> None:
    campaign = _Campaign(tmp_path)
    gaps = {"max_pose_gap_ms": 900.0, "within_thresholds": False}
    window = _window(tmp_path, campaign.sequence_artifact_id, quality=gaps)
    script = checkout_copy(tmp_path, EXPERIMENT, imports=(PILOT, VOXEL)) / CAMPAIGN

    # Um run real sem providers para no primeiro `context run`, depois das camadas voxel.
    code, output = campaign.run(
        "--accept-window-gaps", script=script, window=window, code_identity=None, dry_run=False
    )

    assert code == 1, output
    protocol = _read(campaign.harness / "protocol.json")
    assert protocol["window"]["accept_window_gaps"] is True
    assert protocol["window"]["document"]["quality"]["within_thresholds"] is False
    report = _read(campaign.harness / "report.json")
    assert report["failure"]["arm"] == BASELINE and report["failure"]["step"] == "context-run"
    assert "resources.providers" in report["failure"]["hint"]
    assert report["voxel_layers"]["status"] == "complete"
    assert _read(campaign.harness / "maps.json")["arms"] == {}


def test_a_real_run_from_a_dirty_checkout_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    campaign = _Campaign(tmp_path)
    checkout = checkout_copy(tmp_path, EXPERIMENT, imports=(PILOT, VOXEL))
    with (checkout / "experiments" / EXPERIMENT / "scripts" / "run_campaign.py").open("a") as file:
        file.write("# a local change\n")

    code, output = campaign.run(script=checkout / CAMPAIGN, code_identity=None, dry_run=False)

    assert code == 2, output
    assert "dirty" in output and "commit or stash" in output
    assert not campaign.harness.parent.exists()


def test_a_real_run_from_a_clean_checkout_records_the_full_sha_of_its_commit(
    tmp_path: Path,
) -> None:
    campaign = _Campaign(tmp_path)
    checkout = checkout_copy(tmp_path, EXPERIMENT, imports=(PILOT, VOXEL))
    head = run_git(checkout, "rev-parse", "HEAD")

    code, output = campaign.run(script=checkout / CAMPAIGN, code_identity=head[:12], dry_run=False)

    assert "commit or stash" not in output
    assert code == 1, output
    assert _read(campaign.harness / "protocol.json")["code_identity"] == head


def test_a_second_round_never_overwrites_the_first(tmp_path: Path) -> None:
    campaign = _Campaign(tmp_path)
    campaign.harness.mkdir(parents=True)

    code, output = campaign.run()

    assert code == 2
    assert "already exists" in output


def test_a_template_placeholder_left_in_an_arm_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    campaign = _Campaign(tmp_path)

    code, output = campaign.run(arms=ARMS_DIR)

    assert code == 2, output
    assert "interpreter-eagle2-5" in output and "REPLACE" in output
    assert not campaign.harness.parent.exists()


def test_an_arm_that_varies_an_undeclared_field_is_refused_before_anything_exists(
    tmp_path: Path,
) -> None:
    campaign = _Campaign(tmp_path)
    arms = _filled_arms(tmp_path)
    layer = arms / "association-dinov3-dense.json"
    document = _read(layer)
    # O braço do canal denso não pode mudar também a fusão.
    document["components"]["semantic_fusion"] = {
        "support": {
            "backend": "geometry-jaccard-support-v1",
            "geometry-jaccard-support-v1": {"min_geometry_count": 2, "min_overlap": 0.5},
        }
    }
    _write(layer, document)

    code, output = campaign.run(arms=arms)

    assert code == 2, output
    assert "association-dinov3-dense" in output and "min_geometry_count" in output
    assert not campaign.harness.parent.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("revision", "1" * 40), ("checkpoint", "openai/clip-vit-base-patch32")],
)
def test_a_scorer_outside_the_space_of_the_region_features_is_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    campaign = _Campaign(tmp_path)
    arms = _filled_arms(tmp_path)
    layer = arms / "scoring-clip.json"
    document = _read(layer)
    document["components"]["semantic_scoring"]["scorer"]["clip"][field] = value
    _write(layer, document)

    code, output = campaign.run(arms=arms)

    # O scorer só compara no espaço das features de região que pontua: recusado antes de tudo.
    assert code == 2, output
    assert "scoring-clip" in output and field in output and value in output
    assert not campaign.harness.parent.exists()


def test_a_parameter_that_differs_from_its_registered_value_is_refused(tmp_path: Path) -> None:
    campaign = _Campaign(tmp_path)
    arms = _filled_arms(tmp_path)
    layer = arms / "qwen-view-contextual-crop.json"
    document = _read(layer)
    view = document["components"]["semantic_interpretation"]["interpreter"]["qwen"]
    view["view_policy"]["context_margin_ratio"] = 0.25
    _write(layer, document)

    code, output = campaign.run(arms=arms)

    assert code == 2, output
    assert "context_margin_ratio" in output and "0.25" in output
    assert not campaign.harness.parent.exists()


# --- the window selection ----------------------------------------------------------------


def test_the_selection_takes_every_sixth_image_of_the_window_from_the_first() -> None:
    window = _module("campaign_window")
    images = [(f"frame-{index:04d}", 1_000 * index) for index in range(20)]

    selected = window.every_nth_in_window(images, start_ns=2_000, end_ns=16_000, stride=6)

    # [2 000, 16 000) guarda os frames 2..15; a cada seis, a partir do primeiro.
    assert selected.window_images == 14
    assert selected.observation_ids == ("frame-0002", "frame-0008", "frame-0014")


def test_the_selection_excludes_the_end_of_the_window_and_refuses_an_empty_one() -> None:
    window = _module("campaign_window")
    images = [(f"frame-{index:04d}", 1_000 * index) for index in range(3)]

    assert window.every_nth_in_window(
        images, start_ns=0, end_ns=2_000, stride=6
    ).observation_ids == ("frame-0000",)
    with pytest.raises(ValueError, match="no camera image"):
        window.every_nth_in_window(images, start_ns=2_001, end_ns=3_000, stride=6)


# --- the viewer index --------------------------------------------------------------------


def test_the_index_holds_every_arm_published_so_far(tmp_path: Path) -> None:
    index = _module("campaign_index").ViewerIndex(tmp_path / "harness", schema="test/v1")

    index.publish_voxel_layers({"0.05": "voxel/B"})
    for count, arm in enumerate(("a", "b", "c"), start=1):
        index.publish_arm(arm, {"branch": arm, "context_map_id": f"map-{arm}", "paths": {}})

        document = _read(tmp_path / "harness" / "maps.json")
        assert list(document["arms"]) == ["a", "b", "c"][:count]
        assert document["voxel_layers"]["layers"] == {"0.05": "voxel/B"}
    assert [path.name for path in (tmp_path / "harness").rglob(".tmp-*")] == []


def test_a_write_interrupted_before_its_rename_leaves_the_previous_index_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("campaign_index")
    index = module.ViewerIndex(tmp_path / "harness", schema="test/v1")
    index.publish_arm("a", {"branch": "a", "context_map_id": "map-a", "paths": {}})

    def interrupted(source: Any, target: Any) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        index.publish_arm("b", {"branch": "b", "context_map_id": "map-b", "paths": {}})

    assert list(_read(tmp_path / "harness" / "maps.json")["arms"]) == ["a"]
    assert [path.name for path in (tmp_path / "harness").rglob(".tmp-*")] == []


def test_an_arm_entry_is_never_published_twice(tmp_path: Path) -> None:
    index = _module("campaign_index").ViewerIndex(tmp_path / "harness", schema="test/v1")
    index.publish_arm("a", {"branch": "a", "context_map_id": "map-a", "paths": {}})

    with pytest.raises(FileExistsError):
        index.publish_arm("a", {"branch": "a", "context_map_id": "map-x", "paths": {}})

    assert _read(tmp_path / "harness" / "maps" / "a.json")["context_map_id"] == "map-a"


# --- the matrix and its configuration ----------------------------------------------------


def _matrix() -> Any:
    return _module("campaign_matrix").read_matrix(CONFIG / "campaign.json")


def _templates() -> dict[str, Any]:
    matrix = _matrix()
    arms = _module("pilot_arms")
    return {
        arm.slug: arms.resolve_arm(BASE_TEMPLATE, (POINT_TEMPLATE,), ARMS_DIR / arm.layer)
        for arm in matrix.arms
    }


def test_the_matrix_runs_the_baseline_first_eagle_last_and_reuses_only_the_baseline() -> None:
    matrix = _matrix()

    assert [arm.slug for arm in matrix.arms] == list(ARMS)
    assert matrix.baseline == BASELINE and matrix.replica == REPLICA
    assert {arm.slug: arm.execution for arm in matrix.arms if arm.reuse is not None} == {
        "fusion-quality-aware": "downstream_build",
        "association-dinov3-dense": "association_rerun",
        "scoring-clip": "semantic_scoring",
    }
    assert {arm.reuse for arm in matrix.arms if arm.reuse is not None} == {BASELINE}


def test_the_pilot_arms_and_variables_are_the_pilot_s_own() -> None:
    matrix = _matrix()
    pilot = _module("pilot_arms")

    for arm in PILOT_ARMS:
        assert (ARMS_DIR / f"{arm}.json").read_bytes() == (
            PILOT_CONFIG / "arms" / f"{arm}.json"
        ).read_bytes(), arm
    records = {variable.name: variable.to_record() for variable in matrix.variables}
    for variable in pilot.VARIABLES:
        assert records[variable.name] == variable.to_record()
    assert _read(BASE_TEMPLATE) == _read(PILOT_CONFIG / "base.template.json")
    assert _read(POINT_TEMPLATE) == _read(PILOT_CONFIG / "point-representation.template.json")


def test_every_arm_layer_differs_from_the_baseline_only_in_its_declared_variables() -> None:
    matrix = _matrix()

    differences = _module("campaign_matrix").check_matrix(matrix, _templates())

    declared = {
        arm: {name for item in found for name in item.declared_by}
        for arm, found in differences.items()
    }
    assert declared == {
        arm.slug: {name for name, level in arm.assignments if level != matrix.baseline_level(name)}
        for arm in matrix.arms
    }
    assert declared["scoring-clip"] == {"semantic_scoring_stage", "semantic_scorer"}
    assert declared[REPLICA] == set()


def test_every_registered_parameter_is_the_value_its_arm_resolves() -> None:
    matrix = _matrix()

    checked = _module("campaign_matrix").check_parameters(matrix, _templates())

    assert [item["name"] for item in checked] == [item.name for item in matrix.parameters]


def test_the_scorer_template_names_the_clip_of_the_region_features() -> None:
    matrix = _matrix()

    checked = _module("campaign_matrix").check_scorer_space(matrix, _templates())

    assert checked == {
        "scoring-clip": {
            "backend": "clip",
            "checkpoint": CLIP_CHECKPOINT,
            "revision": "REPLACE-WITH-THE-40-HEX-CLIP-REVISION",
            "region_features_of": BASELINE,
        }
    }


def test_each_qwen_view_arm_resolves_the_one_view_it_names() -> None:
    views = {
        "qwen-view-contextual-crop": SemanticViewPolicy(
            region_views=(VisualViewKind.CONTEXTUAL_CROP,), context_margin_ratio=0.5
        ),
        "qwen-view-masked-subject": SemanticViewPolicy(
            region_views=(VisualViewKind.MASKED_SUBJECT,), mask_fill_rgb=(0, 0, 0)
        ),
    }
    templates = _templates()

    for arm, view in views.items():
        assert resolve_semantic_request_policy(templates[arm]) == (
            SemanticRequestPolicy.from_prompt_policy(
                SemanticPromptPolicy(scene="scene/v1", region="region/v1"), views=view
            )
        ), arm


def _fake_runtime(config: Any, secrets: Any) -> object:
    """Stand in for a model runtime a provider would load; composition never calls it."""
    return object()


def _filled(tmp_path: Path, *sources: Path) -> list[Path]:
    return [
        _write(tmp_path / f"{index}-{source.name}", machine_values(_read(source)))
        for index, source in enumerate(sources)
    ]


@pytest.mark.parametrize(
    "arm", ["interpreter-eagle2-5", "interpreter-florence2-region-category", *PILOT_ARMS[:1]]
)
def test_every_new_interpreter_arm_composes_its_perception_once_filled(
    arm: str, tmp_path: Path
) -> None:
    failures: dict[str, Exception] = {}

    composed = compose_executors(
        resolve_effective_config(files=_filled(tmp_path, BASE_TEMPLATE, ARMS_DIR / f"{arm}.json")),
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


def test_the_reuse_arms_compose_the_one_stage_each_recomputes(tmp_path: Path) -> None:
    def composed(arm: str, stage: str) -> Any:
        files = _filled(tmp_path / arm, BASE_TEMPLATE, ARMS_DIR / f"{arm}.json")
        return compose(
            resolve_effective_config(files=files),
            providers={SCORING: _fake_runtime},
            stages=[stage],
            environ={},
            module_available=lambda _name: True,
        )

    quality = composed("fusion-quality-aware", "semantic_fusion").accumulation_policy
    dense = composed("association-dinov3-dense", "sensor_association")
    scoring = composed("scoring-clip", "semantic_scoring").semantic_scoring

    assert isinstance(quality, QualityAwareAccumulationPolicy)
    assert quality.neutral_factor == 0.5
    assert [(ramp.quality_input.value, ramp.good, ramp.bad) for ramp in quality.ramps] == [
        ("support_depth_median_m", 4.0, 30.0),
        ("border_distance_median_px", 60.0, 10.0),
    ]
    assert dense.association_dense_sampling is InterpolationPolicy.BILINEAR
    assert scoring is not None and scoring.text_template == "a photo of a {hypothesis}."


def test_reuse_checks_all_three_independent_producer_identities() -> None:
    from types import SimpleNamespace

    module = _module("run_campaign")
    campaign = module._Campaign.__new__(module._Campaign)
    producers = {
        stage: {
            "stage_id": stage,
            "contract": contract,
            "artifact_id": f"{stage}-run",
            "content_hash": "sha256:" + "a" * 64,
            "location": f"sequence/run-0001/{stage}",
        }
        for stage, contract in (
            ("region_perception", "RegionPerceptionRunArtifact"),
            ("visual_perception", "VisualEvidenceArtifact"),
            ("semantic_interpretation", "SemanticInterpretationArtifact"),
        )
    }
    source = module._Arm(branch="baseline", artifacts=producers, entry={})
    arm = SimpleNamespace(slug="reuse", reuse="baseline")
    consumed = {stage: [ref] for stage, ref in producers.items()}
    dispositions = dict.fromkeys(producers, "provided")
    record = campaign._reuse(arm, source, consumed=consumed, dispositions=dispositions)
    assert record["holds"] is True
    for key in ("regions", "visual_evidence", "semantic_interpretation"):
        assert record[key]["same_artifact"] is True

    foreign = {**producers["semantic_interpretation"], "artifact_id": "other"}
    with pytest.raises(module.StepFailed, match="exactly the artifacts"):
        campaign._reuse(
            arm,
            source,
            consumed={**consumed, "semantic_interpretation": [foreign]},
            dispositions=dispositions,
        )
