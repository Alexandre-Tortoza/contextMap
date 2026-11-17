"""Stage-major perception: one stage over every frame of a run, then its models are released.

A GPU too small for every model of a run at once (#689) needs each stage to run over all frames
before the next stage's model loads, and each model released once its last stage is done. The
run written must be the same one the frame-major schedule writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime_perception import (
    _FakeFeatureExtractor,
    _FakeRegionDiscovery,
    publish_region_run,
)
from runtime_pillow import fake_pillow  # noqa: F401
from test_independent_perception_executors import _sequence

from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.executors import PerceptionSchedule, VisualPerceptionExecutor
from contextmap.visual_perception.visual_features import FeatureScope


class _Discovery(_FakeRegionDiscovery):
    def __init__(self, log: list[tuple[str, str]]) -> None:
        super().__init__()
        self._log = log

    def discover_audited(self, image: Any) -> Any:
        self._log.append(("discovery", "discover_audited"))
        return super().discover_audited(image)

    def release(self) -> None:
        self._log.append(("discovery", "release"))


class _Features(_FakeFeatureExtractor):
    def __init__(
        self, scope: FeatureScope, name: str, log: list[tuple[str, str]], run_id: str
    ) -> None:
        super().__init__(scope, run_id)
        self._name, self._log = name, log

    def extract(self, *args: Any, **kwargs: Any) -> Any:
        self._log.append((self._name, "extract"))
        return super().extract(*args, **kwargs)

    def release(self) -> None:
        self._log.append((self._name, "release"))


def _executor(log: list[tuple[str, str]], schedule: PerceptionSchedule | None) -> Any:
    return VisualPerceptionExecutor(
        dense_features=lambda scope: _Features(FeatureScope.DENSE, "dense", log, str(scope.run_id)),  # type: ignore[arg-type,return-value]
        region_features=lambda scope: _Features(
            FeatureScope.REGION, "region", log, str(scope.run_id)
        ),  # type: ignore[arg-type,return-value]
        perception_schedule=schedule or PerceptionSchedule.FRAME_MAJOR,
    )


def _perceive(
    workspace: Path, executor: Any, *, frames: int = 3, log: list[tuple[str, str]] | None = None
) -> Path:
    output = workspace / "S1" / "run-0002" / "visual_perception"
    sequence: ArtifactRef = _sequence(workspace, frames=frames)
    # A descoberta roda no stage dela, antes desta percepção, e no mesmo log: o que a percepção
    # agenda é a reprodução daquelas regiões, não um modelo de descoberta.
    regions = publish_region_run(
        workspace, sequence, discovery=_Discovery(log if log is not None else [])
    )
    executor.execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence,), "regions": (regions,)},
            components={},
            config_digest="sha256:test",
            output_dir=output,
            workspace=workspace,
            observation_selection=None,
        )
    )
    return output


def _results_without_clock(output: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in (output / "outputs" / "results.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row.pop("created_at", None)
    return rows


@pytest.mark.usefixtures("fake_pillow")
class TestStageMajorPerception:
    def test_every_frame_runs_a_stage_before_the_next_stage_and_its_model_is_released(
        self, tmp_path: Path
    ) -> None:
        log: list[tuple[str, str]] = []

        _perceive(tmp_path, _executor(log, PerceptionSchedule.STAGE_MAJOR), log=log)

        # O stage de região vem primeiro, com a sua descoberta liberada ao fim do próprio run.
        # Depois, dentro da percepção, cada estágio roda em bloco sobre os 3 frames, na ordem
        # topológica do preset, e cada backend é liberado logo depois do seu último estágio. O
        # intérprete serve a cena (antes das features de região) e as regiões (depois), então só
        # é liberado no fim.
        runs: list[tuple[tuple[str, str], int]] = []
        for entry in log:
            if runs and runs[-1][0] == entry:
                runs[-1] = (entry, runs[-1][1] + 1)
            else:
                runs.append((entry, 1))
        assert runs == [
            (("discovery", "discover_audited"), 3),
            (("discovery", "release"), 1),
            (("dense", "extract"), 3),
            (("dense", "release"), 1),
            (("region", "extract"), 3),
            (("region", "release"), 1),
        ]

    def test_the_stage_major_run_writes_the_frame_major_results(self, tmp_path: Path) -> None:
        frame_major = _perceive(tmp_path / "a", _executor([], PerceptionSchedule.FRAME_MAJOR))
        stage_major = _perceive(tmp_path / "b", _executor([], PerceptionSchedule.STAGE_MAJOR))

        assert _results_without_clock(stage_major) == _results_without_clock(frame_major)
        assert sorted(p.name for p in (stage_major / "outputs").iterdir()) == sorted(
            p.name for p in (frame_major / "outputs").iterdir()
        )

    def test_the_default_schedule_is_frame_major_and_releases_nothing(self, tmp_path: Path) -> None:
        log: list[tuple[str, str]] = []

        _perceive(tmp_path, _executor(log, None), frames=2)

        # Frame-major: o segundo frame começa o primeiro estágio dele — as features densas, já
        # que a descoberta saiu para o stage de região — depois da interpretação do primeiro.
        dense = [i for i, entry in enumerate(log) if entry == ("dense", "extract")]
        assert log.index(("region", "extract")) < dense[1]
        assert log.count(("dense", "release")) == log.count(("region", "release")) == 1


def test_the_schedule_is_selected_by_policy_and_reaches_the_executor(tmp_path: Path) -> None:
    from runtime_documents import effective_from, selected_document

    from contextmap.runtime.composition import compose_executors

    document = selected_document()
    document["policies"] = {"perception_schedule": "stage_major"}
    effective = effective_from(tmp_path, document)
    providers = {
        "region_perception.discovery": lambda _config, _secrets: object(),
        "semantic_interpretation.interpreter": lambda _config, _secrets: object(),
    }
    executor = compose_executors(
        effective, providers=providers, module_available=lambda _: True, environ={}
    )["visual_perception"]

    assert effective.config.policies.perception_schedule == "stage_major"
    assert isinstance(executor, VisualPerceptionExecutor)
    assert executor.perception_schedule is PerceptionSchedule.STAGE_MAJOR


def test_an_unknown_schedule_is_refused_with_its_path(tmp_path: Path) -> None:
    from runtime_documents import effective_from, selected_document

    from contextmap.runtime.configuration import ConfigurationError

    document = selected_document()
    document["policies"] = {"perception_schedule": "sideways"}

    with pytest.raises(ConfigurationError, match=r"policies\.perception_schedule"):
        effective_from(tmp_path, document)
