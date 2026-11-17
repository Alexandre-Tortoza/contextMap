"""A ContextRun explicitly reuses selected regional, visual and semantic producers.

Branches over different geometric maps reuse immutable inference evidence. Each selected
producer must cover the foundation sequence and ContextRun selection, and each consumer must
cite the selected regional producer. All three reused artifacts remain in the audit record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_context import FRAMES, LATER, PROVIDED_RUNTIMES, ContextWorkspace, document, foundation
from runtime_perception import publish_selected_evidence

from contextmap.runtime import (
    ArtifactRef,
    ContextRunError,
    RunJournal,
    check_context_selection,
    context_scope,
    publish_context_run,
    run_plan,
)

PRODUCERS = ("region_perception", "visual_perception", "semantic_interpretation")


def _selected(workspace: Path, **overrides: Any) -> dict[str, tuple[ArtifactRef, ...]]:
    """Publish explicit region, visual and semantic evidence for one ContextRun selection."""
    return publish_selected_evidence(workspace, sequence_id="seq-A", selection=FRAMES, **overrides)


class TestTheScope:
    def test_a_selected_perception_is_supplied_and_only_the_association_runs(
        self, tmp_path: Path
    ) -> None:
        _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
        selected = _selected(tmp_path / "ws")

        execution = context_scope(plan, foundation(), selected=selected)

        assert [stage.stage_id for stage in execution.stages] == ["sensor_association"]
        assert all(execution.reused[stage] == selected[stage] for stage in PRODUCERS)
        assert execution.reused["geometric_mapping"] == (foundation().geometry,)

    @pytest.mark.parametrize("stage_id", ["sensor_association", "geometric_mapping"])
    def test_only_the_perception_may_be_selected(self, tmp_path: Path, stage_id: str) -> None:
        _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
        other = ArtifactRef(stage_id=stage_id, contract="Any", artifact_id="x")

        with pytest.raises(ContextRunError, match="may select only"):
            context_scope(plan, foundation(), selected={stage_id: (other,)})


class TestTheSelectionCheck:
    def test_a_perception_over_the_foundation_and_the_selection_is_accepted(
        self, tmp_path: Path
    ) -> None:
        _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
        selected = _selected(tmp_path / "ws")
        execution = context_scope(plan, foundation(), selected=selected)

        check_context_selection(execution, foundation(), workspace=tmp_path / "ws")

    @pytest.mark.parametrize("stage", PRODUCERS)
    def test_a_perception_of_another_observation_selection_is_refused(
        self, tmp_path: Path, stage: str
    ) -> None:
        _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
        selected = _selected(tmp_path / "ws", selections={stage: LATER})
        execution = context_scope(plan, foundation(), selected=selected)

        with pytest.raises(ContextRunError, match="observation selection"):
            check_context_selection(execution, foundation(), workspace=tmp_path / "ws")

    @pytest.mark.parametrize("stage", PRODUCERS)
    def test_a_perception_of_another_sequence_is_refused(self, tmp_path: Path, stage: str) -> None:
        _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
        selected = _selected(tmp_path / "ws", sequences={stage: "seq-B"})
        execution = context_scope(plan, foundation(), selected=selected)

        with pytest.raises(ContextRunError, match="seq-B"):
            check_context_selection(execution, foundation(), workspace=tmp_path / "ws")


class TestTheRecord:
    def _run(self, tmp_path: Path, selected: dict[str, tuple[ArtifactRef, ...]]) -> Any:
        workspace = ContextWorkspace(tmp_path)
        effective, plan = workspace.plan(document(FRAMES))
        over = foundation(geometry="map-B")
        execution = context_scope(plan, over, selected=selected)
        journal = RunJournal.create(workspace.workspace, effective, execution)
        record = run_plan(
            execution,
            workspace.executors(plan),
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=PROVIDED_RUNTIMES,
            journal=journal,
        )
        return journal.directory, over, execution, record

    def test_it_names_the_selected_perception_as_reused_with_its_selection(
        self, tmp_path: Path
    ) -> None:
        selected = _selected(tmp_path / "ws")
        directory, over, execution, record = self._run(tmp_path, selected)

        context = publish_context_run(
            directory,
            workspace=tmp_path / "ws",
            foundation=over,
            execution=execution,
            record=record,
        )

        assert [(item.ref.stage_id, item.disposition) for item in context.artifacts] == [
            ("region_perception", "reused"),
            ("semantic_interpretation", "reused"),
            ("sensor_association", "produced"),
            ("visual_perception", "reused"),
        ]
        assert {
            item.ref.stage_id: item.ref
            for item in context.artifacts
            if item.disposition == "reused"
        } == {stage: refs[0] for stage, refs in selected.items()}
        assert context.observation_selection == FRAMES

    def test_a_record_over_a_mismatched_perception_is_never_published(self, tmp_path: Path) -> None:
        selected = _selected(tmp_path / "ws", selections={"semantic_interpretation": LATER})
        directory, over, execution, record = self._run(tmp_path, selected)

        with pytest.raises(ContextRunError, match="observation selection"):
            publish_context_run(
                directory,
                workspace=tmp_path / "ws",
                foundation=over,
                execution=execution,
                record=record,
            )
        assert not (directory / "context_run.json").exists()


@pytest.mark.parametrize("stage", ("visual_perception", "semantic_interpretation"))
def test_selected_producers_must_cite_the_selected_regional_artifact(
    tmp_path: Path, stage: str
) -> None:
    _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
    selected = _selected(tmp_path / "ws", regional_dependencies={stage: "region-other"})
    execution = context_scope(plan, foundation(), selected=selected)
    with pytest.raises(ContextRunError, match="outside the selected regional evidence"):
        check_context_selection(execution, foundation(), workspace=tmp_path / "ws")


@pytest.mark.parametrize("stage", ("visual_perception", "semantic_interpretation"))
def test_a_selected_consumer_requires_explicit_regional_selection(
    tmp_path: Path, stage: str
) -> None:
    _, plan = ContextWorkspace(tmp_path).plan(document(FRAMES))
    selected = _selected(tmp_path / "ws")
    with pytest.raises(ContextRunError, match="select region_perception explicitly"):
        context_scope(plan, foundation(), selected={stage: selected[stage]})
