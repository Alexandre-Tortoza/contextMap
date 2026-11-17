"""ContextBuild: one frozen materialization of a branch revision."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from runtime_context import (
    FRAMES,
    LATER,
    PROVIDED_RUNTIMES,
    ContextWorkspace,
    document,
    foundation,
)

from contextmap.runtime import (
    ContextBranch,
    ContextBuild,
    ContextBuildError,
    ExecutionRecord,
    RunJournal,
    append_to_branch,
    create_branch,
    plan_context_build,
    publish_context_build,
    publish_context_run,
    read_context_build,
    read_context_run,
    run_plan,
)
from contextmap.runtime.contextual_mapping.stages import (
    CONTEXT_STAGES,
    NEVER_RECOMPUTED_BY_A_BUILD,
)

MATERIALIZATION = [
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
    "context_map",
]


class _Builds:
    def __init__(
        self, tmp_path: Path, *, name: str = "corridor", context: ContextWorkspace | None = None
    ) -> None:
        """One branch over the default foundation; ``context`` shares a workspace and its reuse."""
        self.context = context or ContextWorkspace(tmp_path)
        self.workspace = self.context.workspace
        self.branch: ContextBranch = create_branch(
            self.workspace, dataset="S1", name=name, foundation=foundation()
        )

    def append(
        self,
        selection: dict[str, Any],
        *,
        point_representation: bool = False,
        representation_radius_m: float | None = None,
    ) -> None:
        self.branch = append_to_branch(
            self.workspace,
            self.branch,
            self.context.run(
                selection=selection,
                point_representation=point_representation,
                representation_radius_m=representation_radius_m,
            ),
        )

    def append_perception_only(self, selection: dict[str, Any]) -> None:
        """Append a ContextRun that holds perception and no association, as a partial run would."""
        effective, plan = self.context.plan(document(selection))
        base = foundation()
        execution = plan.scope(targets=["visual_perception"], provided={"ingestion": base.sequence})
        journal = RunJournal.create(self.workspace, effective, execution)
        record = run_plan(
            execution,
            self.context.executors(plan),
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=PROVIDED_RUNTIMES,
            journal=journal,
        )
        context = publish_context_run(
            journal.directory,
            workspace=self.workspace,
            foundation=base,
            execution=execution,
            record=record,
        )
        self.branch = append_to_branch(self.workspace, self.branch, context)

    def plan(self, config: dict[str, Any] | None = None, **options: Any) -> Any:
        _, plan = self.context.plan(config)
        return plan_context_build(
            self.workspace, self.branch, plan, code_identity="code-1", **options
        )

    def materialize(
        self, config: dict[str, Any] | None = None, **options: Any
    ) -> tuple[ContextBuild, ExecutionRecord, Path]:
        effective, plan = self.context.plan(config)
        planned = plan_context_build(
            self.workspace, self.branch, plan, code_identity="code-1", **options
        )
        journal = RunJournal.create(self.workspace, effective, planned.execution)
        publish_context_build(journal.directory, workspace=self.workspace, build=planned.build)
        record = run_plan(
            planned.execution,
            self.context.executors(plan),
            environ={},
            module_available=lambda _name: True,
            provided_runtimes=PROVIDED_RUNTIMES,
            reuse=self.context.reuse(),
            journal=journal,
        )
        return planned.build, record, journal.directory


class TestFreezing:
    def test_a_build_runs_the_materialization_over_its_frozen_context_runs(
        self, tmp_path: Path
    ) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)

        build, record, _ = builds.materialize()

        assert list(record.order) == MATERIALIZATION
        assert build.branch == "corridor" and build.revision == 1
        assert build.context_run_ids == builds.branch.context_run_ids
        assert set(record.reused) == {
            "ingestion",
            "geometric_mapping",
            "region_perception",
            "visual_perception",
            "semantic_interpretation",
            "sensor_association",
        }

    def test_several_context_runs_feed_the_fusion_as_separate_evidence(
        self, tmp_path: Path
    ) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        builds.append(LATER)

        build = builds.plan().build

        assert len(build.context_run_ids) == 2
        assert len(build.provided["sensor_association"]) == 2
        assert len(build.provided["visual_perception"]) == 2

    def test_the_same_frozen_input_is_the_same_build(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)

        assert builds.plan().build.identity == builds.plan().build.identity

    def test_a_downstream_policy_change_is_another_build_over_the_same_evidence(
        self, tmp_path: Path
    ) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)

        first = builds.plan().build
        changed = builds.plan(document(min_overlap=0.6)).build

        assert changed.identity != first.identity
        assert changed.provided == first.provided

    def test_a_later_append_never_changes_a_frozen_revision(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        frozen = builds.plan().build

        builds.append(LATER)

        assert builds.plan(revision=1).build == frozen
        assert builds.plan().build.identity != frozen.identity

    def test_the_order_of_an_explicit_subset_does_not_matter(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        builds.append(LATER)
        ids = builds.branch.context_run_ids

        forward = builds.plan(context_run_ids=ids).build
        backward = builds.plan(context_run_ids=tuple(reversed(ids))).build

        assert forward == backward

    def test_a_build_never_recomputes_context_evidence_its_runs_lack(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append_perception_only(FRAMES)

        with pytest.raises(ContextBuildError, match="sensor_association"):
            builds.plan()

    def test_a_build_never_recomputes_pose_ingestion_nor_a_foundation_stage(self) -> None:
        # pose_ingestion não é membro da fundação, mas está a montante dela: um build nunca o
        # recomputa.
        upstream = {"ingestion", "pose_ingestion", "state_estimation", "geometric_mapping"}

        assert upstream | set(CONTEXT_STAGES) <= NEVER_RECOMPUTED_BY_A_BUILD

    def test_a_representation_nothing_consumes_is_neither_provided_nor_recomputed(
        self, tmp_path: Path
    ) -> None:
        # O run de Point Representation fica na ContextRun; a fusão não o consome,
        # então o build não o fornece, não o grava na entrada congelada e não o recomputa.
        builds = _Builds(tmp_path)
        builds.append(FRAMES, point_representation=True)
        member = builds.branch.members[0]
        recorded = read_context_run(builds.workspace / member.run, workspace=builds.workspace)
        assert "point_representation" in {artifact.ref.stage_id for artifact in recorded.artifacts}

        build, record, _ = builds.materialize(document(point_representation=True))

        assert "point_representation" not in build.provided
        assert list(record.order) == MATERIALIZATION
        assert "point_representation" not in record.reused

    def test_evidence_the_map_does_not_consume_never_changes_the_build(
        self, tmp_path: Path
    ) -> None:
        # A ContextRun também guarda evidência que o mapa não consome (a representação). Mesma
        # percepção e associação sob outra representação é outra ContextRun, mas o mesmo build:
        # as ContextRuns ficam só como proveniência.
        near = _Builds(tmp_path, name="near")
        far = _Builds(tmp_path, name="far", context=near.context)
        near.append(FRAMES, point_representation=True)
        far.append(FRAMES, point_representation=True, representation_radius_m=1.0)

        first, _, first_run = near.materialize()
        second, record, second_run = far.materialize()

        # A materialização é a mesma: cada estágio de jusante do segundo build é reusado.
        assert [
            (stage.stage_id, stage.decision.kind if stage.decision else None)
            for stage in record.stages
        ] == [(stage_id, "reused") for stage_id in MATERIALIZATION]
        assert first.context_run_ids != second.context_run_ids
        assert first.provided == second.provided
        assert first.identity == second.identity
        workspace = near.workspace
        assert read_context_build(first_run, workspace=workspace).context_run_ids == (
            near.branch.context_run_ids
        )
        assert read_context_build(second_run, workspace=workspace).context_run_ids == (
            far.branch.context_run_ids
        )

    def test_a_branch_without_representation_builds_with_the_stage_enabled(
        self, tmp_path: Path
    ) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)

        with_stage = builds.plan(document(point_representation=True)).build

        assert with_stage.provided == builds.plan().build.provided

    def test_a_build_needs_at_least_one_context_run(self, tmp_path: Path) -> None:
        with pytest.raises(ContextBuildError, match="at least one"):
            _Builds(tmp_path).plan()

    def test_only_members_of_the_frozen_revision_can_be_selected(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        builds.append(LATER)
        later = builds.branch.members[1].context_run_id

        with pytest.raises(ContextBuildError, match="revision 1"):
            builds.plan(revision=1, context_run_ids=(later,))

    def test_a_member_whose_record_changed_is_refused(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        record = builds.workspace / "S1" / "run-0001" / "context_run.json"
        document_ = json.loads(record.read_text("utf-8"))
        document_["artifacts"][0]["content_hash"] = "sha256:other"
        record.write_text(json.dumps(document_), encoding="utf-8")

        with pytest.raises(ContextBuildError, match="run-0001"):
            builds.plan()


class TestTheRecord:
    def test_it_is_written_before_the_materialization_and_reads_back(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)

        build, _, directory = builds.materialize()

        assert read_context_build(directory, workspace=builds.workspace) == build

    def test_moving_the_workspace_keeps_it_readable(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        build, _, directory = builds.materialize()
        moved = tmp_path / "moved"
        shutil.copytree(builds.workspace, moved)

        assert read_context_build(moved / "S1" / directory.name, workspace=moved) == build

    def test_it_is_published_once(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        build, _, directory = builds.materialize()

        with pytest.raises(FileExistsError):
            publish_context_build(directory, workspace=builds.workspace, build=build)

    def test_an_altered_record_is_refused(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        _, _, directory = builds.materialize()
        path = directory / "context_build.json"
        document_ = json.loads(path.read_text("utf-8"))
        document_["provided"]["sensor_association"][0]["content_hash"] = "sha256:other"
        path.write_text(json.dumps(document_), encoding="utf-8")

        with pytest.raises(ContextBuildError, match="identity"):
            read_context_build(directory, workspace=builds.workspace)

    def test_a_record_copied_into_another_run_is_refused(self, tmp_path: Path) -> None:
        builds = _Builds(tmp_path)
        builds.append(FRAMES)
        _, _, directory = builds.materialize()
        other = directory.parent / "run-0099"
        other.mkdir()
        shutil.copy(directory / "context_build.json", other / "context_build.json")

        with pytest.raises(ContextBuildError, match=directory.name):
            read_context_build(other, workspace=builds.workspace)
