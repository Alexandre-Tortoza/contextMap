"""Build index of a map (branch): the immutable ledger of its materializations.

A ``context build`` publishes its ``ContextMapArtifact`` into the maps root and records one
immutable entry per build under ``branches/<map_id>/builds/``, so the map's materializations are
discoverable by ``map_id`` without scanning runs, and each visible artifact keeps its lineage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextmap.runtime import (
    ArtifactRef,
    ContextBuildEntry,
    ContextBuildError,
    ContextBuildId,
    create_branch,
    read_context_builds,
    record_context_build,
    resolve_map_artifact,
)
from contextmap.runtime.contextual_mapping import (
    ContextBranch,
    SpatialFoundation,
    SpatialFoundationId,
    claim_context_build,
)
from contextmap.runtime.contextual_mapping.build_index import builds_directory


def _foundation() -> SpatialFoundation:
    def ref(stage_id: str, contract: str) -> ArtifactRef:
        return ArtifactRef(
            stage_id=stage_id,
            contract=contract,
            artifact_id=f"{stage_id}-A",
            content_hash=f"sha256:{stage_id}-A",
            location=f"S1/run-0001/{stage_id}",
        )

    return SpatialFoundation(
        identity=SpatialFoundationId("sha256:foundation-A"),
        sequence=ref("ingestion", "SequenceArtifact"),
        state_estimation=ref("state_estimation", "StateEstimationRunArtifact"),
        geometry=ref("geometric_mapping", "GeometricMapArtifact"),
    )


def _branch(workspace: Path, name: str = "mapa_01") -> ContextBranch:
    return create_branch(workspace, dataset="S1", name=name, foundation=_foundation())


def _map_ref(map_id: str, index: int) -> ArtifactRef:
    """A handle to the ContextMapArtifact of a build, located in the maps root."""
    return ArtifactRef(
        stage_id="context_map",
        contract="context_map",
        artifact_id=f"context-map-{map_id}-{index}",
        content_hash=f"sha256:map-{index}",
        location=f"{map_id}/{index:06d}",
    )


def _record(
    workspace: Path, branch: ContextBranch, index: int, *, revision: int
) -> ContextBuildEntry:
    return record_context_build(
        workspace,
        branch,
        index=index,
        context_build_id=ContextBuildId(f"sha256:build-{index}"),
        branch_revision=revision,
        context_map=_map_ref(branch.name, index),
        run=f"S1/run-{index + 1:04d}",
    )


class TestRecording:
    def test_the_first_build_is_index_one(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)

        entry = _record(tmp_path, branch, 1, revision=1)

        assert entry.index == 1
        assert (builds_directory(tmp_path, "S1", "mapa_01") / "000001.json").is_file()

    def test_builds_accumulate_in_order(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        _record(tmp_path, branch, 1, revision=1)
        _record(tmp_path, branch, 2, revision=2)

        entries = read_context_builds(tmp_path, dataset="S1", name="mapa_01")

        assert [entry.index for entry in entries] == [1, 2]
        assert (builds_directory(tmp_path, "S1", "mapa_01") / "000002.json").is_file()

    def test_an_index_is_recorded_once(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        _record(tmp_path, branch, 1, revision=1)

        with pytest.raises(ContextBuildError, match="1"):
            _record(tmp_path, branch, 1, revision=1)

    def test_recording_never_changes_an_earlier_entry(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        _record(tmp_path, branch, 1, revision=1)
        directory = builds_directory(tmp_path, "S1", "mapa_01")
        before = (directory / "000001.json").read_bytes()

        _record(tmp_path, branch, 2, revision=2)

        assert (directory / "000001.json").read_bytes() == before


class TestReading:
    def test_entries_round_trip(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        recorded = _record(tmp_path, branch, 1, revision=1)

        (entry,) = read_context_builds(tmp_path, dataset="S1", name="mapa_01")

        assert entry == recorded
        assert ContextBuildEntry.from_document(entry.to_document()) == entry
        assert entry.context_map.location == "mapa_01/000001"

    def test_no_builds_is_empty(self, tmp_path: Path) -> None:
        _branch(tmp_path)

        assert read_context_builds(tmp_path, dataset="S1", name="mapa_01") == ()

    def test_a_gap_is_refused(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        _record(tmp_path, branch, 1, revision=1)
        _record(tmp_path, branch, 2, revision=2)
        (builds_directory(tmp_path, "S1", "mapa_01") / "000001.json").unlink()

        with pytest.raises(ContextBuildError, match="1"):
            read_context_builds(tmp_path, dataset="S1", name="mapa_01")

    def test_an_altered_entry_is_refused(self, tmp_path: Path) -> None:
        branch = _branch(tmp_path)
        _record(tmp_path, branch, 1, revision=1)
        path = builds_directory(tmp_path, "S1", "mapa_01") / "000001.json"
        document = json.loads(path.read_text("utf-8"))
        document["index"] = 7
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(ContextBuildError, match="000001"):
            read_context_builds(tmp_path, dataset="S1", name="mapa_01")


class TestResolving:
    def test_it_points_at_the_map_in_the_maps_root(self, tmp_path: Path) -> None:
        maps_root = tmp_path / "maps"
        branch = _branch(tmp_path / "store")
        entry = _record(tmp_path / "store", branch, 1, revision=1)

        assert resolve_map_artifact(maps_root, entry) == maps_root / "mapa_01" / "000001"


def test_a_second_build_of_one_map_cannot_claim_an_index_while_the_first_holds_it(
    tmp_path: Path,
) -> None:
    """Allocating a build's ordinal is exclusive per map, so two builds never target one map.

    The ledger is strictly ``1..M``, so two builds that read the same ordinal would both publish
    into ``<maps root>/<map_id>/NNNNNN`` and one of them would lose the directory it had already
    written. The claim serializes them instead, and a refused claim writes nothing.
    """
    branch = _branch(tmp_path)
    maps_root = tmp_path.parent / "maps"

    with claim_context_build(tmp_path, branch, maps_root=maps_root) as claimed:
        assert claimed.index == 1
        assert claimed.map_output == maps_root / branch.name / "000001"
        with (
            pytest.raises(ContextBuildError, match="in progress"),
            claim_context_build(tmp_path, branch, maps_root=maps_root),
        ):
            pass

    assert not (maps_root / branch.name).exists()


def test_a_claim_refuses_an_unrecorded_map_left_by_an_interrupted_build(tmp_path: Path) -> None:
    """A published map that no entry records blocks the ordinal instead of being overwritten.

    The ledger has no gaps, so the next build claims the same ordinal. Removing or replacing a
    published artifact is never automatic: the claim reports it and the operator decides.
    """
    branch = _branch(tmp_path)
    maps_root = tmp_path.parent / "maps"
    (maps_root / branch.name / "000001").mkdir(parents=True)

    with (
        pytest.raises(ContextBuildError, match="unrecorded map"),
        claim_context_build(tmp_path, branch, maps_root=maps_root),
    ):
        pass
