"""Explicit joins of independent producer artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import (
    EvidenceSetError,
    PerceptionEvidenceSelection,
    PerceptionEvidenceSet,
)


@dataclass(frozen=True)
class _Manifest:
    run_id: str
    sequence_artifact_id: str = "sequence-A"
    selection_id: str = "selection-A"
    region_artifact_id: str = "regions-A"
    source_observation_ids: tuple[str, ...] = ("frame-A",)


class _Regions:
    def __init__(self, run_id: str = "regions-A") -> None:
        self.manifest = _Manifest(run_id=run_id)

    def read_regions(self) -> tuple[()]:
        return ()


class _Visual:
    def __init__(self, region_artifact_id: str = "regions-A") -> None:
        self.manifest = _Manifest(run_id="visual-A", region_artifact_id=region_artifact_id)

    def iter_results(self) -> Iterator[Any]:
        return iter(())


class _Semantic:
    def __init__(self, region_artifact_id: str = "regions-A") -> None:
        self.manifest = _Manifest(run_id="semantic-A", region_artifact_id=region_artifact_id)

    def iter_observations(self) -> Iterator[Any]:
        return iter(())


def test_a_visual_producer_from_another_regional_artifact_is_rejected() -> None:
    with pytest.raises(EvidenceSetError, match="region"):
        PerceptionEvidenceSelection(
            regions=cast(Any, _Regions()),
            visual=cast(Any, _Visual("regions-B")),
        )


def test_an_explicit_regional_selection_yields_an_identity_free_view() -> None:
    selection = PerceptionEvidenceSelection(regions=cast(Any, _Regions()))
    evidence = PerceptionEvidenceSet((selection,)).evidence_for(SourceObservationId("frame-A"))
    assert len(evidence.results_by_selection) == 1
    (view,) = evidence.results_by_selection.values()
    assert view.region_artifact_id == "regions-A"
    assert view.source_observation_id == "frame-A"
    assert not hasattr(view, "result_id")
    assert not hasattr(view, "run_id")


def test_selected_visual_producer_must_account_for_its_declared_observations() -> None:
    selection = PerceptionEvidenceSelection(
        regions=cast(Any, _Regions()), visual=cast(Any, _Visual())
    )
    with pytest.raises(EvidenceSetError, match=r"visual.*coverage"):
        PerceptionEvidenceSet((selection,))
