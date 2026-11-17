"""Read-only joins of explicitly selected regional, visual, semantic and scoring evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.semantic_interpretation import SemanticInterpretationRunReader, SemanticObservation
from contextmap.semantic_scoring import SemanticScoringRunReader
from contextmap.visual_perception.models import PerceptionResult
from contextmap.visual_perception.visual_evidence_artifact import (
    VisualEvidenceResult,
    VisualEvidenceRunReader,
)


class EvidenceSetError(ValueError):
    """An explicit producer selection cannot be viewed as one coherent evidence set."""


SelectionKey = tuple[str, str | None, str | None, tuple[str, ...]]


@dataclass(frozen=True, kw_only=True)
class PerceptionEvidenceSelection:
    """The exact producer artifacts to join for one view; no source is discovered implicitly."""

    regions: RegionPerceptionRunReader
    visual: VisualEvidenceRunReader | None = None
    semantic: SemanticInterpretationRunReader | None = None
    scoring: tuple[SemanticScoringRunReader, ...] = ()

    def __post_init__(self) -> None:
        """Reject producers from another sequence, selection, or regional artifact."""
        regional = self.regions.manifest
        for name, reader in (("visual", self.visual), ("semantic", self.semantic)):
            if reader is None:
                continue
            manifest = reader.manifest
            if manifest.region_artifact_id != regional.run_id:
                raise EvidenceSetError(f"{name} evidence references another region artifact")
            if (
                manifest.sequence_artifact_id != regional.sequence_artifact_id
                or manifest.selection_id != regional.selection_id
            ):
                raise EvidenceSetError(f"{name} evidence covers another sequence or selection")
        visual = self.visual
        semantic = self.semantic
        if self.scoring and (visual is None or semantic is None):
            raise EvidenceSetError(
                "scoring evidence requires selected visual and semantic producers"
            )
        ids = [str(reader.manifest.run_id) for reader in self.scoring]
        if len(ids) != len(set(ids)):
            raise EvidenceSetError("duplicate scoring artifact selection")
        if visual is not None and semantic is not None:
            for scorer in self.scoring:
                lineage = scorer.manifest.lineage
                if (
                    lineage.sequence_artifact_id != regional.sequence_artifact_id
                    or lineage.selection_id != regional.selection_id
                    or lineage.region_artifact_id != regional.run_id
                    or lineage.visual_evidence_artifact_id != visual.manifest.run_id
                    or lineage.semantic_interpretation_artifact_id != semantic.manifest.run_id
                ):
                    raise EvidenceSetError("scoring evidence references another producer selection")

    @property
    def key(self) -> SelectionKey:
        """Return explicit producer identities, independent of artifact paths."""
        return (
            str(self.regions.manifest.run_id),
            None if self.visual is None else str(self.visual.manifest.run_id),
            None if self.semantic is None else str(self.semantic.manifest.run_id),
            tuple(str(reader.manifest.run_id) for reader in self.scoring),
        )

    @classmethod
    def open(
        cls,
        *,
        regions: Path,
        visual: Path | None = None,
        semantic: Path | None = None,
        scoring: Sequence[Path] = (),
    ) -> PerceptionEvidenceSelection:
        """Open only the producer directories the caller explicitly selected."""
        return cls(
            regions=RegionPerceptionRunReader(regions),
            visual=None if visual is None else VisualEvidenceRunReader(visual),
            semantic=None if semantic is None else SemanticInterpretationRunReader(semantic),
            scoring=tuple(SemanticScoringRunReader(path) for path in scoring),
        )


@dataclass(frozen=True, kw_only=True)
class ObservationEvidence:
    """The selected producer views that cover one physical observation."""

    source_observation_id: SourceObservationId
    results_by_selection: Mapping[SelectionKey, PerceptionResult]


class PerceptionEvidenceSet:
    """Structural grouping of explicitly selected producer bundles by physical observation."""

    def __init__(self, selections: Sequence[PerceptionEvidenceSelection]) -> None:
        """Validate unique selections and the canonical sequence they all cover."""
        if not selections:
            raise EvidenceSetError("at least one producer selection is required")
        self._selections = tuple(selections)
        keys = [selection.key for selection in selections]
        if len(keys) != len(set(keys)):
            raise EvidenceSetError("duplicate producer selection")
        sequences = {selection.regions.manifest.sequence_artifact_id for selection in selections}
        if len(sequences) != 1:
            raise EvidenceSetError("selected producer bundles reference incompatible sequences")
        self._regions = {
            selection.key: self._regions_by_observation(selection.regions)
            for selection in selections
        }
        self._visual = {
            selection.key: self._visual_by_observation(selection.visual) for selection in selections
        }
        self._semantic = {
            selection.key: self._semantic_by_observation(selection.semantic)
            for selection in selections
        }

    @staticmethod
    def _regions_by_observation(
        reader: RegionPerceptionRunReader,
    ) -> dict[SourceObservationId, tuple]:
        groups: dict[SourceObservationId, list] = {
            SourceObservationId(value): [] for value in reader.manifest.source_observation_ids
        }
        for region in reader.read_regions():
            if region.source_observation_id is None or region.source_observation_id not in groups:
                raise EvidenceSetError("regional record names an observation outside its manifest")
            groups[region.source_observation_id].append(region)
        return {observation: tuple(regions) for observation, regions in groups.items()}

    @staticmethod
    def _visual_by_observation(
        reader: VisualEvidenceRunReader | None,
    ) -> dict[SourceObservationId, VisualEvidenceResult]:
        if reader is None:
            return {}
        results = {result.source_observation_id: result for result in reader.iter_results()}
        expected = {SourceObservationId(value) for value in reader.manifest.source_observation_ids}
        if set(results) != expected:
            raise EvidenceSetError("visual evidence coverage disagrees with its manifest")
        return results

    @staticmethod
    def _semantic_by_observation(
        reader: SemanticInterpretationRunReader | None,
    ) -> dict[SourceObservationId, SemanticObservation]:
        if reader is None:
            return {}
        results = {result.source_observation_id: result for result in reader.iter_observations()}
        expected = {SourceObservationId(value) for value in reader.manifest.source_observation_ids}
        if set(results) != expected:
            raise EvidenceSetError("semantic evidence coverage disagrees with its manifest")
        return results

    def observation_ids(self) -> list[SourceObservationId]:
        """Return all physical observations covered by selected regional producers."""
        return sorted({obs for groups in self._regions.values() for obs in groups})

    def evidence_for(self, source_observation_id: SourceObservationId) -> ObservationEvidence:
        """Return one derived view for each selected bundle covering the observation."""
        views: dict[SelectionKey, PerceptionResult] = {}
        for selection in self._selections:
            key = selection.key
            regions = self._regions[key].get(source_observation_id)
            if regions is None:
                continue
            visual = self._visual[key].get(source_observation_id)
            semantic = self._semantic[key].get(source_observation_id)
            scores = tuple(
                score
                for reader in selection.scoring
                for score in reader.iter_scores()
                if score.source_observation_id == source_observation_id
            )
            views[key] = PerceptionResult(
                source_observation_id=source_observation_id,
                sequence_artifact_id=str(selection.regions.manifest.sequence_artifact_id),
                region_artifact_id=key[0],
                visual_evidence_artifact_id=key[1],
                semantic_interpretation_artifact_id=key[2],
                semantic_scoring_artifact_ids=key[3],
                regions=regions,
                features=() if visual is None else visual.features,
                claims=() if semantic is None else semantic.claims,
                scene_context=None if semantic is None else semantic.scene_context,
                semantic_scores=scores,
            )
        return ObservationEvidence(
            source_observation_id=source_observation_id,
            results_by_selection=views,
        )
