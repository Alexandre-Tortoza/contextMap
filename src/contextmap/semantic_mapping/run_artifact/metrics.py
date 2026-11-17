"""As checagens em fluxo de um run e as figuras que o manifest e `metrics/` registram.

Duas coisas acontecem aqui numa única passada pelas entidades, e são a mesma: o run só é publicado
se cada entidade for coerente com o mapa, com a linhagem e com a política única do run, e as
contagens que o manifest grava são exatamente as dessa passada. Separá-las faria o manifest afirmar
números que ninguém verificou.

As checagens recusam com `MappingRunArtifactError` antes de qualquer publicação: ordem estrita por
identidade (o que também recusa duplicata), mapa de outra identidade, mapa geométrico divergente da
linhagem, run de fusão/visual/representação que a linhagem não lista, e mais de uma política ou
configuração no mesmo run.

Este módulo não guarda linha de `outputs/` — isso é de `tables.py` — e não mede qualidade
científica: as figuras são de tamanho e contagem, nunca de acerto semântico.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from contextmap.semantic_mapping.entity import Entity, EntityId, RejectionReason, SemanticMapId
from contextmap.semantic_mapping.run_artifact.layout import MappingRunArtifactError
from contextmap.semantic_mapping.run_artifact.manifest import MappingRunLineage
from contextmap.semantic_mapping.semantic_state import AmbiguityState


class RunTally:
    """Checks the entities as they stream and collects the figures of the run."""

    def __init__(self, semantic_map_id: SemanticMapId, lineage: MappingRunLineage) -> None:
        """Create the tally of one run.

        Args:
            semantic_map_id: The semantic map every entity of the run must belong to.
            lineage: The explicit upstream selection every entity must be consistent with.
        """
        self._semantic_map_id = semantic_map_id
        self._lineage = lineage
        self.entity_count = 0
        self.output_sizes: dict[str, int] = {}
        self.materialization_policy_id: str | None = None
        self.identity_policy_id: str | None = None
        self.configuration_fingerprint: str | None = None
        self._previous: EntityId | None = None
        self._supports: set[str] = set()
        self._physical: set[str] = set()
        self._inference = 0
        self._points = 0
        self._states: Counter[str] = Counter()
        self._with_primary = 0
        self._with_conflicts = 0
        self._without_features = 0
        self._without_representations = 0
        self._per_entity: dict[str, list[int]] = {
            "geometry_points_per_entity": [],
            "physical_observations_per_entity": [],
            "inference_results_per_entity": [],
            "hypotheses_per_entity": [],
        }

    def add(self, entity: Entity) -> None:
        """Fold one entity into the counts and the per-entity distributions."""
        self.entity_count += 1
        geometry, state = entity.geometry, entity.semantic_state
        temporal, links = entity.temporal_state, entity.evidence
        for ref in links.fused_evidence:
            self._supports.add(str(ref.fusion_support_id))
        self._physical.update(str(item) for item in links.physical_observation_ids)
        self._inference += temporal.inference_result_count
        self._points += geometry.statistics.point_count
        self._states[state.ambiguity_state.value] += 1
        self._with_primary += state.primary_hypothesis is not None
        self._with_conflicts += bool(state.conflicts)
        self._without_features += not links.visual_feature_refs
        self._without_representations += not links.point_representation_refs
        per = self._per_entity
        per["geometry_points_per_entity"].append(geometry.statistics.point_count)
        per["physical_observations_per_entity"].append(temporal.physical_observation_count)
        per["inference_results_per_entity"].append(temporal.inference_result_count)
        per["hypotheses_per_entity"].append(len(state.hypotheses))

    def check(self, entity: Entity) -> None:
        """Refuse an entity that breaks the order, the map, the lineage or the single policy.

        Raises:
            MappingRunArtifactError: With what was violated and which entity caused it.
        """
        if self._previous is not None and entity.entity_id <= self._previous:
            raise MappingRunArtifactError(
                f"entities must be strictly sorted by identity: {entity.entity_id!r} follows "
                f"{self._previous!r}"
            )
        self._previous = entity.entity_id
        if entity.semantic_map_id != self._semantic_map_id:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} belongs to semantic map "
                f"{entity.semantic_map_id!r}, but the run holds {self._semantic_map_id!r}"
            )
        if entity.geometry.geometric_map_id != self._lineage.geometric_map_id:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} is over map {entity.geometry.geometric_map_id!r}, "
                f"but the run's lineage names {self._lineage.geometric_map_id!r}"
            )
        for ref in entity.evidence.fused_evidence:
            if (
                ref.fusion_run_id != self._lineage.fusion_run_id
                or ref.fusion_schema_version != self._lineage.fusion_schema_version
                or ref.fusion_artifact_digest != self._lineage.fusion_artifact_digest
            ):
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references fusion run {ref.fusion_run_id!r}, "
                    f"which is not the run the lineage names, {self._lineage.fusion_run_id!r}"
                )
            if ref.sequence_artifact_id != self._lineage.sequence_artifact_id:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} was materialized over sequence "
                    f"{ref.sequence_artifact_id!r}, but the run's lineage names "
                    f"{self._lineage.sequence_artifact_id!r}"
                )
        for feature in entity.evidence.visual_feature_refs:
            if feature.visual_artifact_id not in self._lineage.visual_artifact_ids:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references visual artifact "
                    f"{feature.visual_artifact_id!r}, which the run's lineage does not list"
                )
        for representation in entity.evidence.point_representation_refs:
            if representation.run_id not in self._lineage.point_representation_run_ids:
                raise MappingRunArtifactError(
                    f"entity {entity.entity_id!r} references point representation run "
                    f"{representation.run_id!r}, which the run's lineage does not list"
                )
        self._single_policy(entity)

    def check_rejections(self, supports: list[str]) -> None:
        """Refuse rejections that are unsorted, repeated, or also present as entities.

        Raises:
            MappingRunArtifactError: With the offending support.
        """
        if supports != sorted(set(supports)):
            raise MappingRunArtifactError(
                "rejected candidates must be sorted by support and unique"
            )
        clash = self._supports.intersection(supports)
        if clash:
            raise MappingRunArtifactError(
                f"a candidate cannot be both an entity and a rejection: {sorted(clash)!r}"
            )

    def _single_policy(self, entity: Entity) -> None:
        provenance = entity.provenance
        current = (
            provenance.materialization_policy_id,
            provenance.identity_policy_id,
            provenance.configuration_fingerprint,
        )
        first = (
            self.materialization_policy_id,
            self.identity_policy_id,
            self.configuration_fingerprint,
        )
        if self.entity_count == 0:
            (
                self.materialization_policy_id,
                self.identity_policy_id,
                self.configuration_fingerprint,
            ) = current
        elif current != first:
            raise MappingRunArtifactError(
                f"entity {entity.entity_id!r} was materialized under a different policy or "
                f"configuration than the rest of the run: one run keeps one"
            )

    def counts(self, rejected: list[dict[str, Any]], warnings: int) -> dict[str, Any]:
        """The counts of the run, with optional channels reported as explicit absences."""
        reasons = Counter(row["reason"] for row in rejected)
        return {
            "entities": self.entity_count,
            "rejected_candidates": {
                "total": len(rejected),
                "by_reason": {
                    reason.value: reasons.get(reason.value, 0) for reason in RejectionReason
                },
            },
            "semantic_state": {
                state.value: self._states.get(state.value, 0) for state in AmbiguityState
            },
            "entities_with_primary_hypothesis": self._with_primary,
            "entities_with_conflicts": self._with_conflicts,
            "entities_without_visual_features": self._without_features,
            "entities_without_point_representations": self._without_representations,
            "physical_observations": len(self._physical),
            "inference_results": self._inference,
            "geometry_points": self._points,
            "warnings": warnings,
        }

    def distributions(self) -> dict[str, Any]:
        """The per-entity distributions, each as count, minimum, median and maximum."""
        return {name: _distribution(values) for name, values in self._per_entity.items()}


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }
