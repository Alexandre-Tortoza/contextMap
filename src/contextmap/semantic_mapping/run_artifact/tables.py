"""As tabelas de índice derivadas de um run, uma linha por entidade.

`outputs/entities.jsonl` é a fonte autoritativa; estes índices são resumos derivados dela, para que
um consumidor encontre uma entidade, a sua geometria, a sua evidência, o seu estado ou os frames que
a observaram **sem carregar cada entidade**. `entity-index.jsonl` é o que torna possível resolver
uma `EntityReference` por deslocamento.

Este módulo muda quando um consumidor a jusante precisa de uma coluna nova, nunca quando uma figura
de métrica muda: é por isso que ele não mora em `metrics.py`. Ele não valida nada, não mede nada e
não escreve em disco — só monta as linhas, na ordem em que as entidades chegam.
"""

from __future__ import annotations

from typing import Any

from contextmap.semantic_mapping.entity import CandidateRejection, Entity


class IndexTables:
    """Accumulates the index rows of a run, in the order the entities are streamed."""

    def __init__(self) -> None:
        """Create empty tables."""
        self.entity_rows: list[dict[str, Any]] = []
        self.geometry_rows: list[dict[str, Any]] = []
        self.evidence_rows: list[dict[str, Any]] = []
        self.observation_rows: list[dict[str, Any]] = []
        self.semantic_rows: list[dict[str, Any]] = []
        self.temporal_rows: list[dict[str, Any]] = []

    def add(self, entity: Entity, *, offset: int, length: int) -> None:
        """Append the index rows of one entity.

        Args:
            entity: The entity just written to the payload.
            offset: Byte offset of its record inside the payload.
            length: Byte length of its record, without the newline.
        """
        entity_id = str(entity.entity_id)
        geometry, state = entity.geometry, entity.semantic_state
        temporal, links = entity.temporal_state, entity.evidence
        self.entity_rows.append(
            {
                "entity_id": entity_id,
                "semantic_map_id": str(entity.semantic_map_id),
                "offset": offset,
                "length": length,
            }
        )
        self.geometry_rows.append(
            {
                "entity_id": entity_id,
                "geometric_map_id": str(geometry.geometric_map_id),
                "map_frame": str(geometry.map_frame),
                "point_count": geometry.statistics.point_count,
                "centroid_m": list(geometry.centroid_m),
                "bounds": {
                    "minimum_m": list(geometry.bounds.minimum_m),
                    "maximum_m": list(geometry.bounds.maximum_m),
                },
                "extent_m": list(geometry.extent_m),
                "component_count": geometry.statistics.component_count,
                "diagnostics": sorted(item.kind.value for item in geometry.diagnostics),
            }
        )
        self.evidence_rows.append(
            {
                "entity_id": entity_id,
                "fused_evidence": [
                    {
                        "fusion_run_id": str(ref.fusion_run_id),
                        "fused_evidence_id": str(ref.fused_evidence_id),
                        "fusion_support_id": str(ref.fusion_support_id),
                    }
                    for ref in links.fused_evidence
                ],
                "spatial_observation_count": len(links.spatial_observation_ids),
                "visual_feature_count": len(links.visual_feature_refs),
                "point_representation_count": len(links.point_representation_refs),
            }
        )
        for item in temporal.observation_refs:
            self.observation_rows.append(
                {
                    "entity_id": entity_id,
                    "physical_observation_id": str(item.physical_observation_id),
                    "acquisition_timestamp": item.acquisition_timestamp.to_record(),
                    "inference_result_count": item.inference_result_count,
                }
            )
        primary = state.primary
        self.semantic_rows.append(
            {
                "entity_id": entity_id,
                "ambiguity_state": state.ambiguity_state.value,
                "primary_label": None if primary is None else primary.label,
                "hypothesis_labels": [item.label for item in state.hypotheses],
                "attributes": [
                    {"name": item.name, "value": item.value, "origin": item.origin.value}
                    for item in state.attributes
                ],
                "uncertainty_kinds": sorted({item.record.kind.value for item in state.uncertainty}),
                "conflict_count": len(state.conflicts),
            }
        )
        self.temporal_rows.append(
            {
                "entity_id": entity_id,
                "first_seen": temporal.first_seen.to_record(),
                "last_seen": temporal.last_seen.to_record(),
                "physical_observation_count": temporal.physical_observation_count,
                "inference_result_count": temporal.inference_result_count,
                "lifecycle": None if temporal.lifecycle is None else temporal.lifecycle.value,
            }
        )


def encode_rejection(item: CandidateRejection) -> dict[str, Any]:
    """Encode one rejected candidate as its persisted row."""
    return {
        "fusion_support_id": str(item.fusion_support_id),
        "fused_evidence_id": str(item.fused_evidence_id),
        "reason": item.reason.value,
        "detail": item.detail,
    }
