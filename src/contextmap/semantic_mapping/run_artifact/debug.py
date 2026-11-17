"""A evidência humana de diagnóstico de um run, escrita em `__DEBUG__/` e nunca contratual.

Nada daqui entra no inventário do manifest: remover `__DEBUG__/` não invalida o run, e nenhum
estágio a jusante pode depender destes arquivos. É exatamente por isso que eles podem mostrar o que
for útil para um humano sem criar contrato.

`MappingDebugLevel.STANDARD` cobre uma amostra das primeiras entidades; `FULL` cobre todas.
"""

from __future__ import annotations

from typing import Any

from contextmap.semantic_mapping.entity import Entity
from contextmap.semantic_mapping.run_artifact.layout import (
    DEBUG_ENTITY_LIMIT,
    MappingDebugLevel,
    encode_document,
)
from contextmap.semantic_mapping.serialization import encode_entity
from contextmap.shared import DEBUG_DIRECTORY, AtomicRunDirectory


def write_entity_debug(
    run: AtomicRunDirectory, entity: Entity, *, level: MappingDebugLevel, number: int
) -> None:
    """Write the debug evidence of one entity, when the level asks for it.

    Args:
        run: The run directory being built.
        entity: The entity just streamed.
        level: How much debug evidence the caller asked for.
        number: Ordinal of this entity in the stream, used by the ``STANDARD`` sample.
    """
    if level is MappingDebugLevel.NONE:
        return
    if level is MappingDebugLevel.STANDARD and number > DEBUG_ENTITY_LIMIT:
        return
    base = f"{DEBUG_DIRECTORY}/entities/{entity.entity_id}"
    record = encode_entity(entity)
    files = {
        "summary.json": _debug_summary(entity),
        "geometry-summary.json": record["geometry"],
        "semantic-state.json": record["semantic_state"],
        "evidence-trace.json": record["evidence"],
        "temporal-history.json": record["temporal_state"],
    }
    for name, content in files.items():
        run.write_text(f"{base}/{name}", encode_document(content), contractual=False)


def _debug_summary(entity: Entity) -> dict[str, Any]:
    state = entity.semantic_state
    primary = state.primary
    return {
        "entity_id": str(entity.entity_id),
        "semantic_map_id": str(entity.semantic_map_id),
        "ambiguity_state": state.ambiguity_state.value,
        "primary_label": None if primary is None else primary.label,
        "hypothesis_labels": [item.label for item in state.hypotheses],
        "geometry_points": entity.geometry.statistics.point_count,
        "physical_observation_count": entity.temporal_state.physical_observation_count,
        "inference_result_count": entity.temporal_state.inference_result_count,
        "fused_evidence": [str(ref.fused_evidence_id) for ref in entity.evidence.fused_evidence],
    }
