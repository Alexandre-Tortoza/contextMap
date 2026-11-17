"""Publicação atômica de um suporte espacial semântico como run imutável.

O writer recebe um suporte já construído e a linhagem já montada: ele não deriva nada, não
abre artifact a montante e não interpreta a linhagem, que grava verbatim. As três figuras do
run saem das próprias células. A escrita segue
:class:`~contextmap.shared.AtomicRunDirectory`, então um diretório existente nunca é
substituído e uma escrita interrompida nunca se parece com um run terminado.

A identidade da regra chega como parâmetro: o writer registra a política, não a implementa,
e por isso ``run_artifact`` não depende da regra de junção.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextmap.shared import AtomicRunDirectory, RunDirectoryError
from contextmap.spatial_support.models import SemanticSpatialSupport
from contextmap.spatial_support.policy import EntityVoxelSupportPolicy
from contextmap.spatial_support.run_artifact.layout import (
    CELLS,
    CONFIG,
    ENVIRONMENT,
    LINEAGE,
    METRICS,
    SCHEMA_VERSION,
    SUPPORT,
    SpatialSupportArtifactError,
    cell_record,
    json_document,
)
from contextmap.spatial_support.run_artifact.manifest import (
    SpatialSupportManifest,
    SpatialSupportRunId,
    environment_record,
    load_manifest,
)


class SpatialSupportArtifactWriter:
    """Persists an immutable semantic spatial support atomically."""

    def __init__(self, *, output_dir: Path, run_id: SpatialSupportRunId) -> None:
        """Create a writer for a new derivation.

        Args:
            output_dir: The final directory; created atomically and never replaced.
            run_id: Identity of the derivation, supplied by the caller.
        """
        self._final_dir = output_dir
        self._run_id = run_id

    def finalize(
        self,
        *,
        support: SemanticSpatialSupport,
        source_context_map_id: str,
        policy: EntityVoxelSupportPolicy,
        lineage: dict[str, Any],
        code_version: str | None,
    ) -> SpatialSupportManifest:
        """Persist a support and its lineage atomically.

        Args:
            support: The support to persist.
            source_context_map_id: The ContextMap the support was derived from.
            policy: Identity of the join rule that produced the support.
            lineage: The identities of the three upstream artifacts, recorded verbatim.
            code_version: Code revision that derived the support, when known.

        Returns:
            The manifest of the finalized artifact.

        Raises:
            SpatialSupportArtifactError: If the run directory already exists or a
                write cannot be finalized.
        """
        entities = {item.entity_id for cell in support.cells for item in cell.entity_support}
        total_weight = sum(
            item.support_weight for cell in support.cells for item in cell.entity_support
        )
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                run.write_text(
                    CELLS, "".join(f"{json.dumps(cell_record(cell))}\n" for cell in support.cells)
                )
                run.write_text(
                    SUPPORT,
                    json_document(
                        {
                            "source_map_id": str(support.source_map_id),
                            "frame_id": str(support.frame_id),
                            "grid_fingerprint": support.grid_fingerprint,
                            "aggregation_rule": support.aggregation_rule,
                            "cell_count": len(support.cells),
                        }
                    ),
                )
                run.write_text(
                    CONFIG,
                    json_document(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "policy": policy.to_record(),
                            "policy_fingerprint": policy.fingerprint(),
                            "grid_fingerprint": support.grid_fingerprint,
                            "aggregation_rule": support.aggregation_rule,
                        }
                    ),
                )
                run.write_text(LINEAGE, json_document(lineage))
                run.write_text(ENVIRONMENT, json_document(environment_record(code_version)))
                run.write_text(
                    METRICS,
                    json_document(
                        {
                            "cell_count": len(support.cells),
                            "entity_count": len(entities),
                            "total_support_weight": total_weight,
                        }
                    ),
                )
                run.publish(
                    manifest={
                        "run_id": str(self._run_id),
                        "source_context_map_id": source_context_map_id,
                        "source_map_id": str(support.source_map_id),
                        "grid_fingerprint": support.grid_fingerprint,
                        "policy_fingerprint": policy.fingerprint(),
                        "aggregation_rule": support.aggregation_rule,
                        "frame_id": str(support.frame_id),
                        "cell_count": len(support.cells),
                        "entity_count": len(entities),
                        "total_support_weight": total_weight,
                        "code_version": code_version,
                        "schema_version": SCHEMA_VERSION,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    readme=_readme(self._run_id, support),
                )
        except RunDirectoryError as error:
            raise SpatialSupportArtifactError(str(error)) from error
        return load_manifest(self._final_dir)


def _readme(run_id: SpatialSupportRunId, support: SemanticSpatialSupport) -> str:
    return (
        f"# Semantic spatial support `{run_id}`\n"
        "\n"
        f"- Raw map: `{support.source_map_id}` in frame `{support.frame_id}`\n"
        f"- Rule: `{support.aggregation_rule}` (grid `{support.grid_fingerprint}`)\n"
        f"- Supported cells: {len(support.cells)}\n"
        "\n"
        "A derived representation: the ContextMap and the voxel aggregation stay canonical. "
        "Every cell identity is a pure function of the grid, never of a raw point index.\n"
    )
