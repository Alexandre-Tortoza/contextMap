"""Os contratos de metadata de um run e a decodificação de um manifest já lido do disco.

A linhagem é uma **seleção explícita**: o run de fusão de origem com a sua identidade, versão de
schema e digest, o mapa geométrico, a sequência canônica e, pela linhagem do run de fusão, as runs
de Sensor Association, regiões, interpretação semântica, evidência visual e Point Representation.
Nada aqui descobre upstream sozinho.

O manifest separa duas versões de schema que não se confundem — a do layout do artifact e a do
registro canônico da entidade — e o leitor recusa qualquer uma que não entenda. Este módulo não
varre diretório, não verifica integridade e não escreve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contextmap.geometric_mapping import MapId
from contextmap.point_representation import PointRepresentationRunId
from contextmap.semantic_fusion import SemanticFusionRunId, SemanticFusionRunManifest
from contextmap.semantic_mapping.entity import SemanticMapId
from contextmap.semantic_mapping.evidence import fusion_artifact_digest
from contextmap.semantic_mapping.run_artifact.layout import (
    MANIFEST,
    SCHEMA_VERSION,
    IncompleteMappingRunArtifactError,
    MappingRunArtifactError,
    SemanticMappingRunId,
)
from contextmap.semantic_mapping.serialization import ENTITY_SCHEMA_VERSION
from contextmap.shared import FileEntry


@dataclass(frozen=True, kw_only=True)
class MappingRunLineage:
    """The upstream artifacts a semantic mapping run consumed.

    Attributes:
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The immutable geometric map the geometry belongs to.
        fusion_run_id: The selected Semantic Fusion run the entities were materialized from.
        fusion_schema_version: The schema version of that run.
        fusion_artifact_digest: Digest of that run's identity and inventory.
        association_run_ids: The Sensor Association runs behind the fusion run, sorted and
            unique, through its lineage.
        region_artifact_ids: Regional producer artifacts behind fusion, sorted and unique.
        semantic_interpretation_artifact_ids: Semantic interpretation artifacts behind fusion,
            sorted and unique.
        visual_artifact_ids: Visual evidence artifacts behind fusion, sorted and unique.
        point_representation_run_ids: The Point Representation runs behind it, sorted and
            unique; empty when none was used.
    """

    sequence_artifact_id: str
    geometric_map_id: MapId
    fusion_run_id: SemanticFusionRunId
    fusion_schema_version: str
    fusion_artifact_digest: str
    association_run_ids: tuple[str, ...]
    region_artifact_ids: tuple[str, ...]
    semantic_interpretation_artifact_ids: tuple[str, ...] = ()
    visual_artifact_ids: tuple[str, ...] = ()
    point_representation_run_ids: tuple[PointRepresentationRunId, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every identity is present and every selection sorted and unique.

        Raises:
            ValueError: If an identity is empty, or a selection is not sorted and unique.
        """
        for name in (
            "sequence_artifact_id",
            "geometric_map_id",
            "fusion_run_id",
            "fusion_schema_version",
            "fusion_artifact_digest",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "association_run_ids",
            "region_artifact_ids",
            "semantic_interpretation_artifact_ids",
            "visual_artifact_ids",
            "point_representation_run_ids",
        ):
            items: tuple[str, ...] = getattr(self, name)
            if list(items) != sorted(set(items)):
                raise ValueError(f"{name} must be sorted and unique")


def lineage_from_fusion_manifest(manifest: SemanticFusionRunManifest) -> MappingRunLineage:
    """Derive the lineage of a mapping run from the fusion run it consumes.

    Args:
        manifest: The manifest of the selected Semantic Fusion run.

    Returns:
        The lineage: that run's identity, version and digest, and the map, sequence, association,
        perception and point representation runs behind it.
    """
    lineage = manifest.lineage
    return MappingRunLineage(
        sequence_artifact_id=lineage.sequence_artifact_id,
        geometric_map_id=lineage.geometric_map_id,
        fusion_run_id=manifest.run_id,
        fusion_schema_version=manifest.schema_version,
        fusion_artifact_digest=fusion_artifact_digest(manifest),
        association_run_ids=lineage.association_run_ids,
        region_artifact_ids=lineage.region_artifact_ids,
        semantic_interpretation_artifact_ids=lineage.semantic_interpretation_artifact_ids,
        visual_artifact_ids=lineage.visual_artifact_ids,
        point_representation_run_ids=lineage.point_representation_run_ids,
    )


@dataclass(frozen=True, kw_only=True)
class SemanticMappingRunManifest:
    """Authoritative metadata of a persisted Semantic Mapping run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        semantic_map_id: The semantic map the run holds; the scope of every entity id in it.
        lineage: The upstream artifacts the run consumed.
        materialization_policy_id: The materialization policy, ``None`` for an empty run.
        identity_policy_id: The entity id allocation policy, ``None`` for an empty run.
        configuration_fingerprint: Hash of the materialization configuration.
        code_version: Code revision that produced the run.
        code_digest: Digest of the code that produced the run, exactly as the caller supplied it;
            the capability never computes one from a source tree.
        entity_count: Entities persisted.
        rejected_count: Candidates that could not become entities, persisted explicitly.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        entity_schema_version: Version of the canonical entity record the run stores
            (``ENTITY_SCHEMA_VERSION``), independent of the artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``__DEBUG__/``.
    """

    run_id: SemanticMappingRunId
    run_index: int
    sequence_name: str
    semantic_map_id: SemanticMapId
    lineage: MappingRunLineage
    materialization_policy_id: str | None
    identity_policy_id: str | None
    configuration_fingerprint: str | None
    code_version: str
    code_digest: str
    entity_count: int
    rejected_count: int
    warnings: tuple[str, ...]
    debug_level: str
    schema_version: str
    entity_schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def load_manifest(run_dir: Path) -> SemanticMappingRunManifest:
    """Read and decode the manifest of a published run.

    Args:
        run_dir: The run directory.

    Returns:
        The decoded manifest.

    Raises:
        IncompleteMappingRunArtifactError: If the directory holds no manifest.
        MappingRunArtifactError: If the manifest is malformed or declares a schema version this
            module does not understand.
    """
    manifest_path = run_dir / MANIFEST
    if not manifest_path.is_file():
        raise IncompleteMappingRunArtifactError(f"missing {MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MappingRunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    if raw.get("entity_schema_version") != ENTITY_SCHEMA_VERSION:
        raise MappingRunArtifactError(
            f"unsupported entity_schema_version: {raw.get('entity_schema_version')!r}"
        )
    lineage = raw["lineage"]
    policies = raw["policies"]
    counts = raw["counts"]
    return SemanticMappingRunManifest(
        run_id=SemanticMappingRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        semantic_map_id=SemanticMapId(raw["semantic_map_id"]),
        lineage=MappingRunLineage(
            sequence_artifact_id=lineage["sequence_artifact_id"],
            geometric_map_id=MapId(lineage["geometric_map_id"]),
            fusion_run_id=SemanticFusionRunId(lineage["fusion_run_id"]),
            fusion_schema_version=lineage["fusion_schema_version"],
            fusion_artifact_digest=lineage["fusion_artifact_digest"],
            association_run_ids=tuple(lineage["association_run_ids"]),
            region_artifact_ids=tuple(lineage["region_artifact_ids"]),
            semantic_interpretation_artifact_ids=tuple(
                lineage["semantic_interpretation_artifact_ids"]
            ),
            visual_artifact_ids=tuple(lineage["visual_artifact_ids"]),
            point_representation_run_ids=tuple(
                PointRepresentationRunId(i) for i in lineage["point_representation_run_ids"]
            ),
        ),
        materialization_policy_id=policies["materialization_policy_id"],
        identity_policy_id=policies["identity_policy_id"],
        configuration_fingerprint=policies["configuration_fingerprint"],
        code_version=raw["code_version"],
        code_digest=raw["code_digest"],
        entity_count=counts["entities"],
        rejected_count=counts["rejected_candidates"],
        warnings=tuple(raw["warnings"]),
        debug_level=raw["debug_level"],
        schema_version=raw["schema_version"],
        entity_schema_version=raw["entity_schema_version"],
        created_at=raw["created_at"],
        file_inventory=tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )
