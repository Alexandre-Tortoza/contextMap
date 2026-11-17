"""Linhagem e manifest autoritativo de um run de Semantic Fusion persistido.

O manifest é a metadata autoritativa do run: identidade, linhagem a montante explícita,
as políticas sob as quais o run inteiro foi executado, as identidades que alimentaram cada
canal, contagens e o inventário de todo arquivo contratual com tamanho e hash.

A linhagem é **seleção explícita**, nunca descoberta: um run nomeia exatamente os
artifacts a montante que consumiu. :attr:`FusionRunLineage.point_representation_run_ids`
registra a seleção, não o uso, para que os braços de uma ablação de canal vejam os mesmos
artifacts a montante.

Este módulo decodifica um registro já parseado; ele não varre diretório, não verifica
integridade e não escreve nada.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from contextmap.geometric_mapping import MapId
from contextmap.point_representation import PointRepresentationRunId
from contextmap.semantic_fusion.invariants import require_canonical
from contextmap.semantic_fusion.run_artifact.layout import (
    MANIFEST,
    SCHEMA_VERSION,
    FusionRunArtifactError,
    IncompleteFusionRunArtifactError,
    SemanticFusionRunId,
)
from contextmap.semantic_scoring import SemanticScoringRunId
from contextmap.shared import FileEntry


@dataclass(frozen=True, kw_only=True)
class FusionRunLineage:
    """The upstream artifacts a fusion run consumed.

    Attributes:
        sequence_artifact_id: The canonical sequence everything was built from.
        geometric_map_id: The immutable geometric map the geometry belongs to.
        association_run_ids: The selected Sensor Association runs, sorted and unique.
        region_artifact_ids: The selected regional artifacts, sorted and unique.
        semantic_interpretation_artifact_ids: The selected semantic artifacts, sorted and unique.
        visual_artifact_ids: The selected visual-evidence artifacts, sorted and unique.
        point_representation_run_ids: The Point Representation runs selected for the run, whose
            structure a declared channel may reference, sorted and unique; empty when none was
            selected. It records the selection, not the use: in a channel ablation every arm
            lists the same runs, so that the arms see identical upstream artifacts.
        semantic_scoring_run_ids: Semantic Scoring runs selected as raw-score evidence. They are
            recorded separately from their scorer identity because an identity does not name the
            immutable execution or its pinned perception run.
    """

    sequence_artifact_id: str
    geometric_map_id: MapId
    association_run_ids: tuple[str, ...]
    region_artifact_ids: tuple[str, ...]
    semantic_interpretation_artifact_ids: tuple[str, ...] = ()
    visual_artifact_ids: tuple[str, ...] = ()
    point_representation_run_ids: tuple[PointRepresentationRunId, ...] = ()
    semantic_scoring_run_ids: tuple[SemanticScoringRunId, ...] = ()

    def __post_init__(self) -> None:
        """Validate that every selection is explicit, sorted and unique.

        Raises:
            ValueError: If an identity is empty, a required selection is empty, or a
                selection is not sorted and unique.
        """
        if not self.sequence_artifact_id.strip() or not self.geometric_map_id.strip():
            raise ValueError("sequence_artifact_id and geometric_map_id must not be empty")
        for name in ("association_run_ids", "region_artifact_ids"):
            items: tuple[str, ...] = getattr(self, name)
            if not items:
                raise ValueError(f"{name} must not be empty: upstream runs are selected explicitly")
        for name in (
            "association_run_ids",
            "region_artifact_ids",
            "semantic_interpretation_artifact_ids",
            "visual_artifact_ids",
            "point_representation_run_ids",
            "semantic_scoring_run_ids",
        ):
            require_canonical(name, getattr(self, name), lambda item: (item,))


@dataclass(frozen=True, kw_only=True)
class SemanticFusionRunManifest:
    """Authoritative metadata of a persisted Semantic Fusion run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        lineage: The upstream artifacts the run consumed.
        grouping_policy_id: The physical-observation grouping policy, ``None`` for an empty run.
        support_policy_id: The support construction policy, ``None`` for an empty run.
        support_configuration_fingerprint: Hash of the support configuration.
        fusion_policy_id: The fusion policy, ``None`` for an empty run.
        fusion_configuration_fingerprint: Hash of the fusion configuration.
        evidence_identities: Per evidence channel, the identities (interpreters, scorers,
            embedding spaces, quality definitions, representation spaces, map) that fed it.
        code_version: Code revision that produced the run.
        support_count: Fusion supports persisted.
        fused_evidence_count: Fused evidence records persisted.
        contribution_count: Contributions across all supports.
        excluded_count: Observations left out of every support, persisted explicitly.
        warnings: Human-readable warnings of the run.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes the manifest, the
            README and ``__DEBUG__/``.
    """

    run_id: SemanticFusionRunId
    run_index: int
    sequence_name: str
    lineage: FusionRunLineage
    grouping_policy_id: str | None
    support_policy_id: str | None
    support_configuration_fingerprint: str | None
    fusion_policy_id: str | None
    fusion_configuration_fingerprint: str | None
    evidence_identities: Mapping[str, tuple[str, ...]]
    code_version: str
    support_count: int
    fused_evidence_count: int
    contribution_count: int
    excluded_count: int
    warnings: tuple[str, ...]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def load_manifest(run_dir: Path) -> SemanticFusionRunManifest:
    """Read and decode the manifest of a persisted run.

    Args:
        run_dir: The run's own directory.

    Returns:
        The manifest, with every contract revalidated by its constructor.

    Raises:
        IncompleteFusionRunArtifactError: If the manifest is missing.
        FusionRunArtifactError: If the schema version is not the one this module understands;
            a pre-1.0 schema keeps no compatibility reader.
    """
    manifest_path = run_dir / MANIFEST
    if not manifest_path.is_file():
        raise IncompleteFusionRunArtifactError(f"missing {MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise FusionRunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}; "
            f"this reader understands {SCHEMA_VERSION!r}"
        )
    lineage = raw["lineage"]
    policies = raw["policies"]
    counts = raw["counts"]
    return SemanticFusionRunManifest(
        run_id=SemanticFusionRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        lineage=FusionRunLineage(
            sequence_artifact_id=lineage["sequence_artifact_id"],
            geometric_map_id=MapId(lineage["geometric_map_id"]),
            association_run_ids=tuple(lineage["association_run_ids"]),
            region_artifact_ids=tuple(lineage["region_artifact_ids"]),
            semantic_interpretation_artifact_ids=tuple(
                lineage["semantic_interpretation_artifact_ids"]
            ),
            visual_artifact_ids=tuple(lineage["visual_artifact_ids"]),
            point_representation_run_ids=tuple(
                PointRepresentationRunId(i) for i in lineage["point_representation_run_ids"]
            ),
            semantic_scoring_run_ids=tuple(
                SemanticScoringRunId(i) for i in lineage.get("semantic_scoring_run_ids", ())
            ),
        ),
        grouping_policy_id=policies["grouping_policy_id"],
        support_policy_id=policies["support_policy_id"],
        support_configuration_fingerprint=policies["support_configuration_fingerprint"],
        fusion_policy_id=policies["fusion_policy_id"],
        fusion_configuration_fingerprint=policies["fusion_configuration_fingerprint"],
        evidence_identities={
            channel: tuple(identities) for channel, identities in raw["evidence_identities"].items()
        },
        code_version=raw["code_version"],
        support_count=counts["supports"],
        fused_evidence_count=counts["fused_evidence"],
        contribution_count=counts["contributions"],
        excluded_count=counts["excluded_observations"],
        warnings=tuple(raw["warnings"]),
        debug_level=raw["debug_level"],
        schema_version=raw["schema_version"],
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
