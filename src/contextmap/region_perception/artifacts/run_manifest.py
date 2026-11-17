"""Manifest de uma execução de region_perception: identidade, linhagem e inventário.

O manifest é o que transforma um diretório em artifact: diz qual sequência e quais observações
originaram o run, qual backend descobriu as regiões sob qual configuração, qual política as
normalizou, e quais arquivos contratuais foram publicados com que hash.

``schema_version`` é explícito e verificado na leitura: um run gravado por um esquema que este
código não conhece é recusado em vez de interpretado por adivinhação (AGENTS.md §20, §26).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from contextmap.shared import FileEntry

RegionPerceptionRunId = str

REGION_PERCEPTION_SCHEMA_VERSION = "0.2.0"
"""Schema version of the published region perception run."""


class RegionPerceptionRunError(Exception):
    """A region perception run could not be published or read as a valid artifact."""


@dataclass(frozen=True, kw_only=True)
class RegionPerceptionRunManifest:
    """Metadata for one region perception run (discovery + optional grounding/refinement).

    Tracks upstream dependencies and configuration identity.

    Attributes:
        run_id: Identity of this run.
        run_index: Position of this run among the runs of its plan.
        sequence_artifact_id: Upstream sequence artifact the observations came from.
        source_observation_ids: Physical frames explicitly selected for this execution.
        sequence_name: Human-readable sequence identity of the selected observations.
        selection_id: Identity of the ingestion selection being consumed.
        created_at: ISO-8601 instant the run was published.
        prepared_image_id: Identity of the image preparation the discovery saw.
        discovery_backend_id: Backend that discovered the regions.
        discovery_model: Model the discovery backend ran.
        discovery_version: Version of that model.
        discovery_configuration_fingerprint: Effective configuration of the discovery.
        normalization_policy_fingerprint: Policy that normalized the candidates.
        grounding_backend_id: Grounding backend, or ``None`` when the run had no grounding.
        grounding_configuration_fingerprint: Its effective configuration, or ``None``.
        refinement_backend_id: Refinement backend, or ``None`` when the run had none.
        refinement_configuration_fingerprint: Its effective configuration, or ``None``.
        code_version: Commit the run was produced by, when known.
        region_count: How many regions the run published.
        file_inventory: The contractual files published, with sizes and hashes.
        schema_version: Region perception artifact schema version.
    """

    run_id: RegionPerceptionRunId
    run_index: int
    sequence_artifact_id: str
    source_observation_ids: Sequence[str]
    created_at: str
    sequence_name: str
    selection_id: str
    prepared_image_id: str
    discovery_backend_id: str
    discovery_model: str
    discovery_version: str
    discovery_configuration_fingerprint: str
    normalization_policy_fingerprint: str
    grounding_backend_id: str | None = None
    grounding_configuration_fingerprint: str | None = None
    refinement_backend_id: str | None = None
    refinement_configuration_fingerprint: str | None = None
    code_version: str | None = None
    region_count: int = 0
    file_inventory: tuple[FileEntry, ...] = ()
    schema_version: str = REGION_PERCEPTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze the observation list, so two equal manifests compare equal.

        A caller may pass any sequence; keeping it as given would make a manifest built
        from a list differ from the same manifest read back from disk.
        """
        object.__setattr__(self, "source_observation_ids", tuple(self.source_observation_ids))

    def to_dict(self) -> dict[str, Any]:
        """Render the manifest in the shape persisted as ``manifest.json``.

        ``file_inventory`` is deliberately absent: the atomic publication injects the
        inventory it actually recorded, so emitting it here would let a stale copy
        contradict what was written.

        Returns:
            Every other field of this manifest, with the optional grounding and refinement
            identities present as ``None`` when the run did not include those stages,
            so a reader can tell an absent stage from a missing field.
        """
        return {
            "run_id": self.run_id,
            "run_index": self.run_index,
            "sequence_artifact_id": self.sequence_artifact_id,
            "source_observation_ids": list(self.source_observation_ids),
            "created_at": self.created_at,
            "sequence_name": self.sequence_name,
            "selection_id": self.selection_id,
            "prepared_image_id": self.prepared_image_id,
            "discovery_backend_id": self.discovery_backend_id,
            "discovery_model": self.discovery_model,
            "discovery_version": self.discovery_version,
            "discovery_configuration_fingerprint": self.discovery_configuration_fingerprint,
            "normalization_policy_fingerprint": self.normalization_policy_fingerprint,
            "grounding_backend_id": self.grounding_backend_id,
            "grounding_configuration_fingerprint": self.grounding_configuration_fingerprint,
            "refinement_backend_id": self.refinement_backend_id,
            "refinement_configuration_fingerprint": self.refinement_configuration_fingerprint,
            "code_version": self.code_version,
            "region_count": self.region_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> RegionPerceptionRunManifest:
        """Reconstruct a manifest from the published ``manifest.json``.

        Args:
            record: The decoded manifest, as the publication wrote it.

        Returns:
            The typed manifest, with ``file_inventory`` taken from the published record.

        Raises:
            RegionPerceptionRunError: If the schema version is not the one this code
                reads, or a required field is missing.
        """
        schema_version = record.get("schema_version")
        if schema_version != REGION_PERCEPTION_SCHEMA_VERSION:
            raise RegionPerceptionRunError(
                f"unsupported schema_version: {schema_version} "
                f"(expected {REGION_PERCEPTION_SCHEMA_VERSION})"
            )

        inventory = tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in record.get("file_inventory", ())
        )

        try:
            manifest = cls(
                run_id=record["run_id"],
                run_index=record["run_index"],
                sequence_artifact_id=record["sequence_artifact_id"],
                source_observation_ids=tuple(record["source_observation_ids"]),
                created_at=record["created_at"],
                sequence_name=record["sequence_name"],
                selection_id=record["selection_id"],
                prepared_image_id=record["prepared_image_id"],
                discovery_backend_id=record["discovery_backend_id"],
                discovery_model=record["discovery_model"],
                discovery_version=record["discovery_version"],
                discovery_configuration_fingerprint=record["discovery_configuration_fingerprint"],
                normalization_policy_fingerprint=record["normalization_policy_fingerprint"],
                grounding_backend_id=record.get("grounding_backend_id"),
                grounding_configuration_fingerprint=record.get(
                    "grounding_configuration_fingerprint"
                ),
                refinement_backend_id=record.get("refinement_backend_id"),
                refinement_configuration_fingerprint=record.get(
                    "refinement_configuration_fingerprint"
                ),
                code_version=record.get("code_version"),
                region_count=record.get("region_count", 0),
                schema_version=schema_version,
            )
        except KeyError as error:
            raise RegionPerceptionRunError(
                f"region perception manifest is missing required field {error.args[0]!r}"
            ) from error

        return replace(manifest, file_inventory=inventory)
