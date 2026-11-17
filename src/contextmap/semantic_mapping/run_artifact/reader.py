"""A leitura de um run finalizado, por identidade e sem carregar o resto.

O reader abre um run **só pelo próprio diretório**: nenhum registry, nenhum outro arquivo e nenhum
runtime de percepção, de fusão ou de modelo. `entity()` resolve uma `EntityReference` pelo
deslocamento gravado no índice, então ler uma entidade não custa ler as outras.

Todo registro é decodificado pelos construtores do contrato, então um registro adulterado é
**recusado** em vez de confiado, e `__DEBUG__/` nunca é fonte válida: `read_record` e `read_table`
só aceitam `outputs/` e `metrics/`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from contextmap.geometric_mapping import GeometrySource
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import SemanticFusionRunId
from contextmap.semantic_mapping.entity import (
    CandidateRejection,
    Entity,
    EntityId,
    EntityReference,
    EntitySet,
    ForeignEntityReferenceError,
    RejectionReason,
    SemanticMapId,
    UnknownEntityError,
)
from contextmap.semantic_mapping.evidence_validation import (
    EvidenceIntegrityIssue,
    validate_evidence_of_entities,
)
from contextmap.semantic_mapping.fused_evidence_source import FusedEvidenceSource
from contextmap.semantic_mapping.run_artifact.layout import (
    ENTITIES,
    ENTITY_INDEX,
    OBSERVATION_INDEX,
    REJECTED,
    MappingRunArtifactError,
)
from contextmap.semantic_mapping.run_artifact.manifest import (
    SemanticMappingRunManifest,
    load_manifest,
)
from contextmap.semantic_mapping.serialization import decode_entity
from contextmap.shared import check_file_inventory

_T = TypeVar("_T")


class SemanticMappingRunReader:
    """Reads a finalized Semantic Mapping run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is needed,
                and no perception, fusion or model runtime.

        Raises:
            IncompleteMappingRunArtifactError: If ``manifest.json`` is missing.
            MappingRunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = load_manifest(run_dir)
        self._rows: dict[EntityId, dict[str, Any]] | None = None
        self._observation_entities: dict[SourceObservationId, list[EntityId]] | None = None

    @property
    def manifest(self) -> SemanticMappingRunManifest:
        """The run's manifest."""
        return self._manifest

    @property
    def semantic_map_id(self) -> SemanticMapId:
        """The semantic map the run holds."""
        return self._manifest.semantic_map_id

    def entity_ids(self) -> list[EntityId]:
        """The identities of every entity, in persisted order."""
        return list(self._entity_rows())

    def entity(self, reference: EntityReference) -> Entity:
        """Resolve a reference to its entity without loading the others.

        Args:
            reference: A reference to an entity of this run's semantic map.

        Returns:
            The same canonical entity that was written.

        Raises:
            ForeignEntityReferenceError: If the reference names another semantic map.
            UnknownEntityError: If the run has no such entity.
            MappingRunArtifactError: If the entity's record is malformed or truncated.
        """
        if reference.semantic_map_id != self._manifest.semantic_map_id:
            raise ForeignEntityReferenceError(
                f"reference names semantic map {reference.semantic_map_id!r}, but this run holds "
                f"{self._manifest.semantic_map_id!r}"
            )
        row = self._entity_rows().get(reference.entity_id)
        if row is None:
            raise UnknownEntityError(reference.entity_id)
        raw = self._read_at(ENTITIES, row["offset"], row["length"])
        return _decode(raw, decode_entity, f"entity {reference.entity_id!r}")

    def iter_entities(self) -> Iterator[Entity]:
        """Iterate every entity, in persisted order."""
        with (self._root / ENTITIES).open("rb") as handle:
            for line in handle:
                yield _decode(line, decode_entity, "entity")

    def entities(self) -> EntitySet:
        """Load every entity of the run, addressable by reference."""
        return EntitySet(
            semantic_map_id=self._manifest.semantic_map_id, entities=tuple(self.iter_entities())
        )

    def entities_of_observation(self, observation_id: SourceObservationId) -> list[EntityId]:
        """The entities a physical observation contributed to, sorted.

        Args:
            observation_id: A physical observation.

        Returns:
            Its entities; empty when it contributed to none.
        """
        if self._observation_entities is None:
            index: dict[SourceObservationId, list[EntityId]] = {}
            for row in self._read_rows(OBSERVATION_INDEX):
                index.setdefault(SourceObservationId(row["physical_observation_id"]), []).append(
                    EntityId(row["entity_id"])
                )
            self._observation_entities = index
        return sorted(self._observation_entities.get(observation_id, []))

    def rejected_candidates(self) -> list[CandidateRejection]:
        """The candidates that could not become entities, with why."""
        return [
            CandidateRejection(
                fusion_support_id=row["fusion_support_id"],
                fused_evidence_id=row["fused_evidence_id"],
                reason=RejectionReason(row["reason"]),
                detail=row["detail"],
            )
            for row in self._read_rows(REJECTED)
        ]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``metrics/``.

        Raises:
            MappingRunArtifactError: If the path is not a JSON record of ``metrics/``;
                ``__DEBUG__/``
                is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith("metrics/"):
            raise MappingRunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def read_table(self, relative_path: str) -> list[dict[str, Any]]:
        """Read a contractual JSON Lines table from ``outputs/``.

        Raises:
            MappingRunArtifactError: If the path is not a table of ``outputs/`` or is malformed.
        """
        if not relative_path.endswith(".jsonl") or not relative_path.startswith("outputs/"):
            raise MappingRunArtifactError(f"not a contractual table: {relative_path!r}")
        return self._read_rows(relative_path)

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def validate_references(
        self,
        *,
        fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
        geometry: GeometrySource | None = None,
    ) -> tuple[EvidenceIntegrityIssue, ...]:
        """Check that the evidence every entity references is still there and unchanged.

        Args:
            fusion_runs: The persisted fusion runs to check against, by run identity.
            geometry: The read boundary of the geometric map, to also check the geometry.

        Returns:
            The issues found; empty means every reference resolves. Missing or corrupt upstream
            payloads are reported explicitly.
        """
        return validate_evidence_of_entities(
            self.iter_entities(), fusion_runs=fusion_runs, geometry=geometry
        )

    def _entity_rows(self) -> dict[EntityId, dict[str, Any]]:
        if self._rows is None:
            self._rows = {EntityId(row["entity_id"]): row for row in self._read_rows(ENTITY_INDEX)}
        return self._rows

    def _read_rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise MappingRunArtifactError(f"malformed table {relative_path}: {error}") from error

    def _read_at(self, relative_path: str, offset: int, length: int) -> bytes:
        with (self._root / relative_path).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise MappingRunArtifactError(
                f"{relative_path} is truncated: {length} bytes expected at {offset}, "
                f"found {len(data)}"
            )
        return data


def _decode(raw: bytes, decoder: Callable[[Mapping[str, Any]], _T], what: str) -> _T:
    try:
        return decoder(json.loads(raw))
    except (ValueError, KeyError, TypeError) as error:
        raise MappingRunArtifactError(f"malformed {what}: {error}") from error
