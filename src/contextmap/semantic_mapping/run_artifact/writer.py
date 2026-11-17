"""A publicação atômica de um run de Semantic Mapping, consumindo as entidades como fluxo.

O writer grava cada registro assim que ele chega e publica o run inteiro de uma vez só no fim: uma
falha no meio nunca deixa um diretório parcial parecendo um run válido, e a memória não limita o
número de entidades.

Ele grava **exatamente** no `output_dir` que o chamador entrega: não calcula caminho, não aloca
identidade, não mantém registro de runs e recusa um diretório que já exista — um run finalizado
nunca é sobrescrito.

Uma única passada pelas entidades faz três coisas que precisam concordar: a checagem
(`metrics.RunTally`), as linhas de índice (`tables.IndexTables`) e os deslocamentos dentro do
payload, que só existem porque o writer os conta enquanto escreve.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contextmap.semantic_mapping.entity import CandidateRejection, Entity, SemanticMapId
from contextmap.semantic_mapping.run_artifact.debug import write_entity_debug
from contextmap.semantic_mapping.run_artifact.layout import (
    COUNTS,
    DISTRIBUTIONS,
    ENTITIES,
    ENTITY_INDEX,
    EVIDENCE_INDEX,
    GEOMETRY_INDEX,
    OBSERVATION_INDEX,
    PAYLOAD,
    REJECTED,
    RUNTIME,
    SCHEMA_VERSION,
    SEMANTIC_STATE,
    TEMPORAL_STATE,
    MappingDebugLevel,
    MappingRunArtifactError,
    SemanticMappingRunId,
    encode_document,
    encode_line,
)
from contextmap.semantic_mapping.run_artifact.manifest import (
    MappingRunLineage,
    SemanticMappingRunManifest,
    load_manifest,
)
from contextmap.semantic_mapping.run_artifact.metrics import RunTally
from contextmap.semantic_mapping.run_artifact.tables import IndexTables, encode_rejection
from contextmap.semantic_mapping.serialization import ENTITY_SCHEMA_VERSION, encode_entity
from contextmap.shared import AtomicRunDirectory, RunDirectoryError


class SemanticMappingRunWriter:
    """Builds an immutable Semantic Mapping run artifact on the local filesystem.

    Entities are streamed to disk as they are produced, so the run is not bounded by memory, and
    the run is published atomically when the stream ends.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: SemanticMappingRunId,
        run_index: int,
        semantic_map_id: SemanticMapId,
        lineage: MappingRunLineage,
        code_version: str,
        code_digest: str,
        debug_level: MappingDebugLevel = MappingDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/semantic_mapping``); the writer computes
                no path, creates the directory atomically on writing and refuses to replace one
                that exists.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run among the caller's runs of the same sequence,
                supplied by the caller and recorded as given.
            semantic_map_id: The semantic map the run holds.
            lineage: The explicit upstream selection.
            code_version: Code revision that produced the run.
            code_digest: Digest of the code that produced the run, recorded exactly as given;
                the caller owns how it is computed (for example a commit or a build hash).
            debug_level: Amount of non-contractual debug evidence to persist.

        Raises:
            ValueError: If the code digest is empty.
        """
        if not code_digest.strip():
            raise ValueError("code_digest must not be empty")
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._semantic_map_id = semantic_map_id
        self._lineage = lineage
        self._code_version = code_version
        self._code_digest = code_digest
        self._debug_level = debug_level
        self._final_dir = output_dir

    def write(
        self,
        entities: Iterable[Entity],
        *,
        rejections: Iterable[CandidateRejection] = (),
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> SemanticMappingRunManifest:
        """Persist a run atomically, consuming the entities as a stream.

        Args:
            entities: The entities, sorted by identity.
            rejections: The candidates that could not become entities, sorted by support, kept as
                explicit evidence.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every quality
                measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            MappingRunArtifactError: If a run already exists at the target path, the entities are
                not strictly sorted by identity, an entity belongs to another semantic map or
                breaks the lineage or the run's single policy, a candidate is both an entity and a
                rejection, or the write fails.
        """
        tally = RunTally(self._semantic_map_id, self._lineage)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                tables = self._stream(run, entities, tally)
                rejected = [encode_rejection(item) for item in rejections]
                tally.check_rejections([row["fusion_support_id"] for row in rejected])
                self._write_tables(run, tally, tables, rejected)
                self._write_metrics(run, tally, rejected, warnings, runtime)
                run.publish(
                    manifest=self._manifest_record(tally, rejected, warnings),
                    readme=self._render_readme(tally, rejected),
                )
        except RunDirectoryError as error:
            raise MappingRunArtifactError(str(error)) from error
        return load_manifest(self._final_dir)

    def _stream(
        self, run: AtomicRunDirectory, entities: Iterable[Entity], tally: RunTally
    ) -> IndexTables:
        """Write the payload as it is produced, checking and indexing each entity in one pass."""
        tables = IndexTables()
        with run.open_binary(ENTITIES) as out:
            offset = 0
            for entity in entities:
                tally.check(entity)
                line = encode_line(encode_entity(entity)) + b"\n"
                # O stream não expõe posição: os deslocamentos são contados aqui.
                tables.add(entity, offset=offset, length=len(line) - 1)
                tally.add(entity)
                out.write(line)
                offset += len(line)
                write_entity_debug(run, entity, level=self._debug_level, number=tally.entity_count)
        tally.output_sizes[ENTITIES] = offset
        return tables

    def _write_tables(
        self,
        run: AtomicRunDirectory,
        tally: RunTally,
        tables: IndexTables,
        rejected: list[dict[str, Any]],
    ) -> None:
        """Write every index table and record the size each one contributed."""
        written = {
            ENTITY_INDEX: tables.entity_rows,
            GEOMETRY_INDEX: tables.geometry_rows,
            EVIDENCE_INDEX: tables.evidence_rows,
            OBSERVATION_INDEX: tables.observation_rows,
            SEMANTIC_STATE: tables.semantic_rows,
            TEMPORAL_STATE: tables.temporal_rows,
            REJECTED: rejected,
        }
        for path, rows in written.items():
            text = "".join(encode_line(row).decode() + "\n" for row in rows)
            run.write_text(path, text)
            tally.output_sizes[path] = len(text.encode())

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        tally: RunTally,
        rejected: list[dict[str, Any]],
        warnings: Sequence[str],
        runtime: Mapping[str, float | int | None] | None,
    ) -> None:
        run.write_text(COUNTS, encode_document(tally.counts(rejected, len(warnings))))
        run.write_text(DISTRIBUTIONS, encode_document(tally.distributions()))
        sizes = dict(sorted(tally.output_sizes.items()))
        run.write_text(
            PAYLOAD, encode_document({"files": sizes, "total_bytes": sum(sizes.values())})
        )
        if runtime is not None:
            run.write_text(RUNTIME, encode_document(dict(runtime)))

    def _manifest_record(
        self, tally: RunTally, rejected: list[dict[str, Any]], warnings: Sequence[str]
    ) -> dict[str, Any]:
        lineage = self._lineage
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "semantic_map_id": str(self._semantic_map_id),
            "lineage": {
                "sequence_artifact_id": lineage.sequence_artifact_id,
                "geometric_map_id": str(lineage.geometric_map_id),
                "fusion_run_id": str(lineage.fusion_run_id),
                "fusion_schema_version": lineage.fusion_schema_version,
                "fusion_artifact_digest": lineage.fusion_artifact_digest,
                "association_run_ids": list(lineage.association_run_ids),
                "region_artifact_ids": list(lineage.region_artifact_ids),
                "semantic_interpretation_artifact_ids": list(
                    lineage.semantic_interpretation_artifact_ids
                ),
                "visual_artifact_ids": list(lineage.visual_artifact_ids),
                "point_representation_run_ids": [
                    str(item) for item in lineage.point_representation_run_ids
                ],
            },
            "policies": {
                "materialization_policy_id": tally.materialization_policy_id,
                "identity_policy_id": tally.identity_policy_id,
                "configuration_fingerprint": tally.configuration_fingerprint,
            },
            "code_version": self._code_version,
            "code_digest": self._code_digest,
            "counts": {"entities": tally.entity_count, "rejected_candidates": len(rejected)},
            "warnings": list(warnings),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "entity_schema_version": ENTITY_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self, tally: RunTally, rejected: list[dict[str, Any]]) -> str:
        return (
            f"# Semantic mapping run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Semantic map: `{self._semantic_map_id}`\n"
            f"- Fusion run: `{self._lineage.fusion_run_id}`\n"
            f"- Materialization policy: `{tally.materialization_policy_id}`\n"
            f"- Entities: {tally.entity_count}, rejected candidates: {len(rejected)}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `__DEBUG__/` is human "
            "evidence and no "
            "downstream stage may depend on it. An entity id is local to this semantic map. "
            "Entities keep every hypothesis, conflict, abstention and unscored signal, and no "
            "same-object decision has been made yet.\n"
        )
