"""Construção e publicação atômica de um artifact de Semantic Fusion.

Os outcomes são transmitidos para o disco conforme são produzidos: todo arquivo de
``outputs/``, índices derivados incluídos, é escrito linha a linha, então o writer mantém
um outcome por vez mais os agregados de nível de run. A publicação segue
:class:`~contextmap.shared.AtomicRunDirectory`: uma escrita interrompida nunca se parece
com um run terminado, e um run terminado nunca é modificado.

O writer grava **exatamente** no ``output_dir`` que o chamador entrega. Ele não calcula
caminho, não aloca identidade e não mantém registro: identidade e índice do run vêm de
quem o chamou. Nada a montante é duplicado -- a geometria vira deltas posicionais, e
claims, scores, features, qualidade e representações 3D só são referenciadas.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from contextmap.semantic_fusion.fused_evidence import FusionOutcome
from contextmap.semantic_fusion.run_artifact.debug import write_support_debug
from contextmap.semantic_fusion.run_artifact.layout import (
    CONTRIBUTION_INDEX,
    COUNTS,
    DISTRIBUTIONS,
    EVIDENCE,
    EXCLUDED,
    GROUPS,
    HYPOTHESIS_INDEX,
    PAYLOAD,
    RUNTIME,
    SCHEMA_VERSION,
    STREAMED_OUTPUTS,
    SUPPORT_INDEX,
    SUPPORTS,
    FusionRunArtifactError,
    SemanticFusionDebugLevel,
    SemanticFusionRunId,
    encode_document,
    encode_line,
)
from contextmap.semantic_fusion.run_artifact.manifest import (
    FusionRunLineage,
    SemanticFusionRunManifest,
    load_manifest,
)
from contextmap.semantic_fusion.run_artifact.metrics import RunTally
from contextmap.semantic_fusion.serialization import encode_fused_evidence, encode_fusion_support
from contextmap.semantic_fusion.supports import ExcludedObservation
from contextmap.shared import AtomicRunDirectory, RunDirectoryError


class SemanticFusionRunWriter:
    """Builds an immutable Semantic Fusion run artifact on the local filesystem.

    Outcomes are streamed to disk as they are produced: every ``outputs/`` file, the derived
    indexes included, is written line by line, so the writer holds one outcome at a time plus
    the run-level aggregates behind ``metrics/`` and the manifest (one entry per distinct claim,
    physical observation, inference result and evidence identity, and four integers per
    support). A single support's ``FusedEvidence`` is still encoded whole. The run is published
    atomically when the stream ends.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: SemanticFusionRunId,
        run_index: int,
        lineage: FusionRunLineage,
        code_version: str,
        debug_level: SemanticFusionDebugLevel = SemanticFusionDebugLevel.NONE,
    ) -> None:
        """Create a writer for a new run.

        Args:
            output_dir: The final directory of the artifact. The caller chooses it (in the
                runtime, ``<workspace>/<dataset>/<run>/semantic_fusion``); the writer
                computes no path, creates the directory atomically on finalization and
                refuses to replace one that exists.
            sequence_name: Name of the sequence the run processed.
            run_id: Identity of the run, supplied by the caller and never allocated here.
            run_index: Ordinal of this run among the caller's runs of the same sequence,
                supplied by the caller and recorded as given.
            lineage: The explicit upstream selection.
            code_version: Code revision that produced the run.
            debug_level: Amount of non-contractual debug evidence to persist.
        """
        self._sequence_name = sequence_name
        self._run_id = run_id
        self._run_index = run_index
        self._lineage = lineage
        self._code_version = code_version
        self._debug_level = debug_level
        self._final_dir = output_dir

    def write(
        self,
        outcomes: Iterable[FusionOutcome],
        *,
        excluded: Iterable[ExcludedObservation] = (),
        warnings: Sequence[str] = (),
        runtime: Mapping[str, float | int | None] | None = None,
    ) -> SemanticFusionRunManifest:
        """Persist a run atomically, consuming the outcomes as a stream.

        Args:
            outcomes: The supports with their fused evidence, sorted by support identity.
            excluded: Observations that took part in no support, kept as skipped evidence.
            warnings: Human-readable warnings to record.
            runtime: Figures such as seconds and peak memory, recorded apart from every
                quality measure and only when the caller measured them.

        Returns:
            The manifest of the finalized run.

        Raises:
            FusionRunArtifactError: If a run already exists at the target path, the outcomes are
                not strictly sorted by support identity, a support and its evidence disagree, an
                outcome breaks the lineage or the run's single policy, or the write fails.
        """
        tally = RunTally(self._lineage)
        try:
            with AtomicRunDirectory(self._final_dir) as run:
                self._stream(run, outcomes, excluded, tally)
                self._write_metrics(run, tally, warnings, runtime)
                run.publish(
                    manifest=self._manifest_record(tally, warnings),
                    readme=self._render_readme(tally),
                )
        except RunDirectoryError as error:
            raise FusionRunArtifactError(str(error)) from error
        return load_manifest(self._final_dir)

    def _stream(
        self,
        run: AtomicRunDirectory,
        outcomes: Iterable[FusionOutcome],
        excluded: Iterable[ExcludedObservation],
        tally: RunTally,
    ) -> None:
        with ExitStack() as stack:
            out = {
                path: _JsonLinesOut(stack.enter_context(run.open_binary(path)))
                for path in STREAMED_OUTPUTS
            }
            for outcome in outcomes:
                tally.check(outcome)
                support_offset, support_length = out[SUPPORTS].write(
                    encode_fusion_support(outcome.support)
                )
                evidence_offset, evidence_length = out[EVIDENCE].write(
                    encode_fused_evidence(outcome.evidence)
                )
                tally.add(outcome)
                out[SUPPORT_INDEX].write(
                    support_index_row(
                        outcome,
                        support_offset=support_offset,
                        support_length=support_length,
                        evidence_offset=evidence_offset,
                        evidence_length=evidence_length,
                    )
                )
                out[GROUPS].write_all(group_rows(outcome))
                out[CONTRIBUTION_INDEX].write_all(contribution_rows(outcome))
                out[HYPOTHESIS_INDEX].write_all(hypothesis_rows(outcome))
                write_support_debug(
                    run,
                    outcome,
                    level=self._debug_level,
                    support_number=tally.support_count,
                )
            for item in excluded:
                out[EXCLUDED].write(encode_excluded(item))
                tally.excluded_count += 1
        tally.output_sizes.update((path, lines.size) for path, lines in out.items())

    def _write_metrics(
        self,
        run: AtomicRunDirectory,
        tally: RunTally,
        warnings: Sequence[str],
        runtime: Mapping[str, float | int | None] | None,
    ) -> None:
        run.write_text(COUNTS, encode_document(tally.counts(tally.excluded_count, len(warnings))))
        run.write_text(DISTRIBUTIONS, encode_document(tally.distributions()))
        sizes = dict(sorted(tally.output_sizes.items()))
        run.write_text(
            PAYLOAD, encode_document({"files": sizes, "total_bytes": sum(sizes.values())})
        )
        if runtime is not None:
            run.write_text(RUNTIME, encode_document(dict(runtime)))

    def _manifest_record(self, tally: RunTally, warnings: Sequence[str]) -> dict[str, Any]:
        lineage = self._lineage
        lineage_record: dict[str, Any] = {
            "sequence_artifact_id": lineage.sequence_artifact_id,
            "geometric_map_id": str(lineage.geometric_map_id),
            "association_run_ids": list(lineage.association_run_ids),
            "region_artifact_ids": list(lineage.region_artifact_ids),
            "semantic_interpretation_artifact_ids": list(
                lineage.semantic_interpretation_artifact_ids
            ),
            "visual_artifact_ids": list(lineage.visual_artifact_ids),
            "point_representation_run_ids": [
                str(item) for item in lineage.point_representation_run_ids
            ],
        }
        # Ausência continua a significar que nenhum run de scoring foi selecionado. Omitir o
        # campo vazio preserva a identidade dos artifacts existentes; quando há scores brutos,
        # a seleção vira parte contratual da linhagem.
        if lineage.semantic_scoring_run_ids:
            lineage_record["semantic_scoring_run_ids"] = [
                str(item) for item in lineage.semantic_scoring_run_ids
            ]
        return {
            "run_id": str(self._run_id),
            "run_index": self._run_index,
            "sequence_name": self._sequence_name,
            "lineage": lineage_record,
            "policies": {
                "grouping_policy_id": tally.grouping_policy_id,
                "support_policy_id": tally.support_policy_id,
                "support_configuration_fingerprint": tally.support_fingerprint,
                "fusion_policy_id": tally.fusion_policy_id,
                "fusion_configuration_fingerprint": tally.fusion_fingerprint,
            },
            "evidence_identities": {
                channel: sorted(identities)
                for channel, identities in sorted(tally.identities.items())
            },
            "code_version": self._code_version,
            "counts": {
                "supports": tally.support_count,
                "fused_evidence": tally.support_count,
                "contributions": tally.contribution_count,
                "excluded_observations": tally.excluded_count,
            },
            "warnings": list(warnings),
            "debug_level": self._debug_level.value,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _render_readme(self, tally: RunTally) -> str:
        return (
            f"# Semantic fusion run {self._run_index:04d}\n"
            "\n"
            f"- Run ID: `{self._run_id}`\n"
            f"- Geometric map: `{self._lineage.geometric_map_id}`\n"
            f"- Fusion policy: `{tally.fusion_policy_id}`\n"
            f"- Supports: {tally.support_count}, contributions: {tally.contribution_count}, "
            f"excluded observations: {tally.excluded_count}\n"
            "\n"
            "Contractual data is in `outputs/` and `metrics/`; `__DEBUG__/` is human "
            "evidence and no downstream stage may depend on it. Fused evidence is belief "
            "in formation: every "
            "hypothesis, conflict, abstention and unscored claim is kept, and no entity identity "
            "exists yet.\n"
        )


class _JsonLinesOut:
    """One streamed JSON Lines file of the run, with the byte count its readers index by."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.size = 0

    def write(self, record: Mapping[str, Any]) -> tuple[int, int]:
        """Append one record.

        Returns:
            The record's byte offset and its length without the line break.
        """
        line = encode_line(record)
        # O stream não expõe posição: o deslocamento é contado aqui.
        offset = self.size
        self._stream.write(line + b"\n")
        self.size += len(line) + 1
        return offset, len(line)

    def write_all(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append every record, in order."""
        for record in records:
            self.write(record)


def support_index_row(
    outcome: FusionOutcome,
    *,
    support_offset: int,
    support_length: int,
    evidence_offset: int,
    evidence_length: int,
) -> dict[str, Any]:
    """Build the row that lets a support and its evidence be read by identity.

    Args:
        outcome: The support and its fused evidence.
        support_offset: Byte offset of the encoded support in its JSON Lines file.
        support_length: Length of that record, without the line break.
        evidence_offset: Byte offset of the encoded fused evidence.
        evidence_length: Length of that record, without the line break.

    Returns:
        A JSON-ready row carrying the offsets the reader seeks to, so one support is read
        without loading the others.
    """
    support, evidence = outcome.support, outcome.evidence
    return {
        "fusion_support_id": str(support.fusion_support_id),
        "fused_evidence_id": str(evidence.fused_evidence_id),
        "support_offset": support_offset,
        "support_length": support_length,
        "evidence_offset": evidence_offset,
        "evidence_length": evidence_length,
        "spatial_observation_ids": [str(item) for item in support.spatial_observation_ids],
        "physical_observation_ids": [
            str(g.physical_observation_id) for g in evidence.physical_observation_groups
        ],
    }


def group_rows(outcome: FusionOutcome) -> Iterator[dict[str, Any]]:
    """Yield one row per physical observation of a support, for ``physical-observation-groups``.

    Args:
        outcome: The support and its fused evidence.

    Yields:
        JSON-ready rows that keep physical frames readable apart from inference.
    """
    for group in outcome.evidence.physical_observation_groups:
        yield {
            "fusion_support_id": str(outcome.support.fusion_support_id),
            "physical_observation_id": str(group.physical_observation_id),
            "acquisition_timestamp": group.acquisition_timestamp.to_record(),
            "spatial_observation_ids": [str(i) for i in group.spatial_observation_ids],
            "semantic_interpretation_artifact_ids": [
                str(i) for i in group.semantic_interpretation_artifact_ids
            ],
            "region_artifact_ids": [str(i) for i in group.region_artifact_ids],
        }


def contribution_rows(outcome: FusionOutcome) -> Iterator[dict[str, Any]]:
    """Yield one row per contribution of a support, for the contribution index.

    Args:
        outcome: The support and its fused evidence.

    Yields:
        JSON-ready rows that let a contribution be found by identity without a full scan.
    """
    for item in outcome.evidence.contributions:
        yield {
            "contribution_id": str(item.contribution_id),
            "fusion_support_id": str(outcome.support.fusion_support_id),
            "spatial_observation_id": str(item.spatial_observation_id),
            "physical_observation_id": str(item.physical_observation_id),
            "semantic_interpretation_artifact_id": item.semantic_interpretation_artifact_id,
            "region_artifact_id": str(item.region_artifact_id),
            "region_id": str(item.region_id),
            "claim_ids": [str(ref.claim_id) for ref in item.claim_refs],
            "geometry_count": len(item.geometry_support),
            "has_observation_quality": item.observation_quality is not None,
        }


def hypothesis_rows(outcome: FusionOutcome) -> Iterator[dict[str, Any]]:
    """Yield one row per claim of each hypothesis of a support, for the hypothesis index.

    Args:
        outcome: The support and its fused evidence.

    Yields:
        JSON-ready rows carrying the stance and the upstream identities of each claim, so the
        evidence of a label is queryable without decoding every fused evidence.
    """
    support, evidence = outcome.support, outcome.evidence
    contributions = {item.contribution_id: item for item in evidence.contributions}
    for hypothesis in evidence.hypotheses:
        for item in hypothesis.evidence:
            contribution = contributions[item.contribution_id]
            yield {
                "fusion_support_id": str(support.fusion_support_id),
                "fused_evidence_id": str(evidence.fused_evidence_id),
                "hypothesis_id": str(hypothesis.hypothesis_id),
                "label": hypothesis.label,
                "contribution_id": str(item.contribution_id),
                "claim_id": str(item.claim_id),
                "stance": item.stance.value,
                "role": item.role.value,
                "physical_observation_id": str(contribution.physical_observation_id),
                "semantic_interpretation_artifact_id": str(
                    contribution.semantic_interpretation_artifact_id
                ),
                "region_artifact_id": str(contribution.region_artifact_id),
                "spatial_observation_id": str(contribution.spatial_observation_id),
            }


def encode_excluded(item: ExcludedObservation) -> dict[str, Any]:
    """Encode one observation that took part in no support, with why.

    Args:
        item: The excluded observation.

    Returns:
        A JSON-ready record: exclusion is contractual, never a silent drop.
    """
    return {
        "spatial_observation_id": str(item.spatial_observation_id),
        "geometry_count": item.geometry_count,
        "minimum_geometry_count": item.minimum_geometry_count,
    }
