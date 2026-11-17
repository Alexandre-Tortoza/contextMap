"""Leitura de um run de Semantic Fusion finalizado, a partir do seu próprio diretório.

O reader abre um run sem registro externo, sem NumPy, sem runtime de percepção e sem
biblioteca de modelo. Um suporte, uma evidência fundida ou uma contribuição é lida **por
identidade**, pelos deslocamentos que os índices guardam, sem carregar as outras.

Todo registro é decodificado pelos construtores dos contratos, então um registro adulterado
é recusado em vez de confiado. ``__DEBUG__/`` nunca é fonte válida de leitura.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from contextmap.semantic_fusion.evidence import EvidenceContribution
from contextmap.semantic_fusion.fused_evidence import FusedEvidence, FusionOutcome
from contextmap.semantic_fusion.identity import EvidenceContributionId, FusionSupportId
from contextmap.semantic_fusion.run_artifact.layout import (
    CONTRIBUTION_INDEX,
    EVIDENCE,
    EXCLUDED,
    SUPPORT_INDEX,
    SUPPORTS,
    FusionRunArtifactError,
)
from contextmap.semantic_fusion.run_artifact.manifest import (
    SemanticFusionRunManifest,
    load_manifest,
)
from contextmap.semantic_fusion.serialization import decode_fused_evidence, decode_fusion_support
from contextmap.semantic_fusion.supports import ExcludedObservation, FusionSupport
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import check_file_inventory

_T = TypeVar("_T")


class SemanticFusionRunReader:
    """Reads a finalized Semantic Fusion run from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a run.

        Args:
            run_dir: Path to the run's directory; no registry or other file outside it is
                needed, and no perception or model runtime.

        Raises:
            IncompleteFusionRunArtifactError: If ``manifest.json`` is missing.
            FusionRunArtifactError: If the schema version is not understood.
        """
        self._root = run_dir
        self._manifest = load_manifest(run_dir)
        self._rows: dict[FusionSupportId, dict[str, Any]] | None = None
        self._observation_support: dict[SpatialObservationId, FusionSupportId] | None = None
        self._contribution_support: dict[EvidenceContributionId, FusionSupportId] | None = None

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The run's manifest."""
        return self._manifest

    def support_ids(self) -> list[FusionSupportId]:
        """The identities of every support, in persisted order."""
        return list(self._support_rows())

    def support(self, support_id: FusionSupportId) -> FusionSupport:
        """Read one support by identity without loading the others.

        Raises:
            FusionRunArtifactError: If the run has no such support or its record is malformed.
        """
        row = self._row(support_id)
        raw = self._read_at(SUPPORTS, row["support_offset"], row["support_length"])
        return _decode(raw, decode_fusion_support, f"support {support_id!r}")

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support without loading the others.

        Raises:
            FusionRunArtifactError: If the run has no such support or its record is malformed.
        """
        row = self._row(support_id)
        raw = self._read_at(EVIDENCE, row["evidence_offset"], row["evidence_length"])
        return _decode(raw, decode_fused_evidence, f"fused evidence of {support_id!r}")

    def iter_outcomes(self) -> Iterator[FusionOutcome]:
        """Iterate every support with its fused evidence, in persisted order."""
        with (
            (self._root / SUPPORTS).open("rb") as supports,
            (self._root / EVIDENCE).open("rb") as evidence,
        ):
            for support_line, evidence_line in zip(supports, evidence, strict=True):
                support = _decode(support_line, decode_fusion_support, "support")
                fused = _decode(evidence_line, decode_fused_evidence, "fused evidence")
                yield FusionOutcome(support=support, evidence=fused)

    def support_of_observation(
        self, observation_id: SpatialObservationId
    ) -> FusionSupportId | None:
        """The support a spatial observation was accumulated in, or ``None`` if it was excluded."""
        if self._observation_support is None:
            self._observation_support = {
                SpatialObservationId(observation_id): support_id
                for support_id, row in self._support_rows().items()
                for observation_id in row["spatial_observation_ids"]
            }
        return self._observation_support.get(observation_id)

    def contribution(self, contribution_id: EvidenceContributionId) -> EvidenceContribution:
        """Resolve a contribution to its exact upstream references.

        Args:
            contribution_id: A contribution named by a hypothesis, a conflict or an index.

        Raises:
            FusionRunArtifactError: If the run has no such contribution.
        """
        if self._contribution_support is None:
            self._contribution_support = {
                EvidenceContributionId(row["contribution_id"]): FusionSupportId(
                    row["fusion_support_id"]
                )
                for row in self._read_rows(CONTRIBUTION_INDEX)
            }
        support_id = self._contribution_support.get(contribution_id)
        if support_id is None:
            raise FusionRunArtifactError(f"unknown contribution in this run: {contribution_id!r}")
        for item in self.fused_evidence(support_id).contributions:
            if item.contribution_id == contribution_id:
                return item
        raise FusionRunArtifactError(
            f"contribution {contribution_id!r} is missing from its support"
        )

    def excluded_observations(self) -> list[ExcludedObservation]:
        """The observations that took part in no support, with why."""
        return [
            ExcludedObservation(
                spatial_observation_id=SpatialObservationId(row["spatial_observation_id"]),
                geometry_count=row["geometry_count"],
                minimum_geometry_count=row["minimum_geometry_count"],
            )
            for row in self._read_rows(EXCLUDED)
        ]

    def read_record(self, relative_path: str) -> dict[str, Any]:
        """Read a JSON record from ``outputs/`` or ``metrics/``.

        Raises:
            FusionRunArtifactError: If the path is not a JSON record of a contractual
                directory; ``__DEBUG__/`` is never a valid source.
        """
        if not relative_path.endswith(".json") or not relative_path.startswith(
            ("outputs/", "metrics/")
        ):
            raise FusionRunArtifactError(f"not a contractual JSON record: {relative_path!r}")
        record: dict[str, Any] = json.loads(
            (self._root / relative_path).read_text(encoding="utf-8")
        )
        return record

    def verify_integrity(self) -> list[str]:
        """Check the file inventory against what is actually on disk.

        Returns:
            Human-readable problems; empty means the run is intact.
        """
        return check_file_inventory(self._root, self._manifest.file_inventory)

    def _support_rows(self) -> dict[FusionSupportId, dict[str, Any]]:
        if self._rows is None:
            self._rows = {
                FusionSupportId(row["fusion_support_id"]): row
                for row in self._read_rows(SUPPORT_INDEX)
            }
        return self._rows

    def _row(self, support_id: FusionSupportId) -> dict[str, Any]:
        row = self._support_rows().get(support_id)
        if row is None:
            raise FusionRunArtifactError(f"unknown support in this run: {support_id!r}")
        return row

    def _read_rows(self, relative_path: str) -> list[dict[str, Any]]:
        text = (self._root / relative_path).read_text(encoding="utf-8")
        try:
            return [json.loads(line) for line in text.splitlines() if line]
        except ValueError as error:
            raise FusionRunArtifactError(f"malformed index {relative_path}: {error}") from error

    def _read_at(self, relative_path: str, offset: int, length: int) -> bytes:
        with (self._root / relative_path).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise FusionRunArtifactError(
                f"{relative_path} is truncated: {length} bytes expected at {offset}, "
                f"found {len(data)}"
            )
        return data


def _decode(raw: bytes, decoder: Callable[[Mapping[str, Any]], _T], what: str) -> _T:
    try:
        return decoder(json.loads(raw))
    except (ValueError, KeyError, TypeError) as error:
        raise FusionRunArtifactError(f"malformed {what}: {error}") from error
