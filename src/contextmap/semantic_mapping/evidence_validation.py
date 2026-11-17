"""Integridade das referências de evidência de uma entidade, reportada e nunca levantada.

Uma entidade é um registro compacto que **referencia** a evidência que a criou, então a única
garantia possível é poder conferir essas referências contra os artifacts que elas nomeiam. Este
módulo faz isso e **relata** cada problema como um `EvidenceIntegrityIssue`: artifact ausente, com
outra identidade ou versão, alterado desde a materialização, corrompido na própria verificação,
linhagem incompatível, evidência inexistente, listas divergentes ou geometria fora do suporte.

Relatar em vez de levantar é deliberado: validar um mapa inteiro deve produzir o conjunto completo
de problemas, não parar no primeiro. Quando um fato não pode ser lido, a comparação das listas que
dependem dele é suprimida, porque comparar sem os fatos só geraria falso positivo.

A travessia da proveniência, que **levanta** porque é usada para auditar uma entidade por vez,
está em `evidence_trace.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import Enum

from contextmap.geometric_mapping import GeometrySource
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_fusion import (
    FusedEvidence,
    FusionRunArtifactError,
    FusionSupportId,
    SemanticFusionRunId,
    SemanticFusionRunManifest,
)
from contextmap.semantic_mapping.entity import Entity, EntityReference
from contextmap.semantic_mapping.evidence import (
    EntityFeatureRef,
    FusedEvidenceRef,
    feature_refs_of,
    fusion_artifact_digest,
)
from contextmap.semantic_mapping.fused_evidence_source import FusedEvidenceSource
from contextmap.semantic_mapping.geometry import GeometryResolutionError, resolve_geometry
from contextmap.sensor_association import SpatialObservationId


class EvidenceIntegrityKind(Enum):
    """Why an evidence reference of an entity cannot be trusted.

    Attributes:
        MISSING_ARTIFACT: A referenced fusion run was not offered.
        ARTIFACT_MISMATCH: The offered run has another identity or schema version.
        STALE_REFERENCE: The run's content is not what the entity was materialized from.
        CORRUPT_ARTIFACT: The run fails its own integrity check.
        MISSING_REFERENCE: The fused evidence is not in the run, or is another one.
        INCOMPATIBLE_LINEAGE: The run was built over another geometric map than the entity's, or
            over another sequence than the one the entity was materialized from.
        OBSERVATION_MISMATCH: The observations listed differ from the fused evidence's.
        FEATURE_MISMATCH: The visual features listed differ from the fused evidence's.
        POINT_REPRESENTATION_MISMATCH: The 3D representations listed differ from the fused
            evidence's.
        GEOMETRY_OUTSIDE_SUPPORT: The fused evidence sees geometry the entity does not support.
        MISSING_GEOMETRY: A geometry reference of the entity does not resolve in the map.
    """

    MISSING_ARTIFACT = "missing_artifact"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    STALE_REFERENCE = "stale_reference"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    MISSING_REFERENCE = "missing_reference"
    INCOMPATIBLE_LINEAGE = "incompatible_lineage"
    OBSERVATION_MISMATCH = "observation_mismatch"
    FEATURE_MISMATCH = "feature_mismatch"
    POINT_REPRESENTATION_MISMATCH = "point_representation_mismatch"
    GEOMETRY_OUTSIDE_SUPPORT = "geometry_outside_support"
    MISSING_GEOMETRY = "missing_geometry"


@dataclass(frozen=True, kw_only=True)
class EvidenceIntegrityIssue:
    """One way an evidence reference of an entity is wrong.

    Attributes:
        entity: The entity whose reference is wrong.
        kind: What is wrong.
        detail: A deterministic, human-readable explanation.
    """

    entity: EntityReference
    kind: EvidenceIntegrityKind
    detail: str


def validate_entity_evidence(
    entity: Entity,
    *,
    fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
    geometry: GeometrySource | None = None,
) -> tuple[EvidenceIntegrityIssue, ...]:
    """Check every evidence reference of an entity against the artifacts it names.

    The checks never raise on a bad reference; they report it. An empty result means the
    references resolve, the artifacts are what the entity was materialized from and the entity
    lists exactly the observations, features and 3D representations its fused evidence carries.

    Args:
        entity: The entity to check.
        fusion_runs: The persisted fusion runs to check against, by run identity. Only the
            runs the entity references are read; nothing else is loaded.
        geometry: The read boundary of the geometric map, to also check that every geometry
            reference of the entity resolves; ``None`` skips that check.

    Returns:
        The issues found, in a deterministic order.
    """
    checker = _Checker(entity)
    for ref in entity.evidence.fused_evidence:
        checker.check(ref, fusion_runs.get(ref.fusion_run_id))
    checker.compare_lists()
    if geometry is not None:
        checker.check_geometry(geometry)
    return tuple(checker.issues)


class _VerifiedOnce:
    """Runs the integrity check of a run once, however many entities reference it."""

    def __init__(self, source: FusedEvidenceSource) -> None:
        self._source = source
        self._problems: list[str] | None = None

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The manifest of the run."""
        return self._source.manifest

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support."""
        return self._source.fused_evidence(support_id)

    def verify_integrity(self) -> list[str]:
        """Check the run's files once and remember the result."""
        if self._problems is None:
            self._problems = self._source.verify_integrity()
        return self._problems


def validate_evidence_of_entities(
    entities: Iterable[Entity],
    *,
    fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource],
    geometry: GeometrySource | None = None,
) -> tuple[EvidenceIntegrityIssue, ...]:
    """Check the evidence references of many entities, verifying each run's files only once.

    Args:
        entities: The entities to check.
        fusion_runs: The persisted fusion runs to check against, by run identity.
        geometry: The read boundary of the geometric map, to also check the geometry.

    Returns:
        The issues found for every entity, in the order the entities were given.
    """
    verified = {run_id: _VerifiedOnce(source) for run_id, source in fusion_runs.items()}
    issues: list[EvidenceIntegrityIssue] = []
    for entity in entities:
        issues.extend(validate_entity_evidence(entity, fusion_runs=verified, geometry=geometry))
    return tuple(issues)


class _Checker:
    """Collects the facts of the fused evidence and compares them with the entity's links."""

    def __init__(self, entity: Entity) -> None:
        self._entity = entity
        self._support = set(entity.geometry.geometry_refs)
        self._complete = True
        self._spatial: set[SpatialObservationId] = set()
        self._physical: set[SourceObservationId] = set()
        self._features: set[EntityFeatureRef] = set()
        self._representations: set[tuple[str, str]] = set()
        self.issues: list[EvidenceIntegrityIssue] = []

    def check(self, ref: FusedEvidenceRef, source: FusedEvidenceSource | None) -> None:
        """Check one fused evidence reference and collect what its evidence carries."""
        if source is None:
            self._issue(
                EvidenceIntegrityKind.MISSING_ARTIFACT,
                f"fusion run {ref.fusion_run_id!r} was not offered",
                blocks_comparison=True,
            )
            return
        manifest = source.manifest
        if manifest.run_id != ref.fusion_run_id or manifest.schema_version != (
            ref.fusion_schema_version
        ):
            self._issue(
                EvidenceIntegrityKind.ARTIFACT_MISMATCH,
                f"the entity references run {ref.fusion_run_id!r} at schema "
                f"{ref.fusion_schema_version!r}, the offered run is {manifest.run_id!r} at "
                f"{manifest.schema_version!r}",
                blocks_comparison=True,
            )
            return
        if fusion_artifact_digest(manifest) != ref.fusion_artifact_digest:
            self._issue(
                EvidenceIntegrityKind.STALE_REFERENCE,
                f"the content of fusion run {ref.fusion_run_id!r} changed since the entity was "
                f"materialized",
                blocks_comparison=True,
            )
            return
        for problem in source.verify_integrity():
            self._issue(EvidenceIntegrityKind.CORRUPT_ARTIFACT, problem)
        if manifest.lineage.geometric_map_id != self._entity.geometry.geometric_map_id:
            self._issue(
                EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE,
                f"fusion run {ref.fusion_run_id!r} was built over map "
                f"{manifest.lineage.geometric_map_id!r}, but the entity's geometry is in "
                f"{self._entity.geometry.geometric_map_id!r}",
            )
        if manifest.lineage.sequence_artifact_id != ref.sequence_artifact_id:
            self._issue(
                EvidenceIntegrityKind.INCOMPATIBLE_LINEAGE,
                f"fusion run {ref.fusion_run_id!r} was built over sequence "
                f"{manifest.lineage.sequence_artifact_id!r}, but the entity was materialized from "
                f"sequence {ref.sequence_artifact_id!r}",
            )
        self._collect(ref, source)

    def compare_lists(self) -> None:
        """Compare what the entity lists with what the fused evidence carries."""
        if not self._complete:
            return
        links = self._entity.evidence
        self._compare(
            EvidenceIntegrityKind.OBSERVATION_MISMATCH,
            "spatial observations",
            set(links.spatial_observation_ids),
            self._spatial,
        )
        self._compare(
            EvidenceIntegrityKind.OBSERVATION_MISMATCH,
            "physical observations",
            set(links.physical_observation_ids),
            self._physical,
        )
        self._compare(
            EvidenceIntegrityKind.FEATURE_MISMATCH,
            "visual features",
            set(links.visual_feature_refs),
            self._features,
        )
        self._compare(
            EvidenceIntegrityKind.POINT_REPRESENTATION_MISMATCH,
            "3D representations",
            {(ref.run_id, ref.representation_id) for ref in links.point_representation_refs},
            self._representations,
        )

    def check_geometry(self, geometry: GeometrySource) -> None:
        """Check that every geometry reference of the entity resolves in the map."""
        try:
            resolve_geometry(self._entity.geometry.geometry_refs, source=geometry)
        except GeometryResolutionError as error:
            self._issue(EvidenceIntegrityKind.MISSING_GEOMETRY, str(error))

    def _collect(self, ref: FusedEvidenceRef, source: FusedEvidenceSource) -> None:
        try:
            fused = source.fused_evidence(ref.fusion_support_id)
        except (KeyError, FusionRunArtifactError) as error:
            self._issue(
                EvidenceIntegrityKind.MISSING_REFERENCE,
                f"fusion run {ref.fusion_run_id!r} has no readable support "
                f"{ref.fusion_support_id!r}: {error}",
                blocks_comparison=True,
            )
            return
        if fused.fused_evidence_id != ref.fused_evidence_id:
            self._issue(
                EvidenceIntegrityKind.MISSING_REFERENCE,
                f"support {ref.fusion_support_id!r} holds fused evidence "
                f"{fused.fused_evidence_id!r}, not {ref.fused_evidence_id!r}",
                blocks_comparison=True,
            )
            return
        for contribution in fused.contributions:
            self._spatial.add(contribution.spatial_observation_id)
            self._physical.add(contribution.physical_observation_id)
            outside = [item for item in contribution.geometry_support if item not in self._support]
            if outside:
                self._issue(
                    EvidenceIntegrityKind.GEOMETRY_OUTSIDE_SUPPORT,
                    f"contribution {contribution.contribution_id!r} sees {len(outside)} "
                    f"geometry element(s) outside the entity's support",
                )
        self._features.update(feature_refs_of(fused))
        self._representations.update(
            (ref.run_id, ref.representation_id) for ref in fused.point_representation_refs
        )

    def _compare(
        self,
        kind: EvidenceIntegrityKind,
        what: str,
        listed: AbstractSet[object],
        carried: AbstractSet[object],
    ) -> None:
        if listed != carried:
            self._issue(
                kind,
                f"the entity lists {len(listed)} {what}, the fused evidence carries "
                f"{len(carried)}; {len(listed - carried)} listed are not carried and "
                f"{len(carried - listed)} carried are not listed",
            )

    def _issue(
        self, kind: EvidenceIntegrityKind, detail: str, *, blocks_comparison: bool = False
    ) -> None:
        # Sem os fatos da evidência fundida, comparar as listas só geraria falsos positivos.
        if blocks_comparison:
            self._complete = False
        self.issues.append(
            EvidenceIntegrityIssue(entity=self._entity.reference, kind=kind, detail=detail)
        )
