"""Travessia das cadeias de proveniência de uma entidade, só por identidades.

Duas cadeias precisam ser percorríveis para que uma entidade seja auditável:

```text
Entity -> FusedEvidence -> SpatialObservation -> Region2D / SemanticClaim
       -> SourceObservation

Entity -> EntityGeometry -> GeometryReference -> GeometricMapArtifact -> observação LiDAR
```

A travessia devolve **apenas identidades** e intervalos: nada de imagem, máscara, embedding ou
payload de fusão, e nenhum runtime de percepção, fusão ou modelo é carregado. Diferente da
validação, ela **levanta** `EvidenceTraceError` quando um run não pode ser lido, porque auditar uma
entidade sem conseguir ler a sua evidência não tem resultado parcial útil.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from contextmap.geometric_mapping import GeometrySource
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_fusion import (
    EvidenceContributionId,
    FusedEvidenceId,
    FusionRunArtifactError,
    ScoreReference,
    SemanticFusionRunId,
)
from contextmap.semantic_interpretation import ClaimId
from contextmap.semantic_mapping.entity import Entity, EntityReference
from contextmap.semantic_mapping.fused_evidence_source import FusedEvidenceSource
from contextmap.semantic_mapping.geometry import EntityGeometry, resolve_geometry
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import SourceTimestamp


class EvidenceTraceError(Exception):
    """Raised when the provenance of an entity cannot be traversed."""


@dataclass(frozen=True, kw_only=True)
class ContributionTrace:
    """One view that contributed to an entity, as identities along the provenance chain.

    Attributes:
        fused_evidence_id: The fused evidence the view belongs to.
        contribution_id: The view.
        spatial_observation_id: The spatial observation it comes from.
        region_artifact_id: The regional artifact that holds the region.
        semantic_interpretation_artifact_id: The interpretation artifact holding the claims,
            when semantic evidence exists.
        region_id: The observed region.
        claim_ids: The claims the view made about the region, sorted.
        score_refs: The scorer outputs for those claims, sorted: each names the scored claim and
            the scorer, so the score of a claim is reachable without loading the fusion payload.
        physical_observation_id: The physical frame: the end of the chain.
    """

    fused_evidence_id: FusedEvidenceId
    contribution_id: EvidenceContributionId
    spatial_observation_id: SpatialObservationId
    region_artifact_id: str
    semantic_interpretation_artifact_id: str | None
    region_id: RegionId
    claim_ids: tuple[ClaimId, ...]
    score_refs: tuple[ScoreReference, ...]
    physical_observation_id: SourceObservationId


@dataclass(frozen=True, kw_only=True)
class EntityEvidenceTrace:
    """The provenance chain of an entity, from its fused evidence to the physical frames.

    Attributes:
        entity: The entity that was traced.
        contributions: Every contributing view, sorted by fused evidence and contribution.
    """

    entity: EntityReference
    contributions: tuple[ContributionTrace, ...]

    @property
    def physical_observation_ids(self) -> tuple[SourceObservationId, ...]:
        """The distinct physical frames the entity rests on, sorted."""
        return tuple(sorted({item.physical_observation_id for item in self.contributions}))


def trace_entity_evidence(
    entity: Entity, *, fusion_runs: Mapping[SemanticFusionRunId, FusedEvidenceSource]
) -> EntityEvidenceTrace:
    """Walk an entity back to the views and physical frames that support it.

    The traversal reads only the fused evidence the entity references and returns identities: no
    image, mask, embedding or fusion payload is copied and no inference is run.

    Args:
        entity: The entity to trace.
        fusion_runs: The persisted fusion runs the entity references, by run identity.

    Returns:
        The contributing views, each with its spatial observation, regional artifact, region,
        claims, scorer references and physical frame.

    Raises:
        EvidenceTraceError: If a referenced run was not offered or its evidence cannot be read.
    """
    contributions: list[ContributionTrace] = []
    for ref in entity.evidence.fused_evidence:
        source = fusion_runs.get(ref.fusion_run_id)
        if source is None:
            raise EvidenceTraceError(f"fusion run {ref.fusion_run_id!r} was not offered")
        try:
            fused = source.fused_evidence(ref.fusion_support_id)
        except (KeyError, FusionRunArtifactError) as error:
            raise EvidenceTraceError(
                f"cannot read fused evidence of support {ref.fusion_support_id!r}: {error}"
            ) from error
        contributions.extend(
            ContributionTrace(
                fused_evidence_id=fused.fused_evidence_id,
                contribution_id=item.contribution_id,
                spatial_observation_id=item.spatial_observation_id,
                region_artifact_id=item.region_artifact_id,
                semantic_interpretation_artifact_id=item.semantic_interpretation_artifact_id,
                region_id=item.region_id,
                claim_ids=tuple(claim.claim_id for claim in item.claim_refs),
                score_refs=item.score_refs,
                physical_observation_id=item.physical_observation_id,
            )
            for item in fused.contributions
        )
    return EntityEvidenceTrace(
        entity=entity.reference,
        contributions=tuple(
            sorted(contributions, key=lambda item: (item.fused_evidence_id, item.contribution_id))
        ),
    )


@dataclass(frozen=True, kw_only=True)
class GeometrySourceTrace:
    """The geometry of an entity that came from one physical observation.

    Attributes:
        source_observation_id: The physical (LiDAR) observation the points were measured in.
        point_count: How many of the entity's geometry elements come from it.
        first_acquired: Acquisition time of the earliest of those points.
        last_acquired: Acquisition time of the latest.
    """

    source_observation_id: SourceObservationId
    point_count: int
    first_acquired: SourceTimestamp
    last_acquired: SourceTimestamp


def trace_geometry_sources(
    geometry: EntityGeometry, *, source: GeometrySource
) -> tuple[GeometrySourceTrace, ...]:
    """Walk an entity's geometry back to the physical observations it was measured in.

    Args:
        geometry: The geometry of an entity.
        source: The read boundary of the geometric map its references belong to.

    Returns:
        One trace per source observation, sorted by observation.

    Raises:
        GeometryResolutionError: If a geometry reference cannot be resolved.
    """
    counts: dict[SourceObservationId, int] = defaultdict(int)
    stamps: dict[SourceObservationId, list[SourceTimestamp]] = defaultdict(list)
    for point in resolve_geometry(geometry.geometry_refs, source=source):
        counts[point.source_observation_id] += 1
        stamps[point.source_observation_id].append(point.acquisition_timestamp)
    return tuple(
        GeometrySourceTrace(
            source_observation_id=observation,
            point_count=counts[observation],
            first_acquired=min(stamps[observation], key=SourceTimestamp.total_nanoseconds),
            last_acquired=max(stamps[observation], key=SourceTimestamp.total_nanoseconds),
        )
        for observation in sorted(counts)
    )
