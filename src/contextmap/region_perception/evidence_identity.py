"""Deterministic identities of evidence owned by a regional producer."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception.region_evidence import Region2D, RegionId

if TYPE_CHECKING:
    from contextmap.region_perception.discovery.region_discovery import RegionDiscoveryAudit


def _scope(run_id: str, source_observation_id: SourceObservationId) -> str:
    """Require producer and observation identities before constructing their scope."""
    if not str(run_id).strip() or not str(source_observation_id).strip():
        raise ValueError("regional run_id and source_observation_id must not be empty")
    return f"{run_id}--{source_observation_id}"


def region_id_for(
    *, run_id: str, source_observation_id: SourceObservationId, index: int
) -> RegionId:
    """Identify one discovered region in its producing run and physical frame."""
    if index < 0:
        raise ValueError("region index must be non-negative")
    return RegionId(f"{_scope(run_id, source_observation_id)}--region-{index:04d}")


def grounding_region_id_for(
    *, run_id: str, source_observation_id: SourceObservationId, request_id: str, index: int
) -> RegionId:
    """Identify a grounded region without depending on features or interpretation."""
    if not request_id or index < 0:
        raise ValueError("grounding request_id must not be empty and index must be non-negative")
    return RegionId(f"{_scope(run_id, source_observation_id)}--{request_id}-region-{index:04d}")


def grounding_point_id_for(
    *, run_id: str, source_observation_id: SourceObservationId, request_id: str, index: int
) -> str:
    """Identify a grounded point used as a refinement proposal."""
    if not request_id or index < 0:
        raise ValueError("grounding request_id must not be empty and index must be non-negative")
    return f"{_scope(run_id, source_observation_id)}--{request_id}-point-{index:04d}"


def refined_region_id_for(
    *,
    run_id: str,
    source_observation_id: SourceObservationId,
    proposal_id: str,
    configuration_fingerprint: str,
) -> RegionId:
    """Identify a refined region by its producer, proposal and refiner configuration."""
    if not proposal_id or not configuration_fingerprint:
        raise ValueError("proposal_id and configuration_fingerprint must not be empty")
    digest = hashlib.sha256(f"{proposal_id}\n{configuration_fingerprint}".encode()).hexdigest()
    return RegionId(f"{_scope(run_id, source_observation_id)}--refinement-{digest}")


@dataclass(frozen=True, kw_only=True)
class NamespacedRegions:
    """Publication-ready region identities and the corresponding discovery lineage."""

    regions: tuple[Region2D, ...]
    audit: RegionDiscoveryAudit | None
    region_id_map: Mapping[RegionId, RegionId]


def namespace_regions(
    *,
    run_id: str,
    source_observation_id: SourceObservationId,
    regions: Sequence[Region2D],
    audit: RegionDiscoveryAudit | None = None,
) -> NamespacedRegions:
    """Qualify local normalized identities at publication without changing geometry.

    Masks remain inline until the writer stages them under the qualified identity.
    An already externalized mask cannot be renamed without its payload store, so it
    is rejected at this producer boundary. Candidate IDs remain pass-local audit
    keys; only links to canonical regions are remapped.
    """
    _scope(run_id, source_observation_id)
    if audit is not None and audit.source_observation_id != source_observation_id:
        raise ValueError("discovery audit observation does not match the published frame")
    mapping: dict[RegionId, RegionId] = {}
    renamed: list[Region2D] = []
    for index, region in enumerate(regions):
        if region.source_observation_id not in (None, source_observation_id):
            raise ValueError("region source observation does not match the published frame")
        if region.region_id in mapping:
            raise ValueError(f"duplicate local region identity: {region.region_id!r}")
        if region.mask_reference is not None and region.mask is None:
            raise ValueError("namespace regions before externalizing their masks")
        identity = region_id_for(
            run_id=run_id, source_observation_id=source_observation_id, index=index
        )
        mapping[region.region_id] = identity
        renamed.append(
            replace(region, region_id=identity, source_observation_id=source_observation_id)
        )
    if audit is not None:
        hints = tuple(
            replace(
                hint,
                region_id=None if hint.region_id is None else mapping[hint.region_id],
                perception_run_id=run_id,
                perception_result_id=_scope(run_id, source_observation_id),
            )
            for hint in audit.semantic_hints
        )
        audit = replace(audit, semantic_hints=hints)
    return NamespacedRegions(
        regions=tuple(renamed), audit=audit, region_id_map=MappingProxyType(mapping)
    )
