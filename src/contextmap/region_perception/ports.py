"""Pontos de substituição dos backends de descoberta, grounding e refinamento de região.

Cada ``Protocol`` aqui corresponde a uma variação real já exercida por mais de um
backend concreto (SAM3, Florence-2, LocateAnything), que entram como adapters. Nenhum
port constrói backend nem importa SDK de modelo: isso pertence ao composition root.

Os ports não impõem ordem de execução entre capabilities — quem resolve o grafo de
estágios é o runtime. Um mesmo modelo pode satisfazer ports de capabilities diferentes
por adapters distintos, e isso nunca é motivo para fundir os ports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextmap.region_perception.discovery.region_discovery import (
    AuditedRegions,
    RegionDiscoveryIdentity,
)
from contextmap.region_perception.grounding.region_grounding import (
    RegionGroundingCapabilities,
    RegionGroundingExecution,
    RegionGroundingRequest,
)
from contextmap.region_perception.refinement.region_refinement import (
    RegionRefinementCapabilities,
    RegionRefinementExecution,
    RegionRefinementRequest,
)
from contextmap.region_perception.region_evidence import Region2D
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage


@runtime_checkable
class RegionDiscovery(Protocol):
    """Capability port: discover 2D region candidates from a prepared image."""

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every region this call produces.
        """
        ...

    def discover(self, image: PreparedImage) -> Sequence[Region2D]:
        """Discover the canonical regions of a prepared image.

        Args:
            image: The prepared image to search.

        Returns:
            The regions that survived discovery and normalization. No production backend
            returns a rejected candidate here (``Region2D.is_accepted=False``): rejections and
            merges are reported by :meth:`AuditedRegionDiscovery.discover_audited`.
        """
        ...


@runtime_checkable
class AuditedRegionDiscovery(RegionDiscovery, Protocol):
    """Capability port: region discovery that also reports how each frame's regions were decided.

    The canonical runtime path requires it, so that rejected candidates and merge decisions
    reach the run artifact instead of dying with the run, and so that the run's identity names
    the discovery configuration and algorithms that decided them. A consumer that only needs
    regions keeps depending on :class:`RegionDiscovery`, which is unchanged.
    """

    def discovery_identity(self) -> RegionDiscoveryIdentity:
        """Report the effective discovery identity every frame of this backend is decided under.

        Returns:
            The identity of the pass plan, normalization and discovery algorithms, defaults
            resolved: the same one each :meth:`discover_audited` audit records.
        """
        ...

    def discover_audited(self, image: PreparedImage) -> AuditedRegions:
        """Discover the canonical regions of a prepared image together with their audit.

        Args:
            image: The prepared image to search.

        Returns:
            The same regions :meth:`~RegionDiscovery.discover` returns, and the audit of the
            passes, pass-level rejections, normalization rejections and merge decisions that
            produced them.
        """
        ...


@runtime_checkable
class RegionGrounding(Protocol):
    """Capability port: localize what an explicit language query refers to in an image.

    Distinct from :class:`RegionDiscovery`, which proposes regions from the image alone:
    here the query is a first-class, validated, persisted inference input of each
    request, never backend configuration.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            Provenance whose capability is ``"region_grounding"``.
        """
        ...

    def capabilities(self) -> RegionGroundingCapabilities:
        """Declare the query policies (task and geometry) this backend serves."""
        ...

    def ground(self, request: RegionGroundingRequest) -> RegionGroundingExecution:
        """Answer one grounding request.

        Args:
            request: The image, the query and the configuration fingerprint.

        Returns:
            The request, rendered prompt, raw response, parsed outputs and diagnostics.

        Raises:
            GroundingRequestError: Before any inference, when the backend does not declare
                the requested policy, task or geometry.
        """
        ...


@runtime_checkable
class RegionRefinement(Protocol):
    """Capability port: refine grounding proposals into mask-backed regions.

    A proposal is only a segmentation prompt: the refiner's mask becomes a separately
    identified region whose contributor is the proposal, and the grounding evidence is
    never mutated.
    """

    def backend_provenance(self) -> BackendProvenance:
        """Report this refiner's identity and effective configuration.

        Returns:
            Provenance whose capability is ``"region_refinement"``.
        """
        ...

    def capabilities(self) -> RegionRefinementCapabilities:
        """Declare the prompt geometries this refiner accepts."""
        ...

    def refine(self, request: RegionRefinementRequest) -> RegionRefinementExecution:
        """Refine every prompt of one request.

        Args:
            request: The prepared image, the prompts and the configuration fingerprint.

        Returns:
            One refined region or explicit rejection per prompt, with diagnostics.

        Raises:
            RefinementRequestError: Before any inference, for an unaccepted prompt geometry.
        """
        ...


__all__ = [
    "AuditedRegionDiscovery",
    "RegionDiscovery",
    "RegionGrounding",
    "RegionRefinement",
]
