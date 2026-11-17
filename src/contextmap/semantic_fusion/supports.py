"""O *onde* da fusão: o suporte espacial e as observações que ficaram fora dele.

Um :class:`FusionSupport` afirma apenas que suas observações espaciais veem geometria
compatível sob uma política. Não afirma label, classe nem identidade de objeto, e não
é estável entre versões do mapa: identidade persistente pertence a Semantic Mapping.

Uma :class:`ExcludedObservation` é a contrapartida explícita: toda observação de
entrada está em exatamente um suporte ou aqui, com o motivo. Ela é contratual e é
persistida pelo artifact, por isso é um contrato e não um detalhe da política.

Quem constrói esses contratos é ``support_construction``; este módulo só os define.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from contextmap.geometric_mapping import Bounds3D, GeometryReference, MapId
from contextmap.semantic_fusion.identity import FusionSupportId
from contextmap.semantic_fusion.invariants import (
    require_canonical,
    require_geometry_support,
    require_present,
)
from contextmap.sensor_association import SpatialObservationId
from contextmap.shared import Vector3
from contextmap.state_estimation import TimeBounds


@dataclass(frozen=True, kw_only=True)
class FusionSupportProvenance:
    """How a spatial support was built.

    Attributes:
        support_policy_id: Versioned rule that decided which observations share a support.
        configuration_fingerprint: Hash of the support configuration, when configurable.
        code_version: Code revision that produced the support, when known.
    """

    support_policy_id: str
    configuration_fingerprint: str | None = None
    code_version: str | None = None

    def __post_init__(self) -> None:
        """Require the policy identity.

        Raises:
            ValueError: If ``support_policy_id`` is empty.
        """
        require_present(self, "support_policy_id")


@dataclass(frozen=True, kw_only=True)
class FusionSupport:
    """A spatial support over which evidence can be accumulated.

    It states only that its spatial observations see compatible geometry under the
    support policy. It asserts no label, no class and no object identity, and it is
    not stable across map versions: entity identity belongs to later capabilities.

    Attributes:
        fusion_support_id: Identity of this support.
        geometric_map_id: The immutable map the geometry belongs to.
        geometry_support: References to the geometry, sorted by ``geometry_id`` and unique.
        spatial_observation_ids: The spatial observations accumulated here, sorted and unique.
        bounds: Extent of the geometry, expressed in the frame it declares.
        centroid_m: Centroid of the geometry, in meters, in the frame of ``bounds``.
        time_bounds: Interval spanned by the acquisition of the observations.
        provenance: How the support was built.
    """

    fusion_support_id: FusionSupportId
    geometric_map_id: MapId
    geometry_support: tuple[GeometryReference, ...]
    spatial_observation_ids: tuple[SpatialObservationId, ...]
    bounds: Bounds3D
    centroid_m: Vector3
    time_bounds: TimeBounds
    provenance: FusionSupportProvenance

    def __post_init__(self) -> None:
        """Validate identities, geometry, observations and the centroid.

        Raises:
            ValueError: If an identity or the support is empty, the geometry spans another
                map or is not sorted and unique, the observations are not sorted and unique,
                or the centroid is not finite or lies outside ``bounds``.
        """
        require_present(self, "fusion_support_id", "geometric_map_id")
        require_geometry_support("geometry_support", self.geometry_support)
        foreign = {reference.map_id for reference in self.geometry_support} - {
            self.geometric_map_id
        }
        if foreign:
            raise ValueError(
                f"geometry_support must reference only the support map "
                f"{self.geometric_map_id!r}, got maps {sorted(foreign)!r}"
            )
        if not self.spatial_observation_ids:
            raise ValueError("spatial_observation_ids must not be empty")
        require_canonical(
            "spatial_observation_ids", self.spatial_observation_ids, lambda item: (item,)
        )
        if not all(math.isfinite(value) for value in self.centroid_m):
            raise ValueError(f"centroid_m must be finite, got {self.centroid_m!r}")
        if not self.bounds.contains(self.centroid_m, frame_id=self.bounds.frame_id):
            raise ValueError(
                f"centroid_m {self.centroid_m!r} must lie inside the support bounds "
                f"{self.bounds.minimum_m!r}..{self.bounds.maximum_m!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ExcludedObservation:
    """A spatial observation that did not take part in any support, and why.

    Every input observation of a build is in exactly one support or here: exclusion is
    contractual and persisted, never a silent drop.

    Attributes:
        spatial_observation_id: The excluded observation.
        geometry_count: How many geometry elements it references.
        minimum_geometry_count: The policy minimum it fell short of.
    """

    spatial_observation_id: SpatialObservationId
    geometry_count: int
    minimum_geometry_count: int
