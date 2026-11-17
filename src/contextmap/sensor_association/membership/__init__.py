"""Pertencimento: que geometria visível cada região congelada de um frame suporta.

Duas regras convivem, e qual delas foi aplicada é registrado por região, nunca inferido:

- **máscara** (``mask-membership-v1``): um ponto visível pertence a toda região aceita cuja
  máscara é foreground no pixel mais próximo do seu pixel preparado;
- **caixa** (``box-depth-component-v1``, opcional): para regiões sem máscara, apenas o
  componente 3D que domina o núcleo da caixa vira suporte.

Regiões sobrepostas são válidas e nada aqui escolhe um vencedor semântico. O subpacote termina
montando os ``SpatialObservation``, que é onde a evidência interna vira contrato público.
"""

from contextmap.sensor_association.membership.box_depth_component import (
    BoxDepthComponentPolicy,
)
from contextmap.sensor_association.membership.models import (
    COMPOSITE_MEMBERSHIP_POLICY_ID,
    COVERAGE_DEFINITIONS_VERSION,
    MEMBERSHIP_POLICY_ID,
    FrameMembership,
    MembershipStatistics,
    RegionMaskLoader,
    RegionMembership,
    SkippedRegion,
    SkipReason,
    configured_membership_policy_id,
)
from contextmap.sensor_association.membership.region_association import associate_regions
from contextmap.sensor_association.membership.spatial_observations import (
    build_spatial_observations,
)

__all__ = [
    "COMPOSITE_MEMBERSHIP_POLICY_ID",
    "COVERAGE_DEFINITIONS_VERSION",
    "MEMBERSHIP_POLICY_ID",
    "BoxDepthComponentPolicy",
    "FrameMembership",
    "MembershipStatistics",
    "RegionMaskLoader",
    "RegionMembership",
    "SkipReason",
    "SkippedRegion",
    "associate_regions",
    "build_spatial_observations",
    "configured_membership_policy_id",
]
