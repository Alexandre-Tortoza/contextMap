"""Identidades locais de uma execução de Semantic Fusion.

Toda identidade desta capability é local ao run e derivada deterministicamente da
evidência a montante: dois runs com a mesma entrada e a mesma configuração
reconstroem as mesmas identidades, sem registro nem alocador.

Elas vivem num módulo próprio porque todos os níveis de contrato as citam --
suporte, contribuição, hipótese e artifact -- e nenhum deles as possui sozinho.
Este módulo não tem regra científica, validação de invariante nem I/O.
"""

from __future__ import annotations

from typing import NewType

from contextmap.sensor_association import SpatialObservationId

FusionSupportId = NewType("FusionSupportId", str)
"""Identity of one spatial support, local to a Semantic Fusion run."""

EvidenceContributionId = NewType("EvidenceContributionId", str)
"""Identity of one view's contribution, local to a Semantic Fusion run."""

FusedEvidenceId = NewType("FusedEvidenceId", str)
"""Identity of the evidence fused over one support, local to a Semantic Fusion run."""

FusedHypothesisId = NewType("FusedHypothesisId", str)
"""Identity of one hypothesis, local to its :class:`FusedEvidence`."""


def evidence_contribution_id_for(
    *, fusion_support_id: FusionSupportId, spatial_observation_id: SpatialObservationId
) -> EvidenceContributionId:
    """Compute the deterministic identity of one observation's contribution to one support.

    Args:
        fusion_support_id: The support the evidence is accumulated over.
        spatial_observation_id: The spatial observation the view comes from.

    Returns:
        A pure function of the inputs, so a contribution can be referenced without a registry.
    """
    return EvidenceContributionId(f"contribution--{fusion_support_id}--{spatial_observation_id}")


def fused_evidence_id_for(*, fusion_support_id: FusionSupportId) -> FusedEvidenceId:
    """Compute the deterministic identity of the evidence fused over one support."""
    return FusedEvidenceId(f"fused--{fusion_support_id}")
