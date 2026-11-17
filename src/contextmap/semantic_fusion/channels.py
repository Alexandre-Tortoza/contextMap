"""Que identidades alimentaram cada canal de evidência declarado.

Um canal só participa quando a política o declara: dado disponível nunca o ativa. Este
módulo registra, para cada canal ativo, as identidades do que o alimentou -- o
interpretador ou scorer como ``backend_id/model/version``, o espaço de embedding, a versão
das definições de qualidade, o espaço de representação, o mapa e a política de suporte.

É o que permite atribuir um efeito a um canal numa ablação. Um canal pode estar ativo e
vazio: declarado, mas sem dado para aquele suporte. Nada aqui converte, renomeia ou
combina um canal com outro.
"""

from __future__ import annotations

from collections.abc import Sequence

from contextmap.semantic_fusion.evidence import (
    ChannelProvenance,
    EvidenceChannel,
    EvidenceContribution,
    PointRepresentationRef,
)
from contextmap.semantic_fusion.hypotheses import ViewClaim
from contextmap.semantic_fusion.supports import FusionSupport
from contextmap.shared import BackendProvenance


def _producer_label(producer: BackendProvenance) -> str:
    label = f"{producer.backend_id}/{producer.model}/{producer.version}"
    return (
        label
        if producer.configuration_fingerprint is None
        else (f"{label}@{producer.configuration_fingerprint}")
    )


def channel_provenance(
    active: frozenset[EvidenceChannel],
    support: FusionSupport,
    claims: Sequence[ViewClaim],
    contributions: Sequence[EvidenceContribution],
    structure: Sequence[PointRepresentationRef],
) -> tuple[ChannelProvenance, ...]:
    """Name what fed each active channel, so an effect can be attributed to a channel."""
    fed: dict[EvidenceChannel, set[str]] = {
        EvidenceChannel.SEMANTIC_CLAIMS: {
            _producer_label(item.claim.provenance.backend) for item in claims
        },
        EvidenceChannel.SEMANTIC_SCORES: {
            _producer_label(ref.scorer) for item in contributions for ref in item.score_refs
        }
        | {_producer_label(ref.scorer) for item in contributions for ref in item.raw_score_refs},
        EvidenceChannel.VISUAL_FEATURES: {
            ref.embedding_space_id for item in contributions for ref in item.visual_feature_refs
        },
        EvidenceChannel.OBSERVATION_QUALITY: {
            item.observation_quality.definitions_version
            for item in contributions
            if item.observation_quality is not None
        },
        EvidenceChannel.GEOMETRY_SUPPORT: {
            f"map:{support.geometric_map_id}",
            f"support-policy:{support.provenance.support_policy_id}",
        },
        EvidenceChannel.POINT_REPRESENTATION: {ref.representation_space_id for ref in structure},
    }
    return tuple(
        ChannelProvenance(channel=channel, identities=tuple(sorted(fed[channel])))
        for channel in sorted(active, key=lambda item: item.value)
    )
