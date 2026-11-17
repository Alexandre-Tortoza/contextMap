"""A porta de leitura de um run de Semantic Fusion persistido.

Uma entidade guarda só referências para a evidência que a criou, então validar ou percorrer essas
referências exige abrir o run de fusão de origem. Este protocolo é o contrato mínimo disso —
manifest, evidência por suporte e verificação de integridade — e é satisfeito por
`SemanticFusionRunReader`.

Ele existe como porta por dois motivos concretos: a validação e a travessia precisam da mesma
superfície, e um teste precisa de um substituto leve para um run real em disco. Declará-lo aqui, e
não dentro de um dos dois, evita que um deles importe o outro só para alcançar o tipo.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextmap.semantic_fusion import (
    FusedEvidence,
    FusionSupportId,
    SemanticFusionRunManifest,
)


@runtime_checkable
class FusedEvidenceSource(Protocol):
    """Read boundary of one persisted Semantic Fusion run.

    A ``SemanticFusionRunReader`` satisfies it. Nothing else is needed from the run.
    """

    @property
    def manifest(self) -> SemanticFusionRunManifest:
        """The manifest of the run."""
        ...

    def fused_evidence(self, support_id: FusionSupportId) -> FusedEvidence:
        """Read the evidence fused over one support.

        Args:
            support_id: A support of the run.

        Returns:
            The fused evidence.

        Raises:
            KeyError: If the run has no such support.
        """
        ...

    def verify_integrity(self) -> list[str]:
        """Check the run's files against its inventory; empty means intact."""
        ...
