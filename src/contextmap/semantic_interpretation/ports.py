"""Ponto de substituição dos backends de interpretação semântica.

``SemanticInterpreter`` cobre uma variação já exercida por quatro adapters (Qwen,
Gemini, Eagle-2.5, Florence-2), que diferem nos modos e nas evidências que aceitam
— por isso declaram capabilities antes de qualquer execução.

O port não constrói backend nem importa SDK de modelo: isso pertence ao composition
root.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextmap.semantic_interpretation.interpretation_execution import (
    SemanticInterpretationExecution,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationRequest,
    SemanticInterpreterCapabilities,
)
from contextmap.shared import BackendProvenance


@runtime_checkable
class SemanticInterpreter(Protocol):
    """Capability port: interpret a scene and/or specific regions semantically."""

    def backend_provenance(self) -> BackendProvenance:
        """Report this backend's identity and effective configuration.

        Returns:
            The provenance to attach to every claim/context this call produces.
        """
        ...

    def capabilities(self) -> SemanticInterpreterCapabilities:
        """Declare supported modes and canonical evidence before execution."""
        ...

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        """Interpret one validated canonical semantic request.

        Args:
            request: Exact scene/region evidence selection, prompt identity,
                output schema, and configuration fingerprint.

        Returns:
            Raw, parsed, provenance, and diagnostic records for the call.
        """
        ...


__all__ = ["SemanticInterpreter"]
