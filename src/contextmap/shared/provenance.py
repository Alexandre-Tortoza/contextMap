"""Identidade de uma invocação de backend, comum a todas as capabilities.

Toda evidência produzida por inferência carrega o backend que a produziu, e o
conceito é exatamente o mesmo em region_perception, visual_features,
semantic_interpretation, semantic_scoring, semantic_fusion, semantic_mapping,
sensor_association e evaluation: nenhuma delas introduz ou redefine a semântica
dos campos, apenas anexa o registro à sua própria evidência.

Por não ter dono entre as capabilities e ter significado idêntico em todas, mora
aqui em vez de em uma delas — caso contrário, oito capabilities dependeriam da
nona só para nomear o backend que executou.

Este módulo não conhece capability alguma: ele não sabe quais backends existem,
não valida ``capability`` contra um catálogo e não impõe formato ao
``configuration_fingerprint``. Quem decide isso é o composition root.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class BackendProvenance:
    """Traceability for one backend invocation producing a piece of evidence.

    Attributes:
        backend_id: Stable identity of the adapter, e.g. ``"sam2"``,
            ``"dinov3"``, ``"fake_region_discovery"``.
        capability: Capability port this backend satisfies, e.g.
            ``"region_discovery"``, ``"feature_extractor"``.
        provider: Model family/provider name.
        model: Model/checkpoint identity.
        version: Backend/model version string.
        configuration_fingerprint: Deterministic hash of the effective
            backend configuration, for reproducibility. ``None`` when the
            backend has no configurable parameters worth fingerprinting.
    """

    backend_id: str
    capability: str
    provider: str
    model: str
    version: str
    configuration_fingerprint: str | None = None


__all__ = ["BackendProvenance"]
