"""Superfície de compatibilidade dos adapters que passaram a viver com suas capabilities.

Cada módulo de adapter foi para a capability cujo port ele satisfaz. Este módulo resolve
os nomes antigos para os módulos novos, de forma que consumidores existentes continuem
funcionando; o objeto devolvido é o mesmo módulo, nunca uma cópia.

Código novo deve importar cada adapter de onde ele vive hoje:

```text
florence2, locateanything, prepared_images, sam3  -> region_perception.adapters
alphaclip, clip, dinov2, dinov3, siglip2          -> ..visual_features.adapters
eagle2_5, florence2_semantic, gemini, qwen        -> ..semantic_interpretation.adapters
alphaclip_text, clip_text                         -> semantic_scoring.adapters
```

(``..`` acima é ``contextmap.visual_perception``, o pacote que contém este.)

As torres de texto são o caso que mais se moveu: elas codificam a hipótese de um claim para
compará-la com a feature visual, o que é scoring, e importavam a infra de template de
``semantic_scoring`` — por isso passaram a viver lá, desfazendo o ciclo que existia enquanto
estavam entre as features.
"""

from __future__ import annotations

_REGIONS = "contextmap.region_perception.adapters"
_FEATURES = "contextmap.visual_perception.visual_features.adapters"
_SEMANTICS = "contextmap.semantic_interpretation.adapters"
_SCORING = "contextmap.semantic_scoring.adapters"

_ADAPTER_MODULES = {
    # region_perception
    "florence2": f"{_REGIONS}.florence2",
    "locateanything": f"{_REGIONS}.locateanything",
    "prepared_images": f"{_REGIONS}.prepared_images",
    "sam3": f"{_REGIONS}.sam3",
    # visual_perception.visual_features
    "alphaclip": f"{_FEATURES}.alphaclip",
    "clip": f"{_FEATURES}.clip",
    "dinov2": f"{_FEATURES}.dinov2",
    "dinov3": f"{_FEATURES}.dinov3",
    "siglip2": f"{_FEATURES}.siglip2",
    # visual_perception.semantic_interpretation
    "eagle2_5": f"{_SEMANTICS}.eagle2_5",
    "florence2_semantic": f"{_SEMANTICS}.florence2_semantic",
    "gemini": f"{_SEMANTICS}.gemini",
    "qwen": f"{_SEMANTICS}.qwen",
    # semantic_scoring: as torres de texto seguiram o scorer que as consome
    "alphaclip_text": f"{_SCORING}.alphaclip_text",
    "clip_text": f"{_SCORING}.clip_text",
}


def __getattr__(name: str) -> object:
    """Resolve a legacy adapter name to the module of the capability that owns it.

    Args:
        name: Adapter name as it was spelled under this package.

    Returns:
        The adapter module itself, imported from its current capability.

    Raises:
        AttributeError: If no adapter was ever reachable under this name.
    """
    target = _ADAPTER_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return importlib.import_module(target)


def __dir__() -> list[str]:
    """List the adapters still reachable through this legacy package."""
    return sorted(_ADAPTER_MODULES)


__all__ = sorted(_ADAPTER_MODULES)
