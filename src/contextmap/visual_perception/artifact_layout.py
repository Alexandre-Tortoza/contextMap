"""Convenções de layout do artifact de Visual Perception.

Este módulo possui somente caminhos compartilhados pelo writer do artifact. Ele separa dados
contratuais de diagnósticos humanos e não conhece filesystem, backends ou regras científicas.
"""

DEBUG_ROOT = "__DEBUG__"
"""Diretório raiz, relativo ao artifact, para diagnósticos não contratuais."""


def debug_directory(area: str) -> str:
    """Return the run-relative diagnostic directory for one capability area.

    Args:
        area: Stable area identifier, such as ``"30-feature-extraction"``.

    Raises:
        ValueError: If ``area`` is empty or attempts to escape the debug root.
    """
    if not area or "/" in area or "\\" in area or area in {".", ".."}:
        raise ValueError(f"debug area must be one safe path segment, got {area!r}")
    return f"{DEBUG_ROOT}/{area}"
