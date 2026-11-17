"""Devolução dos recursos do modelo entre estágios de uma execução stage-major (#689).

Um backend repassa ``release()`` ao seu runtime; o runtime solta o modelo e devolve os
blocos CUDA cacheados, de modo que o modelo do estágio seguinte caiba numa GPU pequena.

É infraestrutura de adapter, não domínio: nada aqui conhece região, feature ou claim.
"""

from __future__ import annotations

import gc
import importlib


def release_runtime(runtime: object) -> None:
    """Forward ``release()`` to a runtime that offers it; leave any other runtime alone."""
    release = getattr(runtime, "release", None)
    if callable(release):
        release()


def free_cuda_memory() -> None:
    """Collect garbage and return cached CUDA blocks to the driver, when torch is present."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
