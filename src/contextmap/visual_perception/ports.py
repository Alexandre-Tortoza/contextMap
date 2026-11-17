"""Ports for visual feature production and releasable visual backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextmap.visual_perception.visual_features.ports import (
    FeatureExtractor,
    FeatureResolutionEnhancement,
)


@runtime_checkable
class ReleasableBackend(Protocol):
    """A backend that can free the model resources (GPU memory) it holds.

    A stage-major run releases each backend once its last stage has run for every frame, so
    the next stage's model fits on a small GPU (#689). After ``release()`` a lazily loading
    backend loads again on its next call; one whose model was handed to it already loaded
    refuses further calls instead of failing obscurely.
    """

    def release(self) -> None:
        """Free the model resources this backend holds."""
        ...


__all__ = [
    "FeatureExtractor",
    "FeatureResolutionEnhancement",
    "ReleasableBackend",
]
