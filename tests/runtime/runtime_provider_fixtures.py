"""Importable ``RuntimeProvider`` targets for ``resources.providers`` tests.

These exist only so a test can declare a real ``"module:attribute"`` string in
configuration and let :func:`contextmap.runtime.composition.resolve_provider` import it
for real, exactly as the installed ``contextmap`` binary would for a deployment's own
loader module -- proving the declarative mechanism end to end, not just its plumbing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

CALLS: list[tuple[str, Any]] = []
"""Every call this module's providers received, for a test to assert against."""


class MaterializedPixelsSam2Runtime:
    """A SAM2 runtime that proposes only from the pixels of the pass it is handed.

    Like a real SDK, it cannot open a ``PreparedImage`` reference: it needs the exact
    ``(height, width, 3)`` ``uint8`` pixels of the pass. It proposes one mask covering the pass
    and keeps every image it received, for a test to compare with the prepared image.
    """

    def __init__(self) -> None:
        self.pixels: list[Any] = []

    def predict(self, discovery_input: Any, image: Any, config: Any) -> tuple[Any, ...]:
        """Propose one full-pass mask from the pass's materialized pixels.

        Raises:
            TypeError: If ``image`` is not the pass's ``(height, width, 3)`` ``uint8`` pixels.
        """
        from contextmap.region_perception import InlineMask
        from contextmap.region_perception.adapters.sam2 import Sam2NativeProposal

        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        pixels = np.asarray(image)
        if pixels.shape != (height, width, 3) or pixels.dtype != np.uint8:
            raise TypeError(
                f"SAM2 needs the {width}x{height} pixels of the pass, not {type(image).__name__}"
            )
        self.pixels.append(pixels.copy())
        return (
            Sam2NativeProposal(
                proposal_id="sam2-000000",
                box=(0.0, 0.0, float(width), float(height)),
                mask=InlineMask(np.ones((height, width), dtype=bool)),
                predicted_iou=0.9,
                stability_score=0.9,
            ),
        )


class VerifiedBytesQwenRuntime:
    """A Qwen runtime that answers only from the verified bytes of each view.

    Like a real model, it cannot open a ``SemanticVisualView`` reference: every view must arrive
    with the exact bytes its ``sha256`` identifies. A scene request abstains and a region request
    answers one primary claim; every payload received is kept, in order.
    """

    def __init__(self) -> None:
        self.payloads: list[bytes] = []

    def generate(self, *, visual_views: Any, prompt: str, config: Any) -> Any:
        """Answer one request from its views' verified bytes.

        Raises:
            TypeError: If a view arrives without its bytes.
            ValueError: If a view's bytes are not the ones its ``sha256`` identifies.
        """
        from contextmap.semantic_interpretation import MaterializedSemanticView
        from contextmap.semantic_interpretation.adapters.qwen import (
            QwenGenerationResponse,
        )

        for item in visual_views:
            if not isinstance(item, MaterializedSemanticView):
                raise TypeError(
                    f"Qwen needs the verified bytes of each view, not a {type(item).__name__} "
                    f"reference {getattr(item, 'payload_reference', '')!r}"
                )
            if hashlib.sha256(item.payload).hexdigest() != item.view.sha256:
                raise ValueError(f"view {item.view.view_id!r} bytes do not match its sha256")
            self.payloads.append(item.payload)
        if visual_views[0].view.region_id is None:
            return QwenGenerationResponse(
                text=json.dumps({"abstained": True, "claims": [], "scene_context": None})
            )
        claim: dict[str, Any] = {
            "hypothesis": "chair",
            "role": "primary",
            "category": None,
            "region_kind": "thing",
            "attributes": {},
            "confidence": None,
        }
        return QwenGenerationResponse(
            text=json.dumps({"abstained": False, "claims": [claim], "scene_context": None})
        )


SAM2_RUNTIMES: list[MaterializedPixelsSam2Runtime] = []
"""Every runtime :func:`load_materialized_sam2` returned, in order."""

QWEN_RUNTIMES: list[VerifiedBytesQwenRuntime] = []
"""Every runtime :func:`load_verified_qwen` returned, in order."""


def load_materialized_sam2(config: Any, secrets: Any) -> MaterializedPixelsSam2Runtime:
    """Return a SAM2 runtime that accepts only materialized pass pixels."""
    runtime = MaterializedPixelsSam2Runtime()
    SAM2_RUNTIMES.append(runtime)
    return runtime


def load_verified_qwen(config: Any, secrets: Any) -> VerifiedBytesQwenRuntime:
    """Return a Qwen runtime that accepts only verified view bytes."""
    runtime = VerifiedBytesQwenRuntime()
    QWEN_RUNTIMES.append(runtime)
    return runtime


def load_region_discovery(config: Any, secrets: Any) -> object:
    """Return a placeholder region-discovery runtime, recording that it was asked for."""
    CALLS.append(("region_discovery", config))
    return object()


def load_semantic_interpretation(config: Any, secrets: Any) -> object:
    """Return a placeholder semantic-interpretation runtime, recording the call."""
    CALLS.append(("semantic_interpretation", config))
    return object()


def load_dense_features(config: Any, secrets: Any) -> object:
    """Return a placeholder dense-feature runtime, recording the call."""
    CALLS.append(("dense_features", config))
    return object()


def load_region_features(config: Any, secrets: Any) -> object:
    """Return a placeholder region-feature runtime, recording the call."""
    CALLS.append(("region_features", config))
    return object()


not_callable = 42
"""A non-callable attribute, for testing that :func:`resolve_provider` rejects it."""
