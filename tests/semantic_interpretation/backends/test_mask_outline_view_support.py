"""C6: the instruction-following interpreters consume the mask-outline views (arms B and D)."""

from __future__ import annotations

import pytest

from contextmap.semantic_interpretation import (
    SemanticInterpreterCapabilities,
    SemanticMaskOutline,
    SemanticViewPolicy,
    VisualViewKind,
    check_view_policy_supported,
)
from contextmap.semantic_interpretation.adapters.gemini import (
    GeminiSemanticConfig,
    GeminiSemanticInterpreter,
)
from contextmap.semantic_interpretation.adapters.qwen import (
    QwenSemanticConfig,
    QwenSemanticInterpreter,
)

_OUTLINE = SemanticMaskOutline(rgb=(255, 0, 255), width_px=2)


def _qwen() -> SemanticInterpreterCapabilities:
    return QwenSemanticInterpreter.capabilities_for(
        QwenSemanticConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            precision="float32",
            max_new_tokens=32,
            temperature=0.0,
        )
    )


def _gemini() -> SemanticInterpreterCapabilities:
    return GeminiSemanticInterpreter.capabilities_for(
        GeminiSemanticConfig(
            model="gemini-2.5-pro", timeout_s=30, max_retries=2, temperature=0, thinking_budget=128
        )
    )


@pytest.mark.parametrize("capabilities", [_qwen, _gemini], ids=["qwen", "gemini"])
@pytest.mark.parametrize(
    "views",
    [
        (VisualViewKind.MASK_OUTLINE_CROP,),
        (VisualViewKind.MASK_OUTLINE_FULL_FRAME,),
    ],
    ids=["arm-b", "arm-d"],
)
def test_each_outline_arm_passes_the_preflight_of_the_instruction_following_interpreters(
    capabilities: object, views: tuple[VisualViewKind, ...]
) -> None:
    declared = capabilities()  # type: ignore[operator]

    check_view_policy_supported(
        SemanticViewPolicy(region_views=views, mask_outline=_OUTLINE), declared
    )

    assert set(views) <= declared.supported_view_kinds
