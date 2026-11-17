"""Política resolvida que monta os requests semânticos de um run (#544).

A run's semantic requests are shaped by three explicitly selected choices: the prompt policy
of each mode, the ordered evidence views, and whether region requests are conditioned on the
frame's scene context. :class:`SemanticRequestPolicy` is that selection, resolved, with one
identity (:meth:`SemanticRequestPolicy.fingerprint`), so an experiment arm can name exactly the
treatment its requests received and two equivalent selections are recognisably the same. The
backend, its model and its visual-input budget stay backend configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from contextmap.region_perception import RegionMaskResolver, region_mask_pixels
from contextmap.semantic_interpretation.evidence_views import (
    SemanticViewPolicy,
    check_view_policy_supported,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
)
from contextmap.semantic_interpretation.prompt_policy import SemanticPromptPolicy

if TYPE_CHECKING:
    from contextmap.region_perception import Region2D
from contextmap.region_perception import RegionId

SEMANTIC_REQUEST_POLICY_VERSION = "semantic-request-policy/1"
"""Version of the request-policy document :meth:`SemanticRequestPolicy.to_document` writes."""


@dataclass(frozen=True, kw_only=True)
class RegionRequestBudget:
    """How many regions of one frame receive a region request (``largest-area-regions-v1``).

    The ``max_regions`` largest regions are requested; ties are broken by ``region_id`` so the
    selection never depends on the order of the region set. A region left out is not dropped:
    it stays in the result, reaches association and the map, and is recorded as ineligible
    for interpretation. The area is the region's ``area_pixels``, else its mask's foreground,
    else its box.

    Attributes:
        max_regions: Most region requests of one frame, at least one.
    """

    policy_id: ClassVar[str] = "largest-area-regions-v1"

    max_regions: int

    def __post_init__(self) -> None:
        """Reject a budget that would request nothing."""
        if self.max_regions < 1:
            raise ValueError(f"max_regions must be at least 1, got {self.max_regions}")

    def select(
        self, regions: Sequence[Region2D], *, mask_resolver: RegionMaskResolver | None = None
    ) -> frozenset[RegionId]:
        """Return the identities of the regions this budget requests in one frame.

        Args:
            regions: The frame's region set.
            mask_resolver: Loads the foreground of a region that only carries a
                ``mask_reference`` and no recorded ``area_pixels``; the ranking is the same as
                with the inline mask. A region with a recorded area never loads its pixels.

        Raises:
            MaskStoreError: If a region needs its pixels, only has a reference and no resolver
                was given.
        """
        ranked = sorted(
            regions, key=lambda region: (-_region_area(region, mask_resolver), region.region_id)
        )
        return frozenset(region.region_id for region in ranked[: self.max_regions])

    def to_document(self) -> dict[str, Any]:
        """Return the budget as JSON primitives, with its versioned rule."""
        return {"policy_id": self.policy_id, "max_regions": self.max_regions}


def _region_area(region: Region2D, mask_resolver: RegionMaskResolver | None) -> float:
    """The area the budget ranks by: recorded area, else mask foreground, else box."""
    if region.area_pixels is not None:
        return float(region.area_pixels)
    mask = region_mask_pixels(region, mask_resolver)
    if mask is not None:
        return float(mask.area)
    return float(region.bounding_box.width * region.bounding_box.height)


@dataclass(frozen=True, kw_only=True)
class SemanticModePrompt:
    """The prompt policy identity every request of one mode names.

    Attributes:
        template_id: Becomes the request's ``prompt_template_id``: a free-form catalog
            template, or a task-native prompt policy (Florence-2).
        output_schema: Becomes the request's ``requested_output_schema``.
    """

    template_id: str
    output_schema: str

    def __post_init__(self) -> None:
        """Reject an unnamed prompt policy or output schema."""
        if not self.template_id.strip():
            raise ValueError("mode prompt template_id must not be empty")
        if not self.output_schema.strip():
            raise ValueError("mode prompt output_schema must not be empty")


@dataclass(frozen=True, kw_only=True)
class SemanticRequestPolicy:
    """Prompt, ordered views, scene-context switch and output schema of a run's requests.

    A mode without a prompt is not interpreted: its requests are refused, never defaulted.

    Attributes:
        views: Ordered evidence views and how they are built.
        scene: Prompt of scene requests, or ``None`` when scenes are not interpreted.
        region: Prompt of region requests, or ``None`` when regions are not interpreted.
        region_scene_context: Whether every region request carries the scene context its
            frame's scene request produced (#529).
        region_budget: Limits the region requests of each frame to its largest regions;
            ``None`` requests every region.
    """

    views: SemanticViewPolicy
    scene: SemanticModePrompt | None = None
    region: SemanticModePrompt | None = None
    region_scene_context: bool = False
    region_budget: RegionRequestBudget | None = None

    def __post_init__(self) -> None:
        """Require one interpreted mode, and a region prompt for conditioning or a budget."""
        if self.scene is None and self.region is None:
            raise ValueError("a semantic request policy interprets at least one mode")
        if self.region_scene_context and self.region is None:
            raise ValueError("scene-context conditioning of region requests needs a region prompt")
        if self.region_budget is not None and self.region is None:
            raise ValueError("a region budget needs region requests to limit")

    @classmethod
    def from_prompt_policy(
        cls, prompt_policy: SemanticPromptPolicy, *, views: SemanticViewPolicy
    ) -> SemanticRequestPolicy:
        """Resolve an instruction-following interpreter's configured prompt and view policies."""
        scene = prompt_policy.template_for(SemanticInterpretationMode.SCENE)
        region = prompt_policy.template_for(SemanticInterpretationMode.REGION)
        assert region is not None  # a política de prompt sempre nomeia o template de região
        return cls(
            views=views,
            # Sem template de cena, a run só rotula regiões: o modo cena não é interpretado.
            scene=None
            if scene is None
            else SemanticModePrompt(
                template_id=scene.template_id, output_schema=scene.output_schema_version
            ),
            region=SemanticModePrompt(
                template_id=region.template_id, output_schema=region.output_schema_version
            ),
            region_scene_context=prompt_policy.region_scene_context,
        )

    def prompt_for(self, mode: SemanticInterpretationMode) -> SemanticModePrompt | None:
        """Return the prompt of one mode, or ``None`` when that mode is not interpreted."""
        return self.scene if mode is SemanticInterpretationMode.SCENE else self.region

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record of every choice, the view rules included."""
        return {
            "version": SEMANTIC_REQUEST_POLICY_VERSION,
            "prompts": {
                mode.value: (
                    None
                    if (prompt := self.prompt_for(mode)) is None
                    else {"template_id": prompt.template_id, "output_schema": prompt.output_schema}
                )
                for mode in SemanticInterpretationMode
            },
            "region_scene_context": self.region_scene_context,
            "region_budget": None
            if self.region_budget is None
            else self.region_budget.to_document(),
            "views": self.views.to_document(),
        }

    def fingerprint(self) -> str:
        """Return the ``"sha256:<hex>"`` identity of :meth:`to_document`."""
        canonical = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_request_policy_supported(
    policy: SemanticRequestPolicy, capabilities: SemanticInterpreterCapabilities
) -> None:
    """Refuse, before any inference, a policy an interpreter declares it cannot consume.

    Raises:
        ValueError: If the policy prompts a mode the interpreter does not support, its views
            do not fit the interpreter (:func:`check_view_policy_supported`), or it
            conditions region requests on scene context the interpreter does not accept.
    """
    for mode in SemanticInterpretationMode:
        if policy.prompt_for(mode) is not None and mode not in capabilities.supported_modes:
            raise ValueError(
                f"the request policy prompts {mode.value} requests, which the semantic "
                "interpreter does not support"
            )
    check_view_policy_supported(policy.views, capabilities)
    if policy.region_scene_context and not capabilities.accepts_scene_context:
        raise ValueError(
            "the request policy conditions region requests on scene context, but the semantic "
            "interpreter does not accept scene context"
        )
