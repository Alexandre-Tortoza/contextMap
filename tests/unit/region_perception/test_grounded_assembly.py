"""The regional assembly preserves each independently produced evidence channel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingTask,
    InlineMask,
    RefinementDiagnostics,
    Region2D,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionId,
    RegionRefinementExecution,
    RegionRefinementRequest,
    assemble_regions,
    refinement_outcome,
    refinement_prompts_from,
)
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage

SOURCE_ID = SourceObservationId("frame-0124")
REGION_RUN_ID = "region-run-0001"
WIDTH, HEIGHT = 20, 10


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256="a" * 64, media_type="image/png"
        ),
        width=WIDTH,
        height=HEIGHT,
    )


class _QueryGrounding:
    """Grounding stage backend: one query per image, answered with one box."""

    def __init__(self, *, fingerprint: str = "sha256:grounding", fails: bool = False) -> None:
        self._fingerprint = fingerprint
        self._fails = fails

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint=self._fingerprint,
        )

    def ground_image(self, image: PreparedImage) -> tuple[RegionGroundingExecution, ...]:
        if self._fails:
            raise RuntimeError("grounding runtime failed")
        request = RegionGroundingRequest(
            region_perception_run_id=REGION_RUN_ID,
            image=image,
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="fake.phrase/1",
                geometry=GroundingGeometry.BOX,
                text="the chair",
            ),
            configuration_fingerprint=self._fingerprint,
        )
        return (
            RegionGroundingExecution(
                request=request,
                provenance=self.backend_provenance(),
                rendered_prompt="find the chair",
                raw_response="<box>",
                outputs=(
                    GroundingOutput(
                        output_index=0,
                        native_text="<box>",
                        label="the chair",
                        box=BoundingBox2D(x=2, y=2, width=4, height=4),
                    ),
                ),
                diagnostics=GroundingDiagnostics(latency_ms=1.0),
                effective_configuration=MappingProxyType({"backend": "fake"}),
            ),
        )


class _ProposalRefinement:
    """Refinement stage backend: one request per image, every proposal a full-box mask."""

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake_refiner",
            capability="region_refinement",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:refiner",
        )

    def refine_groundings(
        self, image: PreparedImage, groundings: Sequence[RegionGroundingExecution]
    ) -> tuple[RegionRefinementExecution, ...]:
        prompts = tuple(prompt for item in groundings for prompt in refinement_prompts_from(item))
        request = RegionRefinementRequest(
            region_perception_run_id=REGION_RUN_ID,
            image=image,
            prompts=prompts,
            configuration_fingerprint="sha256:refiner",
        )
        pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
        pixels[3:5, 3:5] = True
        return (
            RegionRefinementExecution(
                request=request,
                provenance=self.backend_provenance(),
                outcomes=tuple(
                    refinement_outcome(
                        request=request,
                        prompt=prompt,
                        provenance=self.backend_provenance(),
                        mask=InlineMask(pixels),
                    )
                    for prompt in prompts
                ),
                diagnostics=RefinementDiagnostics(latency_ms=1.0),
                effective_configuration=MappingProxyType({"backend": "fake"}),
            ),
        )


def _sources() -> tuple[
    tuple[Region2D, ...],
    tuple[RegionGroundingExecution, ...],
    tuple[RegionRefinementExecution, ...],
]:
    image = _image()
    groundings = _QueryGrounding().ground_image(image)
    discovered = (replace(groundings[0].regions[0], region_id=RegionId("discovered-0000")),)
    refinements = _ProposalRefinement().refine_groundings(image, groundings)
    return discovered, groundings, refinements


def test_the_region_set_is_discovered_then_grounded_then_refined() -> None:
    discovered, groundings, refinements = _sources()
    result = assemble_regions(discovered=discovered, groundings=groundings, refinements=refinements)
    assert result == (*discovered, *groundings[0].regions, *refinements[0].regions)
    assert len(result) == len({region.region_id for region in result}) == 3


def test_grounding_keeps_box_evidence_without_refinement() -> None:
    discovered, groundings, _ = _sources()
    result = assemble_regions(discovered=discovered, groundings=groundings)
    assert result == (*discovered, *groundings[0].regions)
    assert all(region.mask is None for region in result)


def test_discovery_is_preserved_after_grounding_failure() -> None:
    discovered, _, _ = _sources()
    with pytest.raises(RuntimeError, match="grounding runtime failed"):
        _QueryGrounding(fails=True).ground_image(_image())
    assert assemble_regions(discovered=discovered) == discovered


def test_colliding_region_id_is_rejected_instead_of_merged() -> None:
    _, groundings, _ = _sources()
    with pytest.raises(ValueError, match="twice"):
        assemble_regions(discovered=groundings[0].regions, groundings=groundings)


def test_unsupported_assembly_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported region assembly policy"):
        assemble_regions(discovered=(), policy_id="region-set-assembly/0")


def test_nonregional_payload_is_rejected_at_the_boundary() -> None:
    with pytest.raises(TypeError, match="requires Region2D"):
        assemble_regions(discovered=("not a region",))  # type: ignore[arg-type]
