"""Grounding and refinement evidence survives regional artifact publication."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.region_perception import (
    BackendScore,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingQuery,
    GroundingTask,
    InlineMask,
    RefinementDiagnostics,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionPerceptionRunError,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    RegionRefinementExecution,
    RegionRefinementRequest,
    refinement_outcome,
    refinement_prompts_from,
)
from contextmap.shared import ArtifactReference

from .test_artifacts import _PROVENANCE, _manifest
from .test_frame_stream import _image


def _grounding() -> RegionGroundingExecution:
    """An exact prompt, verbatim raw response and accepted box."""
    image = replace(
        _image(),
        payload_artifact=ArtifactReference(
            uri=_image().payload_reference, sha256="a" * 64, media_type="image/png"
        ),
    )
    request = RegionGroundingRequest(
        region_perception_run_id="region_run_001",
        image=image,
        query=GroundingQuery(
            task=GroundingTask.PHRASE_GROUNDING,
            geometry=GroundingGeometry.BOX,
            policy_id="ground/1",
            text="chair",
        ),
        configuration_fingerprint="ground",
    )
    return RegionGroundingExecution(
        request=request,
        provenance=replace(
            _PROVENANCE, capability="region_grounding", configuration_fingerprint="ground"
        ),
        rendered_prompt="locate chair",
        raw_response="<chair>(1,1,2,2)\n",
        outputs=(
            GroundingOutput(
                output_index=0, native_text="chair", box=BoundingBox2D(x=1, y=1, width=2, height=2)
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=1),
        effective_configuration={},
    )


def _refinement(grounding: RegionGroundingExecution) -> RegionRefinementExecution:
    """One mask-backed accepted outcome with non-probabilistic native score."""
    request = RegionRefinementRequest(
        region_perception_run_id="region_run_001",
        image=grounding.request.image,
        prompts=refinement_prompts_from(grounding),
        configuration_fingerprint="refine",
    )
    provenance = replace(
        _PROVENANCE, capability="region_refinement", configuration_fingerprint="refine"
    )
    mask = np.zeros((20, 10), dtype=bool)
    mask[1:3, 1:3] = True
    outcome = refinement_outcome(
        request=request,
        prompt=request.prompts[0],
        provenance=provenance,
        mask=InlineMask(mask),
        native_scores=(BackendScore(name="iou", value=0.9, semantics="predicted_mask_iou"),),
    )
    return RegionRefinementExecution(
        request=request,
        provenance=provenance,
        outcomes=(outcome,),
        diagnostics=RefinementDiagnostics(latency_ms=2),
        effective_configuration={},
    )


def test_regional_grounding_roundtrip_preserves_raw_prompt_and_lineage(tmp_path: Path) -> None:
    """Grounding is owned by selected regional artifact, independent of feature runs."""
    execution = _grounding()
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_frame(execution.request.image)
    writer.write_grounding(execution)
    for region in execution.regions:
        writer.write_region(region)
    writer.finalize()
    reader = RegionPerceptionRunReader(tmp_path / "run")
    assert reader.read_grounding() == (execution,)
    assert reader.verify_integrity() == []


def test_refinement_masks_externalized_once_and_shared_by_regions(tmp_path: Path) -> None:
    """Audit outcome and canonical set resolve the same mask payload."""
    grounding = _grounding()
    execution = _refinement(grounding)
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_frame(execution.request.image)
    writer.write_grounding(grounding)
    for region in grounding.regions:
        writer.write_region(region)
    externalized = writer.write_refinement(execution)
    writer.write_region(externalized.regions[0])
    writer.finalize()
    reader = RegionPerceptionRunReader(tmp_path / "run")
    restored = reader.read_refinement()[0]
    assert restored.regions[0].mask is None
    assert restored.regions[0].mask_reference == reader.read_regions()[-1].mask_reference
    assert restored.outcomes[0].native_scores == execution.outcomes[0].native_scores
    assert restored.regions[0].mask_reference is not None
    assert (
        reader.mask_resolver().resolve_mask(restored.regions[0].mask_reference)
        == execution.regions[0].mask
    )
    assert reader.verify_integrity() == []


def test_wrong_regional_producer_is_rejected_before_publication(tmp_path: Path) -> None:
    """A request from another producer cannot silently enter this run's lineage."""
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest(run_id="another-run"))
    with pytest.raises(RegionPerceptionRunError, match="run"):
        writer.write_grounding(_grounding())
