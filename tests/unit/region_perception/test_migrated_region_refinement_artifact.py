"""Refinement executions persisted with, and linked to, the perception run artifact (#568)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BackendScore,
    BoundingBox2D,
    GroundingDiagnostics,
    GroundingGeometry,
    GroundingOutput,
    GroundingPoint,
    GroundingQuery,
    GroundingTask,
    InlineMask,
    RefinementDiagnostics,
    Region2D,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    RegionRefinementExecution,
    RegionRefinementRequest,
    assemble_regions,
    refinement_outcome,
    refinement_prompts_from,
)
from contextmap.region_perception import (
    RegionPerceptionRunError as RunArtifactError,
)
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage

from .test_artifacts import _manifest

RUN_ID = "run-0001"
SOURCE_ID = SourceObservationId("frame-0124")
RESULT_ID = str(RUN_ID)
WIDTH, HEIGHT = 20, 10
REFINER = BackendProvenance(
    backend_id="sam2",
    capability="region_refinement",
    provider="facebook",
    model="sam2.1_hiera_large",
    version="2.1",
    configuration_fingerprint="sha256:refiner",
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SOURCE_ID,
        payload_reference="frame-0124.png",
        payload_artifact=ArtifactReference(
            uri="frame-0124.png", sha256="b" * 64, media_type="image/png"
        ),
        width=WIDTH,
        height=HEIGHT,
    )


def _grounding(text: str = "the chair") -> RegionGroundingExecution:
    return RegionGroundingExecution(
        request=RegionGroundingRequest(
            region_perception_run_id=RESULT_ID,
            image=_image(),
            query=GroundingQuery(
                task=GroundingTask.PHRASE_GROUNDING,
                policy_id="fake.phrase/1",
                geometry=GroundingGeometry.BOX,
                text=text,
            ),
            configuration_fingerprint="sha256:grounding",
        ),
        provenance=BackendProvenance(
            backend_id="fake_grounding",
            capability="region_grounding",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:grounding",
        ),
        rendered_prompt=f"find {text}",
        raw_response="<box><100><200><600><800></box><box><900><900></box>",
        outputs=(
            GroundingOutput(
                output_index=0,
                native_text="<box><100><200><600><800></box>",
                box=BoundingBox2D(x=2.0, y=2.0, width=10.0, height=6.0),
            ),
            GroundingOutput(
                output_index=1,
                native_text="<box><900><900></box>",
                point=GroundingPoint(x=18.0, y=9.0),
            ),
        ),
        diagnostics=GroundingDiagnostics(latency_ms=1.0),
        effective_configuration=MappingProxyType({}),
    )


def _mask(x0: int, y0: int, x1: int, y1: int) -> InlineMask:
    pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pixels[y0:y1, x0:x1] = True
    return InlineMask(pixels)


def _refinement(grounding: RegionGroundingExecution) -> RegionRefinementExecution:
    request = RegionRefinementRequest(
        region_perception_run_id=RESULT_ID,
        image=_image(),
        prompts=refinement_prompts_from(grounding),
        configuration_fingerprint="sha256:refiner",
    )
    box, point = request.prompts
    score = BackendScore(name="predicted_iou", value=0.91, semantics="SAM2-native predicted IoU")
    return RegionRefinementExecution(
        request=request,
        provenance=REFINER,
        outcomes=(
            refinement_outcome(
                request=request,
                prompt=box,
                provenance=REFINER,
                mask=_mask(4, 3, 9, 7),
                native_scores=(score,),
            ),
            refinement_outcome(
                request=request, prompt=point, provenance=REFINER, mask=_mask(0, 0, 0, 0)
            ),
        ),
        diagnostics=RefinementDiagnostics(latency_ms=2.0),
        effective_configuration=MappingProxyType({"checkpoint": "sam2.1_hiera_large"}),
    )


def _regions(
    groundings: tuple[RegionGroundingExecution, ...],
    refinements: tuple[RegionRefinementExecution, ...],
) -> tuple[Region2D, ...]:
    return assemble_regions(discovered=(), groundings=groundings, refinements=refinements)


def _writer(tmp_path: Path) -> RegionPerceptionRunWriter:
    writer = RegionPerceptionRunWriter(tmp_path / "run")
    writer.write_manifest(_manifest(run_id=str(RUN_ID), source_observation_ids=(str(SOURCE_ID),)))
    return writer


def _persist_regions(writer: RegionPerceptionRunWriter, regions: tuple[Region2D, ...]) -> None:
    writer.write_frame(_grounding().request.image)
    for region in regions:
        writer.write_region(writer.externalize_region_mask(SOURCE_ID, region))


def _write(tmp_path: Path) -> tuple[RegionGroundingExecution, RegionRefinementExecution]:
    grounding = _grounding()
    refinement = _refinement(grounding)
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((grounding,), (refinement,)))
    writer.write_grounding(grounding)
    writer.write_refinement(refinement)
    writer.finalize()
    return grounding, refinement


def test_refined_masks_and_their_lineage_survive_the_run_artifact(tmp_path: Path) -> None:
    grounding, refinement = _write(tmp_path)

    reader = RegionPerceptionRunReader(tmp_path / "run")
    (reopened,) = reader.read_refinement()
    regions = reader.read_regions()
    refined_id = refinement.regions[0].region_id

    assert reader.verify_integrity() == []
    assert reopened.request == refinement.request
    assert reopened.outcomes[1] == refinement.outcomes[1]
    refined = next(region for region in regions if region.region_id == refined_id)
    assert reopened.outcomes[0].region == refined
    assert refined.mask is None and refined.mask_reference is not None
    assert refined.contributor_candidate_ids == (refinement.request.prompts[0].proposal_id,)
    assert reader.mask_store().load(SOURCE_ID, refined_id) == _mask(4, 3, 9, 7)
    assert reopened.outcomes[0].native_scores == refinement.outcomes[0].native_scores
    assert regions[0] == grounding.regions[0]
    assert (tmp_path / "run" / "outputs/refinement.jsonl").is_file()


def test_the_refinement_stream_never_inlines_mask_pixels(tmp_path: Path) -> None:
    _write(tmp_path)

    record = json.loads((tmp_path / "run" / "outputs/refinement.jsonl").read_text())

    assert "mask" not in record["outcomes"][0]["region"]
    assert record["outcomes"][0]["region"]["mask_reference"].startswith("outputs/masks/")


@pytest.mark.parametrize(
    "line", ["not json", "[]", "{}"], ids=["not-json", "not-an-object", "empty"]
)
def test_a_malformed_refinement_line_is_reported_with_its_file_and_line(
    tmp_path: Path, line: str
) -> None:
    """#644: every stream the reader decodes names the file and line of a malformed record."""
    _write(tmp_path)
    stream = tmp_path / "run" / "outputs/refinement.jsonl"
    stream.write_text(f"{stream.read_text(encoding='utf-8')}{line}\n", encoding="utf-8")

    with pytest.raises(RunArtifactError, match=r"outputs/refinement\.jsonl:2"):
        RegionPerceptionRunReader(tmp_path / "run").read_refinement()


def test_a_run_without_refinement_has_no_refinement_stream(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((), ()))
    manifest = writer.finalize()

    assert RegionPerceptionRunReader(tmp_path / "run").read_refinement() == ()
    assert any(entry.path == "outputs/refinement.jsonl" for entry in manifest.file_inventory)


def test_finalize_rejects_refined_regions_missing_from_the_result(tmp_path: Path) -> None:
    grounding = _grounding()
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((grounding,), ()))
    writer.write_grounding(grounding)
    writer.write_refinement(_refinement(grounding))

    with pytest.raises(RunArtifactError, match="materialized"):
        writer.finalize()


def test_finalize_rejects_a_prompt_whose_grounding_is_not_in_the_run(tmp_path: Path) -> None:
    grounding = _grounding()
    refinement = _refinement(grounding)
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((grounding,), (refinement,)))
    writer.write_refinement(refinement)

    with pytest.raises(RunArtifactError, match="grounding"):
        writer.finalize()


def test_finalize_rejects_a_prompt_that_is_not_the_grounding_output_it_names(
    tmp_path: Path,
) -> None:
    grounding = _grounding()
    refinement = _refinement(grounding)
    point_prompt = refinement.request.prompts[1]
    moved = replace(point_prompt, point=GroundingPoint(x=1.0, y=1.0))
    tampered_request = replace(refinement.request, prompts=(refinement.request.prompts[0], moved))
    tampered = replace(
        refinement,
        request=tampered_request,
        outcomes=(
            refinement.outcomes[0],
            replace(refinement.outcomes[1], prompt=moved),
        ),
    )
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((grounding,), (tampered,)))
    writer.write_grounding(grounding)
    writer.write_refinement(tampered)

    with pytest.raises(RunArtifactError, match="grounding output"):
        writer.finalize()


def test_writer_rejects_the_same_refinement_request_twice(tmp_path: Path) -> None:
    refinement = _refinement(_grounding())
    writer = _writer(tmp_path)
    writer.write_refinement(refinement)

    with pytest.raises(RunArtifactError, match="duplicate refinement"):
        writer.write_refinement(refinement)


def test_writer_refuses_refinement_after_finalize(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    _persist_regions(writer, _regions((), ()))
    writer.finalize()

    with pytest.raises(RunArtifactError, match="finalize"):
        writer.write_refinement(_refinement(_grounding()))
