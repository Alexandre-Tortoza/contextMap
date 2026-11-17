"""Semantic service preserves regional lineage, scene conditioning and request isolation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_producer_identity import provenance

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox2D, Region2D, RegionId
from contextmap.semantic_interpretation import (
    SemanticBackendDiagnostics,
    SemanticConfidencePolicy,
    SemanticInterpretationExecution,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
    SemanticInterpretationService,
    SemanticInterpreterCapabilities,
    SemanticObservation,
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticResponseParseError,
    SemanticViewPolicy,
    VisualViewKind,
    parse_semantic_response,
    render_semantic_prompt,
    semantic_failure_from_parse_error,
)
from contextmap.semantic_interpretation.evidence_views import materialize_scene_view
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage, SourceImage, prepare_image


class Interpreter:
    """Deterministic interpreter exercising scene conditioning and one real parse failure."""

    def __init__(self, *, fail_first_region: bool = False) -> None:
        self.requests: list[SemanticInterpretationRequest] = []
        self.fail_first_region = fail_first_region
        self.released = False

    def backend_provenance(self) -> BackendProvenance:
        return provenance().backend

    def capabilities(self) -> SemanticInterpreterCapabilities:
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset(
                {SemanticInterpretationMode.SCENE, SemanticInterpretationMode.REGION}
            ),
            supported_view_kinds=frozenset(VisualViewKind),
            accepts_visual_features=False,
            accepts_scene_context=True,
        )

    def release(self) -> None:
        self.released = True

    def interpret(self, request: SemanticInterpretationRequest) -> SemanticInterpretationExecution:
        from contextmap.semantic_interpretation.prompt_policy import semantic_prompt_template

        self.requests.append(request)
        rendered = render_semantic_prompt(
            request,
            semantic_prompt_template(request.prompt_template_id),
            confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
        )
        prov = replace(
            provenance(),
            prompt_template_id=request.prompt_template_id,
            output_schema_version=request.requested_output_schema,
        )
        diagnostics = SemanticBackendDiagnostics(latency_ms=1)
        if request.mode is SemanticInterpretationMode.REGION and self.fail_first_region:
            self.fail_first_region = False
            raise semantic_failure_from_parse_error(
                SemanticResponseParseError("invalid response"),
                request=request,
                rendered_prompt=rendered,
                raw_response="not JSON",
                provenance=prov,
                diagnostics=diagnostics,
                effective_configuration={},
            )
        raw = json.dumps(
            {
                "abstained": False,
                "claims": []
                if request.mode is SemanticInterpretationMode.SCENE
                else [
                    {
                        "hypothesis": "door",
                        "role": "primary",
                        "confidence": None,
                        "category": None,
                        "region_kind": None,
                        "attributes": {},
                    }
                ],
                "scene_context": {"scene_type": "corridor"}
                if request.mode is SemanticInterpretationMode.SCENE
                else None,
            }
        )
        return SemanticInterpretationExecution(
            request=request,
            rendered_prompt=rendered,
            raw_response=raw,
            parsed=parse_semantic_response(
                raw, request, prov, confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY
            ),
            diagnostics=diagnostics,
            effective_configuration={},
        )


def image(root: Path) -> PreparedImage:
    """Prepare a real encoded 4x4 RGB frame for the service's image boundary."""
    view = materialize_scene_view(
        np.zeros((4, 4, 3), dtype=np.uint8),
        source_observation_id=SourceObservationId("obs"),
        source_image_sha256="0" * 64,
        policy=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
    )
    (root / "frame.png").write_bytes(view.payload)
    return prepare_image(
        SourceImage(
            source_observation_id=SourceObservationId("obs"),
            image=ArtifactReference(
                uri="frame.png", sha256=view.view.sha256, media_type="image/png"
            ),
            width=4,
            height=4,
        ),
        operations=(),
    )


def regions() -> tuple[Region2D, ...]:
    """Return two frozen upstream boxes, usable without a fabricated mask."""
    return tuple(
        Region2D(
            region_id=RegionId(f"region-{index}"),
            bounding_box=BoundingBox2D(x=index, y=0, width=1, height=1),
            provenance=replace(provenance().backend, capability="region_discovery"),
        )
        for index in range(2)
    )


def test_service_preserves_each_request_and_conditions_on_same_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Interpreter(fail_first_region=True)
    policy = SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(
            scene="scene/v1", region="region-scene-context/v1", region_scene_context=True
        ),
        views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
    )
    service = SemanticInterpretationService(
        interpreter=backend,
        run_id=SemanticInterpretationRunId("semantic"),
        view_root=tmp_path,
        policy=policy,
    )
    prepared = image(tmp_path)
    monkeypatch.setattr(
        service,
        "_frame",
        lambda image: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            hashlib.sha256((tmp_path / "frame.png").read_bytes()).hexdigest(),
        ),
    )
    with SemanticInterpretationRunWriter(
        output_dir=tmp_path / "artifact",
        sequence_name="seq",
        run_id=SemanticInterpretationRunId("semantic"),
        run_index=1,
        sequence_artifact_id="seq",
        selection_id="selection",
        region_artifact_id="regions",
        configuration_digest="config",
    ) as writer:
        service.bind(writer)
        scene = service.interpret_scene(prepared)
        claims = service.interpret_regions(prepared, regions())
        writer.add_observation(
            SemanticObservation(
                source_observation_id=prepared.source_observation_id,
                semantic_interpretation_run_id=SemanticInterpretationRunId("semantic"),
                scene_context=scene,
                claims=tuple(claims),
            )
        )
        writer.finalize()
    service.release()
    assert backend.released
    assert len(backend.requests) == 3
    assert len(claims) == 1
    assert scene is not None
    for request in backend.requests[1:]:
        assert request.scene_context == scene
        assert request.scene_context_reference is not None
        assert request.scene_context_reference.evidence_id == scene.context_id
        assert request.semantic_interpretation_run_id == "semantic"
    reader = SemanticInterpretationRunReader(tmp_path / "artifact")
    assert reader.manifest.execution_count == 2
    assert reader.manifest.failure_count == 1
    assert next(reader.iter_failed_semantic_interpretations()).raw_response == "not JSON"
    assert reader.verify_integrity() == []


def test_requested_scene_conditioning_never_falls_back_to_unconditioned_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Interpreter()
    policy = SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(
            scene="scene/v1", region="region-scene-context/v1", region_scene_context=True
        ),
        views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
    )
    service = SemanticInterpretationService(
        interpreter=backend,
        run_id=SemanticInterpretationRunId("semantic"),
        view_root=tmp_path,
        policy=policy,
    )
    monkeypatch.setattr(
        service,
        "_frame",
        lambda image: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            hashlib.sha256((tmp_path / "frame.png").read_bytes()).hexdigest(),
        ),
    )
    with SemanticInterpretationRunWriter(
        output_dir=tmp_path / "artifact",
        sequence_name="seq",
        run_id=SemanticInterpretationRunId("semantic"),
        run_index=1,
        sequence_artifact_id="seq",
        selection_id="selection",
        region_artifact_id="regions",
        configuration_digest="config",
    ) as writer:
        service.bind(writer)
        with pytest.raises(ValueError, match="no scene context"):
            service.interpret_regions(image(tmp_path), regions())
    assert backend.requests == []
