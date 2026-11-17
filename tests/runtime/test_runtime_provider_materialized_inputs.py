"""Providers declared in configuration run on inputs the composition side materializes (#661).

The installed ``contextmap`` binary only gets a model runtime from a ``resources.providers``
target, a ``(config, secrets) -> runtime`` callable that knows nothing of a run. The regional
and semantic executors materialize images and semantic views in their own runs. So the
reference has to be resolved on the adapter/composition side, and the model runtime has to
receive what it can consume: the pixels of each discovery pass and the verified bytes of each
view. The runtimes here refuse anything else, as a real SDK would.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import runtime_provider_fixtures
from runtime_documents import effective_from, selected_document
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.runtime.composition import compose
from contextmap.runtime.executors import RegionPerceptionExecutor, SemanticInterpretationExecutor
from contextmap.semantic_interpretation import SemanticInterpretationRunReader
from contextmap.semantic_interpretation.adapters._semantic_views import (
    read_view_payload,
)

REGION = "region_perception.discovery"
INTERPRETER = "semantic_interpretation.interpreter"
_WIDTH, _HEIGHT = 3, 2
_BGR = (10, 20, 30)


def _request(workspace: Path, region_discovery: Any) -> Any:
    """Publish one ``bgr8`` frame, discover regions, and build the semantic request.

    The discovery runs where it belongs now: in the ``region_perception`` stage, over the same
    declared provider. Semantics consumes the artifact it published as its ``regions`` input.
    """
    from contextmap.ingestion import (
        FrameId,
        ImageEncoding,
        ImageObservation,
        SensorId,
        SequenceArtifactId,
        SequenceArtifactWriter,
        SourceObservationId,
        SourceProvenance,
    )
    from contextmap.runtime import ArtifactRef
    from contextmap.runtime.executors import inventory_digest
    from contextmap.runtime.pipeline import StageRequest
    from contextmap.shared import SourceTimestamp

    with SequenceArtifactWriter(
        output_dir=workspace / "corridor-02" / "run-0001" / "ingestion",
        sequence_name="corridor-02",
        artifact_id=SequenceArtifactId("sequence-0001"),
    ) as writer:
        writer.add_observation(
            ImageObservation(
                observation_id=SourceObservationId("frame-0000"),
                sensor_id=SensorId("camera_1"),
                frame_id=FrameId("camera_1_optical"),
                timestamp=SourceTimestamp(seconds=0, nanoseconds=0, clock_id="fixture:header"),
                provenance=SourceProvenance(source_type="fixture", source_path="fixtures/images"),
                width=_WIDTH,
                height=_HEIGHT,
                encoding=ImageEncoding.BGR8,
                data=bytes(_BGR * (_WIDTH * _HEIGHT)),
            )
        )
        manifest = writer.finalize()
    sequence = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=str(manifest.artifact_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location="corridor-02/run-0001/ingestion",
    )
    regions = RegionPerceptionExecutor(region_discovery=region_discovery).execute(
        StageRequest(
            stage_id="region_perception",
            inputs={"sequence": (sequence,)},
            components={},
            config_digest="sha256:test",
            output_dir=workspace / "corridor-02" / "run-0001" / "region_perception",
            workspace=workspace,
        )
    )
    return StageRequest(
        stage_id="semantic_interpretation",
        inputs={"sequence": (sequence,), "regions": (regions,)},
        components={},
        config_digest="sha256:test",
        output_dir=workspace / "corridor-02" / "run-0001" / "semantic_interpretation",
        workspace=workspace,
    )


def _declared_providers_document() -> dict[str, Any]:
    """SAM2 and Qwen, whose runtimes come only from ``resources.providers`` targets."""
    document = selected_document()
    document["components"]["region_perception"]["discovery"] = {
        "backend": "sam2",
        "sam2": {"checkpoint": "sam2.1_hiera_tiny.pt", "model_version": "2.1"},
    }
    document["resources"]["providers"] = {
        REGION: "runtime_provider_fixtures:load_materialized_sam2",
        INTERPRETER: "runtime_provider_fixtures:load_verified_qwen",
    }
    return document


@pytest.mark.usefixtures("fake_pillow")
def test_declared_providers_run_discovery_and_interpretation_on_materialized_inputs(
    tmp_path: Path,
) -> None:
    composed = compose(
        effective_from(tmp_path, _declared_providers_document()),
        stages=["region_perception", "semantic_interpretation"],
        module_available=lambda _name: True,
        environ={},
    )
    sam2 = runtime_provider_fixtures.SAM2_RUNTIMES[-1]
    qwen = runtime_provider_fixtures.QWEN_RUNTIMES[-1]
    assert composed.region_discovery is not None
    assert composed.semantic_interpreter is not None
    assert composed.semantic_request_policy is not None
    executor = SemanticInterpretationExecutor(
        semantic_interpreter=composed.semantic_interpreter,
        semantic_request_policy=composed.semantic_request_policy,
    )
    workspace = tmp_path / "ws"
    request = _request(workspace, composed.region_discovery)
    output_dir: Path = request.output_dir

    executor.execute(request)

    stages = [
        json.loads(line)
        for line in (output_dir / "metrics" / "stage-timings.jsonl").read_text().splitlines()
    ]
    failed = {stage["stage_id"]: stage["error"] for stage in stages if stage["error"]}
    assert failed == {}, f"stages failed on the provider runtimes: {failed}"

    # O SAM2 recebeu os pixels do pass, que são os do frame (bgr8 lido como RGB), nada além.
    expected = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    expected[...] = _BGR[::-1]
    assert len(sam2.pixels) == 1
    assert np.array_equal(sam2.pixels[0], expected)

    reader = SemanticInterpretationRunReader(output_dir)
    assert reader.verify_integrity() == []
    regional = RegionPerceptionRunReader(output_dir.parent / "region_perception")
    assert len(regional.read_regions()) == 1
    (result,) = reader.list_observations()
    assert [claim.hypothesis for claim in result.claims] == ["chair"]

    # O Qwen consumiu exatamente os bytes que o artifact publica para cada view, verificados.
    executions = reader.list_semantic_executions()
    assert sorted(execution.request.mode.value for execution in executions) == [
        "region",
        "scene",
    ]
    published = [
        read_view_payload(output_dir, view)
        for execution in executions
        for view in execution.request.visual_views
    ]
    assert sorted(qwen.payloads) == sorted(published)
    assert all(
        hashlib.sha256(payload).hexdigest()
        in {view.sha256 for e in executions for view in e.request.visual_views}
        for payload in qwen.payloads
    )

    # O diretório de rascunho é removido e nunca aparece no artifact publicado.
    scratch_prefix = f".tmp-{output_dir.name}-prepared-"
    assert list(output_dir.parent.glob(".tmp-*")) == []
    for path in output_dir.rglob("*"):
        if path.is_file():
            assert scratch_prefix.encode() not in path.read_bytes(), path
