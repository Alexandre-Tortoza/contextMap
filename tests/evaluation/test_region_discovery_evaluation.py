import json
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from contextmap.evaluation import (
    EvaluatedDiscoveryFrame,
    EvaluationRunDescriptor,
    GroundTruthRegion,
    ReferenceFrame,
    RegionDiscoveryEvaluator,
    RegionDiscoveryReferenceSet,
    compare_region_discovery_reports,
    write_region_discovery_reference_set,
    write_region_discovery_report,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BackendDiagnostics,
    BoundingBox,
    DiscoveryPass,
    DiscoveryRunResult,
    InlineMask,
    PassKind,
    RegionCandidate,
    RegionProvenance,
    normalize_regions,
)
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage
from sync_recorder import SyncRecorder


def _mask(width: int, height: int, box: BoundingBox) -> InlineMask:
    return InlineMask(
        np.array(
            tuple(
                box.x_min <= x < box.x_max and box.y_min <= y < box.y_max
                for y in range(height)
                for x in range(width)
            ),
            dtype=bool,
        ).reshape(height, width)
    )


def _frame(frame_id: str, condition: str, *, annotated: bool) -> ReferenceFrame:
    width, height = 6, 4
    box = BoundingBox(1, 1, 4, 3)
    prepared = PreparedImage(
        source_observation_id=SourceObservationId(frame_id),
        payload_reference=f"reference/{frame_id}.png",
        payload_artifact=ArtifactReference(
            uri=f"reference/{frame_id}.png",
            sha256=sha256(frame_id.encode()).hexdigest(),
            media_type="image/png",
        ),
        width=width,
        height=height,
        transformations=(),
    )
    annotations = (
        (GroundTruthRegion(region_id="gt-1", mask=_mask(width, height, box)),) if annotated else ()
    )
    return ReferenceFrame(
        frame_id=frame_id,
        source_condition=condition,
        prepared_image=prepared,
        annotations=annotations,
    )


def _evaluated(
    frame: ReferenceFrame,
    backend_id: str,
    duration_ms: float,
    boxes: tuple[BoundingBox, ...] = (BoundingBox(1, 1, 4, 3),),
) -> EvaluatedDiscoveryFrame:
    candidates = tuple(
        RegionCandidate(
            candidate_id=f"{backend_id}-proposal-{index}",
            source_observation_id=frame.prepared_image.source_observation_id,
            perception_run_id=f"run-{backend_id}",
            perception_result_id=f"result-{frame.frame_id}",
            image_width=frame.prepared_image.width,
            image_height=frame.prepared_image.height,
            bounding_box=box,
            mask=_mask(frame.prepared_image.width, frame.prepared_image.height, box),
            provenance=RegionProvenance(
                backend_id=backend_id,
                backend_version="1",
                checkpoint=f"{backend_id}-checkpoint",
                config_digest=f"sha256:{backend_id}",
                discovery_pass_id="full-frame",
                native_proposal_id=f"proposal-{index}",
            ),
        )
        for index, box in enumerate(boxes, start=1)
    )
    discovery = DiscoveryRunResult(
        candidates=candidates,
        rejected=(),
        passes=(
            DiscoveryPass(
                pass_id="full-frame",
                kind=PassKind.FULL_FRAME,
                window=BoundingBox(0, 0, frame.prepared_image.width, frame.prepared_image.height),
            ),
        ),
        diagnostics=(
            BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=(),
                metadata=(("peak_memory_mb", 256.0),),
            ),
        ),
    )
    return EvaluatedDiscoveryFrame(
        discovery=discovery,
        normalization=normalize_regions(
            discovery.candidates,
            frame.prepared_image,
            BackendProvenance(
                backend_id=backend_id,
                capability="region_discovery",
                provider="test-provider",
                model=f"{backend_id}-checkpoint",
                version="1",
                configuration_fingerprint=f"sha256:{backend_id}",
            ),
        ),
    )


def _descriptor(
    backend_id: str, *, tiling: bool = False, duration_tag: str = "baseline"
) -> EvaluationRunDescriptor:
    return EvaluationRunDescriptor(
        region_artifact_id=f"artifact-{backend_id}-{duration_tag}",
        backend_id=backend_id,
        backend_version="1",
        checkpoint=f"{backend_id}-checkpoint",
        config_digest=f"sha256:{backend_id}-controlled-config",
        pipeline_graph_digest="sha256:pipeline-v1",
        strategy="automatic",
        thresholds=(("score", 0.5),),
        variables=(("tiling", tiling), ("query_strategy", "automatic")),
        execution_kind="ci_contract",
    )


def _change_checkpoint(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, checkpoint="sam3-other-checkpoint")


def _change_strategy(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, strategy="text_prompt")


def _change_threshold(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, thresholds=(("score", 0.9),))


def _change_pipeline_graph(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, pipeline_graph_digest="sha256:pipeline-v2")


def _change_config_digest(descriptor: EvaluationRunDescriptor) -> EvaluationRunDescriptor:
    return replace(descriptor, config_digest="sha256:hidden-config-change")


def test_same_reference_selection_uses_one_report_schema_for_all_backends() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1",
        frames=(
            _frame("frame-pinhole", "indoor_pinhole", annotated=True),
            _frame("frame-wide", "outdoor_wide_angle", annotated=False),
        ),
    )
    evaluator = RegionDiscoveryEvaluator(match_iou_threshold=0.5)

    reports = {
        backend: evaluator.evaluate(
            reference_set,
            _descriptor(backend),
            partial(_evaluated, backend_id=backend, duration_ms=5.0),
        )
        for backend in ("sam2", "sam3", "florence2_region_discovery")
    }

    assert {report.schema for report in reports.values()} == {
        "contextmap.region-discovery-evaluation/v2"
    }
    assert {report.frame_selection for report in reports.values()} == {
        ("frame-pinhole", "frame-wide")
    }
    sam3 = reports["sam3"]
    assert sam3.frames[0].accuracy is not None
    assert sam3.frames[0].accuracy.mean_iou == 1.0
    assert sam3.frames[0].accuracy.mean_dice == 1.0
    assert sam3.frames[0].accuracy.region_recall == 1.0
    assert sam3.frames[1].accuracy is None
    assert sam3.frames[1].diagnostics.raw_candidate_count == 1
    assert sam3.frames[1].performance.peak_memory_mb == 256.0


def test_segmentation_rates_are_undefined_when_no_region_is_discovered() -> None:
    # Sem predição as taxas não têm denominador: 0.0 seria o melhor valor possível,
    # atribuído justamente ao backend que não produziu nada.
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )

    report = RegionDiscoveryEvaluator().evaluate(
        reference_set, _descriptor("sam3"), lambda frame: _evaluated(frame, "sam3", 4.0, boxes=())
    )

    accuracy = report.frames[0].accuracy
    assert accuracy is not None
    assert accuracy.over_segmentation_rate is None
    assert accuracy.under_segmentation_rate is None
    assert accuracy.duplicate_region_rate is None
    # A falha em recuperar a região anotada continua sendo uma medição real.
    assert (accuracy.mean_iou, accuracy.region_recall, accuracy.coverage) == (0.0, 0.0, 0.0)
    persisted = json.loads(json.dumps(report.to_dict()))["frames"][0]["accuracy"]
    assert persisted["over_segmentation_rate"] is None
    assert persisted["under_segmentation_rate"] is None
    assert persisted["duplicate_region_rate"] is None


def test_region_recall_matches_each_annotation_to_its_best_prediction_without_exclusivity() -> None:
    # Pin do matching documentado: melhor IoU por ground truth, sem atribuição 1:1, então
    # uma predição que cobre duas anotações conta recall para ambas.
    frame = replace(
        _frame("frame-1", "synthetic", annotated=False),
        annotations=tuple(
            GroundTruthRegion(region_id=region_id, mask=_mask(6, 4, box))
            for region_id, box in (
                ("gt-left", BoundingBox(1, 1, 3, 3)),
                ("gt-right", BoundingBox(3, 1, 5, 3)),
            )
        ),
    )

    report = RegionDiscoveryEvaluator(match_iou_threshold=0.5).evaluate(
        RegionDiscoveryReferenceSet(version="reference-v1", frames=(frame,)),
        _descriptor("sam3"),
        lambda item: _evaluated(item, "sam3", 4.0, boxes=(BoundingBox(1, 1, 5, 3),)),
    )

    accuracy = report.frames[0].accuracy
    assert accuracy is not None
    assert accuracy.mean_iou == 0.5
    assert accuracy.region_recall == 1.0
    assert accuracy.under_segmentation_rate == 1.0


def test_report_serialization_is_deterministic_and_immutable(tmp_path: Path) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    report = RegionDiscoveryEvaluator().evaluate(
        reference_set,
        _descriptor("sam3"),
        lambda frame: _evaluated(frame, "sam3", 4.0),
    )
    output = tmp_path / "sam3-report.json"
    manifest = tmp_path / "reference-set.json"

    write_region_discovery_reference_set(manifest, reference_set)
    write_region_discovery_report(output, report)

    serialized = output.read_text()
    assert json.loads(serialized)["metric_schema_version"] == "2.0.0"
    assert json.loads(manifest.read_text())["frames"][0]["source_condition"] == "synthetic"
    with pytest.raises(FileExistsError, match="already exists"):
        write_region_discovery_report(output, report)
    assert output.read_text() == serialized


def test_the_report_and_the_reference_set_keep_their_published_bytes(tmp_path: Path) -> None:
    """The metric payload preserves its bytes while regional lineage gets its own schema.

    The digests were recorded from the private writer ``region_discovery`` had before #640.
    """
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    report = RegionDiscoveryEvaluator().evaluate(
        reference_set, _descriptor("sam3"), lambda frame: _evaluated(frame, "sam3", 4.0)
    )

    write_region_discovery_reference_set(tmp_path / "reference-set.json", reference_set)
    write_region_discovery_report(tmp_path / "report.json", report)

    assert sha256((tmp_path / "reference-set.json").read_bytes()).hexdigest() == (
        "711de454e9c01088f178313d2fda64a61e8e32232de167dc000a387ee31ee15c"
    )
    published = json.loads((tmp_path / "report.json").read_text())
    assert published["run"]["region_artifact_id"] == "artifact-sam3-baseline"
    assert not {"perception_run_id", "perception_artifact_id"} & published["run"].keys()
    # O oracle original permanece: só a identidade/schema publicados mudaram.
    published["schema"] = "contextmap.region-discovery-evaluation/v1"
    published["run"]["perception_artifact_id"] = published["run"].pop("region_artifact_id")
    published["run"]["perception_run_id"] = "run-sam3-baseline"
    historical = (
        json.dumps(published, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    assert sha256(historical).hexdigest() == (
        "250684c06b907971a29646553063b7f81c7c0ae02da5e8afa2358418e2943267"
    )


@pytest.mark.parametrize("document", ["reference-set.json", "report.json"])
def test_the_report_and_the_reference_set_reach_the_disk_before_their_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document: str
) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    report = RegionDiscoveryEvaluator().evaluate(
        reference_set, _descriptor("sam3"), lambda frame: _evaluated(frame, "sam3", 4.0)
    )
    recorder = SyncRecorder(tmp_path / document, monkeypatch)

    write_region_discovery_reference_set(tmp_path / "reference-set.json", reference_set)
    write_region_discovery_report(tmp_path / "report.json", report)

    assert recorder.synced_before_publication(tmp_path / document)
    assert recorder.synced_after_publication(tmp_path)


def test_ablation_comparison_separates_quality_and_performance() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=True, duration_tag="tiling"),
        lambda frame: _evaluated(frame, "sam3", 10.0),
    )

    comparison = compare_region_discovery_reports(baseline, changed, changed_variable="tiling")

    assert comparison.quality_deltas["mean_iou"] == 0.0
    assert comparison.performance_deltas["mean_runtime_ms"] == 2.0
    assert comparison.changed_variable == "tiling"


@pytest.mark.parametrize(
    ("field", "older"),
    [
        # #609 mudou as taxas de segmentação de 0.0 para null: 1.0.0 e 2.0.0 não medem o mesmo.
        ("metric_schema_version", "1.0.0"),
        ("schema", "contextmap.region-discovery-evaluation/v0"),
    ],
)
def test_ablation_refuses_reports_of_different_metric_schemas(field: str, older: str) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=True, duration_tag="tiling"),
        lambda frame: _evaluated(frame, "sam3", 10.0),
    )
    current = getattr(changed, field)
    older_baseline = (
        replace(baseline, metric_schema_version=older)
        if field == "metric_schema_version"
        else replace(baseline, schema=older)
    )

    with pytest.raises(ValueError, match="schema") as refused:
        compare_region_discovery_reports(older_baseline, changed, changed_variable="tiling")

    assert older in str(refused.value)
    assert current in str(refused.value)


def test_ablation_rejects_uncontrolled_variable_changes() -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed_descriptor = replace(
        _descriptor("sam3", tiling=True, duration_tag="changed"),
        variables=(("tiling", True), ("query_strategy", "text_prompt")),
    )
    changed = evaluator.evaluate(
        reference_set,
        changed_descriptor,
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )

    with pytest.raises(ValueError, match="exactly the declared variable"):
        compare_region_discovery_reports(baseline, changed, changed_variable="tiling")


@pytest.mark.parametrize(
    "mutate",
    [
        _change_checkpoint,
        _change_strategy,
        _change_threshold,
        _change_pipeline_graph,
        _change_config_digest,
    ],
)
def test_ablation_rejects_unreported_run_descriptor_changes(
    mutate: Callable[[EvaluationRunDescriptor], EvaluationRunDescriptor],
) -> None:
    reference_set = RegionDiscoveryReferenceSet(
        version="reference-v1", frames=(_frame("frame-1", "synthetic", annotated=True),)
    )
    evaluator = RegionDiscoveryEvaluator()
    baseline = evaluator.evaluate(
        reference_set,
        _descriptor("sam3", tiling=False, duration_tag="baseline"),
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )
    changed_descriptor = mutate(_descriptor("sam3", tiling=True, duration_tag="changed"))
    changed = evaluator.evaluate(
        reference_set,
        changed_descriptor,
        lambda frame: _evaluated(frame, "sam3", 8.0),
    )

    with pytest.raises(ValueError, match="uncontrolled run fields"):
        compare_region_discovery_reports(baseline, changed, changed_variable="tiling")


def test_versioned_ci_reference_and_sam3_contract_baseline_are_persisted() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "region_discovery"
    reference = json.loads((fixture_root / "reference-set-v1.json").read_text())
    baseline = json.loads((fixture_root / "sam3-ci-contract-baseline-v1.json").read_text())

    assert reference["schema"] == "contextmap.region-discovery-reference-set/v1"
    assert len({frame["source_condition"] for frame in reference["frames"]}) == 2
    assert baseline["schema"] == "contextmap.region-discovery-evaluation/v1"
    assert baseline["frame_selection"] == [frame["frame_id"] for frame in reference["frames"]]
    assert baseline["run"]["backend_id"] == "sam3"
    assert baseline["run"]["execution_kind"] == "ci_contract"
    assert baseline["run"]["checkpoint"].startswith("test://")


@pytest.mark.parametrize(
    "box",
    [
        (0.0, 0.0, 4.0, 3.0),
        (0.5, 0.5, 1.5, 1.5),
        (0.49, 0.51, 2.5, 2.49),
        (1.2, 0.0, 1.4, 3.0),
        (-3.0, -1.0, 9.0, 9.0),
    ],
)
def test_box_rasterization_is_the_full_image_pixel_centre_rule(
    box: tuple[float, float, float, float],
) -> None:
    """The bounded scan of ``box_pixels`` keeps exactly the pixels the full scan kept (#658)."""
    from contextmap.evaluation.region_discovery import box_pixels

    x_min, y_min, x_max, y_max = box
    full_scan = {
        (x, y)
        for y in range(3)
        for x in range(4)
        if x_min <= x + 0.5 < x_max and y_min <= y + 0.5 < y_max
    }

    assert (
        box_pixels(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, width=4, height=3)
        == full_scan
    )


def test_regional_descriptor_publishes_one_producer_identity() -> None:
    """Report lineage names the regional artifact consumed by this evaluator."""
    descriptor = _descriptor("sam2")
    record = descriptor.to_dict()
    assert record["region_artifact_id"] == "artifact-sam2-baseline"
    assert not {"perception_run_id", "perception_artifact_id"} & record.keys()
