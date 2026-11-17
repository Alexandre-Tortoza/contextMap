"""Annotation-free metrics of the perception pilot (#665), read through the public readers.

No reference set exists for corridor-02 (#581), so nothing here is a quality claim. Each arm
reports what its perception run did (regions per frame, semantic requests and their outcomes,
abstentions, the labels it proposed); the baseline and its replica report how often their labels
agree over regions a declared join pairs, which is the noise floor the other arms' differences
are read against. Every count is reported on its own; nothing is combined into a score.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from contextmap.evaluation import (
    Coverage,
    FrameRegionAnnotation,
    FrameRegionJoin,
    ReferenceSampleId,
    RegionAnnotation,
    RegionJoinPolicy,
    account_semantic_attempts,
    join_frame_regions,
    summarize_region_joins,
)
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import BoundingBox, Region2D, RegionPerceptionRunReader
from contextmap.semantic_interpretation import HypothesisRole, SemanticInterpretationRunReader
from contextmap.visual_perception import PerceptionResult

REPLICA_JOIN = RegionJoinPolicy(policy_id="pilot-replica-join/v1", min_iou=0.5)
"""The join that pairs the baseline's regions with the replica's, declared before the run."""
TOP_LABELS = 20
"""How many of the most frequent region labels each arm reports."""


def perception_metrics(run_dir: Path, *, region_run: Path) -> dict[str, Any]:
    """Return the annotation-free metrics of one arm's perception run.

    Args:
        run_dir: The finalized semantic interpretation artifact.
        region_run: The explicitly selected upstream regional artifact.

    Returns:
        ``frames`` (results of the run), ``regions_per_frame`` (accepted regions: min, median,
        max, total), ``semantic_requests`` (the reconciled accounting of the three outcome
        streams), ``backend_call_failures`` (calls that failed before answering, by error
        type), ``abstention`` (by mode, parsed responses that abstained) and ``region_labels``
        (primary hypotheses of region claims).
    """
    reader = SemanticInterpretationRunReader(run_dir)
    results = tuple(_frames(run_dir, region_run=region_run).values())
    regions = [sum(1 for region in result.regions if region.is_accepted) for result in results]
    accounting = account_semantic_attempts(reader)
    call_failures = Counter(
        failed.error_type for failed in reader.iter_failed_semantic_backend_calls()
    )
    parsed: Counter[str] = Counter()
    abstained: Counter[str] = Counter()
    for execution in reader.iter_semantic_executions():
        mode = execution.request.mode.value
        parsed[mode] += 1
        abstained[mode] += execution.parsed.abstained
    # Uma região com mais de uma hipótese primária entra com todas, na ordem alfabética.
    labels = Counter(
        " | ".join(sorted(label))
        for result in results
        for label in _primary_labels(result).values()
    )
    return {
        "frames": len(results),
        "regions_per_frame": {
            "min": min(regions, default=None),
            "median": statistics.median(regions) if regions else None,
            "max": max(regions, default=None),
            "total": sum(regions),
        },
        "semantic_requests": {
            "attempted": accounting.attempted,
            "parsed": accounting.parsed,
            "parse_failed": accounting.parse_failed,
            "call_failed": accounting.call_failed,
            "parse_failure_rate": accounting.parse_failure_rate,
            "call_failure_rate": accounting.call_failure_rate,
            "measurement_status": accounting.measurement_status.value,
        },
        "backend_call_failures": dict(sorted(call_failures.items())),
        "abstention": {
            mode: {
                "parsed": parsed[mode],
                "abstained": abstained[mode],
                "rate": abstained[mode] / parsed[mode],
            }
            for mode in sorted(parsed)
        },
        "region_labels": {
            "labeled_regions": sum(labels.values()),
            "distinct": len(labels),
            "top": [
                [label, count]
                for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0]))
            ][:TOP_LABELS],
        },
    }


def replica_agreement(
    baseline_run: Path,
    replica_run: Path,
    policy: RegionJoinPolicy = REPLICA_JOIN,
    *,
    baseline_regions: Path,
    replica_regions: Path,
) -> dict[str, Any]:
    """Return the label agreement of the baseline and its replica over joined regions.

    Masks are loaded from each run's mask store, so the join measures mask IoU, never the box
    in the mask's place.
    """
    return label_agreement(
        _frames(baseline_run, region_run=baseline_regions),
        _frames(replica_run, region_run=replica_regions),
        policy,
    )


def label_agreement(
    baseline: Mapping[str, PerceptionResult],
    replica: Mapping[str, PerceptionResult],
    policy: RegionJoinPolicy,
) -> dict[str, Any]:
    """Pair the accepted regions of two runs frame by frame and compare their labels.

    The baseline's regions take the reference side of :func:`join_frame_regions` (one to one,
    greedy by IoU, ``policy.min_iou`` at least); no annotation is implied. A region's label is
    the set of its primary region-claim hypotheses, exact text: a region without one is
    unlabeled (it abstained, failed, or was never interpreted). Only pairs labeled on both sides
    are compared, so ``agreement`` is ``agreeing / compared`` and ``None`` when nothing was
    compared; the other pairs are counted, never folded into it.

    Args:
        baseline: The baseline's results, by observation, with masks loaded.
        replica: The replica's results, by observation, with masks loaded.
        policy: The declared join.

    Returns:
        The policy, the frames joined, the join counts and the label counts.

    Raises:
        RegionJoinError: If a frame's regions cannot be joined (see ``join_frame_regions``).
    """
    joins: list[FrameRegionJoin] = []
    counts: Counter[str] = Counter()
    for observation in sorted(set(baseline) | set(replica)):
        if observation not in baseline or observation not in replica:
            counts["frames_in_one_run_only"] += 1
            continue
        left, right = baseline[observation], replica[observation]
        if not _accepted(left) and not _accepted(right):
            counts["frames_without_regions"] += 1
            continue
        join = join_frame_regions(policy, _as_reference(left, right), _accepted(right))
        joins.append(join)
        left_labels, right_labels = _primary_labels(left), _primary_labels(right)
        for match in join.matches:
            first = left_labels.get(match.reference_region_id)
            second = right_labels.get(str(match.run_region_id))
            if first is None and second is None:
                counts["unlabeled_on_both"] += 1
            elif first is None or second is None:
                counts["labeled_on_one_side"] += 1
            else:
                counts["compared"] += 1
                counts["agreeing"] += first == second
    summary = summarize_region_joins(joins) if joins else None
    return {
        "policy_id": policy.policy_id,
        "min_iou": policy.min_iou,
        "frames": len(joins),
        "frames_in_one_run_only": counts["frames_in_one_run_only"],
        "frames_without_regions": counts["frames_without_regions"],
        "matched": 0 if summary is None else summary.matched_count,
        "unmatched_baseline": 0 if summary is None else summary.unmatched_reference_count,
        "unmatched_replica": 0 if summary is None else summary.unmatched_run_count,
        "compared": counts["compared"],
        "agreeing": counts["agreeing"],
        "agreement": counts["agreeing"] / counts["compared"] if counts["compared"] else None,
        "labeled_on_one_side": counts["labeled_on_one_side"],
        "unlabeled_on_both": counts["unlabeled_on_both"],
    }


def _frames(run_dir: Path, *, region_run: Path) -> dict[str, PerceptionResult]:
    """Compose explicitly selected regional/semantic observations, loading regional masks."""
    semantic = SemanticInterpretationRunReader(run_dir)
    regional = RegionPerceptionRunReader(region_run)
    if semantic.manifest.region_artifact_id != str(regional.manifest.run_id):
        raise ValueError("pilot semantic producer does not select the supplied regional artifact")
    if semantic.manifest.sequence_artifact_id != regional.manifest.sequence_artifact_id:
        raise ValueError("pilot producers must describe the same canonical sequence")
    issues = [*semantic.verify_integrity(), *regional.verify_integrity()]
    if issues:
        raise ValueError("; ".join(issues))
    evidence = {item.source_observation_id: item for item in semantic.iter_observations()}
    masks = regional.mask_store()
    region_records = regional.read_regions()
    frames: dict[str, PerceptionResult] = {}
    for frame in regional.iter_frames():
        observation = frame.image.source_observation_id
        if observation not in evidence:
            raise ValueError(f"pilot has no semantic envelope for {observation!r}")
        regions = tuple(
            replace(region, mask=masks.load(observation, region.region_id))
            if region.mask is None and region.mask_reference is not None
            else region
            for region in region_records
            if region.source_observation_id == observation
        )
        item = evidence.pop(observation)
        frames[str(observation)] = PerceptionResult(
            source_observation_id=observation,
            sequence_artifact_id=regional.manifest.sequence_artifact_id,
            region_artifact_id=str(regional.manifest.run_id),
            semantic_interpretation_artifact_id=str(semantic.manifest.run_id),
            regions=regions,
            claims=item.claims,
            scene_context=item.scene_context,
        )
    if evidence:
        raise ValueError("pilot semantic envelopes have no corresponding regional frame")
    return frames


def _accepted(result: PerceptionResult) -> tuple[Region2D, ...]:
    """Return the regions of the frozen canonical set of a result."""
    return tuple(region for region in result.regions if region.is_accepted)


def _as_reference(baseline: PerceptionResult, replica: PerceptionResult) -> FrameRegionAnnotation:
    """Put the baseline's accepted regions on the reference side of the join.

    Raises:
        ValueError: If neither run records the image size of the frame.
    """
    sizes = {
        (region.image_width, region.image_height)
        for region in (*_accepted(baseline), *_accepted(replica))
        if region.image_width is not None and region.image_height is not None
    }
    if len(sizes) != 1:
        raise ValueError(
            f"frame {baseline.source_observation_id!r} needs exactly one image size to join its "
            f"regions, the runs record {sorted(sizes)}"
        )
    ((width, height),) = sizes
    return FrameRegionAnnotation(
        sample_id=ReferenceSampleId(str(baseline.source_observation_id)),
        observation_id=SourceObservationId(str(baseline.source_observation_id)),
        image_width=width,
        image_height=height,
        coverage=Coverage.PARTIAL,
        regions=tuple(
            RegionAnnotation(
                region_id=str(region.region_id),
                mask=region.mask,
                box=BoundingBox(
                    x_min=region.bounding_box.x_min,
                    y_min=region.bounding_box.y_min,
                    x_max=region.bounding_box.x_max,
                    y_max=region.bounding_box.y_max,
                ),
            )
            for region in _accepted(baseline)
        ),
    )


def _primary_labels(result: PerceptionResult) -> dict[str, frozenset[str]]:
    """Return, by region, the set of its primary region-claim hypotheses; unlabeled are absent.

    Scene claims have no region and are never a region's label.
    """
    labels: dict[str, set[str]] = {}
    for claim in result.claims:
        if claim.region_id is not None and claim.role is HypothesisRole.PRIMARY:
            labels.setdefault(str(claim.region_id), set()).add(claim.hypothesis)
    return {region: frozenset(hypotheses) for region, hypotheses in labels.items()}
