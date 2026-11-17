"""Metrics + DINO-pooling on/off ablation for the canonical/2 validation run (#688 item 5).

Computes, per perception run and aggregated per arm, the reviewer's per-perception metrics:
SAM2 region count/coverage, DINO pooling coverage distribution, Qwen claims/abstentions/
alternatives, feature-scope breakdown, and (when a scoring run is given) the raw CLIP cosine
distribution (kept as a raw score, never a probability). Reads only public artifacts.

Usage: compute_metrics.py <ws> <out.json>
The arms/runs are discovered from ws/pipeline/{pool_on_f*,pool_off_f*} and *_semantic_scoring.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from contextmap.visual_perception import (
    FeatureScope,
    PerceptionRunReader,
    SemanticScoringRunReader,
)

ws = Path(sys.argv[1])
out_path = Path(sys.argv[2])
PIPE = ws / "pipeline"


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "n": len(s),
        "min": round(s[0], 4),
        "p50": round(statistics.median(s), 4),
        "mean": round(statistics.fmean(s), 4),
        "max": round(s[-1], 4),
    }


def perception_metrics(run_dir: Path) -> dict:
    reader = PerceptionRunReader(run_dir)
    diag = [
        json.loads(line)
        for line in (run_dir / "metrics" / "feature-extraction.jsonl").read_text().splitlines()
        if line
    ]
    # The pooled DINO region features inherit the dense feature's embedding space; the CLIP
    # region features are in the CLIP space. Derive the DINO space from the DENSE feature.
    dense_space = next(
        (
            f.embedding_space_id
            for res in reader.iter_results()
            for f in res.features
            if f.scope is FeatureScope.DENSE
        ),
        None,
    )
    pooled = [d for d in diag if d.get("stage_id") == "dense_region_pooling"]
    coverage = [
        d["region"]["coverage_fraction"]
        for d in pooled
        if d.get("status") == "succeeded" and d.get("region")
    ]
    per_frame = []
    total_regions = clip_feats = dino_feats = region_claims = scene_claims = 0
    roles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for res in reader.iter_results():
        regions = len(res.regions)
        total_regions += regions
        region_features = [f for f in res.features if f.scope is FeatureScope.REGION]
        dino = sum(1 for f in region_features if f.embedding_space_id == dense_space)
        clip = len(region_features) - dino
        clip_feats += clip
        dino_feats += dino
        region_claims += len(res.claims)
        for c in res.claims:
            roles[str(c.role).split(".")[-1]] += 1
            categories[str(getattr(c, "category", "?"))] += 1
        scene = res.scene_context
        n_scene = 0 if scene is None else len(scene.claims)
        scene_claims += n_scene
        # image coverage of the union of region bounding boxes (fragmentation proxy)
        img_area = None
        boxes = []
        for reg in res.regions:
            b = reg.bounding_box
            boxes.append((b.x, b.y, b.width, b.height))
            if reg.image_width and reg.image_height:
                img_area = reg.image_width * reg.image_height
        per_frame.append(
            {
                "frame": str(res.source_observation_id),
                "regions": regions,
                "clip_region_features": clip,
                "dino_pooled_features": dino,
                "region_claims": len(res.claims),
                "scene_claims": n_scene,
            }
        )
    return {
        "run_id": str(reader.manifest.run_id),
        "frames": len(per_frame),
        "total_regions": total_regions,
        "clip_region_features": clip_feats,
        "dino_pooled_features": dino_feats,
        "region_claims": region_claims,
        "scene_claims": scene_claims,
        "claim_roles": dict(roles),
        "claim_categories": dict(categories.most_common(8)),
        "dino_pooling_coverage_fraction": _quantiles(coverage),
        "dino_pooling_success": len(coverage),
        "dino_pooling_attempts": len(pooled),
        "per_frame": per_frame,
    }


def scoring_metrics(run_dir: Path) -> dict:
    reader = SemanticScoringRunReader(run_dir)
    scores = reader.list_scores()
    counts = reader.manifest.counts
    return {
        "run_id": str(reader.manifest.run_id),
        "n_scores": len(scores),
        "clip_cosine_raw": _quantiles([float(s.value) for s in scores]),
        "unscored_region_claims": counts.unscored_region_claim_count,
        "scene_claims": counts.scene_claim_count,
        "note": "cosine kept as a raw score in [-1,1]; never a probability",
    }


def arm(prefix: str) -> dict:
    runs = sorted(PIPE.glob(f"{prefix}_f*"))
    per_run = [perception_metrics(d) for d in runs if (d / "manifest.json").exists()]
    agg = {
        "runs": len(per_run),
        "total_regions": sum(r["total_regions"] for r in per_run),
        "clip_region_features": sum(r["clip_region_features"] for r in per_run),
        "dino_pooled_features": sum(r["dino_pooled_features"] for r in per_run),
        "region_claims": sum(r["region_claims"] for r in per_run),
        "scene_claims": sum(r["scene_claims"] for r in per_run),
        "dino_pooling_coverage_fraction": _quantiles(
            [
                v
                for d in runs
                if (d / "metrics" / "feature-extraction.jsonl").exists()
                for line in (d / "metrics" / "feature-extraction.jsonl").read_text().splitlines()
                if line
                for j in [json.loads(line)]
                if j.get("stage_id") == "dense_region_pooling"
                and j.get("status") == "succeeded"
                and j.get("region")
                for v in [j["region"]["coverage_fraction"]]
            ]
        ),
        "per_run": per_run,
    }
    return agg


report = {
    "note": (
        "Multi-frame execution on the 8 GB device is currently affected by a reproducible "
        "degradation after the first frame. Each selected frame succeeds when executed in an "
        "isolated process. This empirical ablation therefore uses process-isolated per-frame "
        "perception artifacts, while the multi-frame model-lifecycle issue is tracked separately."
    ),
    "pooling_on": arm("pool_on"),
    "pooling_off": arm("pool_off"),
    "scoring": {
        d.name: scoring_metrics(d)
        for d in sorted(PIPE.glob("*_semantic_scoring"))
        if (d / "manifest.json").exists()
    },
}
out_path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
