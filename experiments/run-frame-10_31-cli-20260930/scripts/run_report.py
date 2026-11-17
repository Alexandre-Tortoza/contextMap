"""Métricas da run de dois frames: um braço por build (raw, voxel) e a comparação entre eles.

Uso: python run_report.py <workspace> raw=<run-NNNN> voxel=<run-NNNN> > report.json

Cada braço parte do `context_build.json` do seu build, que nomeia a fundação e os runs de
percepção e associação fornecidos; o mapa publicado é `<maps root>/<branch>/000001`, com o maps
root = pai do workspace. Lê só `outputs/`, `metrics/` e manifests (contratuais) e não recalcula
nada que um estágio já grava: conta e resume.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def _attribute(state: dict[str, Any], name: str) -> list[Any]:
    return [a["value"] for a in state.get("attributes", []) if a["name"] == name]


def arm(workspace: Path, run: str) -> dict[str, Any]:
    """Resume um braço a partir do run do build."""
    build_dir = next(workspace.glob(f"*/{run}"))
    build = _json(build_dir / "context_build.json")
    geometry = _json(workspace / build["foundation"]["geometry"]["location"] / "manifest.json")
    perception = [ref["artifact_id"] for ref in build["provided"]["visual_perception"]]

    association = {}
    for ref in build["provided"]["sensor_association"]:
        stage = workspace / ref["location"]
        metrics = _json(next((stage / "metrics").glob("*.json")))
        observations = _jsonl(stage / "outputs" / "spatial-observations.jsonl")
        supports = [o["support"]["count"] for o in observations]
        frame = observations[0]["source_observation_id"] if observations else ref["artifact_id"]
        association[frame] = {
            "run": ref["location"],
            "observations": metrics["observation_count"],
            "observations_without_support": metrics["observations_without_support"],
            "visible_points": metrics["state_counts"].get("visible"),
            "support_points": _stats([float(s) for s in supports if s]),
            "support_points_total": sum(supports),
        }

    fusion = build_dir / "semantic_fusion" / "outputs"
    entities = _jsonl(build_dir / "semantic_mapping" / "outputs" / "entity-semantic-state.jsonl")
    resolved = _jsonl(build_dir / "entity_resolution" / "outputs" / "resolved-entities.jsonl")
    relations = _jsonl(build_dir / "spatial_relations" / "outputs" / "relations.jsonl")
    candidates = _jsonl(build_dir / "spatial_relations" / "outputs" / "relation-candidates.jsonl")
    policies = _json(build_dir / "spatial_relations" / "manifest.json")["policies"]
    map_dir = workspace.parent / build["branch"] / "000001"
    map_manifest = _json(map_dir / "manifest.json")

    radii: list[float] = []
    several_radii = without_radius = 0
    belief = Counter()
    both_frames = 0
    labels = Counter()
    extent_max: list[float] = []
    for entity in resolved:
        state = entity["semantic_state"]
        values = {float(v) for v in _attribute(state, "interaction_radius_m")}
        if not values:
            without_radius += 1
        if len(values) > 1:
            several_radii += 1
        radii.extend(values)
        status = (state.get("belief") or {}).get("support_status", "no_belief")
        belief[status] += 1
        if entity["temporal_state"]["physical_observation_count"] >= 2:
            both_frames += 1
        labels.update({h["label"] for h in state.get("hypotheses", [])})
        bounds = entity["geometry"]["bounds"]
        extent_max.append(
            max(hi - lo for hi, lo in zip(bounds["maximum_m"], bounds["minimum_m"], strict=True))
        )

    by_predicate: dict[str, Counter[str]] = {}
    for relation in relations:
        by_predicate.setdefault(relation["predicate"], Counter())[relation["state"]] += 1
    generated = Counter(c["predicate"] for c in candidates if c.get("record") == "candidate")
    excluded = {
        c["nearest"][0]["predicate"]: {"count": c["count"], "min_gap_m": c["min_gap_m"]}
        for c in candidates
        if c.get("record") == "exclusion_summary" and c.get("nearest")
    }
    candidate_set = next(c for c in candidates if c.get("record") == "set")

    return {
        "build_run": str(build_dir.relative_to(workspace)),
        "context_map": {
            "location": str(map_dir),
            "context_map_id": map_manifest["context_map_id"],
            "entity_count": map_manifest["entity_count"],
            "relation_count": map_manifest["relation_count"],
        },
        "geometry": {
            "map_run": build["foundation"]["geometry"]["location"],
            "point_count": geometry["point_count"],
            "source_point_count": geometry["source_point_count"],
            "scan_count": geometry["scan_count"],
            "rejected_scan_count": geometry["rejected_scan_count"],
        },
        "perception_runs": perception,
        "association": association,
        "fusion": {
            "supports": len(_jsonl(fusion / "fusion-supports.jsonl")),
            "excluded_observations": len(_jsonl(fusion / "excluded-observations.jsonl")),
        },
        "semantic_mapping": {
            "entities": len(entities),
            "ambiguity_state": dict(Counter(e["ambiguity_state"] for e in entities)),
            "with_primary_label": sum(1 for e in entities if e.get("primary_label")),
        },
        "entity_resolution": {
            "resolved": len(resolved),
            "merged_from_several_entities": sum(1 for e in resolved if len(e["members"]) > 1),
            "seen_in_both_frames": both_frames,
            "belief_support_status": dict(belief),
            "without_interaction_radius": without_radius,
            "with_several_distinct_radii": several_radii,
            "interaction_radius_m": _stats(radii),
            "max_extent_m": _stats(extent_max),
            "distinct_labels": len(labels),
        },
        "spatial_relations": {
            "derived_interaction_reach_m": policies["candidate"]["parameters"].get(
                "interaction_radius_m"
            ),
            "interaction_reach": policies["candidate"]["parameters"].get("interaction_reach"),
            "entity_count": candidate_set["entity_count"],
            "pairs_not_enumerated": candidate_set["pairs_not_enumerated"],
            "candidates_by_predicate": dict(generated),
            "excluded_by_predicate": excluded,
            "relations_by_predicate_and_state": {p: dict(s) for p, s in by_predicate.items()},
        },
    }


def main() -> None:
    """Imprime o relatório dos braços e a comparação entre raw e voxel."""
    workspace = Path(sys.argv[1])
    arms = dict(argument.split("=", 1) for argument in sys.argv[2:])
    report: dict[str, Any] = {name: arm(workspace, run) for name, run in arms.items()}
    if {"raw", "voxel"} <= set(report):
        raw, voxel = report["raw"], report["voxel"]
        report["same_perception_in_both_arms"] = raw["perception_runs"] == voxel["perception_runs"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
