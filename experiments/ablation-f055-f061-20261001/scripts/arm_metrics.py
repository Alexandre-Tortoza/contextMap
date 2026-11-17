"""Métricas de qualidade e custo de um braço da ablação f055/f061.

Uso: python arm_metrics.py <workspace> <map_id> [<map_id> ...] > results/metrics.json

Cada braço é uma branch (``map_id``) do run store ``workspace``. O script acha o último build da
branch, lê os runs de percepção e associação que ele fundiu e o mapa publicado, e mede contra a
anotação fraca dos dois frames (``annotations/frame-NN.json``). Lê só ``outputs/``, ``metrics/``,
manifests e os ``*.cost.json`` gravados pelo ``env.sh`` (tempo de parede e pico de GPU por
processo); não recalcula nada que um estágio já grava, conta e resume.

Definições (declaradas antes de rodar os braços):

- casamento região↔instância: IoU de caixas >= 0,5 (a caixa de uma região com máscara é a da
  máscara), a mesma representação para todos os braços; cada instância casa com a melhor região;
- ``known_object_recall``: instâncias não estruturais (sem parede/teto/piso) com casamento;
  recortes: vocabulário do C4 (door, door_handle, window) e classes extras;
- ``fragments_per_object``: regiões com >= 50 % dos seus pixels (máscara ou caixa) dentro da caixa
  da instância, por instância com pelo menos uma; mede super-segmentação;
- ``label_correct``: label primária da região casada pertence a ``accepted_labels`` (casefold
  exato, com ``_`` trocado por espaço na classe);
- ``depth_outlier_fraction``: por observação com suporte, fração de pontos cuja profundidade se
  afasta mais de 0,5 m da mediana do suporte (mistura objeto/fundo);
- false merge: entidade resolvida com observações casadas a instâncias diferentes;
  false split: instância cujas observações casadas acabam em duas ou mais entidades resolvidas.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
FRAMES = ("000055", "000061")
WIDTH, HEIGHT = 640, 480
MATCH_IOU = 0.5
INSIDE_FRACTION = 0.5
DEPTH_OUTLIER_M = 0.5
C4_CLASSES = {"door", "door_handle", "window"}

sys.path.insert(0, str(BUNDLE.parent / "run-frame-10_31-cli-20260930" / "scripts"))
from run_report import arm as map_report  # noqa: E402


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _mask(stage: Path, frame: str, region_id: str) -> np.ndarray | None:
    path = stage / "outputs" / "masks" / f"camera_1_image_raw-{frame}" / f"{region_id}.npy"
    if not path.exists():
        return None
    raw = np.load(path)
    if raw.shape != (HEIGHT, WIDTH):
        raw = np.unpackbits(raw)[: HEIGHT * WIDTH].reshape(HEIGHT, WIDTH)
    return raw.astype(bool)


def _region_box(region: dict[str, Any], mask: np.ndarray | None) -> tuple[float, ...]:
    if mask is not None and mask.any():
        rows, cols = np.nonzero(mask)
        return (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    b = region["bounding_box"]
    return (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])


def _inside_fraction(region_box: tuple[float, ...], mask: np.ndarray | None, box: list[float]) -> float:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if mask is not None:
        total = int(mask.sum())
        return float(mask[y0:y1, x0:x1].sum()) / total if total else 0.0
    area = (region_box[2] - region_box[0]) * (region_box[3] - region_box[1])
    inter = max(0.0, min(region_box[2], x1) - max(region_box[0], x0)) * max(
        0.0, min(region_box[3], y1) - max(region_box[1], y0)
    )
    return inter / area if area else 0.0


def _label_ok(label: str | None, instance: dict[str, Any]) -> bool:
    if not label:
        return False
    accepted = {a.casefold() for a in instance["accepted_labels"]}
    accepted.add(instance["class"].replace("_", " ").casefold())
    return label.strip().casefold() in accepted


def perception_frame(
    stage: Path, frame: str, annotation: dict[str, Any], result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Regiões, labels e custo de um run de percepção contra a anotação do frame.

    ``result`` é o resultado do frame quando o run cobre vários frames; por padrão, o único
    resultado do run (os braços da ablação rodaram um frame por run).
    """
    result = result or _jsonl(stage / "outputs" / "results.jsonl")[0]
    regions = [r for r in result["regions"] if r.get("is_accepted", True)]
    primary = {c["region_id"]: c["hypothesis"] for c in result["claims"] if c["role"] == "primary"}
    masks = {r["region_id"]: _mask(stage, frame, r["region_id"]) for r in regions}
    boxes = {r["region_id"]: _region_box(r, masks[r["region_id"]]) for r in regions}
    areas = [float(r["area_pixels"]) for r in regions]

    objects = [i for i in annotation["instances"] if not i.get("structure")]
    matches: dict[str, dict[str, Any]] = {}
    fragments: dict[str, int] = {}
    for instance in annotation["instances"]:
        box = tuple(float(v) for v in instance["box_xyxy"])
        scored = [(rid, _iou(boxes[rid], box)) for rid in boxes]
        best = max(scored, key=lambda item: item[1], default=(None, 0.0))
        if best[0] is not None and best[1] >= MATCH_IOU:
            matches[instance["instance_id"]] = {
                "region_id": best[0],
                "iou": round(best[1], 3),
                "label": primary.get(best[0]),
                "label_correct": _label_ok(primary.get(best[0]), instance),
            }
        inside = sum(
            1
            for rid in boxes
            if _inside_fraction(boxes[rid], masks[rid], instance["box_xyxy"]) >= INSIDE_FRACTION
        )
        if not instance.get("structure"):
            fragments[instance["instance_id"]] = inside

    def recall(selection: list[dict[str, Any]]) -> dict[str, Any]:
        hit = [i["instance_id"] for i in selection if i["instance_id"] in matches]
        return {"matched": len(hit), "total": len(selection), "recall": round(len(hit) / len(selection), 3) if selection else None}

    matched_objects = [matches[i["instance_id"]] for i in objects if i["instance_id"] in matches]
    observation = result["source_observation_id"]
    interpretations = [
        i
        for i in _jsonl(stage / "outputs" / "semantic-interpretations.jsonl")
        if _request_frame(i) in (None, observation)
    ]
    failures = [
        i
        for i in _jsonl(stage / "outputs" / "semantic-interpretation-failures.jsonl")
        if _request_frame(i) in (None, observation)
    ]
    # Run com vários frames: o tempo de cada estágio é dividido igualmente entre eles.
    share = 1 / max(1, len(_jsonl(stage / "outputs" / "results.jsonl")))
    timings: dict[str, float] = {}
    for t in _jsonl(stage / "metrics" / "stage-timings.jsonl"):
        timings[t["stage_id"]] = timings.get(t["stage_id"], 0.0) + share * t["duration_ms"] / 1000
    abstained = sum(1 for i in interpretations if (i.get("parsed") or {}).get("abstained"))
    implicated = {m["region_id"] for m in matches.values()}
    return {
        "perception_run": str(stage.parent.name),
        "regions": len(regions),
        "region_area_px": {
            "median": statistics.median(areas) if areas else None,
            "below_0_5pct_frame": sum(1 for a in areas if a < 0.005 * WIDTH * HEIGHT),
        },
        "known_object_recall": recall(objects),
        "known_object_recall_c4": recall([i for i in objects if i["class"] in C4_CLASSES]),
        "known_object_recall_extra": recall([i for i in objects if i["class"] not in C4_CLASSES]),
        "structure_recall": recall([i for i in annotation["instances"] if i.get("structure")]),
        "fragments_per_object": {
            "mean_over_hit": round(statistics.mean(v for v in fragments.values() if v), 2)
            if any(fragments.values())
            else None,
            "max": max(fragments.values(), default=0),
            "by_instance": fragments,
        },
        "regions_not_matching_any_instance": len(set(boxes) - implicated),
        "label_correct": {
            "correct": sum(1 for m in matched_objects if m["label_correct"]),
            "matched": len(matched_objects),
        },
        "matches": matches,
        "semantic_calls": len(interpretations) + len(failures),
        "semantic_parse_failures": len(failures),
        "abstentions": abstained,
        "primary_labels": dict(Counter(primary.values()).most_common()),
        "stage_seconds": timings,
        "perception_seconds": round(sum(timings.values()), 2),
    }


def _request_frame(record: dict[str, Any]) -> str | None:
    """A observação de origem de um registro de interpretação, quando ele a nomeia."""
    request = record.get("request") or {}
    return request.get("source_observation_id") or record.get("source_observation_id")


def _discovery_backend(stage: Path) -> str | None:
    audits = _jsonl(stage / "outputs" / "region-discovery-audit.jsonl")
    return audits[0]["backend"]["backend_id"] if audits else None


def association_frame(stage: Path, frame: str | None = None) -> dict[str, Any]:
    """Suporte 3D e regras de pertencimento de um run de associação, num frame ou em todos."""
    manifest = _json(stage / "manifest.json")

    def mine(record: dict[str, Any]) -> bool:
        return frame is None or str(record.get("source_observation_id", "")).endswith(frame)

    records = [
        v
        for v in _jsonl(stage / "outputs" / "visibility-records.jsonl")
        if "regions" in v and mine(v)
    ]
    qualities = [q for q in _jsonl(stage / "outputs" / "observation-quality.jsonl") if mine(q)]
    observations = [
        o for o in _jsonl(stage / "outputs" / "spatial-observations.jsonl") if mine(o)
    ]
    supported = [o for o in observations if o["support"]["count"]]
    skipped = Counter(s["reason"] for v in records for s in v["skipped_regions"])
    excluded = sum(r.get("excluded_visible_count", 0) for v in records for r in v["regions"])
    return {
        "association_run": str(stage.parent.name),
        "membership_policy_id": manifest["membership_policy_id"],
        "observations": len(observations),
        "observations_with_support": len(supported),
        "support_points_total": sum(o["support"]["count"] for o in observations),
        "skipped_regions": dict(skipped),
        "excluded_visible_points": excluded,
        "footprint_kinds": dict(Counter(q.get("footprint_kind") for q in qualities)),
    }


def depth_outliers(stage: Path, frame: str | None = None) -> dict[str, Any]:
    """Fração de pontos de suporte longe da mediana de profundidade, por observação com suporte.

    Usa a profundidade gravada por ponto em ``projection-records.jsonl`` quando existe; senão,
    a amplitude (máximo − mínimo) do resumo de profundidade da qualidade.
    """
    qualities = [
        q
        for q in _jsonl(stage / "outputs" / "observation-quality.jsonl")
        if frame is None or str(q.get("source_observation_id", "")).endswith(frame)
    ]
    spreads = [
        q["support_depth_m"]["maximum"] - q["support_depth_m"]["minimum"]
        for q in qualities
        if q.get("support_depth_m")
    ]
    return {
        "support_depth_range_m": {
            "median": round(statistics.median(spreads), 3) if spreads else None,
            "share_over_1m": round(sum(1 for s in spreads if s > 1.0) / len(spreads), 3) if spreads else None,
        }
    }


def merges_and_splits(build: Path, matches_by_frame: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """False merge / false split das entidades resolvidas contra as instâncias anotadas."""
    region_instance: dict[tuple[str, str], str] = {}
    for frame, matches in matches_by_frame.items():
        for instance_id, match in matches.items():
            region_instance[(frame, match["region_id"])] = instance_id
    resolved = _jsonl(build / "entity_resolution" / "outputs" / "resolved-entities.jsonl")
    instance_entities: dict[str, set[str]] = defaultdict(set)
    merges = []
    for entity in resolved:
        instances = set()
        for observation_id in entity["evidence"]["spatial_observation_ids"]:
            parts = observation_id.split("--")
            frame, region = parts[-2][-6:], parts[-1]
            instance = region_instance.get((frame, region))
            if instance is not None:
                instances.add(instance)
                instance_entities[instance].add(entity["resolved_entity_id"])
        if len(instances) > 1:
            merges.append(sorted(instances))
    splits = {i: len(e) for i, e in instance_entities.items() if len(e) > 1}
    return {
        "resolved_with_annotated_instance": sum(
            1
            for entity in resolved
            if any(
                region_instance.get((o.split("--")[-2][-6:], o.split("--")[-1]))
                for o in entity["evidence"]["spatial_observation_ids"]
            )
        ),
        "false_merges": merges,
        "false_splits": splits,
        "instances_in_one_entity_in_both_frames": sum(
            1
            for instance, entities in instance_entities.items()
            if len(entities) == 1
            and all(instance in matches_by_frame[f] for f in FRAMES)
        ),
    }


def cost(map_id: str, frame_runs: dict[str, Any]) -> dict[str, Any]:
    logs = BUNDLE / "logs"
    per_process = {}
    for name in [*FRAMES, "build"]:
        path = logs / f"{map_id}-{name}.cost.json"
        if path.exists():
            per_process[name] = _json(path)
    perception = sum(f["perception_seconds"] for f in frame_runs.values())
    calls = sum(f["semantic_calls"] for f in frame_runs.values())
    return {
        "perception_seconds_two_frames": round(perception, 2),
        "semantic_calls_two_frames": calls,
        "per_process": per_process,
        "gpu_peak_mib": max((p["gpu_peak_mib"] for p in per_process.values()), default=None),
    }


def arm(workspace: Path, map_id: str) -> dict[str, Any]:
    builds = sorted(
        run
        for run in workspace.glob("*/run-*")
        if (run / "context_build.json").exists()
        and _json(run / "context_build.json")["branch"] == map_id
    )
    build = builds[-1]
    record = _json(build / "context_build.json")
    annotations = {f: _json(BUNDLE / "annotations" / f"frame-{f[-2:]}.json") for f in FRAMES}
    frames: dict[str, Any] = {}
    association: dict[str, Any] = {}
    human: dict[str, Any] = {}
    human_runs: set[str] = set()
    for ref in record["provided"]["visual_perception"]:
        stage = workspace / ref["location"]
        # A busca de pessoas (SAM3) é um canal à parte: entra no mapa, mas não nas métricas
        # de descoberta contra a anotação de objetos.
        if _discovery_backend(stage) == "sam3":
            human_runs.add(ref["artifact_id"])
            for result in _jsonl(stage / "outputs" / "results.jsonl"):
                human[result["source_observation_id"][-6:]] = [
                    [p.get("query") for p in region.get("discovery_provenance", [])]
                    for region in result["regions"]
                ]
            continue
        for result in _jsonl(stage / "outputs" / "results.jsonl"):
            frame = result["source_observation_id"][-6:]
            if frame in annotations:
                frames[frame] = perception_frame(stage, frame, annotations[frame], result)
    for ref in record["provided"]["sensor_association"]:
        stage = workspace / ref["location"]
        if set(_json(stage / "manifest.json").get("perception_run_ids", [])) & human_runs:
            continue
        for frame in annotations:
            measured = association_frame(stage, frame)
            # Um run por frame (ablação) ou vários frames por run (run longa): só o run que
            # associou o frame o descreve.
            if measured["observations"] or measured["skipped_regions"]:
                association[frame] = {**measured, **depth_outliers(stage, frame)}
    report = map_report(workspace, build.name)
    statuses = {a["annotator"]["id"] for a in annotations.values()}
    return {
        "map_id": map_id,
        "annotation_status": sorted({a["status"] for a in annotations.values()}),
        "annotators": sorted(statuses),
        "build_run": build.name,
        "frames": frames,
        "association": association,
        "entities": merges_and_splits(build, {f: frames[f]["matches"] for f in frames}),
        "human_search": human,
        "map": {
            "location": report["context_map"]["location"],
            "context_map_id": report["context_map"]["context_map_id"],
            "entity_count": report["context_map"]["entity_count"],
            "relation_count": report["context_map"]["relation_count"],
            "geometry_points": report["geometry"]["point_count"],
            "fusion": report["fusion"],
            "semantic_mapping": report["semantic_mapping"],
            "entity_resolution": report["entity_resolution"],
            "relations": report["spatial_relations"]["relations_by_predicate_and_state"],
        },
        "cost": cost(map_id, frames),
    }


def main() -> None:
    workspace = Path(sys.argv[1])
    print(json.dumps({m: arm(workspace, m) for m in sys.argv[2:]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
