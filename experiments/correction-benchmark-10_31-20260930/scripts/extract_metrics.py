"""Extrator único de métricas do benchmark B1: um diretório de run -> ``metrics.json``.

Layout esperado (o de ``workspace-old/frame-10_31-raw``): ``<run>/<estágio>/`` com
``geometric_mapping``, ``visual_perception``, ``sensor_association``, ``semantic_fusion``,
``semantic_mapping``, ``entity_resolution``, ``spatial_relations`` e ``context_map``. Um braço
parcial (por exemplo, só ``visual_perception`` nos braços SAM2 do C4) mede só as camadas que
tem; as demais saem como ``{"status": "not_in_run"}`` e o gate não as compara.

Camadas: geometria, regiões, associação, semântica, entidades, relações e as métricas que
dependem da anotação fraca do frame 55 (``annotations/frame-55.json``). Sem anotação, cada
métrica dependente sai como ``"pending_annotation"`` (nunca como zero).

Regiões usam ``contextmap.evaluation.region_structure`` quando o módulo existe no momento da
execução; senão, uma portabilidade das definições de ``p6_mask_quality.py`` com os mesmos
nomes de campo. O ``metrics.json`` registra qual implementação mediu.

Léxico de cor/material: só de avaliação, copiado de ``p7_label_provenance.py`` (investigação).
Ele nunca entra no pipeline.

    PYTHONPATH=src $MISE_PY \
      experiments/correction-benchmark-10_31-20260930/scripts/extract_metrics.py \
      --run-dir workspace-old/frame-10_31-raw --arm-id E0 --out .../results/E0/metrics.json
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (
    ANNOTATIONS,
    CAMERA_CALIBRATION_ID,
    IMAGE_ID,
    IMAGE_SIZE,
    ROOT,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    write_json,
)
from common import GEOMETRY_DTYPE, mei_project, rigid, slerp
from p7_label_provenance import COLORS, MATERIALS

METRICS_SCHEMA = "correction-benchmark/metrics/v1"
PENDING = "pending_annotation"
CONTAINMENT_THRESHOLD = 0.8  # "mask_overlap (pares contidos >= 80%)", plano C4
DUPLICATE_IOU_THRESHOLD = 0.8  # mesmo limiar de duplicata da normalização (dup IoU 0,8)
RECALL_IOU_THRESHOLD = 0.5  # critério de seleção do C4: objeto anotado com IoU >= 0,5
SMALL_MASK_PX = 500
VOXEL_OCCUPANCY_CELL_M = 0.10
SCENE_LABEL_MAX_MASK_FRACTION = 0.5  # plano C6: regiões com máscara < 50% da imagem
HALLUCINATION_MIN_OVERLAP = 0.5  # plano C6: objeto fraco sobreposto a >= 50% da máscara
VALID_ASSET = ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-valid.npy"
EGO_ASSET = ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-ego.npy"
DEFAULT_ANNOTATION = ANNOTATIONS / "frame-55.json"


# --------------------------------------------------------------------------- utilidades


def _fraction(numerator: float, denominator: float) -> float | None:
    return None if not denominator else numerator / denominator


def _norm_label(label: str) -> str:
    """Casefold + espaços colapsados; não afirma sinonímia (como ``casefold-exact/1``)."""
    return " ".join(label.casefold().split())


def _entropy_bits(counts: list[int]) -> float | None:
    total = sum(counts)
    if not total:
        return None
    return -sum(c / total * math.log2(c / total) for c in counts if c)


def _has_color_or_material_prefix(label: str) -> bool:
    """Regra de p7: mais de uma palavra e a primeira está no léxico de avaliação."""
    words = re.findall(r"[a-z]+", label.lower())
    return len(words) > 1 and words[0] in COLORS | MATERIALS


def _usable_area() -> np.ndarray:
    return np.load(VALID_ASSET).astype(bool) & ~np.load(EGO_ASSET).astype(bool)


# --------------------------------------------------------------------------- geometria


def geometry_metrics(run: Path) -> dict[str, Any]:
    """sha256/map_id/contagens do mapa geométrico; ocupação de voxel declarada."""
    stage = run / "geometric_mapping"
    if not stage.is_dir():
        return {"status": "not_in_run"}
    manifest = read_json(stage / "manifest.json")
    config = read_json(stage / "config.json")
    geometry_bin = stage / "outputs/geometry.bin"
    digest = sha256_file(geometry_bin)
    recorded = next(
        (
            f["content_hash"]
            for f in manifest["file_inventory"]
            if f["path"] == "outputs/geometry.bin"
        ),
        None,
    )
    points = np.fromfile(geometry_bin, dtype=GEOMETRY_DTYPE)
    keys = np.floor(points["map"] / VOXEL_OCCUPANCY_CELL_M).astype(np.int64)
    occupancy = int(np.unique(keys, axis=0).shape[0])
    aggregation = config.get("aggregation")
    return {
        "status": "measured",
        "geometry_bin_sha256": digest,
        "geometry_bin_sha256_matches_manifest": digest == recorded,
        "geometry_bin_size_bytes": geometry_bin.stat().st_size,
        "map_id": manifest["map_id"],
        "run_id": manifest["run_id"],
        "point_count": int(manifest["point_count"]),
        "point_count_in_file": len(points),
        "source_point_count": int(manifest.get("source_point_count", manifest["point_count"])),
        "scan_count": int(manifest["scan_count"]),
        "rejected_scan_count": int(manifest["rejected_scan_count"]),
        "calibration_identity": manifest["calibration_identity"],
        "trajectory_id": manifest["trajectory_id"],
        "configuration_fingerprint": manifest["configuration_fingerprint"],
        "aggregation_rule": manifest.get("aggregation_rule"),
        # Com agregação voxel por scan, cada ponto do mapa É uma célula de voxel.
        "aggregation_voxel_count": int(manifest["point_count"]) if aggregation else None,
        "occupied_voxels_0p10m_global": occupancy,
        "motion_correction_policy": config.get("motion_correction_policy"),
    }


# --------------------------------------------------------------------------- regiões


def _load_region_masks(perception: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accepted-region masks (InlineMask) by region id, and the frame's result record."""
    from contextmap.ingestion import SourceObservationId
    from contextmap.visual_perception import RegionId
    from contextmap.visual_perception.artifacts.region_mask_store import MaskStoreReader

    result = next(
        r
        for r in read_jsonl(perception / "outputs/results.jsonl")
        if r["source_observation_id"] == IMAGE_ID
    )
    store = MaskStoreReader.open(perception / "outputs/masks")
    masks = {
        region["region_id"]: store.load(
            SourceObservationId(IMAGE_ID), RegionId(region["region_id"])
        )
        for region in result["regions"]
        if region["is_accepted"] and region.get("mask_reference")
    }
    return masks, result


def _structure_fallback(
    masks: dict[str, np.ndarray], usable: np.ndarray, proposals: int | None
) -> dict[str, Any]:
    """Port of ``p6_mask_quality.py`` definitions under the region_structure field names."""
    from scipy import ndimage

    ids = sorted(masks)
    stack = np.stack([masks[r].reshape(-1) for r in ids]).astype(np.float32)
    area = stack.sum(1)
    inter = stack @ stack.T
    union = area[:, None] + area[None, :] - inter
    upper = np.triu_indices(len(ids), 1)
    inside = inter / np.maximum(area[:, None], 1)
    np.fill_diagonal(inside, 0)
    components = [int(ndimage.label(masks[r])[1]) for r in ids]
    depth = stack.sum(0).astype(np.int64)
    covered = depth > 0
    hist = np.bincount(depth[covered])
    pair_min = np.minimum(area[upper[0]], area[upper[1]])
    return {
        "metric_version": "fallback-p6-port",
        "mask_count": len(ids),
        "proposal_count": proposals,
        "accepted_mask_ratio": _fraction(len(ids), proposals) if proposals else None,
        "area_px_p10": float(np.quantile(area, 0.1)),
        "area_px_median": float(np.median(area)),
        "area_px_p90": float(np.quantile(area, 0.9)),
        "fragmented_mask_count": sum(c > 1 for c in components),
        "fragmentation_rate": _fraction(sum(c > 1 for c in components), len(ids)),
        "masks_contained_in_another": int((inside.max(1) >= CONTAINMENT_THRESHOLD).sum()),
        "overlapping_pair_count": int((inter[upper] > 0).sum()),
        "contained_pair_count": int((inter[upper] / pair_min >= CONTAINMENT_THRESHOLD).sum()),
        "duplicate_pair_count": int((inter[upper] / union[upper] >= DUPLICATE_IOU_THRESHOLD).sum()),
        "covered_pixel_count": int(covered.sum()),
        "pixels_with_multiple_masks": int((depth >= 2).sum()),
        "multiple_mask_pixel_fraction": _fraction(int((depth >= 2).sum()), int(covered.sum())),
        "nesting_depth_histogram": {str(k): int(v) for k, v in enumerate(hist) if k and v},
        "max_nesting_depth": int(depth.max()),
        "usable_pixel_count": int(usable.sum()),
        "usable_area_coverage": float(covered[usable.reshape(-1)].mean()),
        "masks": [
            {"region_id": r, "area_px": int(area[i]), "component_count": components[i]}
            for i, r in enumerate(ids)
        ],
    }


def region_metrics(
    run: Path, annotation: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Structure of the accepted masks + the discovery audit; recall only with annotation."""
    perception = run / "visual_perception"
    if not perception.is_dir():
        return {"status": "not_in_run"}, {}
    masks, result = _load_region_masks(perception)
    audit = next(
        (
            a
            for a in read_jsonl(perception / "outputs/region-discovery-audit.jsonl")
            if a["source_observation_id"] == IMAGE_ID
        ),
        None,
    )
    raw_proposals = None
    filtered = None
    if audit is not None:
        meta = {
            m["name"]: m["value"] for p in audit["passes"] for m in p["diagnostics"]["metadata"]
        }
        raw_proposals = meta.get("raw_proposal_count")
        filtered = meta.get("filtered_proposal_count")
    # proposal_count = propostas que chegaram à normalização (saída do SAM2 depois do filtro de
    # limiar do backend), a população da qual as regiões aceitas foram escolhidas.
    candidates = None if raw_proposals is None else raw_proposals - (filtered or 0)
    usable = _usable_area()
    implementation = "contextmap.evaluation.region_structure"
    try:
        from contextmap.evaluation import region_structure as rs
        from contextmap.visual_perception import InlineMask

        report = rs.measure_region_structure(
            masks,
            containment_threshold=CONTAINMENT_THRESHOLD,
            duplicate_iou_threshold=DUPLICATE_IOU_THRESHOLD,
            proposal_count=candidates if candidates else None,
            usable_area=InlineMask(usable),
        ).to_dict()
        implementation += f" {report['metric_version']}"
    except ImportError:
        implementation = "fallback: port of p6_mask_quality.py"
        report = _structure_fallback(
            {r: m.as_array() for r, m in masks.items()}, usable, candidates if candidates else None
        )
    per_mask = report.pop("masks")
    areas = [m["area_px"] for m in per_mask]
    components = [m["component_count"] for m in per_mask]
    metrics: dict[str, Any] = {
        "status": "measured",
        "implementation": implementation,
        "definitions": {
            "containment_threshold": CONTAINMENT_THRESHOLD,
            "duplicate_iou_threshold": DUPLICATE_IOU_THRESHOLD,
            "proposal_count": "SAM2 raw proposals minus those filtered by the backend thresholds",
            "usable_area": "camera_1-valid.npy AND NOT camera_1-ego.npy",
        },
        "region_count_total": len(result["regions"]),
        "accepted_region_count": sum(1 for r in result["regions"] if r["is_accepted"]),
        "masks_measured": len(masks),
        **report,
        "masks_below_500px": sum(a < SMALL_MASK_PX for a in areas),
        "masks_with_gt_10_components": sum(c > 10 for c in components),
        "median_components": float(np.median(components)) if components else None,
        "sam2_raw_proposal_count": raw_proposals,
        "sam2_filtered_by_threshold_count": filtered,
        "discovery_backend": audit["backend"] if audit else None,
        "discovery_identity": audit["discovery_identity"] if audit else None,
    }
    if audit is not None:
        metrics["pass_rejections_by_reason"] = dict(
            Counter(r["reason"] for r in audit["pass_rejections"])
        )
        metrics["normalization_rejections_by_reason"] = dict(
            Counter(r["reason"] for r in audit["normalization_rejections"])
        )
        metrics["merge_decisions_by_kind"] = dict(
            Counter(m["kind"] for m in audit["merge_decisions"])
        )
    metrics["mask_sha256_by_region"] = {
        row["region_id"]: row["content_hash"]
        for row in read_jsonl(perception / "outputs/masks/mask-index.jsonl")
        if "region_id" in row
    }
    return metrics, masks


def annotated_recall(masks: dict[str, Any], annotation: dict[str, Any] | None) -> Any:
    """Recall of annotated instances (best IoU >= 0.5), via region_structure when present."""
    if annotation is None:
        return PENDING
    from contextmap.evaluation.annotations import RegionAnnotation
    from contextmap.visual_perception import BoundingBox, InlineMask

    objects = []
    for inst in annotation["instances"]:
        mask = _polygon_mask(inst.get("polygon_xy"))
        box = inst.get("box_xyxy")
        objects.append(
            RegionAnnotation(
                region_id=inst["instance_id"],
                mask=None if mask is None else InlineMask(mask),
                box=None if box is None else BoundingBox(*map(float, box)),
            )
        )
    try:
        from contextmap.evaluation.region_structure import known_object_recall
    except ImportError:
        return {"status": "region_structure module unavailable"}
    report = known_object_recall(masks, objects, match_iou_threshold=RECALL_IOU_THRESHOLD).to_dict()
    classes = {i["instance_id"]: i["class"] for i in annotation["instances"]}
    for match in report["matches"]:
        match["class"] = classes[match["object_id"]]
    return report


def _polygon_mask(polygon: list[list[float]] | None) -> np.ndarray | None:
    if not polygon:
        return None
    from PIL import Image, ImageDraw

    canvas = Image.new("1", IMAGE_SIZE, 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon], fill=1, outline=1)
    return np.array(canvas, dtype=bool)


# --------------------------------------------------------------------------- associação


def association_metrics(run: Path, point_count: int | None) -> dict[str, Any]:
    """Counts per stage + an independent recount of multi-mask membership from the support."""
    stage = run / "sensor_association"
    if not stage.is_dir():
        return {"status": "not_in_run"}
    summary = read_json(stage / "metrics/summary.json")
    diag = next(
        d
        for d in read_jsonl(stage / "metrics/frame-diagnostics.jsonl")
        if d["source_observation_id"] == IMAGE_ID
    )
    membership = diag["membership"]
    support = np.fromfile(stage / "outputs/geometry-support.u32", dtype="<u4").astype(np.int64)
    observations = read_jsonl(stage / "outputs/spatial-observations.jsonl")
    per_point: Counter[int] = Counter()
    for obs in observations:
        offset, count = obs["support"]["offset"], obs["support"]["count"]
        per_point.update(np.unique(support[offset : offset + count]).tolist())
    associated = len(per_point)
    multi = sum(1 for c in per_point.values() if c >= 2)
    total = point_count if point_count is not None else membership["point_count"]
    return {
        "status": "measured",
        "calibration_identity": diag["calibration_ref"]["calibration_identity"],
        "camera_calibration_id": diag["calibration_ref"]["camera_calibration_id"],
        "map_id": diag["map"]["map_id"],
        "pose_ref": diag["pose_ref"],
        "visibility_policy": diag["visibility_policy"],
        "state_counts": summary["state_counts"],
        "projected_in_domain_count": total - summary["state_counts"]["behind_camera"],
        "in_image_count": total
        - summary["state_counts"]["behind_camera"]
        - summary["state_counts"]["outside_image"],
        "visible_count": membership["visible_count"],
        "associated_count": membership["associated_count"],
        "visible_unassigned_count": membership["visible_unassigned_count"],
        "membership_pair_count": membership["membership_pair_count"],
        "overlap_point_count": membership["overlap_point_count"],
        "points_by_region_count": membership["points_by_region_count"],
        "multi_mask_point_fraction": _fraction(
            membership["overlap_point_count"], membership["associated_count"]
        ),
        "uncovered_point_fraction": 1 - membership["associated_count"] / total,
        "observation_count": summary["observation_count"],
        "observations_without_support": summary["observations_without_support"],
        "recount": {
            "associated_unique_points": associated,
            "points_in_2plus_observations": multi,
            "max_observations_per_point": max(per_point.values(), default=0),
            "agrees_with_diagnostics": associated == membership["associated_count"]
            and multi == membership["overlap_point_count"],
        },
    }


# --------------------------------------------------------------------------- semântica


def semantic_metrics(run: Path) -> dict[str, Any]:
    """Claims, labels, modifiers (evaluation-only lexicon), entropy and parse failures."""
    perception = run / "visual_perception"
    if not perception.is_dir():
        return {"status": "not_in_run"}
    result = next(
        r
        for r in read_jsonl(perception / "outputs/results.jsonl")
        if r["source_observation_id"] == IMAGE_ID
    )
    claims = result["claims"]
    if not claims and not (perception / "outputs/semantic-interpretations.jsonl").is_file():
        return {"status": "not_in_run"}
    interpretations = read_jsonl(perception / "outputs/semantic-interpretations.jsonl")
    failures = read_jsonl(perception / "outputs/semantic-interpretation-failures.jsonl")
    call_failures = read_jsonl(perception / "outputs/semantic-backend-call-failures.jsonl")
    primary = [c for c in claims if c.get("role") == "primary"]
    labels = [_norm_label(c["hypothesis"]) for c in primary]
    counts = Counter(labels)
    with_prefix = [c for c in primary if _has_color_or_material_prefix(c["hypothesis"])]
    prefix_also_structured = [
        c
        for c in with_prefix
        if any(str(a["value"]).lower() in c["hypothesis"].lower() for a in c.get("attributes", []))
    ]
    accepted = {r["region_id"] for r in result["regions"] if r["is_accepted"]}
    regions_with_claims = {c["region_id"] for c in claims if c.get("region_id")}
    model = None
    if interpretations:
        cfg = interpretations[0]["effective_configuration"]
        model = {
            "model": cfg["model"],
            "revision": cfg["revision"],
            "quantization": cfg.get("quantization"),
            "temperature": cfg.get("temperature"),
            "prompt_template_id": interpretations[0]["rendered_prompt"]["template_id"],
            # Um prompt renderizado por região; o digest da lista ordenada permite comparar
            # braços (mesmo prompt por região) sem guardar 77 fingerprints.
            "rendered_prompts_digest": sha256_bytes(
                "\n".join(
                    sorted(
                        f"{i['request']['region_id']}={i['rendered_prompt']['fingerprint']}"
                        for i in interpretations + failures
                    )
                ).encode()
            ),
            "visual_views_digest": sha256_bytes(
                "\n".join(
                    sorted(
                        f"{v['view_id']}={v['sha256']}"
                        for i in interpretations + failures
                        for v in i["request"]["visual_views"]
                    )
                ).encode()
            ),
            "configuration_fingerprint": interpretations[0]["request"]["configuration_fingerprint"],
            "view_kinds": sorted(
                {v["kind"] for i in interpretations for v in i["request"]["visual_views"]}
            ),
        }
    return {
        "status": "measured",
        "model": model,
        "claim_count": len(claims),
        "primary_claim_count": len(primary),
        "alternative_claim_count": sum(1 for c in claims if c.get("role") == "alternative"),
        "regions_with_claims": len(regions_with_claims & accepted),
        "accepted_regions_without_claims": len(accepted - regions_with_claims),
        "parse_failures": len(failures),
        "backend_call_failures": len(call_failures),
        "unique_labels": len(counts),
        "label_entropy_bits": _entropy_bits(list(counts.values())),
        "label_entropy_max_bits": math.log2(len(counts)) if counts else None,
        "labels_with_color_or_material_prefix": len(with_prefix),
        "color_or_material_prefix_fraction": _fraction(len(with_prefix), len(primary)),
        "prefix_value_also_in_attributes": len(prefix_also_structured),
        "claims_with_structured_attributes": sum(1 for c in primary if c.get("attributes")),
        "unique_categories": len({c.get("category") for c in primary}),
        "region_kind_counts": dict(Counter(str(c.get("region_kind")) for c in primary)),
        "label_counts": dict(sorted(counts.items())),
        "lexicon": {
            "source": "experiments/semantic-geometry-investigation-20260930/scripts/"
            "p7_label_provenance.py",
            "rule": "label has >1 word and its first word is in COLORS | MATERIALS "
            "(evaluation only)",
            "colors": sorted(COLORS),
            "materials": sorted(MATERIALS),
        },
        "primary_by_region": {
            c["region_id"]: {
                "label": c["hypothesis"],
                "category": c.get("category"),
                "attributes": {a["name"]: a["value"] for a in c.get("attributes", [])},
            }
            for c in primary
            if c.get("region_id")
        },
    }


# --------------------------------------------------------------------------- entidades


def entity_metrics(run: Path) -> dict[str, Any]:
    """Supports, semantic entities, resolved entities and context-map entities."""
    if not (run / "semantic_mapping").is_dir():
        return {"status": "not_in_run"}
    fusion_counts = read_json(run / "semantic_fusion/metrics/counts.json")
    supports = read_jsonl(run / "semantic_fusion/outputs/fusion-supports.jsonl")
    fused = read_jsonl(run / "semantic_fusion/outputs/fused-evidence.jsonl")
    sm_entities = read_jsonl(run / "semantic_mapping/outputs/entities.jsonl")
    physical = {e["entity_id"]: set(e["evidence"]["physical_observation_ids"]) for e in sm_entities}
    states = Counter(e["semantic_state"]["ambiguity_state"] for e in sm_entities)
    with_primary = sum(1 for e in sm_entities if e["semantic_state"].get("primary_hypothesis"))
    er_counts = read_json(run / "entity_resolution/metrics/counts.json")["counts"]
    context: list[dict[str, Any]] = []
    entity_points: set[str] = set()
    cm_path = run / "context_map/entities/entities.jsonl"
    for row in read_jsonl(cm_path):
        record = row["record"]
        members = [m["entity_id"] for m in record["member_entities"]]
        entity_points.update(ref["geometry_id"] for ref in record["geometry_refs"])
        observations = set().union(*(physical.get(m, set()) for m in members)) if members else set()
        context.append(
            {
                "status": record["semantic_state"]["status"],
                "labels": [h["label"] for h in record["semantic_state"]["hypotheses"]],
                "members": len(members),
                "physical_observations": len(observations),
                "points": len(record["geometry_refs"]),
            }
        )
    obs_counts = [c["physical_observations"] for c in context]
    labels_per_entity = [len(c["labels"]) for c in context]
    return {
        "status": "measured",
        "fusion_supports": len(supports),
        "fusion_excluded_observations": fusion_counts["excluded_observations"],
        "fusion_physical_observations": fusion_counts["physical_observations"],
        "supports_with_multiple_labels": sum(
            1 for f in fused if len({_norm_label(h["label"]) for h in f.get("hypotheses", [])}) > 1
        ),
        "semantic_entities": len(sm_entities),
        "semantic_entities_with_primary": with_primary,
        "semantic_state_counts": dict(states),
        "resolved_entities": er_counts["resolved_entities"],
        "merged_entities": er_counts["merged_entities"],
        "resolution_decisions": {
            k.removeprefix("decision_"): v
            for k, v in er_counts.items()
            if k.startswith("decision_")
        },
        "channels_measured": sorted(
            k.removeprefix("channel_").removesuffix("_measured")
            for k, v in er_counts.items()
            if k.startswith("channel_") and k.endswith("_measured") and v
        ),
        "context_map_entities": len(context),
        "context_map_status_counts": dict(Counter(c["status"] for c in context)),
        # Primária no mapa público = status "unambiguous" (o artefato não carrega outro campo).
        "context_map_entities_with_single_label": sum(
            1 for c in context if c["status"] == "unambiguous"
        ),
        "observations_per_entity_mean": float(np.mean(obs_counts)) if obs_counts else None,
        "observations_per_entity_max": max(obs_counts, default=0),
        "isolated_entity_rate": _fraction(sum(1 for c in obs_counts if c <= 1), len(obs_counts)),
        "labels_per_entity_mean": float(np.mean(labels_per_entity)) if labels_per_entity else None,
        "labels_per_entity_max": max(labels_per_entity, default=0),
        "unique_labels_in_map": len({_norm_label(lbl) for c in context for lbl in c["labels"]}),
        # Uma entidade pode compartilhar pontos com outra (supports vêm de máscaras sobrepostas):
        # a soma conta referências; os únicos são os pontos do mapa cobertos por alguma entidade.
        "entity_point_references": sum(c["points"] for c in context),
        "unique_points_in_entities": len(entity_points),
    }


def relation_metrics(run: Path) -> dict[str, Any]:
    """Relation counts by state (context only; no C-arm targets relations yet)."""
    stage = run / "spatial_relations"
    if not stage.is_dir():
        return {"status": "not_in_run"}
    counts = read_json(stage / "metrics/counts.json")
    return {
        "status": "measured",
        "relations": counts["relations"],
        "by_state": counts["by_state"],
        "entities_with_relations": counts["entities_with_relations"],
    }


# --------------------------------------------------------------------------- anotação


def calibration_metrics(
    run: Path, annotation: dict[str, Any] | None, geometry: dict[str, Any]
) -> dict[str, Any]:
    """Reprojection residual on the annotated 2D<->3D correspondences (pilot, weak)."""
    if annotation is None:
        return {"status": PENDING, "reprojection_residual_px": PENDING}
    correspondences = annotation.get("correspondences", [])
    if not correspondences:
        return {"status": "no correspondences annotated"}
    sequence = next((run / "sequences/corridor-02").iterdir())
    calibration = read_json(sequence / "calibration/calibration.json")
    camera = next(
        e for e in calibration["entries"] if e["calibration_id"] == CAMERA_CALIBRATION_ID
    )["camera_model"]
    extrinsic = next(
        s
        for s in calibration["static_transforms"]
        if s["parent_frame"] == "epson" and s["child_frame"] == "d"
    )
    t_epson_d = rigid(extrinsic["translation"], extrinsic["rotation"])
    t_map_epson = _pose_at(run, _image_time_ns(sequence))
    t_cam_map = np.linalg.inv(t_map_epson @ t_epson_d)
    points = np.fromfile(run / "geometric_mapping/outputs/geometry.bin", dtype=GEOMETRY_DTYPE)
    residuals = []
    for c in correspondences:
        if c.get("geometry_index") is not None:
            if c.get("map_id") not in (None, geometry.get("map_id")):
                residuals.append(
                    {"correspondence_id": c["correspondence_id"], "status": "map_id mismatch"}
                )
                continue
            xyz = points["map"][int(c["geometry_index"])]
        else:
            xyz = np.asarray(c["point_map_m"], float)
        cam = xyz @ t_cam_map[:3, :3].T + t_cam_map[:3, 3]
        uv, valid = mei_project(cam[None, :], camera)
        if not valid[0]:
            residuals.append(
                {"correspondence_id": c["correspondence_id"], "status": "outside projection domain"}
            )
            continue
        error = float(np.linalg.norm(uv[0] - np.asarray(c["pixel_uv"], float)))
        residuals.append({"correspondence_id": c["correspondence_id"], "residual_px": error})
    values = [r["residual_px"] for r in residuals if "residual_px" in r]
    return {
        "status": "measured",
        "reprojection_residual_px": {
            "count": len(values),
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, 0.9)) if values else None,
            "per_correspondence": residuals,
            "calibration_identity": geometry.get("calibration_identity"),
        },
    }


def _image_time_ns(sequence: Path) -> int:
    entry = next(e for e in read_jsonl(sequence / "index.jsonl") if e["observation_id"] == IMAGE_ID)
    return int(entry["timestamp"]["seconds"]) * 1_000_000_000 + int(
        entry["timestamp"]["nanoseconds"]
    )


def _pose_at(run: Path, t_ns: int) -> np.ndarray:
    """T_map_epson(t) por lerp+slerp entre as poses válidas do state_estimation do run."""
    rows = [
        r
        for r in read_jsonl(run / "state_estimation/outputs/poses.jsonl")
        if r.get("validity") == "valid"
    ]
    times = np.array(
        [
            int(r["timestamp"]["seconds"]) * 1_000_000_000 + int(r["timestamp"]["nanoseconds"])
            for r in rows
        ]
    )
    order = np.argsort(times)
    times = times[order]
    rows = [rows[i] for i in order]
    i = int(np.searchsorted(times, t_ns))
    if i < len(times) and times[i] == t_ns:
        return rigid(rows[i]["translation_m"], rows[i]["orientation_xyzw"])
    a, b = rows[i - 1], rows[i]
    f = (t_ns - times[i - 1]) / (times[i] - times[i - 1])
    p = np.asarray(a["translation_m"]) + f * (
        np.asarray(b["translation_m"]) - np.asarray(a["translation_m"])
    )
    q = slerp(a["orientation_xyzw"], b["orientation_xyzw"], f)
    return rigid(p, q)


def semantic_annotation_metrics(
    semantics: dict[str, Any], masks: dict[str, Any], annotation: dict[str, Any] | None
) -> dict[str, Any]:
    """Scene-label rate and hallucination rate (plan C6 definitions), pilot and weak."""
    if annotation is None or semantics.get("status") != "measured":
        return {"scene_label_rate": PENDING, "hallucination_rate": PENDING}
    scene_terms = {_norm_label(t) for t in annotation.get("scene_terms", [])}
    instances = []
    for inst in annotation["instances"]:
        pixels = _polygon_mask(inst.get("polygon_xy"))
        if pixels is None and inst.get("box_xyxy"):
            x0, y0, x1, y1 = (round(v) for v in inst["box_xyxy"])
            pixels = np.zeros(IMAGE_SIZE[::-1], dtype=bool)
            pixels[y0:y1, x0:x1] = True
        accepted = {_norm_label(inst["class"].replace("_", " "))} | {
            _norm_label(v) for v in inst.get("accepted_labels", [])
        }
        instances.append((pixels, accepted))
    image_area = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    scene_hits = scene_population = halluc = halluc_population = 0
    for region_id, claim in semantics["primary_by_region"].items():
        if region_id not in masks:
            continue
        mask = masks[region_id].as_array()
        label = _norm_label(claim["label"])
        if mask.sum() / image_area < SCENE_LABEL_MAX_MASK_FRACTION:
            scene_population += 1
            scene_hits += label in scene_terms
        # A região só é julgada quando algum objeto fraco cobre >= 50% da máscara; fora disso
        # a anotação parcial não diz nada (ausência não é verdade negativa).
        overlapping = [
            acc
            for pixels, acc in instances
            if (mask & pixels).sum() / mask.sum() >= HALLUCINATION_MIN_OVERLAP
        ]
        if overlapping:
            halluc_population += 1
            halluc += not any(label in acc for acc in overlapping)
    return {
        "scene_label_rate": {
            "count": scene_hits,
            "population": scene_population,
            "rate": _fraction(scene_hits, scene_population),
        },
        "hallucination_rate": {
            "count": halluc,
            "population": halluc_population,
            "rate": _fraction(halluc, halluc_population),
        },
        "matching_rule": "casefold-exact against the instance class (underscore -> space) "
        "or its accepted_labels",
    }


# --------------------------------------------------------------------------- orquestração


def _shown(path: Path) -> str:
    """Repository-relative path when inside the repository, absolute otherwise."""
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def extract(
    run_dir: Path, *, arm_id: str, annotation_path: Path | None, notes: str = ""
) -> dict[str, Any]:
    """Measure every layer present in ``run_dir``."""
    annotation = None
    if annotation_path is not None and annotation_path.is_file():
        annotation = read_json(annotation_path)
        if annotation.get("status") != "annotated":
            annotation = None
    geometry = geometry_metrics(run_dir)
    regions, masks = region_metrics(run_dir, annotation)
    semantics = semantic_metrics(run_dir)
    point_count = geometry.get("point_count") if geometry.get("status") == "measured" else None
    return {
        "schema": METRICS_SCHEMA,
        "arm_id": arm_id,
        "run_dir": _shown(run_dir),
        "benchmark": "B1",
        "image_id": IMAGE_ID,
        "notes": notes,
        "annotation": {
            "path": None if annotation_path is None else _shown(annotation_path),
            "status": "used" if annotation is not None else PENDING,
            "sha256": sha256_file(annotation_path)
            if annotation is not None and annotation_path
            else None,
        },
        "layers": {
            "geometry": geometry,
            "calibration": calibration_metrics(run_dir, annotation, geometry)
            if geometry.get("status") == "measured"
            else {"status": "not_in_run"},
            "regions": regions
            | ({"known_object_recall": annotated_recall(masks, annotation)} if masks else {}),
            "association": association_metrics(run_dir, point_count),
            "semantics": semantics | semantic_annotation_metrics(semantics, masks, annotation)
            if semantics.get("status") == "measured"
            else semantics,
            "entities": entity_metrics(run_dir),
            "relations": relation_metrics(run_dir),
        },
    }


def main() -> int:
    """CLI: extract the metrics of one run directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--arm-manifest",
        type=Path,
        default=None,
        help="arm.json do braço (declara, p.ex., changes_geometry/changes_calibration)",
    )
    args = parser.parse_args()
    metrics = extract(
        args.run_dir, arm_id=args.arm_id, annotation_path=args.annotation, notes=args.notes
    )
    if args.arm_manifest is not None:
        metrics["arm"] = read_json(args.arm_manifest)
    write_json(args.out, metrics)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
