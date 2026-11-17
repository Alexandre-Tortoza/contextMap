"""Parte 7-8: proveniência de cada label, da resposta bruta do modelo até a entidade final.

Uma linha por claim: frame, máscara, label (hypothesis), category, attributes, region_kind,
role, resposta bruta do Qwen, similaridade CLIP (cosseno cru, não calibrado), modelo/revisão,
template/fingerprint do prompt, qualidade da máscara, e o caminho support -> entidade ->
entidade resolvida -> entidade do context_map. Também mede quantas labels embutem atributo
(cor/material) no texto e quanto a `category` já agruparia.
"""
from __future__ import annotations

import csv
import json
import re

from common import RESULTS, dump, load_jsonl, run_dir

COLORS = {"blue", "yellow", "white", "black", "red", "green", "gray", "grey", "brown", "dark", "metallic", "wooden"}
MATERIALS = {"metal", "wooden", "wood", "concrete", "glass", "plastic", "metallic"}


def main() -> None:
    arm = "raw"
    base = run_dir(arm)
    interp = {r["request"]["region_id"]: r for r in load_jsonl(base / "visual_perception/outputs/semantic-interpretations.jsonl")}
    failures = {r["request"]["region_id"]: r for r in load_jsonl(base / "visual_perception/outputs/semantic-interpretation-failures.jsonl")}
    scores = {s["claim_id"]: s for s in load_jsonl(base / "semantic_scoring/outputs/scores.jsonl")}
    masks = {m["mask_id"]: m for m in json.loads((RESULTS / "p6_mask_quality.json").read_text())["masks"]}
    result = json.loads((base / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    views = {r["request"]["region_id"]: r["request"]["visual_views"][0]["construction"]["pixel_bounds"] for r in interp.values()}

    # caminho região -> support -> entidade -> resolvida -> context_map
    support_of = {}
    for s in load_jsonl(base / "semantic_fusion/outputs/fusion-supports.jsonl"):
        for so in s["spatial_observation_ids"]:
            support_of[so.rsplit("--", 1)[1]] = s["fusion_support_id"]
    excluded = {e["spatial_observation_id"].rsplit("--", 1)[1]: e["geometry_count"]
                for e in load_jsonl(base / "semantic_fusion/outputs/excluded-observations.jsonl")}
    entity_of_support = {e["entity_id"].replace("entity--", ""): e["entity_id"]
                         for e in load_jsonl(base / "semantic_mapping/outputs/entities.jsonl")}
    resolved_of = {}
    for r in load_jsonl(base / "entity_resolution/outputs/resolved-entities.jsonl"):
        for m in r["members"]:
            resolved_of[m["entity_ref"]["entity_id"]] = r["resolved_entity_id"]
    cm_of = {}
    cm_labels = {}
    for line in load_jsonl(base / "context_map/entities/entities.jsonl"):
        rec = line["record"]
        cm_of[rec["source"]["resolved_entity_id"]] = rec["entity_id"]
        cm_labels[rec["entity_id"]] = ([h["label"] for h in rec["semantic_state"]["hypotheses"]], rec["semantic_state"]["status"])

    rows = []
    for region in result["regions"]:
        rid = region["region_id"]
        rec = interp.get(rid)
        claims = rec["parsed"]["claims"] if rec else []
        support = support_of.get(rid)
        entity = entity_of_support.get(support) if support else None
        resolved = resolved_of.get(entity) if entity else None
        cm = cm_of.get(resolved) if resolved else None
        bx = views.get(rid)
        bbox_frac = None if bx is None else round((bx[2] - bx[0]) * (bx[3] - bx[1]) / (640 * 480), 4)
        base_row = {
            "frame_id": region["source_observation_id"], "mask_id": rid,
            "mask_area_px": masks[rid]["area_px"], "mask_components": masks[rid]["connected_components"],
            "sam2_predicted_iou": masks[rid]["predicted_iou"], "sam2_stability": masks[rid]["stability"],
            "view_bbox_fraction_of_image": bbox_frac,
            "fusion_support": support, "excluded_from_fusion_geometry_count": excluded.get(rid),
            "semantic_entity": entity, "resolved_entity": resolved, "context_map_entity": cm,
            "context_map_hypotheses": cm_labels.get(cm, (None, None))[0] if cm else None,
            "context_map_status": cm_labels.get(cm, (None, None))[1] if cm else None,
        }
        if not claims:
            f = failures.get(rid)
            rows.append({**base_row, "label": None, "role": None, "category": None, "attributes": None,
                         "region_kind": None, "clip_cosine": None,
                         "raw_model_output": f.get("raw_response") if f else None,
                         "failure": f["parse_failure"]["message"] if f else "no request",
                         "model": "Qwen/Qwen3-VL-4B-Instruct", "model_revision": None,
                         "prompt_template": None, "prompt_fingerprint": None})
            continue
        for c in claims:
            s = scores.get(c["claim_id"])
            rows.append({**base_row, "label": c["hypothesis"], "role": c["role"], "category": c.get("category"),
                         "attributes": {a["name"]: a["value"] for a in c.get("attributes", [])},
                         "region_kind": c.get("region_kind"),
                         "clip_cosine": None if s is None else round(s["value"], 4),
                         "raw_model_output": rec["raw_response"], "failure": None,
                         "model": rec["effective_configuration"]["model"],
                         "model_revision": rec["effective_configuration"]["revision"],
                         "prompt_template": rec["rendered_prompt"]["template_id"],
                         "prompt_fingerprint": rec["rendered_prompt"]["fingerprint"]})

    labeled = [r for r in rows if r["label"]]
    words = lambda s: re.findall(r"[a-z]+", s.lower())
    with_attr_word = [r for r in labeled if len(words(r["label"])) > 1 and words(r["label"])[0] in COLORS | MATERIALS]
    attr_also_structured = [r for r in with_attr_word if r["attributes"] and any(
        str(v).lower() in r["label"].lower() for v in r["attributes"].values())]
    by_category = {}
    for r in labeled:
        by_category.setdefault(r["category"], set()).add(r["label"])
    summary = {
        "regions": len(result["regions"]), "claims": len(labeled), "parse_failures": sum(1 for r in rows if r["failure"]),
        "claims_per_region_max": max((sum(1 for x in labeled if x["mask_id"] == m) for m in {x["mask_id"] for x in labeled}), default=0),
        "roles": sorted({r["role"] for r in labeled}),
        "unique_labels_frame": len({r["label"] for r in labeled}),
        "unique_categories_frame": len({r["category"] for r in labeled}),
        "labels_with_color_or_material_prefix": len(with_attr_word),
        "of_which_same_value_also_in_attributes": len(attr_also_structured),
        "claims_with_structured_attributes": sum(1 for r in labeled if r["attributes"]),
        "labels_per_category": {str(k): sorted(v) for k, v in sorted(by_category.items(), key=lambda kv: str(kv[0]))},
        "region_kind_counts": {k: sum(1 for r in labeled if r["region_kind"] == k) for k in {r["region_kind"] for r in labeled}},
        "clip_cosine_range": [min(r["clip_cosine"] for r in labeled if r["clip_cosine"] is not None),
                              max(r["clip_cosine"] for r in labeled if r["clip_cosine"] is not None)],
    }
    dump("p7_label_provenance.json", {"summary": summary, "rows": rows})
    with open(RESULTS / "p7_label_provenance.csv", "w", newline="") as fh:
        cols = ["frame_id", "mask_id", "label", "role", "category", "attributes", "region_kind", "clip_cosine",
                "sam2_predicted_iou", "sam2_stability", "mask_area_px", "mask_components", "view_bbox_fraction_of_image",
                "fusion_support", "semantic_entity", "context_map_entity", "context_map_hypotheses", "context_map_status",
                "model", "model_revision", "prompt_template", "prompt_fingerprint", "failure", "raw_model_output"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
