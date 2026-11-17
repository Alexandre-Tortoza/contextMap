"""Parte 6: qualidade regional das 83 máscaras aceitas do frame 55, sem usar rótulo.

Por máscara: área, razão de área (sobre a área útil), bbox, componentes conexos, porosidade
(área / área com buracos preenchidos), IoU predito e estabilidade do SAM2 (do re-run casado),
e sobreposição com as demais (IoU máx., contenção máx. = |A∩B|/|A|). Por frame: máscaras,
mediana de área, taxa de sobreposição, fragmentação, profundidade de aninhamento por pixel.
"""
from __future__ import annotations

import json

import numpy as np
from assoc import load_frame
from common import RESULTS, dump, run_dir
from scipy import ndimage


def main() -> None:
    frame = load_frame("raw")
    sam = {m["region_id"]: m for m in json.loads((RESULTS / "p6_sam2_rerun.json").read_text())["accepted_regions"]}
    result = json.loads((run_dir("raw") / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    claims = {}
    for c in result["claims"]:
        claims.setdefault(c["region_id"], []).append(c)
    usable = (frame.valid & ~frame.exclusion).sum()
    ids = sorted(frame.masks)
    stack = np.stack([frame.masks[r].reshape(-1) for r in ids]).astype(np.uint8)
    area = stack.sum(1).astype(float)
    inter = stack.astype(np.float32) @ stack.T.astype(np.float32)
    union = area[:, None] + area[None, :] - inter
    iou = inter / np.maximum(union, 1)
    contain = inter / np.maximum(area[:, None], 1)  # fração de A dentro de B
    np.fill_diagonal(iou, 0)
    np.fill_diagonal(contain, 0)
    rows = []
    for i, rid in enumerate(ids):
        m = frame.masks[rid]
        lab, ncomp = ndimage.label(m)
        filled = ndimage.binary_fill_holes(m).sum()
        comp_sizes = np.bincount(lab.reshape(-1))[1:]
        ys, xs = np.nonzero(m)
        cl = claims.get(rid, [])
        primary = next((c for c in cl if c.get("role") == "primary"), None)
        rows.append({
            "mask_id": rid, "area_px": int(area[i]), "area_ratio": round(float(area[i] / usable), 5),
            "bbox_xywh": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "predicted_iou": round(sam[rid]["predicted_iou"], 4), "stability": round(sam[rid]["stability"], 4),
            "connected_components": int(ncomp),
            "largest_component_fraction": round(float(comp_sizes.max() / area[i]), 4),
            "porosity": round(float(1 - area[i] / filled), 4),
            "max_iou_other": round(float(iou[i].max()), 4), "max_iou_with": ids[int(iou[i].argmax())],
            "max_containment_in_other": round(float(contain[i].max()), 4), "contained_in": ids[int(contain[i].argmax())],
            "overlapping_masks": int((inter[i] > 0).sum() - 1),
            "label": primary["hypothesis"] if primary else None,
            "category": primary.get("category") if primary else None,
        })
    depth = stack.sum(0).reshape(480, 640)
    covered = depth > 0
    frame_metrics = {
        "masks": len(ids),
        "median_area_px": float(np.median(area)),
        "area_px_quantiles": [float(q) for q in np.quantile(area, [0, 0.1, 0.5, 0.9, 1])],
        "masks_below_500px": int((area < 500).sum()),
        "masks_with_pred_iou_below_0p5": int(sum(r["predicted_iou"] < 0.5 for r in rows)),
        "masks_with_stability_below_0p8": int(sum(r["stability"] < 0.8 for r in rows)),
        "overlap_rate_pairs_iou_gt_0": float((inter[np.triu_indices(len(ids), 1)] > 0).mean()),
        "pairs_iou_ge_0p5": int((iou[np.triu_indices(len(ids), 1)] >= 0.5).sum()),
        "pairs_containment_ge_0p8": int((contain >= 0.8).sum()),
        "masks_mostly_inside_another_ge_0p8": int(sum(r["max_containment_in_other"] >= 0.8 for r in rows)),
        "fragmentation_rate_multi_component": float(np.mean([r["connected_components"] > 1 for r in rows])),
        "masks_with_gt_10_components": int(sum(r["connected_components"] > 10 for r in rows)),
        "median_components": float(np.median([r["connected_components"] for r in rows])),
        "pixel_nesting_depth_hist_over_covered": {str(k): int(v) for k, v in enumerate(np.bincount(depth[covered])) if v},
        "usable_pixels_covered_fraction": float(covered[frame.valid & ~frame.exclusion].mean()),
        "pearson_pred_iou_vs_components": float(np.corrcoef([r["predicted_iou"] for r in rows], [np.log1p(r["connected_components"]) for r in rows])[0, 1]),
    }
    dump("p6_mask_quality.json", {"frame": frame_metrics, "masks": rows})
    print(json.dumps(frame_metrics, indent=1))
    for r in sorted(rows, key=lambda r: -r["area_px"])[:25]:
        print(r["mask_id"], r["label"], r["area_px"], "iou", r["predicted_iou"], "stab", r["stability"], "cc", r["connected_components"],
              "por", r["porosity"], "maxIoU", r["max_iou_other"], r["max_iou_with"], "inside", r["max_containment_in_other"], r["contained_in"])


if __name__ == "__main__":
    main()
