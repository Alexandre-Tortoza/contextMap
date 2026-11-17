"""Parte 6: re-executa o SAM2 do run no frame 55 e exporta TODAS as propostas com scores.

Mesma configuração efetiva do run (sam2.1-hiera-tiny, revisão pinada, float32, cuda,
pred_iou_thresh=0.0, stability_score_thresh=0.0, demais parâmetros nos defaults do SAM2
AutomaticMaskGenerator). As 83 máscaras aceitas persistidas são casadas com as propostas por
IoU, o que atribui predicted_iou/stability a cada região (o artefato não guarda esses scores).
Roda no venv de ablação (sam2 + torch 2.8, HF offline).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import dump, load_rgb, run_dir

OUT = Path(os.environ["SCRATCH"]) / "sam2_frame55_proposals.npz"


def main() -> None:
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    torch.manual_seed(0)
    model = build_sam2_hf("facebook/sam2.1-hiera-tiny", device="cuda")
    generator = SAM2AutomaticMaskGenerator(model=model, pred_iou_thresh=0.0, stability_score_thresh=0.0,
                                           output_mode="binary_mask")
    rgb = load_rgb()
    with torch.inference_mode():
        records = generator.generate(rgb)
    masks = np.stack([r["segmentation"].astype(bool) for r in records])
    iou = np.array([r["predicted_iou"] for r in records])
    stab = np.array([r["stability_score"] for r in records])
    area = masks.reshape(len(masks), -1).sum(1)
    np.savez_compressed(OUT, masks=np.packbits(masks.reshape(len(masks), -1), axis=1), iou=iou, stability=stab, area=area)

    # casa as regiões aceitas persistidas com as propostas
    result = json.loads((run_dir("raw") / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    flat = masks.reshape(len(masks), -1)
    matches = []
    for region in result["regions"]:
        packed = np.load(run_dir("raw") / "visual_perception" / region["mask_reference"])
        m = np.unpackbits(packed, count=480 * 640).astype(bool)
        inter = (flat & m).sum(1)
        union = (flat | m).sum(1)
        j = inter / np.maximum(union, 1)
        best = int(np.argmax(j))
        contributor_idx = [int(c.rsplit("-", 1)[1]) for c in region["contributor_candidate_ids"]]
        matches.append({"region_id": region["region_id"], "best_proposal": best, "best_iou": float(j[best]),
                        "contributors": contributor_idx,
                        "representative_index_matches": best == contributor_idx[0] if contributor_idx else None,
                        "predicted_iou": float(iou[best]), "stability": float(stab[best]), "area_px": int(m.sum())})
    summary = {
        "proposal_count": len(records),
        "order_is_descending_predicted_iou": bool(np.all(np.diff(iou) <= 1e-6)),
        "predicted_iou_quantiles": [round(float(q), 4) for q in np.quantile(iou, [0, 0.1, 0.5, 0.9, 1])],
        "stability_quantiles": [round(float(q), 4) for q in np.quantile(stab, [0, 0.1, 0.5, 0.9, 1])],
        "would_pass_sam2_defaults_0p8_0p95": int(((iou >= 0.8) & (stab >= 0.95)).sum()),
        "accepted_regions_matched_iou_ge_0p99": int(sum(m["best_iou"] >= 0.99 for m in matches)),
        "accepted_regions_passing_sam2_defaults": int(sum(m["predicted_iou"] >= 0.8 and m["stability"] >= 0.95 for m in matches)),
        "accepted_regions": matches,
    }
    dump("p6_sam2_rerun.json", summary)
    print({k: v for k, v in summary.items() if k != "accepted_regions"})


if __name__ == "__main__":
    main()
