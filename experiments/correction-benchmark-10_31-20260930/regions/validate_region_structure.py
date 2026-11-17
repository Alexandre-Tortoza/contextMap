"""C4: reproduz as métricas de máscara da investigação (p6) com `evaluation.region_structure`.

Entrada: as 83 regiões aceitas do E0 (frame 55, `workspace-old/frame-10_31-raw`), lidas pelas
APIs públicas. Referência: `semantic-geometry-investigation-20260930/results/p6_mask_quality.json`.
Oráculo de componentes conexos: `scipy.ndimage.label` (conectividade 4, o default em 2D), só
aqui no experimento; o módulo de produção usa apenas numpy.

Rodar (da raiz do repo):
    PYTHONPATH=src <python3.12 do mise> \
        experiments/correction-benchmark-10_31-20260930/regions/validate_region_structure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from e0_regions import INVESTIGATION, ROOT, code_identity, load_e0_regions

from contextmap.evaluation.region_structure import (
    MASK_CONNECTIVITY,
    connected_component_count,
    measure_region_structure,
)
from contextmap.visual_perception import InlineMask

OUT = Path(__file__).parent / "results"
CONTAINMENT = 0.8  # o mesmo de p6 ("≥80 % dentro de outra") e do C4 no plano
DUPLICATE_IOU_P6 = 0.5  # p6 contou pares com IoU ≥ 0,5
DUPLICATE_IOU_PLAN = 0.8  # limiar de duplicata do C5 no plano (e da normalização)
ORACLE_RANDOM_MASKS = 400


def oracle_components(masks: dict[str, InlineMask]) -> dict[str, object]:
    """Compara a contagem de componentes com o scipy em máscaras reais e aleatórias."""
    real_mismatch = [
        rid
        for rid, mask in masks.items()
        if connected_component_count(mask) != ndimage.label(mask.as_array())[1]
    ]
    rng = np.random.default_rng(0)
    random_mismatch = 0
    for trial in range(ORACLE_RANDOM_MASKS):
        density = 0.2 + 0.6 * (trial / ORACLE_RANDOM_MASKS)  # passa pelo limiar de percolação
        pixels = rng.random((48, 64)) < density
        if not pixels.any():
            continue
        if connected_component_count(InlineMask(pixels)) != ndimage.label(pixels)[1]:
            random_mismatch += 1
    return {
        "oracle": "scipy.ndimage.label (default 2D structure = 4-connectivity)",
        "real_masks": len(masks),
        "real_mismatches": real_mismatch,
        "random_masks": ORACLE_RANDOM_MASKS,
        "random_shape": [48, 64],
        "random_density_range": [0.2, 0.8],
        "random_mismatches": random_mismatch,
    }


def ordered_contained_pairs(masks: dict[str, InlineMask], threshold: float) -> dict[str, int]:
    """Reconcilia a contagem ordenada de p6 com a não ordenada do módulo."""
    ids = sorted(masks)
    stack = np.stack([masks[r].as_array().reshape(-1) for r in ids]).astype(np.float32)
    inter = stack @ stack.T
    area = stack.sum(1)
    inside = inter / area[:, None]
    np.fill_diagonal(inside, 0)
    ordered = int((inside >= threshold).sum())
    both = int(
        ((inside >= threshold) & (threshold <= inside.T))[np.triu_indices(len(ids), 1)].sum()
    )
    return {"ordered": ordered, "both_directions": both, "unordered": ordered - both}


def main() -> None:
    e0 = load_e0_regions()
    kwargs = {"proposal_count": e0.proposal_count, "usable_area": e0.usable_area}
    at_p6 = measure_region_structure(
        e0.masks,
        containment_threshold=CONTAINMENT,
        duplicate_iou_threshold=DUPLICATE_IOU_P6,
        **kwargs,
    )
    at_plan = measure_region_structure(
        e0.masks,
        containment_threshold=CONTAINMENT,
        duplicate_iou_threshold=DUPLICATE_IOU_PLAN,
        **kwargs,
    )
    p6 = json.loads((INVESTIGATION / "results/p6_mask_quality.json").read_text())
    ref = p6["frame"]
    rows = {m["mask_id"]: m for m in p6["masks"]}
    comps = [m.component_count for m in at_p6.masks]
    n = at_p6.mask_count
    hist = dict(at_p6.nesting_depth_histogram)
    ordered = ordered_contained_pairs(e0.masks, CONTAINMENT)

    def row(metric: str, reference: object, measured: object, note: str = "") -> dict[str, object]:
        same = (
            np.isclose(reference, measured, rtol=0, atol=1e-9)
            if isinstance(reference, float | int) and isinstance(measured, float | int)
            else reference == measured
        )
        return {
            "metric": metric,
            "p6": reference,
            "module": measured,
            "equal": bool(same),
            "note": note,
        }

    comparison = [
        row("masks", ref["masks"], n),
        row("median_area_px", ref["median_area_px"], at_p6.area_px_median),
        row("area_px_p10", ref["area_px_quantiles"][1], at_p6.area_px_p10),
        row("area_px_p90", ref["area_px_quantiles"][3], at_p6.area_px_p90),
        row(
            "fragmentation_rate",
            ref["fragmentation_rate_multi_component"],
            at_p6.fragmentation_rate,
        ),
        row(
            "masks_with_gt_10_components",
            ref["masks_with_gt_10_components"],
            sum(c > 10 for c in comps),
        ),
        row("median_components", ref["median_components"], float(np.median(comps))),
        row(
            "masks_contained_in_another_ge_0p8",
            ref["masks_mostly_inside_another_ge_0p8"],
            at_p6.masks_contained_in_another,
        ),
        row("pairs_iou_ge_0p5", ref["pairs_iou_ge_0p5"], at_p6.duplicate_pair_count),
        row(
            "overlap_rate_pairs_iou_gt_0",
            ref["overlap_rate_pairs_iou_gt_0"],
            at_p6.overlapping_pair_count / (n * (n - 1) / 2),
        ),
        row(
            "pairs_containment_ge_0p8",
            ref["pairs_containment_ge_0p8"],
            at_p6.contained_pair_count,
            f"p6 conta pares ORDENADOS (A em B e B em A): {ordered['ordered']} ordenados = "
            f"{ordered['unordered']} não ordenados + {ordered['both_directions']} pares com as duas "
            "direções ≥ 0,8; o módulo conta pares não ordenados",
        ),
        row(
            "pixel_nesting_depth_hist",
            {int(k): v for k, v in ref["pixel_nesting_depth_hist_over_covered"].items()},
            hist,
        ),
        row(
            "single_mask_pixel_fraction",
            ref["pixel_nesting_depth_hist_over_covered"]["1"]
            / sum(ref["pixel_nesting_depth_hist_over_covered"].values()),
            1 - at_p6.multiple_mask_pixel_fraction,
        ),
        row(
            "max_nesting_depth",
            max(int(k) for k in ref["pixel_nesting_depth_hist_over_covered"]),
            at_p6.max_nesting_depth,
        ),
        row(
            "usable_area_coverage",
            ref["usable_pixels_covered_fraction"],
            at_p6.usable_area_coverage,
        ),
    ]
    per_mask_mismatch, container_undefined = [], []
    for mask in at_p6.masks:
        ref_row = rows[mask.region_id]
        if (
            ref_row["area_px"] != mask.area_px
            or ref_row["connected_components"] != mask.component_count
            or abs(ref_row["max_containment_in_other"] - round(mask.max_fraction_inside_other, 4))
            > 1e-9
        ):
            per_mask_mismatch.append(mask.region_id)
        elif ref_row["contained_in"] != mask.most_containing_region_id:
            # Sem sobreposição, p6 grava o argmax de um vetor de zeros (a 1ª máscara); o módulo,
            # None. Só isso é aceito como diferença explicada.
            if (
                ref_row["max_containment_in_other"] == 0.0
                and mask.most_containing_region_id is None
            ):
                container_undefined.append(mask.region_id)
            else:
                per_mask_mismatch.append(mask.region_id)

    out = {
        "experiment": "C4 — validação das métricas estruturais de região contra p6",
        "provenance": {**e0.provenance, "code": code_identity(), "connectivity": MASK_CONNECTIVITY},
        "comparison": comparison,
        "per_mask_mismatches": per_mask_mismatch,
        "per_mask_container_undefined_in_module_argmax_of_zeros_in_p6": container_undefined,
        "component_oracle": oracle_components(e0.masks),
        "contained_pairs_reconciliation": ordered,
        "report_duplicate_iou_0p5": at_p6.to_dict(),
        "report_duplicate_iou_0p8": at_plan.to_dict(),
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "e0_region_structure_validation.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for item in comparison:
        status = "OK" if item["equal"] else "DIFF"
        print(
            f"{item['metric']:36s} p6={item['p6']!s:>32.32s} mod={item['module']!s:>32.32s} {status}"
        )
    print("per-mask mismatches:", per_mask_mismatch, "container undefined:", container_undefined)
    print("oracle:", out["component_oracle"])
    print(
        "accepted_mask_ratio:",
        at_p6.accepted_mask_ratio,
        "duplicates@0.8:",
        at_plan.duplicate_pair_count,
    )
    print("->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
