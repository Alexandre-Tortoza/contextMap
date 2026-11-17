"""C5: reproduz multi-pertencimento, contaminação por camada e oclusão próxima de p5.

Entradas: pertença persistida do E0 (API pública), máscaras do E0 (API pública) e, por ponto,
alcance do raio / profundidade de suporte / oclusão da reimplementação validada (`assoc.py`).
Referências: `p5_association_ablation.json` (config `accumulated_map_81_scans|base|pipeline`)
e `p5_occlusion_anatomy.json`. Também grava a primeira taxonomia de conflitos do E0.

Rodar (da raiz do repo):
    PYTHONPATH=src <python3.12 do mise> \
        experiments/correction-benchmark-10_31-20260930/association/validate_association_conflicts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_association import (
    INVESTIGATION,
    ROOT,
    assoc,
    code_identity,
    load_e0_regions,
    persisted_membership,
)

from contextmap.evaluation.association_conflicts import (
    DepthLayerCriterion,
    measure_association_conflicts,
    measure_occlusion_proximity,
)

OUT = Path(__file__).parent / "results"
# Definições de p5 (depth_layers) e de p5_occlusion_anatomy, sem mudança.
DEPTH_LAYERS = DepthLayerCriterion(
    gap_m=0.3, gap_ratio=0.08, main_layer_min_share=0.2, min_points=30
)
PROXIMITY_M = 0.5
# Limiares do plano (C5): aninhado = contenção 2D ≥ 0,9; duplicata = IoU ≥ 0,8.
NESTED_CONTAINMENT = 0.9
DUPLICATE_IOU = 0.8
LARGE_MASK_USABLE_SHARE = 0.10  # só para o diagnóstico descritivo da classe "partial"


def main() -> None:
    from common import load_geometry

    e0 = load_e0_regions()
    membership, association_provenance = persisted_membership()
    geo = load_geometry("raw")
    frame = assoc.load_frame("raw")
    res = assoc.associate(geo["map"], frame)
    reimpl_equal = all(
        np.array_equal(np.sort(membership[rid]), res["per_region"][rid]) for rid in membership
    ) and set(membership) == set(res["per_region"])

    conflicts = measure_association_conflicts(
        membership,
        e0.masks,
        res["range"],
        nested_containment_threshold=NESTED_CONTAINMENT,
        duplicate_iou_threshold=DUPLICATE_IOU,
        depth_layers=DEPTH_LAYERS,
    )
    occlusion = measure_occlusion_proximity(
        res["range"], res["support_depth"], res["occluded"], proximity_m=PROXIMITY_M
    )

    p5 = json.loads((INVESTIGATION / "results/p5_association_ablation.json").read_text())
    ref = p5["configs"]["accumulated_map_81_scans|base|pipeline"]
    anatomy = json.loads((INVESTIGATION / "results/p5_occlusion_anatomy.json").read_text())
    hist = {int(k): v for k, v in ref["regions_per_associated_point_hist"].items()}
    ref_rows = {row["region_id"]: row["off_main_layer_fraction"] for row in ref["per_region"]}

    def row(metric: str, reference: object, measured: object, note: str = "") -> dict[str, object]:
        if isinstance(reference, float) and isinstance(measured, float):
            same = abs(reference - measured) <= 1e-12
        else:
            same = reference == measured
        return {"metric": metric, "p5": reference, "module": measured, "equal": same, "note": note}

    weighted = conflicts.off_main_layer_fraction_weighted
    comparison = [
        row("visible", ref["visible"], int(res["visible"].sum())),
        row("occluded", ref["occluded"], occlusion.occluded_point_count),
        row("associated_unique", ref["associated_unique"], conflicts.associated_point_count),
        row(
            "membership_pairs",
            ref["membership_count"] if "membership_count" in ref else ref["membership_pairs"],
            conflicts.membership_count,
        ),
        row("regions_per_associated_point_hist", hist, dict(conflicts.regions_per_point_histogram)),
        row(
            "multi_mask_point_rate",
            1 - hist[1] / ref["associated_unique"],
            conflicts.multi_mask_point_rate,
        ),
        row("max_regions_per_point", max(hist), conflicts.max_regions_per_point),
        row(
            "regions_with_ge30_points",
            ref["regions_with_ge30_points"],
            conflicts.depth_layer_mask_count,
        ),
        row(
            "off_main_layer_fraction_weighted",
            ref["off_main_layer_fraction_weighted"],
            None if weighted is None else round(weighted, 4),
            "p5 grava arredondado a 4 casas; comparado depois do mesmo arredondamento",
        ),
        row(
            "occluded_within_0p5m_of_support_fraction",
            anatomy["occluded_within_0p5m_of_support_fraction"],
            occlusion.near_support_fraction,
        ),
    ]
    per_region_mismatch = [
        mask.region_id
        for mask in conflicts.masks
        if (
            None if mask.off_main_layer_fraction is None else round(mask.off_main_layer_fraction, 4)
        )
        != ref_rows[mask.region_id]
    ]

    # Diagnóstico descritivo da classe "partial" (não é definição de métrica nem política).
    usable_px = int(e0.usable_area.as_array().sum())
    large = {rid for rid, m in e0.masks.items() if m.area >= LARGE_MASK_USABLE_SHARE * usable_px}
    partial = [p for p in conflicts.pairs if p.conflict_class.value == "partial"]
    partial_total = sum(p.shared_point_count for p in partial)
    bins = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, NESTED_CONTAINMENT))
    partial_breakdown = {
        "shared_point_incidences": partial_total,
        "by_containment_2d": [
            {
                "range": [lo, hi],
                "pairs": sum(1 for p in partial if lo <= p.containment_2d < hi),
                "incidences": (
                    n := sum(p.shared_point_count for p in partial if lo <= p.containment_2d < hi)
                ),
                "share": n / partial_total if partial_total else None,
            }
            for lo, hi in bins
        ],
        "large_mask_definition": f"área ≥ {LARGE_MASK_USABLE_SHARE:.0%} da área útil ({usable_px} px)",
        "large_masks": sorted(large),
        "incidences_involving_a_large_mask": (
            m := sum(p.shared_point_count for p in partial if set(p.region_ids) & large)
        ),
        "share_involving_a_large_mask": m / partial_total if partial_total else None,
    }

    out = {
        "experiment": "C5 — validação da taxonomia de conflitos contra p5 (E0, calibração base)",
        "provenance": {
            **e0.provenance,
            **association_provenance,
            "code": code_identity(),
            "depth": "alcance do raio (RAY_RANGE, métrica do pipeline para MEI), de assoc.py",
        },
        "reimplementation_membership_equals_artifact": reimpl_equal,
        "comparison": comparison,
        "per_region_off_main_layer_mismatches": per_region_mismatch,
        "conflicts": conflicts.to_dict(),
        "occlusion_proximity": occlusion.to_dict(),
        "partial_class_breakdown": partial_breakdown,
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "e0_association_conflicts_validation.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("reimplementation == artifact:", reimpl_equal)
    for item in comparison:
        status = "OK" if item["equal"] else "DIFF"
        print(
            f"{item['metric']:42s} p5={item['p5']!s:>28.28s} mod={item['module']!s:>28.28s} {status}"
        )
    print("per-region mismatches:", per_region_mismatch)
    summary = {k: v for k, v in conflicts.to_dict().items() if k not in ("pairs", "masks")}
    print(json.dumps(summary, indent=1))
    print(json.dumps(partial_breakdown, indent=1))
    print("->", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
