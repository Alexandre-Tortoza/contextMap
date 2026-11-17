"""C5 (preliminar): ablação offline da associação do frame 55 com as regiões do E0.

Grade: calibração {base, corrigida yaw 4,0° / pitch −1,75° / roll 0 (convenção de p4/p5,
`rotation_correction` = Rz(roll)·Rx(pitch)·Ry(yaw) aplicada no frame óptico)} × candidatos
{mapa inteiro, scans com início a ±50/±150/±500 ms da imagem, scan em andamento no instante da
imagem (substituto declarado do ±50 ms, que é vazio), alcance ≤ 20 m do centro óptico} ×
oclusão `neighborhood_radius_cells` ∈ {0, 1, 2} (célula 4 px, margem max(0,1 m; 2 %), as
atuais). Tudo com `assoc.py` (reimplementação validada = artefato) e medido com
`evaluation.association_conflicts`. Nenhuma política entra em produção.

Rodar (da raiz do repo):
    PYTHONPATH=src <python3.12 do mise> \
        experiments/correction-benchmark-10_31-20260930/association/c5_offline_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_association import (
    INVESTIGATION,
    ROOT,
    RUN,
    assoc,
    code_identity,
    load_e0_regions,
    persisted_membership,
)

from contextmap.evaluation.association_conflicts import (
    ConflictClass,
    DepthLayerCriterion,
    measure_association_conflicts,
    measure_occlusion_proximity,
)

OUT = Path(__file__).parent / "results"
CALIBRATIONS = {"base": (0.0, 0.0, 0.0), "corrected_yaw4.0_pitch-1.75": (4.0, -1.75, 0.0)}
SCAN_WINDOWS_MS = (50, 150, 500)
MAX_RANGE_M = 20.0
RADII = (0, 1, 2)
DEPTH_LAYERS = DepthLayerCriterion(
    gap_m=0.3, gap_ratio=0.08, main_layer_min_share=0.2, min_points=30
)
PROXIMITY_M = 0.5
NESTED_CONTAINMENT = 0.9
DUPLICATE_IOU = 0.8
NESTED_SENSITIVITY = (0.5, 0.8, 0.9)


def scan_catalogue(t_img: int) -> list[dict]:
    """Scans do mapa com início, deslocamento para a imagem e fatia no geometry.bin."""
    from common import load_jsonl, ts_ns

    scans = load_jsonl(RUN / "geometric_mapping/outputs/source-index.jsonl")
    return [
        {
            "observation_id": s["observation_id"],
            "start_offset_ms": (ts_ns(s["acquisition_timestamp"]) - t_img) / 1e6,
            "first": s["first_geometry_index"],
            "count": s["geometry_count"],
        }
        for s in scans
    ]


def sweep_duration_ms(observation_id: str) -> float:
    """Duração da varredura (máximo do campo `time` por ponto) do payload bruto do scan."""
    from common import load_scan_payload, sequence_index

    entry = next(e for e in sequence_index() if e["observation_id"] == observation_id)
    _, raw = load_scan_payload(entry)
    return float(raw["time"].max()) * 1e3


def candidate_sets(geo: np.ndarray, frame, scans: list[dict]) -> dict[str, dict]:
    """Índices globais de cada política de candidatos, com o que a define."""
    everything = np.arange(len(geo))
    sets = {"full_map": {"indices": everything, "definition": "todos os pontos do mapa"}}

    def from_scans(selected: list[dict]) -> np.ndarray:
        if not selected:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(s["first"], s["first"] + s["count"]) for s in selected])

    for window in SCAN_WINDOWS_MS:
        selected = [s for s in scans if abs(s["start_offset_ms"]) <= window]
        sets[f"scan_start_within_{window}ms"] = {
            "indices": from_scans(selected),
            "definition": f"|início do scan − t_imagem| ≤ {window} ms",
            "scans": [(s["observation_id"], round(s["start_offset_ms"], 2)) for s in selected],
        }
    in_progress = max(
        (s for s in scans if s["start_offset_ms"] <= 0), key=lambda s: s["start_offset_ms"]
    )
    sweep = sweep_duration_ms(in_progress["observation_id"])
    sets["scan_in_progress"] = {
        "indices": from_scans([in_progress]),
        "definition": "último scan com início ≤ t_imagem (varredura em andamento no instante da imagem)",
        "scans": [(in_progress["observation_id"], round(in_progress["start_offset_ms"], 2))],
        "sweep_duration_ms": round(sweep, 2),
        "image_inside_sweep": bool(-in_progress["start_offset_ms"] <= sweep),
    }
    centre = frame.t_map_cam[:3, 3]
    within = np.linalg.norm(geo["map"] - centre, axis=1) <= MAX_RANGE_M
    sets[f"range_{MAX_RANGE_M:g}m"] = {
        "indices": np.flatnonzero(within),
        "definition": f"||P − C|| ≤ {MAX_RANGE_M:g} m, C = centro óptico (calibração base; a "
        "correção é só rotação, C não muda)",
    }
    return sets


def run_arm(geo, frame, masks, indices, correction, radius, e0_associated) -> dict:
    """Associa os candidatos, mede estágios, conflitos, camadas e oclusão próxima."""
    assoc.RADIUS_CELLS = radius  # assoc lê a constante de módulo a cada chamada
    res = assoc.associate(geo["map"][indices], frame, correction=correction)
    membership = {rid: indices[local] for rid, local in res["per_region"].items()}
    depth = np.full(len(geo), np.nan)
    depth[indices] = res["range"]
    conflicts = measure_association_conflicts(
        membership,
        masks,
        depth,
        nested_containment_threshold=NESTED_CONTAINMENT,
        duplicate_iou_threshold=DUPLICATE_IOU,
        depth_layers=DEPTH_LAYERS,
    )
    occlusion = measure_occlusion_proximity(
        res["range"], res["support_depth"], res["occluded"], proximity_m=PROXIMITY_M
    )
    associated = np.unique(np.concatenate(list(membership.values())))
    kept = np.intersect1d(associated, e0_associated, assume_unique=True).size
    report = conflicts.to_dict()
    return {
        "stages": {
            "candidates": len(indices),
            "projected": int(res["in_domain"].sum()),
            "in_image": int(res["in_image"].sum()),
            "in_valid_support": int(res["supported"].sum()),
            "occluded": int(res["occluded"].sum()),
            "visible": int(res["visible"].sum()),
            "associated": conflicts.associated_point_count,
        },
        "regions_with_points": sum(1 for v in membership.values() if len(v)),
        "e0_associated_retained": kept,
        "e0_associated_retained_fraction": kept / len(e0_associated),
        "multi_mask_point_rate": conflicts.multi_mask_point_rate,
        "mean_regions_per_point": conflicts.mean_regions_per_point,
        "max_regions_per_point": conflicts.max_regions_per_point,
        "regions_per_point_histogram": report["regions_per_point_histogram"],
        "multi_mask_points_by_class": report["multi_mask_points_by_class"],
        "multi_mask_point_rate_by_class": report["multi_mask_point_rate_by_class"],
        "pair_count_by_class": report["pair_count_by_class"],
        "depth_layer_mask_count": conflicts.depth_layer_mask_count,
        "off_main_layer_fraction_weighted": conflicts.off_main_layer_fraction_weighted,
        "occlusion_near_support": occlusion.to_dict(),
    }


def nested_sensitivity(membership, masks, depth) -> dict[str, dict]:
    """Sensibilidade da taxonomia do E0 ao limiar de aninhamento (definição, não política)."""
    out = {}
    for threshold in NESTED_SENSITIVITY:
        report = measure_association_conflicts(
            membership,
            masks,
            depth,
            nested_containment_threshold=threshold,
            duplicate_iou_threshold=DUPLICATE_IOU,
            depth_layers=DEPTH_LAYERS,
        ).to_dict()
        out[str(threshold)] = {
            "multi_mask_point_rate_by_class": report["multi_mask_point_rate_by_class"],
            "pair_count_by_class": report["pair_count_by_class"],
        }
    return out


def main() -> None:
    from common import load_geometry

    started = time.time()
    e0 = load_e0_regions()
    e0_membership, association_provenance = persisted_membership()
    e0_associated = np.unique(np.concatenate(list(e0_membership.values())))
    geo = load_geometry("raw")
    frame = assoc.load_frame("raw")
    scans = scan_catalogue(frame.t_ns)
    candidates = candidate_sets(geo, frame, scans)

    arms, skipped = {}, {}
    for cname, correction in CALIBRATIONS.items():
        for kname, spec in candidates.items():
            if len(spec["indices"]) == 0:
                skipped[f"{cname}|{kname}"] = "nenhum candidato: " + spec["definition"]
                continue
            for radius in RADII:
                key = f"{cname}|{kname}|r{radius}"
                arms[key] = {
                    "calibration": cname,
                    "correction_yaw_pitch_roll_deg": list(correction),
                    "candidates": kname,
                    "neighborhood_radius_cells": radius,
                    **run_arm(
                        geo, frame, e0.masks, spec["indices"], correction, radius, e0_associated
                    ),
                }
                a = arms[key]
                print(
                    f"{key:55s} vis={a['stages']['visible']:>7d} assoc={a['stages']['associated']:>6d} "
                    f"multi={a['multi_mask_point_rate']:.3f} "
                    f"partial={a['multi_mask_point_rate_by_class']['partial']:.3f} "
                    f"layer={a['off_main_layer_fraction_weighted']} "
                    f"near={a['occlusion_near_support']['near_support_fraction']}",
                    flush=True,
                )

    # Sanidade: o braço base|mapa|r2 é o E0 persistido; corrigido|mapa|r2 é o fit_p4 de p5.
    p5 = json.loads((INVESTIGATION / "results/p5_association_ablation.json").read_text())
    fit = p5["configs"]["accumulated_map_81_scans|fit_p4|pipeline"]
    base_arm = arms["base|full_map|r2"]
    corr_arm = arms["corrected_yaw4.0_pitch-1.75|full_map|r2"]
    sanity = {
        "base_full_r2_equals_e0_artifact": base_arm["stages"]["visible"] == 23948
        and base_arm["stages"]["associated"] == len(e0_associated)
        and base_arm["e0_associated_retained"] == len(e0_associated),
        "corrected_full_r2_equals_p5_fit_p4": corr_arm["stages"]["visible"] == fit["visible"]
        and corr_arm["stages"]["associated"] == fit["associated_unique"]
        and corr_arm["regions_per_point_histogram"] == fit["regions_per_associated_point_hist"]
        and round(corr_arm["off_main_layer_fraction_weighted"], 4)
        == fit["off_main_layer_fraction_weighted"],
    }
    depth = np.full(len(geo), np.nan)
    assoc.RADIUS_CELLS = 2
    depth[:] = assoc.associate(geo["map"], frame)["range"]
    sensitivity = nested_sensitivity(e0_membership, e0.masks, depth)

    out = {
        "experiment": "C5 preliminar — ablação offline de calibração × candidatos × raio de oclusão",
        "status": "preliminar: regiões do E0 (SAM2 0/0), mapa sem deskew; nada entra em produção",
        "provenance": {
            **e0.provenance,
            **association_provenance,
            "code": code_identity(),
            "reimplementation": str((INVESTIGATION / "scripts/assoc.py").relative_to(ROOT))
            + " (validada = artefato; results/p4_reimplementation_validation.json)",
            "image_time_ns": int(frame.t_ns),
        },
        "definitions": {
            "calibrations": {k: list(v) for k, v in CALIBRATIONS.items()},
            "candidates": {
                k: {kk: vv for kk, vv in v.items() if kk != "indices"}
                | {"points": len(v["indices"])}
                for k, v in candidates.items()
            },
            "occlusion": {
                "cell_size_px": assoc.CELL_PX,
                "depth_margin_m": assoc.MARGIN_M,
                "depth_margin_ratio": assoc.MARGIN_RATIO,
                "radii": list(RADII),
            },
            "nested_containment_threshold": NESTED_CONTAINMENT,
            "duplicate_iou_threshold": DUPLICATE_IOU,
            "depth_layers": DEPTH_LAYERS.to_dict(),
            "occlusion_proximity_m": PROXIMITY_M,
            "depth_metric": "alcance do raio (RAY_RANGE), métrica do pipeline para MEI",
            "conflict_classes": [c.value for c in ConflictClass],
        },
        "sanity": sanity,
        "skipped_arms": skipped,
        "arms": arms,
        "e0_taxonomy_nested_threshold_sensitivity": sensitivity,
        "runtime_s": round(time.time() - started, 1),
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "c5_offline_ablation.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("sanity:", sanity)
    print("skipped:", skipped)
    print("->", path.relative_to(ROOT), f"({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
