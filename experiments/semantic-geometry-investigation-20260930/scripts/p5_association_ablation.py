"""Partes 4-5: ablação da associação do frame 55, uma variável por vez.

Variáveis: calibração (base / correção V1 / melhor ajuste da parte 4), geometria (mapa
acumulado de 81 scans sem deskew / só o scan local mais próximo, com deskew) e oclusão
(regra do pipeline / z-buffer por pixel / nenhuma). A reimplementação foi validada contra o
artefato (0/83 regiões divergentes) antes desta variação.

Métrica de contaminação por região: os alcances dos pontos associados são agrupados em
camadas separadas por saltos > max(0.3 m, 8 % do alcance); a fração fora da camada mais
próxima com >= 20 % dos pontos mede pontos de superfícies diferentes pintados pela mesma
máscara (fundo através de bordas, ou superfícies de trás). Não usa rótulo.
"""
from __future__ import annotations

import json

import numpy as np
from assoc import associate, load_frame
from common import (
    IMAGE_ID,
    Trajectory,
    dump,
    load_geometry,
    load_jsonl,
    load_scan_payload,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from p3_time_offset_deskew import place

CALIBRATIONS = {"base": (0.0, 0.0, 0.0), "v1_correction": (4.25, -2.25, 0.0), "fit_p4": (4.0, -1.75, 0.0)}


def depth_layers(r: np.ndarray) -> float:
    """Fração de pontos fora da camada de profundidade principal (mais próxima com >=20%)."""
    if len(r) < 30:
        return float("nan")
    s = np.sort(r)
    gaps = np.diff(s) > np.maximum(0.3, 0.08 * s[:-1])
    labels = np.concatenate([[0], np.cumsum(gaps)])
    counts = np.bincount(labels)
    main = next(i for i, c in enumerate(counts) if c >= 0.2 * len(s))
    return float(1 - counts[main] / len(s))


def local_scan_points() -> tuple[np.ndarray, str]:
    """Scan LiDAR mais próximo da imagem, deskew por ponto, no frame do mapa."""
    geo = load_geometry("raw")
    scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
    index = {e["observation_id"]: e for e in sequence_index()}
    t_img = ts_ns(index[IMAGE_ID]["timestamp"])
    s = min(scans, key=lambda s: abs(ts_ns(s["acquisition_timestamp"]) - t_img))
    seg = geo[s["first_geometry_index"] : s["first_geometry_index"] + s["geometry_count"]]
    _, raw = load_scan_payload(index[s["observation_id"]])
    times = raw["time"].astype(np.float64)[seg["source_index"]]
    pts = place(Trajectory(), static_transform("epson", "cmu_rc1_velodyne"),
                ts_ns(s["acquisition_timestamp"]), times, seg["source"], 0, True)
    return pts, s["observation_id"], s["first_geometry_index"]


def summarize(res: dict, labels: dict) -> dict:
    per_region = res["per_region"]
    assoc = np.unique(np.concatenate([v for v in per_region.values()] or [np.array([], int)]))
    member_count = np.zeros(len(res["visible"]), dtype=np.int32)
    for v in per_region.values():
        member_count[v] += 1
    contamination, weights, rows = [], [], []
    for rid, idx in per_region.items():
        c = depth_layers(res["range"][idx])
        rows.append({"region_id": rid, "label": labels.get(rid), "associated": len(idx),
                     "off_main_layer_fraction": None if np.isnan(c) else round(c, 4)})
        if not np.isnan(c):
            contamination.append(c)
            weights.append(len(idx))
    hist = np.bincount(member_count[assoc]) if len(assoc) else np.array([0])
    return {
        "visible": int(res["visible"].sum()),
        "occluded": int(res["occluded"].sum()),
        "associated_unique": len(assoc),
        "membership_pairs": int(sum(len(v) for v in per_region.values())),
        "regions_per_associated_point_hist": {str(k): int(v) for k, v in enumerate(hist) if v},
        "regions_with_ge30_points": len(contamination),
        "off_main_layer_fraction_weighted": round(float(np.average(contamination, weights=weights)), 4) if weights else None,
        "off_main_layer_fraction_median": round(float(np.median(contamination)), 4) if contamination else None,
        "per_region": rows,
    }


def main() -> None:
    frame = load_frame("raw")
    result = json.loads((run_dir("raw") / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    labels = {}
    for c in result["claims"]:
        if c.get("role") == "primary" and c.get("region_id"):
            labels[c["region_id"]] = c["hypothesis"]
    geo = load_geometry("raw")
    scan_pts, scan_id, _ = local_scan_points()
    geometries = {"accumulated_map_81_scans": geo["map"], f"local_scan_{scan_id}_deskewed": scan_pts}
    out = {"image": IMAGE_ID, "configs": {}}
    base_sets = {}
    for gname, pts in geometries.items():
        for cname, corr in CALIBRATIONS.items():
            for occ in ("pipeline", "exact_pixel", "none"):
                if gname.startswith("accumulated") and occ == "none":
                    continue  # sem oclusão, o mapa inteiro atravessa as paredes: não informativo
                res = associate(pts, frame, correction=corr, occlusion=occ)
                key = f"{gname}|{cname}|{occ}"
                s = summarize(res, labels)
                if cname == "base":
                    base_sets[(gname, occ)] = res["per_region"]
                else:
                    ref = base_sets[(gname, occ)]
                    jac = []
                    for rid, idx in res["per_region"].items():
                        a, b = set(ref[rid].tolist()), set(idx.tolist())
                        if a or b:
                            jac.append(len(a & b) / len(a | b))
                    s["jaccard_vs_base_median"] = round(float(np.median(jac)), 4)
                    s["jaccard_vs_base_mean"] = round(float(np.mean(jac)), 4)
                out["configs"][key] = s
                print(key, {k: v for k, v in s.items() if k != "per_region"}, flush=True)
    # deslocamento em pixel entre base e correção, para os pontos visíveis do pipeline
    base = associate(geo["map"], frame)
    corr = associate(geo["map"], frame, correction=CALIBRATIONS["v1_correction"])
    vis = base["visible"] & corr["in_image"]
    shift = np.linalg.norm(corr["uv"][vis] - base["uv"][vis], axis=1)
    out["pixel_shift_base_to_v1_correction_px"] = {
        "median": round(float(np.median(shift)), 2), "p10": round(float(np.percentile(shift, 10)), 2),
        "p90": round(float(np.percentile(shift, 90)), 2), "max": round(float(shift.max()), 2)}
    dump("p5_association_ablation.json", out)


if __name__ == "__main__":
    main()
