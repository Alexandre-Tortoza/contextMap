"""Parte 3: consistência do mapa acumulado (81 scans) e offset temporal GT <-> LiDAR.

Para um scan de referência (o mais próximo da imagem 55), ajusta planos locais (PCA k-NN) e
mede a distância ponto-plano dos pontos de outros scans na mesma superfície. Depois
re-posiciona cada scan com a pose GT em t_scan + delta e procura o delta que minimiza o
resíduo: se o mínimo não estiver em 0, há offset entre o relógio da GT e o do LiDAR, e a mesma
pose deslocada é a que posiciona a câmera na projeção.
"""
from __future__ import annotations

import numpy as np
from common import (
    IMAGE_ID,
    Trajectory,
    apply,
    dump,
    load_geometry,
    load_jsonl,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from scipy.spatial import cKDTree

MAX_RANGE_M = 12.0     # só superfícies próximas, bem amostradas
PLANARITY_MAX = 0.02   # lambda_min / soma: exige vizinhança planar
NEIGHBOR_RADIUS_M = 0.25


def plane_model(ref: np.ndarray):
    tree = cKDTree(ref)
    _, nbr = tree.query(ref, k=12)
    pts = ref[nbr]
    centered = pts - pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / 12
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    planarity = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)
    return tree, normals, planarity


def residuals(points: np.ndarray, ref: np.ndarray, tree, normals, planarity) -> np.ndarray:
    d, i = tree.query(points, distance_upper_bound=NEIGHBOR_RADIUS_M)
    ok = np.isfinite(d)
    ok[ok] &= planarity[i[ok]] < PLANARITY_MAX
    r = np.abs(np.einsum("ij,ij->i", points[ok] - ref[i[ok]], normals[i[ok]]))
    return r


def main() -> None:
    geo = load_geometry("raw")
    scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
    image = next(e for e in sequence_index() if e["observation_id"] == IMAGE_ID)
    t_img = ts_ns(image["timestamp"])
    traj = Trajectory()
    t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
    ref_scan = min(scans, key=lambda s: abs(ts_ns(s["acquisition_timestamp"]) - t_img))

    def scan_xyz(s, delta_ns=0, frame="map"):
        seg = geo[s["first_geometry_index"] : s["first_geometry_index"] + s["geometry_count"]]
        src = seg["source"]
        near = np.linalg.norm(src, axis=1) < MAX_RANGE_M
        if delta_ns == 0:
            return seg["map"][near]
        t_map_epson, _ = traj.pose(ts_ns(s["acquisition_timestamp"]) + delta_ns)
        return apply(t_map_epson @ t_epson_vel, src[near])

    ref = scan_xyz(ref_scan)
    tree, normals, planarity = plane_model(ref)
    t_ref = ts_ns(ref_scan["acquisition_timestamp"])

    # 1) resíduo por scan, como persistido no mapa (delta = 0)
    per_scan = []
    for s in scans:
        if s is ref_scan:
            continue
        dt = (ts_ns(s["acquisition_timestamp"]) - t_ref) / 1e9
        r = residuals(scan_xyz(s), ref, tree, normals, planarity)
        if len(r) < 200:
            continue
        per_scan.append({
            "observation_id": s["observation_id"], "dt_s": round(dt, 3), "matched": len(r),
            "median_m": round(float(np.median(r)), 4), "p90_m": round(float(np.percentile(r, 90)), 4),
        })

    # 2) varredura de offset temporal: mesma métrica, scans a <= 3 s da referência
    nearby = [s for s in scans if s is not ref_scan and abs(ts_ns(s["acquisition_timestamp"]) - t_ref) <= 3e9]
    sweep = []
    for delta_ms in range(-300, 301, 25):
        meds = []
        for s in nearby:
            # a referência também se move com delta: o offset é do relógio, não de um scan
            pass
        t_ref_pose, _ = traj.pose(t_ref + delta_ms * 1_000_000)
        seg_ref = geo[ref_scan["first_geometry_index"] : ref_scan["first_geometry_index"] + ref_scan["geometry_count"]]
        src_ref = seg_ref["source"][np.linalg.norm(seg_ref["source"], axis=1) < MAX_RANGE_M]
        ref_d = apply(t_ref_pose @ t_epson_vel, src_ref)
        tree_d, normals_d, planarity_d = plane_model(ref_d)
        all_r = []
        for s in nearby:
            r = residuals(scan_xyz(s, delta_ns=delta_ms * 1_000_000), ref_d, tree_d, normals_d, planarity_d)
            if len(r) >= 200:
                all_r.append(r)
        all_r = np.concatenate(all_r)
        sweep.append({"delta_ms": delta_ms, "matched": len(all_r),
                      "median_m": round(float(np.median(all_r)), 5), "p90_m": round(float(np.percentile(all_r, 90)), 5)})
        print(sweep[-1], flush=True)

    # velocidade da plataforma no instante da imagem (para converter erro de tempo em erro espacial)
    a, _ = traj.pose(t_img - 50_000_000)
    b, _ = traj.pose(t_img + 50_000_000)
    lin = np.linalg.norm(b[:3, 3] - a[:3, 3]) / 0.1
    rot = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))) / 0.1
    i = int(np.searchsorted(traj.t, t_img))
    gaps = np.diff(traj.t[max(0, i - 60) : i + 60]) / 1e6
    dump("p3_map_consistency.json", {
        "reference_scan": ref_scan["observation_id"],
        "max_range_m": MAX_RANGE_M, "planarity_max": PLANARITY_MAX,
        "per_scan_residual_delta0": per_scan,
        "time_offset_sweep": sweep,
        "platform_speed_at_image": {"linear_m_s": round(float(lin), 3), "angular_deg_s": round(float(rot), 2)},
        "gt_sample_gaps_ms_around_image": {
            "values_hist": {str(int(round(g))): int(c) for g, c in zip(*np.unique(np.round(gaps), return_counts=True))},
        },
    })


if __name__ == "__main__":
    main()
