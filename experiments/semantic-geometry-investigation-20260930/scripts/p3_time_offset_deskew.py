"""Parte 3b: offset de relógio GT <-> header, isolado do efeito de scan sem deskew.

Cada ponto é posicionado com a pose GT em t_header + time_i + delta (time_i = campo "time"
do Velodyne, 0..~100 ms). Assim o delta ótimo mede só o desalinhamento de relógio entre a
trajetória e os headers dos sensores — que é o mesmo desalinhamento que a projeção da câmera
herda, pois câmera, LiDAR e GT compartilham o relógio corrigido.
"""
from __future__ import annotations

import numpy as np
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
from p3_map_consistency import MAX_RANGE_M, plane_model, residuals


def place(traj, t_epson_vel, t0_ns, times_s, src, delta_ns, deskew=True):
    """Posiciona pontos com pose por ponto (deskew) ou por scan."""
    out = np.empty_like(src)
    if not deskew:
        t_map_epson, _ = traj.pose(t0_ns + delta_ns)
        m = t_map_epson @ t_epson_vel
        return src @ m[:3, :3].T + m[:3, 3]
    # agrupa em fatias de 5 ms para não interpolar a pose ponto a ponto
    bins = np.floor(times_s / 0.005).astype(int)
    for b in np.unique(bins):
        sel = bins == b
        t = t0_ns + int(round((b + 0.5) * 5e6)) + delta_ns
        t_map_epson, _ = traj.pose(t)
        m = t_map_epson @ t_epson_vel
        out[sel] = src[sel] @ m[:3, :3].T + m[:3, 3]
    return out


def main() -> None:
    geo = load_geometry("raw")
    scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
    index = {e["observation_id"]: e for e in sequence_index()}
    t_img = ts_ns(index[IMAGE_ID]["timestamp"])
    traj = Trajectory()
    t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
    ref_scan = min(scans, key=lambda s: abs(ts_ns(s["acquisition_timestamp"]) - t_img))
    t_ref = ts_ns(ref_scan["acquisition_timestamp"])
    chosen = [s for s in scans if abs(ts_ns(s["acquisition_timestamp"]) - t_ref) <= 3e9]

    data = {}
    for s in chosen:
        seg = geo[s["first_geometry_index"] : s["first_geometry_index"] + s["geometry_count"]]
        _, raw = load_scan_payload(index[s["observation_id"]])
        times = raw["time"].astype(np.float64)[seg["source_index"]]
        near = np.linalg.norm(seg["source"], axis=1) < MAX_RANGE_M
        data[s["observation_id"]] = (ts_ns(s["acquisition_timestamp"]), times[near], seg["source"][near])

    sweep = []
    for deskew in (True, False):
        for delta_ms in range(-150, 201, 25):
            d = delta_ms * 1_000_000
            t0, times, src = data[ref_scan["observation_id"]]
            ref = place(traj, t_epson_vel, t0, times, src, d, deskew)
            tree, normals, planarity = plane_model(ref)
            all_r = []
            for oid, (t0, times, src) in data.items():
                if oid == ref_scan["observation_id"]:
                    continue
                r = residuals(place(traj, t_epson_vel, t0, times, src, d, deskew), ref, tree, normals, planarity)
                if len(r) >= 200:
                    all_r.append(r)
            all_r = np.concatenate(all_r)
            row = {"deskew": deskew, "delta_ms": delta_ms, "matched": len(all_r),
                   "median_m": round(float(np.median(all_r)), 5),
                   "p90_m": round(float(np.percentile(all_r, 90)), 5)}
            sweep.append(row)
            print(row, flush=True)
    dump("p3_time_offset_deskew.json", {"reference_scan": ref_scan["observation_id"],
                                         "scans_used": len(chosen), "sweep": sweep})


if __name__ == "__main__":
    main()
