"""Parte 3: o sentido de laser_to_imu (T_epson_velodyne) e de rgb_camera_to_imu está certo?

LiDAR: consistência multi-scan (deskew, delta=0) com a extrínseca como está vs invertida.
Câmera: consistência fotométrica (critério da parte 4) com T_epson_d vs sua inversa.
"""
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
from p3_time_offset_deskew import place

geo = load_geometry("raw")
scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
index = {e["observation_id"]: e for e in sequence_index()}
t_img = ts_ns(index[IMAGE_ID]["timestamp"])
traj = Trajectory()
ref_scan = min(scans, key=lambda s: abs(ts_ns(s["acquisition_timestamp"]) - t_img))
t_ref = ts_ns(ref_scan["acquisition_timestamp"])
chosen = [s for s in scans if abs(ts_ns(s["acquisition_timestamp"]) - t_ref) <= 3e9]
data = {}
for s in chosen:
    seg = geo[s["first_geometry_index"]: s["first_geometry_index"] + s["geometry_count"]]
    _, raw = load_scan_payload(index[s["observation_id"]])
    times = raw["time"].astype(np.float64)[seg["source_index"]]
    near = np.linalg.norm(seg["source"], axis=1) < MAX_RANGE_M
    data[s["observation_id"]] = (ts_ns(s["acquisition_timestamp"]), times[near], seg["source"][near])
out = {}
t = static_transform("epson", "cmu_rc1_velodyne")
for name, tf in (("as_ingested", t), ("inverted", np.linalg.inv(t)), ("identity", np.eye(4))):
    t0, times, src = data[ref_scan["observation_id"]]
    ref = place(traj, tf, t0, times, src, 0, True)
    tree, normals, planarity = plane_model(ref)
    rs = [residuals(place(traj, tf, *v, 0, True), ref, tree, normals, planarity) for k, v in data.items() if k != ref_scan["observation_id"]]
    r = np.concatenate(rs)
    out[name] = {"median_m": round(float(np.median(r)), 5), "p90_m": round(float(np.percentile(r, 90)), 5), "matched": len(r)}
    print(name, out[name], flush=True)
dump("p3_lidar_extrinsic_direction.json", out)
