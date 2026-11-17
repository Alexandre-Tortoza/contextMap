"""Parte 4: a extrínseca câmera<->corpo está alinhada nos dados do ContextMap2 (pose GT)?

Replica o critério do V1 (estimate_corridor02_camera_rotation.py, #285): o mesmo ponto 3D visto
por vários frames deve amostrar a mesma textura local; uma rotação errada faz o ponto cair em
superfícies diferentes a cada distância e a textura diverge. Aqui a geometria é a do próprio
ContextMap2 (scans da janela 10:31 com pose GT e deskew por ponto) e a extrínseca base é a do
calibration.json da sequência. Mede a inconsistência para: base, correção publicada pelo V1
(yaw 4.25, pitch -2.25, roll 0) e a melhor da grade, com trechos de ajuste e de validação.
"""
from __future__ import annotations

import numpy as np
from common import (
    ROOT,
    Trajectory,
    camera_model,
    dump,
    load_geometry,
    load_jsonl,
    load_rgb,
    load_scan_payload,
    mei_project,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from p3_time_offset_deskew import place
from PIL import Image
from scipy import ndimage

MAX_RANGE_M = 12.0
MIN_VIEWS = 4
MAX_POINTS = 60_000
SEED = 7
V1_CORRECTION = (4.25, -2.25, 0.0)


def rotation_correction(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
    """Rz(roll) @ Rx(pitch) @ Ry(yaw), mesma convenção do manifest do V1."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def usable_area() -> np.ndarray:
    valid = np.load(ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-valid.npy").astype(bool)
    ego = np.load(ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-ego.npy").astype(bool)
    return ndimage.binary_erosion(valid & ~ego, iterations=6)


def local_texture(rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float)
    high = gray - ndimage.gaussian_filter(gray, 6.0)
    local = high / np.sqrt(ndimage.gaussian_filter(high * high, 6.0) + 4.0)
    return np.clip(ndimage.gaussian_filter(local, 0.8), -3, 3)


def pixels(cam_pts, cam, shape):
    uv, ok = mei_project(cam_pts, cam)
    ok &= np.isfinite(uv).all(axis=1)
    ui = np.zeros(len(uv), dtype=int)
    vi = np.zeros(len(uv), dtype=int)
    ui[ok] = np.round(uv[ok, 0]).astype(int)
    vi[ok] = np.round(uv[ok, 1]).astype(int)
    ok &= (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
    return ui, vi, ok


def main() -> None:
    cam = camera_model()
    t_epson_d = static_transform("epson", "d")
    t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
    traj = Trajectory()
    index = sequence_index()
    images = [e for e in index if e["modality"] == "image"]
    lidar = {e["observation_id"]: e for e in index if e["modality"] == "lidar"}
    geo = load_geometry("raw")
    scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
    usable = usable_area()
    rng = np.random.default_rng(SEED)

    # nuvem deskew por trecho: scans do mapa cujo instante cai no trecho (+-0.5 s)
    def cloud_for(t_lo, t_hi):
        parts = []
        for s in scans:
            t0 = ts_ns(s["acquisition_timestamp"])
            if not (t_lo - 5e8 <= t0 <= t_hi + 5e8):
                continue
            seg = geo[s["first_geometry_index"] : s["first_geometry_index"] + s["geometry_count"]]
            _, raw = load_scan_payload(lidar[s["observation_id"]])
            times = raw["time"].astype(np.float64)[seg["source_index"]]
            near = np.linalg.norm(seg["source"], axis=1) < MAX_RANGE_M
            parts.append(place(traj, t_epson_vel, t0, times[near], seg["source"][near], 0, True))
        return np.concatenate(parts)

    # trechos: 8 frames espaçados de 3 (≈0.12 s) -> ~1 s de movimento por trecho
    groups = []
    for start in range(0, len(images) - 24, 27):
        chosen = images[start : start + 24 : 3]
        frames = []
        for e in chosen:
            t = ts_ns(e["timestamp"])
            try:
                t_map_epson, bracket = traj.pose(t)
            except Exception:
                continue
            if bracket["gap_ns"] > 450_000_000:
                continue
            t_map_cam = t_map_epson @ t_epson_d
            frames.append({"id": e["observation_id"], "t": t, "map_to_camera": np.linalg.inv(t_map_cam),
                           "texture": local_texture(load_rgb(e["observation_id"]))})
        if len(frames) >= 6:
            groups.append(frames)
    print("groups", [len(g) for g in groups], flush=True)

    observations = []
    for frames in groups:
        cloud = cloud_for(frames[0]["t"], frames[-1]["t"])
        visible = np.zeros((len(cloud), len(frames)), dtype=bool)
        cams = np.zeros((len(cloud), len(frames), 3), dtype=np.float32)
        for c, f in enumerate(frames):
            m = f["map_to_camera"]
            cp = cloud @ m[:3, :3].T + m[:3, 3]
            ui, vi, ok = pixels(cp, cam, usable.shape)
            dist = np.linalg.norm(cp, axis=1)
            ok &= dist < MAX_RANGE_M
            ok[ok] &= usable[vi[ok], ui[ok]]
            z = np.full(usable.shape, np.inf)
            np.minimum.at(z, (vi[ok], ui[ok]), dist[ok])
            near = ndimage.minimum_filter(z, size=3)
            seen = ok.copy()
            seen[ok] = dist[ok] <= near[vi[ok], ui[ok]] + 0.05 + 0.03 * dist[ok]
            visible[:, c] = seen
            cams[:, c] = cp
        keep = np.where(visible.sum(1) >= MIN_VIEWS)[0]
        if len(keep) > MAX_POINTS:
            keep = np.sort(rng.choice(keep, MAX_POINTS, replace=False))
        observations.append((frames, visible[keep], cams[keep]))
        print("group", frames[0]["id"], "points", len(keep), flush=True)

    def group_cost(item, rot):
        frames, visible, cams = item
        samples = np.full(visible.shape, np.nan)
        for c, f in enumerate(frames):
            rows = np.where(visible[:, c])[0]
            ui, vi, ok = pixels(cams[rows, c].astype(float) @ rot.T, cam, f["texture"].shape)
            samples[rows[ok], c] = f["texture"][vi[ok], ui[ok]]
        enough = np.sum(~np.isnan(samples), axis=1) >= 3
        return float(np.nansum(np.nanvar(samples[enough], axis=1))), int(enough.sum())

    def cost(items, yaw, pitch, roll=0.0):
        rot = rotation_correction(yaw, pitch, roll)
        tot = cnt = 0
        for it in items:
            a, b = group_cost(it, rot)
            tot, cnt = tot + a, cnt + b
        return tot / cnt

    fit = observations[0::2]
    held = observations[1::2]
    coarse = min((cost(fit, y, p), y, p) for y in np.arange(-8, 8.1, 1.0) for p in np.arange(-8, 8.1, 1.0))
    _, y0, p0 = coarse
    fine = min((cost(fit, y, p), y, p) for y in np.arange(y0 - 1.5, y0 + 1.6, 0.25) for p in np.arange(p0 - 1.5, p0 + 1.6, 0.25))
    best_cost, yb, pb = fine
    per_group = []
    for it in observations:
        c0 = min((cost([it], y, p), y, p) for y in np.arange(-8, 8.1, 1.0) for p in np.arange(-8, 8.1, 1.0))
        c1 = min((cost([it], y, p), y, p) for y in np.arange(c0[1] - 1, c0[1] + 1.1, 0.25) for p in np.arange(c0[2] - 1, c0[2] + 1.1, 0.25))
        per_group.append({"first_frame": it[0][0]["id"], "points": len(it[1]),
                          "yaw_deg": float(c1[1]), "pitch_deg": float(c1[2]),
                          "cost_base": round(cost([it], 0, 0), 5), "cost_best": round(c1[0], 5),
                          "cost_v1": round(cost([it], *V1_CORRECTION[:2]), 5)})
        print(per_group[-1], flush=True)
    result = {
        "groups": len(observations), "fit_groups": len(fit), "held_out_groups": len(held),
        "best_fit": {"yaw_deg": float(yb), "pitch_deg": float(pb), "roll_deg": 0.0},
        "inconsistency": {
            "fit_base": round(cost(fit, 0, 0), 5),
            "fit_v1_correction": round(cost(fit, *V1_CORRECTION[:2]), 5),
            "fit_best": round(best_cost, 5),
            "held_out_base": round(cost(held, 0, 0), 5),
            "held_out_v1_correction": round(cost(held, *V1_CORRECTION[:2]), 5),
            "held_out_best_fit": round(cost(held, yb, pb), 5),
        },
        "per_group": per_group,
        "yaw_profile_at_best_pitch": [
            {"yaw": float(y), "cost": round(cost(fit, y, pb), 5)} for y in np.arange(-6, 8.1, 1.0)
        ],
        "pitch_profile_at_best_yaw": [
            {"pitch": float(p), "cost": round(cost(fit, yb, p), 5)} for p in np.arange(-6, 6.1, 1.0)
        ],
    }
    dump("p4_calibration_photometric.json", result)
    print(result["best_fit"], result["inconsistency"])


if __name__ == "__main__":
    main()
