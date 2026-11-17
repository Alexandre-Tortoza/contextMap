"""Leitores e utilitários compartilhados da investigação (somente leitura dos artefatos).

Nada aqui importa código de produção para *calcular* geometria: transforms, quatérnios e o
modelo MEI são reimplementados de forma independente, para que uma concordância com os
artefatos seja evidência e não tautologia.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/alexmrtr/Projects/contextmap2")
RUNS = ROOT / "workspace-old"
BUNDLE = ROOT / "experiments/semantic-geometry-investigation-20260930"
RESULTS = BUNDLE / "results"
FIGURES = BUNDLE / "figures"
SEQ_ID = "aa0551d92f7b4a7aa3508dbaa1ee8ea3"
POSE_SEQ_ID = "d422227f15f14d0cb8e071652c5c20b5"
IMAGE_ID = "camera_1_image_raw-000055"

# Registro empacotado do geometry.bin: "<3d3dIqI" (64 bytes, sem padding).
GEOMETRY_DTYPE = np.dtype(
    {
        "names": ["map", "source", "scan", "source_index", "count"],
        "formats": [("<f8", 3), ("<f8", 3), "<u4", "<i8", "<u4"],
        "offsets": [0, 24, 48, 52, 60],
        "itemsize": 64,
    }
)


def run_dir(arm: str) -> Path:
    return RUNS / f"frame-10_31-{arm}"


def seq_dir(arm: str = "raw") -> Path:
    return run_dir(arm) / "sequences/corridor-02" / SEQ_ID


def load_geometry(arm: str) -> np.ndarray:
    path = run_dir(arm) / "geometric_mapping/outputs/geometry.bin"
    return np.fromfile(path, dtype=GEOMETRY_DTYPE)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ts_ns(stamp: dict) -> int:
    return int(stamp["seconds"]) * 1_000_000_000 + int(stamp["nanoseconds"])


def quat_to_matrix(q_xyzw) -> np.ndarray:
    """Rotação ativa de um quatérnio unitário (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q_xyzw)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Inverso de :func:`quat_to_matrix` (x, y, z, w), w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    q = np.array(q)
    return q if q[3] >= 0 else -q


def rigid(translation, q_xyzw) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(q_xyzw)
    t[:3, 3] = np.asarray(translation, dtype=float)
    return t


def apply(t: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ t[:3, :3].T + t[:3, 3]


def slerp(q0, q1, f: float) -> np.ndarray:
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


class Trajectory:
    """Trajetória GT como persistida pelo state_estimation (T_map_epson)."""

    def __init__(self, arm: str = "raw") -> None:
        rows = load_jsonl(run_dir(arm) / "state_estimation/outputs/poses.jsonl")
        rows = [r for r in rows if r.get("validity") == "valid"]
        self.t = np.array([ts_ns(r["timestamp"]) for r in rows], dtype=np.int64)
        self.p = np.array([r["translation_m"] for r in rows], float)
        self.q = np.array([r["orientation_xyzw"] for r in rows], float)
        self.ids = [r["estimate_id"] for r in rows]
        order = np.argsort(self.t)
        self.t, self.p, self.q = self.t[order], self.p[order], self.q[order]
        self.ids = [self.ids[i] for i in order]

    def pose(self, t_ns: int) -> tuple[np.ndarray, dict]:
        """T_map_epson por interpolação (lerp + slerp), com o bracket usado."""
        i = int(np.searchsorted(self.t, t_ns))
        if i < len(self.t) and self.t[i] == t_ns:
            return rigid(self.p[i], self.q[i]), {"gap_ns": 0, "fraction": 0.0, "ids": [self.ids[i]]}
        a, b = i - 1, i
        f = (t_ns - self.t[a]) / (self.t[b] - self.t[a])
        p = self.p[a] + f * (self.p[b] - self.p[a])
        q = slerp(self.q[a], self.q[b], f)
        return rigid(p, q), {
            "gap_ns": int(self.t[b] - self.t[a]),
            "fraction": float(f),
            "ids": [self.ids[a], self.ids[b]],
        }


def calibration(arm: str = "raw") -> dict:
    return json.loads((seq_dir(arm) / "calibration/calibration.json").read_text())


def static_transform(parent: str, child: str, arm: str = "raw") -> np.ndarray:
    for st in calibration(arm)["static_transforms"]:
        if st["parent_frame"] == parent and st["child_frame"] == child:
            return rigid(st["translation"], st["rotation"])
    raise KeyError((parent, child))


def camera_model(arm: str = "raw") -> dict:
    for entry in calibration(arm)["entries"]:
        if entry["camera_model"] is not None:
            return entry["camera_model"]
    raise KeyError("camera")


def mei_project(points_cam: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Projeção MEI/CamOdoCal independente: (N,3) no frame óptico -> pixels (N,2), válido."""
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    xi = cam["xi"]
    k1, k2, p1, p2 = cam["distortion_coefficients"]
    den = z + xi * r
    valid = den > 1e-9
    mx = np.where(valid, x / np.where(valid, den, 1), np.nan)
    my = np.where(valid, y / np.where(valid, den, 1), np.nan)
    r2 = mx * mx + my * my
    radial = k1 * r2 + k2 * r2 * r2
    dx = mx * radial + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
    dy = my * radial + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
    u = cam["fx"] * (mx + dx) + cam["cx"]
    v = cam["fy"] * (my + dy) + cam["cy"]
    return np.column_stack([u, v]), valid


def mei_unproject(pixels: np.ndarray, cam: dict, iterations: int = 30) -> np.ndarray:
    """Raio unitário de cada pixel (inversão iterativa da distorção + esfera unificada)."""
    xi = cam["xi"]
    k1, k2, p1, p2 = cam["distortion_coefficients"]
    xd = (pixels[:, 0] - cam["cx"]) / cam["fx"]
    yd = (pixels[:, 1] - cam["cy"]) / cam["fy"]
    mx, my = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = mx * mx + my * my
        radial = k1 * r2 + k2 * r2 * r2
        dx = mx * radial + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
        dy = my * radial + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
        mx, my = xd - dx, yd - dy
    rho2 = mx * mx + my * my
    scale = (xi + np.sqrt(1 + (1 - xi * xi) * rho2)) / (1 + rho2)
    rays = np.column_stack([scale * mx, scale * my, scale - xi])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def load_rgb(observation_id: str = IMAGE_ID) -> np.ndarray:
    """Imagem RGB (H, W, 3) uint8 a partir do payload bgr8 da sequência."""
    path = seq_dir() / "rgb" / f"{observation_id}.bin"
    bgr = np.fromfile(path, dtype=np.uint8).reshape(480, 640, 3)
    return bgr[:, :, ::-1].copy()


def sequence_index() -> list[dict]:
    return load_jsonl(seq_dir() / "index.jsonl")


def load_scan_payload(entry: dict) -> np.ndarray:
    """XYZ (float64) de um payload PointCloud2 canônico (campos declarados no índice)."""
    fields = {f["name"]: f for f in entry["fields"]}
    kinds = {"float32": "<f4", "float64": "<f8", "uint16": "<u2", "uint32": "<u4"}
    dtype = np.dtype(
        {
            "names": list(fields),
            "formats": [kinds[f["data_type"]] for f in fields.values()],
            "offsets": [f["offset_bytes"] for f in fields.values()],
            "itemsize": entry["point_step_bytes"],
        }
    )
    raw = np.fromfile(seq_dir() / entry["payload_path"], dtype=dtype)
    return np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64), raw


def cloud_stats(xyz: np.ndarray, *, nn_sample: int = 100_000, voxel_m: float = 0.10, seed: int = 0) -> dict:
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n == 0:
        return {"point_count": 0}
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    rng = np.random.default_rng(seed)
    sample = xyz if n <= nn_sample else xyz[rng.choice(n, nn_sample, replace=False)]
    tree = cKDTree(xyz)
    d, _ = tree.query(sample, k=2)
    nn = d[:, 1]
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    occupancy = int(np.unique(keys, axis=0).shape[0])
    return {
        "point_count": n,
        "centroid_m": [round(float(v), 4) for v in xyz.mean(axis=0)],
        "min_m": [round(float(v), 4) for v in lo],
        "max_m": [round(float(v), 4) for v in hi],
        "bbox_dims_m": [round(float(v), 4) for v in hi - lo],
        "mean_nn_distance_m": round(float(nn.mean()), 5),
        "median_nn_distance_m": round(float(np.median(nn)), 5),
        "duplicate_xyz_fraction": round(float((nn == 0).mean()), 5),
        f"voxel_occupancy_{voxel_m:.2f}m": occupancy,
    }


def deltas(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=1)
    return {
        "matched": len(d),
        "max_delta_m": float(d.max()) if len(d) else 0.0,
        "mean_delta_m": float(d.mean()) if len(d) else 0.0,
        "median_delta_m": float(np.median(d)) if len(d) else 0.0,
        "rmse_m": float(np.sqrt((d * d).mean())) if len(d) else 0.0,
        "exactly_equal_fraction": float((d == 0).mean()) if len(d) else 1.0,
    }


def dump(name: str, payload) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    return path
