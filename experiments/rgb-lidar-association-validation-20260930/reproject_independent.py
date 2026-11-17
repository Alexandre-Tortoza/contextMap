"""Fase 1: reprojeção INDEPENDENTE do pipeline.

Implementa do zero (numpy puro) o modelo MEI + slerp de pose + composição de
extrínseca, projeta as coords 3D de geometry.bin e compara com os pixels
prepared_u/v gravados pelo sensor_association. Se casar em sub-pixel, a cadeia
de projeção (pose->extrínseca->MEI->pixel) está correta.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

B = Path("workspace/frame-10_29-full-run")

# --- calibração MEI ---
fx, fy, cx, cy = 761.6791126513665, 678.7676595096547, 317.73997456826834, 233.50210820810244
xi = 1.5629934283981157
k1, k2, p1, p2 = -0.07083665039247518, 0.3094574364801575, 0.0004607746627279861, 0.00015704097191542886

# --- poses T_map_epson ---
q0 = np.array([0.010705, 0.030073, 0.973681, 0.22567])       # xyzw
t0 = np.array([46.554513, -94.152253, -1.531836])
q1 = np.array([0.013246, 0.011357, 0.973705, 0.227143])
t1 = np.array([46.374691, -94.06502, -1.532075])
frac = 0.11729991076740036

# --- extrínseca T_epson_d ---
q_ed = np.array([0.502638542173682, -0.5028457380724255, 0.5050125794996384, -0.48934952147829336])
t_ed = np.array([0.17313616, 0.0476788, -0.01933398])


def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(qa: np.ndarray, qb: np.ndarray, f: float) -> np.ndarray:
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    d = float(np.dot(qa, qb))
    if d < 0:
        qb = -qb
        d = -d
    if d > 0.9995:
        q = qa + f * (qb - qa)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d)
    th = th0 * f
    q2 = qb - qa * d
    q2 = q2 / np.linalg.norm(q2)
    return qa * np.cos(th) + q2 * np.sin(th)


# T_map_epson interpolado
R_me = quat_to_R(slerp(q0, q1, frac))
t_me = (1 - frac) * t0 + frac * t1
# T_epson_d
R_ed = quat_to_R(q_ed)
# T_map_d = T_map_epson ∘ T_epson_d
R_md = R_me @ R_ed
t_md = R_me @ t_ed + t_me

# --- carrega coords 3D (primeiros 3 doubles de cada registro de 64 bytes) ---
raw = (B / "geometric_mapping" / "outputs" / "geometry.bin").read_bytes()
n = len(raw) // 64
coords = np.frombuffer(raw, dtype="<f8").reshape(n, 8)[:, :3]  # 8 slots de 8B; 3 primeiros = xyz
print(f"pontos: {n}")

# P_camera = R_md^T (P_map - t_md)
Pc = (coords - t_md) @ R_md  # (v @ R) == R^T v

# --- MEI forward ---
norm = np.linalg.norm(Pc, axis=1)
Xs, Ys, Zs = Pc[:, 0] / norm, Pc[:, 1] / norm, Pc[:, 2] / norm
denom = Zs + xi
with np.errstate(divide="ignore", invalid="ignore"):
    xu = Xs / denom
    yu = Ys / denom
r2 = xu * xu + yu * yu
radial = 1 + k1 * r2 + k2 * r2 * r2
xd = xu * radial + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
yd = yu * radial + p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
u_me = fx * xd + cx
v_me = fy * yd + cy

# --- pixels gravados (só os pontos in-support presentes no samples.csv) ---
SAMPLES = B / "sensor_association" / "debug" / "frames" / "camera_1_image_raw-000000" / "samples.csv"
rec_u: dict[int, float] = {}
rec_v: dict[int, float] = {}
with open(SAMPLES) as f:
    for row in csv.DictReader(f):
        gi = int(row["geometry_index"])
        rec_u[gi] = float(row["prepared_u"])
        rec_v[gi] = float(row["prepared_v"])

idx = np.array(sorted(rec_u))
ru = np.array([rec_u[i] for i in idx])
rv = np.array([rec_v[i] for i in idx])
mu = u_me[idx]
mv = v_me[idx]
du = mu - ru
dv = mv - rv
err = np.sqrt(du * du + dv * dv)
print(f"comparados: {idx.size} pontos in-support")
print(f"erro de reprojeção (px): mediana={np.median(err):.4f}  p99={np.percentile(err,99):.4f}  max={np.max(err):.4f}  média={np.mean(err):.4f}")
print(f"erro > 1px: {int(np.sum(err>1))}  ({100*np.mean(err>1):.3f}%)")
print(f"erro > 0.5px: {int(np.sum(err>0.5))}  ({100*np.mean(err>0.5):.3f}%)")
