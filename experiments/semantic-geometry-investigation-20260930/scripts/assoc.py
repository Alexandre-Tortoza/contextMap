"""Reimplementação independente (numpy) da associação do ContextMap2 para um frame.

Mesma regra do pipeline: projeção MEI, domínio z > (-min(xi,1/xi)+1e-9)*alcance, pixel no
intervalo [-0.5, size-0.5), suporte válido (região válida e fora das exclusões), oclusão por
suporte de profundidade local (célula 4 px, raio 2 células, margem max(0.1 m, 2%) sobre o
alcance do raio) e pertencimento a TODA máscara aceita que contém o pixel. Validada contra
o geometry-support.u32 persistido antes de qualquer variação (calibração, geometria, oclusão).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from common import (
    IMAGE_ID,
    ROOT,
    Trajectory,
    camera_model,
    load_jsonl,
    mei_project,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from p4_calibration_photometric import rotation_correction

CELL_PX, RADIUS_CELLS, MARGIN_M, MARGIN_RATIO = 4, 2, 0.1, 0.02


@dataclass
class Frame:
    t_ns: int
    t_map_cam: np.ndarray       # câmera óptica 'd' no mapa (base)
    valid: np.ndarray
    exclusion: np.ndarray
    masks: dict                 # region_id -> bool (480, 640)
    accepted: list


def load_frame(arm: str = "raw") -> Frame:
    image = next(e for e in sequence_index() if e["observation_id"] == IMAGE_ID)
    t = ts_ns(image["timestamp"])
    t_map_epson, _ = Trajectory(arm).pose(t)
    t_map_cam = t_map_epson @ static_transform("epson", "d", arm)
    valid = np.load(ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-valid.npy").astype(bool)
    ego = np.load(ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-ego.npy").astype(bool)
    import json

    result = json.loads((run_dir(arm) / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    masks, accepted = {}, []
    for region in result["regions"]:
        if not region["is_accepted"]:
            continue
        accepted.append(region)
        packed = np.load(run_dir(arm) / "visual_perception" / region["mask_reference"], allow_pickle=False)
        masks[region["region_id"]] = np.unpackbits(packed, count=480 * 640).reshape(480, 640).astype(bool)
    return Frame(t, t_map_cam, valid, ego, masks, accepted)


def camera_points(points_map: np.ndarray, frame: Frame, correction=(0.0, 0.0, 0.0)) -> np.ndarray:
    t_cam_map = np.linalg.inv(frame.t_map_cam)
    cam = points_map @ t_cam_map[:3, :3].T + t_cam_map[:3, 3]
    if any(correction):
        cam = cam @ rotation_correction(*correction).T
    return cam


def associate(points_map: np.ndarray, frame: Frame, *, correction=(0.0, 0.0, 0.0),
              occlusion: str = "pipeline", cam_points: np.ndarray | None = None) -> dict:
    """Retorna estados por ponto e o conjunto associado por região.

    occlusion: "pipeline" (suporte local 4px/r2), "exact_pixel" (z-buffer por pixel, 1 ponto
    por pixel, como o V1 para um scan local) ou "none".
    """
    cam_model = camera_model()
    cam = camera_points(points_map, frame, correction) if cam_points is None else cam_points
    rng = np.linalg.norm(cam, axis=1)
    xi = cam_model["xi"]
    in_domain = cam[:, 2] > (-min(xi, 1 / xi) + 1e-9) * rng
    uv = np.full((len(cam), 2), np.nan)
    uv_d, _ = mei_project(cam[in_domain], cam_model)
    uv[in_domain] = uv_d
    h, w = frame.valid.shape
    with np.errstate(invalid="ignore"):
        in_image = in_domain & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)
    col = np.zeros(len(cam), dtype=np.int64)
    row = np.zeros(len(cam), dtype=np.int64)
    col[in_image] = np.floor(uv[in_image, 0] + 0.5).astype(np.int64)
    row[in_image] = np.floor(uv[in_image, 1] + 0.5).astype(np.int64)
    supported = np.zeros(len(cam), dtype=bool)
    supported[in_image] = frame.valid[row[in_image], col[in_image]] & ~frame.exclusion[row[in_image], col[in_image]]
    occluded = np.zeros(len(cam), dtype=bool)
    support_depth = np.full(len(cam), np.nan)
    idx = np.flatnonzero(in_image)
    if occlusion == "pipeline":
        cols_c, rows_c = -(-w // CELL_PX), -(-h // CELL_PX)
        edges = uv[idx] + 0.5
        cx = np.minimum((edges[:, 0] // CELL_PX).astype(np.int64), cols_c - 1)
        cy = np.minimum((edges[:, 1] // CELL_PX).astype(np.int64), rows_c - 1)
        grid = np.full(rows_c * cols_c, np.inf)
        np.minimum.at(grid, cy * cols_c + cx, rng[idx])
        grid = grid.reshape(rows_c, cols_c)
        from scipy.ndimage import minimum_filter

        window = minimum_filter(grid, size=2 * RADIUS_CELLS + 1, mode="constant", cval=np.inf)
        sup = window[cy, cx]
        support_depth[idx] = sup
        occluded[idx] = (rng[idx] > sup + np.maximum(MARGIN_M, MARGIN_RATIO * sup)) & supported[idx]
    elif occlusion == "exact_pixel":
        # um ponto por pixel: o mais próximo; os demais do mesmo pixel ficam ocluídos
        order = idx[np.lexsort((rng[idx], row[idx] * w + col[idx]))]
        keys = row[order] * w + col[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = keys[1:] != keys[:-1]
        occluded[order[~first]] = True
        occluded &= supported
    visible = supported & ~occluded
    per_region = {}
    for rid, mask in frame.masks.items():
        inside = np.zeros(len(cam), dtype=bool)
        inside[idx] = mask[row[idx], col[idx]]
        per_region[rid] = np.flatnonzero(inside & visible)
    return {
        "uv": uv, "row": row, "col": col, "range": rng, "cam": cam,
        "in_domain": in_domain, "in_image": in_image, "supported": supported,
        "occluded": occluded, "visible": visible, "support_depth": support_depth,
        "per_region": per_region,
    }


def pipeline_support(arm: str = "raw") -> dict:
    """Conjunto associado por região, como persistido pelo pipeline."""
    base = run_dir(arm) / "sensor_association/outputs"
    support = np.fromfile(base / "geometry-support.u32", dtype="<u4").astype(np.int64)
    out = {}
    for rec in load_jsonl(base / "spatial-observations.jsonl"):
        o, c = rec["support"]["offset"], rec["support"]["count"]
        out[rec["region_id"]] = np.sort(support[o : o + c])
    return out
