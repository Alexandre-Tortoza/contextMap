"""Parte 4/6: debug visual do frame 55 — projeção, máscaras e pontos por máscara.

Gera PNGs em figures/: RGB; scan local projetado (base vs correção do V1) colorido por
alcance; pontos visíveis do pipeline (mapa acumulado); máscaras numeradas; mapa de
sobreposição de máscaras; e, por região, contorno da máscara com os pontos associados
sob as duas calibrações.
"""
from __future__ import annotations

import json

import numpy as np
from assoc import associate, load_frame
from common import FIGURES, load_geometry, load_rgb, run_dir
from p5_association_ablation import CALIBRATIONS, local_scan_points
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FIGURES.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.load_default()


def turbo(t: np.ndarray) -> np.ndarray:
    """Colormap simples (azul perto -> vermelho longe), t em [0,1]."""
    t = np.clip(t, 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def draw_points(rgb, uv, mask, colors, radius=1, dim=0.55):
    img = (rgb.astype(float) * dim).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for (u, v), c in zip(uv[mask], colors[mask] if colors.ndim == 2 else [colors] * int(mask.sum())):
        d.ellipse([u - radius, v - radius, u + radius, v + radius], fill=tuple(int(x) for x in c))
    return im


def side_by_side(images, titles):
    w, h = images[0].size
    out = Image.new("RGB", (w * len(images), h + 16), (0, 0, 0))
    d = ImageDraw.Draw(out)
    for i, (im, t) in enumerate(zip(images, titles)):
        out.paste(im, (i * w, 16))
        d.text((i * w + 4, 2), t, fill=(255, 255, 255), font=FONT)
    return out


def main() -> None:
    rgb = load_rgb()
    Image.fromarray(rgb).save(FIGURES / "f01_frame55_rgb.png")
    frame = load_frame("raw")
    result = json.loads((run_dir("raw") / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    labels = {c["region_id"]: c["hypothesis"] for c in result["claims"] if c.get("role") == "primary" and c.get("region_id")}

    # F2: scan local (deskew) — base vs correção V1, z-buffer por pixel, cor = alcance
    scan_pts, scan_id, _ = local_scan_points()
    ims = []
    for name in ("base", "v1_correction"):
        res = associate(scan_pts, frame, correction=CALIBRATIONS[name], occlusion="exact_pixel")
        col = turbo(np.clip((res["range"] - 0.5) / 8.0, 0, 1))
        ims.append(draw_points(rgb, res["uv"], res["visible"], col, radius=1))
    side_by_side(ims, [f"{scan_id} base extrinsic", f"{scan_id} V1 correction (yaw 4.25, pitch -2.25)"]).save(
        FIGURES / "f02_local_scan_projection_base_vs_corrected.png")
    # recortes 2x em áreas de borda (porta/janela à direita, batente à esquerda)
    crops = [(380, 60, 640, 300), (120, 60, 380, 300), (200, 250, 460, 470)]
    for k, box in enumerate(crops):
        side_by_side([im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST) for im in ims],
                     ["base", "V1 correction"]).save(FIGURES / f"f02_crop{k + 1}_base_vs_corrected.png")

    # F3: pontos visíveis do pipeline (mapa acumulado, regra de oclusão do pipeline)
    geo = load_geometry("raw")
    ims = []
    for name in ("base", "v1_correction"):
        res = associate(geo["map"], frame, correction=CALIBRATIONS[name])
        col = turbo(np.clip((res["range"] - 0.5) / 8.0, 0, 1))
        ims.append(draw_points(rgb, res["uv"], res["visible"], col, radius=1))
    side_by_side(ims, ["pipeline visible (81-scan map) base", "pipeline visible, V1 correction"]).save(
        FIGURES / "f03_pipeline_visible_base_vs_corrected.png")

    # F4: máscaras numeradas + sobreposição
    rng = np.random.default_rng(3)
    order = sorted(frame.masks, key=lambda r: -frame.masks[r].sum())
    canvas = (rgb.astype(float) * 0.35)
    overlap = np.zeros(rgb.shape[:2], dtype=np.int32)
    for rid in order:
        m = frame.masks[rid]
        overlap += m
        color = rng.integers(60, 255, 3)
        canvas[m] = canvas[m] * 0.45 + color * 0.55
    im = Image.fromarray(canvas.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for rid in order:
        m = frame.masks[rid]
        edge = m & ~ndimage.binary_erosion(m)
        ys, xs = np.nonzero(edge)
        for x, y in zip(xs[::3], ys[::3]):
            d.point((int(x), int(y)), fill=(255, 255, 255))
        cy, cx = ndimage.center_of_mass(m)
        d.text((cx - 8, cy - 5), rid[-4:], fill=(255, 255, 0), font=FONT)
    im.save(FIGURES / "f04_masks_numbered.png")
    heat = turbo(np.clip(overlap / max(overlap.max(), 1), 0, 1).reshape(-1)).reshape(480, 640, 3)
    heat[overlap == 0] = (rgb[overlap == 0] * 0.3).astype(np.uint8)
    im = Image.fromarray(heat)
    ImageDraw.Draw(im).text((4, 4), f"mask overlap count (max {overlap.max()})", fill=(255, 255, 255), font=FONT)
    im.save(FIGURES / "f05_mask_overlap_count.png")

    # F6: por região — contorno + pontos associados (scan local, z-buffer por pixel)
    res_b = associate(scan_pts, frame, correction=CALIBRATIONS["base"], occlusion="exact_pixel")
    res_c = associate(scan_pts, frame, correction=CALIBRATIONS["v1_correction"], occlusion="exact_pixel")
    for rid in ("region-0020", "region-0062", "region-0070", "region-0006", "region-0017", "region-0035"):
        m = frame.masks[rid]
        ys, xs = np.nonzero(m)
        box = (max(xs.min() - 30, 0), max(ys.min() - 30, 0), min(xs.max() + 30, 639), min(ys.max() + 30, 479))
        panels = []
        for res, color in ((res_b, (255, 60, 60)), (res_c, (60, 255, 60))):
            canvas = (rgb.astype(float) * 0.6).astype(np.uint8)
            edge = m & ~ndimage.binary_erosion(m, iterations=1)
            canvas[edge] = (255, 255, 0)
            im = Image.fromarray(canvas)
            dd = ImageDraw.Draw(im)
            vis = res["visible"]
            for (u, v), inside in zip(res["uv"][vis], np.isin(np.flatnonzero(vis), res["per_region"][rid])):
                dd.point((u, v), fill=color if inside else (90, 90, 200))
            crop = im.crop(box)
            scale = max(1, int(300 / max(crop.size)))
            panels.append(crop.resize((crop.size[0] * scale, crop.size[1] * scale), Image.NEAREST))
        side_by_side(panels, [f"{rid} '{labels.get(rid)}' base", "V1 correction"]).save(FIGURES / f"f06_{rid}_points_base_vs_corrected.png")
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
