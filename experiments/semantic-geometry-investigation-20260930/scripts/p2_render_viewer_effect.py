"""Parte 2: ilustra por que a nuvem "parece mudar" sem que nenhum XYZ mude.

Renderização ortográfica de topo (XY do frame map, janela de 24 m em torno da câmera do
frame 55), reproduzindo as regras de cor do viewer (GeometryChunks.client.vue):
  (a) mapa 1-scan só-geometria (frame-10_28-raw) — o tipo de mapa "sem semântica" em disco;
  (b) mapa do run semântico (81 scans), sem context_map: cor base 0x7dd3fc, opaco;
  (c) mesmo mapa, com cobertura de evidência: pontos sem observação -> 0x333333 a 40%
      de opacidade (UNOBSERVED_RGB/UNOBSERVED_OPACITY), pontos observados opacos.
Os três usam os XYZ autoritativos dos geometry.bin; (b) e (c) usam exatamente os mesmos.
"""
from __future__ import annotations

import numpy as np
from assoc import load_frame
from common import FIGURES, GEOMETRY_DTYPE, RUNS, load_geometry, run_dir
from PIL import Image, ImageDraw, ImageFont

SIZE, SPAN = 800, 24.0


def raster(xy, colors, alpha, center):
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    px = ((xy - center) / SPAN * SIZE + SIZE / 2).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < SIZE) & (px[:, 1] >= 0) & (px[:, 1] < SIZE)
    px, colors, alpha = px[ok], colors[ok], alpha[ok]
    # composição "over" na ordem dos pontos, como pontos transparentes sem depthWrite
    for a in np.unique(alpha):
        sel = alpha == a
        img[SIZE - 1 - px[sel, 1], px[sel, 0]] = img[SIZE - 1 - px[sel, 1], px[sel, 0]] * (1 - a) + colors[sel] * a
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def main() -> None:
    frame = load_frame("raw")
    cam_xy = frame.t_map_cam[:2, 3]
    base = np.array([0x7D, 0xD3, 0xFC]) / 255
    geo = load_geometry("raw")
    xy = geo["map"][:, :2]
    one_scan = np.fromfile(RUNS / "pointcloud-raw-vs-voxel-20260930/frame-10_28-raw/geometric_mapping/outputs/geometry.bin",
                           dtype=GEOMETRY_DTYPE)
    support = np.fromfile(run_dir("raw") / "sensor_association/outputs/geometry-support.u32", dtype="<u4")
    covered = np.zeros(len(geo), bool)
    covered[support] = True
    panels = []
    # (a) mapa só-geometria de 1 scan: centrado no próprio sensor (outro instante e lugar)
    one_center = one_scan["map"][:, :2].mean(0)
    panels.append((raster(one_scan["map"][:, :2], np.tile(base, (len(one_scan), 1)), np.ones(len(one_scan)), one_center),
                   f"(a) frame-10_28-raw: 1 scan, {len(one_scan):,} pts"))
    panels.append((raster(xy, np.tile(base, (len(geo), 1)), np.ones(len(geo)), cam_xy),
                   f"(b) frame-10_31-raw sem cobertura: 81 scans, {len(geo):,} pts"))
    colors = np.where(covered[:, None], base, np.array([0x33, 0x33, 0x33]) / 255)
    alpha = np.where(covered, 1.0, 0.4)
    order = np.argsort(covered)  # observados por cima, como o z-order visual típico
    panels.append((raster(xy[order], colors[order], alpha[order], cam_xy),
                   f"(c) mesmo XYZ, com cobertura: {covered.sum():,} opacos ({covered.mean():.2%})"))
    out = Image.new("RGB", (SIZE * 3 + 20, SIZE + 18), (20, 20, 20))
    d = ImageDraw.Draw(out)
    for i, (im, title) in enumerate(panels):
        out.paste(im, (i * (SIZE + 10), 18))
        d.text((i * (SIZE + 10) + 4, 3), title, fill=(255, 255, 255), font=ImageFont.load_default())
    out.save(FIGURES / "f08_viewer_coverage_effect_topdown.png")
    print("ok", covered.sum(), len(geo))


if __name__ == "__main__":
    main()
