"""Gera as imagens de apoio da anotação fraca do frame 55 em ``annotations/``.

* ``frame-55.png``: a imagem RGB do frame (payload bgr8 da sequência do E0, 640x480). A
  preparação de imagem do run não redimensiona nem recorta (escala 1,0; só declara região
  válida e exclusão), então os pixels desta PNG são os da imagem preparada.
* ``frame-55-grid.png``: a mesma imagem com grade a cada 50 px e a região inutilizável
  (fora da vinheta válida ou sobre o rig) escurecida, para ler coordenadas à mão.

    PYTHONPATH=src $MISE_PY \
      experiments/correction-benchmark-10_31-20260930/scripts/prepare_annotation_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import ANNOTATIONS, sha256_file
from c4_evaluate import load_rgb
from extract_metrics import _usable_area

GRID_PX = 50


def main() -> int:
    """Write the frame image and its gridded companion."""
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb()
    Image.fromarray(rgb).save(ANNOTATIONS / "frame-55.png")
    shaded = rgb.astype(float)
    shaded[~_usable_area()] *= 0.35
    image = Image.fromarray(shaded.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for x in range(0, 640, GRID_PX):
        draw.line([(x, 0), (x, 479)], fill=(0, 255, 255), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 0), font=font)
    for y in range(0, 480, GRID_PX):
        draw.line([(0, y), (639, y)], fill=(0, 255, 255), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 0), font=font)
    image.save(ANNOTATIONS / "frame-55-grid.png")
    print("frame-55.png", sha256_file(ANNOTATIONS / "frame-55.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
