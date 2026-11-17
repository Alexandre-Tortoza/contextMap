"""Exporta o mapa raw do E0 perto da câmera do frame 55 como PLY, com o índice de cada ponto.

Apoio à anotação das correspondências 2D<->3D: abrir o PLY num visualizador de nuvem
(p.ex. CloudCompare) e escolher o ponto 3D (quina de porta/janela) pela GEOMETRIA, sem usar a
projeção do pipeline — a correspondência precisa ser independente da calibração que ela vai
avaliar. O campo escalar ``index`` é o índice do ponto no ``geometry.bin`` (``geometry_index``
da anotação); as coordenadas são as do frame ``map``, em metros.

Saída (gitignored):
``outputs/correction-benchmark-10_31-20260930/annotation/e0-raw-map-near-frame55.ply``.

    PYTHONPATH=src $MISE_PY \
      experiments/correction-benchmark-10_31-20260930/scripts/export_map_ply.py [--radius-m 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import ARM_OUTPUTS, E0_RUNS, SEQUENCE_ID, read_json
from common import GEOMETRY_DTYPE
from extract_metrics import _image_time_ns, _pose_at


def main() -> int:
    """Write the cropped, indexed PLY."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius-m", type=float, default=15.0)
    args = parser.parse_args()
    run = E0_RUNS["raw"]
    points = np.fromfile(run / "geometric_mapping/outputs/geometry.bin", dtype=GEOMETRY_DTYPE)
    sequence = run / f"sequences/corridor-02/{SEQUENCE_ID}"
    camera_position = _pose_at(run, _image_time_ns(sequence))[:3, 3]
    keep = np.flatnonzero(np.linalg.norm(points["map"] - camera_position, axis=1) <= args.radius_m)
    out = ARM_OUTPUTS / "annotation/e0-raw-map-near-frame55.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = np.zeros(len(keep), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("index", "<u4")])
    record["x"], record["y"], record["z"] = points["map"][keep].T
    record["index"] = keep.astype(np.uint32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment map_id {read_json(run / 'geometric_mapping/manifest.json')['map_id']}\n"
        f"comment frame map; points within {args.radius_m} m of the camera at "
        "camera_1_image_raw-000055\n"
        f"element vertex {len(keep)}\nproperty double x\nproperty double y\nproperty double z\n"
        "property uint index\nend_header\n"
    )
    with out.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(record.tobytes())
    print(out, len(keep), "points; camera at", camera_position.round(3).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
