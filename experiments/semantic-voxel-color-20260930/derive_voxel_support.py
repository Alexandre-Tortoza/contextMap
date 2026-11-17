"""Fase 3: gera a agregação voxel de um mapa real, para alimentar o viewer.

Fecha a metade que ainda não tem comando: o viewer (cliente+servidor) consome um
VoxelAggregationArtifact (fonte voxel) e um SemanticSpatialSupportArtifact (cor por entidade
por aggregate_index). O suporte agora sai de `contextmap derive-spatial-support`; aqui fica
só a agregação, que é de geometric_mapping e ainda não tem caminho de produção próprio.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    InterScanVoxelPolicy,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    VoxelGridSpec,
)

BUNDLE = Path("workspace/frame-10_29-full-run")
RAW = BUNDLE / "geometric_mapping"
CONTEXT_MAP = BUNDLE / "context_map"
OUT = Path("outputs/semantic-voxel-color-20260930")
CELL_M = 0.10

# frame_id e origem do mapa a partir do artefato bruto
with GeometricMapArtifactReader(RAW) as raw:
    frame = raw.geometry().geometric_map.frame_id
    raw_map_id = str(raw.manifest.map_id)
print(f"raw map: {raw_map_id}  frame={frame}")

voxel_dir = OUT / "voxel" / f"cell-0_{int(CELL_M * 100):02d}"
support_dir = OUT / "semantic-support"
if voxel_dir.exists():
    print(f"ERRO: {voxel_dir} já existe; remova antes.", file=sys.stderr)
    sys.exit(1)

voxel_run = VoxelAggregationRunId(f"voxel-cell-0_{int(CELL_M * 100):02d}")
vmanifest = VoxelAggregationArtifactWriter(output_dir=voxel_dir, run_id=voxel_run).finalize(
    source_dir=RAW,
    policy=InterScanVoxelPolicy(
        grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=CELL_M)
    ),
    code_version="fase3-semantic-voxel-color",
)
print(
    f"voxel aggregation: run_id={voxel_run}  voxels={vmanifest.aggregate_count}  "
    f"source_map_id={vmanifest.source_map_id}"
)

command = [
    "contextmap",
    "derive-spatial-support",
    "--context-map",
    str(CONTEXT_MAP),
    "--voxel-aggregation",
    str(voxel_dir),
    "--raw-geometry",
    str(RAW),
    "--output",
    str(support_dir),
    "--run-id",
    "support-frame-10_29",
    "--code-version",
    "fase3-semantic-voxel-color",
]
print("\n=== derive o suporte semântico ===")
print(shlex.join(command))

print("\n=== viewer env (aponte e reinicie o viewer) ===")
root = Path.cwd()
print(f"NUXT_CONTEXT_MAP_ARTIFACTS_ROOT={root / 'workspace'}")
print(f"NUXT_VOXEL_ARTIFACTS_ROOT={root / voxel_dir}")
print(f"NUXT_SPATIAL_SUPPORT_ARTIFACTS_ROOT={root / support_dir}")
