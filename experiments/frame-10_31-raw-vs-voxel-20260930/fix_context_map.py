"""Re-gera só o context_map de cada arm com os inputs nos caminhos do bundle auto-contido,
para que o locator da geometria vire '../geometric_mapping' (irmão) e o viewer plote.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
from contextmap.runtime.composition import compose_executors

WS = Path("workspace")
SEQ_ID = "aa0551d92f7b4a7aa3508dbaa1ee8ea3"
CONFIGS = [
    "examples/v0.2.0/canonical-config.toml",
    "experiments/canonical-2-empirical-validation-20260929/config/base.template.toml",
    "experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml",
    "examples/v0.2.0/corridor-02-image-masks.toml",
    "experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml",
]


def _id(run_dir: Path) -> str:
    m = json.loads((run_dir / "manifest.json").read_text())
    for k in ("map_id", "run_id", "context_map_id", "artifact_id"):
        if k in m:
            return str(m[k])
    raise KeyError(run_dir)


def ref(stage: str, contract: str, artifact_id: str, location: str) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage, contract=contract, artifact_id=artifact_id,
        content_hash="sha256:fix", location=location,
    )


def main() -> int:
    effective = resolve_effective_config(files=CONFIGS)
    ex = compose_executors(
        effective, module_available=lambda _m: True, environ={}, code_version="frame-10_31"
    )
    for arm in ("frame-10_31-raw", "frame-10_31-voxel"):
        b = WS / arm
        geo_id = _id(b / "geometric_mapping")
        ent_id = _id(b / "entity_resolution")
        rel_id = _id(b / "spatial_relations")
        # remove o context_map antigo (imutável) para regravar com locator correto
        shutil.rmtree(b / "context_map", ignore_errors=True)
        out = b / "context_map"
        req = StageRequest(
            stage_id="context_map",
            inputs={
                "sequence": (ref("ingestion", "SequenceArtifact", SEQ_ID,
                                 f"{arm}/sequences/corridor-02/{SEQ_ID}"),),
                "geometry": (ref("geometric_mapping", "GeometricMapArtifact", geo_id,
                                 f"{arm}/geometric_mapping"),),
                "entities": (ref("entity_resolution", "EntityResolutionRunArtifact", ent_id,
                                 f"{arm}/entity_resolution"),),
                "relations": (ref("spatial_relations", "SpatialRelationsRunArtifact", rel_id,
                                  f"{arm}/spatial_relations"),),
            },
            components={}, config_digest=f"{arm}-ctx-fix", output_dir=out, workspace=WS,
        )
        r = ex["context_map"].execute(req)
        # confirma o locator
        m = json.loads((out / "manifest.json").read_text())
        import re
        loc = re.search(r'"locator"\s*:\s*"([^"]+)"', json.dumps(m))
        print(f"{arm}: context_map={r.artifact_id}  geo_locator={loc.group(1) if loc else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
