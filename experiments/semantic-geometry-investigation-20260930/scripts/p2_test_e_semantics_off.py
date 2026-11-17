"""Teste E: semantics=false vs semantics=true, ao nível do artefato geométrico.

Reconstrói o GeometricMapArtifact (raw e voxel) a partir das MESMAS entradas do run
frame-10_31 (sequência + trajetória), sem executar nenhum estágio semântico, num workspace
de scratch, e compara o geometry.bin byte a byte com o que o context_map semântico referencia.
Se forem idênticos, a semântica não tem como ter alterado XYZ: o mapa geométrico é produzido
antes e só é referenciado depois.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from common import ROOT, SEQ_ID, dump, run_dir

SCRATCH = Path(os.environ["SCRATCH"]) / "test-e-ws"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def main() -> None:
    from contextmap.geometric_mapping import ScanVoxelPolicy
    from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
    from contextmap.runtime.composition import compose
    from contextmap.runtime.executors import GeometricMappingExecutor

    SCRATCH.mkdir(parents=True, exist_ok=True)
    # layout esperado pelo executor: <ws>/sequences/... e <ws>/pipeline/state_estimation
    (SCRATCH / "pipeline").mkdir(exist_ok=True)
    for link, target in (
        (SCRATCH / "sequences", run_dir("raw") / "sequences"),
        (SCRATCH / "pipeline/state_estimation", run_dir("raw") / "state_estimation"),
    ):
        if not link.exists():
            link.symlink_to(target)
    configs = [
        ROOT / "examples/v0.2.0/canonical-config.toml",
        ROOT / "experiments/canonical-2-empirical-validation-20260929/config/base.template.toml",
        ROOT / "experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml",
        ROOT / "examples/v0.2.0/corridor-02-image-masks.toml",
    ]
    effective = resolve_effective_config(files=[str(c) for c in configs])
    g = compose(effective, stages=["geometric_mapping"])
    state_id = json.loads((run_dir("raw") / "state_estimation/manifest.json").read_text())["run_id"]

    def ref(stage, contract, artifact_id, location):
        return ArtifactRef(stage_id=stage, contract=contract, artifact_id=artifact_id,
                           content_hash="sha256:driven", location=location)

    seq_ref = ref("ingestion", "SequenceArtifact", SEQ_ID, f"sequences/corridor-02/{SEQ_ID}")
    state_ref = ref("state_estimation", "StateEstimationRunArtifact", state_id, "pipeline/state_estimation")
    results = {}
    for arm, aggregation in (("raw", None), ("voxel", ScanVoxelPolicy(cell_m=0.10))):
        out = SCRATCH / f"geometric_mapping_{arm}"
        if not (out / "manifest.json").exists():
            executor = GeometricMappingExecutor(
                pose_lookup=g.geometric_mapping_pose_lookup,
                motion_correction=g.motion_correction,
                aggregation=aggregation,
                code_version="frame-10_31",
            )
            executor.execute(StageRequest(
                stage_id="geometric_mapping",
                inputs={"sequence": (seq_ref,), "trajectory": (state_ref,)},
                components={}, config_digest=f"geo-{arm}", output_dir=out, workspace=SCRATCH,
            ))
        rebuilt = sha256(out / "outputs/geometry.bin")
        used = sha256(run_dir(arm) / "geometric_mapping/outputs/geometry.bin")
        man_new = json.loads((out / "manifest.json").read_text())
        man_old = json.loads((run_dir(arm) / "geometric_mapping/manifest.json").read_text())
        results[arm] = {
            "semantics_off_geometry_sha256": rebuilt,
            "semantics_on_geometry_sha256": used,
            "identical": rebuilt == used,
            "point_count_off": man_new["point_count"],
            "point_count_on": man_old["point_count"],
            "map_id_off": man_new["map_id"],
            "map_id_on": man_old["map_id"],
            "scan_count_off": man_new["scan_count"],
            "rejected_scan_count_off": man_new["rejected_scan_count"],
        }
        print(arm, results[arm], flush=True)
    dump("p2_test_e_semantics_off_vs_on.json", results)


if __name__ == "__main__":
    sys.exit(main())
