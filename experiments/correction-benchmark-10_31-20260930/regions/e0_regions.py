"""Regiões aceitas do E0 (frame 10:31, braço raw), lidas só pelas APIs públicas.

E0 = `workspace-old/frame-10_31-raw`, o run da investigação
`experiments/semantic-geometry-investigation-20260930`. Nada aqui escreve no run: o
`PerceptionRunReader` lê o índice, o `MaskStoreReader` verifica o hash de cada máscara.
A área útil é a mesma da investigação (`p6_mask_quality.py` via `assoc.load_frame`):
região válida menos a exclusão do ego, dos `.npy` do overlay de máscaras de imagem.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextmap.visual_perception import InlineMask, PerceptionRunReader

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "workspace-old/frame-10_31-raw"
IMAGE_ID = "camera_1_image_raw-000055"
ASSETS = ROOT / "examples/v0.2.0/assets/corridor-02"
INVESTIGATION = ROOT / "experiments/semantic-geometry-investigation-20260930"


@dataclass(frozen=True)
class E0Regions:
    """Máscaras aceitas do frame 55, com o que é preciso para medir e rastrear."""

    masks: dict[str, InlineMask]
    proposal_count: int
    usable_area: InlineMask
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


MEASURED_MODULES = (
    ROOT / "src/contextmap/evaluation/region_structure.py",
    ROOT / "src/contextmap/evaluation/association_conflicts.py",
)


def code_identity() -> dict[str, object]:
    """HEAD, hash do diff rastreado e hash dos módulos de métrica (ainda não rastreados).

    A árvore tem WIP de outras sessões em paralelo, então o hash do diff muda entre execuções;
    o que define as métricas é o hash dos dois módulos novos.
    """
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "HEAD"], capture_output=True, check=True
    ).stdout
    return {
        "head": head,
        "tracked_worktree_diff_sha256": "sha256:" + hashlib.sha256(diff).hexdigest(),
        "metric_modules": {str(p.relative_to(ROOT)): _sha256(p) for p in MEASURED_MODULES},
    }


def load_e0_regions() -> E0Regions:
    """Carrega as regiões aceitas, a contagem de propostas e a área útil do frame 55."""
    reader = PerceptionRunReader(RUN / "visual_perception")
    result = next(r for r in reader.list_results() if r.source_observation_id == IMAGE_ID)
    store = reader.mask_store()
    masks = {
        str(region.region_id): region.mask
        if region.mask is not None
        else store.load(result.source_observation_id, region.region_id)
        for region in result.regions
        if region.is_accepted
    }
    audit = next(
        a for a in reader.iter_region_discovery_audits() if a.source_observation_id == IMAGE_ID
    )
    proposal_count = sum(diagnostic.proposal_count for diagnostic in audit.diagnostics)
    valid_path, ego_path = ASSETS / "camera_1-valid.npy", ASSETS / "camera_1-ego.npy"
    valid = np.load(valid_path).astype(bool)
    ego = np.load(ego_path).astype(bool)
    manifest = reader.manifest
    return E0Regions(
        masks=masks,
        proposal_count=proposal_count,
        usable_area=InlineMask(valid & ~ego),
        provenance={
            "run": str(RUN.relative_to(ROOT)),
            "image": IMAGE_ID,
            "perception_run_id": manifest.run_id,
            "perception_configuration_digest": manifest.configuration_digest,
            "region_discovery_backend": {
                "backend_id": audit.backend.backend_id,
                "model": audit.backend.model,
                "version": audit.backend.version,
                "configuration_fingerprint": audit.backend.configuration_fingerprint,
            },
            "usable_area": {
                "valid": {"path": str(valid_path.relative_to(ROOT)), "sha256": _sha256(valid_path)},
                "ego_exclusion": {
                    "path": str(ego_path.relative_to(ROOT)),
                    "sha256": _sha256(ego_path),
                },
            },
        },
    )
