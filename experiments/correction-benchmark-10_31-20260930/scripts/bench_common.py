"""Constantes e utilitários compartilhados do benchmark B1 (frame 10:31, imagem 55).

Tudo aqui é leitura: os runs baseline em ``workspace-old/`` são artefatos imutáveis e nenhuma
função deste módulo escreve neles. Os parâmetros congelados (janela, ego-mask, preset, prompt)
são os de ``experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py``, repetidos
literalmente, nunca reinventados.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "experiments/correction-benchmark-10_31-20260930"
RESULTS = BUNDLE / "results"
MANIFESTS = BUNDLE / "manifests"
ANNOTATIONS = BUNDLE / "annotations"
# Saídas volumosas (artefatos de percepção dos braços) ficam fora do bundle, em outputs/
# (gitignored); o bundle guarda só métricas, tabelas e figuras.
ARM_OUTPUTS = ROOT / "outputs/correction-benchmark-10_31-20260930"

INVESTIGATION = ROOT / "experiments/semantic-geometry-investigation-20260930"
INVESTIGATION_SCRIPTS = INVESTIGATION / "scripts"
RUN_DRIVER_BUNDLE = ROOT / "experiments/frame-10_31-raw-vs-voxel-20260930"

# Runs baseline (E0): configuração atual, produzidos por run_two_arms.py em 2026-09-30.
E0_RUNS = {
    "raw": ROOT / "workspace-old/frame-10_31-raw",
    "voxel": ROOT / "workspace-old/frame-10_31-voxel",
}

SEQUENCE_NAME = "corridor-02"
SEQUENCE_ID = "aa0551d92f7b4a7aa3508dbaa1ee8ea3"
POSE_SEQUENCE_ID = "d422227f15f14d0cb8e071652c5c20b5"
IMAGE_ID = "camera_1_image_raw-000055"
FRAME_INDEX = 55
IMAGE_TIMESTAMP = "1724621714.984856033"
NEAREST_SCAN_ID = "velodyne_points-000022"
CAMERA_CALIBRATION_ID = "camera_1_image_raw-calib"
IMAGE_SIZE = (640, 480)

# Pilha de configuração do E0, na ordem de precedência (a última vence). É a mesma do README de
# frame-10_31-raw-vs-voxel-20260930 mais o overlay region-label.toml, sem o qual as claims não
# teriam o template region-label/v1 que o artefato registra.
CONFIG_STACK = (
    "examples/v0.2.0/canonical-config.toml",
    "experiments/canonical-2-empirical-validation-20260929/config/base.template.toml",
    "experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml",
    "examples/v0.2.0/corridor-02-image-masks.toml",
    "experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml",
)

MISE_PYTHON = str(Path("~/.local/share/mise/installs/python/3.12.14/bin/python3.12").expanduser())
ABLATION_PYTHON = str(Path("~/.cache/contextmap2-ablation/venv/bin/python").expanduser())
ABLATION_PROVIDERS = str(Path("~/.cache/contextmap2-ablation/providers").expanduser())

# O import dos scripts da investigação é feito por caminho: eles não são um pacote.
if str(INVESTIGATION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(INVESTIGATION_SCRIPTS))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return ``sha256:<hex>`` of a file, streamed so large payloads never load at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_bytes(payload: bytes) -> str:
    """Return ``sha256:<hex>`` of an in-memory payload."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def read_json(path: Path) -> Any:
    """Read one JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; a missing file reads as no records."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> Path:
    """Write a JSON document with stable key order (results are diffed between arms)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def code_identity(relevant_untracked: tuple[str, ...] = ()) -> dict[str, Any]:
    """Identity of the code that produced a result: HEAD plus the hash of the local diff.

    Everything is local and uncommitted, so HEAD alone does not identify the code: the sha256
    of ``git diff HEAD`` (tracked changes, staged or not) and the list of untracked files that
    matter for the run complete it. Untracked files are hashed individually.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout

    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    untracked = []
    for entry in relevant_untracked:
        path = ROOT / entry
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if "__pycache__" in file.parts or not file.is_file():
                continue
            untracked.append({"path": str(file.relative_to(ROOT)), "sha256": sha256_file(file)})
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "diff_sha256": sha256_bytes(diff),
        "diff_bytes": len(diff),
        "tracked_files_changed": sorted(git("diff", "HEAD", "--name-only").split()),
        "relevant_untracked_files": untracked,
    }


# Arquivos não rastreados que participam do caminho de execução dos runs (máscaras de imagem
# do runtime, suporte espacial, predicados semânticos) e os drivers de experimento reusados.
RELEVANT_UNTRACKED = (
    "src/contextmap/runtime/image_masks.py",
    "src/contextmap/spatial_relations/semantic_predicates.py",
    "src/contextmap/spatial_support",
    "experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py",
    "experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml",
    "experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml",
    "experiments/semantic-geometry-investigation-20260930/scripts",
)
