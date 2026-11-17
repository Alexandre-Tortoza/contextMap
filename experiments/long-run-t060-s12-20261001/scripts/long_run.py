"""Run longa por blocos, com orçamento: corridor-02 de 01:00 em diante, stride 12.

Uso (da raiz do repositório, com o ambiente de ``env.sh``):
    python long_run.py --budget-hours 7 --chunk 60

Cada bloco de ``--chunk`` frames (de ``frames.json``, em ordem):
1. ContextRun principal (perfil SOTA, estágio-major, PTv3) num processo;
2. ContextRun da busca de pessoas (SAM3) sobre os mesmos frames, na mesma branch;
3. build de checkpoint, que publica ``workspace/<map_id>/NNNNNN``.

Antes de cada bloco, projeta o tempo pelo último bloco medido e não começa um bloco que
ultrapasse o orçamento. Depois do 1º build, aplica as guardas declaradas no plano (build > 15 min,
mais de 50 mil relações, ou falha de processo): para e reporta. Grava ``results/progress.json``
depois de cada passo, com tempo de parede e pico de GPU (amostrado pelo nvidia-smi a 200 ms).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
PREVIOUS = "experiments/run-frame-10_31-cli-20260930"
WORKSPACE = "workspace/run-t060-s12"
MAPS_ROOT = "workspace"
MAP_ID = "corridor02-t060-s12-sam2t-dinov3-qwen8b-top8-ptv3-sam3human-vox10"
BUILD_LIMIT_S = 15 * 60
RELATION_LIMIT = 50_000

STACK = [
    "-c", "examples/v0.2.0/canonical-config.toml",
    "-c", "experiments/canonical-2-empirical-validation-20260929/config/base.template.toml",
    "-c", "experiments/frame-10_31-raw-vs-voxel-20260930/providers.toml",
    "-c", "examples/v0.2.0/corridor-02-image-masks.toml",
    "-c", "experiments/frame-10_31-raw-vs-voxel-20260930/region-label.toml",
    "-c", f"{PREVIOUS}/run.toml",
]  # fmt: skip
SOTA = ["-c", f"{BUNDLE.relative_to(ROOT)}/config/sota.toml", "-c", f"{BUNDLE.relative_to(ROOT)}/config/ptv3.json"]
HUMAN = ["-c", f"{BUNDLE.relative_to(ROOT)}/config/human.toml"]
NO_SCENE = ["--set", "components.semantic_interpretation.interpreter.qwen.prompt_policy.scene=null"]
NO_MODELS = ["--set", "resources.providers={}"]


class GpuSampler:
    """Amostra a memória usada da GPU a cada 200 ms enquanto um processo roda."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=False, timeout=5,
                ).stdout.strip()  # fmt: skip
                if out:
                    self.samples.append(int(out.splitlines()[0]))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(0.2)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()


def run_step(name: str, argv: list[str], log_dir: Path) -> dict[str, Any]:
    """Roda um comando do CLI, guarda stdout/stderr e devolve tempo, pico de GPU e saída."""
    started = time.monotonic()
    with GpuSampler() as gpu, open(log_dir / f"{name}.err", "w") as err:
        proc = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True, check=False)
    (log_dir / f"{name}.json").write_text(proc.stdout)
    return {
        "step": name,
        "exit": proc.returncode,
        "wall_s": round(time.monotonic() - started, 2),
        "gpu_peak_mib": max(gpu.samples, default=-1),
    }


def selection(frames: list[dict[str, Any]]) -> list[str]:
    ids = [frame["observation_id"] for frame in frames]
    return ["--set", "inputs.observation_selection=" + json.dumps({"kind": "explicit_ids", "observation_ids": ids})]


def map_counts(build_json: Path) -> dict[str, Any]:
    """Entidades e relações do mapa publicado pelo build (lidas do manifest do mapa)."""
    try:
        record = json.loads(build_json.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    text = json.dumps(record)
    maps = sorted((ROOT / MAPS_ROOT / MAP_ID).glob("[0-9]" * 6))
    if not maps:
        return {}
    manifest = json.loads((maps[-1] / "manifest.json").read_text())
    return {
        "map": str(maps[-1].relative_to(ROOT)),
        "entity_count": manifest.get("entity_count"),
        "relation_count": manifest.get("relation_count"),
        "build_record_bytes": len(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-hours", type=float, default=7.0)
    parser.add_argument("--chunk", type=int, default=60)
    parser.add_argument("--start-chunk", type=int, default=0)
    args = parser.parse_args()

    python = os.path.expanduser("~/.cache/contextmap2-ablation/venv/bin/python")
    code = (BUNDLE / "code-identity.txt").read_text().strip()
    frames = json.loads((BUNDLE / "frames.json").read_text())["frames"]
    chunks = [frames[i : i + args.chunk] for i in range(0, len(frames), args.chunk)]
    log_dir = BUNDLE / "logs" / "chunks"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = BUNDLE / "results" / "progress.json"
    progress: dict[str, Any] = {
        "map_id": MAP_ID,
        "budget_s": args.budget_hours * 3600,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chunks": [],
        "status": "running",
    }
    started = time.monotonic()
    base = [python, "-m", "contextmap"]
    common = ["--workspace", WORKSPACE, "--maps-root", MAPS_ROOT, "--code-identity", code, "--json"]

    def save() -> None:
        progress["elapsed_s"] = round(time.monotonic() - started, 1)
        progress_path.write_text(json.dumps(progress, indent=2))

    last_chunk_s: float | None = None
    for index, chunk in enumerate(chunks):
        if index < args.start_chunk:
            continue
        elapsed = time.monotonic() - started
        if last_chunk_s is not None and elapsed + last_chunk_s > progress["budget_s"]:
            progress["status"] = "budget_reached"
            progress["stopped_before_chunk"] = index
            save()
            return 0
        tag = f"chunk-{index:02d}"
        entry: dict[str, Any] = {
            "chunk": index,
            "frames": len(chunk),
            "t_start_s": chunk[0]["t_s"],
            "t_end_s": chunk[-1]["t_s"],
            "steps": [],
        }
        progress["chunks"].append(entry)
        chunk_started = time.monotonic()
        steps = [
            (f"{tag}-main", [*base, "context", "run", "--branch", MAP_ID, *STACK, *SOTA, *NO_SCENE, *selection(chunk), *common]),
            (f"{tag}-human", [*base, "context", "run", "--branch", MAP_ID, *STACK, *SOTA, *HUMAN, *NO_SCENE, *selection(chunk), *common]),
            (f"{tag}-build", [*base, "context", "build", "--branch", MAP_ID, *STACK, *SOTA, *NO_MODELS, *NO_SCENE, *common]),
        ]  # fmt: skip
        for name, argv in steps:
            result = run_step(name, argv, log_dir)
            entry["steps"].append(result)
            save()
            if result["exit"] != 0:
                progress["status"] = f"failed_at_{name}"
                save()
                return 1
        entry.update(map_counts(log_dir / f"{tag}-build.json"))
        last_chunk_s = time.monotonic() - chunk_started
        entry["chunk_wall_s"] = round(last_chunk_s, 1)
        save()
        build_s = entry["steps"][-1]["wall_s"]
        relations = entry.get("relation_count") or 0
        if index == args.start_chunk and (build_s > BUILD_LIMIT_S or relations > RELATION_LIMIT):
            progress["status"] = "guard_tripped"
            progress["guard"] = {"build_s": build_s, "relations": relations}
            save()
            return 2
    progress["status"] = "completed"
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
