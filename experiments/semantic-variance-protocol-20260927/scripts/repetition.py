"""One repetition of one arm of the #580 variance protocol, in its own fresh process.

The repetition is one real execution of the canonical perception path through the runtime's
public entry point (:class:`contextmap.runtime.Runtime`), over the frozen 20-frame sequence the
orchestrator derived: the same composition ``contextmap stage visual_perception`` performs, with
the model runtimes the configuration's ``resources.providers`` declares. Perception spans two
executed stages since the region capability was extracted: ``region_perception`` discovers the
regions and ``visual_perception`` describes them, so requesting the measured stage runs both and
the record names both artifacts.

Arm ``B`` differs from arm ``A`` only by the determinism flags of the issue, applied here before
any model is loaded: fixed seeds (Python, NumPy, torch, CUDA), and
``torch.use_deterministic_algorithms(True)``. ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` has to be in
the environment before CUDA starts, so the orchestrator sets it on the command line and this
process only verifies it. Arm ``A`` sets nothing and refuses to start if the variable leaked in.

A fresh process per repetition is the point of the experiment: the baseline divergence of
run0005 was measured between two independent executions, while repeated requests of an earlier
validation (``src/contextmap/evaluation/docs/semantic-interpretation.md``), with no record of
separate processes, returned identical text.

Usage:
    python repetition.py --arm A --repetition 1 --workspace DIR --config FILE [--config FILE]
        --sequence-location inputs/sequence --dataset corridor-02 --code-identity SHA
        --record FILE [--seed 0]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from contextmap.ingestion import SequenceArtifactReader
from contextmap.runtime import ArtifactRef, Runtime
from contextmap.runtime.executors import inventory_digest

ARMS = ("A", "B")
DETERMINISTIC_ARM = "B"
CUBLAS_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
"""The cuBLAS workspace setting PyTorch documents for deterministic cuBLAS kernels."""

STAGE = "semantic_interpretation"
"""The measured stage: the perception run the variance report compares."""
REGION_STAGE = "region_perception"
"""The region run the measured stage consumes; it is executed in the same repetition."""
RECORDED_VARIABLES = (CUBLAS_VARIABLE, "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")
PACKAGES = (
    "contextmap",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sam2",
    "numpy",
    "Pillow",
)


class RepetitionError(Exception):
    """Raised when a repetition would not run under the conditions its arm declares."""


def apply_determinism(
    arm: str,
    *,
    seed: int,
    environ: Mapping[str, str],
    torch: Any,
    python_random: Any,
    numpy_random: Any,
) -> dict[str, Any]:
    """Apply the arm's determinism flags before any model is loaded, and say what was applied.

    Args:
        arm: ``"A"`` (canonical configuration as is) or ``"B"`` (A plus the flags).
        seed: The seed arm ``B`` fixes everywhere; ignored by arm ``A``.
        environ: The process environment, to verify ``CUBLAS_WORKSPACE_CONFIG``.
        torch: The ``torch`` module, or ``None`` when it is not installed (arm ``A`` only).
        python_random: The ``random`` module.
        numpy_random: ``numpy.random``.

    Returns:
        ``seed``, ``use_deterministic_algorithms`` and ``cublas_workspace_config`` as applied.

    Raises:
        RepetitionError: If the arm is unknown, arm ``A`` finds ``CUBLAS_WORKSPACE_CONFIG`` set,
            or arm ``B`` finds it missing or different, or has no ``torch``.
    """
    if arm not in ARMS:
        raise RepetitionError(f"unknown arm {arm!r}: the protocol has arms {ARMS}")
    configured = environ.get(CUBLAS_VARIABLE)
    if arm != DETERMINISTIC_ARM:
        if configured is not None:
            raise RepetitionError(
                f"arm A runs the canonical configuration without determinism flags, but "
                f"{CUBLAS_VARIABLE}={configured!r} is set in its environment"
            )
        return {
            "seed": None,
            "use_deterministic_algorithms": False,
            "cublas_workspace_config": None,
        }
    if configured != CUBLAS_WORKSPACE_CONFIG:
        raise RepetitionError(
            f"arm B needs {CUBLAS_VARIABLE}={CUBLAS_WORKSPACE_CONFIG} in its environment before "
            f"CUDA starts, got {configured!r}"
        )
    if torch is None:
        raise RepetitionError("arm B fixes torch's determinism flags, but torch is not installed")
    python_random.seed(seed)
    numpy_random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": configured,
    }


def _stage_artifact(record: Any, stage_id: str) -> dict[str, Any] | None:
    """Return the identity and location of what one stage published, or ``None`` if it did not."""
    stage = next((item for item in record.stages if item.stage_id == stage_id), None)
    output = None if stage is None else stage.output
    if output is None:
        return None
    return {
        "artifact_id": output["artifact_id"],
        "content_hash": output["content_hash"],
        "location": output["location"],
    }


def run_repetition(
    *,
    runtime: Runtime,
    config_files: Sequence[Path],
    sequence_location: str,
    dataset: str,
    code_identity: str,
) -> dict[str, Any]:
    """Run the perception path once, as a new run of the runtime's workspace.

    The frozen sequence is the only input provided by reference; ``region_perception`` is the
    one upstream stage the measured stage needs, and the runtime executes it in the same run.
    Nothing is reused, so every repetition is a real execution in its own ``run-NNNN`` directory
    and both artifacts it publishes are named in the record.

    Args:
        runtime: The runtime, bound to the experiment's workspace.
        config_files: The scenario's backend selection and the maintainer's parameter layer.
        sequence_location: Where the frozen sequence lives, relative to the workspace.
        dataset: ``inputs.sequence``: the dataset directory runs are created under.
        code_identity: The code identity the run records.

    Returns:
        What the run record says, with every location relative to the workspace.

    Raises:
        RepetitionError: If the runtime has no workspace.
    """
    status = runtime.status()
    if status.workspace is None:
        raise RepetitionError("a repetition persists its run: the runtime needs a workspace")
    workspace = Path(status.workspace)
    config = runtime.resolve_config(files=config_files, overrides=[f"inputs.sequence={dataset}"])
    manifest = SequenceArtifactReader(workspace / sequence_location).manifest
    sequence = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=str(manifest.artifact_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location=sequence_location,
    )
    result = runtime.run(
        config, targets=[STAGE], provided={"ingestion": sequence}, code_identity=code_identity
    )
    record = result.record
    stage = next((item for item in record.stages if item.stage_id == STAGE), None)
    return {
        "status": record.status,
        "run": Path(record.directory).relative_to(workspace).as_posix(),
        "config_digest": record.config_digest,
        "code_identity": record.code_identity,
        "sequence": {
            "artifact_id": sequence.artifact_id,
            "content_hash": sequence.content_hash,
            "location": sequence_location,
        },
        # O run de regiões é a entrada a montante do run medido: sem a identidade dele, a
        # comparação entre repetições não diria sobre quais regiões as claims foram feitas.
        "regions": _stage_artifact(record, REGION_STAGE),
        "semantic_interpretation": _stage_artifact(record, STAGE),
        "stage_elapsed_s": None if stage is None else stage.elapsed_s,
        "failure": None if record.failure is None else dict(record.failure),
        "blocked_problems": [str(problem) for problem in record.blocked_problems],
        "environment": dict(record.environment),
    }


def hardware(torch: Any) -> dict[str, Any]:
    """Describe the GPU, driver and torch build the repetition ran on.

    Nothing here may stop the repetition: a missing ``nvidia-smi`` is recorded as missing.
    """
    executable = shutil.which("nvidia-smi")
    nvidia_smi: dict[str, Any] | None = None
    if executable is not None:
        query = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total,pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        nvidia_smi = {
            "returncode": query.returncode,
            "gpus": query.stdout.strip().splitlines(),
            "stderr": query.stderr.strip()[-500:],
        }
    torch_record: dict[str, Any] | None = None
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        torch_record = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "device": torch.cuda.get_device_name(0) if cuda else None,
            "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }
    return {"nvidia_smi": nvidia_smi, "torch": torch_record}


def packages() -> dict[str, str | None]:
    """Return the installed version of every package the canonical stage depends on."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run one repetition and write its record; exit non-zero unless the run completed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--sequence-location", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--code-identity", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args(argv)
    if options.record.exists():
        raise RepetitionError(f"{options.record} already exists: a record is never overwritten")

    torch = _optional_module("torch")
    numpy = importlib.import_module("numpy")
    # As flags entram antes de qualquer composição: é a composição que carrega os modelos.
    applied = apply_determinism(
        options.arm,
        seed=options.seed,
        environ=os.environ,
        torch=torch,
        python_random=random,
        numpy_random=numpy.random,
    )
    started = time.perf_counter()
    record = run_repetition(
        runtime=Runtime(workspace=options.workspace),
        config_files=options.config,
        sequence_location=options.sequence_location,
        dataset=options.dataset,
        code_identity=options.code_identity,
    )
    record.update(
        {
            "arm": options.arm,
            "repetition": options.repetition,
            "wall_time_s": round(time.perf_counter() - started, 3),
            "determinism": applied,
            "environment_variables": {name: os.environ.get(name) for name in RECORDED_VARIABLES},
            "python_hash_randomization": bool(sys.flags.hash_randomization),
            "hardware": hardware(torch),
            "packages": packages(),
        }
    )
    options.record.parent.mkdir(parents=True, exist_ok=True)
    with options.record.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
