"""Variância do Qwen (#556): re-executa a interpretação das regiões do E0 no frame 55.

Um rerun = um processo (1 frame por processo, #689). Pilha de configuração idêntica à do E0
(``bench_common.CONFIG_STACK``, sem override). O preset é o do E0 (``canonical/1`` sem
``scene_interpretation``) menos as duas etapas de features (DINOv2/CLIP), que não alimentam o
Qwen: ``region_interpretation`` depende só de ``{image, regions}``. As regiões vêm do próprio
SAM2, que é determinístico nesta configuração (verificado: 83/83 máscaras com o mesmo sha256
do E0 e auditoria idêntica); o script confere isso de novo depois de cada rerun e marca o rerun
como inválido se uma máscara divergir. Assim as regiões ficam congeladas sem substituir o
backend de descoberta nem mudar código de produção.

Saída: ``outputs/correction-benchmark-10_31-20260930/qwen-n3/<rerun>/visual_perception``.

    PYTHONPATH=src:$PROV HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY experiments/correction-benchmark-10_31-20260930/scripts/qwen_rerun.py --rerun-id r1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (
    ARM_OUTPUTS,
    CONFIG_STACK,
    E0_RUNS,
    IMAGE_ID,
    RELEVANT_UNTRACKED,
    ROOT,
    SEQUENCE_ID,
    SEQUENCE_NAME,
    code_identity,
    read_jsonl,
    write_json,
)

KEPT_STAGES = {"image_preparation", "region_discovery", "region_interpretation"}


def _mask_hashes(perception_dir: Path) -> dict[str, str]:
    """Region id -> sha256 of its persisted mask payload, from the run's mask index."""
    rows = read_jsonl(perception_dir / "outputs/masks/mask-index.jsonl")
    return {row["region_id"]: row["content_hash"] for row in rows if "region_id" in row}


def main() -> int:
    """Run one Qwen rerun over the E0 regions and check that the regions were the same."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--replicate-e0-identity",
        action="store_true",
        help=(
            "usa o mesmo request do run_two_arms.py (config_digest='perc', content_hash "
            "'sha256:driven'): o run id sai igual ao do E0 e, como o prompt renderizado contém o "
            "perception_result_id, a entrada do modelo fica byte-idêntica à do E0"
        ),
    )
    args = parser.parse_args()

    import torch

    from contextmap.ingestion import SequenceArtifactReader, SourceObservationId, encode_selection
    from contextmap.ingestion.sequence_observation_selection import ExplicitIdsSelection
    from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
    from contextmap.runtime.composition import compose_executors
    from contextmap.runtime.executors import inventory_digest
    from contextmap.visual_perception.execution.pipeline_resolution import (
        CANONICAL_PRESET_V1,
        PipelinePreset,
    )

    effective = resolve_effective_config(files=[ROOT / path for path in CONFIG_STACK])
    arm_dir = ARM_OUTPUTS / "qwen-n3" / args.rerun_id
    output = arm_dir / "visual_perception"
    if output.exists():
        if not args.force:
            print(f"{output} já existe; use --force para refazer", file=sys.stderr)
            return 2
        shutil.rmtree(output)
    arm_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    executors = compose_executors(
        effective,
        module_available=lambda _module: True,
        environ={},
        code_version="correction-benchmark-10_31/qwen-n3",
    )
    perception = executors["visual_perception"]
    preset = PipelinePreset(
        preset_id="canonical/1-labels-no-scene-no-features",
        stages=tuple(s for s in CANONICAL_PRESET_V1.stages if s.stage_id in KEPT_STAGES),
        description="canonical/1 sem scene_interpretation e sem features (variância do Qwen).",
    )
    perception._preset = lambda: preset

    workspace = E0_RUNS["raw"]
    sequence_location = f"sequences/{SEQUENCE_NAME}/{SEQUENCE_ID}"
    sequence = SequenceArtifactReader(workspace / sequence_location)
    if args.replicate_e0_identity:
        # Literalmente o request de run_two_arms.py: mesmo run id do E0, mesmo prompt em bytes.
        content_hash, config_digest = "sha256:driven", "perc"
    else:
        # O digest identifica a execução: cada rerun precisa de um run id próprio, então o
        # identificador do rerun entra no digest declarado (a configuração efetiva é a mesma).
        content_hash = inventory_digest(sequence.manifest.file_inventory)
        config_digest = f"{effective.digest}|rerun={args.rerun_id}"
    request = StageRequest(
        stage_id="visual_perception",
        inputs={
            "sequence": (
                ArtifactRef(
                    stage_id="ingestion",
                    contract="SequenceArtifact",
                    artifact_id=SEQUENCE_ID,
                    content_hash=content_hash,
                    location=sequence_location,
                ),
            )
        },
        components={},
        config_digest=config_digest,
        output_dir=output,
        workspace=workspace,
        output_root=arm_dir,
        observation_selection=encode_selection(
            ExplicitIdsSelection(observation_ids=frozenset({SourceObservationId(IMAGE_ID)}))
        ),
    )
    ref = perception.execute(request)
    elapsed = time.perf_counter() - started

    e0_masks = _mask_hashes(E0_RUNS["raw"] / "visual_perception")
    rerun_masks = _mask_hashes(output)
    same_regions = e0_masks == rerun_masks
    write_json(arm_dir / "effective_config.json", effective.to_document())
    write_json(
        arm_dir / "arm.json",
        {
            "arm_id": f"E0-qwen-{args.rerun_id}",
            "variable": "nenhuma (rerun do Qwen com a mesma configuração do E0)",
            "replicates_e0_run_identity": args.replicate_e0_identity,
            "request_config_digest": config_digest,
            "request_sequence_content_hash": content_hash,
            "config_stack": list(CONFIG_STACK),
            "effective_config_digest": effective.digest,
            "preset": preset.preset_id,
            "semantic_request_policy": perception._semantic_request_policy.to_document(),
            "region_discovery_backend": dataclasses.asdict(perception._region_discovery.backend),
            "regions_identical_to_e0": same_regions,
            "e0_region_count": len(e0_masks),
            "rerun_region_count": len(rerun_masks),
            "input_workspace": str(workspace.relative_to(ROOT)),
            "perception_run_id": ref.artifact_id,
            "perception_dir": str(output.relative_to(ROOT)),
            "elapsed_s": round(elapsed, 2),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch": torch.__version__,
            "code_identity": code_identity(RELEVANT_UNTRACKED),
        },
    )
    print(
        json.dumps(
            {"rerun": args.rerun_id, "run_id": ref.artifact_id, "same_regions": same_regions}
        )
    )
    return 0 if same_regions else 3


if __name__ == "__main__":
    raise SystemExit(main())
