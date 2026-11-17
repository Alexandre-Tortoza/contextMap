"""Braço C4: re-executa só a descoberta de regiões (SAM2) do frame 55 com limiares declarados.

Um braço = um processo (1 frame por processo, workaround #689). Usa o executor real do
runtime (``VisualPerceptionExecutor``) com a MESMA pilha de configuração do E0
(``bench_common.CONFIG_STACK``: ego-mask e normalização do overlay de máscaras incluídas) e muda
uma única variável: ``predicted_iou_threshold``/``stability_threshold`` do ``Sam2Config``, por
override declarado. O preset é o ``canonical/1`` reduzido a ``image_preparation`` +
``region_discovery``: as métricas de região não dependem de features nem de VLM, e sem eles o
braço não carrega DINOv2/CLIP/Qwen na GPU.

O artefato de percepção do braço vai para ``outputs/correction-benchmark-10_31-20260930/c4/<arm>/``
(gitignored); o workspace de entrada (``workspace-old/frame-10_31-raw``) só é lido.

Rodar da raiz do repositório (os assets da ego-mask resolvem contra o diretório atual)::

    PYTHONPATH=src:$PROV HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY experiments/correction-benchmark-10_31-20260930/scripts/c4_sam2_arm.py \
      --arm-id E3-g-iou0.5-stab0.80 --pred-iou 0.5 --stability 0.80
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
    write_json,
)

SAM2_KEY = "components.region_perception.discovery.sam2"


def main() -> int:
    """Run one region-discovery arm and record its effective configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--pred-iou", type=float, required=True)
    parser.add_argument("--stability", type=float, required=True)
    parser.add_argument("--checkpoint", default=None, help="override do checkpoint (E3-L)")
    parser.add_argument("--model-version", default=None, help="revisão do checkpoint override")
    parser.add_argument("--force", action="store_true", help="apaga uma saída anterior do braço")
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

    overrides = [
        f"{SAM2_KEY}.predicted_iou_threshold={args.pred_iou}",
        f"{SAM2_KEY}.stability_threshold={args.stability}",
    ]
    if args.checkpoint is not None:
        if args.model_version is None:
            parser.error("--checkpoint exige --model-version (entra no digest do Sam2Config)")
        overrides += [
            f"{SAM2_KEY}.checkpoint={args.checkpoint}",
            f"{SAM2_KEY}.model_version={args.model_version}",
        ]
    effective = resolve_effective_config(
        files=[ROOT / path for path in CONFIG_STACK], overrides=overrides
    )

    arm_dir = ARM_OUTPUTS / "c4" / args.arm_id
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
        code_version="correction-benchmark-10_31/c4",
    )
    perception = executors["visual_perception"]
    # Só as duas etapas que decidem as regiões; mesma topologia do canonical/1 até aqui.
    regions_only = PipelinePreset(
        preset_id="canonical/1-regions-only",
        stages=tuple(
            stage
            for stage in CANONICAL_PRESET_V1.stages
            if stage.stage_id in {"image_preparation", "region_discovery"}
        ),
        description="canonical/1 reduzido a image_preparation + region_discovery (C4).",
    )
    perception._preset = lambda: regions_only

    workspace = E0_RUNS["raw"]
    sequence_location = f"sequences/{SEQUENCE_NAME}/{SEQUENCE_ID}"
    sequence = SequenceArtifactReader(workspace / sequence_location)
    sequence_ref = ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id=SEQUENCE_ID,
        content_hash=inventory_digest(sequence.manifest.file_inventory),
        location=sequence_location,
    )
    request = StageRequest(
        stage_id="visual_perception",
        inputs={"sequence": (sequence_ref,)},
        components={},
        config_digest=effective.digest,
        output_dir=output,
        workspace=workspace,
        # O artefato do braço fica fora do workspace de entrada (que é imutável).
        output_root=arm_dir,
        observation_selection=encode_selection(
            ExplicitIdsSelection(observation_ids=frozenset({SourceObservationId(IMAGE_ID)}))
        ),
    )
    ref = perception.execute(request)
    elapsed = time.perf_counter() - started

    plan = perception._region_discovery
    write_json(arm_dir / "effective_config.json", effective.to_document())
    write_json(
        arm_dir / "arm.json",
        {
            "arm_id": args.arm_id,
            "variable": "sam2 predicted_iou_threshold / stability_threshold"
            + (" + checkpoint" if args.checkpoint else ""),
            "predicted_iou_threshold": args.pred_iou,
            "stability_threshold": args.stability,
            "checkpoint_override": args.checkpoint,
            "config_stack": list(CONFIG_STACK),
            "overrides": overrides,
            "effective_config_digest": effective.digest,
            "region_discovery_backend": dataclasses.asdict(plan.backend),
            "discovery_identity": plan.discovery_identity.to_dict(),
            "preset": regions_only.preset_id,
            "image_id": IMAGE_ID,
            "input_workspace": str(workspace.relative_to(ROOT)),
            "perception_run_id": ref.artifact_id,
            "perception_dir": str(output.relative_to(ROOT)),
            "elapsed_s": round(elapsed, 2),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "torch": torch.__version__,
            "code_identity": code_identity(RELEVANT_UNTRACKED),
        },
    )
    print(json.dumps({"arm": args.arm_id, "run_id": ref.artifact_id, "elapsed_s": elapsed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
