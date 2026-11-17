"""Avalia os braços SAM2 do C4: métricas de região (mesmo extrator do E0), gate, figuras e tabela.

Para cada braço em ``outputs/correction-benchmark-10_31-20260930/c4/<arm>/``:

* ``results/C4-sam2-grid/<arm>/metrics.json`` (``extract_metrics.extract``; só a camada de
  regiões e a de semântica/percepção existem no braço);
* ``gate.json`` contra o E0 (``arm_kind = none``, camada-alvo ``regions``);
* ``masks_numbered.png`` (máscaras numeradas) e ``mask_overlap_count.png``.

E, no diretório da grade: ``table.md``/``table.csv``/``summary.json`` com a tabela por braço, a
checagem de determinismo (E0-sam2 == E0; E2 == E3-g 0,80x0,95) e o estado da seleção.

A **seleção** do braço depende da anotação fraca (critério declarado no plano, antes de rodar:
descartar braços que perdem objeto anotado encontrado com IoU >= 0,5; entre os restantes,
minimizar ``pixels_with_multiple_masks``; desempate por ``fragmentation_rate`` e depois menos
máscaras). Sem anotação a seleção fica ``pending_annotation``.

    PYTHONPATH=src $MISE_PY experiments/correction-benchmark-10_31-20260930/scripts/c4_evaluate.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (
    ARM_OUTPUTS,
    E0_RUNS,
    IMAGE_ID,
    RESULTS,
    SEQUENCE_ID,
    read_json,
    write_json,
)
from extract_metrics import DEFAULT_ANNOTATION, PENDING, _load_region_masks, extract
from gate import check

GRID = RESULTS / "C4-sam2-grid"
POLICY = Path(__file__).resolve().parents[1] / "manifests/gate-policy.json"
FONT = ImageFont.load_default()


def load_rgb() -> np.ndarray:
    """RGB (480, 640, 3) do frame 55 a partir do payload bgr8 da sequência do E0."""
    path = E0_RUNS["raw"] / f"sequences/corridor-02/{SEQUENCE_ID}/rgb/{IMAGE_ID}.bin"
    return np.fromfile(path, dtype=np.uint8).reshape(480, 640, 3)[:, :, ::-1].copy()


def render_masks(rgb: np.ndarray, masks: dict[str, np.ndarray], title: str, out_dir: Path) -> None:
    """Numbered masks (largest first, so small ones stay visible) and the overlap count."""
    rng = np.random.default_rng(3)
    order = sorted(masks, key=lambda r: -int(masks[r].sum()))
    canvas = rgb.astype(float) * 0.35
    overlap = np.zeros(rgb.shape[:2], dtype=np.int32)
    for rid in order:
        m = masks[rid]
        overlap += m
        color = rng.integers(60, 255, 3)
        canvas[m] = canvas[m] * 0.45 + color * 0.55
    image = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for rid in order:
        m = masks[rid]
        edge = m & ~ndimage.binary_erosion(m)
        ys, xs = np.nonzero(edge)
        for x, y in zip(xs[::3], ys[::3], strict=True):
            draw.point((int(x), int(y)), fill=(255, 255, 255))
        cy, cx = ndimage.center_of_mass(m)
        draw.text((cx - 8, cy - 5), rid[-4:], fill=(255, 255, 0), font=FONT)
    draw.rectangle([0, 0, 640, 13], fill=(0, 0, 0))
    draw.text((4, 1), f"{title}: {len(masks)} masks", fill=(255, 255, 255), font=FONT)
    image.save(out_dir / "masks_numbered.png")

    top = max(int(overlap.max()), 1)
    t = np.clip(overlap / top, 0, 1).reshape(-1)
    heat = (
        (np.stack([np.clip(1.5 - np.abs(4 * t - k), 0, 1) for k in (3, 2, 1)], axis=1) * 255)
        .astype(np.uint8)
        .reshape(480, 640, 3)
    )
    heat[overlap == 0] = (rgb[overlap == 0] * 0.3).astype(np.uint8)
    heat_image = Image.fromarray(heat)
    ImageDraw.Draw(heat_image).text(
        (4, 4),
        f"{title}: mask overlap count (max {int(overlap.max())})",
        fill=(255, 255, 255),
        font=FONT,
    )
    heat_image.save(out_dir / "mask_overlap_count.png")


def evaluate_arm(
    arm_dir: Path, rgb: np.ndarray, baseline: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Metrics, gate and figures of one C4 arm."""
    arm = read_json(arm_dir / "arm.json")
    arm_id = arm["arm_id"]
    out = GRID / arm_id
    metrics = extract(
        arm_dir, arm_id=arm_id, annotation_path=DEFAULT_ANNOTATION, notes="braço C4 (só regiões)"
    )
    metrics["arm"] = arm | {"arm_kind": "none", "target_layer": "regions"}
    write_json(out / "metrics.json", metrics)
    gate = check(baseline, metrics, arm_kind="none", target_layer="regions", policy=policy)
    write_json(out / "gate.json", gate)
    masks, _ = _load_region_masks(arm_dir / "visual_perception")
    render_masks(rgb, {r: m.as_array() for r, m in masks.items()}, arm_id, out)
    return metrics


def _row(metrics: dict[str, Any]) -> dict[str, Any]:
    r = metrics["layers"]["regions"]
    arm = metrics["arm"]
    recall = r.get("known_object_recall")
    return {
        "arm": metrics["arm_id"],
        "pred_iou": arm.get("predicted_iou_threshold"),
        "stability": arm.get("stability_threshold"),
        # Saída do AutomaticMaskGenerator: os limiares são aplicados DENTRO dele, antes do NMS,
        # então com limiares > 0 este número não é um subconjunto das 431 propostas do 0/0.
        "sam2_amg_proposals": r["sam2_raw_proposal_count"],
        "accepted_masks": r["mask_count"],
        "median_area_px": r["area_px_median"],
        "masks_below_500px": r["masks_below_500px"],
        "fragmentation_rate": r["fragmentation_rate"],
        "masks_with_gt_10_components": r["masks_with_gt_10_components"],
        "masks_contained_in_another": r["masks_contained_in_another"],
        "contained_pairs": r["contained_pair_count"],
        "pixels_with_multiple_masks": r["pixels_with_multiple_masks"],
        "multiple_mask_pixel_fraction": r["multiple_mask_pixel_fraction"],
        "max_nesting_depth": r["max_nesting_depth"],
        "usable_area_coverage": r["usable_area_coverage"],
        "exclusion_overlap_rejections": r.get("normalization_rejections_by_reason", {}).get(
            "exclusion_overlap", 0
        ),
        "known_object_recall": recall if recall == PENDING else (recall or {}).get("recall"),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "—" if value is None else str(value)


def main() -> int:
    """Evaluate every arm found under the C4 output directory."""
    baseline = read_json(RESULTS / "E0/metrics.json")
    policy = read_json(POLICY)
    rgb = load_rgb()
    arm_dirs = sorted(p for p in (ARM_OUTPUTS / "c4").iterdir() if (p / "arm.json").is_file())
    results = {}
    for arm_dir in arm_dirs:
        metrics = evaluate_arm(arm_dir, rgb, baseline, policy)
        results[metrics["arm_id"]] = metrics
        print("evaluated", metrics["arm_id"])
    rows = [_row(m) for m in results.values()]

    def masks_of(arm_id: str) -> dict[str, str] | None:
        return (
            results[arm_id]["layers"]["regions"]["mask_sha256_by_region"]
            if arm_id in results
            else None
        )

    e0_masks = baseline["layers"]["regions"]["mask_sha256_by_region"]
    determinism = {
        "E0-sam2 masks == E0 masks (sha256 of every mask)": masks_of("E0-sam2-iou0.00-stab0.00")
        == e0_masks,
        "E2 masks == E3-g 0.80x0.95 masks (same configuration, two processes)": masks_of(
            "E2-iou0.80-stab0.95"
        )
        == masks_of("E3-g-iou0.80-stab0.95"),
    }
    annotated = all(isinstance(row["known_object_recall"], float) for row in rows)
    selection: dict[str, Any] = {
        "criterion": [
            "1. descartar braços que perdem qualquer objeto anotado que o E1c/E0 encontra "
            "com IoU >= 0,5",
            "2. entre os restantes, minimizar pixels_with_multiple_masks",
            "3. desempate: menor fragmentation_rate, depois menos máscaras",
        ],
        "status": "selected" if annotated else PENDING,
        "selected_arm": None,
        "note": (
            "Sem a anotação fraca do frame 55 o passo 1 não pode ser avaliado. Minimizar só o "
            "passo 2 escolheria trivialmente o braço mais restritivo (menos máscaras, menos "
            "cobertura), que é exatamente o risco que o passo 1 existe para barrar."
        ),
    }
    write_json(
        GRID / "summary.json", {"rows": rows, "determinism": determinism, "selection": selection}
    )
    columns = list(rows[0])
    with (GRID / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *("| " + " | ".join(_fmt(row[c]) for c in columns) + " |" for row in rows),
    ]
    (GRID / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(determinism)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
