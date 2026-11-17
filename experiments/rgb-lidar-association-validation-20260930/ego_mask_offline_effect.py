"""Fase 2 (offline): efeito das máscaras ego+valid sobre as regiões do frame-10_29.

Aplica as duas regras de normalização com os thresholds default
(minimum_valid_fraction=1.0, maximum_exclusion_fraction=0.0) às 95 regiões já
descobertas, usando os assets reais (V1 ego + valid), SEM re-rodar SAM/Qwen.
Quantifica quantos labels de veículo somem e as rejeições colaterais.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

B = Path("workspace/frame-10_29-full-run")
M = B / "visual_perception" / "outputs" / "masks" / "camera_1_image_raw-000000"
ASSETS = Path("examples/v0.2.0/assets/corridor-02")
W, H = 640, 480

ego = np.load(ASSETS / "camera_1-ego.npy")
valid = np.load(ASSETS / "camera_1-valid.npy")

label: dict[str, tuple[str, str]] = {}
with open(B / "visual_perception" / "outputs" / "semantic-interpretations.jsonl") as f:
    for line in f:
        o = json.loads(line)
        rid = o.get("request", {}).get("region_id")
        claims = o.get("parsed", {}).get("claims", [])
        if rid and claims:
            label[rid] = (claims[0].get("hypothesis") or "?", claims[0].get("category") or "?")

VEHICLE_WORDS = ("vehicle", "wheel", "tire", "dashboard", "off-road", "all-terrain", "chassis")


def is_ego_label(text: str, cat: str) -> bool:
    t = f"{text} {cat}".lower()
    return any(w in t for w in VEHICLE_WORDS)


rejected_excl = rejected_valid = kept = 0
ego_labels_removed = ego_labels_kept = 0
collateral: list[tuple[str, str, str]] = []
for path in sorted(M.glob("region-*.npy")):
    rid = path.stem
    packed = np.load(path)
    m = np.unpackbits(packed, count=W * H).reshape(H, W).astype(bool)
    area = int(m.sum())
    if area == 0:
        continue
    excl_frac = float((m & ego).sum()) / area
    valid_frac = float((m & valid).sum()) / area
    rej_excl = excl_frac > 0.0
    rej_valid = valid_frac < 1.0
    lab, cat = label.get(rid, ("?", "?"))
    ego_label = is_ego_label(lab, cat)
    if rej_excl:
        rejected_excl += 1
    if rej_valid and not rej_excl:
        rejected_valid += 1
    if not rej_excl and not rej_valid:
        kept += 1
    if rej_excl or rej_valid:
        if ego_label:
            ego_labels_removed += 1
        elif cat in ("structure", "surface", "ceiling", "door") or "wall" in lab or "floor" in lab:
            reason = "exclusion" if rej_excl else "valid"
            collateral.append((rid, f"{lab} [{cat}]", reason))
    else:
        if ego_label:
            ego_labels_kept += 1

print(f"regions with support-eligible mask: {rejected_excl + rejected_valid + kept}")
print(f"  rejected by EGO exclusion (touch>0%): {rejected_excl}")
print(f"  rejected by VALID region (<100% inside circle): {rejected_valid}")
print(f"  kept: {kept}")
print(f"\nego/vehicle-labelled regions removed: {ego_labels_removed}")
print(f"ego/vehicle-labelled regions still surviving: {ego_labels_kept}")
print(f"\ncollateral (structure/floor/wall) regions also removed: {len(collateral)}")
for rid, lab, reason in collateral:
    print(f"  {rid} {lab} ({reason})")
