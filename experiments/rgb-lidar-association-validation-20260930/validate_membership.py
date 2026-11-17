"""Fase 1: valida se os pontos 'pintados' de cada regiao caem dentro da mascara dela.

Reusa APENAS os artefatos ja produzidos pelo bundle (nada de re-execucao):
- geometry-support.u32  : indices de ponto por regiao (fatiado por offset/count)
- spatial-observations  : offset/count e region_id de cada regiao
- samples.csv           : pixel projetado (prepared_u/v) e estado por ponto
- masks/*.npy           : mascara booleana 640x480 no espaco da imagem preparada

Regra de pertencimento (mask-membership-v1): pixel = floor(coord + 0.5).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

B = Path("workspace/frame-10_29-full-run")
SA = B / "sensor_association" / "outputs"
MASKS = B / "visual_perception" / "outputs" / "masks" / "camera_1_image_raw-000000"
SAMPLES = B / "sensor_association" / "debug" / "frames" / "camera_1_image_raw-000000" / "samples.csv"

# label por regiao
label: dict[str, str] = {}
with open(B / "visual_perception" / "outputs" / "semantic-interpretations.jsonl") as f:
    for line in f:
        o = json.loads(line)
        rid = o.get("request", {}).get("region_id")
        claims = o.get("parsed", {}).get("claims", [])
        if rid and claims:
            label[rid] = claims[0].get("hypothesis")

# support (offset,count,region_id) por regiao
regions: list[tuple[str, int, int]] = []
with open(SA / "spatial-observations.jsonl") as f:
    for line in f:
        o = json.loads(line)
        sup = o["support"]
        if sup["count"] > 0:
            regions.append((o["region_id"], sup["offset"], sup["count"]))

support = np.fromfile(SA / "geometry-support.u32", dtype="<u4")
print(f"geometry-support.u32: {support.size} entries (membership_pair_count)")

# pixel projetado por geometry_index (so linhas presentes no samples.csv)
u = {}
v = {}
state = {}
with open(SAMPLES) as f:
    for row in csv.DictReader(f):
        gi = int(row["geometry_index"])
        u[gi] = float(row["prepared_u"])
        v[gi] = float(row["prepared_v"])
        state[gi] = row["state"]

W, H = 640, 480
print(f"\n{'region':<12}{'label':<22}{'painted':>8}{'in_mask':>9}{'pct':>7}{'states':>10}")
grand_in = grand_tot = 0
for rid, off, cnt in regions:
    idx = support[off : off + cnt]
    packed = np.load(MASKS / f"{rid}.npy")
    mask = np.unpackbits(packed, count=W * H).reshape(H, W).astype(bool)
    assert mask.shape == (H, W), (rid, mask.shape)
    inside = 0
    st: Counter[str] = Counter()
    missing = 0
    for gi in idx.tolist():
        if gi not in u:
            missing += 1
            continue
        st[state.get(gi, "?")] += 1
        col = int(np.floor(u[gi] + 0.5))
        rowp = int(np.floor(v[gi] + 0.5))
        if 0 <= col < W and 0 <= rowp < H and bool(mask[rowp, col]):
            inside += 1
    tot = cnt - missing
    grand_in += inside
    grand_tot += tot
    pct = 100.0 * inside / tot if tot else 0.0
    lab = (label.get(rid) or "?")[:20]
    top = ",".join(f"{k}:{n}" for k, n in st.most_common(2))
    flag = "" if pct > 99.5 else "  <-- CHECK"
    print(f"{rid:<12}{lab:<22}{cnt:>8}{inside:>9}{pct:>6.1f}%{'':>2}{top}{flag}")

print(f"\nTOTAL: {grand_in}/{grand_tot} painted points land inside their region mask "
      f"= {100.0*grand_in/grand_tot:.2f}%")

# reforco: estado dos pontos pintados deve ser 'associated' (nao occluded/visible_unassigned)
