"""Partes 10-11: entidades finais, sobreposição 3D e contagem observação->entidade."""
import json

import numpy as np
from common import dump, load_geometry, load_jsonl, run_dir

arm = "raw"
geo = load_geometry(arm)
ents = []
for line in load_jsonl(run_dir(arm) / "context_map/entities/entities.jsonl"):
    rec = line["record"]
    idx = np.array([int(g["geometry_id"].rsplit("--geom-", 1)[1]) for g in rec["geometry_refs"]])
    vox = {tuple(v) for v in np.floor(geo["map"][idx] / 0.1).astype(int)}
    ents.append({"id": rec["entity_id"], "labels": [h["label"] for h in rec["semantic_state"]["hypotheses"]],
                 "status": rec["semantic_state"]["status"], "points": len(idx), "vox": vox,
                 "centroid": geo["map"][idx].mean(0).round(2).tolist(),
                 "extent": (geo["map"][idx].max(0) - geo["map"][idx].min(0)).round(2).tolist()})
door = [e for e in ents if any("door" in l for l in e["labels"])]
pairs = []
for i in range(len(door)):
    for j in range(i + 1, len(door)):
        a, b = door[i]["vox"], door[j]["vox"]
        inter = len(a & b)
        pairs.append({"a": door[i]["id"], "b": door[j]["id"], "voxel_iou": round(inter / len(a | b), 3),
                      "containment_min": round(inter / min(len(a), len(b)), 3)})
# cadeia observação -> support -> entidade -> resolvida -> context_map
sa = load_jsonl(run_dir(arm) / "sensor_association/outputs/spatial-observations.jsonl")
sup = load_jsonl(run_dir(arm) / "semantic_fusion/outputs/fusion-supports.jsonl")
exc = load_jsonl(run_dir(arm) / "semantic_fusion/outputs/excluded-observations.jsonl")
sm = load_jsonl(run_dir(arm) / "semantic_mapping/outputs/entities.jsonl")
rej = load_jsonl(run_dir(arm) / "semantic_mapping/outputs/rejected-candidates.jsonl")
res = load_jsonl(run_dir(arm) / "entity_resolution/outputs/resolved-entities.jsonl")
dec = load_jsonl(run_dir(arm) / "entity_resolution/outputs/resolution-decisions.jsonl")
from collections import Counter

chain = {
    "regions_accepted": 83, "spatial_observations": len(sa),
    "observations_with_zero_support": sum(1 for s in sa if s["support"]["count"] == 0),
    "excluded_from_fusion_lt_min_geometry": len(exc),
    "fusion_supports": len(sup),
    "observations_per_support_hist": dict(Counter(len(s["spatial_observation_ids"]) for s in sup)),
    "semantic_entities": len(sm), "semantic_mapping_rejected_candidates": len(rej),
    "resolved_entities": len(res),
    "members_per_resolved_hist": dict(Counter(len(r["members"]) for r in res)),
    "resolution_decisions": Counter(d.get("outcome", d.get("decision")) for d in dec),
    "context_map_entities": len(ents),
    "physical_observations_per_entity": dict(Counter(1 for _ in ents)),
}
out = {"chain": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in chain.items()},
       "door_family": [{k: v for k, v in e.items() if k != "vox"} for e in door],
       "door_pairs_3d": sorted(pairs, key=lambda p: -p["containment_min"])}
dump("p10_entity_chain_and_overlap.json", out)
print(json.dumps(out["chain"], indent=1, default=str))
for e in out["door_family"]: print(e)
for p in out["door_pairs_3d"][:12]: print(p)
