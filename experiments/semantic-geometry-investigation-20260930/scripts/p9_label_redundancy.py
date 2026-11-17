"""Parte 9: redundância semântica das labels, e em que estágio ela surge. Não faz merge.

Similaridade: encoder de texto do MESMO CLIP do scorer (openai/clip-vit-large-patch14, revisão
pinada, template "a photo of a {label}"), cosseno entre embeddings L2-normalizados. Clusters
aproximados por ligação média com limiar explícito (dois limiares reportados) e, como segunda
visão, agrupamento léxico pelo núcleo (última palavra, com lista curta de sinônimos de núcleo
declarada aqui só para a medição). Para cada cluster, mede se as máscaras envolvidas se
sobrepõem (mesma área física: mecanismo B) e em quantas entidades finais elas terminam.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, dump

CLIP = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
HEAD_SYNONYMS = {"railing": "rail", "rod": "pole", "tile": "panel", "handle": "handle", "latch": "latch"}


def clip_text_embeddings(labels: list[str]) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP, revision=REVISION)
    model = CLIPModel.from_pretrained(CLIP, revision=REVISION).eval()
    with torch.inference_mode():
        batch = tok([f"a photo of a {label}" for label in labels], padding=True, return_tensors="pt")
        emb = model.get_text_features(**batch)
        emb = emb if isinstance(emb, torch.Tensor) else emb.pooler_output
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.numpy()


def average_linkage(sim: np.ndarray, threshold: float) -> list[list[int]]:
    clusters = [[i] for i in range(len(sim))]
    while True:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = float(np.mean(sim[np.ix_(clusters[a], clusters[b])]))
                if s > best:
                    best, pair = s, (a, b)
        if pair is None or best < threshold:
            return clusters
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]


def main() -> None:
    prov = json.loads((RESULTS / "p7_label_provenance.json").read_text())["rows"]
    masks = {m["mask_id"]: m for m in json.loads((RESULTS / "p6_mask_quality.json").read_text())["masks"]}
    rows = [r for r in prov if r["label"]]
    labels = sorted({r["label"] for r in rows})
    final_labels = sorted({h for r in rows if r["context_map_hypotheses"] for h in r["context_map_hypotheses"]})
    emb = clip_text_embeddings(labels)
    sim = emb @ emb.T
    off = sim[np.triu_indices(len(labels), 1)]

    # sobreposição de máscaras (mesma área?) entre regiões de um cluster
    from assoc import load_frame

    frame = load_frame("raw")

    def mask_overlap(ra, rb):
        a, b = frame.masks[ra], frame.masks[rb]
        inter = (a & b).sum()
        return inter / (a | b).sum(), inter / min(a.sum(), b.sum())

    def describe(cluster):
        names = [labels[i] for i in cluster]
        regs = [r for r in rows if r["label"] in names]
        pairs = []
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                iou, cont = mask_overlap(regs[i]["mask_id"], regs[j]["mask_id"])
                pairs.append((iou, cont))
        overlapping = sum(1 for _, c in pairs if c >= 0.3)
        return {
            "labels": names,
            "regions": [f'{r["mask_id"]}:{r["label"]}' for r in regs],
            "region_count": len(regs),
            "region_pairs": len(pairs),
            "region_pairs_containment_ge_0p3": overlapping,
            "fusion_supports": sorted({r["fusion_support"] for r in regs if r["fusion_support"]}),
            "context_map_entities": sorted({r["context_map_entity"] for r in regs if r["context_map_entity"]}),
            "regions_not_in_map": sum(1 for r in regs if not r["context_map_entity"]),
            "mean_intra_cosine": round(float(np.mean(sim[np.ix_(cluster, cluster)][np.triu_indices(len(cluster), 1)])), 4) if len(cluster) > 1 else None,
        }

    out = {
        "encoder": {"model": CLIP, "revision": REVISION, "template": "a photo of a {label}"},
        "labels_frame": labels, "labels_in_context_map": final_labels,
        "cosine_offdiag_quantiles": [round(float(q), 4) for q in np.quantile(off, [0, 0.25, 0.5, 0.75, 0.9, 1])],
        "nearest_neighbour": {labels[i]: {"label": labels[int(np.argsort(-sim[i])[1])], "cosine": round(float(np.sort(sim[i])[-2]), 4)} for i in range(len(labels))},
        "clusters": {},
    }
    for th in (0.90, 0.85):
        cl = [c for c in average_linkage(sim, th)]
        out["clusters"][f"average_linkage_cos_ge_{th}"] = {
            "cluster_count": len(cl), "label_count": len(labels),
            "multi_label_clusters": [describe(c) for c in sorted(cl, key=len, reverse=True) if len(c) > 1],
        }
    heads = {}
    for i, label in enumerate(labels):
        head = label.split()[-1]
        heads.setdefault(HEAD_SYNONYMS.get(head, head), []).append(i)
    out["clusters"]["lexical_head"] = {
        "cluster_count": len(heads), "label_count": len(labels),
        "multi_label_clusters": [describe(c) for c in sorted(heads.values(), key=len, reverse=True) if len(c) > 1],
    }
    # mecanismos
    per_mask = {}
    for r in rows:
        per_mask.setdefault(r["mask_id"], set()).add(r["label"])
    supports = {}
    for r in rows:
        if r["fusion_support"]:
            supports.setdefault(r["fusion_support"], set()).add(r["label"])
    out["mechanisms"] = {
        "A_same_mask_multiple_labels": sum(1 for v in per_mask.values() if len(v) > 1),
        "B_supports_with_gt1_distinct_label": sum(1 for v in supports.values() if len(v) > 1),
        "B_distinct_labels_inside_multi_label_supports": sum(len(v) for v in supports.values() if len(v) > 1),
        "C_multi_frame": "não aplicável: o run tem 1 frame (ver relation-graph-diagnostic para o run de 4 frames)",
        "D_label_key_is_exact_text": "accumulation.py: NFKC+casefold+espaços; 'door'/'blue door'/'metal door' são hipóteses distintas",
    }
    dump("p9_label_redundancy.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("nearest_neighbour",)}, indent=1)[:9000])


if __name__ == "__main__":
    main()
