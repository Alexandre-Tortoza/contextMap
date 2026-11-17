"""Anatomia da oclusão do pipeline: quanto do 'ocluído' é a mesma superfície?"""
import numpy as np
from assoc import associate, load_frame
from common import dump, load_geometry

geo = load_geometry("raw"); frame = load_frame("raw")
res = associate(geo["map"], frame)
occ = res["occluded"]
gap = res["range"][occ] - res["support_depth"][occ]
rel = gap / res["support_depth"][occ]
bins = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf]
hist = {}
lo = 0
for b in bins:
    hist[f"{lo}-{b} m"] = int(((gap > lo) & (gap <= b)).sum()); lo = b
# oclusão exclusivamente por vizinhança (o próprio pixel/célula não ocluiria)
# e pontos em células onde o próprio ponto é o mais próximo da célula
supported = res["supported"]
out = {"supported_points": int(supported.sum()), "occluded": int(occ.sum()), "visible": int(res["visible"].sum()),
       "occluded_gap_hist": hist,
       "occluded_within_0p5m_of_support_fraction": float((gap <= 0.5).mean()),
       "occluded_within_1m_or_20pct_fraction": float(((gap <= 1.0) | (rel <= 0.2)).mean())}
# sobre as máscaras: pontos 'ocluídos' que caem dentro de alguma máscara aceita
inside_any = np.zeros(len(occ), bool)
idx = np.flatnonzero(res["in_image"])
for m in frame.masks.values():
    inside_any[idx] |= m[res["row"][idx], res["col"][idx]]
out["occluded_inside_some_mask"] = int((occ & inside_any).sum())
out["occluded_inside_some_mask_within_0p5m"] = int((occ & inside_any)[occ].sum() and ((gap <= 0.5) & inside_any[occ]).sum())
print(out)
dump("p5_occlusion_anatomy.json", out)
