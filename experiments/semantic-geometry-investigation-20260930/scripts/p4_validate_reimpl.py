"""Valida a reimplementação da associação contra o artefato persistido (calibração base)."""
import json

import numpy as np
from assoc import associate, load_frame, pipeline_support
from common import dump, load_geometry, run_dir

out = {}
for arm in ("raw", "voxel"):
    geo = load_geometry(arm)
    frame = load_frame(arm)
    ours = associate(geo["map"], frame)
    theirs = pipeline_support(arm)
    mismatched = {r: (len(ours["per_region"].get(r, [])), len(s)) for r, s in theirs.items()
                  if not np.array_equal(np.sort(ours["per_region"].get(r, np.array([], int))), s)}
    summary = json.loads((run_dir(arm) / "sensor_association/metrics/summary.json").read_text())["state_counts"]
    mine = {
        "behind_camera": int((~ours["in_domain"]).sum()),
        "outside_image": int((ours["in_domain"] & ~ours["in_image"]).sum()),
        "outside_valid_support": int((ours["in_image"] & ~ours["supported"]).sum()),
        "occluded": int(ours["occluded"].sum()),
        "visible": int(ours["visible"].sum()),
    }
    out[arm] = {"regions_compared": len(theirs), "regions_mismatched": len(mismatched),
                "mismatch_detail": mismatched, "state_counts_pipeline": summary, "state_counts_reimpl": mine}
    print(arm, out[arm]["regions_mismatched"], mine, summary, flush=True)
dump("p4_reimplementation_validation.json", out)
