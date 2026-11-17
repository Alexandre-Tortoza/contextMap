"""Entradas de associação do E0 (frame 10:31, braço raw) e acesso à reimplementação validada.

- Pertença persistida: `SensorAssociationRunReader` (API pública) + `geometry_index_of`.
- Profundidade/suporte/oclusão por ponto: `scripts/assoc.py` da investigação, validado bit a
  bit contra o artefato (`results/p4_reimplementation_validation.json`: 0/83 regiões
  divergentes, contagens de estado idênticas). O artefato não persiste profundidade por ponto.
- Máscaras e área útil: `regions/e0_regions.py` (APIs públicas de visual_perception).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "regions"))
from e0_regions import INVESTIGATION, ROOT, RUN, code_identity, load_e0_regions  # noqa: E402

sys.path.insert(0, str(INVESTIGATION / "scripts"))
import assoc  # noqa: E402

from contextmap.geometric_mapping import geometry_index_of  # noqa: E402
from contextmap.sensor_association import SensorAssociationRunReader  # noqa: E402

__all__ = [
    "INVESTIGATION",
    "ROOT",
    "RUN",
    "assoc",
    "code_identity",
    "load_e0_regions",
    "persisted_membership",
]


def persisted_membership() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Índices globais associados a cada região, como o pipeline os persistiu."""
    reader = SensorAssociationRunReader(RUN / "sensor_association")
    manifest = reader.manifest
    membership = {}
    for observation in reader.observations():
        membership[str(observation.region_id)] = np.array(
            [
                geometry_index_of(map_id=ref.map_id, geometry_id=ref.geometry_id)
                for ref in observation.geometry_support
            ],
            dtype=np.int64,
        )
    provenance = {
        "sensor_association_run_id": manifest.run_id,
        "geometric_map_id": manifest.geometric_map_id,
        "calibration_identity": manifest.calibration_identity,
        "candidate_policy": manifest.candidate_policy,
        "visibility_policy": manifest.visibility_policy,
        "membership_policy_id": manifest.membership_policy_id,
    }
    return membership, provenance
