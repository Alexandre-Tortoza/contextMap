"""C2: CalibrationSet base do E0 + refinamento versionado = calibração do braço E1.

A base é exatamente a de
``experiments/canonical-2-empirical-validation-20260929/scripts/ingest_window_fresh.py::build_calibration``
(a que ingeriu a sequência do E0; identidade conferida abaixo). O refinamento vem do documento
``calibration-refinement/1`` deste diretório e é aplicado uma única vez por
``contextmap.ingestion.apply_calibration_refinement``, que recusa base divergente, dupla
aplicação e ângulo fora do limite. Antes de aplicar, os YAMLs do dataset precisam ter os sha256
que o refinamento declara: um dataset diferente exige reestimar, não reaproveitar.

Uso como módulo (driver de ingestão do E1):
    sys.path.insert(0, ".../correction-benchmark-10_31-20260930/calibration")
    from build_refined_calibration import build_refined_calibration
    calibration = build_refined_calibration()

Uso como script: confere tudo, imprime as identidades e escreve ``refined-calibration.json``
(o ``calibration.json`` que a sequência do E1 deve conter), sem ingerir nada; na raiz do repo:
    PYTHONPATH=src python \
        experiments/correction-benchmark-10_31-20260930/calibration/build_refined_calibration.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/canonical-2-empirical-validation-20260929/scripts"))

import ingest_window_fresh  # noqa: E402

from contextmap.ingestion import (  # noqa: E402
    CalibrationSet,
    apply_calibration_refinement,
    load_calibration_refinement,
)
from contextmap.ingestion.calibration import encode_calibration_set  # noqa: E402
from contextmap.state_estimation.preflight import calibration_identity  # noqa: E402

REFINEMENT = HERE / "corridor-02-camera-body-refinement-v1.json"
DATASET = ROOT / "datasets/corridor-02"
OUTPUT = HERE / "refined-calibration.json"
E0_CALIBRATION_IDENTITY = "sha256:f7d3288e82305a34960dee8b841d625c7de643a506cb5706395a9f9bbe3242d6"
"""Identidade que o E0 (workspace-old/frame-10_31-raw) registrou para a calibração base."""


def build_refined_calibration(refinement_path: Path = REFINEMENT) -> CalibrationSet:
    """Return the E0 base calibration with the refinement applied.

    Raises:
        RuntimeError: If the base is not the one E0 used or a dataset YAML differs from the
            one the refinement was estimated on.
        contextmap.ingestion.CalibrationError: If the refinement does not apply to the base.
    """
    base = ingest_window_fresh.build_calibration()
    if calibration_identity(base) != E0_CALIBRATION_IDENTITY:
        raise RuntimeError(
            f"the base calibration is not E0's: {calibration_identity(base)} "
            f"!= {E0_CALIBRATION_IDENTITY}"
        )
    refinement = load_calibration_refinement(refinement_path)
    for name, expected in refinement.source_files.items():
        actual = "sha256:" + hashlib.sha256((DATASET / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"{name} has {actual}, but {refinement.refinement_id} was estimated on "
                f"{expected}: re-estimate the refinement"
            )
    return apply_calibration_refinement(base, refinement)


def main() -> int:
    """Check, print the identities and write ``refined-calibration.json``."""
    refined = build_refined_calibration()
    OUTPUT.write_text(json.dumps(encode_calibration_set(refined), indent=2) + "\n", "utf-8")
    print("BASE_CALIBRATION_IDENTITY", E0_CALIBRATION_IDENTITY)
    print("REFINED_CALIBRATION_IDENTITY", calibration_identity(refined))
    print("REFINEMENTS", [item.refinement_id for item in refined.refinements])
    print("WROTE", OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
