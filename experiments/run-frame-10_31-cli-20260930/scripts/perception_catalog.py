"""Catálogo de seleção com o run de percepção de uma context run do braço raw.

Uso: python perception_catalog.py <workspace> <run-NNNN> <saída.json>

O braço voxel reusa a percepção do raw (`--select visual_perception=<id> --catalog <saída>`), então
os dois mapas partem das mesmas regiões e labels, e o Qwen não roda de novo. A entrada é copiada
do registro `context_run.json` (id, hash, local), sem reescrever nada do run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def perception_catalog(workspace: Path, run: str) -> dict:
    """Monta o catálogo com a entrada `visual_perception` da context run `run`."""
    record = json.loads(next(workspace.glob(f"*/{run}/context_run.json")).read_text())
    (perception,) = (a for a in record["artifacts"] if a["stage_id"] == "visual_perception")
    if perception["disposition"] != "produced":
        raise SystemExit(f"{run}: a percepção não foi produzida por este run ({perception})")
    sequence = record["foundation"]["sequence"]
    return {
        "schema_version": "0.1.0",
        "entries": [
            {
                "stage_id": "visual_perception",
                "contract": perception["contract"],
                "artifact_id": perception["artifact_id"],
                "content_hash": perception["content_hash"],
                "location": perception["location"],
                "run_index": int(run.removeprefix("run-")),
                "lineage": {"upstream": {"ingestion": [sequence["artifact_id"]]}},
            }
        ],
    }


def main() -> None:
    """Grava o catálogo e imprime o id do run de percepção (o valor do `--select`)."""
    workspace, run, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    catalog = perception_catalog(workspace, run)
    output.write_text(json.dumps(catalog, indent=2) + "\n")
    print(catalog["entries"][0]["artifact_id"])


if __name__ == "__main__":
    main()
