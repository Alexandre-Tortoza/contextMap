"""Contrato físico do run: onde cada arquivo vive e como seus bytes são formados.

Este módulo é o único lugar que decide o layout em disco e a forma dos registros de célula.
O writer e o reader precisam concordar byte a byte, então ambos dependem daqui: um bump de
``SCHEMA_VERSION`` é o diff de um arquivo só.

Ele não conhece ``AtomicRunDirectory``, nem a regra de junção, nem os artifacts a montante.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextmap.artifact import ContextEntityId
from contextmap.geometric_mapping import Bounds3D
from contextmap.ingestion import FrameId
from contextmap.shared import Vector3
from contextmap.spatial_support.models import EntitySupport, SpatialCell, SpatialCellId

SCHEMA_VERSION = "0.2.0"
"""Semantic spatial support artifact schema version written and understood by this package."""

MANIFEST = "manifest.json"
"""Authoritative metadata of the run, written by ``AtomicRunDirectory.publish``."""

LINEAGE = "lineage.json"
"""Identities of the three upstream artifacts, recorded verbatim."""

CONFIG = "config.json"
"""Schema version, policy identity and the grid the support was joined onto."""

ENVIRONMENT = "environment.json"
"""Interpreter and platform the derivation ran on."""

CELLS = "outputs/cells.jsonl"
"""One JSON record per supported cell, in the support's own order."""

SUPPORT = "outputs/support.json"
"""Header the cell records share: source map, frame, grid and aggregation rule."""

METRICS = "metrics/support.json"
"""The three figures of the run: cells, distinct entities and total support weight."""


class SpatialSupportArtifactError(ValueError):
    """Raised when the support cannot be derived or read without breaking an invariant."""


def cell_record(cell: SpatialCell) -> dict[str, Any]:
    """Encode one cell as the JSON record ``outputs/cells.jsonl`` holds.

    Args:
        cell: The cell to encode.

    Returns:
        A JSON-ready record. The cell's frame is omitted on purpose: it lives once in the
        ``outputs/support.json`` header, and :func:`cell_of` reinjects it.
    """
    return {
        "cell_id": str(cell.cell_id),
        "aggregate_index": cell.aggregate_index,
        "key": [cell.key[0], cell.key[1], cell.key[2]],
        "centroid_m": [cell.centroid_m[0], cell.centroid_m[1], cell.centroid_m[2]],
        "bounds": {
            "minimum_m": list(cell.bounds.minimum_m),
            "maximum_m": list(cell.bounds.maximum_m),
        },
        "point_count": cell.point_count,
        "entity_support": [
            {"entity_id": str(item.entity_id), "support_weight": item.support_weight}
            for item in cell.entity_support
        ],
    }


def cell_of(record: dict[str, Any], frame_id: FrameId) -> SpatialCell:
    """Reconstruct one cell from its persisted record, revalidated by its own constructors.

    Args:
        record: One decoded line of ``outputs/cells.jsonl``.
        frame_id: The frame of the support the record belongs to.

    Returns:
        The cell, with the support frame reinjected into its bounds.

    Raises:
        ValueError: If the record violates a domain invariant of :class:`SpatialCell`.
    """
    key = tuple(int(component) for component in record["key"])
    bounds = record["bounds"]
    return SpatialCell(
        cell_id=SpatialCellId(record["cell_id"]),
        aggregate_index=int(record["aggregate_index"]),
        key=(key[0], key[1], key[2]),
        centroid_m=_vector3(record["centroid_m"]),
        bounds=Bounds3D(
            frame_id=frame_id,
            minimum_m=_vector3(bounds["minimum_m"]),
            maximum_m=_vector3(bounds["maximum_m"]),
        ),
        point_count=int(record["point_count"]),
        entity_support=tuple(
            EntitySupport(
                entity_id=ContextEntityId(item["entity_id"]),
                support_weight=int(item["support_weight"]),
            )
            for item in record["entity_support"]
        ),
    )


def json_document(payload: dict[str, Any]) -> str:
    """Render one contractual JSON document.

    Args:
        payload: The document to render.

    Returns:
        Indented, key-sorted JSON terminated by a newline, as every sibling artifact
        renders it. The bytes are hashed into the manifest's file inventory, so the exact
        rendering is part of the schema.
    """
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def read_json_document(path: Path) -> dict[str, Any]:
    """Decode one contractual JSON document of a finalized run.

    Args:
        path: Absolute path of the document.

    Returns:
        The decoded mapping.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _vector3(values: Any) -> Vector3:
    """Parse a persisted three-component coordinate.

    Args:
        values: The persisted sequence.

    Returns:
        The coordinate as the shared ``Vector3`` primitive.

    Raises:
        SpatialSupportArtifactError: If the sequence does not hold three components.
    """
    components = [float(value) for value in values]
    if len(components) != 3:
        raise SpatialSupportArtifactError(
            f"a coordinate must have three components, got {len(components)}"
        )
    return (components[0], components[1], components[2])
