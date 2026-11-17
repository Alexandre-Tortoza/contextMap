"""Physical layout, schema version and byte format of the Point Representation run artifact.

One place declares *where* each file of a run lives and *how* its bytes are formed, because
the writer and the reader must agree byte for byte: ``outputs/representation-index.jsonl``
stores offsets into ``outputs/representations.jsonl``, so a change of separator or key
ordering in the encoding below would break reading a representation by identity.

This module reads and writes nothing: it only declares the physical contract and the
artifact's error vocabulary, and imports nothing from the capability.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

SCHEMA_VERSION = "0.2.0"
"""Point Representation run artifact schema version written and understood by this module."""

MANIFEST = "manifest.json"
REPRESENTATIONS = "outputs/representations.jsonl"
REPRESENTATION_INDEX = "outputs/representation-index.jsonl"
SPACES = "outputs/representation-spaces.json"
GEOMETRY_INDEX = "outputs/geometry-representation-index.jsonl"
FAILED_SUPPORTS = "outputs/failed-supports.jsonl"
COUNTS = "metrics/counts.json"
SUPPORT_SIZE = "metrics/support-size.json"
NORMS = "metrics/norms.json"
RUNTIME = "metrics/runtime.json"

DTYPES = {"float32": ("f", 4, "f32"), "float64": ("d", 8, "f64")}
"""``struct`` code, item size in bytes and file suffix per payload dtype."""

TRACE_LIMIT = 20
"""Support traces kept at the ``standard`` and ``full`` debug levels."""


class RunArtifactError(Exception):
    """Base class for Point Representation run artifact read/write failures."""


class IncompleteRunArtifactError(RunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class PointRepresentationDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds representative support traces and a norm per representation.
        FULL: Adds per-component statistics of the stored vectors.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


def encode_document(record: Mapping[str, Any]) -> str:
    """Encode one standalone JSON document of the artifact, indented for reading.

    Args:
        record: The document to encode.

    Returns:
        The encoded text, with keys sorted and newline-terminated.
    """
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def encode_lines(lines: Iterable[str]) -> str:
    """Join already-encoded JSON Lines records into the text of one ``.jsonl`` file.

    Args:
        lines: The encoded records, without their line break.

    Returns:
        The records, each followed by a newline; empty when there are none.
    """
    return "".join(f"{line}\n" for line in lines)
