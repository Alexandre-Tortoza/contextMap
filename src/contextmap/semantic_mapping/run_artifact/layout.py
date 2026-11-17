"""O contrato físico de um run de Semantic Mapping: caminhos, versão de schema e formato de bytes.

Um lugar só declara **onde** cada arquivo de um run mora e **como** os seus bytes são formados,
porque o writer e o reader precisam concordar byte a byte: `outputs/entity-index.jsonl` guarda
deslocamentos dentro de `outputs/entities.jsonl`, então mudar um separador ou a ordem das chaves da
codificação abaixo quebraria a leitura de uma entidade por identidade.

Este módulo não lê nem escreve nada: declara o contrato físico e o vocabulário de erro do artifact,
e não importa nada da capability.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, NewType

SCHEMA_VERSION = "0.2.0"
"""Semantic Mapping run artifact schema version written and understood by this module."""

SemanticMappingRunId = NewType("SemanticMappingRunId", str)
"""Identity of one Semantic Mapping run."""

MANIFEST = "manifest.json"
ENTITIES = "outputs/entities.jsonl"
ENTITY_INDEX = "outputs/entity-index.jsonl"
GEOMETRY_INDEX = "outputs/entity-geometry-index.jsonl"
EVIDENCE_INDEX = "outputs/entity-evidence-index.jsonl"
OBSERVATION_INDEX = "outputs/entity-observation-index.jsonl"
SEMANTIC_STATE = "outputs/entity-semantic-state.jsonl"
TEMPORAL_STATE = "outputs/entity-temporal-state.jsonl"
REJECTED = "outputs/rejected-candidates.jsonl"
COUNTS = "metrics/counts.json"
DISTRIBUTIONS = "metrics/distributions.json"
PAYLOAD = "metrics/payload.json"
RUNTIME = "metrics/runtime.json"

DEBUG_ENTITY_LIMIT = 20
"""Entities that get debug evidence at the ``standard`` level; ``full`` covers all of them."""


class MappingRunArtifactError(Exception):
    """Base class for Semantic Mapping run artifact read/write failures."""


class IncompleteMappingRunArtifactError(MappingRunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class MappingDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds a summary, the geometry, the semantic state, the evidence links and the
            temporal history of a sample of entities.
        FULL: The same for every entity.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


def encode_line(record: Mapping[str, Any]) -> bytes:
    """Encode one JSONL record: compact, key-sorted and refusing non-finite floats."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def encode_document(record: Mapping[str, Any]) -> str:
    """Encode one standalone JSON document: indented, key-sorted and newline-terminated."""
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
