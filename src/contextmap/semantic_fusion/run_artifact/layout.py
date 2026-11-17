"""Layout físico, versão de schema e formato de bytes do artifact de Semantic Fusion.

Um único lugar declara *onde* cada arquivo do run vive e *como* seus bytes são escritos,
porque reader, writer, métricas e debug precisam concordar byte a byte: os índices
guardam deslocamentos dentro dos arquivos de ``outputs/``, então uma mudança de separador
ou de ordenação de chaves invalidaria a leitura por identidade.

Dados contratuais ficam em ``outputs/`` e ``metrics/`` e são inventariados com tamanho e
SHA-256 no manifest. ``__DEBUG__/`` é evidência humana, nunca inventariada: removê-lo não
pode invalidar o run, e nenhum estágio a jusante pode depender dele.

Este módulo não lê nem escreve arquivo: ele só declara o contrato físico e os erros.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from typing import Any, NewType

SCHEMA_VERSION = "0.4.0"
"""Semantic Fusion run artifact schema version written and understood by this module.

``0.4.0`` records regional, visual and semantic producers independently, without a
monolithic perception-result identity. Inference totals count distinct semantic artifact
and physical-observation pairs. ``0.3.0`` introduced raw semantic-score evidence and its
scoring-run lineage. ``0.2.0`` also redefined ``metrics/counts.json``'s ``inference_results``:
``0.1.0`` summed the results of every support, counting a result whose regions fall in several
supports once per support; ``0.2.0`` counts the distinct results of the whole run, like
``physical_observations``. This is a pre-1.0 schema: no compatibility reader for older schemas
is kept and such a run is refused when opened.
"""

SemanticFusionRunId = NewType("SemanticFusionRunId", str)
"""Identity of one Semantic Fusion run."""

MANIFEST = "manifest.json"
SUPPORTS = "outputs/fusion-supports.jsonl"
EVIDENCE = "outputs/fused-evidence.jsonl"
SUPPORT_INDEX = "outputs/support-observation-index.jsonl"
HYPOTHESIS_INDEX = "outputs/hypothesis-evidence-index.jsonl"
GROUPS = "outputs/physical-observation-groups.jsonl"
CONTRIBUTION_INDEX = "outputs/contribution-index.jsonl"
EXCLUDED = "outputs/excluded-observations.jsonl"
COUNTS = "metrics/counts.json"
DISTRIBUTIONS = "metrics/distributions.json"
PAYLOAD = "metrics/payload.json"
RUNTIME = "metrics/runtime.json"

STREAMED_OUTPUTS = (
    SUPPORTS,
    EVIDENCE,
    SUPPORT_INDEX,
    HYPOTHESIS_INDEX,
    GROUPS,
    CONTRIBUTION_INDEX,
    EXCLUDED,
)

DEBUG_SUPPORT_LIMIT = 20
"""Supports that get debug evidence at the ``standard`` level; ``full`` covers all of them."""


class FusionRunArtifactError(Exception):
    """Base class for Semantic Fusion run artifact read/write failures."""


class IncompleteFusionRunArtifactError(FusionRunArtifactError):
    """Raised when a directory does not contain a complete, valid run artifact."""


class SemanticFusionDebugLevel(Enum):
    """How much non-contractual debug evidence to persist.

    Attributes:
        NONE: Contractual outputs, lineage and required metrics only.
        STANDARD: Adds a summary, the hypotheses, the conflicts and the physical observations
            of a sample of supports.
        FULL: The same for every support, plus a per-contribution trace and a geometry summary.
    """

    NONE = "none"
    STANDARD = "standard"
    FULL = "full"


def encode_line(record: Mapping[str, Any]) -> bytes:
    """Encode one JSON Lines record exactly as the artifact stores it.

    Keys are sorted and separators are tight, so equivalent records are byte-identical and
    the offsets the indexes record stay valid. ``NaN`` and infinities are refused rather
    than written as non-standard JSON.

    Args:
        record: The record to encode.

    Returns:
        The encoded bytes, without the line break.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def encode_document(record: Mapping[str, Any]) -> str:
    """Encode one standalone JSON document of the artifact, indented for reading.

    Args:
        record: The document to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
