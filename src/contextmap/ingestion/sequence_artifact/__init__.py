"""Fachada pública do formato persistente produzido pela ingestion.

O pacote expõe manifest, reader, writer e erros do SequenceArtifact sem obrigar consumidores a
conhecer sua divisão interna. O artifact é a fronteira reutilizável para replay: depois de
publicado, etapas downstream não precisam reabrir a fonte ROS ou o dataset original.
"""

from .sequence_artifact_manifest import (
    SCHEMA_VERSION,
    IncompleteSequenceArtifactError,
    SequenceArtifactError,
    SequenceArtifactFileEntry,
    SequenceArtifactId,
    SequenceArtifactManifest,
)
from .sequence_artifact_reader import IndexEntry, SequenceArtifactReader
from .sequence_artifact_writer import SequenceArtifactWriter

__all__ = [
    "SCHEMA_VERSION",
    "IncompleteSequenceArtifactError",
    "IndexEntry",
    "SequenceArtifactError",
    "SequenceArtifactFileEntry",
    "SequenceArtifactId",
    "SequenceArtifactManifest",
    "SequenceArtifactReader",
    "SequenceArtifactWriter",
]
