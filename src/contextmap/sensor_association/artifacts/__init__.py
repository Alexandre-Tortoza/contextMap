"""Fachada interna da persistência de Sensor Association.

Expõe formato, leitura e escrita; as regras científicas permanecem fora deste pacote.
"""

from .run_manifest import (
    SCHEMA_VERSION,
    IncompleteRunArtifactError,
    RunArtifactError,
    SensorAssociationDebugLevel,
    SensorAssociationRunId,
    SensorAssociationRunManifest,
)
from .run_reader import SensorAssociationRunReader
from .run_writer import SensorAssociationRunWriter

__all__ = [
    "SCHEMA_VERSION",
    "IncompleteRunArtifactError",
    "RunArtifactError",
    "SensorAssociationDebugLevel",
    "SensorAssociationRunId",
    "SensorAssociationRunManifest",
    "SensorAssociationRunReader",
    "SensorAssociationRunWriter",
]
