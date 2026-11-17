"""Artifact de run de Semantic Fusion: persistido, imutável e autodescritivo.

Um ``SemanticFusionRunArtifact`` é um diretório com os suportes de fusão de um run e a
evidência fundida sobre cada um, mais a linhagem, as métricas e a evidência pulada
necessárias para confiar neles. Ver
``src/contextmap/semantic_fusion/docs/artifact.md`` para o layout e os trade-offs deste
primeiro schema.

As responsabilidades ficam separadas: ``layout`` declara o contrato físico, ``manifest``
os contratos de metadata, ``writer`` publica, ``reader`` lê por identidade, ``metrics``
checa e agrega em fluxo, e ``debug`` escreve o que nunca é contratual.
"""

from contextmap.semantic_fusion.run_artifact.layout import (
    SCHEMA_VERSION,
    FusionRunArtifactError,
    IncompleteFusionRunArtifactError,
    SemanticFusionDebugLevel,
    SemanticFusionRunId,
)
from contextmap.semantic_fusion.run_artifact.manifest import (
    FusionRunLineage,
    SemanticFusionRunManifest,
)
from contextmap.semantic_fusion.run_artifact.reader import SemanticFusionRunReader
from contextmap.semantic_fusion.run_artifact.writer import SemanticFusionRunWriter

__all__ = [
    "SCHEMA_VERSION",
    "FusionRunArtifactError",
    "FusionRunLineage",
    "IncompleteFusionRunArtifactError",
    "SemanticFusionDebugLevel",
    "SemanticFusionRunId",
    "SemanticFusionRunManifest",
    "SemanticFusionRunReader",
    "SemanticFusionRunWriter",
]
