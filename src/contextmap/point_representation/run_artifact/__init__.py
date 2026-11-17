"""Artifact de run de Point Representation: persistido, imutável e autodescritivo.

Um ``PointRepresentationRunArtifact`` é um diretório com as representações que um encoder
produziu sobre um mapa geométrico, mais a linhagem, as métricas e os suportes que falharam
necessários para confiar nelas. Ele abre sem NumPy, biblioteca de modelo nem o próprio mapa
geométrico, e uma representação ou um vetor isolado pode ser lido por identidade ou pela
geometria à qual está ancorado, sem carregar os demais. Ver
``src/contextmap/point_representation/docs/artifact.md`` para o layout e os trade-offs
deste primeiro schema.

Dado contratual vive em ``outputs/`` e ``metrics/`` e é inventariado com tamanho e SHA-256
no manifest; ``__DEBUG__/`` é evidência humana, nunca inventariada, então removê-lo não
invalida o run e nenhum estágio a jusante pode depender dele. A escrita segue
:class:`~contextmap.shared.AtomicRunDirectory`: uma escrita interrompida nunca se parece
com um run terminado, e um run terminado nunca é modificado.

O artifact não copia o mapa geométrico (só as referências e a linhagem que o identificam),
nem feature visual ou claim semântico, e uma representação é persistida como evidência, não
como verdade semântica.

As responsabilidades ficam separadas: ``layout`` declara o contrato físico e os erros,
``manifest`` o contrato de metadata e a identidade dos centros, ``metrics`` calcula as
figuras do run, ``writer`` publica e ``reader`` lê por identidade.
"""

from contextmap.point_representation.run_artifact.layout import (
    SCHEMA_VERSION,
    IncompleteRunArtifactError,
    PointRepresentationDebugLevel,
    RunArtifactError,
)
from contextmap.point_representation.run_artifact.manifest import (
    PointRepresentationRunManifest,
    center_selection_id,
)
from contextmap.point_representation.run_artifact.reader import PointRepresentationRunReader
from contextmap.point_representation.run_artifact.writer import PointRepresentationRunWriter

__all__ = [
    "SCHEMA_VERSION",
    "IncompleteRunArtifactError",
    "PointRepresentationDebugLevel",
    "PointRepresentationRunManifest",
    "PointRepresentationRunReader",
    "PointRepresentationRunWriter",
    "RunArtifactError",
    "center_selection_id",
]
