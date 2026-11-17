"""Artifact de run de Spatial Support: persistido, imutável e autodescritivo.

Um ``SemanticSpatialSupportArtifact`` é um diretório com uma :class:`SpatialCell` por voxel
ocupado que alguma entidade suporta, mais as identidades dos três artifacts a montante de
que foi derivado. Ele abre sem NumPy e sem o mapa bruto: o consumidor lê as células e alinha
cada uma ao voxel renderizado pelo ``aggregate_index``, nunca por índice de ponto.

Dado contratual vive em ``outputs/`` e ``metrics/`` e é inventariado com tamanho e SHA-256 no
manifest. A escrita segue :class:`~contextmap.shared.AtomicRunDirectory`: uma escrita
interrompida nunca se parece com um run terminado, e um run terminado nunca é modificado.

As responsabilidades ficam separadas: ``layout`` declara o contrato físico, os erros e o
codec de células, ``manifest`` o contrato de metadata e a recusa de schema, ``writer``
publica o run e ``reader`` o lê de volta.

Este pacote é interno à capability: a superfície cross-capability é
``contextmap.spatial_support``. Ele não possui a regra de junção nem a seleção dos artifacts
a montante.
"""

from contextmap.spatial_support.run_artifact.layout import (
    SCHEMA_VERSION,
    SpatialSupportArtifactError,
)
from contextmap.spatial_support.run_artifact.manifest import (
    SpatialSupportManifest,
    SpatialSupportRunId,
)
from contextmap.spatial_support.run_artifact.reader import (
    SpatialSupportArtifactReader,
)
from contextmap.spatial_support.run_artifact.writer import SpatialSupportArtifactWriter

__all__ = [
    "SCHEMA_VERSION",
    "SpatialSupportArtifactError",
    "SpatialSupportArtifactReader",
    "SpatialSupportArtifactWriter",
    "SpatialSupportManifest",
    "SpatialSupportRunId",
]
