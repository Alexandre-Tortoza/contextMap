"""O run de Semantic Mapping persistido: contrato físico, metadata, publicação e leitura.

O package é dividido pelas razões de mudança que o run tem, e não por tamanho:

- `layout` declara o contrato físico — caminhos, versão de schema, erros e formato de bytes — e é
  o que writer e reader precisam compartilhar byte a byte;
- `manifest` declara a metadata do run e a linhagem explícita, e decodifica um manifest lido;
- `tables` monta as tabelas de índice derivadas, que mudam quando um consumidor precisa de coluna;
- `metrics` checa as entidades em fluxo e calcula as figuras que o manifest afirma;
- `writer` publica atomicamente; `reader` abre por identidade;
- `debug` escreve evidência humana em `__DEBUG__/`, que nunca é contratual.

Esta fachada expõe apenas o que a capability publica; os módulos acima não são API cross-capability.
"""

from contextmap.semantic_mapping.run_artifact.layout import (
    SCHEMA_VERSION,
    IncompleteMappingRunArtifactError,
    MappingDebugLevel,
    MappingRunArtifactError,
    SemanticMappingRunId,
)
from contextmap.semantic_mapping.run_artifact.manifest import (
    MappingRunLineage,
    SemanticMappingRunManifest,
    lineage_from_fusion_manifest,
)
from contextmap.semantic_mapping.run_artifact.reader import SemanticMappingRunReader
from contextmap.semantic_mapping.run_artifact.writer import SemanticMappingRunWriter

__all__ = [
    "SCHEMA_VERSION",
    "IncompleteMappingRunArtifactError",
    "MappingDebugLevel",
    "MappingRunArtifactError",
    "MappingRunLineage",
    "SemanticMappingRunId",
    "SemanticMappingRunManifest",
    "SemanticMappingRunReader",
    "SemanticMappingRunWriter",
    "lineage_from_fusion_manifest",
]
