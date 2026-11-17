"""Opções configuráveis do runtime: o que pode ser selecionado e como os estágios se ligam.

O subpacote reúne três coisas que mudam por razões independentes, cada uma no seu módulo:

- ``contracts``: os nomes dos contratos que atravessam as arestas do DAG -- uma folha que não
  importa nada; quem só precisa desse vocabulário o importa de lá diretamente;
- ``components``: o registro de pontos de variação e backends, com seus fatos de implantação
  (módulos opcionais, dica de instalação, segredos, parâmetro de dispositivo);
- ``presets``: as topologias versionadas (estágios, fiação, presets e identidades de perfil).

O nome é ``components``, não ``catalog``, de propósito: ``selection`` já usa ``catalog`` para
os runs que de fato foram produzidos (``ArtifactCatalog``, ``StaticCatalog``,
``load_catalog``); aqui estão as opções que podem ser configuradas.
"""

from contextmap.runtime.components.components import COMPONENTS, BackendSpec, ComponentSpec
from contextmap.runtime.components.presets import (
    CANONICAL2_PRESET,
    CANONICAL2_PROFILE_ID,
    CANONICAL_PRESET,
    CANONICAL_PROFILE_ID,
    PRESETS,
    Interception,
    RuntimePreset,
    StageDeclaration,
    StageInput,
)

__all__ = [
    "CANONICAL2_PRESET",
    "CANONICAL2_PROFILE_ID",
    "CANONICAL_PRESET",
    "CANONICAL_PROFILE_ID",
    "COMPONENTS",
    "PRESETS",
    "BackendSpec",
    "ComponentSpec",
    "Interception",
    "RuntimePreset",
    "StageDeclaration",
    "StageInput",
]
