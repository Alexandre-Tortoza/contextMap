"""Configuração de runtime: schema, resolução em camadas, verificações, segredos e persistência.

A configuração diz *o que executar e com quais implementações*; nunca diz como uma
capability funciona. O subpacote separa as razões de mudança que antes dividiam um único
módulo:

- ``problems``: o vocabulário de problemas (:class:`ConfigProblem`,
  :class:`ConfigurationError`), consumido também fora da configuração;
- ``schema``: o documento tipado, a versão do schema e a serialização que o digest cobre;
  os defaults das políticas são os defaults dos seus campos;
- ``documents``: a camada não tipada — perfil, arquivos JSON/TOML, mescla e overrides;
- ``resolution``: a validação estrutural, a normalização e o digest da configuração
  efetiva; é o único lugar onde defaults se aplicam;
- ``validation``: completude da seleção, disponibilidade de módulos e segredos, e as
  regras entre capabilities que o runtime ainda hospeda;
- ``secrets``: a heurística de nomes de credencial e os segredos lidos do ambiente;
- ``persistence``: gravação e leitura verificada de ``effective_config.json``.

Este ``__init__`` é a superfície que o resto do runtime importa; os módulos acima são
internos ao subpacote.
"""

from contextmap.runtime.configuration.documents import parse_override
from contextmap.runtime.configuration.persistence import (
    EFFECTIVE_CONFIG_FILENAME,
    read_effective_config,
    write_effective_config,
)
from contextmap.runtime.configuration.problems import ConfigProblem, ConfigurationError
from contextmap.runtime.configuration.resolution import (
    LATEST,
    NAMED_PREFIX,
    resolve_effective_config,
)
from contextmap.runtime.configuration.schema import (
    CONFIG_SCHEMA_VERSION,
    DEBUG_LEVELS,
    CameraPreparationConfig,
    ComponentConfig,
    ConfigurationSource,
    EffectiveConfig,
    InputsConfig,
    PipelineConfig,
    PoliciesConfig,
    ResourcesConfig,
    RuntimeConfig,
)
from contextmap.runtime.configuration.secrets import ResolvedSecrets, resolve_secrets
from contextmap.runtime.configuration.validation import (
    check_availability,
    check_component_availability,
    check_component_selection,
    check_selection,
    module_installed,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEBUG_LEVELS",
    "EFFECTIVE_CONFIG_FILENAME",
    "LATEST",
    "NAMED_PREFIX",
    "CameraPreparationConfig",
    "ComponentConfig",
    "ConfigProblem",
    "ConfigurationError",
    "ConfigurationSource",
    "EffectiveConfig",
    "InputsConfig",
    "PipelineConfig",
    "PoliciesConfig",
    "ResolvedSecrets",
    "ResourcesConfig",
    "RuntimeConfig",
    "check_availability",
    "check_component_availability",
    "check_component_selection",
    "check_selection",
    "module_installed",
    "parse_override",
    "read_effective_config",
    "resolve_effective_config",
    "resolve_secrets",
    "write_effective_config",
]
