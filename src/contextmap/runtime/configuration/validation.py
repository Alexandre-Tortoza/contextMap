"""Verificações sobre uma configuração resolvida e regras semânticas entre capabilities.

Três níveis de verificação, nenhum carrega modelo: a resolução diz se o documento é
válido e coerente; :func:`check_selection` diz se todo ponto de variação de um estágio
habilitado tem backend, porque o runtime nunca escolhe um backend em nome do usuário; e
:func:`check_availability` diz se os módulos opcionais e os segredos dos backends
selecionados existem, consultando módulos sem importá-los.

Também moram aqui, temporariamente, regras cujo dono semântico é outra capability (o
alcance da associação de sensores e os canais de evidência da fusão semântica). Cada uma
está marcada com um ``FIXME`` que nomeia o dono real e o motivo do adiamento.
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Callable, Mapping

from contextmap.runtime.components import COMPONENTS
from contextmap.runtime.configuration.problems import ConfigProblem
from contextmap.runtime.configuration.schema import (
    ComponentConfig,
    ConfigValue,
    RuntimeConfig,
)


def check_selection(config: RuntimeConfig) -> tuple[ConfigProblem, ...]:
    """Report the variation points of enabled stages that still have no backend.

    The runtime never picks a backend on the user's behalf, so a configuration that
    resolves can still be incomplete. This lightweight check needs neither models nor
    optional modules and is meant to run before any execution.

    Args:
        config: A resolved configuration.

    Returns:
        One problem per unselected variation point; empty when the selection is
        complete.
    """
    problems = []
    for component_id, component in config.components.items():
        problem = check_component_selection(component_id, component)
        if problem is not None:
            problems.append(problem)
    return tuple(problems)


def check_component_selection(
    component_id: str, component: ComponentConfig
) -> ConfigProblem | None:
    """Report a variation point that has no backend selected.

    A component the catalog marks ``optional`` (see :class:`~contextmap.runtime.components.
    ComponentSpec`) is never reported for having no backend: its absence is a valid,
    explicit configuration, not an incomplete one.

    Args:
        component_id: Identity of the variation point, ``"<capability>.<slot>"``.
        component: Its resolved configuration.

    Returns:
        The problem, or ``None`` when a backend is selected or the component is optional.
    """
    if component.backend is not None or COMPONENTS[component_id].optional:
        return None
    supported = ", ".join(sorted(COMPONENTS[component_id].backends))
    capability, slot = component_id.split(".", 1)
    return ConfigProblem(
        path=f"components.{component_id}",
        message=(
            f"no backend selected; choose one of: {supported} "
            f"(components.{capability}.{slot}.backend=<id>)"
        ),
    )


def check_availability(
    config: RuntimeConfig,
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
) -> tuple[ConfigProblem, ...]:
    """Report selected backends whose optional modules or secrets are missing.

    Modules are only looked up, never imported, so this stays cheap and loads no
    model. A secret is reported by name; its value is never read into the report.

    Args:
        config: A resolved configuration.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether a top-level module can be
            imported; defaults to an :mod:`importlib` lookup.

    Returns:
        One problem per missing module or secret; empty when everything is present.
    """
    problems: list[ConfigProblem] = []
    for component_id, component in config.components.items():
        problems.extend(
            check_component_availability(
                component_id, component, environ=environ, module_available=module_available
            )
        )
    return tuple(problems)


def check_component_availability(
    component_id: str,
    component: ComponentConfig,
    *,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    check_modules: bool = True,
) -> tuple[ConfigProblem, ...]:
    """Report what one selected backend is missing, without loading anything.

    Args:
        component_id: Identity of the variation point, ``"<capability>.<slot>"``.
        component: Its resolved configuration.
        environ: Environment to look secrets up in; defaults to ``os.environ``.
        module_available: Predicate telling whether a top-level module can be
            imported; defaults to an :mod:`importlib` lookup.
        check_modules: Whether the backend's optional modules are required. A caller
            that supplies the model runtime itself passes ``False``: the modules are
            then the supplied runtime's concern, not the bundled code's.

    Returns:
        One problem per missing module or secret; empty when nothing is missing or no
        backend is selected.
    """
    if component.backend is None:
        return ()
    environment = os.environ if environ is None else environ
    available = module_available or module_installed
    spec = COMPONENTS[component_id].backends[component.backend]
    path = f"components.{component_id}"
    problems = []
    for module in spec.requires if check_modules else ():
        if not available(module):
            hint = f" ({spec.install_hint})" if spec.install_hint else ""
            problems.append(
                ConfigProblem(
                    path=path,
                    message=(
                        f"backend {component.backend!r} needs the optional module "
                        f"{module!r}, which is not installed{hint}"
                    ),
                )
            )
    for name in spec.secrets:
        if not environment.get(name):
            problems.append(
                ConfigProblem(
                    path=path,
                    message=(
                        f"backend {component.backend!r} needs the environment secret "
                        f"{name}; set it in the environment, it is never stored in "
                        "configuration"
                    ),
                )
            )
    return tuple(problems)


def module_installed(name: str) -> bool:
    """Tell whether a top-level module can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# FIXME: o limiar de alcance pertence a sensor_association, que é quem descarta os elementos
# do mapa além dele; validar o seu domínio é regra dela, não do runtime. Fica aqui enquanto
# sensor_association é refatorada em paralelo e enquanto uma capability não pode importar o
# runtime (tests/architecture/test_boundaries.py); a mudança é adiada até lá.
def association_range(
    section: Mapping[str, ConfigValue], problems: list[ConfigProblem]
) -> float | None:
    """Read the optional candidate range; anything but a positive finite number is a problem."""
    declared = section.get("association_max_range_m")
    if declared is None:
        return None
    if isinstance(declared, bool) or not isinstance(declared, int | float):
        problems.append(
            ConfigProblem(
                path="policies.association_max_range_m",
                message=f"is {declared!r}; expected a distance in meters or no value at all",
            )
        )
        return None
    value = float(declared)
    if not math.isfinite(value) or value <= 0:
        problems.append(
            ConfigProblem(
                path="policies.association_max_range_m",
                message=f"is {value!r}; expected a positive finite distance in meters",
            )
        )
        return None
    return value


# FIXME: os canais de evidência e a política ciente de qualidade pertencem a semantic_fusion;
# ler os canais declarados e recusar um canal que não pode ser alimentado são regras dela.
# Ficam aqui enquanto semantic_fusion é refatorada em paralelo e enquanto uma capability não
# pode importar o runtime (tests/architecture/test_boundaries.py); a mudança é adiada até lá.
def _declared_channels(parameters: Mapping[str, ConfigValue]) -> tuple[ConfigValue, ...]:
    """Read the evidence channels a fusion policy declares, baseline or quality-aware."""
    baseline = parameters.get("baseline")
    scopes = [parameters]
    if isinstance(baseline, Mapping):
        scopes.append(baseline)  # a política ciente de qualidade guarda os canais em `baseline`.
    channels: list[ConfigValue] = []
    for scope in scopes:
        declared = scope.get("channels")
        if isinstance(declared, tuple):
            channels.extend(declared)
    return tuple(channels)


def incompatibilities(config: RuntimeConfig) -> list[ConfigProblem]:
    """Find selections that contradict each other, independent of what is installed."""
    problems = []
    accumulation = config.components.get("semantic_fusion.accumulation")
    # O estágio point_representation roda, mas a fusão não recebe o run dele: o canal
    # nunca seria alimentado, então é recusado em vez de acumular sem evidência 3D.
    if (
        accumulation is not None
        and accumulation.backend is not None
        and "point_representation" in _declared_channels(accumulation.parameters)
    ):
        problems.append(
            ConfigProblem(
                path="components.semantic_fusion.accumulation",
                message=(
                    "the 'point_representation' evidence channel cannot be fed: "
                    "semantic_fusion has no representation input, a consumption contract "
                    "#657 keeps out of scope; drop the channel"
                ),
            )
        )
    return problems
