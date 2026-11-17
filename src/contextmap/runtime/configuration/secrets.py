"""Segredos: nunca na configuração, só no ambiente e só em memória.

Um segredo nunca faz parte de uma configuração persistida nem do seu digest. Este módulo
reúne as duas metades dessa regra: a heurística que reconhece um nome de parâmetro que
denota uma credencial (para recusá-lo durante a resolução, apontando o ambiente), e
:class:`ResolvedSecrets`, que lê do ambiente apenas os nomes que os backends selecionados
declaram no catálogo, nunca mostra um valor em ``repr``/``str`` e sabe redigir esses valores
de um texto antes que ele seja registrado.
"""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Mapping, Sequence

from contextmap.runtime.components import COMPONENTS
from contextmap.runtime.configuration.problems import ConfigurationError
from contextmap.runtime.configuration.schema import RuntimeConfig

_SECRET_TOKENS = frozenset({"secret", "secrets", "password", "passwd", "credential", "credentials"})
_SECRET_PAIRS = frozenset(
    {
        ("api", "key"),
        ("access", "key"),
        ("private", "key"),
        ("secret", "key"),
        ("auth", "token"),
        ("access", "token"),
        ("bearer", "token"),
        ("refresh", "token"),
        ("session", "token"),
    }
)
_NAME_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def looks_like_secret(name: str) -> bool:
    """Tell whether a parameter name denotes a credential rather than a setting."""
    tokens = [token.lower() for token in _NAME_TOKEN.findall(name)]
    if tokens == ["token"] or any(token in _SECRET_TOKENS for token in tokens):
        return True
    return any(pair in _SECRET_PAIRS for pair in itertools.pairwise(tokens))


class ResolvedSecrets:
    """The environment secrets the selected backends declare, held in memory only.

    A value is reachable only through :meth:`get`; ``repr`` and ``str`` show names
    and never values, so a secret cannot leak into a log line or a manifest by
    formatting this object.

    Attributes:
        names: Every secret the selected backends declare, present or not.
        missing: The declared secrets absent from the environment.
    """

    __slots__ = ("_values", "missing", "names")

    def __init__(self, names: Sequence[str], values: Mapping[str, str]) -> None:
        """Keep the declared names and the values found for them."""
        self.names = tuple(names)
        self._values = dict(values)
        self.missing = tuple(name for name in self.names if name not in self._values)

    def get(self, name: str) -> str:
        """Return one secret's value.

        Args:
            name: A declared secret name.

        Returns:
            The value from the environment.

        Raises:
            ConfigurationError: If the secret is undeclared or not in the environment.
        """
        if name not in self._values:
            raise ConfigurationError.single(
                f"secret {name} is not available; set it in the environment"
            )
        return self._values[name]

    def redact(self, text: str) -> str:
        """Replace every held secret value inside ``text`` with a placeholder.

        This is how an error message or an event is cleaned before it is recorded: the value
        never leaves this object, it is only searched for.

        Args:
            text: Any text, for example an exception message.

        Returns:
            ``text`` with each secret value replaced by ``***``.
        """
        for value in sorted(self._values.values(), key=len, reverse=True):
            if value:
                text = text.replace(value, "***")
        return text

    def __repr__(self) -> str:
        """Render names only."""
        return f"ResolvedSecrets(names={list(self.names)}, missing={list(self.missing)})"

    __str__ = __repr__


def resolve_secrets(
    config: RuntimeConfig, *, environ: Mapping[str, str] | None = None
) -> ResolvedSecrets:
    """Collect the secrets the selected backends declare from the environment.

    Args:
        config: A resolved configuration.
        environ: Environment to read; defaults to ``os.environ``.

    Returns:
        The secrets found and the ones missing. Nothing here is persisted.
    """
    environment = os.environ if environ is None else environ
    names: list[str] = []
    for component_id, component in config.components.items():
        if component.backend is None:
            continue
        for name in COMPONENTS[component_id].backends[component.backend].secrets:
            if name not in names:
                names.append(name)
    values = {name: environment[name] for name in names if environment.get(name)}
    return ResolvedSecrets(sorted(names), values)
