"""Vocabulário de problemas de uma configuração de runtime.

Um :class:`ConfigProblem` é a unidade que toda verificação de configuração devolve, e um
:class:`ConfigurationError` carrega todos os problemas encontrados de uma vez, para que o
usuário corrija tudo numa só passada. O vocabulário vive em um módulo próprio porque é
consumido além da configuração (seleção de upstream, preflight, serviço de ingestion), e
esses consumidores precisam dele sem depender do schema nem da resolução.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ConfigProblem:
    """One reason a configuration cannot be used as written.

    Attributes:
        path: Dotted location of the offending entry, empty when it concerns the
            configuration as a whole.
        message: What is wrong and, when known, how to fix it.
    """

    path: str
    message: str

    def __str__(self) -> str:
        """Render the problem as ``path: message``."""
        return f"{self.path}: {self.message}" if self.path else self.message


class ConfigurationError(Exception):
    """Raised when a runtime configuration is invalid or cannot be resolved.

    Attributes:
        problems: Every problem found, so a user fixes them in one pass.
    """

    def __init__(self, problems: Sequence[ConfigProblem]) -> None:
        """Build the error from every problem found."""
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"invalid runtime configuration:\n{lines}")

    @classmethod
    def single(cls, message: str, *, path: str = "") -> ConfigurationError:
        """Build an error that carries exactly one problem."""
        return cls([ConfigProblem(path=path, message=message)])
