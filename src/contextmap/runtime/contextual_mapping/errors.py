"""Erros do mapeamento contextual incremental, um por elo da cadeia que ele constrói.

Fundação espacial, ContextRun, ContextBranch e ContextBuild recusam cada um o próprio registro
ou a própria operação com o próprio erro, para que quem chama saiba qual elo falhou. Ficam
fora de ``runtime/errors.py`` de propósito: aqueles são erros de composição e de execução do
DAG, que este subdomínio apenas usa.
"""

from __future__ import annotations

from collections.abc import Sequence


class SpatialFoundationError(ValueError):
    """The three artifacts do not form one spatial foundation.

    Attributes:
        problems: Every problem found, each prefixed with the role it concerns.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Build the error from every problem found."""
        self.problems = tuple(problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"the artifacts do not form one spatial foundation:\n{listed}")


class ContextRunError(ValueError):
    """A ContextRun record cannot be published or read as it stands."""


class ContextBranchError(ValueError):
    """A branch cannot be created, opened or grown as asked."""


class ContextBuildError(ValueError):
    """A build cannot be frozen, published or read as asked."""
