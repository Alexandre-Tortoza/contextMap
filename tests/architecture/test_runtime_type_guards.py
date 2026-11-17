"""Um guard de tipo não pode depender de um nome que só existe para o type checker.

Duas vezes nesta branch a decomposição resolveu um ciclo de import pondo o contrato sob
``if TYPE_CHECKING`` e deixando um ``isinstance`` logo abaixo:

- ``perception_result_assembly`` ainda declarava
  ``SemanticInterpretationExecution = None`` como stand-in de runtime, e o ``isinstance``
  passava a levantar ``TypeError: isinstance() arg 2 must be a type``;
- ``pipeline_resolution._run_semantic_interpreter`` só tinha
  ``SemanticInterpretationRequest`` sob ``TYPE_CHECKING``, e o ``isinstance`` levantava
  ``NameError`` em toda execução — a isolação de falha do grafo registrava o estágio como
  ``FAILED`` e nenhuma run com semântica produzia claim.

Os dois são invisíveis ao lint e ao type checker: `mypy` vê o nome, `ruff` vê o import usado,
e o erro só aparece quando a linha roda. Este teste olha o que o lint não olha — o nome usado
em posição de runtime versus onde ele foi importado — em todo `src/`.

Um import tardio dentro da função é a forma correta e é aceita aqui: o ciclo já está resolvido
no momento da chamada.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"

_RUNTIME_CALLS = frozenset({"isinstance", "issubclass"})


def _type_checking_imports(tree: ast.Module) -> set[str]:
    """Return the names imported only inside an ``if TYPE_CHECKING:`` block."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        guarded = getattr(test, "id", None) == "TYPE_CHECKING" or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not guarded:
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.ImportFrom | ast.Import):
                names.update(alias.asname or alias.name for alias in statement.names)
    return names


def _module_level_names(tree: ast.Module) -> set[str]:
    """Return the names bound at module level outside a ``TYPE_CHECKING`` block."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
    return names


def _local_imports(scope: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return the names a function imports in its own body."""
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _runtime_type_names(tree: ast.Module) -> list[tuple[int, str]]:
    """Return every (line, name) used as a type in an isinstance/issubclass check."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if getattr(function, "id", getattr(function, "attr", "")) not in _RUNTIME_CALLS:
            continue
        if len(node.args) != 2:
            continue
        for inner in ast.walk(node.args[1]):
            if isinstance(inner, ast.Name):
                found.append((inner.lineno, inner.id))
    return found


def _enclosing_scopes(
    tree: ast.Module,
) -> list[tuple[int, int, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return every function's (first line, last line, node), innermost resolvable by span."""
    scopes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            last = max(
                (getattr(child, "lineno", node.lineno) for child in ast.walk(node)),
                default=node.lineno,
            )
            scopes.append((node.lineno, last, node))
    return scopes


def test_no_isinstance_depends_on_a_type_checking_only_name() -> None:
    """Every type in an isinstance check must be bound at runtime where it is used."""
    problems: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deferred = _type_checking_imports(tree)
        if not deferred:
            continue
        available = _module_level_names(tree)
        scopes = _enclosing_scopes(tree)
        for line, name in _runtime_type_names(tree):
            if name not in deferred or name in available:
                continue
            in_scope = any(
                start <= line <= end and name in _local_imports(scope)
                for start, end, scope in scopes
            )
            if not in_scope:
                problems.append(
                    f"{path.relative_to(SOURCE_ROOT)}:{line}: isinstance uses {name!r}, "
                    "which is imported only under TYPE_CHECKING and not in scope at runtime"
                )
    assert not problems, "Runtime type guards on deferred names:\n" + "\n".join(problems)


def test_no_module_assigns_none_as_a_runtime_stand_in_for_a_deferred_type() -> None:
    """``Name = None`` in the ``else`` of a TYPE_CHECKING block silently disables a guard."""
    problems: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deferred = _type_checking_imports(tree)
        if not deferred:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in deferred:
                    problems.append(
                        f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {target.id!r} is "
                        "assigned None as a runtime stand-in for a deferred type; import it "
                        "late inside the function that needs it instead"
                    )
    assert not problems, "Deferred types stubbed with None:\n" + "\n".join(problems)
