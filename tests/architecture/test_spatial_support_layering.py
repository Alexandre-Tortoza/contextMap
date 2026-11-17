"""As camadas internas de spatial_support, verificadas em vez de apenas documentadas.

O `code-map.md` da capability declara quatro camadas -- vocabulário, regra, persistência e
derivação -- e afirma que nenhuma delas importa a de cima. `test_boundaries.py` só verifica
imports *entre* capabilities, então esta é a rede que impede o run persistido de voltar a
depender da regra de junção.

A regra que paga este arquivo é a última: `derivation.py` é o único módulo autorizado a abrir
artifact de outra capability. `test_boundaries.py` permite `artifact` e `geometric_mapping`
para esta capability inteira, então ele nunca pegaria um reader estrangeiro reaparecendo no
writer -- que é exatamente o estado de que esta capability saiu.
"""

from __future__ import annotations

import ast
from pathlib import Path

CAPABILITY = Path(__file__).resolve().parents[2] / "src" / "contextmap" / "spatial_support"

VOCABULARY = frozenset({"models", "policy"})
RULE = frozenset({"service"})
PERSISTENCE = frozenset({"run_artifact"})
DERIVATION = frozenset({"derivation"})

# O que cada camada pode importar de dentro da capability. A fachada é o único módulo que
# alcança tudo, porque é exatamente o seu trabalho.
ALLOWED: dict[frozenset[str], frozenset[str]] = {
    VOCABULARY: frozenset(),
    RULE: VOCABULARY,
    PERSISTENCE: VOCABULARY | PERSISTENCE,
    DERIVATION: VOCABULARY | RULE | PERSISTENCE | DERIVATION,
}

FACADE = "contextmap.spatial_support"
"""A fachada, proibida de dentro: ela alcança todas as camadas, então passar por ela
esconderia uma dependência para cima. Não pertence a camada nenhuma de `ALLOWED`, portanto
qualquer módulo que a importe vira violação."""

FOREIGN_ARTIFACT_READERS = frozenset(
    {
        "ContextMapArtifactReader",
        "GeometricMapArtifactReader",
        "VoxelAggregationArtifactReader",
    }
)
"""Readers de artifact de outras capabilities: seleção de upstream, não persistência."""


def _layer_of(module: str) -> frozenset[str] | None:
    for layer in ALLOWED:
        if module in layer:
            return layer
    return None


def _internal_imports(path: Path, capability: Path = CAPABILITY) -> set[str]:
    """Nomes de dentro da capability que um módulo importa, em qualquer forma sintática.

    Args:
        path: O arquivo a analisar.
        capability: Raiz do package, para resolver import relativo.

    Returns:
        O primeiro nível de cada import interno, ou :data:`FACADE` quando o módulo passa
        pela fachada.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    prefix = f"{FACADE}."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FACADE:
                    imported.add(FACADE)
                elif alias.name.startswith(prefix):
                    imported.add(alias.name.removeprefix(prefix))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = path.relative_to(capability).parts[:-1]
                if node.level > len(package) + 1:
                    continue
                base = package[: len(package) + 1 - node.level]
                module = ".".join((*base, node.module)) if node.module else ".".join(base)
                imported.add(module or FACADE)
            elif node.module == FACADE or (
                node.module == "contextmap"
                and any(alias.name == "spatial_support" for alias in node.names)
            ):
                imported.add(FACADE)
            elif node.module and node.module.startswith(prefix):
                imported.add(node.module.removeprefix(prefix))
    return imported


def _imported_names(path: Path) -> set[str]:
    """Nomes que um módulo importa de outras capabilities do ContextMap."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module != FACADE
            and node.module.startswith("contextmap")
            and not node.module.startswith(f"{FACADE}.")
        ):
            names.update(alias.name for alias in node.names)
    return names


def _top_level_modules(capability: Path) -> set[str]:
    """Nomes de primeiro nível da capability: módulos soltos e subpackages."""
    modules = {path.stem for path in capability.glob("*.py")}
    modules.update(path.name for path in capability.iterdir() if (path / "__init__.py").is_file())
    return modules


def _layer_violations(capability: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(capability.rglob("*.py")):
        relative = path.relative_to(capability)
        module = relative.parts[0] if len(relative.parts) > 1 else path.stem
        if module == "__init__":
            continue
        layer = _layer_of(module)
        if layer is None:
            continue
        allowed = (
            frozenset({"run_artifact"})
            if relative == Path("run_artifact/__init__.py")
            else ALLOWED[layer]
        )
        for imported in sorted(_internal_imports(path, capability)):
            target = imported.split(".")[0]
            if target not in allowed:
                violations.append(f"{relative} must not import {imported}")
    return violations


def test_no_layer_imports_the_one_above_it() -> None:
    violations = _layer_violations(CAPABILITY)
    assert not violations, "Violações de camada em spatial_support:\n" + "\n".join(violations)


def test_the_vocabulary_depends_on_nothing_of_the_capability() -> None:
    reaching = {
        module: sorted(_internal_imports(CAPABILITY / f"{module}.py"))
        for module in sorted(VOCABULARY)
    }
    assert {module: deps for module, deps in reaching.items() if deps} == {}


def test_only_the_derivation_opens_another_capabilitys_artifact() -> None:
    # Abrir artifact a montante é seleção de upstream, e a recusa de três artifacts que não
    # compartilham um mapa bruto mora junto dela. O run persistido não pode saber quem o
    # alimenta, ou o formato volta a depender da origem.
    opening = {
        str(path.relative_to(CAPABILITY)): sorted(_imported_names(path) & FOREIGN_ARTIFACT_READERS)
        for path in sorted(CAPABILITY.rglob("*.py"))
    }
    assert {module: names for module, names in opening.items() if names} == {
        "derivation.py": [
            "ContextMapArtifactReader",
            "GeometricMapArtifactReader",
            "VoxelAggregationArtifactReader",
        ]
    }


def test_every_module_of_the_capability_has_a_declared_layer() -> None:
    # Um arquivo novo sem camada declarada é uma decisão de ownership que precisa ser tomada
    # aqui e no code-map, não um módulo que escapa da verificação.
    known = VOCABULARY | RULE | PERSISTENCE | DERIVATION | {"__init__"}
    assert _top_level_modules(CAPABILITY) == known
