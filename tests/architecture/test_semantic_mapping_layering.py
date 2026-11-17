"""As camadas internas de semantic_mapping, verificadas em vez de apenas documentadas.

O `code-map.md` da capability declara cinco camadas -- vocabulário, contratos, políticas,
auditoria e persistência -- e afirma que nenhuma delas importa a de cima. `test_boundaries.py`
só verifica imports *entre* capabilities, então esta é a rede que impede um contrato de voltar
a depender de uma política ou de um codec.

Duas regras desta capability merecem destaque, porque foram decisões da refatoração e não
consequências óbvias:

* **auditoria** (`evidence_validation`, `evidence_trace`) é uma camada própria, e não uma
  política. Ela confere e percorre referências de evidência *já persistida*, então a
  persistência precisa alcançá-la -- `reader.validate_references()` existe por isso -- enquanto
  ela não pode alcançar nenhuma regra científica;
* **persistência nunca importa política.** É por isso que `CandidateRejection` vive em
  `entity.py` e `resolve_geometry` em `geometry.py`: ambas são alcançadas pela camada de fora.

A fachada também é verificada aqui, mas na direção oposta à de `test_boundaries.py`: de fora
ela é a única porta, de dentro ela é proibida, porque alcança todas as camadas e deixaria um
contrato chegar à camada de cima sem nomeá-la.
"""

from __future__ import annotations

import ast
from pathlib import Path

CAPABILITY = Path(__file__).resolve().parents[2] / "src" / "contextmap" / "semantic_mapping"

VOCABULARY = frozenset({"invariants", "fingerprint", "support_shape", "fused_evidence_source"})
CONTRACTS = frozenset({"entity", "geometry", "semantic_state", "evidence", "temporal"})
POLICIES = frozenset(
    {
        "geometry_summary",
        "state_mapping",
        "primary_hypothesis",
        "interaction_radius",
        "materialization",
    }
)
AUDIT = frozenset({"evidence_validation", "evidence_trace"})
PERSISTENCE = frozenset({"serialization", "run_artifact"})

# O que cada camada pode importar de dentro da capability. A fachada é o único módulo que
# alcança tudo, porque é exatamente o seu trabalho.
ALLOWED: dict[frozenset[str], frozenset[str]] = {
    VOCABULARY: frozenset(),
    CONTRACTS: VOCABULARY | CONTRACTS,
    POLICIES: VOCABULARY | CONTRACTS | POLICIES,
    AUDIT: VOCABULARY | CONTRACTS | AUDIT,
    PERSISTENCE: VOCABULARY | CONTRACTS | AUDIT | PERSISTENCE,
}

LEAVES = frozenset({"invariants", "fingerprint", "support_shape", "fused_evidence_source"})
"""Módulos que não podem importar nada de dentro da capability."""

FACADE = "contextmap.semantic_mapping"
"""A fachada, proibida de dentro: ela alcança todas as camadas, então passar por ela
esconderia uma dependência para cima. Não pertence a camada nenhuma de `ALLOWED`, portanto
qualquer módulo que a importe vira violação."""


def test_import_syntax_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "geometry.py"
    path.write_text(
        "import contextmap.semantic_mapping.materialization\n"
        "from contextmap.semantic_mapping.state_mapping import derive_ambiguity_state\n",
        encoding="utf-8",
    )
    assert _internal_imports(path, tmp_path) == {"materialization", "state_mapping"}


def test_the_facade_is_recognised_in_every_form_it_can_be_written(tmp_path: Path) -> None:
    path = tmp_path / "geometry.py"
    path.write_text(
        "import contextmap.semantic_mapping\n"
        "from contextmap import semantic_mapping\n"
        "from contextmap.semantic_mapping import entity\n"
        "from . import invariants\n",
        encoding="utf-8",
    )
    assert _internal_imports(path, tmp_path) == {FACADE}


def test_the_facade_cannot_be_imported_from_inside_the_capability(tmp_path: Path) -> None:
    # De dentro, o import é sempre direto ao módulo dono.
    (tmp_path / "geometry.py").write_text(
        "from contextmap.semantic_mapping import invariants\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == [f"geometry.py must not import {FACADE}"]


def test_a_contract_cannot_import_the_policy_that_fills_it(tmp_path: Path) -> None:
    (tmp_path / "geometry.py").write_text(
        "from contextmap.semantic_mapping.geometry_summary import summarize_geometry\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == ["geometry.py must not import geometry_summary"]


def test_persistence_cannot_import_a_policy(tmp_path: Path) -> None:
    # A razão de `CandidateRejection` morar em `entity.py`: o writer precisa dela para gravar
    # `rejected-candidates.jsonl`, e alcançá-la pela materialização inverteria a direção.
    artifact = tmp_path / "run_artifact"
    artifact.mkdir()
    (artifact / "writer.py").write_text(
        "from contextmap.semantic_mapping.materialization import CandidateRejection\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == ["run_artifact/writer.py must not import materialization"]


def test_audit_cannot_import_a_policy(tmp_path: Path) -> None:
    # A razão de `resolve_geometry` morar no contrato: a validação precisa resolver geometria
    # para relatar `missing_geometry`, sem alcançar a política que a resume.
    (tmp_path / "evidence_validation.py").write_text(
        "from contextmap.semantic_mapping.geometry_summary import summarize_geometry\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == [
        "evidence_validation.py must not import geometry_summary"
    ]


def test_artifact_modules_are_checked_recursively(tmp_path: Path) -> None:
    artifact = tmp_path / "run_artifact"
    artifact.mkdir()
    (artifact / "reader.py").write_text(
        "from contextmap.semantic_mapping.state_mapping import derive_ambiguity_state\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == ["run_artifact/reader.py must not import state_mapping"]


def test_relative_imports_cannot_bypass_artifact_layer(tmp_path: Path) -> None:
    artifact = tmp_path / "run_artifact"
    artifact.mkdir()
    (artifact / "writer.py").write_text(
        "from ..materialization import EntityMaterializationPolicy\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == ["run_artifact/writer.py must not import materialization"]


def test_artifact_facade_only_exports_artifact_modules(tmp_path: Path) -> None:
    artifact = tmp_path / "run_artifact"
    artifact.mkdir()
    (artifact / "__init__.py").write_text(
        "from contextmap.semantic_mapping.materialization import materialize_entities\n",
        encoding="utf-8",
    )
    assert _layer_violations(tmp_path) == [
        "run_artifact/__init__.py must not import materialization"
    ]


def _layer_of(module: str) -> frozenset[str] | None:
    for layer in ALLOWED:
        if module in layer:
            return layer
    return None


def _internal_imports(path: Path, capability: Path = CAPABILITY) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    prefix = "contextmap.semantic_mapping."
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
                and any(alias.name == "semantic_mapping" for alias in node.names)
            ):
                imported.add(FACADE)
            elif node.module and node.module.startswith(prefix):
                imported.add(node.module.removeprefix(prefix))
    return imported


def _top_level_modules(capability: Path) -> set[str]:
    """Nomes de primeiro nível da capability: módulos soltos e subpackages.

    Args:
        capability: Raiz do package da capability.

    Returns:
        O stem de cada ``.py`` na raiz, mais o nome de cada subpackage.
    """
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
    assert not violations, "Violações de camada em semantic_mapping:\n" + "\n".join(violations)


def test_the_leaf_modules_depend_on_nothing_of_the_capability() -> None:
    reaching = {
        module: sorted(_internal_imports(CAPABILITY / f"{module}.py")) for module in sorted(LEAVES)
    }
    assert {module: deps for module, deps in reaching.items() if deps} == {}


def test_the_artifact_package_is_the_outermost_layer() -> None:
    # O artifact pode conhecer contratos, auditoria e codec; o contrário tornaria o contrato
    # dependente do seu formato persistido.
    inward = {
        path.stem: sorted(
            item for item in _internal_imports(path) if item.startswith("run_artifact")
        )
        for path in sorted(CAPABILITY.glob("*.py"))
        if path.stem != "__init__"
    }
    assert {module: deps for module, deps in inward.items() if deps} == {}


def test_every_module_of_the_capability_has_a_declared_layer() -> None:
    # Um arquivo novo sem camada declarada é uma decisão de ownership que precisa ser tomada
    # aqui e no code-map, não um módulo que escapa da verificação.
    known = VOCABULARY | CONTRACTS | POLICIES | AUDIT | PERSISTENCE | {"__init__"}
    assert _top_level_modules(CAPABILITY) == known


def test_a_new_subpackage_cannot_escape_the_layer_declaration(tmp_path: Path) -> None:
    # Sem isto, `semantic_mapping/foo/bar.py` sairia de `_layer_violations()` sem camada e
    # sem ninguém notar; só um `.py` solto na raiz seria detectado.
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "codecs").mkdir()
    (tmp_path / "codecs" / "__init__.py").write_text("", encoding="utf-8")
    assert _top_level_modules(tmp_path) == {"__init__", "codecs"}
