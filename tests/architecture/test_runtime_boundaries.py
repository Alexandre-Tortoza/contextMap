"""The runtime composes; only its composition root knows other capabilities and backends.

``test_boundaries.py`` lets the runtime import anything, because a composition root has to
name concrete backends. These tests narrow that permission: the composition root may import
capabilities, lazily; the ingestion application service, the stage executors, the spatial
foundation and the check of a selected perception run may import the public root of the
capabilities they run or read (and nothing below
it, so never an adapter or a backend, and never eagerly from the package root); the CLI may
import only the public root of ``contextmap.artifact``, and only inside the one function that
hands a ContextMapArtifact to that capability's own validator (``contextmap validate``); every
other runtime module, meaning configuration, DAG, reuse, selection and lifecycle code, can
never grow a dependency on a capability or on a concrete backend.

Every rule here names a module by its path under the capability, and the scan that feeds them
is recursive: a module inside a subpackage is held to the same rules as a flat one, so growing
a subpackage is never a way around them.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
RUNTIME = SRC / "contextmap" / "runtime"
COMPOSITION_MODULES = frozenset({"composition.py"})
"""The composition root, by module key: the only part of the runtime free to name a capability."""
# Módulos de aplicação que podem importar a raiz pública de uma capability, e só ela.
PUBLIC_ROOT_IMPORTERS = {
    "ingestion_service.py": {"contextmap.ingestion"},
    # Decodifica os assets de máscara de config nas regiões públicas de visual_perception
    # (ValidRegion/ExclusionRegion/InlineMask) que a composição injeta nas duas etapas de
    # preparação de imagem. Só a API pública.
    "image_masks.py": {"contextmap.visual_perception", "contextmap.region_perception"},
    # Um ContextRun pode selecionar um run de percepção em vez de executá-lo; o registro só o
    # referencia depois de ler o manifest dele pelo leitor público e conferir a sequência e a
    # seleção (a identidade de seleção é a regra de ingestion). Import tardio da percepção.
    "contextual_mapping/perception_selection.py": {
        "contextmap.ingestion",
        "contextmap.region_perception",
        "contextmap.visual_perception",
        "contextmap.semantic_interpretation",
    },
    "contextual_mapping/context_run.py": {
        "contextmap.visual_perception",
        "contextmap.semantic_interpretation",
    },
    # A fundação espacial só lê os três artifacts pelos leitores públicos.
    "contextual_mapping/foundation.py": {
        "contextmap.geometric_mapping",
        "contextmap.ingestion",
        "contextmap.state_estimation",
    },
    "executors.py": {
        "contextmap.artifact",
        "contextmap.entity_resolution",
        "contextmap.geometric_mapping",
        "contextmap.ingestion",
        "contextmap.point_representation",
        "contextmap.region_perception",
        "contextmap.semantic_fusion",
        "contextmap.semantic_mapping",
        "contextmap.semantic_interpretation",
        "contextmap.semantic_scoring",
        "contextmap.sensor_association",
        "contextmap.spatial_relations",
        "contextmap.state_estimation",
        "contextmap.visual_perception",
    },
}
CAPABILITIES = frozenset(
    {
        "ingestion",
        "visual_perception",
        "region_perception",
        "semantic_interpretation",
        "semantic_scoring",
        "state_estimation",
        "geometric_mapping",
        "sensor_association",
        "point_representation",
        "semantic_fusion",
        "semantic_mapping",
        "entity_resolution",
        "spatial_relations",
        "spatial_support",
        "artifact",
        "evaluation",
    }
)
BACKEND_PARTS = frozenset({"backends", "adapters", "infrastructure"})
# Exceções restritas a uma função, não ao módulo: o `validate` da CLI delega um
# ContextMapArtifact ao validador do dono (issue #603), e só essa função pode importar a raiz
# pública de `contextmap.artifact`. Fora dela, a CLI continua sem nenhuma capability.
# A derivação do suporte espacial semântico é o segundo caso: a capability não é estágio do
# DAG, então a CLI é o seu caminho de produção, e só o handler do comando pode importá-la.
FUNCTION_SCOPED_IMPORTERS = {
    "cli.py": {
        "_context_map_findings": {"contextmap.artifact"},
        "_derive_spatial_support": {"contextmap.spatial_support"},
    },
}


def _module_level_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Imports that run when the module is imported, skipping ``if TYPE_CHECKING`` blocks."""
    found: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def _all_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def _capability_of(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "contextmap" and parts[1] in CAPABILITIES:
        return parts[1]
    return None


def _modules_under(root: Path) -> list[Path]:
    """Every Python module under ``root``, including the ones inside a subpackage.

    The scan is recursive on purpose: a flat one would let a whole subpackage escape every rule
    in this file. Bytecode caches are not source and are skipped.
    """
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.relative_to(root).parts
    )


def _module_key(path: Path, root: Path = RUNTIME) -> str:
    """Return the key a rule names a module by: its path under the capability, POSIX style."""
    return path.relative_to(root).as_posix()


def _runtime_modules() -> list[Path]:
    return _modules_under(RUNTIME)


def _function_scoped_imports(key: str, tree: ast.Module) -> set[tuple[int, str]]:
    """The imports a module may make only inside the function its exception names."""
    allowed: set[tuple[int, str]] = set()
    for function, modules in FUNCTION_SCOPED_IMPORTERS.get(key, {}).items():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
                allowed.update(item for item in _all_imports(node) if item[1] in modules)
    return allowed


def _capability_violations(key: str, tree: ast.Module) -> list[str]:
    allowed = PUBLIC_ROOT_IMPORTERS.get(key, set())
    scoped = _function_scoped_imports(key, tree)
    return [
        f"{key}:{line} imports {module} ({capability})"
        for line, module in _all_imports(tree)
        if (capability := _capability_of(module)) is not None
        and module not in allowed
        and (line, module) not in scoped
    ]


def test_no_runtime_module_escapes_the_capability_scan(tmp_path: Path) -> None:
    """Every module under the capability is scanned, including one inside a subpackage.

    A flat scan would let a whole subpackage escape every rule in this file, so the scan and
    the key the rules are written against are verified here before they are used.
    """
    (tmp_path / "flat.py").write_text("", encoding="utf-8")
    nested = tmp_path / "group" / "inner"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "stale.py").write_text("", encoding="utf-8")

    found = {_module_key(path, tmp_path) for path in _modules_under(tmp_path)}

    assert found == {"flat.py", "group/inner/deep.py"}


def test_only_the_composition_root_imports_a_capability() -> None:
    violations = []
    for path in _runtime_modules():
        key = _module_key(path)
        if key in COMPOSITION_MODULES:
            continue
        violations.extend(_capability_violations(key, ast.parse(path.read_text(encoding="utf-8"))))

    assert not violations, "\n".join(violations)


def test_the_cli_exception_is_scoped_to_the_context_map_validation() -> None:
    # A permissão é da função do `validate`, não do módulo: o mesmo import em qualquer outro
    # lugar da CLI continua sendo violação.
    elsewhere = ast.parse(
        "from contextmap.artifact import validate_context_map_artifact\n"
        "def _run_command():\n"
        "    from contextmap.artifact import ValidationLevel\n"
        "def _context_map_findings():\n"
        "    from contextmap.artifact import Severity\n"
        "    from contextmap.spatial_relations import Relation\n"
    )

    assert _capability_violations("cli.py", elsewhere) == [
        "cli.py:1 imports contextmap.artifact (artifact)",
        "cli.py:3 imports contextmap.artifact (artifact)",
        "cli.py:6 imports contextmap.spatial_relations (spatial_relations)",
    ]


def test_the_cli_exception_is_scoped_to_the_spatial_support_derivation() -> None:
    # A segunda exceção é da função do comando, não do módulo: a capability não é estágio do
    # DAG, então a CLI a produz, mas só ali.
    elsewhere = ast.parse(
        "from contextmap.spatial_support import SpatialSupportRunId\n"
        "def _validate():\n"
        "    from contextmap.spatial_support import SpatialSupportArtifactReader\n"
        "def _derive_spatial_support():\n"
        "    from contextmap.spatial_support import derive_semantic_spatial_support_artifact\n"
        "    from contextmap.artifact import ContextMapArtifactReader\n"
    )

    assert _capability_violations("cli.py", elsewhere) == [
        "cli.py:1 imports contextmap.spatial_support (spatial_support)",
        "cli.py:3 imports contextmap.spatial_support (spatial_support)",
        "cli.py:6 imports contextmap.artifact (artifact)",
    ]


def test_the_composition_root_imports_capabilities_lazily() -> None:
    eager = [
        f"{key}:{line} imports {module} at module level"
        for key in sorted(COMPOSITION_MODULES)
        for line, module in _module_level_imports(
            ast.parse((RUNTIME / key).read_text(encoding="utf-8"))
        )
        if _capability_of(module) is not None
    ]

    assert not eager, "\n".join(eager)


def test_concrete_backends_are_imported_only_inside_the_factory_that_uses_them() -> None:
    misplaced = []
    for key in sorted(COMPOSITION_MODULES):
        tree = ast.parse((RUNTIME / key).read_text(encoding="utf-8"))
        inside_functions: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                inside_functions.update(
                    child.lineno for child in ast.walk(node) if isinstance(child, ast.ImportFrom)
                )

        for line, module in _all_imports(tree):
            parts = set(module.split("."))
            is_backend = _capability_of(module) is not None and bool(parts & BACKEND_PARTS)
            if is_backend and line not in inside_functions:
                misplaced.append(f"{key}:{line} imports backend {module} outside a factory")

    assert not misplaced, "\n".join(misplaced)


def test_importing_the_runtime_loads_no_backend_no_sdk_and_only_the_ingestion_root() -> None:
    code = (
        "import sys\n"
        "import contextmap.runtime, contextmap.runtime.cli\n"
        f"capabilities = {sorted(CAPABILITIES)!r}\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('contextmap.') "
        "and m.split('.')[1] in capabilities)\n"
        "sdks = [m for m in ('torch', 'transformers', 'rosbags', 'numpy') if m in sys.modules]\n"
        "print('\\n'.join(loaded + sdks))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
    )

    loaded = result.stdout.split()
    assert not [m for m in loaded if not m.startswith("contextmap.ingestion")], loaded
    assert not [m for m in loaded if ".adapters" in m or ".backends" in m], loaded


def test_the_public_api_module_depends_only_on_the_standard_library_and_the_runtime() -> None:
    tree = ast.parse((RUNTIME / "api.py").read_text(encoding="utf-8"))

    foreign = sorted(
        {
            module
            for _, module in _all_imports(tree)
            if module.split(".")[0] not in sys.stdlib_module_names
            and module != "contextmap"
            and not module.startswith("contextmap.runtime")
        }
    )

    # Nenhuma capability, backend, adaptador ou biblioteca de terceiros entra na API pública.
    assert not foreign, foreign


def test_no_runtime_module_depends_on_a_ui_framework() -> None:
    frameworks = {
        "textual",
        "typer",
        "click",
        "rich",
        "prompt_toolkit",
        "urwid",
        "curses",
        "tkinter",
    }
    violations = [
        f"{_module_key(path)}:{line} imports {module}"
        for path in _runtime_modules()
        for line, module in _all_imports(ast.parse(path.read_text(encoding="utf-8")))
        if module.split(".")[0] in frameworks or module.split(".")[0].startswith(("PyQt", "PySide"))
    ]

    assert not violations, "\n".join(violations)


def test_the_public_api_exposes_no_ros_native_type_and_no_backend_class() -> None:
    import contextmap.runtime as runtime
    from contextmap.runtime import api

    contracts = [getattr(api, name) for name in api.__all__ if isinstance(getattr(api, name), type)]
    foreign = {
        f"{contract.__name__}.{annotation}"
        for contract in contracts
        for annotation in getattr(contract, "__annotations__", {}).values()
        if any(token in str(annotation) for token in ("rosbags", "rclpy", "rospy", "torch", "PIL"))
    }

    assert foreign == set()
    assert set(api.__all__) <= set(runtime.__all__)


def test_the_runtime_public_api_hides_the_concrete_backend_classes() -> None:
    import contextmap.runtime as runtime

    exported = set(runtime.__all__)
    concrete = {
        name
        for name in exported
        if name.startswith(("Sam", "Dino", "Clip", "Qwen", "Gemini", "Florence", "FastLio", "PTv3"))
    }

    assert concrete == set()
