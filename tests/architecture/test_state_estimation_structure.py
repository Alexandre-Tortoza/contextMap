"""Regras estruturais internas de ``contextmap.state_estimation``.

As regras de fronteira entre capabilities ficam em ``test_boundaries.py``; aqui ficam as
dependências que só fazem sentido dentro da capability.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from test_boundaries import _imported_modules as scan_imports

CAPABILITY_ROOT = Path(__file__).resolve().parents[2] / "src" / "contextmap" / "state_estimation"


def _imported_modules(module: str) -> set[str]:
    # Reutiliza o scanner de ``test_boundaries``: ele cobre `import x`, `from x import y` e
    # imports relativos, e manter dois scanners com cobertura diferente deixaria regras
    # verdes mesmo diante de uma violação.
    path = CAPABILITY_ROOT / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {name for _, name in scan_imports(path, tree)}


def _scan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> set[str]:
    root = tmp_path / "contextmap" / "state_estimation"
    root.mkdir(parents=True)
    (root / "fixture.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "CAPABILITY_ROOT", root)
    return _imported_modules("fixture.py")


def test_the_scanner_sees_a_plain_import_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sem isto, `import contextmap.state_estimation.backends.fast_lio` passaria despercebido
    # pelas regras de direção de dependência abaixo.
    imported = _scan_fixture(
        tmp_path, monkeypatch, "import contextmap.state_estimation.backends.fast_lio\n"
    )
    assert "contextmap.state_estimation.backends.fast_lio" in imported


def test_the_scanner_resolves_a_relative_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = _scan_fixture(tmp_path, monkeypatch, "from .backends import fast_lio\n")
    assert "contextmap.state_estimation.backends" in imported


def test_the_estimator_port_does_not_depend_on_the_preflight_evaluator() -> None:
    # O vocabulário do port (``GeometryRequirements``) pertence a ``requirements``; o port
    # não deve conhecer quem avalia os requisitos.
    assert "contextmap.state_estimation.preflight" not in _imported_modules("ports.py")


def test_the_deployment_wrapper_imports_nothing_from_contextmap() -> None:
    # O wrapper roda dentro do container ROS 1, onde `contextmap` não está instalado.
    imported = _imported_modules("backends/fast_lio_wrapper.py")
    assert not {name for name in imported if name.split(".")[0] == "contextmap"}


def test_core_modules_do_not_import_a_backend() -> None:
    # A direção é backend -> núcleo; o núcleo só conhece o port. Quem escolhe backend é o runtime.
    core_modules = [*CAPABILITY_ROOT.glob("*.py"), *(CAPABILITY_ROOT / "preflight").glob("*.py")]
    for module in sorted(core_modules):
        imported = _imported_modules(module.relative_to(CAPABILITY_ROOT).as_posix())
        backends = {name for name in imported if ".backends" in name}
        assert not backends, f"{module.name} imports {sorted(backends)}"


def test_the_pose_contracts_depend_on_no_other_module_of_the_capability() -> None:
    imported = _imported_modules("models.py")
    internal = {name for name in imported if name.startswith("contextmap.state_estimation")}
    assert not internal


def test_the_requirements_vocabulary_depends_on_no_other_module_of_the_capability() -> None:
    imported = _imported_modules("requirements.py")
    internal = {name for name in imported if name.startswith("contextmap.state_estimation")}
    assert not internal


PREFLIGHT = "contextmap.state_estimation.preflight"
PREFLIGHT_EVALUATORS = ("calibration", "clocks", "readiness")


def _preflight_siblings(module: str) -> set[str]:
    imported = _imported_modules(f"preflight/{module}.py")
    prefix = f"{PREFLIGHT}."
    return {name.removeprefix(prefix).split(".")[0] for name in imported if name.startswith(prefix)}


def test_the_preflight_contracts_import_no_other_preflight_module() -> None:
    # `models` só descreve; se importasse um avaliador, os contratos passariam a depender de
    # quem os produz.
    assert not _preflight_siblings("models")


def test_the_observation_facts_depend_only_on_the_contracts_they_summarize() -> None:
    assert not _preflight_siblings("facts")


@pytest.mark.parametrize("evaluator", PREFLIGHT_EVALUATORS)
def test_a_preflight_evaluator_does_not_import_the_orchestration_or_another_evaluator(
    evaluator: str,
) -> None:
    # Cada avaliador responde a uma pergunta; quem os combina é `service`.
    forbidden = {"service", *PREFLIGHT_EVALUATORS} - {evaluator}
    assert not _preflight_siblings(evaluator) & forbidden


def test_the_preflight_facade_exposes_only_contracts_and_operations() -> None:
    # Consumidores não dependem de `calibration`, `clocks` ou `readiness` por fora do pacote.
    for module in sorted(CAPABILITY_ROOT.rglob("*.py")):
        relative = module.relative_to(CAPABILITY_ROOT)
        if relative.parts[0] == "preflight":
            continue
        reached = {
            name
            for name in _imported_modules(relative.as_posix())
            if name.startswith(f"{PREFLIGHT}.")
        }
        assert not reached, f"{relative} imports {sorted(reached)}"
