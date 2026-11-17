"""The code identity every run of the incremental real-data harnesses records (#501, #504, #665).

``resolve_code_identity`` in the #501 ``steps.py`` is the one rule the three drivers share: a
real run is evidence, so it runs exactly the commit it records, from a clean checkout, and
records that commit's full SHA whatever form named it; a dry run (CI, development) may run from
a dirty tree or outside git and records what it is given. The rule is exercised here against
temporary git repositories, each isolated from any repository enclosing the test's directory.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from incremental_harness import PARTITION_SCRIPTS, REPOSITORY, commit_all, run_git

REAL = "real"
DRY_RUN = "dry_run"


def _steps() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "steps", REPOSITORY / PARTITION_SCRIPTS / "steps.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Uma dataclass resolve o próprio módulo em sys.modules durante a execução dele.
    sys.modules["steps"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository with one commit and a clean tree."""
    # Sem o teto, o git subiria até um repositório que contivesse o diretório do teste.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "code.py").write_text("print('first')\n", encoding="utf-8")
    run_git(checkout, "init", "--quiet")
    commit_all(checkout, "first")
    return checkout


def _resolve(checkout: Path, evidence_class: str, requested: str | None = None) -> str:
    steps = _steps()
    identity: str = steps.resolve_code_identity(
        steps.git_identity(checkout),
        checkout=checkout,
        evidence_class=evidence_class,
        requested=requested,
    )
    return identity


def _head(checkout: Path) -> str:
    return run_git(checkout, "rev-parse", "HEAD")


# --- a real run ---------------------------------------------------------------------------


def test_a_real_run_from_a_clean_checkout_records_the_full_sha_of_its_commit(
    repository: Path,
) -> None:
    identity = _resolve(repository, REAL)

    assert identity == _head(repository)
    assert re.fullmatch(r"[0-9a-f]{40}", identity)


@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_a_real_run_from_a_dirty_checkout_is_refused(repository: Path, change: str) -> None:
    # Uma mudança não commitada, rastreada ou não, é código que o commit registrado não identifica.
    name = "code.py" if change == "modified" else "new.py"
    (repository / name).write_text("print('local')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dirty") as refused:
        _resolve(repository, REAL)

    assert "commit or stash" in str(refused.value)


@pytest.mark.parametrize("git", ["initialized", "none"])
def test_a_real_run_without_a_commit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: str
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if git == "initialized":
        run_git(checkout, "init", "--quiet")

    with pytest.raises(ValueError, match="no commit") as refused:
        _resolve(checkout, REAL)

    assert "commit or stash" in str(refused.value)


def test_a_real_run_accepts_the_full_sha_of_the_checked_out_commit(repository: Path) -> None:
    head = _head(repository)

    assert _resolve(repository, REAL, head) == head


def test_a_real_run_records_an_abbreviated_sha_of_the_checked_out_commit_in_full(
    repository: Path,
) -> None:
    head = _head(repository)

    # O valor gravado é sempre o SHA completo, nunca a abreviação que o nomeou.
    assert _resolve(repository, REAL, head[:12]) == head


def test_a_real_run_naming_another_commit_is_refused(repository: Path) -> None:
    first = _head(repository)
    second = commit_all(repository, "second")

    with pytest.raises(ValueError, match="--code-identity") as refused:
        _resolve(repository, REAL, first)

    message = str(refused.value)
    assert first in message and second in message
    assert "omit" in message


@pytest.mark.parametrize("named", ["dry-run", "tree"])
def test_a_real_run_naming_something_that_is_not_a_commit_is_refused(
    repository: Path, named: str
) -> None:
    # Uma árvore é um objeto do repositório, mas não um commit: o código não é identificado.
    requested = run_git(repository, "rev-parse", "HEAD^{tree}") if named == "tree" else named

    with pytest.raises(ValueError, match="does not name a commit") as refused:
        _resolve(repository, REAL, requested)

    assert "omit" in str(refused.value)


def test_a_refused_real_run_lists_every_reason_and_how_to_fix_each(repository: Path) -> None:
    (repository / "code.py").write_text("print('local')\n", encoding="utf-8")

    with pytest.raises(ValueError) as refused:
        _resolve(repository, REAL, "dry-run")

    message = str(refused.value)
    assert "dirty" in message and "does not name a commit" in message
    assert "commit or stash" in message and "omit" in message


# --- a dry run ----------------------------------------------------------------------------


def test_a_dry_run_from_a_dirty_checkout_records_its_commit(repository: Path) -> None:
    (repository / "code.py").write_text("print('local')\n", encoding="utf-8")

    assert _resolve(repository, DRY_RUN) == _head(repository)


def test_a_dry_run_records_an_explicit_identity_as_given(repository: Path) -> None:
    (repository / "code.py").write_text("print('local')\n", encoding="utf-8")

    assert _resolve(repository, DRY_RUN, "dry-run") == "dry-run"


def test_a_dry_run_outside_git_records_an_unknown_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    assert _resolve(checkout, DRY_RUN) == "unknown"


def test_an_unknown_evidence_class_is_refused(repository: Path) -> None:
    with pytest.raises(ValueError, match="evidence class"):
        _resolve(repository, "real-ish")
