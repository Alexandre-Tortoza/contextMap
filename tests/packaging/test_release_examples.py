"""The committed examples work on a base install, with no extra and no GPU.

These run in the ``lightweight-install`` job against the wheel installed with NumPy only, which
is the point: issue #188 promises that a clean base installation opens and validates the demo
artifact and that the documented commands are covered by release smoke tests. A README command
that stops working fails here.

The demo artifact is the one the v0.1.0 release shipped, and ``examples/v0.1.0`` stays frozen as
the record of what it shipped. The canonical configuration this code resolves is the current
one, ``examples/perception-dag``: the perception DAG decomposition renamed the
component capabilities
(``components.visual_perception.region_discovery`` became
``components.region_perception.discovery``, and the other three likewise), so the v0.1.0
document no longer resolves at all, and renaming a key in it is a change the frozen-transition
rule below forbids (``docs/perception-dag-decomposition.md``).

The committed documents are frozen. The code may resolve the canonical example to another
effective configuration or plan topology only through a transition recorded under
``[Não lançado]`` in ``CHANGELOG.md`` (``canonical effective configuration <frozen digest> →
<code digest>``, the same rule as the schema versions in ``test_release_scope.py``), and only by
adding to it: every value and every line of the frozen documents must still resolve unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from release_record import unrecorded_changes

from contextmap.artifact import (
    ContextMapArtifactReader,
    Severity,
    ValidationLevel,
    ValidationStatus,
    validate_context_map_artifact,
)
from contextmap.runtime.cli import main as cli

ROOT = Path(__file__).resolve().parents[2]
RELEASED = ROOT / "examples" / "v0.1.0"
"""What the v0.1.0 release shipped, never rewritten: the demo artifact and its own documents."""
PREVIOUS = ROOT / "examples" / "v0.2.0"
"""Earlier corrected template and topology, preserved as a draft snapshot."""
CURRENT = ROOT / "examples" / "perception-dag"
"""The canonical configuration this code resolves, with its generated documents."""
DEMO = RELEASED / "demo" / "context-map"
CONFIG = CURRENT / "canonical-config.toml"
CHANGELOG = ROOT / "CHANGELOG.md"

EFFECTIVE_CONFIGURATION = "canonical effective configuration"
"""Name of the frozen example's effective-configuration digest in a recorded transition."""
PLAN_TOPOLOGY = "canonical plan topology"
"""Name of the frozen example's resolved-plan topology digest in a recorded transition."""


def _changed_values(frozen: object, resolved: object, path: str = "") -> list[str]:
    """Every value of ``frozen`` that ``resolved`` does not hold unchanged, by dotted path.

    ``resolved`` may add keys to any mapping; it may not drop or change one ``frozen`` has.
    Anything that is not a mapping, lists included, is compared as a whole.
    """
    if isinstance(frozen, dict) and isinstance(resolved, dict):
        return [
            changed
            for key, value in frozen.items()
            for changed in (
                _changed_values(value, resolved[key], f"{path}.{key}".lstrip("."))
                if key in resolved
                else [f"{path}.{key}".lstrip(".")]
            )
        ]
    return [] if frozen == resolved else [path]


def _missing_lines(frozen: list[str], resolved: list[str]) -> list[str]:
    """Every line of ``frozen`` that ``resolved`` does not keep, in the frozen order.

    ``resolved`` may insert lines anywhere; it may not drop, change or reorder a frozen one.
    """
    remaining = iter(resolved)
    return [line for line in frozen if not any(candidate == line for candidate in remaining)]


def test_a_base_install_validates_the_demo_artifact_with_no_dependency_paths() -> None:
    """Nothing is passed: the relative hints the manifest recorded must resolve on their own."""
    report = validate_context_map_artifact(DEMO, level=ValidationLevel.FULL)

    assert report.status is ValidationStatus.VERIFIED
    errors = [finding for finding in report.findings if finding.severity is Severity.ERROR]
    assert errors == []
    # Os avisos esperados são só as dependências opcionais ausentes, que não vão no pacote.
    assert {finding.code for finding in report.findings} <= {"dependency.optional_missing"}


def test_the_demo_artifact_exercises_entities_geometry_alternatives_and_relations() -> None:
    """#188 forbids an empty smoke artifact; this pins what it actually carries."""
    reader = ContextMapArtifactReader.open(DEMO, verify_hashes=True)
    entities = list(reader.entities())
    relations = list(reader.relations())

    assert len(entities) == 3
    assert len(relations) == 10
    assert all(entity.geometry_refs for entity in entities)
    assert any(len(entity.semantic_state.hypotheses) > 1 for entity in entities)
    assert {entity.semantic_state.status.value for entity in entities} == {
        "unambiguous",
        "ambiguous",
        "insufficient_evidence",
    }


def test_the_documented_validate_command_reports_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(["validate", str(DEMO)]) == 0

    assert "ok" in capsys.readouterr().out


def test_the_documented_inspect_artifact_command_summarises_the_demo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli(["inspect", "artifact", str(DEMO)]) == 0

    output = capsys.readouterr().out
    assert "schema_version: 0.1.0" in output
    assert "integrity: ok" in output


def test_the_canonical_configuration_resolves_with_no_undocumented_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The example must parse and resolve cleanly, which is an acceptance criterion of #188."""
    assert cli(["inspect", "config", "-c", str(CONFIG), "--json"]) == 0

    document = json.loads(capsys.readouterr().out)
    assert document.get("problems", []) == []
    assert document["config"]["pipeline"]["preset"] == "canonical/1"
    assert (
        document["config"]["components"]["semantic_interpretation"]["interpreter"]["backend"]
        == "qwen"
    )


def test_the_committed_effective_configuration_is_what_the_example_resolves_to(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A committed example that drifts from what the code produces is worse than none.

    The digest covers the whole effective document, so a recorded transition pins exactly the
    document this code resolves; any further drift needs another one.
    """
    assert cli(["inspect", "config", "-c", str(CONFIG), "--json"]) == 0
    resolved = json.loads(capsys.readouterr().out)

    committed = json.loads((CURRENT / "effective-config.json").read_text(encoding="utf-8"))

    # O digest cobre o documento efetivo, não onde o arquivo estava: é ele que prova que o
    # exemplo commitado é o que este código resolve, ou a transição registrada até ele.
    assert not unrecorded_changes(
        {EFFECTIVE_CONFIGURATION: resolved["digest"]},
        {EFFECTIVE_CONFIGURATION: committed["digest"]},
        CHANGELOG.read_text(encoding="utf-8"),
    ), (
        "the frozen example resolves to another effective configuration without its "
        "transition in CHANGELOG.md [Não lançado] (write "
        f"'{EFFECTIVE_CONFIGURATION} <frozen digest> → <code digest>')"
    )
    assert _changed_values(committed["config"], resolved["config"]) == []
    if committed["digest"] == resolved["digest"]:
        assert committed["config"] == resolved["config"]
    assert committed["schema_version"] == resolved["schema_version"]
    # A identidade de uma fonte é um caminho, e um caminho depende de onde a suíte roda; o que
    # é comparável é o hash do conteúdo do arquivo lido.
    assert [source["kind"] for source in committed["sources"]] == [
        source["kind"] for source in resolved["sources"]
    ]
    assert [source["content_hash"] for source in committed["sources"]] == [
        source["content_hash"] for source in resolved["sources"]
    ]


def test_the_committed_resolved_plan_is_what_the_example_resolves_to(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The topology digest covers the whole plan, so it moves only with a recorded transition."""
    assert cli(["inspect", "plan", "-c", str(CONFIG)]) == 0
    resolved = capsys.readouterr().out
    committed = (CURRENT / "resolved-plan.txt").read_text(encoding="utf-8")

    frozen_header, *frozen_body = committed.splitlines()
    header, *body = resolved.splitlines()
    *frozen_topology, frozen_digest = frozen_header.split()
    *topology, digest = header.split()
    assert topology == frozen_topology == ["topology", "canonical/1"]
    assert not unrecorded_changes(
        {PLAN_TOPOLOGY: digest},
        {PLAN_TOPOLOGY: frozen_digest},
        CHANGELOG.read_text(encoding="utf-8"),
    ), (
        "the frozen example resolves to another plan topology without its transition in "
        f"CHANGELOG.md [Não lançado] (write '{PLAN_TOPOLOGY} <frozen digest> → <code digest>')"
    )
    assert _missing_lines(frozen_body, body) == []
    if digest == frozen_digest:
        assert resolved == committed


def test_the_previous_example_to_independent_producers_is_recorded() -> None:
    """Name the transition while keeping the earlier generated documents untouched."""
    previous_config = json.loads((PREVIOUS / "effective-config.json").read_text("utf-8"))
    current_config = json.loads((CURRENT / "effective-config.json").read_text("utf-8"))
    previous_plan = (PREVIOUS / "resolved-plan.txt").read_text("utf-8").splitlines()[0].split()[-1]
    current_plan = (CURRENT / "resolved-plan.txt").read_text("utf-8").splitlines()[0].split()[-1]
    assert not unrecorded_changes(
        {
            EFFECTIVE_CONFIGURATION: current_config["digest"],
            PLAN_TOPOLOGY: current_plan,
        },
        {
            EFFECTIVE_CONFIGURATION: previous_config["digest"],
            PLAN_TOPOLOGY: previous_plan,
        },
        CHANGELOG.read_text("utf-8"),
    )


def test_a_context_map_distributed_without_its_required_lineage_is_invalid(tmp_path: Path) -> None:
    """The README says so, and it is the contract working: the map references, never copies.

    Copying the artifact alone, away from the upstream runs its manifest points at, must be
    reported as invalid rather than silently accepted.
    """
    import shutil

    alone = tmp_path / "context-map"
    shutil.copytree(DEMO, alone)

    report = validate_context_map_artifact(alone, level=ValidationLevel.FULL)

    assert report.status is ValidationStatus.INVALID
    assert {finding.code for finding in report.findings if finding.severity is Severity.ERROR} == {
        "dependency.required_missing"
    }


def test_a_frozen_example_identity_moves_only_with_its_recorded_transition() -> None:
    frozen = {EFFECTIVE_CONFIGURATION: "sha256:aaaa"}
    code = {EFFECTIVE_CONFIGURATION: "sha256:bbbb"}
    unrecorded = "## [Não lançado]\n\n## [0.1.0] - 2026-09-25\n"
    recorded = (
        "## [Não lançado]\n\n### Alterado\n\n"
        "- `canonical effective configuration` `sha256:aaaa` → `sha256:bbbb`: novo componente.\n\n"
        "## [0.1.0] - 2026-09-25\n"
    )

    assert unrecorded_changes(code, frozen, unrecorded) == [
        "canonical effective configuration sha256:aaaa → sha256:bbbb"
    ]
    assert unrecorded_changes(code, frozen, recorded) == []
    # Uma transição para outro valor não cobre o que o código produz hoje.
    assert unrecorded_changes({EFFECTIVE_CONFIGURATION: "sha256:cccc"}, frozen, recorded) == [
        "canonical effective configuration sha256:aaaa → sha256:cccc"
    ]


def test_a_recorded_transition_may_only_add_to_the_frozen_documents() -> None:
    frozen = {"components": {"a": {"backend": "x"}}, "resources": {"device": "cpu"}}

    assert _changed_values(frozen, {**frozen, "extra": 1}) == []
    assert _changed_values(frozen, {**frozen, "components": {"a": {"backend": "x"}, "b": {}}}) == []
    assert _changed_values(frozen, {**frozen, "components": {"a": {"backend": "y"}}}) == [
        "components.a.backend"
    ]
    assert _changed_values(frozen, {"components": frozen["components"]}) == ["resources"]
    assert _missing_lines(["a", "b", "c"], ["a", "x", "b", "c"]) == []
    assert _missing_lines(["a", "b", "c"], ["a", "c", "b"]) == ["c"]
    assert _missing_lines(["a", "b"], ["a", "B"]) == ["b"]
