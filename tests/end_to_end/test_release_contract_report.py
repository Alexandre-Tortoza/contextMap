"""The committed v0.1.0 release acceptance report stays consistent with scenario 1.0.5.

The report decides the release contract's matrix from the evidence milestone #19's campaign
recorded against 1.0.4. CI cannot reproduce that campaign -- its artifacts were pruned -- but it
can refuse a report that drifts from the scenario it claims to be about, that hides a failed
gate, or that quietly drops the caveat saying no hash in it was re-verified.

The scenario it claims to be about is ``solution-1-canonical`` **1.0.5**, and that never moves:
the report decided that matrix with that evidence. So these tests read 1.0.5's committed
snapshot, not the live scenario. Coupling them to ``SCENARIO_VERSION`` would make a later freeze
-- 1.0.6 renamed the profile's component ids and promoted ``region_perception`` to a stage --
look like a drift of a report that is finished.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from contextmap.evaluation import ACCEPTANCE_REPORT_SCHEMA, GateKind

EVALUATION_DOCS = Path(__file__).resolve().parents[2] / "src/contextmap/evaluation/docs"
REPORT = EVALUATION_DOCS / "validation/v0.1.0-release-contract-20260925.acceptance-report.json"
RELEASE_CONTRACT_VERSION = "1.0.5"
SNAPSHOT = (
    EVALUATION_DOCS
    / "scenarios"
    / f"solution-1-canonical-{RELEASE_CONTRACT_VERSION}-corridor-02-sample.json"
)


def _report() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(REPORT.read_text(encoding="utf-8"))
    return document


def _release_contract() -> dict[str, Any]:
    """Return the committed snapshot of the scenario the report decided."""
    document: dict[str, Any] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return document


def _gate_kinds() -> dict[str, GateKind]:
    """Return the gate kinds 1.0.5 froze, read from its snapshot instead of live code."""
    return {gate["gate_id"]: GateKind(gate["kind"]) for gate in _release_contract()["gates"]}


def test_the_release_report_is_about_the_frozen_release_contract() -> None:
    """The report must keep matching the snapshot of the scenario it decided.

    It is the evidence the release ships with, so it is about the contract that ships, and that
    contract is 1.0.5 whatever version the live scenario has reached. If the report ever stops
    matching 1.0.5's snapshot, one of the two was edited, which is the failure worth catching.
    """
    contract = _release_contract()
    document = _report()

    assert document["schema"] == ACCEPTANCE_REPORT_SCHEMA
    assert contract["version"] == RELEASE_CONTRACT_VERSION
    assert document["scenario"] == {
        "scenario_id": contract["scenario_id"],
        "version": RELEASE_CONTRACT_VERSION,
        "digest": contract["digest"],
        "matrix_digest": contract["matrix_digest"],
        "evidence_class": "real",
    }
    assert [gate["gate_id"] for gate in document["gates"]] == [
        gate["gate_id"] for gate in contract["gates"]
    ]


def test_every_invariant_gate_of_the_release_contract_is_met_with_real_evidence() -> None:
    """Re-derives the release-readiness rule instead of trusting a recorded count."""
    kind = _gate_kinds()

    unmet = [
        gate["gate_id"]
        for gate in _report()["gates"]
        if kind[gate["gate_id"]] is GateKind.INVARIANT
        and not (gate["status"] == "passed" and gate["evidence_class"] == "real")
    ]

    assert unmet == []


def test_the_release_report_hides_no_failed_gate_and_blocks_only_report_gates() -> None:
    kind = _gate_kinds()
    document = _report()
    gates = document["gates"]

    assert Counter(gate["status"] for gate in gates) == {
        "passed": 18,
        "blocked": 5,
        "not_evaluated": 3,
    }
    assert "passed" not in document and "score" not in document
    # Nada que não seja gate de relatório pode ficar sem decisão: um invariante bloqueado ou não
    # avaliado seria uma lacuna do contrato, não uma medida ausente.
    for gate in gates:
        if gate["status"] in ("blocked", "not_evaluated"):
            assert kind[gate["gate_id"]] is GateKind.REPORT, gate["gate_id"]
        if gate["status"] == "blocked":
            assert gate["failing_capabilities"]


def test_the_report_keeps_saying_that_nothing_in_it_was_re_measured() -> None:
    """The caveat is the honesty of the document; dropping it would overstate the evidence."""
    limitations = " ".join(_report()["limitations"])

    assert "not a rerun" in limitations
    assert "pruned" in limitations
    assert "re-verified" in limitations
    # E que a equivalência vale sob a condição estreitada, não de sensores a artifact.
    assert "same PerceptionRunArtifact" in limitations
    assert "#556" in limitations
