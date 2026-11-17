"""Requisitos de clock domain: cada modalidade exigida tem um clock explícito e compartilhado?

Timestamps de domínios diferentes nunca são comparados implicitamente. Este módulo decide se os
metadados de clock das observações permitem a run começar; a conversão entre domínios, quando
existir, não é responsabilidade do preflight.
"""

from __future__ import annotations

from contextmap.state_estimation.preflight.facts import ObservationFacts
from contextmap.state_estimation.preflight.models import ClockCheck, PreflightFinding
from contextmap.state_estimation.requirements import GeometryRequirements


def evaluate_clock_requirements(
    execution: GeometryRequirements, facts: ObservationFacts
) -> tuple[list[ClockCheck], list[PreflightFinding]]:
    """Check that the required modalities carry one, explicit and shared clock domain."""
    checks: list[ClockCheck] = []
    blockers: list[PreflightFinding] = []
    required_present = sorted(execution.modalities & facts.modalities)
    all_clocks: set[str] = set()
    for modality in required_present:
        clocks = facts.clocks[modality]
        all_clocks |= clocks
        if "" in clocks:
            checks.append(
                ClockCheck(
                    subject=f"modality:{modality}", passed=False, detail="empty clock identity"
                )
            )
            blockers.append(
                PreflightFinding(
                    code="missing_clock",
                    message=f"observations of {modality!r} carry no clock identity",
                )
            )
        elif len(clocks) > 1:
            checks.append(
                ClockCheck(
                    subject=f"modality:{modality}",
                    passed=False,
                    detail=f"several clock domains {sorted(clocks)}",
                )
            )
            blockers.append(
                PreflightFinding(
                    code="clock_domain_mismatch",
                    message=f"observations of {modality!r} use several clock domains "
                    f"{sorted(clocks)}",
                )
            )
        else:
            checks.append(
                ClockCheck(subject=f"modality:{modality}", passed=True, detail=next(iter(clocks)))
            )
    if len(required_present) > 1:
        aligned = len(all_clocks) <= 1
        checks.append(
            ClockCheck(
                subject="required modalities",
                passed=aligned,
                detail=f"clock domains {sorted(all_clocks)}",
            )
        )
        if not aligned and "" not in all_clocks:
            blockers.append(
                PreflightFinding(
                    code="clock_domain_mismatch",
                    message="required modalities use different clock domains "
                    f"{sorted(all_clocks)}; clock domains are never compared implicitly",
                )
            )
    return checks, blockers
