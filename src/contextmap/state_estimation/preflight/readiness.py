"""Satisfação dos requisitos geométricos de uma capability pelos inputs disponíveis.

``requirements`` declara o que a capability precisa; este módulo verifica se isso existe nas
observações e na calibração (modalidades presentes, frames sem ambiguidade, caminho estático
entre os extremos de cada relação e modelo de câmera). A validade numérica da própria
calibração é de ``calibration``.
"""

from __future__ import annotations

from contextmap.ingestion import CalibrationSet, FrameId
from contextmap.state_estimation.frame_graph import StaticFrameGraph
from contextmap.state_estimation.preflight.facts import ObservationFacts
from contextmap.state_estimation.preflight.models import PreflightFinding, TransformCheck
from contextmap.state_estimation.requirements import BODY_ENDPOINT, GeometryRequirements


def evaluate_requirements(
    requirements: GeometryRequirements,
    *,
    facts: ObservationFacts,
    calibration: CalibrationSet | None,
    graph: StaticFrameGraph,
    body_frame: FrameId,
) -> tuple[list[PreflightFinding], list[TransformCheck], set[FrameId]]:
    """Evaluate one capability's requirements; return findings, checks and used frames."""
    findings: list[PreflightFinding] = []
    checks: list[TransformCheck] = []
    touched: set[FrameId] = set()

    for modality in sorted(requirements.modalities - facts.modalities):
        findings.append(
            PreflightFinding(
                code="missing_input",
                message=f"{requirements.capability} requires {modality!r} observations, "
                "but the selection has none",
            )
        )

    def endpoint_frame(endpoint: str) -> FrameId | None:
        if endpoint == BODY_ENDPOINT:
            return body_frame
        frames = facts.frames.get(endpoint, set())
        if len(frames) > 1:
            findings.append(
                PreflightFinding(
                    code="ambiguous_endpoint",
                    message=f"modality {endpoint!r} spans several frames "
                    f"{sorted(map(str, frames))}, so a static relation to it is ambiguous",
                )
            )
            return None
        # Sem observações da modalidade o problema já foi reportado como `missing_input`.
        return next(iter(frames), None)

    for relation in requirements.static_relations:
        first = endpoint_frame(relation.from_endpoint)
        second = endpoint_frame(relation.to_endpoint)
        if first is None or second is None:
            continue
        subject = f"{first}<-{second}"
        if first == second:
            checks.append(
                TransformCheck(
                    subject=subject,
                    check="relation_available",
                    passed=True,
                    detail="same frame, identity",
                )
            )
            continue
        if calibration is None:
            findings.append(
                PreflightFinding(
                    code="missing_calibration",
                    message=f"{requirements.capability} needs the static transform between "
                    f"{first!r} and {second!r}, but the sequence has no calibration",
                )
            )
            continue
        connected = first in graph.frames and second in graph.component_of(first)
        checks.append(
            TransformCheck(
                subject=subject,
                check="relation_available",
                passed=connected,
                detail="static path found" if connected else "no static path",
            )
        )
        if not connected:
            findings.append(
                PreflightFinding(
                    code="missing_static_transform",
                    message=f"{requirements.capability} needs a static transform between "
                    f"{first!r} and {second!r}, but calibration has no path connecting them",
                )
            )
            continue
        touched |= graph.component_of(first)

    for modality in sorted(requirements.camera_model_modalities & facts.modalities):
        entries = calibration.entries if calibration is not None else {}
        lacks_model = any(
            calibration_id is None
            or calibration_id not in entries
            or entries[calibration_id].camera_model is None
            for calibration_id in facts.calibration_ids[modality]
        )
        if lacks_model:
            findings.append(
                PreflightFinding(
                    code="missing_camera_model",
                    message=f"{requirements.capability} needs a camera model (intrinsics) for "
                    f"{modality!r}, but an observation has no calibration entry with one",
                )
            )
    return findings, checks, touched
