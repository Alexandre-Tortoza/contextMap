"""Preflight de geometria: frames, extrínsecos e relógios de uma run são confiáveis?

Falhas de geometria costumam vir de semântica de frames, extrínsecos estáticos ou metadados de
relógio ausentes ou inconsistentes, não do estimador. O preflight bloqueia a run *antes* de o
estimador iniciar quando o que ela precisa falta ou é inválido, e registra quais capabilities
posteriores ainda carecem de pré-requisitos sem fingir que estão validadas.

Os requisitos são declarados por capability (``requirements``): uma run LiDAR-inercial nunca é
bloqueada por falta de intrínsecos de câmera, que viram prontidão downstream. O preflight não
possui, não interpreta e não reescreve calibração (isso é de Ingestion), não fixa rig nem
dataset, e não afirma que uma calibração câmera-LiDAR está fisicamente correta só porque as
matrizes são válidas: a validação por reprojeção é de Sensor Association.

Convenções assumidas pelas checagens, e declaradas no relatório: transforms ``T_parent_child``
(``p_parent = R * p_child + t``), translações em metros e rotações como quaternions unitários
``(x, y, z, w)``. A avaliação segue três passos independentes: calibração estática (checagens
numéricas e consistência de laços), relações exigidas pela capability e metadados de relógio.
"""

from contextmap.state_estimation.preflight.models import (
    ClockCheck,
    DownstreamReadiness,
    FrameGraphSummary,
    GeometryPreflightReport,
    PreflightFinding,
    PreflightStatus,
    PreflightTolerances,
    TransformCheck,
)
from contextmap.state_estimation.preflight.service import (
    calibration_identity,
    run_geometry_preflight,
)

__all__ = [
    "ClockCheck",
    "DownstreamReadiness",
    "FrameGraphSummary",
    "GeometryPreflightReport",
    "PreflightFinding",
    "PreflightStatus",
    "PreflightTolerances",
    "TransformCheck",
    "calibration_identity",
    "run_geometry_preflight",
]
