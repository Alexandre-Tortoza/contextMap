"""Diagnósticos de calibração, reprojeção e alinhamento temporal de um frame.

Três responsabilidades distintas vivem aqui:

- ``reprojection.py`` mede o resíduo contra uma referência confiável, e só contra ela;
- ``frame_report.py`` reúne tudo de que o frame dependeu e emite achados explícitos quando uma
  tolerância configurada é ultrapassada;
- ``time_offset_sweep.py`` reprojeta o mesmo frame em timestamps deslocados, apenas para
  separar erro de calibração de erro de tempo.

Nenhum deles vira confiança semântica, e nenhum altera calibração, timestamp ou claim: os
componentes brutos são preservados para que uma política a jusante decida o que fazer com eles.
"""

from contextmap.sensor_association.diagnostics.frame_report import (
    DIAGNOSTICS_DEFINITIONS_VERSION,
    DenseSamplingSummary,
    DiagnosticFinding,
    DiagnosticTolerances,
    FindingCode,
    FindingSeverity,
    FrameDiagnostics,
    ReprojectionAttempt,
    ReprojectionOutcome,
    diagnose_frame,
)
from contextmap.sensor_association.diagnostics.reprojection import (
    TrustedCorrespondences,
    evaluated_correspondences,
    reprojection_statistics,
)
from contextmap.sensor_association.diagnostics.time_offset_sweep import (
    TimeOffsetOutcome,
    time_offset_sweep,
)

__all__ = [
    "DIAGNOSTICS_DEFINITIONS_VERSION",
    "DenseSamplingSummary",
    "DiagnosticFinding",
    "DiagnosticTolerances",
    "FindingCode",
    "FindingSeverity",
    "FrameDiagnostics",
    "ReprojectionAttempt",
    "ReprojectionOutcome",
    "TimeOffsetOutcome",
    "TrustedCorrespondences",
    "diagnose_frame",
    "evaluated_correspondences",
    "reprojection_statistics",
    "time_offset_sweep",
]
