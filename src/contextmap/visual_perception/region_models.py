"""Nomes antigos dos contratos de região, reexportados do dono que os possui.

Os contratos que moravam aqui são de ``region_perception``: a geometria de uma região, sua
máscara, a convenção de coordenadas de pixel, a proposta bruta do backend e a recusa dela
(ver ``contextmap.region_perception.region_geometry``). ``ArtifactReference`` e
``JsonScalar`` não eram contrato de região nem de percepção e foram para
``contextmap.shared``, onde atravessam capabilities com o mesmo significado.

Este módulo não redefine nada: cada nome aqui **é** o objeto do dono, não uma cópia, para
que ``isinstance`` e ``==`` continuem valendo entre os dois caminhos de import
(``tests/architecture/test_compatibility_facade.py``). Código novo importa do dono.
"""

from __future__ import annotations

from contextmap.region_perception import (
    BackendScore,
    BoundingBox,
    CoordinateConvention,
    InlineMask,
    NativeRegionText,
    RegionCandidate,
    RegionProvenance,
    RejectedRegionCandidate,
    RejectionReason,
    mask_bounding_box,
)
from contextmap.shared import ArtifactReference, JsonScalar

__all__ = [
    "ArtifactReference",
    "BackendScore",
    "BoundingBox",
    "CoordinateConvention",
    "InlineMask",
    "JsonScalar",
    "NativeRegionText",
    "RegionCandidate",
    "RegionProvenance",
    "RejectedRegionCandidate",
    "RejectionReason",
    "mask_bounding_box",
]
