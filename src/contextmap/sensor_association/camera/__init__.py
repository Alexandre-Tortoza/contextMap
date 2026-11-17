"""Projeção de câmera calibrada, atrás de um contrato único.

Este subpacote é a borda da capability com a óptica: recebe pontos no frame óptico da câmera e
devolve pixels da imagem bruta, e vice-versa. O modelo vem apenas da calibração canônica de
Ingestion — pinhole, fisheye equidistante ou omnidirecional unificado (MEI) — e nenhum deles é
substituído por outro.

Quem consome a capability usa :func:`camera_projection_for` e o port
:class:`CameraProjection`; as três implementações e a numérica que elas compartilham são
internas ao subpacote.
"""

from contextmap.sensor_association.camera.projection import (
    CameraIdentity,
    CameraProjection,
    PixelProjection,
)
from contextmap.sensor_association.camera.selection import camera_projection_for

__all__ = [
    "CameraIdentity",
    "CameraProjection",
    "PixelProjection",
    "camera_projection_for",
]
