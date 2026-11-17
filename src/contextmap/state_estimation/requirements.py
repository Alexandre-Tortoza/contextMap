"""Declaração do que uma capability exige da sequência e da calibração.

Este módulo é o vocabulário que um backend usa para dizer o que precisa (modalidades,
extrínsecos estáticos, modelo de câmera, frames dinâmicos). Ele não avalia nada: quem
verifica esses requisitos contra as observações e a calibração é ``preflight``. A separação
mantém o port ``StateEstimator`` dependente apenas da declaração, nunca do avaliador.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.ingestion import MODALITY_NAMES, FrameId

BODY_ENDPOINT = "body"
"""Endpoint naming the body frame of the dynamic transform, instead of a modality."""

_ENDPOINTS = MODALITY_NAMES | {BODY_ENDPOINT}


@dataclass(frozen=True, kw_only=True)
class StaticRelationRequirement:
    """A static transform a capability needs between two frames.

    Attributes:
        from_endpoint: A modality name (``"lidar"``, ``"imu"``, ``"image"``,
            ``"external_pose"``) or :data:`BODY_ENDPOINT`.
        to_endpoint: A modality name or :data:`BODY_ENDPOINT`.
    """

    from_endpoint: str
    to_endpoint: str

    def __post_init__(self) -> None:
        """Reject endpoints that are neither a modality nor the body frame.

        Raises:
            ValueError: If an endpoint is unknown.
        """
        for endpoint in (self.from_endpoint, self.to_endpoint):
            if endpoint not in _ENDPOINTS:
                raise ValueError(
                    f"unknown endpoint {endpoint!r}, expected one of {sorted(_ENDPOINTS)}"
                )


@dataclass(frozen=True, kw_only=True)
class GeometryRequirements:
    """What one capability needs from the sequence and its calibration.

    Attributes:
        capability: Name shown in reports, e.g. ``"state_estimation:external_pose"``.
        modalities: Observation modalities that must be present.
        static_relations: Static transforms that must exist between frames.
        camera_model_modalities: Modalities whose observations must reference
            a calibration entry with a camera model (intrinsics).
        reference_frame: For the *executing* capability only, the reference
            frame of the dynamic transform it publishes.
        body_frame: For the executing capability only, the body frame of the
            dynamic transform; :data:`BODY_ENDPOINT` resolves to it.
    """

    capability: str
    modalities: frozenset[str]
    static_relations: tuple[StaticRelationRequirement, ...] = ()
    camera_model_modalities: frozenset[str] = frozenset()
    reference_frame: FrameId | None = None
    body_frame: FrameId | None = None

    def __post_init__(self) -> None:
        """Reject unknown modality names.

        Raises:
            ValueError: If a modality is not one of the canonical modalities.
        """
        unknown = (self.modalities | self.camera_model_modalities) - MODALITY_NAMES
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
