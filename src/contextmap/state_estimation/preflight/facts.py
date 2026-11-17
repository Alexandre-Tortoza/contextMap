"""O que as observações selecionadas dizem sobre modalidades, frames, clocks e calibração.

``readiness`` e ``clocks`` avaliam requisitos diferentes sobre os mesmos fatos; resumir as
observações uma única vez aqui evita que um avaliador dependa do outro.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from contextmap.ingestion import (
    CalibrationReferenceId,
    FrameId,
    SourceObservation,
    observation_modality,
)


@dataclass(frozen=True)
class ObservationFacts:
    """What the observations say about modalities, frames, clocks and calibration ids."""

    modalities: frozenset[str]
    frames: dict[str, set[FrameId]]
    clocks: dict[str, set[str]]
    calibration_ids: dict[str, list[CalibrationReferenceId | None]]


def gather_observation_facts(observations: Iterable[SourceObservation]) -> ObservationFacts:
    """Summarize the observations by modality.

    Args:
        observations: The selected canonical observations.

    Returns:
        The modalities present and, per modality, the frames, clock identities and calibration
        references the observations carry.
    """
    frames: dict[str, set[FrameId]] = {}
    clocks: dict[str, set[str]] = {}
    calibration_ids: dict[str, list[CalibrationReferenceId | None]] = {}
    for observation in observations:
        modality = observation_modality(observation)
        frames.setdefault(modality, set()).add(observation.frame_id)
        clocks.setdefault(modality, set()).add(observation.timestamp.clock_id)
        calibration_ids.setdefault(modality, []).append(observation.calibration_id)
    return ObservationFacts(
        modalities=frozenset(frames),
        frames=frames,
        clocks=clocks,
        calibration_ids=calibration_ids,
    )
