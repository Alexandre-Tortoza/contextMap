"""Varredura de deslocamentos temporais para diagnóstico de reprojeção.

Reprojeta timestamps deslocados e reporta cada resultado sem modificar as entradas.
Não aplica correção de calibração ou alinhamento temporal ao run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from contextmap.ingestion import ImageObservation
from contextmap.sensor_association.diagnostics.reprojection import (
    TrustedCorrespondences,
    reprojection_statistics,
)
from contextmap.sensor_association.projection_chain.frame_projection import (
    FrameProjector,
    RejectedProjection,
)
from contextmap.sensor_association.quality import ReprojectionStatistics
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import LookupRejection
from contextmap.visual_perception import PreparedImage


@dataclass(frozen=True, kw_only=True)
class TimeOffsetOutcome:
    """The reprojection residual of a frame whose timestamp was shifted by one offset.

    Attributes:
        offset_ns: The offset applied to the frame timestamp, in nanoseconds.
        statistics: The residual against the trusted reference, or ``None`` when the shifted
            frame could not be evaluated.
        rejection: Why the pose lookup rejected the shifted timestamp, when it did.
    """

    offset_ns: int
    statistics: ReprojectionStatistics | None
    rejection: LookupRejection | None


def time_offset_sweep(
    projector: FrameProjector,
    observation: ImageObservation,
    prepared_image: PreparedImage,
    correspondences: TrustedCorrespondences,
    offsets_ns: Sequence[int],
) -> tuple[TimeOffsetOutcome, ...]:
    """Measure the reprojection residual as the frame timestamp is shifted, for diagnosis only.

    Each offset re-projects the frame with the pose looked up at the shifted timestamp and
    measures the residual of the reference against it. The observation, the calibration and
    every timestamp are left untouched; the sweep only reports, and choosing to act on it is
    a separate, explicit decision. Each offset selects its own candidates, because a shifted
    pose moves the camera, so a reference whose geometry falls outside the candidate policy
    at some offset is reported as unevaluated there.

    Args:
        projector: The projector of the map, trajectory and calibration.
        observation: The camera frame.
        prepared_image: The image prepared from that frame.
        correspondences: The trusted reference.
        offsets_ns: The offsets to try, in nanoseconds, added to the frame timestamp.

    Returns:
        One outcome per offset, in the given order.
    """
    outcomes: list[TimeOffsetOutcome] = []
    for offset in offsets_ns:
        shifted = dataclasses.replace(
            observation, timestamp=_shifted(observation.timestamp, offset)
        )
        result = projector.project(shifted, prepared_image)
        if isinstance(result, RejectedProjection):
            outcomes.append(
                TimeOffsetOutcome(offset_ns=offset, statistics=None, rejection=result.rejection)
            )
        else:
            outcomes.append(
                TimeOffsetOutcome(
                    offset_ns=offset,
                    statistics=reprojection_statistics(result, correspondences),
                    rejection=None,
                )
            )
    return tuple(outcomes)


def _shifted(timestamp: SourceTimestamp, offset_ns: int) -> SourceTimestamp:
    """Copy an observation with a diagnostic timestamp offset in nanoseconds."""
    seconds, nanoseconds = divmod(timestamp.total_nanoseconds() + offset_ns, 1_000_000_000)
    return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=timestamp.clock_id)
