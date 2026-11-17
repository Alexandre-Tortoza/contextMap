"""Regiões não processadas por um estágio regional, com motivo explícito (#650).

Region features and region interpretation run over every region of a frame's region set:
discovered, grounded and refined regions alike. A stage conditioned on evidence a region does
not carry cannot process that region: AlphaCLIP embeds a region through its mask, and the
``masked_subject`` semantic view paints everything outside the mask. A grounded box has no
mask, and inventing one from the box would fabricate evidence the grounding never produced.

The stage therefore skips such a region, processes the rest of the frame, and reports one
:class:`IneligibleRegion` per skipped region, which the run persists in
``outputs/ineligible-regions.jsonl``. A skipped region is never silently dropped and never
makes the stage fail for the regions that could be processed. This is the per-region
counterpart of Sensor Association's ``SkippedRegion(NO_INLINE_MASK)``: the same box-only
region one capability later, recorded the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception.region_evidence import RegionId
from contextmap.visual_perception import FeatureId


class IneligibleRegionReason(Enum):
    """Why a region-scoped stage did not process a region."""

    NO_MASK = "no_mask"
    """The stage is conditioned on the region's mask, and the region is box-only."""

    POOLING_UNAVAILABLE = "pooling_unavailable"
    """The dense map cannot provide valid support for this region's mask."""

    REGION_BUDGET = "region_budget"
    """The run's semantic request budget interprets only the frame's largest regions."""


@dataclass(frozen=True, kw_only=True)
class IneligibleRegion:
    """One region of one frame that one region-scoped stage did not process.

    Attributes:
        source_observation_id: The frame whose region set held the region.
        region_id: The region, as it appears in that frame's result.
        stage_id: The pipeline stage that skipped it, e.g. ``"region_feature_extraction"``
            or ``"region_interpretation"``.
        reason: Why the stage could not process it.
        detail: Human-readable explanation naming what the stage needed.
        source_feature_id: The dense feature the pooling could not use, for a
            ``POOLING_UNAVAILABLE`` record. A stage may pool a region from several
            dense sources, so the source is part of the record's identity; it is
            ``None`` for any other reason, which is not per-source.
    """

    source_observation_id: SourceObservationId
    region_id: RegionId
    stage_id: str
    reason: IneligibleRegionReason
    detail: str
    source_feature_id: FeatureId | None = None

    def __post_init__(self) -> None:
        """Require the frame, region and stage identities, an explanation and a coherent source.

        Raises:
            ValueError: If an identity, the stage or the detail is empty, or if
                ``source_feature_id`` does not match the reason: a
                ``POOLING_UNAVAILABLE`` record names its dense source, and any other
                reason leaves it unset.
        """
        if not str(self.source_observation_id).strip() or not str(self.region_id).strip():
            raise ValueError("an ineligible region needs its source_observation_id and region_id")
        if not self.stage_id.strip():
            raise ValueError("an ineligible region needs the stage_id that skipped it")
        if not self.detail.strip():
            raise ValueError("an ineligible region needs a detail explaining its reason")
        if self.reason is IneligibleRegionReason.POOLING_UNAVAILABLE:
            if self.source_feature_id is None or not str(self.source_feature_id).strip():
                raise ValueError(
                    "a pooling_unavailable region needs the source_feature_id it could not pool"
                )
        elif self.source_feature_id is not None:
            raise ValueError(
                "source_feature_id is only for a pooling_unavailable region, not "
                f"{self.reason.value!r}"
            )


def encode_ineligible_region(record: IneligibleRegion) -> dict[str, Any]:
    """Encode one record for ``outputs/ineligible-regions.jsonl``."""
    encoded = {
        "source_observation_id": str(record.source_observation_id),
        "region_id": str(record.region_id),
        "stage_id": record.stage_id,
        "reason": record.reason.value,
        "detail": record.detail,
    }
    if record.source_feature_id is not None:
        encoded["source_feature_id"] = str(record.source_feature_id)
    return encoded


def decode_ineligible_region(record: Mapping[str, Any]) -> IneligibleRegion:
    """Decode one record written by :func:`encode_ineligible_region`.

    Raises:
        KeyError: If a field is missing.
        TypeError: If an identity, the stage or the detail is not a string.
        ValueError: If the reason is unknown or a field is empty.
    """
    fields = ("source_observation_id", "region_id", "stage_id", "detail")
    for name in fields:
        if not isinstance(record[name], str):
            raise TypeError(f"ineligible region {name} must be a string")
    raw_source_feature_id = record.get("source_feature_id")
    if raw_source_feature_id is not None and not isinstance(raw_source_feature_id, str):
        raise TypeError("ineligible region source_feature_id must be a string")
    return IneligibleRegion(
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        region_id=RegionId(record["region_id"]),
        stage_id=record["stage_id"],
        reason=IneligibleRegionReason(record["reason"]),
        detail=record["detail"],
        source_feature_id=(
            None if raw_source_feature_id is None else FeatureId(raw_source_feature_id)
        ),
    )
