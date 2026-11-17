"""Canonical regional codecs: metadata only, mask pixels live in the mask store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception.region_evidence import Region2D, RegionId
from contextmap.region_perception.region_geometry import (
    BoundingBox2D,
    CoordinateConvention,
    RegionProvenance,
)
from contextmap.shared import BackendProvenance

if TYPE_CHECKING:
    from contextmap.region_perception.discovery.region_discovery import RegionDiscoveryAudit


def encode_provenance(provenance: BackendProvenance) -> dict[str, Any]:
    """Encode a :class:`BackendProvenance` into a JSON-serializable dict."""
    return {
        "backend_id": provenance.backend_id,
        "capability": provenance.capability,
        "provider": provenance.provider,
        "model": provenance.model,
        "version": provenance.version,
        "configuration_fingerprint": provenance.configuration_fingerprint,
    }


def decode_provenance(record: dict[str, Any]) -> BackendProvenance:
    """Decode a :class:`BackendProvenance` from :func:`encode_provenance`'s output."""
    return BackendProvenance(**record)


def encode_bounding_box(box: BoundingBox2D) -> dict[str, float]:
    """Encode a :class:`BoundingBox2D` into a JSON-serializable dict."""
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def decode_bounding_box(record: dict[str, Any]) -> BoundingBox2D:
    """Decode a :class:`BoundingBox2D` from :func:`encode_bounding_box`'s output."""
    return BoundingBox2D(**record)


def encode_region(region: Region2D) -> dict[str, Any]:
    """Encode a :class:`Region2D` into a JSON-serializable dict.

    A region's mask pixels, when it has one, are never inlined here —
    only ``mask_reference`` is encoded (#378). A full-frame mask used to
    cost about 0.92 MB of JSON per region regardless of how small the
    region's own bounding box was; a run's actual pixel payload, when
    persisted, lives in a compact, lazily-loaded store instead (see
    :mod:`contextmap.region_perception.artifacts.mask_store` and
    :meth:`~contextmap.visual_perception.run_artifact.PerceptionRunWriter.finalize`).
    A ``Region2D`` built and encoded directly (no writer involved, as in
    this module's own tests) simply keeps whatever ``mask_reference`` it
    was given, which is ``None`` unless something else persisted the
    mask and attached its reference.
    """
    return {
        "region_id": str(region.region_id),
        "bounding_box": encode_bounding_box(region.bounding_box),
        "provenance": encode_provenance(region.provenance),
        "mask_reference": region.mask_reference,
        "region_kind": region.region_kind,
        "is_accepted": region.is_accepted,
        "rejection_reason": region.rejection_reason,
        "source_observation_id": (
            None if region.source_observation_id is None else str(region.source_observation_id)
        ),
        "image_width": region.image_width,
        "image_height": region.image_height,
        "area_pixels": region.area_pixels,
        "contributor_candidate_ids": list(region.contributor_candidate_ids),
        "discovery_provenance": [item.to_dict() for item in region.discovery_provenance],
        "coordinate_convention": region.coordinate_convention.value,
    }


def decode_region(record: dict[str, Any]) -> Region2D:
    """Decode a :class:`Region2D` from :func:`encode_region`'s output.

    The decoded region's ``mask`` is always ``None``: pixel data is
    never inlined in the encoded record (#378). A caller that needs the
    actual pixels for a region whose ``mask_reference`` is set loads
    them lazily and hash-verified through
    :class:`~contextmap.region_perception.MaskStoreReader`.
    """
    # A região pode não conhecer seu frame de origem, e o encoder grava null nesse caso;
    # exigir o campo aqui quebraria o roundtrip de uma região legitimamente sem provenance.
    source_observation_id = record.get("source_observation_id")
    return Region2D(
        region_id=RegionId(record["region_id"]),
        bounding_box=decode_bounding_box(record["bounding_box"]),
        provenance=decode_provenance(record["provenance"]),
        mask_reference=record["mask_reference"],
        region_kind=record["region_kind"],
        is_accepted=record["is_accepted"],
        rejection_reason=record["rejection_reason"],
        source_observation_id=(
            None if source_observation_id is None else SourceObservationId(source_observation_id)
        ),
        image_width=record.get("image_width"),
        image_height=record.get("image_height"),
        area_pixels=record.get("area_pixels"),
        contributor_candidate_ids=tuple(record.get("contributor_candidate_ids", ())),
        discovery_provenance=tuple(
            RegionProvenance.from_dict(item) for item in record.get("discovery_provenance", ())
        ),
        coordinate_convention=CoordinateConvention(
            record.get("coordinate_convention", CoordinateConvention.PIXEL_XY_TOP_LEFT.value)
        ),
    )


def encode_discovery_audit(audit: RegionDiscoveryAudit, *, run_id: str) -> dict[str, Any]:
    """Publish normalization audit with regional lineage, preserving local golden contracts.

    The normalizer's candidate scopes remain local implementation inputs. Published
    hints cite their regional producer and physical frame directly, so neither their
    wire format nor their region references depend on a fused perception result.
    """
    record = audit.to_dict()
    record["region_artifact_id"] = run_id
    hints = []
    for hint in audit.semantic_hints:
        encoded = hint.to_dict()
        encoded.pop("perception_result_id")
        encoded.pop("perception_run_id")
        encoded["region_artifact_id"] = run_id
        hints.append(encoded)
    record["semantic_hints"] = hints
    return record


def decode_discovery_audit(record: dict[str, Any], *, run_id: str) -> RegionDiscoveryAudit:
    """Decode published regional audit into the normalization owner's local view."""
    from contextmap.region_perception.discovery.region_discovery import RegionDiscoveryAudit

    if record["region_artifact_id"] != run_id:
        raise ValueError("discovery audit region_artifact_id does not match its producer")
    local = dict(record)
    local.pop("region_artifact_id")
    hints = []
    for hint in record["semantic_hints"]:
        if hint["region_artifact_id"] != run_id:
            raise ValueError("region hint region_artifact_id does not match its producer")
        decoded = dict(hint)
        decoded.pop("region_artifact_id")
        decoded["perception_run_id"] = run_id
        decoded["perception_result_id"] = f"{run_id}--{hint['source_observation_id']}"
        hints.append(decoded)
    local["semantic_hints"] = hints
    return RegionDiscoveryAudit.from_dict(local)
