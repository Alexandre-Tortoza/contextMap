"""Manifest of a persisted Point Representation run: its fields, decoding and center identity.

The manifest is the authoritative metadata of a run: identity, the geometric map it was
built over, the support policy and encoder, the representation counts, and the inventory of
every contractual file with size and hash. This module owns the shape of that record, how it
is decoded from disk (refusing a schema version it does not understand) and the identity of
the set of centers the run records, which lets an evaluation prove two encoders were
compared over the same centers.

Reading it needs only the standard library and the identities of the geometric map: nothing
here touches NumPy, a model library or the capability that selects the centers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contextmap.geometric_mapping import GeometryReference, MapId
from contextmap.point_representation.models import (
    EncoderIdentity,
    PointRepresentationRunId,
    SupportPolicy,
)
from contextmap.point_representation.run_artifact.layout import (
    MANIFEST,
    SCHEMA_VERSION,
    IncompleteRunArtifactError,
    RunArtifactError,
)
from contextmap.point_representation.serialization import decode_support_policy
from contextmap.shared import FileEntry


@dataclass(frozen=True, kw_only=True)
class PointRepresentationRunManifest:
    """Authoritative metadata of a persisted Point Representation run.

    Attributes:
        run_id: Identity of the run, supplied by the caller.
        run_index: Ordinal of the run among the caller's runs of this sequence, supplied by the
            caller.
        sequence_name: Name of the processed sequence.
        geometric_map_id: The immutable geometric map the geometry came from.
        geometric_map_frame: Frame of that map.
        geometric_map_point_count: Elements in that map.
        sequence_artifact_id: Canonical sequence the map was built from.
        geometry_selection_id: Selection identity the map was built for.
        trajectory_id: Trajectory whose poses placed the map's geometry.
        association_context_id: Association run used as selection/context input
            when explicitly given, else ``None``; never a hidden feature source.
        center_selection_id: Identity of the set of requested centers,
            independent of request order.
        center_selection_policy_id: Versioned policy that selected the centers (for
            example ``observation-support-anchor-v1``), or ``None`` when the caller
            requested them itself.
        support_policy: The support policy of the run, with its coordinate
            preparation.
        representation_space_id: Fingerprint of the run's representation space.
        dimension: Vector dimensionality.
        dtype: Payload element type.
        payload_path: Run-relative path of the vector payload.
        encoder: Encoder identity, including the checkpoint hash when learned.
        code_version: Code revision that produced the run.
        representation_count: Representations persisted, partial ones included.
        partial_count: Representations with undefined components.
        failed_count: Supports that produced no representation.
        failed_by_reason: Failed supports per reason.
        debug_level: Debug evidence level that was requested.
        schema_version: Run artifact schema version.
        created_at: ISO 8601 UTC creation timestamp.
        file_inventory: Every contractual file, with size and hash; excludes
            the manifest, the README and ``__DEBUG__/``.
    """

    run_id: PointRepresentationRunId
    run_index: int
    sequence_name: str
    geometric_map_id: MapId
    geometric_map_frame: str
    geometric_map_point_count: int
    sequence_artifact_id: str
    geometry_selection_id: str
    trajectory_id: str
    association_context_id: str | None
    center_selection_id: str
    center_selection_policy_id: str | None
    support_policy: SupportPolicy
    representation_space_id: str
    dimension: int
    dtype: str
    payload_path: str
    encoder: EncoderIdentity
    code_version: str
    representation_count: int
    partial_count: int
    failed_count: int
    failed_by_reason: Mapping[str, int]
    debug_level: str
    schema_version: str
    created_at: str
    file_inventory: tuple[FileEntry, ...]


def center_selection_id(*, map_id: MapId, centers: Iterable[GeometryReference]) -> str:
    """Identity of a set of requested centers of one map, independent of request order.

    The manifest records it, so an evaluation that fixes the same centers across
    encoders can prove it compared like with like.

    Args:
        map_id: The geometric map the centers belong to.
        centers: The requested centers, each a reference into ``map_id``.

    Returns:
        ``"sha256:<hex digest>"`` of the map identity and the sorted center ids.
    """
    text = "\n".join([str(map_id), *sorted(str(center.geometry_id) for center in centers)])
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def load_manifest(run_dir: Path) -> PointRepresentationRunManifest:
    """Decode the manifest of the run in ``run_dir``.

    Raises:
        IncompleteRunArtifactError: If ``manifest.json`` is missing.
        RunArtifactError: If the schema version is not the one this code understands.
    """
    manifest_path = run_dir / MANIFEST
    if not manifest_path.is_file():
        raise IncompleteRunArtifactError(f"missing {MANIFEST} in {run_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RunArtifactError(
            f"unsupported run artifact schema_version: {raw.get('schema_version')!r}"
        )
    geometric_map = raw["geometric_map"]
    encoder = raw["encoder"]
    return PointRepresentationRunManifest(
        run_id=PointRepresentationRunId(raw["run_id"]),
        run_index=raw["run_index"],
        sequence_name=raw["sequence_name"],
        geometric_map_id=MapId(geometric_map["map_id"]),
        geometric_map_frame=geometric_map["frame_id"],
        geometric_map_point_count=geometric_map["point_count"],
        sequence_artifact_id=geometric_map["sequence_artifact_id"],
        geometry_selection_id=geometric_map["selection_id"],
        trajectory_id=geometric_map["trajectory_id"],
        association_context_id=raw["association_context_id"],
        center_selection_id=raw["center_selection_id"],
        center_selection_policy_id=raw["center_selection_policy_id"],
        support_policy=decode_support_policy(raw["support_policy"]),
        representation_space_id=raw["representation_space_id"],
        dimension=raw["dimension"],
        dtype=raw["dtype"],
        payload_path=raw["payload_path"],
        encoder=EncoderIdentity(
            backend_id=encoder["backend_id"],
            backend_version=encoder["backend_version"],
            configuration_fingerprint=encoder["configuration_fingerprint"],
            checkpoint_hash=encoder["checkpoint_hash"],
        ),
        code_version=raw["code_version"],
        representation_count=raw["representation_count"],
        partial_count=raw["partial_count"],
        failed_count=raw["failed_count"],
        failed_by_reason=dict(raw["failed_by_reason"]),
        debug_level=raw["debug_level"],
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        file_inventory=tuple(
            FileEntry(
                path=entry["path"],
                size_bytes=entry["size_bytes"],
                content_hash=entry["content_hash"],
            )
            for entry in raw["file_inventory"]
        ),
    )
