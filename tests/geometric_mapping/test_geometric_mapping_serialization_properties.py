"""Round-trip property of the source index: an entry read back is the entry that was written.

``outputs/source-index.jsonl`` holds one ``ScanRecord`` per scan, with the transform chain and
the bounds every later query prunes with. Whatever the contract accepts must survive the file
exactly: degenerate and inverted-sign bounds, floats down to the last bit, dynamic and static
steps, empty chains, scans without geometry and counts beyond 32 bits.
"""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from contextmap.geometric_mapping import (
    Bounds3D,
    MotionCorrectionState,
    ScanRecord,
    TracedTransform,
    TransformKind,
    TransformStep,
)
from contextmap.geometric_mapping.serialization import decode_scan_record, encode_scan_record
from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.state_estimation import PoseEstimateId
from property_strategies import finite_floats, finite_vectors, identifiers, rotations, timestamps

counts = st.integers(min_value=0, max_value=2**40)


@st.composite
def bounds(draw: st.DrawFn) -> Bounds3D:
    """Tight boxes of any finite extent, including a single point on some or every axis."""
    corners = [sorted(draw(st.tuples(finite_floats, finite_floats))) for _ in range(3)]
    return Bounds3D(
        frame_id=FrameId(draw(identifiers)),
        minimum_m=(corners[0][0], corners[1][0], corners[2][0]),
        maximum_m=(corners[0][1], corners[1][1], corners[2][1]),
    )


@st.composite
def traced_transforms(draw: st.DrawFn) -> TracedTransform:
    parent, child = draw(st.lists(identifiers, min_size=2, max_size=2, unique=True))
    kind = draw(st.sampled_from(TransformKind))
    sources = (
        draw(st.lists(identifiers, min_size=1, max_size=2))
        if kind is TransformKind.DYNAMIC_POSE
        else []
    )
    return TracedTransform(
        step=TransformStep(
            kind=kind,
            parent_frame=FrameId(parent),
            child_frame=FrameId(child),
            reference=draw(identifiers),
            source_estimate_ids=tuple(PoseEstimateId(item) for item in sources),
        ),
        translation_m=draw(finite_vectors()),
        rotation=draw(rotations()),
    )


@st.composite
def scan_records(draw: st.DrawFn) -> ScanRecord:
    source_point_count = draw(counts)
    geometry_count = draw(counts)
    return ScanRecord(
        ordinal=draw(counts),
        observation_id=SourceObservationId(draw(identifiers)),
        source_frame=FrameId(draw(identifiers)),
        acquisition_timestamp=draw(timestamps),
        payload_hash=draw(identifiers),
        motion_correction=draw(st.sampled_from(MotionCorrectionState)),
        transform_chain=tuple(draw(st.lists(traced_transforms(), max_size=3))),
        source_point_count=source_point_count,
        dropped_non_finite_count=draw(st.integers(min_value=0, max_value=source_point_count)),
        first_geometry_index=draw(counts),
        geometry_count=geometry_count,
        bounds=None if geometry_count == 0 else draw(bounds()),
    )


def _persisted(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True)


@given(scan_records())
def test_a_source_index_entry_read_back_is_the_same_entry_bit_for_bit(scan: ScanRecord) -> None:
    line = _persisted(encode_scan_record(scan))

    decoded = decode_scan_record(json.loads(line))

    assert decoded == scan
    # A igualdade de dataclass aceita 0.0 == -0.0; a linha reescrita não aceita nada disso.
    assert _persisted(encode_scan_record(decoded)) == line
