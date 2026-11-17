"""Round-trip property of a persisted pose: decoding what was encoded changes nothing.

A trajectory artifact stores one encoded pose per line, and every later stage reads it back.
Whatever the contract accepts must survive the JSON file exactly: floats down to the last bit
(signed zero, subnormals, the largest magnitudes), an orientation anywhere inside the unit
tolerance, any clock offset and any Unicode in the identities.
"""

from __future__ import annotations

import json

from hypothesis import assume, given
from hypothesis import strategies as st

from contextmap.ingestion import FrameId, SourceObservationId
from contextmap.shared import UNIT_QUATERNION_TOLERANCE, Quaternion, is_unit_quaternion
from contextmap.state_estimation import (
    PoseEstimate,
    PoseEstimateId,
    PoseProvenance,
    PoseValidity,
)
from contextmap.state_estimation.serialization import decode_pose_estimate, encode_pose_estimate
from property_strategies import finite_floats, finite_vectors, identifiers, rotations, timestamps


@st.composite
def orientations(draw: st.DrawFn) -> Quaternion:
    """Unit quaternions, and quaternions stretched up to the edge of the unit tolerance."""
    rotation = draw(rotations())
    stretch = draw(
        st.floats(
            min_value=1.0 - UNIT_QUATERNION_TOLERANCE, max_value=1.0 + UNIT_QUATERNION_TOLERANCE
        )
    )
    stretched = (
        rotation[0] * stretch,
        rotation[1] * stretch,
        rotation[2] * stretch,
        rotation[3] * stretch,
    )
    assume(is_unit_quaternion(stretched))
    return stretched


@st.composite
def poses(draw: st.DrawFn) -> PoseEstimate:
    parent, child = draw(st.lists(identifiers, min_size=2, max_size=2, unique=True))
    return PoseEstimate(
        estimate_id=PoseEstimateId(draw(identifiers)),
        timestamp=draw(timestamps),
        parent_frame=FrameId(parent),
        child_frame=FrameId(child),
        translation_m=draw(finite_vectors()),
        orientation=draw(orientations()),
        validity=draw(st.sampled_from(PoseValidity)),
        provenance=PoseProvenance(
            source_observation_ids=tuple(
                SourceObservationId(item)
                for item in draw(st.lists(identifiers, min_size=1, max_size=3))
            ),
            conversions_applied=tuple(draw(st.lists(st.text(max_size=24), max_size=2))),
            derived_from=tuple(
                PoseEstimateId(item) for item in draw(st.lists(identifiers, max_size=2))
            ),
        ),
        covariance=draw(st.none() | st.tuples(*(finite_floats for _ in range(36)))),
    )


def _persisted(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True)


@given(poses())
def test_a_pose_read_back_from_its_record_is_the_same_pose_bit_for_bit(pose: PoseEstimate) -> None:
    record = _persisted(encode_pose_estimate(pose))

    decoded = decode_pose_estimate(json.loads(record))

    assert decoded == pose
    # A igualdade de dataclass aceita 0.0 == -0.0; o registro reescrito não aceita nada disso.
    assert _persisted(encode_pose_estimate(decoded)) == record
