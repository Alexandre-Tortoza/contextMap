"""Regional serialization keeps full image masks outside JSON records."""

import json

import numpy as np

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionId,
    decode_region,
    encode_region,
)
from contextmap.shared import BackendProvenance

_PROVENANCE = BackendProvenance(
    backend_id="fake", capability="region_discovery", provider="fake", model="fake", version="1"
)


def test_encode_region_never_inlines_a_full_frame_mask() -> None:
    """Regression test for #378: a full-frame mask must not blow up encode_region() output.

    Mirrors the issue's own reproduction: a 640x480 mask used to cost
    about 922 KB of JSON per region regardless of the region's actual
    bounding box; encoded output must now scale with region metadata,
    not with image resolution.
    """
    width, height = 640, 480
    bits = tuple((x // 40 + y // 40) % 2 == 0 for y in range(height) for x in range(width))
    region = Region2D(
        region_id=RegionId("r1"),
        bounding_box=BoundingBox2D(x=0.0, y=0.0, width=float(width), height=float(height)),
        provenance=_PROVENANCE,
        source_observation_id=SourceObservationId("o1"),
        image_width=width,
        image_height=height,
        area_pixels=float(sum(bits)),
        mask=InlineMask(np.array(bits, dtype=bool).reshape(height, width)),
    )

    encoded_size = len(json.dumps(encode_region(region), sort_keys=True))

    assert encoded_size < 2_000  # was 922,285 bytes before the fix
    assert "mask" not in encode_region(region)

    decoded = decode_region(encode_region(region))
    assert decoded.mask is None
    assert decoded.mask_reference is None
    assert decoded.image_width == width
    assert decoded.image_height == height
