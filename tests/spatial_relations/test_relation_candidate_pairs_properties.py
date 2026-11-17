"""Property of the candidate sweep at the reach boundary: it prunes exactly by the measured gap.

The differential of ``test_relation_candidate_pairs_differential.py`` samples scenes; this property
lands the gap of two boxes on every float within two units in the last place of the reach. There
the sweep's early stop once computed ``upper face + reach`` while the candidate stage measures
``lower face - upper face``: the two roundings disagree, and a pair whose measured gap is exactly
the reach was left out (#641). The sweep now stops on the same subtraction, so a pair is
enumerated if and only if its measured gap is within the reach.
"""

from __future__ import annotations

import math

from hypothesis import example, given
from hypothesis import strategies as st
from relation_scene import Scene

from contextmap.spatial_relations._bounds import bounds_gap_m
from contextmap.spatial_relations.candidates import _neighbor_pairs


# O contraexemplo reduzido da #614: gap medido 0.5 == alcance, mas 0.8372...1 > 0.3372... + 0.5.
@example(upper_face=0.337290862214726, reach=0.5, ulps=1)
@given(
    upper_face=st.floats(min_value=-100.0, max_value=100.0),
    # De um nanômetro para cima: abaixo, o quadrado de uma separação some nos subnormais e o
    # gap euclidiano do estágio vira zero, uma escala sem sentido físico.
    reach=st.floats(min_value=1e-9, max_value=5.0),
    ulps=st.integers(min_value=-2, max_value=2),
)
def test_a_pair_is_enumerated_exactly_when_its_gap_is_within_reach(
    upper_face: float, reach: float, ulps: int
) -> None:
    lower_face = upper_face + reach
    for _ in range(abs(ulps)):
        lower_face = math.nextafter(lower_face, math.inf if ulps > 0 else -math.inf)
    scene = Scene()
    scene.add_box("first", (upper_face - 1.0, 0.0, 0.0), (upper_face, 1.0, 1.0))
    scene.add_box("second", (lower_face, 0.0, 0.0), (lower_face + 1.0, 1.0, 1.0))
    first, second = scene.geometry("first"), scene.geometry("second")

    pairs = list(_neighbor_pairs([first, second], reach))

    # As caixas só se separam em x, então o gap medido é a separação nesse eixo: o par é
    # enumerado quando ela cabe no alcance, e só então (nem par perdido, nem par a mais).
    within_reach = bounds_gap_m(first.bounds, second.bounds) <= reach
    assert pairs == ([(0, 1)] if within_reach else [])
