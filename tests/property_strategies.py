"""Hypothesis strategies shared by the property modules (``test_*_properties.py``).

The strategies reach the corners that fixed examples miss: the extremes of the float range,
subnormals, signed zero, identical values and empty collections. Only the property modules
import this one, so a base install without ``hypothesis`` never does (see ``conftest.py``).
"""

from __future__ import annotations

import math
import sys

from hypothesis import assume
from hypothesis import strategies as st

from contextmap.shared import Quaternion, SourceTimestamp, normalize_quaternion, quaternion_norm

MAX_NANOSECONDS = 999_999_999

identifiers = st.text(min_size=1, max_size=24)
"""Non-empty identities of any Unicode text: contracts only ask for them to be non-empty."""

BOUNDARY_FLOATS = (
    0.0,
    -0.0,
    math.ulp(0.0),
    -math.ulp(0.0),
    sys.float_info.min,
    -sys.float_info.min,
    1.0,
    -1.0,
    sys.float_info.max,
    -sys.float_info.max,
)
"""Signed zero, the smallest subnormal and normal, one, and the largest finite magnitude."""

finite_floats = st.sampled_from(BOUNDARY_FLOATS) | st.floats(allow_nan=False, allow_infinity=False)
"""Every finite double, with the boundary values drawn about as often as all the others.

A value such as ``-0.0`` in one of several coordinates would otherwise show up in only a few of
the examples a CI run draws, not in every run.
"""


def finite_vectors(bound: float | None = None) -> st.SearchStrategy[tuple[float, float, float]]:
    """``(x, y, z)`` of finite doubles, within ``[-bound, bound]`` when a bound is given."""
    component = finite_floats
    if bound is not None:
        within = tuple(value for value in BOUNDARY_FLOATS if abs(value) <= bound)
        component = st.sampled_from((*within, bound, -bound)) | st.floats(
            min_value=-bound, max_value=bound
        )
    return st.tuples(component, component, component)


_HALF_TURNS: tuple[Quaternion, ...] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


@st.composite
def rotations(draw: st.DrawFn) -> Quaternion:
    """Unit quaternions: arbitrary, near the identity, half turns, and both signs of each.

    The raw vector's norm is kept a normal double: below it ``normalize_quaternion`` does not
    return a unit quaternion (a strict expected failure in ``shared/test_geometry_properties``),
    and no stored rotation comes from such a vector.
    """
    raw = draw(
        st.one_of(
            st.tuples(*(st.floats(min_value=-1.0, max_value=1.0) for _ in range(4))),
            # Perto da identidade: componentes vetoriais minúsculos, inclusive subnormais.
            st.tuples(
                st.floats(min_value=-1e-9, max_value=1e-9),
                st.floats(min_value=-1e-9, max_value=1e-9),
                st.floats(min_value=-1e-9, max_value=1e-9),
                st.just(1.0),
            ),
            st.sampled_from(_HALF_TURNS),
        )
    )
    assume(quaternion_norm(raw) >= sys.float_info.min)
    rotation = normalize_quaternion(raw)
    # q e -q são a mesma rotação; os dois sinais precisam sair.
    if draw(st.booleans()):
        rotation = (-rotation[0], -rotation[1], -rotation[2], -rotation[3])
    assert math.isclose(quaternion_norm(rotation), 1.0, abs_tol=1e-12)
    return rotation


timestamps = st.builds(
    SourceTimestamp,
    seconds=st.integers(),
    nanoseconds=st.integers(min_value=0, max_value=MAX_NANOSECONDS),
    clock_id=st.text(max_size=24),
)
"""Timestamps across any epoch offset, both ends of the sub-second range included."""
