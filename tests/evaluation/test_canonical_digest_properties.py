"""Properties of ``canonical_digest``: the identity of every evaluation manifest and record.

A digest is only an identity if it depends on the content and on nothing else: not on the
order a dict was built in, not on a tuple standing where the JSON has a list, not on having
gone through the file and back. And it must lose nothing of the content: two values whose JSON
differs, even by one unit in the last place of a float, must not share a digest.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from contextmap.evaluation._persistence import canonical_digest, canonical_json
from property_strategies import finite_floats

JsonValue = Any

json_leaves = st.one_of(
    st.none(), st.booleans(), st.integers(), finite_floats, st.text(max_size=12)
)
json_values = st.recursive(
    json_leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=8), children, max_size=4),
    ),
    max_leaves=24,
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _reordered(value: JsonValue) -> JsonValue:
    """The same content with every dict built in reverse order and every list as a tuple."""
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return tuple(_reordered(item) for item in value)
    return value


def _leaf_paths(value: JsonValue, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """The path of every scalar and of every empty container, the places a change can be made."""
    if isinstance(value, dict) and value:
        return [leaf for key in value for leaf in _leaf_paths(value[key], (*path, key))]
    if isinstance(value, list) and value:
        return [
            leaf for index, item in enumerate(value) for leaf in _leaf_paths(item, (*path, index))
        ]
    return [path]


def _replaced(value: JsonValue, path: tuple[Any, ...], leaf: JsonValue) -> JsonValue:
    if not path:
        return leaf
    head, *rest = path
    if isinstance(value, dict):
        return {
            key: _replaced(item, tuple(rest), leaf) if key == head else item
            for key, item in value.items()
        }
    return [
        _replaced(item, tuple(rest), leaf) if index == head else item
        for index, item in enumerate(value)
    ]


def _at(value: JsonValue, path: tuple[Any, ...]) -> JsonValue:
    for step in path:
        value = value[step]
    return value


@given(json_values)
def test_the_digest_names_the_content_not_how_it_was_built(value: JsonValue) -> None:
    digest = canonical_digest(value)

    assert DIGEST.fullmatch(digest)
    assert canonical_digest(value) == digest
    assert canonical_digest(_reordered(value)) == digest


@given(json_values)
def test_the_digest_survives_the_trip_through_the_persisted_json(value: JsonValue) -> None:
    # Um manifest é relido do disco e redigerido: o valor relido tem de dar o mesmo digest.
    persisted = json.loads(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))

    assert canonical_digest(persisted) == canonical_digest(value)


@given(json_values, st.data())
def test_changing_any_value_changes_the_digest(value: JsonValue, data: st.DataObject) -> None:
    path = data.draw(st.sampled_from(_leaf_paths(value)))
    original = _at(value, path)
    replacement = data.draw(json_leaves)
    assume(canonical_json(replacement) != canonical_json(original))

    assert canonical_digest(_replaced(value, path, replacement)) != canonical_digest(value)


@given(json_values, st.data())
def test_a_float_one_unit_in_the_last_place_away_changes_the_digest(
    value: JsonValue, data: st.DataObject
) -> None:
    path = data.draw(st.sampled_from(_leaf_paths(value)))
    number = data.draw(finite_floats)
    neighbour = math.nextafter(number, data.draw(st.sampled_from([math.inf, -math.inf])))
    assume(math.isfinite(neighbour))

    assert canonical_digest(_replaced(value, path, number)) != canonical_digest(
        _replaced(value, path, neighbour)
    )
