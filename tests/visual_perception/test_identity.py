"""Feature identity distinguishes visual producers and physical acquisition."""

from typing import Any

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.visual_perception import feature_id_for


def test_feature_identity_is_stable() -> None:
    arguments: dict[str, Any] = dict(
        run_id="visual-a",
        source_observation_id=SourceObservationId("frame-a"),
        producer_id="dense",
        index=0,
    )
    assert feature_id_for(**arguments) == feature_id_for(**arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "visual-b"),
        ("source_observation_id", SourceObservationId("frame-b")),
        ("producer_id", "region"),
        ("index", 1),
    ],
)
def test_every_feature_identity_axis_changes_the_evidence_identity(
    field: str, value: object
) -> None:
    arguments: dict[str, Any] = dict(
        run_id="visual-a",
        source_observation_id=SourceObservationId("frame-a"),
        producer_id="dense",
        index=0,
    )
    assert feature_id_for(**arguments) != feature_id_for(**{**arguments, field: value})
