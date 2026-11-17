"""The visual capability exposes visual contracts without legacy cross-owner aliases."""

from __future__ import annotations

from dataclasses import fields

import contextmap.visual_perception as visual


def test_visual_public_api_owns_only_visual_evidence() -> None:
    """Region and semantic contracts stay on their owners' public APIs."""
    names = set(visual.__all__)
    assert {
        "PreparedImage",
        "VisualFeature",
        "VisualEvidenceRunReader",
        "PerceptionResult",
    } <= names
    assert not names & {
        "Region2D",
        "RegionDiscovery",
        "SemanticClaim",
        "SemanticInterpreter",
        "PerceptionRun",
        "PerceptionRunReader",
        "PerceptionRunWriter",
        "PerceptionResultId",
    }


def test_derived_perception_result_has_no_execution_identity() -> None:
    """Joining selected artifacts does not create a fourth producer or persisted result ID."""
    names = {item.name for item in fields(visual.PerceptionResult)}
    assert {"region_artifact_id", "source_observation_id"} <= names
    assert not names & {"result_id", "run_id", "created_at"}


def test_every_advertised_visual_symbol_resolves() -> None:
    """The minimal public surface cannot advertise absent symbols."""
    assert all(hasattr(visual, name) for name in visual.__all__)
