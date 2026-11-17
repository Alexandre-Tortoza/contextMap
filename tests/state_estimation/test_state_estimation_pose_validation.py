"""Regras de publicação de pose compartilhadas pelos backends."""

from pose_builders import make_pose

from contextmap.ingestion import SourceObservationId
from contextmap.state_estimation import DiagnosticSeverity, TrajectoryGap
from contextmap.state_estimation.pose_validation import timestamp_gap_diagnostic

MS = 1_000_000


def test_a_timestamp_gap_diagnostic_is_a_warning_coded_with_the_backend_identity() -> None:
    earlier = make_pose(0, time_ns=0)
    later = make_pose(1, time_ns=900 * MS)
    gap = TrajectoryGap(
        previous_estimate_id=earlier.estimate_id,
        next_estimate_id=later.estimate_id,
        duration_ns=900 * MS,
    )

    diagnostic = timestamp_gap_diagnostic(
        backend_id="some_backend",
        gap=gap,
        max_gap_ns=250 * MS,
        observation_id=SourceObservationId("obs-1"),
    )

    assert diagnostic.severity is DiagnosticSeverity.WARNING
    assert diagnostic.code == "some_backend.timestamp_gap"
    assert diagnostic.observation_id == SourceObservationId("obs-1")
    assert str(gap.duration_ns) in diagnostic.message
    assert "max_gap_ns=250000000" in diagnostic.message
