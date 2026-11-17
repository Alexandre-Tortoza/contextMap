"""Group spatial observations by the physical observation they were inferred from.

Repeated inference over one physical frame -- several runs, backends or prompts --
is *correlated* evidence about one observation, not several independent views.
:func:`group_by_physical_observation` keeps the two levels apart: it groups the
spatial observations of an explicit selection of perception runs by physical
:class:`~contextmap.ingestion.SourceObservationId` and reports the counts a fusion
policy needs to avoid turning repeated inference into artificial confidence.

It never picks a winner, never discards a repeated inference and assumes no
statistical independence between backends. Runs are never merged implicitly: the
caller names exactly the runs to group, and an observation from any other run is
an error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionPerceptionRunManifest
from contextmap.semantic_fusion.evidence import PhysicalObservationGroup
from contextmap.semantic_fusion.invariants import require_canonical
from contextmap.sensor_association import SpatialObservation, SpatialObservationId
from contextmap.shared import SourceTimestamp

PHYSICAL_OBSERVATION_GROUPING_POLICY_ID = "physical-observation-grouping-v1"
"""Versioned rule: one group per physical observation of one sequence, from selected runs."""


@dataclass(frozen=True, kw_only=True)
class PhysicalObservationGrouping:
    """Spatial observations grouped by physical observation, with the counts kept apart.

    Attributes:
        grouping_policy_id: The versioned grouping rule.
        selected_region_artifact_ids: The perception runs the caller selected, sorted and unique.
        groups: One group per physical observation, sorted by observation.
        inference_variant_count: Number of distinct backend identities among the runs that
            produced evidence; runs of one backend configuration are repeated inference,
            not new variants.
    """

    grouping_policy_id: str
    selected_region_artifact_ids: tuple[str, ...]
    groups: tuple[PhysicalObservationGroup, ...]
    inference_variant_count: int

    def __post_init__(self) -> None:
        """Validate the ordering, that groups use only selected runs and the variant count.

        Raises:
            ValueError: If the policy is empty, a collection is not sorted and unique, a group
                uses a run outside the selected artifacts, or an inconsistent variant count
                with the runs that produced evidence.
        """
        if not self.grouping_policy_id.strip():
            raise ValueError("grouping_policy_id must not be empty")
        require_canonical(
            "selected_region_artifact_ids", self.selected_region_artifact_ids, lambda item: (item,)
        )
        require_canonical("groups", self.groups, lambda group: (group.physical_observation_id,))
        outside = set(self._used_run_ids()) - set(self.selected_region_artifact_ids)
        if outside:
            raise ValueError(
                f"groups use perception runs {sorted(outside)!r} that are not in "
                f"selected_region_artifact_ids {list(self.selected_region_artifact_ids)!r}"
            )
        runs = self.perception_run_count
        if not (0 if runs == 0 else 1) <= self.inference_variant_count <= runs:
            raise ValueError(
                f"inference_variant_count must be between 1 and the {runs} runs that produced "
                f"evidence (0 when there is none), got {self.inference_variant_count}"
            )

    @property
    def physical_observation_count(self) -> int:
        """Number of distinct physical observations."""
        return len(self.groups)

    @property
    def inference_result_count(self) -> int:
        """Number of inference results, correlated within each physical observation."""
        return sum(group.inference_result_count for group in self.groups)

    @property
    def perception_run_count(self) -> int:
        """Number of perception runs that produced evidence."""
        return len(self._used_run_ids())

    def _used_run_ids(self) -> set[str]:
        return {run_id for group in self.groups for run_id in group.region_artifact_ids}


def group_by_physical_observation(
    observations: Iterable[SpatialObservation],
    *,
    selected_region_runs: Sequence[RegionPerceptionRunManifest],
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
) -> PhysicalObservationGrouping:
    """Group spatial observations by physical observation under an explicit run selection.

    Args:
        observations: The spatial observations to group, in any order.
        selected_region_runs: The perception runs whose evidence may be grouped. Nothing outside
            this selection is ever merged in.
        acquisition_timestamps: When each physical observation was acquired, in one clock
            domain chosen by the caller (normally taken from the canonical sequence).

    Returns:
        The grouping; it does not depend on the order of ``observations`` or of
        ``selected_region_runs``.

    Raises:
        ValueError: If no run is selected, a run is selected twice or the runs process
            different sequences; an observation comes from an unselected run, from another
            sequence than its run, or is repeated; a run has two results for one physical
            observation, or one result describes two; or a physical observation has no
            acquisition timestamp.
    """
    runs = _index_selected_region_runs(selected_region_runs)
    seen: set[SpatialObservationId] = set()
    by_frame: dict[SourceObservationId, list[SpatialObservation]] = {}
    associated: dict[tuple[str, SourceObservationId, str], SpatialObservationId] = {}

    for observation in observations:
        run_id = observation.provenance.region_artifact_id
        run = runs.get(run_id)
        if run is None:
            raise ValueError(
                f"spatial observation {observation.spatial_observation_id!r} comes from "
                f"perception run {run_id!r}, which is not among the selected runs "
                f"{sorted(runs)!r}"
            )
        if observation.provenance.sequence_artifact_id != run.sequence_artifact_id:
            raise ValueError(
                f"spatial observation {observation.spatial_observation_id!r} belongs to "
                f"sequence artifact {observation.provenance.sequence_artifact_id!r}, but its "
                f"run {run_id!r} processed {run.sequence_artifact_id!r}"
            )
        if observation.spatial_observation_id in seen:
            raise ValueError(
                f"duplicate spatial observation {observation.spatial_observation_id!r}"
            )
        seen.add(observation.spatial_observation_id)
        regional_key = (run_id, observation.source_observation_id, observation.region_id)
        if regional_key in associated:
            raise ValueError(f"ambiguous association for regional evidence {regional_key!r}")
        associated[regional_key] = observation.spatial_observation_id

        frame = observation.source_observation_id
        by_frame.setdefault(frame, []).append(observation)

    groups = tuple(
        _group_of(frame, by_frame[frame], acquisition_timestamps) for frame in sorted(by_frame)
    )
    used_runs = {run_id for group in groups for run_id in group.region_artifact_ids}
    return PhysicalObservationGrouping(
        grouping_policy_id=PHYSICAL_OBSERVATION_GROUPING_POLICY_ID,
        selected_region_artifact_ids=tuple(sorted(runs)),
        groups=groups,
        inference_variant_count=len({_variant_of(runs[run_id]) for run_id in used_runs}),
    )


def _index_selected_region_runs(
    selected_region_runs: Sequence[RegionPerceptionRunManifest],
) -> dict[str, RegionPerceptionRunManifest]:
    if not selected_region_runs:
        raise ValueError("a grouping needs at least one perception run selected explicitly")
    runs: dict[str, RegionPerceptionRunManifest] = {}
    for run in selected_region_runs:
        if run.run_id in runs:
            raise ValueError(f"perception run {run.run_id!r} is selected more than once")
        runs[run.run_id] = run
    sequences = {run.sequence_artifact_id for run in runs.values()}
    if len(sequences) > 1:
        raise ValueError(
            f"selected perception runs must process one sequence artifact, "
            f"got {sorted(sequences)!r}"
        )
    return runs


def _group_of(
    frame: SourceObservationId,
    members: list[SpatialObservation],
    acquisition_timestamps: Mapping[SourceObservationId, SourceTimestamp],
) -> PhysicalObservationGroup:
    timestamp = acquisition_timestamps.get(frame)
    if timestamp is None:
        raise ValueError(f"no acquisition timestamp for physical observation {frame!r}")
    return PhysicalObservationGroup(
        physical_observation_id=frame,
        acquisition_timestamp=timestamp,
        spatial_observation_ids=tuple(sorted(item.spatial_observation_id for item in members)),
        semantic_interpretation_artifact_ids=(),
        region_artifact_ids=tuple(sorted({item.provenance.region_artifact_id for item in members})),
    )


def _variant_of(run: RegionPerceptionRunManifest) -> tuple[str, ...]:
    """Identify the actual regional backend/configuration used by the selected run."""
    return (
        run.discovery_backend_id,
        run.discovery_model,
        run.discovery_version,
        run.discovery_configuration_fingerprint,
        run.normalization_policy_fingerprint,
    )
