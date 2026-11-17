"""Stage executors: each runs one capability and writes only where the request says.

An executor is thin. It opens its inputs through :meth:`StageRequest.directory_of`, calls the
capability's public service with the scientific policies it was built with, hands the capability's
writer the directory it was given (``StageRequest.output_dir``) and reports the artifact as an
:class:`ArtifactRef` with its location. It computes no path, allocates no identity and decides no
science: every threshold and policy is supplied by whoever builds the executor, and the run id it
records is derived from the execution (:meth:`StageRequest.identity`), so an identical execution
produces the identical artifact.

This module is imported explicitly (``contextmap.runtime.executors``), never by importing
``contextmap.runtime``: it depends on the public root of every capability it executes, and the
runtime package itself must load none of them. It imports public roots only, never a backend.

A stage produces exactly one artifact, and the executors consume exactly one run per input: a
request with several runs for one input is refused, never resolved by picking one. Comparing runs
means running the runtime more than once and reading the artifacts through their references.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from contextmap.artifact import (
    CONTEXT_MAP_ASSEMBLY_POLICY_ID,
    ArtifactKind,
    ContextMapId,
    ContextMapMetadata,
    DeclaredCapabilities,
    MapCapability,
    MapCreation,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
    UpstreamArtifact,
    ValidationLevel,
    ValidationStatus,
    artifact_digest,
    assemble_context_map_with_metrics,
    estimator_local_map_frame,
    validate_context_map_artifact,
    write_context_map_with_metrics,
)
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ConservativeResolutionPolicy,
    EntityResolutionRunId,
    EntityResolutionRunReader,
    EntityResolutionRunWriter,
    MatchEvidenceBuilder,
    ResolutionDebugLevel,
    ResolvedBeliefPolicy,
    lineage_from_mapping_manifest,
    materialize_resolved_entities,
    resolve_candidate_pairs,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    MapDebugLevel,
    MotionCorrectionPolicy,
    PoseInterpolationDeskewPolicy,
    ScanVoxelPolicy,
    build_geometry_mapping_plan_from_artifacts,
)
from contextmap.ingestion import (
    ClockPlausibilityError,
    FullSequenceSelection,
    ImageEncoding,
    ImageObservation,
    SequenceArtifactReader,
    SequenceSelection,
    SequenceSelectionError,
    SourceObservation,
    SourceObservationId,
    decode_selection,
    encode_selection,
    resolve_selection_offsets,
    selection_identity,
    validate_cross_source_clock_plausibility,
)
from contextmap.point_representation import (
    ObservationSupportAnchorPolicy,
    PointEncoder,
    PointRepresentationDebugLevel,
    PointRepresentationRunId,
    PointRepresentationRunWriter,
    RepresentationService,
    representation_space_fingerprint,
)
from contextmap.region_perception import (
    REFINEMENT_ACCEPTANCE_POLICY,
    AuditedRegionDiscovery,
    GroundingQuery,
    IneligibleRegion,
    IneligibleRegionReason,
    Region2D,
    RegionDiscoveryAudit,
    RegionDiscoveryIdentity,
    RegionGrounding,
    RegionGroundingExecution,
    RegionGroundingRequest,
    RegionMaskResolver,
    RegionPerceptionRunManifest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    RegionRefinement,
    RegionRefinementExecution,
    RegionRefinementRequest,
    assemble_regions,
    namespace_regions,
    refinement_prompts_from,
    region_mask_pixels,
)
from contextmap.runtime.artifacts import ArtifactRef, inventory_digest
from contextmap.runtime.components.contracts import (
    ASSOCIATION,
    CONTEXT_MAP,
    ENTITIES,
    FUSION,
    GEOMETRY,
    REGIONS,
    RELATIONS,
    REPRESENTATION,
    RESOLUTION,
    SEMANTIC_INTERPRETATION,
    SEMANTIC_SCORES,
    TRAJECTORY,
    VISUAL_EVIDENCE,
)
from contextmap.runtime.composition import (
    FeatureBuildScope,
    FeatureFactory,
    RegionDiscoveryPlan,
    RegionGroundingPlan,
    RegionRefinementPlan,
    SemanticInterpreterFactory,
    SemanticScoringPlan,
)
from contextmap.runtime.image_masks import CameraImageMasks
from contextmap.runtime.pipeline import StageRequest
from contextmap.semantic_fusion import (
    BaselineAccumulationPolicy,
    FusionOutcome,
    FusionRunLineage,
    GeometryOverlapSupportPolicy,
    QualityAwareAccumulationPolicy,
    SemanticFusionDebugLevel,
    SemanticFusionRunId,
    SemanticFusionRunReader,
    SemanticFusionRunWriter,
    accumulate_baseline_evidence,
    accumulate_quality_aware_evidence,
    build_fusion_supports,
    group_by_physical_observation,
)
from contextmap.semantic_interpretation import (
    SceneContext,
    SemanticDebugLevel,
    SemanticInterpretationRunId,
    SemanticInterpretationRunReader,
    SemanticInterpretationRunWriter,
    SemanticInterpretationService,
    SemanticObservation,
    SemanticObservationStatus,
    SemanticRequestPolicy,
)
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    MappingDebugLevel,
    SemanticMapId,
    SemanticMappingRunId,
    SemanticMappingRunReader,
    SemanticMappingRunWriter,
    lineage_from_fusion_manifest,
    materialize_entities,
)
from contextmap.semantic_scoring import (
    SemanticScore,
    SemanticScoringLineage,
    SemanticScoringRunId,
    SemanticScoringRunReader,
    SemanticScoringRunWriter,
)
from contextmap.sensor_association import (
    AssociationFrameInput,
    BoxDepthComponentPolicy,
    CandidateGeometryPolicy,
    DenseChannel,
    DiagnosticTolerances,
    InterpolationPolicy,
    OcclusionPolicy,
    SensorAssociationDebugLevel,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
    SpatialObservation,
    SpatialObservationId,
)
from contextmap.shared import (
    ArtifactReference,
    BackendProvenance,
    SourceTimestamp,
)
from contextmap.spatial_relations import (
    AxisDirection,
    RelationsRunDebugLevel,
    RelationsRunPolicies,
    SpatialRelationsRunId,
    SpatialRelationsRunReader,
    SpatialRelationsRunWriter,
    derive_relations,
    lineage_from_resolution_manifest,
)
from contextmap.state_estimation import (
    GeometryRequirements,
    LookupPolicy,
    StateEstimationDebugLevel,
    StateEstimationRequest,
    StateEstimationRunId,
    StateEstimationRunReader,
    StateEstimationRunWriter,
    StateEstimator,
    TrajectoryId,
    TrajectoryLookup,
    execute_state_estimation,
)
from contextmap.visual_perception import (
    CANONICAL_PRESET_V1,
    POOLING_POLICY,
    DenseFeatureMap,
    ExclusionRegion,
    FeatureDebugLevel,
    FeatureEventStatus,
    FeatureExtractionDiagnostic,
    FeatureId,
    FeatureScope,
    PerceptionResult,
    PipelinePreset,
    PreparedImage,
    RegionAssociationError,
    RegionFeatureDiagnostic,
    ReleasableBackend,
    SourceImage,
    StageDefinition,
    StageOutcome,
    StageSpec,
    StageStatus,
    ValidRegion,
    VisualEvidenceResult,
    VisualEvidenceRunError,
    VisualEvidenceRunId,
    VisualEvidenceRunReader,
    VisualEvidenceRunWriter,
    VisualFeature,
    box_mask_shape,
    dense_feature_map_of,
    execute_stage_graph,
    execute_stage_graphs_stage_major,
    feature_id_for,
    pool_region_feature,
    pooling_backend_provenance,
    prepare_image,
    resolve_pipeline,
    with_dense_region_pooling,
)

__all__ = [
    "EntityResolutionExecutor",
    "ExecutorError",
    "GeometricMappingExecutor",
    "PointRepresentationExecutor",
    "RegionPerceptionExecutor",
    "SemanticFusionExecutor",
    "SemanticInterpretationExecutor",
    "SemanticScoringExecutor",
    "SensorAssociationExecutor",
    "SpatialRelationsExecutor",
    "StateEstimationExecutor",
    "VisualPerceptionExecutor",
    "inventory_digest",
]


def _preparation_constraints(
    image_masks: Mapping[str, CameraImageMasks],
    calibration_id: object | None,
    width: int,
    height: int,
) -> tuple[ValidRegion | None, tuple[ExclusionRegion, ...]]:
    """Resolve one image's static valid/exclusion regions from the per-camera masks.

    Args:
        image_masks: The decoded masks keyed by camera calibration id.
        calibration_id: The image's camera calibration id, or ``None`` when it has none.
        width: Prepared image width in pixels.
        height: Prepared image height in pixels.

    Returns:
        The valid region (or ``None``) and the exclusion regions for the image; empty when no
        mask is configured for the camera.

    Raises:
        ExecutorError: If a configured mask's dimensions differ from the prepared image.
    """
    if not image_masks or calibration_id is None:
        return None, ()
    camera = str(calibration_id)
    masks = image_masks.get(camera)
    if masks is None:
        return None, ()
    try:
        return masks.constraints_for(width=width, height=height, camera=camera)
    except ValueError as error:
        raise ExecutorError(str(error)) from error


class ExecutorError(ValueError):
    """Raised when a request cannot be served: no directory, or not exactly one run per input."""


def _output(request: StageRequest) -> Path:
    if request.output_dir is None or request.workspace is None:
        raise ExecutorError(
            f"stage {request.stage_id!r} needs the run's output directory: run it with a journal"
        )
    return request.output_dir


def _one(request: StageRequest, name: str) -> Path:
    refs = request.inputs.get(name, ())
    if len(refs) != 1:
        raise ExecutorError(
            f"stage {request.stage_id!r} consumes exactly one run of input {name!r}, got "
            f"{len(refs)}: run the runtime once per run instead of choosing one"
        )
    return request.directory_of(refs[0])


def _many(request: StageRequest, name: str) -> tuple[Path, ...]:
    """Return the distinct runs of an input that accepts several, in a deterministic order.

    One run reached twice (through two ContextRuns of a build, say) is one piece of evidence and
    is read once; two different contents under one identity are refused.

    Raises:
        ExecutorError: If the input is empty, or names one run with two different contents.
    """
    distinct: dict[str, ArtifactRef] = {}
    for ref in request.inputs.get(name, ()):
        seen = distinct.setdefault(ref.artifact_id, ref)
        if seen.content_hash != ref.content_hash:
            raise ExecutorError(
                f"input {name!r} of stage {request.stage_id!r} names run {ref.artifact_id!r} "
                "twice with different content"
            )
    if not distinct:
        raise ExecutorError(f"stage {request.stage_id!r} needs at least one run of {name!r}")
    return tuple(request.directory_of(distinct[run_id]) for run_id in sorted(distinct))


def _optional_one(request: StageRequest, name: str) -> Path | None:
    refs = request.inputs.get(name, ())
    if not refs:
        return None
    if len(refs) != 1:
        raise ExecutorError(
            f"stage {request.stage_id!r} consumes at most one run of input {name!r}, got "
            f"{len(refs)}: run the runtime once per run instead of choosing one"
        )
    return request.directory_of(refs[0])


def _observation_selection(request: StageRequest) -> SequenceSelection:
    """Decode the observation selection of an observation-scoped stage (issue #497).

    The document must be the canonical form :func:`~contextmap.ingestion.encode_selection`
    gives the selection it decodes to: fields of another kind, which a merge of configuration
    layers can leave behind, are refused rather than silently ignored.

    Returns:
        The selection, or the whole sequence when the stage has none.

    Raises:
        ExecutorError: If the document is not a selection in canonical form.
    """
    document = request.observation_selection
    if document is None:
        return FullSequenceSelection()
    plain = _plain(document)
    try:
        selection = decode_selection(plain)
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutorError(
            f"inputs.observation_selection {plain!r} is not a sequence selection: {error}"
        ) from error
    canonical = encode_selection(selection)
    if canonical != plain:
        raise ExecutorError(
            f"inputs.observation_selection {plain!r} is not in canonical form; write {canonical!r}"
        )
    return selection


def _plain(value: Any) -> Any:
    """Turn a frozen configuration value back into plain JSON containers."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _recorded_code_version(code_version: str) -> str:
    """Return the code version a run must record, refusing an empty one.

    Semantic Fusion, Semantic Mapping, Entity Resolution and Spatial Relations write it into
    their run, and a ``ContextMap`` refuses to cite a run that records none (#659): an unknown
    code is an explicit error here, never an empty string in an artifact.

    Raises:
        ValueError: If ``code_version`` is empty or blank.
    """
    if not code_version.strip():
        raise ValueError("code_version must not be empty: a run records the code that produced it")
    return code_version


def _configured_stage_identity(request: StageRequest, configuration_identity: str) -> str:
    """Combine a stage's execution identity with the effective configuration its backends apply.

    ``request.identity()`` only covers the parameters the configuration declares; the run id of
    a :class:`~contextmap.runtime.pipeline.ConfiguredStageExecutor` also has to change when a
    default or algorithm its backends resolve changes, as its reuse key does.
    """
    combined = f"{request.identity()}|configuration={configuration_identity}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]


def _reference(
    request: StageRequest, contract: str, artifact_id: str, inventory: Sequence[object]
) -> ArtifactRef:
    output = _output(request)
    assert request.workspace is not None  # garantido por `_output`
    # A location é relativa ao root de saída (o run store para um estágio comum; o maps root
    # para um artefato terminal redirecionado), nunca ao workspace de resolução de inputs.
    output_root = request.output_root or request.workspace
    return ArtifactRef(
        stage_id=request.stage_id,
        contract=contract,
        artifact_id=artifact_id,
        content_hash=inventory_digest(inventory),
        location=output.relative_to(output_root).as_posix(),
    )


_GROUND_TRUTH_POSE_ROLE = "ground_truth"
"""issue #555: an auxiliary pose sequence tagged with this role never drives the trajectory
without :attr:`StateEstimationExecutor._allow_ground_truth_trajectory`'s explicit opt-in."""


class StateEstimationExecutor:
    """Estimates the trajectory of the ingested sequence with the selected estimator."""

    def __init__(
        self,
        estimator: StateEstimator,
        *,
        downstream: Sequence[GeometryRequirements] = (),
        allow_ground_truth_trajectory: bool = False,
        debug_level: StateEstimationDebugLevel = StateEstimationDebugLevel.NONE,
    ) -> None:
        """Bind the executor to a composed estimator and the readiness it reports downstream.

        Args:
            estimator: The selected state estimation backend.
            downstream: Requirements later capabilities report as readiness, never blocking.
            allow_ground_truth_trajectory: Explicit opt-in (issue #555) letting an optional
                auxiliary pose sequence tagged ``pose_role="ground_truth"`` become the
                operational trajectory. ``False`` (default) drops a ground-truth-tagged
                auxiliary pose from the merge entirely: the run behaves exactly as if no
                auxiliary pose sequence were configured.
            debug_level: Amount of non-contractual debug evidence the run persists, from
                ``policies.debug_level``. ``NONE`` (default) preserves the prior behavior.
        """
        self._estimator = estimator
        self._downstream = tuple(downstream)
        self._allow_ground_truth_trajectory = allow_ground_truth_trajectory
        self._debug_level = debug_level

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Estimate, persist and reference the trajectory of the ``sequence`` input.

        When the optional ``pose_sequence`` input is present (issue #555's auxiliary pose
        bridge), its observations are merged in after its clock relationship with ``sequence``
        passes an explicit plausibility check (never accepted on a matching
        ``timestamp_clock_id`` string alone) and its declared ``pose_role`` is checked. When the
        merge actually contributes observations, ``StateEstimationRequest`` also names the
        auxiliary artifact and selection explicitly (``auxiliary_sequence_artifact_id``/
        ``auxiliary_selection_id``), so the trajectory's own persisted provenance stays
        self-portable even when a reader only has the ``StateEstimationRunArtifact``.
        """
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        observations: tuple[SourceObservation, ...] = tuple(sequence.list_observations())

        auxiliary_sequence_artifact_id = None
        auxiliary_selection_id = None
        pose_path = _optional_one(request, "pose_sequence")
        if pose_path is not None:
            pose_sequence = SequenceArtifactReader(pose_path)
            auxiliary = tuple(pose_sequence.list_observations())
            merged = self._select_auxiliary_pose(observations, auxiliary)
            observations = observations + merged
            if merged:
                auxiliary_sequence_artifact_id = pose_sequence.manifest.artifact_id
                auxiliary_selection_id = selection_identity(
                    pose_sequence.manifest.artifact_id, FullSequenceSelection()
                )

        identity = request.identity()
        outcome = execute_state_estimation(
            self._estimator,
            StateEstimationRequest(
                trajectory_id=TrajectoryId(f"{identity}--trajectory"),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id=selection_identity(
                    sequence.manifest.artifact_id, FullSequenceSelection()
                ),
                observations=observations,
                calibration=sequence.read_calibration(),
                auxiliary_sequence_artifact_id=auxiliary_sequence_artifact_id,
                auxiliary_selection_id=auxiliary_selection_id,
            ),
            downstream=self._downstream,
        )
        manifest = StateEstimationRunWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=StateEstimationRunId(identity),
            run_index=request.run_number(),
            debug_level=self._debug_level,
        ).finalize(outcome)
        return _reference(request, TRAJECTORY, str(manifest.run_id), manifest.file_inventory)

    def _select_auxiliary_pose(
        self, primary: Sequence[SourceObservation], auxiliary: Sequence[SourceObservation]
    ) -> tuple[SourceObservation, ...]:
        """Validate an auxiliary pose sequence and decide whether it joins the merge.

        Args:
            primary: The main sequence's observations, before any merge.
            auxiliary: The auxiliary pose sequence's observations.

        Returns:
            ``auxiliary`` unchanged, or ``()`` when its declared ``pose_role`` is
            ``"ground_truth"`` and :attr:`_allow_ground_truth_trajectory` is not set.

        Raises:
            ExecutorError: If ``auxiliary`` does not declare one single, recognized
                ``pose_role``, or if the two sequences' clock relationship fails the
                plausibility check (issue #555 requires this before ever combining two
                artifacts' observations). The ground-truth safeguard is checked *before* the
                clock check: a ground-truth-tagged auxiliary sequence that will be dropped
                anyway must not block the run just because its clock is implausible --
                ``operational_only`` must behave exactly as if the auxiliary sequence were
                never configured, including when it is malformed in a way that is irrelevant to
                a dropped sequence.
        """
        pose_role = self._auxiliary_pose_role(auxiliary)
        if pose_role == _GROUND_TRUTH_POSE_ROLE and not self._allow_ground_truth_trajectory:
            return ()
        try:
            validate_cross_source_clock_plausibility(primary, auxiliary)
        except ClockPlausibilityError as error:
            raise ExecutorError(f"cannot merge auxiliary pose sequence: {error}") from error
        return tuple(auxiliary)

    @staticmethod
    def _auxiliary_pose_role(auxiliary: Sequence[SourceObservation]) -> str:
        """Return the auxiliary sequence's single, consistent ``pose_role``.

        Raises:
            ExecutorError: If ``auxiliary`` is empty, declares no ``pose_role`` on some
                observation, or declares more than one distinct value -- inconsistent or
                missing pose role is never silently resolved (issue #555).
        """
        roles = {observation.provenance.raw_metadata.get("pose_role") for observation in auxiliary}
        if len(roles) != 1 or None in roles:
            raise ExecutorError(
                "auxiliary pose sequence observations must declare a single, consistent "
                f"pose_role in their provenance; got {sorted(str(role) for role in roles)!r}"
            )
        (role,) = roles
        assert isinstance(role, str)
        return role


class GeometricMappingExecutor:
    """Accumulates the LiDAR scans into a persistent map in the trajectory's frame."""

    def __init__(
        self,
        *,
        pose_lookup: LookupPolicy,
        motion_correction: MotionCorrectionPolicy,
        deskew: PoseInterpolationDeskewPolicy | None = None,
        aggregation: ScanVoxelPolicy | None = None,
        code_version: str | None = None,
        debug_level: MapDebugLevel = MapDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the policies that decide which scans become geometry.

        ``deskew`` corrects each scan for platform motion before it is mapped; ``None`` maps each
        scan with the single pose of its timestamp, exactly as before the corrector existed.
        """
        self._pose_lookup = pose_lookup
        self._motion_correction = motion_correction
        self._deskew = deskew
        self._aggregation = aggregation
        self._code_version = code_version
        self._debug_level = debug_level

    @property
    def deskew(self) -> PoseInterpolationDeskewPolicy | None:
        """The composed deskew policy, or ``None`` when scans are mapped without correction."""
        return self._deskew

    @property
    def aggregation(self) -> ScanVoxelPolicy | None:
        """The composed intra-scan voxel rule, or ``None`` when scans are not aggregated."""
        return self._aggregation

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Map the ``sequence`` input through the ``trajectory`` input and reference the map."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        plan = build_geometry_mapping_plan_from_artifacts(
            sequence=sequence,
            selection=FullSequenceSelection(),
            run=StateEstimationRunReader(_one(request, "trajectory")),
            pose_lookup=self._pose_lookup,
            motion_correction_policy=self._motion_correction,
            deskew=self._deskew,
        )
        manifest = GeometricMapArtifactWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=GeometricMapRunId(request.identity()),
            run_index=request.run_number(),
            debug_level=self._debug_level,
        ).finalize(plan=plan, aggregation=self._aggregation, code_version=self._code_version)
        return _reference(request, GEOMETRY, str(manifest.run_id), manifest.file_inventory)


class SensorAssociationExecutor:
    """Associates the regions of one perception run with the persistent map.

    With ``dense_sampling`` it also samples the dense feature maps of that run, one declared
    :class:`~contextmap.sensor_association.DenseChannel` per dense feature (#651). Without it,
    the association is the geometry-only one and declares no channel.
    """

    def __init__(
        self,
        *,
        candidates: CandidateGeometryPolicy,
        occlusion: OcclusionPolicy,
        tolerances: DiagnosticTolerances,
        pose_policy: LookupPolicy,
        dense_sampling: InterpolationPolicy | None = None,
        box_policy: BoxDepthComponentPolicy | None = None,
        code_version: str | None = None,
        debug_level: SensorAssociationDebugLevel = SensorAssociationDebugLevel.NONE,
        image_masks: Mapping[str, CameraImageMasks] | None = None,
    ) -> None:
        """Bind the executor to the candidate, visibility, tolerance and pose rules.

        Args:
            candidates: Which map geometry each frame evaluates.
            occlusion: The visibility rule.
            tolerances: The diagnostic tolerances.
            pose_policy: Which pose lookups are acceptable for an image timestamp.
            dense_sampling: How every dense channel is read at a pixel, when the optional
                ``sensor_association.dense_sampling`` component is selected. ``None`` keeps
                the geometry-only association: no channel is declared and nothing dense is
                written.
            box_policy: The box rule of membership for regions without mask, when the
                optional ``sensor_association.membership`` component is selected. ``None``
                keeps the mask rule: a box-only region is skipped.
            code_version: Code revision that produces the run, when known.
            debug_level: Amount of non-contractual debug evidence the run persists, from
                ``policies.debug_level``. ``NONE`` (default) preserves the prior behavior.
            image_masks: Static per-camera valid/exclusion regions, keyed by camera
                calibration id. A point projecting into an exclusion region (or outside the
                valid region) becomes ``OUTSIDE_VALID_SUPPORT`` and never supports a region.
                ``None`` or empty keeps the whole image usable.
        """
        self._candidates = candidates
        self._occlusion = occlusion
        self._tolerances = tolerances
        self._pose_policy = pose_policy
        self._dense_sampling = dense_sampling
        self._box_policy = box_policy
        self._code_version = code_version
        self._debug_level = debug_level
        self._image_masks = image_masks or {}

    @property
    def box_policy(self) -> BoxDepthComponentPolicy | None:
        """The box rule of membership, or ``None`` when the run keeps the mask rule."""
        return self._box_policy

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Associate every perceived frame of the ``perception`` input and reference the run."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        trajectory = StateEstimationRunReader(_one(request, "trajectory"))
        calibration = sequence.read_calibration()
        if calibration is None:
            raise ExecutorError("the sequence artifact carries no calibration")
        # Só as imagens são decodificadas: list_observations() decodificaria também os 364 MB de
        # pointcloud e os 20781 registros de IMU do corridor-02, que esta associação nunca usa
        # (#511). O índice guarda apenas o offset de cada imagem: o payload é lido quando o
        # frame daquela observação é construído e solto logo depois, porque mantê-los todos
        # vivos custava 2,19 GB no corridor-02 e faria a entrada escalar com o número de
        # frames, exatamente o que o #563 removeu do lado da saída.
        offsets = {
            str(entry.observation_id): entry.offset
            for entry in sequence.iter_index()
            if entry.modality == "image"
        }
        regions = RegionPerceptionRunReader(_one(request, "regions"))
        if regions.manifest.sequence_artifact_id != sequence.manifest.artifact_id:
            raise ExecutorError("regional artifact belongs to another sequence")
        visual = (
            VisualEvidenceRunReader(_one(request, "visual_evidence"))
            if self._dense_sampling is not None
            else None
        )
        if visual is not None and (
            visual.manifest.region_artifact_id != regions.manifest.run_id
            or visual.manifest.sequence_artifact_id != sequence.manifest.artifact_id
            or set(visual.manifest.source_observation_ids)
            != set(regions.manifest.source_observation_ids)
        ):
            raise ExecutorError(
                "dense visual evidence does not cover the selected regional artifact"
            )
        # Os canais são declarados antes do primeiro frame, então a run de percepção é lida
        # uma vez a mais só para descobri-los; sem o componente, nada é lido.
        channels = () if visual is None else self._dense_channels(visual)

        grouped: dict[str, list[Region2D]] = {}
        for region in regions.read_regions():
            grouped.setdefault(str(region.source_observation_id), []).append(region)

        def frames() -> Iterator[AssociationFrameInput]:
            for frame in regions.iter_frames():
                source_id = str(frame.image.source_observation_id)
                offset = offsets.get(source_id)
                if offset is None:
                    raise ExecutorError(f"regional frame {source_id!r} is absent from the sequence")
                image = sequence.observation_at(offset)
                if isinstance(image, ImageObservation):
                    result = (
                        None if visual is None else visual.result(SourceObservationId(source_id))
                    )
                    yield AssociationFrameInput(
                        observation=image,
                        prepared_image=frame.image,
                        region_artifact_id=str(regions.manifest.run_id),
                        regions=tuple(grouped.get(source_id, ())),
                        visual_artifact_id=None if visual is None else str(visual.manifest.run_id),
                        visual_evidence=result,
                        dense_maps={}
                        if result is None
                        else _dense_maps(cast(VisualEvidenceRunReader, visual), result),
                    )

        writer = SensorAssociationRunWriter(
            output_dir=output,
            sequence_name=sequence.manifest.sequence_name,
            run_id=SensorAssociationRunId(request.identity()),
            run_index=request.run_number(),
            debug_level=self._debug_level,
        )
        # Cada frame é construído sob demanda, persistido e liberado dentro da transação: nem
        # a entrada nem o resultado do run ficam inteiros em memória (#563).
        with (
            GeometricMapArtifactReader(_one(request, "geometry")) as geometry,
            writer.transaction() as run,
        ):
            outcome = SensorAssociationService().run(
                SensorAssociationRequest(
                    association_run_id=request.identity(),
                    sequence_artifact_id=sequence.manifest.artifact_id,
                    # A associação cobre exatamente os frames da run de percepção: a seleção é a
                    # dela, nunca uma reconstruída aqui (#497).
                    selection_id=regions.manifest.selection_id,
                    geometry=geometry.geometry(),
                    trajectory=TrajectoryLookup(trajectory.trajectory()),
                    pose_policy=self._pose_policy,
                    calibration=calibration,
                    candidate_policy=self._candidates,
                    occlusion_policy=self._occlusion,
                    tolerances=self._tolerances,
                    frames=frames(),
                    dense_channels=channels,
                    state_estimation_run_id=trajectory.manifest.run_id,
                    code_version=self._code_version,
                    # A reopened perception run never inlines mask pixels (#378); this
                    # is how association resolves a region's mask_reference on demand.
                    mask_loader=regions.mask_store(),
                    box_policy=self._box_policy,
                ),
                sink=run,
            )
            manifest = run.finalize(outcome)
        return _reference(request, ASSOCIATION, str(manifest.run_id), manifest.file_inventory)

    def _dense_channels(self, perception: VisualEvidenceRunReader) -> tuple[DenseChannel, ...]:
        """Declare one channel per dense feature of the perception run, before any frame.

        A channel is a (feature stage, embedding space) pair of the run, identified by
        :func:`_dense_channel_id`; every dense feature of the run must have its recorded
        geometry. A frame that lacks the map of a declared channel is refused later by the
        service, never sampled as an empty channel.

        Returns:
            The channels, ordered by identity; ``()`` when dense sampling is not selected.

        Raises:
            ExecutorError: If dense sampling is selected and the run has no dense feature, or
                a dense feature whose geometry the run did not record.
        """
        if self._dense_sampling is None:
            return ()
        identities: set[str] = set()
        for result in perception.iter_results():
            identities.update(_dense_maps(perception, result))
        if not identities:
            raise ExecutorError(
                f"dense sampling is selected, but perception run "
                f"{perception.manifest.run_id!r} has no dense feature to sample"
            )
        return tuple(
            DenseChannel(channel_id=channel_id, interpolation=self._dense_sampling)
            for channel_id in sorted(identities)
        )


def _dense_channel_id(stage_id: str, embedding_space_id: str) -> str:
    """Name the dense evidence channel of one feature stage in one embedding space.

    The identity is a function of the run's evidence, never of the backend's name: a stage
    whose space changes (another checkpoint, layer or normalization) is another channel.
    """
    return f"{stage_id}@{embedding_space_id}"


def _dense_maps(
    perception: VisualEvidenceRunReader, result: VisualEvidenceResult
) -> dict[str, DenseFeatureMap]:
    """Rebuild the dense maps of one frame, keyed by the channel each belongs to.

    Raises:
        ExecutorError: If a dense feature has no recorded geometry, or two dense features of
            the frame fall in one channel (the channel would not name one map).
    """
    maps: dict[str, DenseFeatureMap] = {}
    for feature in result.features:
        if feature.scope is not FeatureScope.DENSE:
            continue
        try:
            recorded = perception.dense_feature_map(result.source_observation_id, feature)
        except VisualEvidenceRunError as error:
            raise ExecutorError(
                f"dense sampling needs the dense geometry of every dense feature of perception "
                f"run {perception.manifest.run_id!r}: {error}"
            ) from error
        channel_id = _dense_channel_id(recorded.stage_id, feature.embedding_space_id)
        if channel_id in maps:
            raise ExecutorError(
                f"frame {result.source_observation_id!r} has more than one dense feature in "
                f"channel {channel_id!r}"
            )
        maps[channel_id] = recorded.dense_map
    return maps


class PointRepresentationExecutor:
    """Represents the local 3D structure the association observed, as a derived output (#663).

    The executor only composes Point Representation's public pieces: the capability's versioned
    policy picks the centers from the observations of the ``association`` run, its execution
    service represents them with the selected encoder over the ``geometry`` map, and its writer
    publishes the run, which references the map and never copies it. Which centers, which support
    and which vector are decided by the capability; no stage consumes the run (#657).
    """

    def __init__(
        self,
        *,
        center_selection: ObservationSupportAnchorPolicy,
        encoder: PointEncoder,
        code_version: str,
        debug_level: PointRepresentationDebugLevel = PointRepresentationDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the center selection policy and the encoder it runs.

        Raises:
            ValueError: If ``code_version`` is empty: the run records the code that produced it.
        """
        self._center_selection = center_selection
        self._encoder = encoder
        self._code_version = _recorded_code_version(code_version)
        self._debug_level = debug_level

    @property
    def configuration_identity(self) -> str:
        """Return the identity of the encoder and the center rule beyond the declared parameters.

        The encoder identity carries its adapter version, its resolved configuration (the device
        included) and its checkpoint hash, and the representation space carries the support
        policy: changing any of them, or the center selection policy, changes the reuse key and
        the run id (#636).
        """
        encoder = self._encoder.encoder_identity()
        document = {
            "center_selection_policy_id": self._center_selection.policy_id,
            "encoder": dataclasses.asdict(encoder),
            "representation_space_id": representation_space_fingerprint(
                self._encoder.representation_space()
            ),
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Represent the anchors of the ``association`` run over ``geometry``; reference the run.

        Raises:
            ExecutorError: If the manifest of the ``association`` run records another
                ``geometric_map_id`` than the ``map_id`` of the ``geometry`` map, whether or not
                the run holds any spatial observation. It is raised before any center is
                selected or any output is written, so nothing is published.
        """
        output = _output(request)
        association = SensorAssociationRunReader(_one(request, "association"))
        run_id = PointRepresentationRunId(
            _configured_stage_identity(request, self.configuration_identity)
        )
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            source = geometry.geometry()
            # A política de centros confere o mapa de cada observação, mas uma associação sem
            # observações não passaria por essa checagem: sem esta, o run publicado citaria a
            # associação de um mapa e registraria outro como o seu.
            associated_map = association.manifest.geometric_map_id
            represented_map = source.geometric_map.map_id
            if associated_map != represented_map:
                raise ExecutorError(
                    f"association run {str(association.manifest.run_id)!r} was built over "
                    f"geometric map {str(associated_map)!r}, which is not the geometric map "
                    f"{str(represented_map)!r} this point representation represents"
                )
            centers = self._center_selection.select_centers(association.observations(), source)
            service = RepresentationService(
                source, self._encoder, run_id=run_id, code_version=self._code_version
            )
            writer = PointRepresentationRunWriter(
                output_dir=output,
                sequence_name=association.manifest.sequence_name,
                run_id=run_id,
                run_index=request.run_number(),
                geometric_map=source.geometric_map,
                space=self._encoder.representation_space(),
                encoder_identity=self._encoder.encoder_identity(),
                code_version=self._code_version,
                association_context_id=str(association.manifest.run_id),
                center_selection_policy_id=self._center_selection.policy_id,
                debug_level=self._debug_level,
            )
            for outcome in service.represent(centers):
                writer.add(outcome)
            manifest = writer.finalize(metrics=service.metrics)
        return _reference(request, REPRESENTATION, str(manifest.run_id), manifest.file_inventory)


class SemanticFusionExecutor:
    """Fuses the association evidence over the persistent geometry, without creating identity.

    The accumulation runs under exactly the policy the executor was built with: the baseline, or
    the quality-aware policy, which also reads the measured quality every association run
    persists for each of its observations. There is no fallback from one to the other.
    """

    def __init__(
        self,
        *,
        support_policy: GeometryOverlapSupportPolicy,
        accumulation_policy: BaselineAccumulationPolicy | QualityAwareAccumulationPolicy,
        code_version: str,
        debug_level: SemanticFusionDebugLevel = SemanticFusionDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the support policy and the accumulation policy it runs.

        Raises:
            ValueError: If ``code_version`` is empty: the run records the code that produced it.
        """
        self._support_policy = support_policy
        self._accumulation_policy = accumulation_policy
        self._code_version = _recorded_code_version(code_version)
        self._debug_level = debug_level

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Fuse the ``association`` runs over the ``geometry`` map and reference the fusion run.

        Several association and perception runs (the ContextRuns of one build, issue #500) are
        fused as separate evidence: Semantic Fusion groups them by physical observation, so
        repeated inference over one frame never counts as another observation. Each association
        run must come from perception runs that are inputs of this fusion, and each perception
        run may be associated once: two associations of one perception run would describe the
        same regions twice.

        Under the quality-aware policy, every association run must have measured its quality
        under the definitions the policy declares, checked before any support is built; the
        quality of each fused observation is read from the association run that produced it.
        """
        output = _output(request)
        associations = [SensorAssociationRunReader(path) for path in _many(request, "association")]
        self._require_quality_definitions(associations)
        region_readers = [RegionPerceptionRunReader(path) for path in _many(request, "regions")]
        runs = tuple(reader.manifest for reader in region_readers)
        region_ids = {str(run.run_id) for run in runs}
        _check_one_association_per_region(associations, region_ids)
        visual_readers = [
            VisualEvidenceRunReader(request.directory_of(ref))
            for ref in request.inputs.get("visual_evidence", ())
        ]
        semantic_readers = [
            SemanticInterpretationRunReader(request.directory_of(ref))
            for ref in request.inputs.get("semantic_interpretation", ())
        ]
        sequence_id = str(SequenceArtifactReader(_one(request, "sequence")).manifest.artifact_id)
        regional_evidence: dict[tuple[str, SourceObservationId], list[Region2D]] = {}
        for regional_reader in region_readers:
            if str(regional_reader.manifest.sequence_artifact_id) != sequence_id:
                raise ExecutorError("regional fusion input belongs to another sequence")
            for source_id in regional_reader.manifest.source_observation_ids:
                regional_evidence.setdefault(
                    (str(regional_reader.manifest.run_id), SourceObservationId(source_id)), []
                )
            for region in regional_reader.read_regions():
                regional_evidence[
                    (
                        str(regional_reader.manifest.run_id),
                        cast(SourceObservationId, region.source_observation_id),
                    )
                ].append(region)
        visuals: dict[tuple[str, SourceObservationId], list[tuple[str, VisualEvidenceResult]]] = {}
        semantics: dict[tuple[str, SourceObservationId], list[tuple[str, SemanticObservation]]] = {}
        producers: list[Any] = [*visual_readers, *semantic_readers]
        for producer in producers:
            if (
                producer.manifest.region_artifact_id not in region_ids
                or str(producer.manifest.sequence_artifact_id) != sequence_id
            ):
                raise ExecutorError(
                    "fusion producer references an unselected regional artifact or sequence"
                )
        for visual_reader in visual_readers:
            for item in visual_reader.iter_results():
                key = (
                    visual_reader.manifest.region_artifact_id,
                    item.source_observation_id,
                )
                if key not in regional_evidence:
                    raise ExecutorError(
                        "visual evidence references an unselected regional observation"
                    )
                visuals.setdefault(key, []).append((str(visual_reader.manifest.run_id), item))
        for semantic_reader in semantic_readers:
            for semantic_item in semantic_reader.iter_observations():
                key = (
                    semantic_reader.manifest.region_artifact_id,
                    semantic_item.source_observation_id,
                )
                if key not in regional_evidence:
                    raise ExecutorError(
                        "semantic evidence references an unselected regional observation"
                    )
                semantics.setdefault(key, []).append(
                    (str(semantic_reader.manifest.run_id), semantic_item)
                )
        visual_ids = {str(reader.manifest.run_id) for reader in visual_readers}
        semantic_ids = {str(reader.manifest.run_id) for reader in semantic_readers}
        raw_scores, scoring_run_ids = self._semantic_scores(
            request, region_ids, visual_ids, semantic_ids
        )
        results: dict[tuple[str, SourceObservationId], list[PerceptionResult]] = {}
        for key, regions in regional_evidence.items():
            for visual_id, visual in visuals.get(key, [(None, None)]):
                for semantic_id, semantic in semantics.get(key, [(None, None)]):
                    results.setdefault(key, []).append(
                        PerceptionResult(
                            source_observation_id=key[1],
                            sequence_artifact_id=sequence_id,
                            region_artifact_id=key[0],
                            visual_evidence_artifact_id=visual_id,
                            semantic_interpretation_artifact_id=semantic_id,
                            regions=tuple(regions),
                            features=() if visual is None else visual.features,
                            claims=() if semantic is None else semantic.claims,
                            scene_context=None if semantic is None else semantic.scene_context,
                            semantic_scoring_artifact_ids=tuple(
                                str(item) for item in scoring_run_ids
                            ),
                        )
                    )
        observations: list[SpatialObservation] = []
        association_of: dict[SpatialObservationId, SensorAssociationRunReader] = {}
        for association in associations:
            for observation in association.observations():
                observations.append(observation)
                association_of[observation.spatial_observation_id] = association
        # Pelo índice, não por list_observations(): esta fusão só precisa do timestamp de cada
        # imagem, e decodificar a sequência inteira para isso custava os 2,6 GB de payload do
        # corridor-02 (#514). O índice já carrega identidade, timestamp e modalidade.
        timestamps = {
            entry.observation_id: entry.timestamp
            for entry in SequenceArtifactReader(_one(request, "sequence")).iter_index()
            if entry.modality == "image"
        }
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            source = geometry.geometry()
            grouping = group_by_physical_observation(
                observations, selected_region_runs=runs, acquisition_timestamps=timestamps
            )
            build = build_fusion_supports(
                observations,
                geometry=source,
                acquisition_timestamps=timestamps,
                policy=self._support_policy,
                code_version=self._code_version,
            )
            by_id = {item.spatial_observation_id: item for item in observations}
            policy = self._accumulation_policy
            # O despacho é pelo tipo da política selecionada, nunca por um nome de backend, e não
            # há fallback: uma política que nenhum ramo conhece é recusada.
            if isinstance(policy, QualityAwareAccumulationPolicy):
                # Só a qualidade das observações que entram em algum suporte é lida, e sempre do
                # run de associação que produziu a observação.
                qualities = {
                    observation_id: association_of[observation_id].quality(observation_id)
                    for support in build.supports
                    for observation_id in support.spatial_observation_ids
                }
                evidence = [
                    accumulate_quality_aware_evidence(
                        support,
                        observations=by_id,
                        grouping=grouping,
                        perception_results=results,
                        observation_qualities=qualities,
                        policy=policy,
                        raw_semantic_scores=raw_scores,
                        code_version=self._code_version,
                    )
                    for support in build.supports
                ]
            elif isinstance(policy, BaselineAccumulationPolicy):
                evidence = [
                    accumulate_baseline_evidence(
                        support,
                        observations=by_id,
                        grouping=grouping,
                        perception_results=results,
                        policy=policy,
                        raw_semantic_scores=raw_scores,
                        code_version=self._code_version,
                    )
                    for support in build.supports
                ]
            else:
                raise ExecutorError(f"no accumulation runs policy {type(policy).__name__}")
            outcomes = tuple(
                FusionOutcome(support=support, evidence=fused)
                for support, fused in zip(build.supports, evidence, strict=True)
            )
            first = associations[0].manifest
            manifest = SemanticFusionRunWriter(
                output_dir=output,
                sequence_name=first.sequence_name,
                run_id=SemanticFusionRunId(request.identity()),
                run_index=request.run_number(),
                lineage=FusionRunLineage(
                    sequence_artifact_id=str(first.sequence_artifact_id),
                    geometric_map_id=geometry.manifest.map_id,
                    association_run_ids=tuple(
                        sorted(str(association.manifest.run_id) for association in associations)
                    ),
                    region_artifact_ids=tuple(sorted(region_ids)),
                    visual_artifact_ids=tuple(sorted(visual_ids)),
                    semantic_interpretation_artifact_ids=tuple(sorted(semantic_ids)),
                    # A fusão não consome runs de Point Representation (#657).
                    point_representation_run_ids=(),
                    semantic_scoring_run_ids=scoring_run_ids,
                ),
                code_version=self._code_version,
                debug_level=self._debug_level,
            ).write(outcomes, excluded=build.excluded)
        return _reference(request, FUSION, str(manifest.run_id), manifest.file_inventory)

    def _semantic_scores(
        self,
        request: StageRequest,
        regional_ids: set[str],
        visual_ids: set[str],
        semantic_ids: set[str],
    ) -> tuple[
        dict[str, tuple[tuple[str, SemanticScore], ...]] | None, tuple[SemanticScoringRunId, ...]
    ]:
        """Validate scoring producers and preserve their native scores by semantic producer."""
        scores: dict[str, list[tuple[str, SemanticScore]]] = {}
        run_ids: set[SemanticScoringRunId] = set()
        for ref in request.inputs.get("semantic_scores", ()):
            reader = SemanticScoringRunReader(request.directory_of(ref))
            lineage = reader.manifest.lineage
            if (
                lineage.region_artifact_id not in regional_ids
                or lineage.visual_evidence_artifact_id not in visual_ids
                or lineage.semantic_interpretation_artifact_id not in semantic_ids
            ):
                raise ExecutorError(
                    "semantic scoring run references an unselected evidence producer"
                )
            run_id = SemanticScoringRunId(str(reader.manifest.run_id))
            run_ids.add(run_id)
            scores.setdefault(lineage.semantic_interpretation_artifact_id, []).extend(
                (str(run_id), score) for score in reader.iter_scores()
            )
        ordered = {
            key: tuple(sorted(items, key=lambda item: (item[0], str(item[1].score_id))))
            for key, items in scores.items()
        }
        return ordered or None, tuple(sorted(run_ids))

    def _require_quality_definitions(
        self, associations: Sequence[SensorAssociationRunReader]
    ) -> None:
        """Refuse, before any support is built, quality the selected policy cannot read.

        Every association run records the quality definitions it measured under; the
        quality-aware policy declares the one its ramps are defined over. The comparison is the
        policy's own rule, applied here to each input run as a whole, so a mismatch never
        surfaces midway through the accumulation. The baseline reads no quality.

        Raises:
            ExecutorError: If an association run measured quality under other definitions.
        """
        policy = self._accumulation_policy
        if not isinstance(policy, QualityAwareAccumulationPolicy):
            return
        for association in associations:
            try:
                policy.require_definitions(
                    association.manifest.definitions["quality"],
                    source=f"association run {str(association.manifest.run_id)!r}",
                )
            except ValueError as error:
                raise ExecutorError(str(error)) from error


def _check_one_association_per_region(
    associations: Sequence[SensorAssociationRunReader], regional_runs: set[str]
) -> None:
    """Reject alternative associations of the same regional evidence before fusion."""
    associated: dict[str, str] = {}
    for association in associations:
        association_run = str(association.manifest.run_id)
        for region_run in association.manifest.region_artifact_ids:
            if region_run not in regional_runs:
                raise ExecutorError(
                    f"association {association_run!r} references unselected region {region_run!r}"
                )
            other = associated.setdefault(region_run, association_run)
            if other != association_run:
                raise ExecutorError(
                    f"regional artifact {region_run!r} is associated by both "
                    f"{other!r} and {association_run!r}; "
                    "a fusion takes one association per regional artifact"
                )


class SemanticMappingExecutor:
    """Materializes the fused evidence as persistent entities, without merging supports."""

    def __init__(
        self,
        *,
        policy: EntityMaterializationPolicy,
        semantic_map_id: SemanticMapId,
        code_digest: str,
        code_version: str,
        debug_level: MappingDebugLevel = MappingDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the materialization policy and the semantic map it fills.

        Args:
            policy: The materialization policy.
            semantic_map_id: The semantic map the run fills.
            code_digest: Digest of the code producing the run; ``SemanticMappingRunWriter``
                refuses an empty one.
            code_version: Code revision that produced the run.
            debug_level: Amount of non-contractual debug evidence the run persists, from
                ``policies.debug_level``. ``NONE`` (default) preserves the prior behavior.

        Raises:
            ValueError: If ``code_version`` is empty: the run records the code that produced it.
        """
        self._policy = policy
        self._semantic_map_id = semantic_map_id
        self._code_digest = code_digest
        self._code_version = _recorded_code_version(code_version)
        self._debug_level = debug_level

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Materialize the ``fusion`` run over its geometry and reference the entity run."""
        output = _output(request)
        fusion = SemanticFusionRunReader(_one(request, "fusion"))
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            materialization = materialize_entities(
                fusion.iter_outcomes(),
                fusion_manifest=fusion.manifest,
                geometry=geometry.geometry(),
                semantic_map_id=self._semantic_map_id,
                policy=self._policy,
                code_version=self._code_version,
            )
        manifest = SemanticMappingRunWriter(
            output_dir=output,
            sequence_name=fusion.manifest.sequence_name,
            run_id=SemanticMappingRunId(request.identity()),
            run_index=request.run_number(),
            semantic_map_id=self._semantic_map_id,
            lineage=lineage_from_fusion_manifest(fusion.manifest),
            code_version=self._code_version,
            code_digest=self._code_digest,
            debug_level=self._debug_level,
        ).write(materialization.entities, rejections=materialization.rejections)
        return _reference(request, ENTITIES, str(manifest.run_id), manifest.file_inventory)


class _DeferredFeaturePayloadSink:
    """Forwards feature payloads and their diagnostics to a writer that does not exist yet.

    ``VisualEvidenceRunWriter`` needs ``configuration_digest``
    (:meth:`~contextmap.visual_perception.execution.pipeline_resolution.ResolvedPipeline.configuration_digest`)
    at construction time, but that digest can only be computed *after*
    :func:`~contextmap.visual_perception.resolve_pipeline` has built every
    backend — including the feature extractors, which need a payload sink
    already at their own construction time. This breaks that ordering
    cycle: backends are given this forwarding sink immediately, and it is
    bound to the real writer once one exists, before any observation is
    processed and therefore before any payload is actually queued.
    """

    def __init__(self) -> None:
        """Create an unbound sink."""
        self._writer: VisualEvidenceRunWriter | None = None
        self._current_arrays: dict[tuple[str, str], object] = {}
        self._persisted_features: set[tuple[str, str]] = set()
        self._current_observation: str | None = None
        self._current_diagnostics: dict[tuple[str, str], FeatureExtractionDiagnostic] = {}

    def bind(self, writer: VisualEvidenceRunWriter) -> None:
        """Bind every subsequent call to the real writer."""
        self._writer = writer

    def add_feature_payload(
        self, feature: object, source_observation_id: object, array: object
    ) -> None:
        """Forward one payload to the bound writer.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("feature payload queued before the run writer was bound")
        self._writer.add_feature_payload(feature, source_observation_id, array)  # type: ignore[arg-type]
        feature_id = getattr(feature, "feature_id", None)
        if feature_id is not None:
            observation = str(source_observation_id)
            if observation != self._current_observation:
                self._current_arrays.clear()
                self._current_observation = observation
            key = (observation, str(feature_id))
            self._persisted_features.add(key)
            self._current_arrays[key] = array

    def current_payload(self, source_observation_id: object, feature_id: object) -> Any:
        """Return one current-frame payload for an in-transaction derived feature.

        The cache is intentionally bounded to the caller's current frame: it permits a
        mask-weighted DINO region feature to reuse the dense array already produced by the
        backend, without rereading a staging artifact or running DINO again.
        """
        key = (str(source_observation_id), str(feature_id))
        if key in self._current_arrays:
            return self._current_arrays[key]
        if key not in self._persisted_features:
            raise ExecutorError(f"no current payload for feature {feature_id!r}")
        return self.writer().staged_feature_payload(SourceObservationId(key[0]), FeatureId(key[1]))

    def has_current_dense_evidence(self, source_observation_id: object, feature_id: object) -> bool:
        """Whether a backend persisted both payload and diagnostic for the current feature."""
        key = (str(source_observation_id), str(feature_id))
        return key in self._persisted_features and key in self._current_diagnostics

    def discard_for(self, source_observation_id: object) -> None:
        """Release the temporary arrays and diagnostics of one observation.

        Called once the last stage that reads that frame's dense payload is done, so a
        stage-major run does not keep one dense map per frame alive until the run ends.
        Idempotent: discarding an observation with nothing cached is a no-op.
        """
        observation = str(source_observation_id)
        self._persisted_features.difference_update(
            key for key in tuple(self._persisted_features) if key[0] == observation
        )
        for cache in (self._current_arrays, self._current_diagnostics):
            for key in [key for key in cache if key[0] == observation]:
                del cache[key]

    def add_feature_diagnostic(self, diagnostic: object) -> None:
        """Forward the diagnostic of one produced feature to the bound writer.

        A dense extractor sends, with each payload, the sampling geometry it computed; the
        writer persists it in ``metrics/feature-extraction.jsonl``, which is what reopens the
        payload as a spatial map (#651).

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("feature diagnostic queued before the run writer was bound")
        if not isinstance(diagnostic, FeatureExtractionDiagnostic):
            raise ExecutorError("feature diagnostic must use the public diagnostic contract")
        self._writer.add_feature_diagnostic(diagnostic)
        if diagnostic.feature_id is not None:
            self._current_diagnostics[
                (str(diagnostic.source_observation_id), str(diagnostic.feature_id))
            ] = diagnostic

    def current_diagnostic(
        self, source_observation_id: object, feature_id: object
    ) -> FeatureExtractionDiagnostic:
        """Return the current-frame diagnostic belonging to one persisted feature payload."""
        key = (str(source_observation_id), str(feature_id))
        try:
            return self._current_diagnostics[key]
        except KeyError as error:
            raise ExecutorError(f"no current diagnostic for feature {feature_id!r}") from error

    def writer(self) -> VisualEvidenceRunWriter:
        """Return the writer after the executor has bound the transaction."""
        if self._writer is None:
            raise ExecutorError("dense region pooling ran before the run writer was bound")
        return self._writer

    def add_ineligible_region(self, record: IneligibleRegion) -> None:
        """Forward one region a mask-conditioned extractor skipped (#650) to the bound writer.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("ineligible region reported before the run writer was bound")
        self._writer.add_ineligible_region(record)


def _append_mask_pooled_dense_features(
    *,
    result: PerceptionResult,
    prepared: PreparedImage,
    payload_sink: _DeferredFeaturePayloadSink,
    writer: VisualEvidenceRunWriter,
    run_id: str,
    mask_resolver: RegionMaskResolver,
) -> PerceptionResult:
    """Add one mask-weighted region vector for every dense map and masked region.

    The dense backend already computed and queued its array. This adapter only reuses that
    current-frame evidence with the backend-recorded sampling geometry; it never invokes a
    second model or invents a score. Box-only regions are intentionally excluded: this channel
    is specifically mask-conditioned appearance evidence.
    """
    pooled_features: list[VisualFeature] = []
    for region in (
        item for item in result.regions if item.mask is None and item.mask_reference is None
    ):
        writer.add_ineligible_region(
            IneligibleRegion(
                source_observation_id=result.source_observation_id,
                region_id=region.region_id,
                stage_id="dense_region_pooling",
                reason=IneligibleRegionReason.NO_MASK,
                detail="mask-weighted dense region pooling requires an inline region mask",
            )
        )
    for dense in (item for item in result.features if item.scope is FeatureScope.DENSE):
        # A generic FeatureExtractor is allowed to publish metadata-only dense evidence. Such a
        # feature remains valid evidence but cannot be pooled without its numerical payload and
        # sampling geometry; real DINO/SigLIP adapters publish both through the sink.
        masked_regions = tuple(
            region
            for region in result.regions
            if region.mask is not None or region.mask_reference is not None
        )
        if not payload_sink.has_current_dense_evidence(
            result.source_observation_id, dense.feature_id
        ):
            for region in masked_regions:
                writer.add_ineligible_region(
                    IneligibleRegion(
                        source_observation_id=result.source_observation_id,
                        region_id=region.region_id,
                        stage_id="dense_region_pooling",
                        reason=IneligibleRegionReason.POOLING_UNAVAILABLE,
                        detail=(
                            f"dense feature {dense.feature_id!r} has no current payload and "
                            "sampling diagnostic"
                        ),
                        source_feature_id=dense.feature_id,
                    )
                )
            continue
        diagnostic = payload_sink.current_diagnostic(result.source_observation_id, dense.feature_id)
        if diagnostic.status is not FeatureEventStatus.SUCCEEDED or diagnostic.dense is None:
            raise ExecutorError(
                f"dense feature {dense.feature_id!r} has no successful dense extraction diagnostic"
            )
        if diagnostic.embedding_space is None:
            raise ExecutorError(
                f"dense feature {dense.feature_id!r} has no embedding-space diagnostic"
            )
        dense_map = dense_feature_map_of(
            dense,
            diagnostic.dense,
            source_image_width=prepared.width,
            source_image_height=prepared.height,
        )
        array = payload_sink.current_payload(result.source_observation_id, dense.feature_id)
        for index, region in enumerate(masked_regions):
            mask = _RegionMaskSource(mask_resolver).load_box_local_mask(region)
            try:
                pooled = pool_region_feature(array, dense_map=dense_map, region=region, mask=mask)
            except RegionAssociationError as error:
                writer.add_ineligible_region(
                    IneligibleRegion(
                        source_observation_id=result.source_observation_id,
                        region_id=region.region_id,
                        stage_id="dense_region_pooling",
                        reason=IneligibleRegionReason.POOLING_UNAVAILABLE,
                        detail=str(error),
                        source_feature_id=dense.feature_id,
                    )
                )
                continue
            feature_id = feature_id_for(
                run_id=run_id,
                source_observation_id=result.source_observation_id,
                producer_id=f"dense-region-pooling/{dense.feature_id}",
                index=index,
            )
            feature = VisualFeature(
                feature_id=feature_id,
                source_observation_id=result.source_observation_id,
                scope=FeatureScope.REGION,
                embedding_space_id=pooled.embedding_space_id,
                shape=tuple(pooled.vector.shape),
                dtype=pooled.dtype,
                normalization=pooled.normalization,
                payload_reference=f"dense-region-pooling/{feature_id}.npy",
                provenance=pooled.provenance.backend,
                region_id=region.region_id,
            )
            writer.add_feature_payload(feature, result.source_observation_id, pooled.vector)
            writer.add_feature_diagnostic(
                FeatureExtractionDiagnostic(
                    event_id=str(feature.feature_id),
                    source_observation_id=result.source_observation_id,
                    source_prepared_image_reference=prepared.payload_reference,
                    source_image_width=prepared.width,
                    source_image_height=prepared.height,
                    stage_id="dense_region_pooling",
                    status=FeatureEventStatus.SUCCEEDED,
                    backend=pooled.provenance.backend,
                    feature_id=feature.feature_id,
                    scope=feature.scope,
                    embedding_space=diagnostic.embedding_space,
                    output_shape=feature.shape,
                    dtype=feature.dtype,
                    normalization=feature.normalization,
                    payload_reference=feature.payload_reference,
                    region=RegionFeatureDiagnostic(
                        region_id=region.region_id,
                        bounding_box=region.bounding_box,
                        source_mask_reference=region.mask_reference,
                        mask_content_hash=pooled.provenance.mask_content_hash,
                        coordinate_transform_id=pooled.provenance.coordinate_transform_id,
                        pooling_policy=pooled.provenance.pooling_policy,
                        contributing_cell_count=pooled.diagnostics.contributing_cell_count,
                        total_weight=pooled.diagnostics.total_weight,
                        coverage_fraction=pooled.diagnostics.coverage_fraction,
                        source_feature_id=pooled.provenance.source_feature_id,
                        source_payload_reference=pooled.provenance.source_payload_reference,
                        source_artifact_id=pooled.provenance.source_artifact_id,
                    ),
                )
            )
            pooled_features.append(feature)
    return replace(result, features=(*result.features, *pooled_features))


class _DenseRegionPoolingStage:
    """Pipeline backend that publishes pooled appearance evidence inside the stage graph."""

    def __init__(
        self,
        *,
        run_id: str,
        payload_sink: _DeferredFeaturePayloadSink,
        mask_resolver: RegionMaskResolver,
    ) -> None:
        self._run_id = run_id
        self._payload_sink = payload_sink
        self._mask_resolver = mask_resolver

    def backend_provenance(self) -> BackendProvenance:
        # A mesma identidade que _pooling_provenance() põe em cada VisualFeature pooled, para que
        # a feature seja lida como produzida pelo backend que o manifest registra (#688).
        return pooling_backend_provenance()

    def pool(
        self,
        image: PreparedImage,
        regions: Sequence[Region2D],
        dense_features: Sequence[VisualFeature],
    ) -> Sequence[VisualFeature]:
        result = PerceptionResult(
            source_observation_id=image.source_observation_id,
            sequence_artifact_id="stage-local",
            region_artifact_id="stage-regions",
            visual_evidence_artifact_id=self._run_id,
            regions=regions,
            features=dense_features,
        )
        pooled = _append_mask_pooled_dense_features(
            result=result,
            prepared=image,
            payload_sink=self._payload_sink,
            writer=self._payload_sink.writer(),
            run_id=self._run_id,
            mask_resolver=self._mask_resolver,
        )
        return pooled.features[len(dense_features) :]


class _RegionPublishingWriter(Protocol):
    """The two writer operations a frame's discovered regions and their audit reach.

    Both the region run writer and the perception run writer serve it, with the same meaning:
    the audit is persisted as the frame's evidence, and each region's mask pixels leave RAM
    for the run's staging store, the region keeping only ``mask_reference``. That is what lets
    one wrapper publish the audit of a discovery whichever of the two stages is executing it.
    """

    def add_region_discovery_audit(self, audit: RegionDiscoveryAudit) -> None:
        """Persist how one frame's canonical regions were decided."""
        ...

    def externalize_region_mask(
        self, source_observation_id: SourceObservationId, region: Region2D
    ) -> Region2D:
        """Stage one region's inline mask and return it carrying only its reference."""
        ...


class _AuditPublishingRegionDiscovery:
    """Serves the plain ``RegionDiscovery`` port to the stage graph and publishes each audit.

    ``CANONICAL_PRESET_V1``'s ``region_discovery`` stage calls ``discover()`` and hands its regions
    to the stages that depend on them. This asks the composed backend for
    ``discover_audited()`` instead, gives the frame's
    :class:`~contextmap.visual_perception.RegionDiscoveryAudit` to the writer and returns exactly
    the regions ``discover()`` would have. What the audit contains is decided in
    :mod:`contextmap.visual_perception`; this only routes it, so rejected candidates and merge
    decisions reach the artifact instead of dying with the stage output.

    It is also where the frame's region masks leave RAM: each region is handed to the writer's
    staging mask store (``externalize_region_mask``) and continues through the graph carrying
    only ``mask_reference``. This is infrastructure at the producer boundary, not a scientific
    stage; the scheduler stays unaware of masks.
    """

    def __init__(self, discovery: AuditedRegionDiscovery, run_id: str) -> None:
        """Wrap the composed backend; nothing is published until :meth:`bind`."""
        self._discovery = discovery
        self._run_id = run_id
        self._writer: _RegionPublishingWriter | None = None

    def bind(self, writer: _RegionPublishingWriter) -> None:
        """Bind every subsequent audit to the real writer.

        Same ordering cycle as :class:`_DeferredFeaturePayloadSink`: this has to exist before
        ``resolve_pipeline()`` builds the stage graph, but the writer needs that graph's
        ``configuration_digest``. Binding happens before any image is processed. The region
        stage has no such cycle and binds its own writer right away.
        """
        self._writer = writer

    def backend_provenance(self) -> BackendProvenance:
        """Pass the wrapped backend's provenance through: the configuration digest is unchanged."""
        return self._discovery.backend_provenance()

    def discovery_identity(self) -> RegionDiscoveryIdentity:
        """Pass the wrapped backend's effective discovery identity through, for the same reason."""
        return self._discovery.discovery_identity()

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover one frame's regions and hand its audit to the writer before returning them.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("region discovery used before bind(): no writer for its audit")
        audited = self._discovery.discover_audited(image)
        named = namespace_regions(
            run_id=self._run_id,
            source_observation_id=image.source_observation_id,
            regions=audited.regions,
            audit=audited.audit,
        )
        if named.audit is not None:
            self._writer.add_region_discovery_audit(named.audit)
        # Fronteira do produtor: a máscara vai para o staging agora e a região segue só com a
        # referência, então os pixels não esperam o último estágio do último frame (F02b).
        return tuple(
            self._writer.externalize_region_mask(image.source_observation_id, region)
            for region in named.regions
        )

    def release(self) -> None:
        """Release the wrapped backend's model, when it can release one."""
        _release(self._discovery)


_REGION_INTERPRETATION_STAGE = "region_interpretation"
"""The stage every preset the executor runs serves through ``interpret_regions``."""


def _materialize_prepared_image(
    image: ImageObservation,
    root: Path,
    image_masks: Mapping[str, CameraImageMasks] | None = None,
) -> PreparedImage:
    """Decode one raw observation into an RGB/L PNG file a real backend can open.

    Ingestion's :class:`~contextmap.ingestion.ImageObservation` carries a raw,
    uncompressed pixel buffer; every real Visual Perception backend (SAM2,
    SAM3, DINOv2/v3, CLIP, AlphaCLIP) decodes ``PreparedImage.payload_reference``
    as an actual image file through Pillow. This is the one format
    conversion this executor performs on its own: no resize, crop or
    rectification decision is made — exactly the identity transform
    :func:`~contextmap.visual_perception.prepare_image` already models with
    ``operations=()``, the same shape ``SensorAssociationExecutor._frame``
    already uses for its own (non-decodable) placeholder reference.

    Args:
        image: The raw observation to materialize.
        root: Scratch directory prepared images are written under; never
            part of the published ``PerceptionRunArtifact``.
        image_masks: Static per-camera valid/exclusion regions, keyed by camera
            calibration id. ``None`` or empty keeps the whole image usable.

    Returns:
        A ``PreparedImage`` whose reference resolves, under ``root``, to a
        real, hash-verifiable PNG file.

    Raises:
        ExecutorError: If the observation's encoding is not supported.
    """
    import importlib

    import numpy as np

    image_module = importlib.import_module("PIL.Image")

    if image.encoding is ImageEncoding.RGB8:
        rgb = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        pil_image = image_module.fromarray(rgb, mode="RGB")
    elif image.encoding is ImageEncoding.BGR8:
        bgr = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        pil_image = image_module.fromarray(bgr[:, :, ::-1], mode="RGB")
    elif image.encoding is ImageEncoding.MONO8:
        mono8 = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width)
        pil_image = image_module.fromarray(mono8, mode="L")
    elif image.encoding is ImageEncoding.MONO16:
        mono16 = np.frombuffer(image.data, dtype=np.uint16).reshape(image.height, image.width)
        pil_image = image_module.fromarray(mono16, mode="I;16")
    else:
        raise ExecutorError(
            f"unsupported image encoding for prepared-image materialization: {image.encoding!r}"
        )

    filename = f"{image.observation_id}.png"
    output_path = root / filename
    pil_image.save(output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    valid_region, exclusion_regions = _preparation_constraints(
        image_masks or {}, image.calibration_id, image.width, image.height
    )
    return prepare_image(
        SourceImage(
            source_observation_id=str(image.observation_id),
            image=ArtifactReference(uri=filename, sha256=digest, media_type="image/png"),
            width=image.width,
            height=image.height,
        ),
        operations=(),
        valid_region=valid_region,
        exclusion_regions=exclusion_regions,
    )


class _RegionMaskSource:
    """Crops each region's mask to its box-local raster envelope, wherever its pixels live.

    A mask-conditioned region-features backend (AlphaCLIP) is conditioned on the region's
    mask, not only its crop, and asks for it box-local (``box_mask_shape``), not full-image
    (``RegionMaskSource`` in ``visual_perception.visual_features.adapters.alphaclip``). A
    mask-based Region Discovery backend (SAM2, SAM3) attaches it to each ``Region2D`` at full
    resolution; the executor externalizes it at the producer boundary, so the region the stage
    receives only carries ``mask_reference`` and the pixels are read from the run's staging
    store here (:func:`~contextmap.visual_perception.region_mask_pixels`, hash-verified, not
    cached). A region that still carries its inline mask is cropped directly. No new evidence is
    produced and no threshold applied: it is the exact inverse of how the backend itself
    re-expands a box-local mask back into full-image coordinates.

    Raises:
        ValueError: If ``region`` carries neither an inline mask nor a ``mask_reference``.
            AlphaCLIP records a box-only region (a grounded box) as ineligible before asking
            for its mask (#650), so reaching this means a box-only region entered the stage.
    """

    def __init__(self, resolver: RegionMaskResolver | None) -> None:
        """Bind the source to where this run's ``mask_reference`` values are resolved."""
        self._resolver = resolver

    def load_box_local_mask(self, region: Region2D) -> Any:
        """Return ``region``'s own mask, cropped to ``region.bounding_box``'s raster envelope."""
        import numpy as np

        mask = region_mask_pixels(region, self._resolver)
        if mask is None:
            raise ValueError(
                f"region {region.region_id!r} has no mask: a mask-conditioned region "
                "features backend needs a mask-based region discovery backend"
            )
        full = mask.as_array()
        height, width = box_mask_shape(region.bounding_box)
        top = math.floor(region.bounding_box.y)
        left = math.floor(region.bounding_box.x)
        cropped = np.zeros((height, width), dtype=np.bool_)
        source_top, source_left = max(0, top), max(0, left)
        source_bottom = min(full.shape[0], top + height)
        source_right = min(full.shape[1], left + width)
        copy_height, copy_width = source_bottom - source_top, source_right - source_left
        if copy_height > 0 and copy_width > 0:
            cropped[
                source_top - top : source_top - top + copy_height,
                source_left - left : source_left - left + copy_width,
            ] = full[source_top:source_bottom, source_left:source_right]
        return cropped


class _QueryGroundingStage:
    """Serves a grounded preset's ``region_grounding`` stage: every configured query, per image.

    Each (image, query) pair becomes one explicit
    :class:`~contextmap.visual_perception.RegionGroundingRequest`, owned by the image's result in
    this run and issued to the configuration the backend reports. The stage returns the
    executions in query order; the executor persists them from the stage's outcome, so a failed
    grounding call fails only this frame's ``region_grounding`` stage (#650).
    """

    def __init__(
        self,
        grounding: RegionGrounding,
        queries: Sequence[GroundingQuery],
        run_id: str,
    ) -> None:
        """Bind the run's grounding backend to its ordered queries and to the run.

        Raises:
            ExecutorError: If the backend reports no configuration fingerprint: its requests
                could not be tied to the configuration that served them.
        """
        fingerprint = grounding.backend_provenance().configuration_fingerprint
        if not fingerprint:
            raise ExecutorError(
                "the region grounding backend reports no configuration fingerprint; its "
                "requests could not be tied to the configuration that served them"
            )
        self._grounding = grounding
        self._queries = tuple(queries)
        self._run_id = run_id
        self._fingerprint = fingerprint

    def backend_provenance(self) -> BackendProvenance:
        """Pass the grounding backend's provenance through, into the configuration digest."""
        return self._grounding.backend_provenance()

    def ground_image(self, image: PreparedImage) -> tuple[RegionGroundingExecution, ...]:
        """Ask one image every configured query, each as its own explicit request."""
        return tuple(
            self._grounding.ground(
                RegionGroundingRequest(
                    region_perception_run_id=self._run_id,
                    image=image,
                    query=query,
                    configuration_fingerprint=self._fingerprint,
                )
            )
            for query in self._queries
        )


class _ProposalRefinementStage:
    """Serves the ``region_refinement`` stage: one request per image with all its proposals.

    An image whose grounding answered no geometry has nothing to refine, so no request is
    issued for it; every other image gets exactly one request (the image is encoded once).
    """

    def __init__(self, refiner: RegionRefinement, run_id: str) -> None:
        """Bind the run's refiner to the run.

        Raises:
            ExecutorError: If the refiner reports no configuration fingerprint.
        """
        fingerprint = refiner.backend_provenance().configuration_fingerprint
        if not fingerprint:
            raise ExecutorError(
                "the region refinement backend reports no configuration fingerprint; its "
                "requests could not be tied to the configuration that served them"
            )
        self._refiner = refiner
        self._run_id = run_id
        self._fingerprint = fingerprint
        self._writer: RegionPerceptionRunWriter | None = None

    def bind(self, writer: RegionPerceptionRunWriter) -> None:
        """Bind the stage to the writer whose staging store receives the refined masks."""
        self._writer = writer

    def backend_provenance(self) -> BackendProvenance:
        """Pass the refiner's provenance through, into the configuration digest."""
        return self._refiner.backend_provenance()

    def refine_groundings(
        self, image: PreparedImage, groundings: Sequence[RegionGroundingExecution]
    ) -> tuple[RegionRefinementExecution, ...]:
        """Refine every grounding proposal of one image in one request; none without proposals."""
        prompts = tuple(prompt for item in groundings for prompt in refinement_prompts_from(item))
        if not prompts:
            return ()
        request = RegionRefinementRequest(
            region_perception_run_id=self._run_id,
            image=image,
            prompts=prompts,
            configuration_fingerprint=self._fingerprint,
        )
        return (self._externalized(image, self._refiner.refine(request)),)

    def _externalized(
        self, image: PreparedImage, execution: RegionRefinementExecution
    ) -> RegionRefinementExecution:
        """Stage every refined mask now, as discovery does, and keep only its reference.

        Raises:
            ExecutorError: If called before :meth:`bind`.
        """
        if self._writer is None:
            raise ExecutorError("region refinement used before bind(): no writer for its masks")
        writer = self._writer
        return replace(
            execution,
            outcomes=tuple(
                outcome
                if outcome.region is None
                else replace(
                    outcome,
                    region=writer.externalize_region_mask(
                        image.source_observation_id, outcome.region
                    ),
                )
                for outcome in execution.outcomes
            ),
        )


def _succeeded_output(outcomes: Sequence[StageOutcome], stage_id: str) -> Any:
    """Return what ``stage_id`` produced for this frame, or ``()`` if it did not succeed."""
    for outcome in outcomes:
        if outcome.stage_id == stage_id and outcome.status is StageStatus.SUCCEEDED:
            return outcome.output
    return ()


class PerceptionSchedule(Enum):
    """The order in which a perception run executes its stages over its frames.

    Attributes:
        FRAME_MAJOR: Every stage of a frame, then the next frame; every model stays loaded for
            the whole run. The default.
        STAGE_MAJOR: One stage over every frame, then the next stage, releasing each backend
            once its last stage is done, so a GPU too small for all models at once runs the
            run (#689). It writes the same run as ``FRAME_MAJOR``; it holds every frame's stage
            outputs until the last stage.
    """

    FRAME_MAJOR = "frame_major"
    STAGE_MAJOR = "stage_major"


def _release(backend: object) -> None:
    """Release a backend's model resources when the backend can release them."""
    if isinstance(backend, ReleasableBackend):
        backend.release()


def _release_once(backend: object, released: set[int]) -> None:
    """Release a constructed backend at most once, including failure cleanup."""
    if id(backend) not in released:
        _release(backend)
        released.add(id(backend))


def _release_points(
    stage_ids: Sequence[str], backends: Mapping[str, object]
) -> dict[str, tuple[object, ...]]:
    """Map each stage to the backends whose last stage it is, so each is released once."""
    last: dict[int, str] = {}
    objects: dict[int, object] = {}
    for stage_id in stage_ids:
        backend = backends.get(stage_id)
        if backend is not None:
            last[id(backend)] = stage_id
            objects[id(backend)] = backend
    points: dict[str, list[object]] = {}
    for key, stage_id in last.items():
        points.setdefault(stage_id, []).append(objects[key])
    return {stage_id: tuple(items) for stage_id, items in points.items()}


_IDENTITY_IMAGE_PREPARATION_POLICY = "image-preparation/identity-v1"
"""Identity of the preparation the perception stages apply before any backend sees a frame.

It is the identity transform (:func:`~contextmap.visual_perception.prepare_image` with
``operations=()``), the raw observation only decoded into a file a backend can open, plus the
configured static per-camera valid and exclusion regions. No resize, crop or rectification
decision is made, so the policy is versioned rather than fingerprinted per run; a change of
what the preparation does changes this identity.
"""


class RegionPerceptionExecutor:
    """Publish final regional evidence, including optional grounding and refinement."""

    def __init__(
        self,
        *,
        region_discovery: RegionDiscoveryPlan,
        region_grounding: RegionGroundingPlan | None = None,
        region_refinement: RegionRefinementPlan | None = None,
        image_masks: Mapping[str, CameraImageMasks] | None = None,
        code_version: str | None = None,
        perception_schedule: PerceptionSchedule = PerceptionSchedule.FRAME_MAJOR,
    ) -> None:
        """Bind all regional producers and their execution policy to this artifact owner."""
        if region_refinement is not None and region_grounding is None:
            raise ValueError("region refinement requires grounding proposals")
        self._region_discovery = region_discovery
        self._region_grounding = region_grounding
        self._region_refinement = region_refinement
        self._image_masks = image_masks or {}
        self._code_version = code_version
        self._perception_schedule = perception_schedule

    @property
    def required_modules(self) -> Mapping[str, str]:
        """Declare raw-image materialization dependencies for preflight."""
        return {"PIL": "region perception requires Pillow; install contextmap[vision]"}

    @property
    def configuration_identity(self) -> str:
        """Identify all regional algorithms, queries and effective discovery policies."""
        document = {
            "region_discovery": {
                "backend": dataclasses.asdict(self._region_discovery.backend),
                "discovery": self._region_discovery.discovery_identity.to_dict(),
            },
            "image_preparation": _IDENTITY_IMAGE_PREPARATION_POLICY,
        }
        if self._region_grounding is not None:
            document["region_grounding"] = {
                "backend": dataclasses.asdict(self._region_grounding.backend),
                "queries": [
                    {
                        "task": query.task.value,
                        "policy_id": query.policy_id,
                        "geometry": query.geometry.value,
                        "text": query.text,
                        "categories": list(query.categories),
                    }
                    for query in self._region_grounding.queries
                ],
            }
        if self._region_refinement is not None:
            document["region_refinement"] = {
                "backend": dataclasses.asdict(self._region_refinement.backend),
                "acceptance_policy": REFINEMENT_ACCEPTANCE_POLICY,
            }
        return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Run regional graphs and publish frame coverage even for empty or failed evidence."""
        output = _output(request)
        sequence = SequenceArtifactReader(_one(request, "sequence"))
        selection = _observation_selection(request)
        try:
            selected = frozenset(resolve_selection_offsets(sequence, selection))
        except SequenceSelectionError as error:
            raise ExecutorError(f"inputs.observation_selection: {error}") from error
        observation_ids = tuple(
            str(entry.observation_id)
            for entry in sequence.iter_index()
            if entry.modality == "image" and entry.offset in selected
        )
        run_id = _configured_stage_identity(request, self.configuration_identity)
        backend = self._region_discovery.backend
        discovery_identity = self._region_discovery.discovery_identity
        scratch = output.parent / f".tmp-{output.name}-prepared-{uuid4().hex[:8]}"
        scratch.mkdir(parents=True, exist_ok=False)
        backends: dict[str, Any] = {}
        released: set[int] = set()
        try:
            discovery = _AuditPublishingRegionDiscovery(self._build_discovery(scratch), run_id)
            backends["region_discovery"] = discovery
            grounding = (
                None if self._region_grounding is None else self._region_grounding.factory(scratch)
            )
            refiner = (
                None
                if self._region_refinement is None
                else self._region_refinement.factory(scratch)
            )
            for built, plan in (
                (grounding, self._region_grounding),
                (refiner, self._region_refinement),
            ):
                if (
                    built is not None
                    and plan is not None
                    and built.backend_provenance() != plan.backend
                ):
                    raise ExecutorError(
                        "regional backend provenance differs from its composed plan"
                    )
            grounding_stage = None
            refinement_stage = None
            if grounding is not None:
                assert self._region_grounding is not None
                grounding_stage = _QueryGroundingStage(
                    grounding, self._region_grounding.queries, run_id
                )
                backends["region_grounding"] = grounding
            if refiner is not None:
                refinement_stage = _ProposalRefinementStage(refiner, run_id)
                backends["region_refinement"] = refiner
            with RegionPerceptionRunWriter(output) as writer:
                writer.write_manifest(
                    RegionPerceptionRunManifest(
                        run_id=run_id,
                        sequence_name=sequence.manifest.sequence_name,
                        run_index=request.run_number(),
                        sequence_artifact_id=sequence.manifest.artifact_id,
                        source_observation_ids=observation_ids,
                        created_at=datetime.now(UTC).isoformat(),
                        selection_id=selection_identity(sequence.manifest.artifact_id, selection),
                        prepared_image_id=_IDENTITY_IMAGE_PREPARATION_POLICY,
                        discovery_backend_id=backend.backend_id,
                        discovery_model=backend.model,
                        discovery_version=backend.version,
                        discovery_configuration_fingerprint=backend.configuration_fingerprint or "",
                        normalization_policy_fingerprint=discovery_identity.normalization_config_digest,
                        grounding_backend_id=None
                        if grounding is None
                        else grounding.backend_provenance().backend_id,
                        grounding_configuration_fingerprint=None
                        if grounding is None
                        else grounding.backend_provenance().configuration_fingerprint,
                        refinement_backend_id=None
                        if refiner is None
                        else refiner.backend_provenance().backend_id,
                        refinement_configuration_fingerprint=None
                        if refiner is None
                        else refiner.backend_provenance().configuration_fingerprint,
                        code_version=self._code_version,
                    )
                )
                discovery.bind(writer)
                if refinement_stage is not None:
                    refinement_stage.bind(writer)

                def graph(
                    image: ImageObservation,
                ) -> tuple[PreparedImage, Sequence[StageDefinition]]:
                    prepared = _materialize_prepared_image(image, scratch, self._image_masks)
                    stages = [
                        StageDefinition(
                            stage_id="region_discovery",
                            capability="region_discovery",
                            run=lambda _inputs: discovery.discover(prepared),
                        )
                    ]
                    if grounding_stage is not None:
                        stages.append(
                            StageDefinition(
                                stage_id="region_grounding",
                                capability="region_grounding",
                                run=lambda _inputs: grounding_stage.ground_image(prepared),
                            )
                        )
                    if refinement_stage is not None:
                        stages.append(
                            StageDefinition(
                                stage_id="region_refinement",
                                capability="region_refinement",
                                depends_on=frozenset({"region_grounding"}),
                                run=lambda inputs: refinement_stage.refine_groundings(
                                    prepared,
                                    cast(
                                        Sequence[RegionGroundingExecution],
                                        inputs["region_grounding"],
                                    ),
                                ),
                            )
                        )
                    return prepared, stages

                def record(prepared: PreparedImage, outcomes: Sequence[StageOutcome]) -> None:
                    discovered = _succeeded_output(outcomes, "region_discovery")
                    groundings = _succeeded_output(outcomes, "region_grounding")
                    refinements = _succeeded_output(outcomes, "region_refinement")
                    for item in groundings:
                        writer.write_grounding(item)
                    for item in refinements:
                        writer.write_refinement(item)
                    final_regions = assemble_regions(
                        discovered=discovered, groundings=groundings, refinements=refinements
                    )
                    for region in final_regions:
                        writer.write_region(region)
                    failures = [item for item in outcomes if item.status is StageStatus.FAILED]
                    durable_image = replace(
                        prepared,
                        payload_reference=f"source-observation:{prepared.source_observation_id}",
                        payload_artifact=None,
                    )
                    writer.write_frame(
                        durable_image,
                        outcome="failed" if failures else "succeeded",
                        failure_detail="; ".join(
                            f"{item.stage_id}: {item.error}" for item in failures
                        )
                        if failures
                        else None,
                    )

                _execute_perception_frames(
                    sequence,
                    selected,
                    graph,
                    record,
                    backends,
                    self._perception_schedule,
                    released_backend_ids=released,
                )
                manifest = writer.finalize()
        finally:
            for value in {id(value): value for value in backends.values()}.values():
                _release_once(value, released)
            shutil.rmtree(scratch, ignore_errors=True)
        return _reference(request, REGIONS, str(manifest.run_id), manifest.file_inventory)

    def _build_discovery(self, prepared_image_root: Path) -> AuditedRegionDiscovery:
        """Build the run's discovery backend and hold it to the identity its plan names.

        Raises:
            TypeError: If the built backend cannot report the audit of its regions
                (``discover_audited``): the run would silently lose every rejection and merge.
            ExecutorError: If it reports another provenance or discovery identity than the
                plan declared; the run would be identified by a discovery it does not run.
        """
        plan = self._region_discovery
        discovery = plan.factory(prepared_image_root)
        if not isinstance(discovery, AuditedRegionDiscovery):
            raise TypeError(
                "the canonical region perception path needs a region discovery backend that "
                f"implements AuditedRegionDiscovery.discover_audited(); "
                f"{type(discovery).__name__} only returns regions, so its rejections and "
                "merges could not be persisted"
            )
        if discovery.backend_provenance() != plan.backend:
            raise ExecutorError(
                f"the region discovery backend reports {discovery.backend_provenance()!r}, but "
                f"the run's identity names {plan.backend!r}"
            )
        if discovery.discovery_identity() != plan.discovery_identity:
            raise ExecutorError(
                f"the region discovery backend reports {discovery.discovery_identity()!r}, but "
                f"the run's identity names {plan.discovery_identity!r}"
            )
        return discovery


class VisualPerceptionExecutor:
    """Extract visual features independently from explicitly selected immutable regions."""

    def __init__(
        self,
        *,
        dense_features: FeatureFactory,
        region_features: FeatureFactory,
        dense_region_pooling_enabled: bool = False,
        feature_debug_level: FeatureDebugLevel = FeatureDebugLevel.NONE,
        image_masks: Mapping[str, CameraImageMasks] | None = None,
        perception_schedule: PerceptionSchedule = PerceptionSchedule.FRAME_MAJOR,
        code_version: str | None = None,
    ) -> None:
        """Bind feature factories and image/pooling policies without semantic backends."""
        self._dense_features = dense_features
        self._region_features = region_features
        self._dense_region_pooling_enabled = dense_region_pooling_enabled
        self._feature_debug_level = feature_debug_level
        self._image_masks = image_masks or {}
        self._perception_schedule = perception_schedule
        self._code_version = code_version

    @property
    def required_modules(self) -> Mapping[str, str]:
        """Declare materialization dependencies before model construction."""
        return {"PIL": "visual perception requires Pillow; install contextmap[vision]"}

    @property
    def perception_schedule(self) -> PerceptionSchedule:
        """Return the selected scheduling policy."""
        return self._perception_schedule

    @property
    def configuration_identity(self) -> str:
        """Identify only the image preparation and optional pooling applied by this producer."""
        document = {
            "image_preparation": _IDENTITY_IMAGE_PREPARATION_POLICY,
            "dense_region_pooling": POOLING_POLICY if self._dense_region_pooling_enabled else None,
        }
        return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Execute selected feature stages, preserving source region IDs and lazy mask access."""
        output = _output(request)
        sequence, selection, selected, regions, _frames, by_observation = _perception_inputs(
            request
        )
        identity = _configured_stage_identity(request, self.configuration_identity)
        run_id = VisualEvidenceRunId(identity)
        payload_sink = _DeferredFeaturePayloadSink()
        scratch = output.parent / f".tmp-{output.name}-prepared-{uuid4().hex[:8]}"
        scratch.mkdir(parents=True, exist_ok=False)
        backends: dict[str, Any] = {}
        released: set[int] = set()
        try:
            dense_scope = FeatureBuildScope(
                run_id=run_id,
                feature_stage_id="dense_feature_extraction",
                source_artifact_id=identity,
                payload_sink=payload_sink,
                prepared_image_root=scratch,
            )
            region_scope = replace(
                dense_scope,
                feature_stage_id="region_feature_extraction",
                mask_source=_RegionMaskSource(regions.mask_resolver()),
            )
            backends = {
                "dense_feature_extraction": self._dense_features(dense_scope),
                "region_feature_extraction": self._region_features(region_scope),
            }
            if self._dense_region_pooling_enabled:
                backends["dense_region_pooling"] = _DenseRegionPoolingStage(
                    run_id=run_id, payload_sink=payload_sink, mask_resolver=regions.mask_resolver()
                )
            feature_stages = tuple(
                stage
                for stage in CANONICAL_PRESET_V1.stages
                if stage.capability == "feature_extractor"
            )
            stages = (
                StageSpec(stage_id="image_preparation", capability="image_preparation"),
                StageSpec(stage_id="region_discovery", capability="region_discovery"),
                *feature_stages,
            )
            preset = PipelinePreset(preset_id="visual-evidence/1", stages=stages)
            if self._dense_region_pooling_enabled:
                preset = with_dense_region_pooling(preset)
            resolved = resolve_pipeline(
                preset,
                backend_factories={
                    key: _constant_backend(value) for key, value in backends.items()
                },
            )
            with VisualEvidenceRunWriter(
                output_dir=output,
                run_id=run_id,
                run_index=request.run_number(),
                sequence_name=sequence.manifest.sequence_name,
                sequence_artifact_id=sequence.manifest.artifact_id,
                region_artifact_id=regions.manifest.run_id,
                selection_id=selection_identity(sequence.manifest.artifact_id, selection),
                configuration_digest=resolved.configuration_digest(),
                backend_provenance={
                    key: value.backend_provenance() for key, value in backends.items()
                },
                code_version=self._code_version,
                feature_debug_level=self._feature_debug_level,
            ) as writer:
                payload_sink.bind(writer)

                def graph(
                    image: ImageObservation,
                ) -> tuple[PreparedImage, Sequence[StageDefinition]]:
                    prepared = _materialize_prepared_image(image, scratch, self._image_masks)
                    _validate_preparation(prepared, regions.frame(image.observation_id).image)
                    region_values = by_observation.get(str(image.observation_id), ())
                    definitions = resolved.build_stage_graph(
                        {"image_preparation": prepared, "region_discovery": region_values}
                    )
                    return prepared, _with_regional_outcome(
                        definitions, regions.frame(image.observation_id)
                    )

                def record(prepared: PreparedImage, outcomes: Sequence[StageOutcome]) -> None:
                    produced = tuple(
                        feature
                        for item in outcomes
                        if item.stage_id in backends and item.status is StageStatus.SUCCEEDED
                        for feature in cast(Sequence[VisualFeature], item.output)
                    )  # type: ignore[union-attr]
                    writer.add_result(
                        VisualEvidenceResult(
                            source_observation_id=prepared.source_observation_id,
                            prepared_image=prepared,
                            features=produced,
                        )
                    )
                    writer.add_stage_outcomes(outcomes, prepared.source_observation_id)
                    payload_sink.discard_for(prepared.source_observation_id)

                _execute_perception_frames(
                    sequence,
                    selected,
                    graph,
                    record,
                    backends,
                    self._perception_schedule,
                    released_backend_ids=released,
                    payload_sink=payload_sink,
                    last_dense_consumer=(
                        "dense_region_pooling"
                        if self._dense_region_pooling_enabled
                        else "dense_feature_extraction"
                    ),
                )
                manifest = writer.finalize()
        finally:
            for backend in {id(value): value for value in backends.values()}.values():
                _release_once(backend, released)
            shutil.rmtree(scratch, ignore_errors=True)
        return _reference(request, VISUAL_EVIDENCE, str(manifest.run_id), manifest.file_inventory)


class SemanticInterpretationExecutor:
    """Interpret frozen regions as a producer independent from visual feature extraction."""

    def __init__(
        self,
        *,
        semantic_interpreter: SemanticInterpreterFactory,
        semantic_request_policy: SemanticRequestPolicy,
        semantic_debug_level: SemanticDebugLevel = SemanticDebugLevel.FULL,
        image_masks: Mapping[str, CameraImageMasks] | None = None,
        perception_schedule: PerceptionSchedule = PerceptionSchedule.FRAME_MAJOR,
        code_version: str | None = None,
    ) -> None:
        """Bind the semantic backend, request policy and auditable execution settings."""
        self._semantic_interpreter = semantic_interpreter
        self._semantic_request_policy = semantic_request_policy
        self._semantic_debug_level = semantic_debug_level
        self._image_masks = image_masks or {}
        self._perception_schedule = perception_schedule
        self._code_version = code_version

    @property
    def required_modules(self) -> Mapping[str, str]:
        """Declare the exact-view materialization dependency."""
        return {"PIL": "semantic interpretation requires Pillow; install contextmap[vision]"}

    @property
    def configuration_identity(self) -> str:
        """Identify the prompts, views and scene conditioning this producer applies."""
        document = self._semantic_request_policy.to_document()
        return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Execute semantic requests over frozen regions and persist exact views and failures."""
        output = _output(request)
        sequence, selection, selected, regions, _frames, by_observation = _perception_inputs(
            request
        )
        identity = _configured_stage_identity(request, self.configuration_identity)
        run_id = SemanticInterpretationRunId(identity)
        scratch = output.parent / f".tmp-{output.name}-prepared-{uuid4().hex[:8]}"
        scratch.mkdir(parents=True, exist_ok=False)
        service: SemanticInterpretationService | None = None
        released: set[int] = set()
        try:
            service = SemanticInterpretationService(
                interpreter=self._semantic_interpreter(scratch),
                run_id=run_id,
                view_root=scratch,
                policy=self._semantic_request_policy,
                mask_resolver=regions.mask_resolver(),
            )
            selected_stages = tuple(
                stage
                for stage in CANONICAL_PRESET_V1.stages
                if stage.capability in {"scene_interpretation", "region_interpretation"}
                and (
                    self._semantic_request_policy.scene is not None
                    if stage.capability == "scene_interpretation"
                    else self._semantic_request_policy.region is not None
                )
            )
            stages = (
                StageSpec(stage_id="image_preparation", capability="image_preparation"),
                StageSpec(stage_id="region_discovery", capability="region_discovery"),
                *selected_stages,
            )
            preset = PipelinePreset(preset_id="semantic-interpretation/1", stages=stages)
            backends = {stage.stage_id: service for stage in selected_stages}
            resolved = resolve_pipeline(
                preset, backend_factories={key: lambda _parameters: service for key in backends}
            )
            with SemanticInterpretationRunWriter(
                output_dir=output,
                sequence_name=sequence.manifest.sequence_name,
                run_id=run_id,
                run_index=request.run_number(),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id=selection_identity(sequence.manifest.artifact_id, selection),
                region_artifact_id=regions.manifest.run_id,
                configuration_digest=resolved.configuration_digest(),
                backend_provenance={"semantic_interpreter": service.backend_provenance()},
                code_version=self._code_version,
                semantic_debug_level=self._semantic_debug_level,
            ) as writer:
                service.bind(writer)

                def graph(
                    image: ImageObservation,
                ) -> tuple[PreparedImage, Sequence[StageDefinition]]:
                    prepared = _materialize_prepared_image(image, scratch, self._image_masks)
                    _validate_preparation(prepared, regions.frame(image.observation_id).image)
                    definitions = resolved.build_stage_graph(
                        {
                            "image_preparation": prepared,
                            "region_discovery": by_observation.get(str(image.observation_id), ()),
                        }
                    )
                    if (
                        self._semantic_request_policy.region_scene_context
                        and self._semantic_request_policy.scene is not None
                    ):
                        definitions = tuple(
                            replace(stage, depends_on=stage.depends_on | {"scene_interpretation"})
                            if stage.stage_id == "region_interpretation"
                            else stage
                            for stage in definitions
                        )
                    return prepared, _with_regional_outcome(
                        definitions, regions.frame(image.observation_id)
                    )

                def record(prepared: PreparedImage, outcomes: Sequence[StageOutcome]) -> None:
                    claims = tuple(_succeeded_output(outcomes, "region_interpretation"))
                    scene = next(
                        (
                            item.output
                            for item in outcomes
                            if item.stage_id == "scene_interpretation"
                            and item.status is StageStatus.SUCCEEDED
                        ),
                        None,
                    )
                    failed = any(item.status is StageStatus.FAILED for item in outcomes)
                    writer.add_observation(
                        SemanticObservation(
                            source_observation_id=prepared.source_observation_id,
                            semantic_interpretation_run_id=run_id,
                            claims=claims,
                            scene_context=cast(SceneContext | None, scene),
                            created_at=datetime.now(UTC).isoformat(),
                            status=(
                                SemanticObservationStatus.FAILED
                                if failed
                                else SemanticObservationStatus.SUCCEEDED
                            ),
                        )
                    )
                    writer.add_stage_outcomes(outcomes)

                _execute_perception_frames(
                    sequence,
                    selected,
                    graph,
                    record,
                    backends,
                    self._perception_schedule,
                    released_backend_ids=released,
                )
                manifest = writer.finalize()
        finally:
            if service is not None:
                _release_once(service, released)
            shutil.rmtree(scratch, ignore_errors=True)
        return _reference(
            request, SEMANTIC_INTERPRETATION, str(manifest.run_id), manifest.file_inventory
        )


def _with_regional_outcome(
    stages: Sequence[StageDefinition], frame: Any
) -> Sequence[StageDefinition]:
    """Preserve failed regional sources so their consumers are skipped."""
    if frame.outcome != "failed":
        return stages

    def failed(_inputs: Mapping[str, object]) -> object:
        raise ExecutorError(f"regional source failed: {frame.failure_detail}")

    return tuple(
        replace(stage, run=failed) if stage.stage_id == "region_discovery" else stage
        for stage in stages
    )


def _constant_backend(backend: Any) -> Any:
    """Bind one already-built backend without closing over a mutable loop variable."""
    return lambda _parameters: backend


def _perception_inputs(
    request: StageRequest,
) -> tuple[Any, Any, frozenset[int], Any, dict[str, Any], dict[str, tuple[Region2D, ...]]]:
    """Check producer selection and physical frames before preparing images or loading models."""
    sequence = SequenceArtifactReader(_one(request, "sequence"))
    regions = RegionPerceptionRunReader(_one(request, "regions"))
    selection = _observation_selection(request)
    selected = frozenset(resolve_selection_offsets(sequence, selection))
    wanted = {
        str(item.observation_id)
        for item in sequence.iter_index()
        if item.modality == "image" and item.offset in selected
    }
    if regions.manifest.sequence_artifact_id != sequence.manifest.artifact_id:
        raise ExecutorError("regional artifact belongs to another sequence")
    frames = {str(source_id): source_id for source_id in regions.manifest.source_observation_ids}
    if not wanted <= set(frames):
        raise ExecutorError("regional artifact does not cover the requested physical observations")
    grouped: dict[str, list[Region2D]] = {key: [] for key in wanted}
    for region in regions.read_regions():
        key = str(region.source_observation_id)
        if key not in frames:
            raise ExecutorError("regional evidence references an undeclared observation")
        if key in grouped:
            grouped[key].append(region)
    return (
        sequence,
        selection,
        selected,
        regions,
        frames,
        {key: tuple(value) for key, value in grouped.items()},
    )


def _validate_preparation(prepared: PreparedImage, upstream: PreparedImage) -> None:
    """Reject rematerialization that would alter the coordinate support used by regions."""
    if (prepared.width, prepared.height) != (upstream.width, upstream.height):
        raise ExecutorError("prepared image dimensions differ from the regional upstream")
    if prepared.valid_region != upstream.valid_region or tuple(prepared.exclusion_regions) != tuple(
        upstream.exclusion_regions
    ):
        raise ExecutorError("image eligibility differs from the regional preparation")


def _execute_perception_frames(
    sequence: SequenceArtifactReader,
    selected: frozenset[int],
    graph: Any,
    record: Any,
    backends: Mapping[str, Any],
    schedule: PerceptionSchedule,
    *,
    released_backend_ids: set[int] | None = None,
    payload_sink: _DeferredFeaturePayloadSink | None = None,
    last_dense_consumer: str | None = None,
) -> None:
    """Schedule independent producer graphs while releasing numerical state at its last consumer."""

    def images() -> Iterator[ImageObservation]:
        for entry in sequence.iter_index():
            if entry.modality == "image" and entry.offset in selected:
                image = sequence.observation_at(entry.offset)
                if isinstance(image, ImageObservation):
                    yield image

    if schedule is PerceptionSchedule.FRAME_MAJOR:
        for image in images():
            prepared, stages = graph(image)
            record(prepared, execute_stage_graph(stages))
        return
    materialized = [graph(image) for image in images()]
    graphs = [stages for _, stages in materialized]
    if not graphs:
        return
    released = set() if released_backend_ids is None else released_backend_ids
    releases = _release_points([stage.stage_id for stage in graphs[0]], backends)

    def after_stage(stage_id: str) -> None:
        for backend in releases.get(stage_id, ()):
            _release_once(backend, released)

    def after_frame_stage(index: int, stage_id: str, outcome: StageOutcome) -> None:
        if payload_sink is not None and stage_id == last_dense_consumer:
            payload_sink.discard_for(materialized[index][0].source_observation_id)

    try:
        outcomes = execute_stage_graphs_stage_major(
            graphs, after_stage=after_stage, after_frame_stage=after_frame_stage
        )
        for (prepared, _), items in zip(materialized, outcomes, strict=True):
            record(prepared, items)
    finally:
        if payload_sink is not None:
            for prepared, _ in materialized:
                payload_sink.discard_for(prepared.source_observation_id)


class SemanticScoringExecutor:
    """Scores the claims of one perception run against that run's own region features (#652).

    The perception run is only read: its claims, features and payloads stay where they are and
    the scoring run references them by identity, so whether the run was pinned or just produced,
    every scorer arm over it scores exactly the same claims without re-running its VLM. The
    executor decides nothing: which claim pairs with which feature, the cosine rule and the
    refusal of a feature of another embedding space belong to the composed scorer, and what the
    run may record to :class:`~contextmap.visual_perception.SemanticScoringRunWriter`.
    """

    def __init__(self, *, plan: SemanticScoringPlan, code_version: str | None = None) -> None:
        """Bind the executor to the composed scorer and the code revision it records."""
        self._plan = plan
        self._code_version = code_version

    @property
    def configuration_identity(self) -> str:
        """Return the identity of the scorer beyond the stage's declared configuration.

        The scorer's provenance fingerprint resolves every default and version that changes a
        score value (text template, precision, adapter and scoring rule), so changing any of
        them changes the reuse key and the run id (#636).
        """
        document = {"semantic_scorer": dataclasses.asdict(self._plan.backend)}
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Score selected visual and semantic producers sharing frozen regional evidence."""
        output = _output(request)
        regions = RegionPerceptionRunReader(_one(request, "regions"))
        visual = VisualEvidenceRunReader(_one(request, "visual_evidence"))
        semantic = SemanticInterpretationRunReader(_one(request, "semantic_interpretation"))
        first, second = visual.manifest, semantic.manifest
        if (
            first.sequence_artifact_id != second.sequence_artifact_id
            or first.region_artifact_id != second.region_artifact_id
            or first.selection_id != second.selection_id
            or set(first.source_observation_ids) != set(second.source_observation_ids)
        ):
            raise ExecutorError("scoring requires equivalent visual and semantic selections")
        if (
            first.region_artifact_id != regions.manifest.run_id
            or first.sequence_artifact_id != regions.manifest.sequence_artifact_id
        ):
            raise ExecutorError("scoring regional upstream differs from its evidence producers")
        grouped: dict[str, list[Region2D]] = {}
        for region in regions.read_regions():
            grouped.setdefault(str(region.source_observation_id), []).append(region)
        semantic_by_observation = {
            item.source_observation_id: item for item in semantic.iter_observations()
        }
        composed = self._plan.factory(visual.feature_store())
        if composed.scorer.backend_provenance() != self._plan.backend:
            raise ExecutorError("semantic scorer provenance differs from the composed plan")
        writer = SemanticScoringRunWriter(
            output_dir=output,
            sequence_name=first.sequence_name,
            run_id=SemanticScoringRunId(
                _configured_stage_identity(request, self.configuration_identity)
            ),
            run_index=request.run_number(),
            lineage=SemanticScoringLineage(
                sequence_artifact_id=first.sequence_artifact_id,
                selection_id=first.selection_id,
                region_artifact_id=regions.manifest.run_id,
                visual_evidence_artifact_id=first.run_id,
                semantic_interpretation_artifact_id=str(second.run_id),
            ),
            scorer=self._plan.backend,
            embedding_space=composed.embedding_space,
            text_template=self._plan.text_template,
            code_version=self._code_version,
        )
        try:
            for evidence in visual.iter_results():
                interpretation = semantic_by_observation[evidence.source_observation_id]
                view = PerceptionResult(
                    source_observation_id=evidence.source_observation_id,
                    sequence_artifact_id=first.sequence_artifact_id,
                    region_artifact_id=first.region_artifact_id,
                    visual_evidence_artifact_id=first.run_id,
                    semantic_interpretation_artifact_id=str(second.run_id),
                    regions=tuple(grouped.get(str(evidence.source_observation_id), ())),
                    features=evidence.features,
                    claims=interpretation.claims,
                    scene_context=interpretation.scene_context,
                )
                writer.add_result(view, composed.scorer.score(view.claims, view.features))
            manifest = writer.finalize()
        finally:
            _release(composed.scorer)
        return _reference(request, SEMANTIC_SCORES, str(manifest.run_id), manifest.file_inventory)


class EntityResolutionExecutor:
    """Decides identity across the materialized entities; without proof it stays unresolved."""

    def __init__(
        self,
        *,
        retrieval: CandidateRetrievalPolicy,
        builder: MatchEvidenceBuilder,
        resolution: ConservativeResolutionPolicy,
        code_version: str,
        belief_policy: ResolvedBeliefPolicy | None = None,
        debug_level: ResolutionDebugLevel = ResolutionDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the retrieval, the evidence channels and the decision policy.

        ``belief_policy`` derives the belief of every resolved entity under
        ``resolved-belief-min-support-v1``; ``None`` derives none, as before the rule existed.

        Raises:
            ValueError: If ``code_version`` is empty: the run records the code that produced it.
        """
        self._retrieval = retrieval
        self._builder = builder
        self._resolution = resolution
        self._code_version = _recorded_code_version(code_version)
        self._belief_policy = belief_policy
        self._debug_level = debug_level

    @property
    def belief_policy(self) -> ResolvedBeliefPolicy | None:
        """The composed resolved-belief rule, or ``None`` when no belief is derived."""
        return self._belief_policy

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Resolve the ``entities`` run without mutating it and reference the resolution run."""
        output = _output(request)
        mapping = SemanticMappingRunReader(_one(request, "entities"))
        entities = list(mapping.iter_entities())
        candidate_sets = retrieve_candidate_sets(entities, self._retrieval)
        resolutions = resolve_candidate_pairs(
            entities,
            candidate_sets,
            self._builder,
            self._resolution,
            code_version=self._code_version,
        )
        run_id = EntityResolutionRunId(request.identity())
        materialization = materialize_resolved_entities(
            entities,
            [item.decision for item in resolutions],
            resolution_run_id=run_id,
            code_version=self._code_version,
            belief_policy=self._belief_policy,
        )
        manifest = EntityResolutionRunWriter(
            output_dir=output,
            run_id=run_id,
            lineage=lineage_from_mapping_manifest(mapping.manifest),
            code_version=self._code_version,
            debug_level=self._debug_level,
        ).write(
            candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization
        )
        return _reference(request, RESOLUTION, str(manifest.run_id), manifest.file_inventory)


class SpatialRelationsExecutor:
    """Derives the relations between resolved entities and keeps the evidence behind each one."""

    def __init__(
        self,
        *,
        policies: RelationsRunPolicies,
        code_version: str,
        debug_level: RelationsRunDebugLevel = RelationsRunDebugLevel.NONE,
    ) -> None:
        """Bind the executor to the frame conventions and the predicate policies.

        ``policies.geometry_summary`` is the only source of the summary policy, and it is also
        what gets persisted in the run's own provenance, so there is exactly one place that can
        say which policy actually produced the evidence. A second parameter for it could name a
        different policy than the one the run records, so there is none.

        Raises:
            ValueError: If ``code_version`` is empty: the run records the code that produced it.
        """
        self._policies = policies
        self._code_version = _recorded_code_version(code_version)
        self._debug_level = debug_level

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Relate the ``entities`` resolution over the ``geometry`` map and reference the run."""
        output = _output(request)
        resolution = EntityResolutionRunReader(_one(request, "entities"))
        with GeometricMapArtifactReader(_one(request, "geometry")) as geometry:
            derived = derive_relations(
                resolution.resolved_entities(),
                source=geometry.geometry(),
                policies=self._policies,
            )
        manifest = SpatialRelationsRunWriter(
            output_dir=output,
            run_id=SpatialRelationsRunId(request.identity()),
            lineage=lineage_from_resolution_manifest(resolution.manifest),
            policies=derived.policies,
            code_version=self._code_version,
            debug_level=self._debug_level,
        ).write(
            candidates=derived.candidates,
            evidence=derived.evidence,
            decisions=derived.decisions,
        )
        return _reference(request, RELATIONS, str(manifest.run_id), manifest.file_inventory)


def _observation_window(clock_id: str, *, start_ns: int, end_ns: int) -> ObservationWindow:
    def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
        seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
        return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=clock_id)

    return ObservationWindow(start=_timestamp(start_ns), end=_timestamp(end_ns))


def _context_map_configuration_fingerprint(
    *, assembly_policy: PolicyRef, up_direction: tuple[float, float, float] | None
) -> str:
    """Deterministic hash of the configuration ``ContextMapExecutor`` actually assembled with.

    ``ContextMapExecutor`` has no backend to select, but it does have two effective inputs of
    its own that change what it assembles: the assembly policy's identity and the map's
    declared up direction (``None`` is itself a distinct, meaningful configuration, not an
    absence of one). ``runtime.provenance_identity`` requires every stage artifact to record an
    effective configuration digest; this was previously hardcoded to ``None`` (PR #438 review).
    """
    payload = json.dumps(
        {
            "assembly_policy_id": assembly_policy.policy_id,
            "assembly_policy_version": assembly_policy.version,
            "up_direction": up_direction,
        },
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class ContextMapExecutor:
    """Assembles the repository's public product: the ``ContextMap``, by reference.

    Every entity, relation and piece of evidence was already decided by Entity Resolution and
    Spatial Relations; this only translates their real, finished runs -- and the geometric map
    and sequence they were built over -- into what :func:`~contextmap.artifact.
    assemble_context_map_with_metrics` needs, and never infers anything itself (that function
    already owns the translation rule, see :data:`~contextmap.artifact.
    CONTEXT_MAP_ASSEMBLY_POLICY_ID`).
    """

    def __init__(
        self,
        *,
        up_axis: AxisDirection | None = None,
        assembly_policy: PolicyRef | None = None,
        code_version: str | None = None,
    ) -> None:
        """Bind the executor to the map frame's declared up axis and the assembly policy.

        Args:
            up_axis: Spatial Relations' own composed ``FrameConventions.up_axis``, when
                declared -- the same value that already shaped this run's relations, never
                re-derived independently here. ``None`` when no run declared one.
            assembly_policy: Identity of the assembly rule; defaults to
                :data:`~contextmap.artifact.CONTEXT_MAP_ASSEMBLY_POLICY_ID` version ``"2"``, the
                one :func:`~contextmap.artifact.assemble_context_map_with_metrics` actually
                implements.
            code_version: Code revision that produced the run.
        """
        self._up_direction = up_axis.vector if up_axis is not None else None
        self._assembly_policy = assembly_policy or PolicyRef(
            policy_id=CONTEXT_MAP_ASSEMBLY_POLICY_ID, version="2"
        )
        self._code_version = code_version
        self._configuration_fingerprint = _context_map_configuration_fingerprint(
            assembly_policy=self._assembly_policy, up_direction=self._up_direction
        )

    def execute(self, request: StageRequest) -> ArtifactRef:
        """Assemble and write the final ``ContextMap`` from this run's real upstream artifacts.

        Raises:
            ExecutorError: If the ``sequence`` input is not the artifact the ``geometry`` input
                was actually built over. ``ArtifactKind.SEQUENCE`` is not a structural
                dependency (:attr:`~contextmap.artifact.ArtifactKind.is_structural`), so
                :class:`~contextmap.artifact.ContextMapArtifactWriter` never opens or verifies
                it; without this check, a caller wiring the wrong ``sequence`` run could publish
                a lineage entry whose ``artifact_id`` (taken from the geometry manifest) and
                ``content_identity`` (the digest of whatever directory was actually passed)
                silently name two different artifacts.
        """
        output = _output(request)
        sequence_dir = _one(request, "sequence")
        geometry_dir = _one(request, "geometry")
        resolution_dir = _one(request, "entities")
        relations_dir = _one(request, "relations")

        sequence = SequenceArtifactReader(sequence_dir)
        with GeometricMapArtifactReader(geometry_dir) as geometry_reader:
            geometry_manifest = geometry_reader.manifest
            bounds = geometry_reader.geometry().geometric_map.bounds
        if sequence.manifest.artifact_id != geometry_manifest.sequence_artifact_id:
            raise ExecutorError(
                f"stage {request.stage_id!r} was given sequence "
                f"{sequence.manifest.artifact_id!r}, but the geometric map it was also given "
                f"was built over sequence {geometry_manifest.sequence_artifact_id!r}"
            )
        expected_selection_id = selection_identity(
            sequence.manifest.artifact_id, FullSequenceSelection()
        )
        if geometry_manifest.selection_id != expected_selection_id:
            raise ExecutorError(
                f"stage {request.stage_id!r} was given a sequence selection "
                f"{expected_selection_id!r} that does not match the geometric map's own "
                f"selection {geometry_manifest.selection_id!r}"
            )
        relations_reader = SpatialRelationsRunReader(relations_dir)
        predicates = sorted(
            {relation.predicate for relation in relations_reader.iter_relations()},
            key=lambda predicate: predicate.value,
        )

        metadata = ContextMapMetadata(
            creation=MapCreation(
                assembly_policy=self._assembly_policy,
                code_version=self._code_version,
                configuration_fingerprint=self._configuration_fingerprint,
            ),
            source_sequences=(
                SourceSequence(
                    sequence_artifact_id=str(geometry_manifest.sequence_artifact_id),
                    selection_id=geometry_manifest.selection_id,
                ),
            ),
            frame=estimator_local_map_frame(
                frame_id=str(geometry_manifest.map_frame), up_direction=self._up_direction
            ),
            bounds=bounds,
            time_bounds=_observation_window(
                geometry_manifest.clock_id,
                start_ns=geometry_manifest.start_time_ns,
                end_ns=geometry_manifest.end_time_ns,
            ),
            capabilities=DeclaredCapabilities(
                content=(MapCapability.ENTITIES, MapCapability.GEOMETRY, MapCapability.RELATIONS),
                relation_predicates=tuple(predicates),
            ),
        )
        sequence_lineage = (
            UpstreamArtifact(
                artifact_id=str(geometry_manifest.sequence_artifact_id),
                kind=ArtifactKind.SEQUENCE,
                content_identity=artifact_digest(sequence_dir),
                configuration_fingerprint=None,
                code_version=None,
                model_identities=(),
            ),
        )
        result = assemble_context_map_with_metrics(
            context_map_id=ContextMapId(request.identity()),
            metadata=metadata,
            geometric_map_location=geometry_dir,
            entity_resolution_location=resolution_dir,
            spatial_relations_location=relations_dir,
            additional_lineage=sequence_lineage,
        )
        location_by_kind = {
            ArtifactKind.GEOMETRIC_MAP: geometry_dir,
            ArtifactKind.ENTITY_RESOLUTION_RUN: resolution_dir,
            ArtifactKind.SPATIAL_RELATIONS_RUN: relations_dir,
        }
        upstream_locations = {
            item.artifact_id: location_by_kind[item.kind]
            for item in result.context_map.lineage
            if item.kind.is_structural
        }
        manifest, _write_metrics = write_context_map_with_metrics(
            result.context_map, output_dir=output, upstream_locations=upstream_locations
        )
        # A validação completa é do artifact; o estágio só se recusa a publicar como sucesso um
        # mapa que ela não verifica (#499): um build nunca termina bem sobre um mapa inválido.
        report = validate_context_map_artifact(
            output,
            level=ValidationLevel.FULL,
            dependency_paths={
                **upstream_locations,
                str(sequence.manifest.artifact_id): sequence_dir,
            },
        )
        if report.status is not ValidationStatus.VERIFIED:
            findings = "; ".join(f"{item.code}: {item.message}" for item in report.findings[:5])
            raise ExecutorError(
                f"the context map written to {output} does not verify ({report.status.value}): "
                f"{findings}"
            )
        return _reference(
            request, CONTEXT_MAP, str(manifest.context_map_id), manifest.file_inventory
        )
