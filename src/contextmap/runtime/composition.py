"""Composition root: turn an effective configuration into concrete implementations.

This is the single place where concrete backends are named and constructed. A
capability never imports the runtime and a downstream stage never learns which backend
produced its input: every implementation leaves here behind the port its capability
publishes, so swapping a backend is a configuration change and no downstream source
changes.

Three rules keep the composition honest:

- Constructing is not loading. A backend's parameters are validated by the capability's
  own configuration class, and its optional modules and secrets are checked, before any
  model is requested. Bundled loaders are lazy; a model the repository does not know how
  to load is supplied by the caller through a :data:`RuntimeProvider`.
- There is no fallback. A selected backend that cannot be built raises a specific error;
  nothing substitutes another backend or drops the stage.
- No scientific policy lives here. Factories only translate configured values into the
  capability's own types; defaults, ranges and meaning belong to the capability.

Backend modules are imported lazily, inside the factory that needs them, so composing a
configuration imports only the backends it selects. There is no registry, plugin system
or service locator: the table of factories below is explicit and closed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from contextmap.runtime.coercion import ParameterError, build_config
from contextmap.runtime.components import (
    CANONICAL2_PROFILE_ID,
    COMPONENTS,
    PRESETS,
    StageDeclaration,
)
from contextmap.runtime.configuration import (
    ComponentConfig,
    ConfigurationError,
    EffectiveConfig,
    ResolvedSecrets,
    check_component_availability,
    check_selection,
)
from contextmap.runtime.errors import (
    BackendConfigurationError,
    BackendRuntimeMissingError,
    BackendUnavailableError,
    CompositionError,
    ExecutorIdentityMissingError,
    ProviderConfigurationError,
    StageUnavailableError,
)

if TYPE_CHECKING:
    from contextmap.entity_resolution import (
        CandidateRetrievalPolicy,
        ComparisonChannels,
        ConservativeResolutionPolicy,
        ResolvedBeliefPolicy,
    )
    from contextmap.geometric_mapping import (
        MotionCorrectionPolicy,
        PoseInterpolationDeskewPolicy,
        ScanVoxelPolicy,
    )
    from contextmap.ingestion import SourceAdapter, SourceAdapterConfig
    from contextmap.point_representation import ObservationSupportAnchorPolicy, PointEncoder
    from contextmap.region_perception import (
        AuditedRegionDiscovery,
        DiscoveryInput,
        GroundingQuery,
        RegionDiscoveryIdentity,
        RegionGrounding,
        RegionRefinement,
    )
    from contextmap.runtime.pipeline import StageExecutor
    from contextmap.semantic_fusion import (
        BaselineAccumulationPolicy,
        GeometryOverlapSupportPolicy,
        QualityAwareAccumulationPolicy,
    )
    from contextmap.semantic_interpretation import (
        SemanticInterpreter,
        SemanticInterpreterCapabilities,
        SemanticRequestPolicy,
    )
    from contextmap.semantic_mapping import GeometrySummaryPolicy, SemanticMapId
    from contextmap.semantic_scoring import (
        SemanticScorer,
    )
    from contextmap.sensor_association import (
        BoxDepthComponentPolicy,
        CandidateGeometryPolicy,
        DiagnosticTolerances,
        InterpolationPolicy,
        OcclusionPolicy,
    )
    from contextmap.shared import BackendProvenance
    from contextmap.spatial_relations import RelationsRunPolicies
    from contextmap.state_estimation import LookupPolicy, StateEstimator
    from contextmap.visual_perception import (
        EmbeddingSpace,
        FeatureExtractor,
        FeatureStoreReader,
        VisualEvidenceRunId,
    )

RuntimeProvider = Callable[[Any, ResolvedSecrets], object]
"""Builds the model runtime or client of a backend the repository cannot load itself.

It receives the capability's own configuration object, already validated, and the secrets
that backend declares (and only those), and returns whatever runtime the backend's
adapter expects (for example a SAM3 runtime or a Gemini client). Loading the model is the
provider's business, so heavy SDKs stay out of the runtime package. Nothing about a run
reaches it: a runtime receives each input already materialized (the pixels of a discovery
pass, the verified bytes of a semantic view), resolved against the run's directory by the
adapter the composition builds for that run (#661).
"""


def resolve_provider(component_id: str, target: str) -> RuntimeProvider:
    """Resolve a ``"resources.providers"`` target into the :data:`RuntimeProvider` it names.

    This is the declarative counterpart of passing a :data:`RuntimeProvider` in Python: a
    configuration names, instead of embeds, the callable the composition root asks for a
    model runtime. The imported attribute *is* the provider -- there is no intermediate
    factory or wrapper, so it must already accept ``(config, secrets)`` and return the
    runtime, exactly like a provider supplied through ``compose(providers=...)``.

    Security posture: only the ``"module:attribute"`` syntax is accepted, never ``eval``;
    no module is ever installed automatically; and this is called lazily, by
    :meth:`_Context.runtime`/:meth:`_Context.optional_runtime`, only for a component that is
    actually being composed -- never eagerly for every target a document declares. A
    ``resources.providers`` target is Python code that is imported and then called at
    runtime, the same trust boundary as any other configuration that names executable code:
    configuration from an untrusted source must never be resolved this way.

    Args:
        component_id: Variation point the target was declared for, used only to build an
            actionable error.
        target: ``"module_name:attribute"``.

    Returns:
        The imported attribute, already verified callable.

    Raises:
        ProviderConfigurationError: If ``target`` is malformed, its module cannot be
            imported, the attribute does not exist on it, or the attribute is not callable.
    """
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ProviderConfigurationError(
            component_id, target, "must look like 'module:attribute' with both parts non-empty"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ProviderConfigurationError(
            component_id, target, f"module {module_name!r} could not be imported: {error}"
        ) from error
    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as error:
        raise ProviderConfigurationError(
            component_id, target, f"module {module_name!r} has no attribute {attribute_name!r}"
        ) from error
    if not callable(attribute):
        raise ProviderConfigurationError(component_id, target, "is not callable")
    return cast("RuntimeProvider", attribute)


def provided_runtimes(
    effective: EffectiveConfig, providers: Mapping[str, RuntimeProvider] | None = None
) -> frozenset[str]:
    """Name the components whose model runtime a provider supplies instead of a bundled loader.

    A provider is expected for a component when the caller passes one explicitly or the
    configuration declares a ``resources.providers`` target for it. The composition then never
    imports that component's bundled modules (loading the model is the provider's business),
    so :func:`~contextmap.runtime.pipeline.preflight` must not require them either: this is the
    ``provided_runtimes`` it receives, by the same rule the composition applies.

    Args:
        effective: The resolved configuration, with its declared targets.
        providers: The providers the caller passes explicitly, by component identity.

    Returns:
        The component identities, ``"<capability>.<slot>"``.
    """
    return frozenset(providers or ()) | frozenset(effective.config.resources.providers)


class ExecutorIdentities(TypedDict):
    """What a run's executors record that no configuration carries (issue #659).

    The keys are the keyword arguments of :func:`compose_executors` of the same names.

    Attributes:
        code_version: The code identity of the run, unchanged; ``None`` when it has none.
        code_digest: Digest of that code identity; ``None`` when it has none.
        semantic_map_id: The semantic map the run's entities belong to; ``None`` when the run
            names no dataset.
    """

    code_version: str | None
    code_digest: str | None
    semantic_map_id: SemanticMapId | None


def executor_identities(
    effective: EffectiveConfig, code_identity: str | None
) -> ExecutorIdentities:
    """Derive, from a run's code identity and dataset, what its executors record.

    One rule for every production caller of :func:`compose_executors` (the CLI and
    :class:`~contextmap.runtime.api.Runtime`), so a run composed by either records the same:

    - ``code_version`` is the code identity itself (``--code-identity``, a commit, a tag);
    - ``code_digest`` is ``"sha256:"`` and the SHA-256 of the code identity's UTF-8 bytes. The
      runtime knows its code only by the identity a run declares and never hashes a source
      tree, so the digest pins that declared identity, not the bytes of the code;
    - ``semantic_map_id`` is ``"semantic-map-<dataset>"``: the persistent semantic map of the
      dataset (``inputs.sequence``) the workspace files the run under. It is a function of
      the dataset alone, never of the build or run that asks, so repeating an execution
      repeats its content; which run materialized each version of the map is in the lineage
      of the ``ContextMap`` that cites it.

    Nothing is invented for what the run does not name: without a code identity the stages
    that record one are not composed, and preflight reports why
    (:class:`~contextmap.runtime.errors.ExecutorIdentityMissingError`).

    Args:
        effective: The resolved configuration, with the run's dataset.
        code_identity: The run's code identity, or ``None`` when it declares none.

    Returns:
        The keyword arguments of :func:`compose_executors` of the same names.

    Raises:
        ValueError: If ``code_identity`` is empty or blank: an empty identity is no identity.
    """
    from contextmap.semantic_mapping import SemanticMapId

    if code_identity is not None and not code_identity.strip():
        raise ValueError("a code identity must not be empty: pass None when there is none")
    dataset = effective.config.inputs.sequence
    return ExecutorIdentities(
        code_version=code_identity,
        code_digest=(
            None
            if code_identity is None
            else "sha256:" + hashlib.sha256(code_identity.encode("utf-8")).hexdigest()
        ),
        semantic_map_id=None if dataset is None else SemanticMapId(f"semantic-map-{dataset}"),
    )


FeatureFactory = Callable[["FeatureBuildScope"], "FeatureExtractor"]
GroundingFactory = Callable[[Path], "RegionGrounding"]
"""Builds a run's grounding backend from the directory holding its prepared images."""
RefinementFactory = Callable[[Path], "RegionRefinement"]
"""Builds a run's grounding refiner from the directory holding its prepared images."""
RegionDiscoveryFactory = Callable[[Path], "AuditedRegionDiscovery"]
"""Builds a run's region discovery backend from the directory holding its prepared images."""
SemanticInterpreterFactory = Callable[[Path], "SemanticInterpreter"]
"""Builds a run's semantic interpreter from the directory holding its semantic views."""
SourceAdapterFactory = Callable[["SourceAdapterConfig"], "SourceAdapter"]


@dataclass(frozen=True, kw_only=True)
class FeatureBuildScope:
    """Values a feature backend needs that exist only while one run is executing.

    Feature extractors write their payloads into the run that owns them, so they can only
    be constructed once that run exists. Everything here is execution state, not
    configuration.

    Attributes:
        run_id: Perception run that owns the produced features.
        feature_stage_id: Stage identity that namespaces the features inside the run.
        source_artifact_id: Run or artifact that owns the persisted payloads.
        payload_sink: Receives payload arrays before the run artifact is finalized.
        prepared_image_root: Directory that resolves prepared-image payload references.
        mask_source: Supplies region masks to a mask-conditioned backend.
        checkpoint_root: Directory that resolves local checkpoint paths.
    """

    run_id: VisualEvidenceRunId
    feature_stage_id: str
    source_artifact_id: str
    payload_sink: Any
    prepared_image_root: Path | None = None
    mask_source: Any = None
    checkpoint_root: Path | None = None


@dataclass(frozen=True, kw_only=True)
class RegionDiscoveryPlan:
    """The region discovery backend of a run and the identity it discovers under.

    Attributes:
        factory: Builds the backend once per run, from the directory holding that run's
            prepared images: the backend cuts each discovery pass from them and hands the
            model runtime its pixels (#661). The runtime was obtained once, at composition.
        backend: The provenance the built backend reports, known before any run: its
            configuration fingerprint covers the backend configuration, so the executor names
            it in the run's identity (#636).
        discovery_identity: The effective discovery identity (passes, normalization and
            algorithm versions, defaults resolved) the built backend reports, which the
            executor names in the run's identity too.
    """

    factory: RegionDiscoveryFactory
    backend: BackendProvenance
    discovery_identity: RegionDiscoveryIdentity


@dataclass(frozen=True, kw_only=True)
class RegionGroundingPlan:
    """The grounding backend of a run and the explicit queries it answers.

    The queries come from the backend's reserved ``query_set`` parameter group, never from
    the backend's own configuration, so they are part of the run's configuration identity
    but not of the backend fingerprint. The executor turns each (image, query) pair into
    one :class:`~contextmap.visual_perception.RegionGroundingRequest`.

    Attributes:
        queries: The queries, in configured order; unique and already validated against
            the backend's declared capabilities.
        factory: Builds the backend once per run, from the directory holding that run's
            prepared images; the bundled runtime loads its model lazily, once.
        backend: The provenance the built backend reports, known before any run: its
            configuration fingerprint resolves every default and parser version, so the
            executor names it in the run's identity (#636).
    """

    queries: tuple[GroundingQuery, ...]
    factory: GroundingFactory
    backend: BackendProvenance


@dataclass(frozen=True, kw_only=True)
class RegionRefinementPlan:
    """The refiner of a run's grounding proposals and the identity it applies.

    Attributes:
        factory: Builds the refiner once per run, from the directory holding that run's
            prepared images; the model runtime was obtained once, at composition.
        backend: The provenance the built refiner reports, known before any run: its
            configuration fingerprint resolves every default, so the executor names it in the
            run's identity next to the acceptance policy (#636).
    """

    factory: RefinementFactory
    backend: BackendProvenance


@dataclass(frozen=True, kw_only=True)
class ComposedSemanticScorer:
    """A scorer built over one perception run, and the one space every score of it lives in.

    Attributes:
        scorer: The region scorer, reading the feature payloads of that perception run.
        embedding_space: The space of the scorer's text tower. A region feature of another
            space is refused at the first request, never scored across spaces.
    """

    scorer: SemanticScorer
    embedding_space: EmbeddingSpace


SemanticScorerFactory = Callable[["FeatureStoreReader"], ComposedSemanticScorer]
"""Builds the scorer of one execution over the feature store of the perception run it scores."""


@dataclass(frozen=True, kw_only=True)
class SemanticScoringPlan:
    """The scorer of the optional ``semantic_scoring`` stage and what identifies it (#652).

    Attributes:
        factory: Builds the scorer over one pinned perception run's feature payloads. The
            text tower's runtime was obtained once, at composition; its model loads on first
            use, when the stage runs.
        backend: The provenance every score carries, known before any run: its fingerprint
            resolves every setting that changes a value (text template included), so the
            executor names it in the run's identity.
        text_template: How each hypothesis is rendered for the text tower, recorded in the
            scoring run.
    """

    factory: SemanticScorerFactory
    backend: BackendProvenance
    text_template: str


@dataclass(frozen=True, kw_only=True)
class ComposedRuntime:
    """The implementations built for one effective configuration.

    A field is ``None`` when its stage was not composed: not selected, not requested, or
    without an implemented capability (see ``unavailable_stages``).

    Attributes:
        effective: The configuration everything below was built from.
        stages: Identities of the stages that were composed, in topology order.
        unavailable_stages: Enabled stages skipped because their capability does not
            exist yet, with the reason. Nothing stands in for them.
        source_adapter: Builds the configured source adapter for one ingestion request.
        region_discovery: Builds the run's region discovery backend, which also reports each
            frame's audit, from the run's prepared images, and names its identity.
        region_grounding: Prompt-conditioned grounding backend and its queries, only when
            the optional ``region_perception.grounding`` component is selected.
        region_refinement: The run's refiner of grounding proposals and its identity, only
            when the optional ``region_perception.refinement`` component is selected
            (it requires grounding).
        dense_features: Builds the dense feature extractor once a run scope exists.
        region_features: Builds the region feature extractor once a run scope exists.
        semantic_interpreter: Builds the run's semantic interpretation backend from the
            directory holding the run's semantic views.
        semantic_scoring: The scorer of the optional ``semantic_scoring`` stage (#652), only
            when that stage is enabled.
        semantic_request_policy: The resolved request policy of the run (#544): the prompt of
            each interpreted mode (the configured ``prompt_policy`` of an instruction-following
            backend, or Florence-2's task-native prompt for the one mode its task serves), the
            configured ``view_policy`` and the scene-context switch, with one fingerprint.
        state_estimator: State estimation backend.
        geometric_mapping_pose_lookup: Pose lookup rule Geometric Mapping uses to place
            each scan.
        motion_correction: Disposition of scans that are not known to be corrected.
        geometric_mapping_deskew: The deskew policy that corrects each scan for platform motion
            before it is mapped, only when the optional ``geometric_mapping.deskew`` component
            is selected; ``None`` maps every scan as its source recorded it.
        geometric_mapping_aggregation: The intra-scan voxel rule, only when the optional
            ``geometric_mapping.aggregation`` component is selected; ``None`` keeps every raw point.
        point_encoder: Point encoder, only when the optional stage is selected.
        center_selection: The versioned policy that selects the centers the point encoder
            represents, only when the optional ``point_representation`` stage is selected (#663).
        support_policy: Semantic Fusion support policy.
        accumulation_policy: Semantic Fusion accumulation policy.
        association_candidates: Which map geometry each Sensor Association frame evaluates
            before projection; its range is ``None`` unless ``policies.association_max_range_m``
            sets one, which keeps the whole map evaluated exactly as before (#562).
        occlusion_policy: Sensor Association visibility rule.
        association_tolerances: Sensor Association diagnostic tolerances.
        association_pose_policy: Pose lookup rule Sensor Association uses per frame.
        association_dense_sampling: How Sensor Association reads every dense channel of the
            perception run, only when the optional ``sensor_association.dense_sampling``
            component is selected (#651); ``None`` keeps the geometry-only association.
        semantic_mapping_geometry_summary: Semantic Mapping's entity-geometry summary policy.
        entity_retrieval_policy: Entity Resolution candidate retrieval policy.
        entity_comparison_channels: The evidence channels Entity Resolution evaluates; geometry
            is always present, every other channel is ``None`` when not selected.
        entity_resolution_policy: Entity Resolution's conservative decision policy.
        entity_belief_policy: The minimum support of ``resolved-belief-min-support-v1``, only when
            the optional ``entity_resolution.belief`` component is selected; ``None`` derives no
            belief, and the resolution run is exactly the one without it.
        spatial_relations_policies: Spatial Relations' effective policies: frame conventions,
            candidate generation and the geometry-summary policy are always present, the
            predicate evaluators are ``None`` when not selected. ``geometry_summary`` here is
            the only source ``SpatialRelationsExecutor`` reads it from, so the policy it uses
            to summarize geometry and the one persisted in the run's own provenance can never
            diverge (review of PR #540, second round).
    """

    effective: EffectiveConfig
    stages: tuple[str, ...]
    unavailable_stages: Mapping[str, str] = field(default_factory=dict)
    source_adapter: SourceAdapterFactory | None = None
    region_discovery: RegionDiscoveryPlan | None = None
    region_grounding: RegionGroundingPlan | None = None
    region_refinement: RegionRefinementPlan | None = None
    dense_features: FeatureFactory | None = None
    region_features: FeatureFactory | None = None
    semantic_interpreter: SemanticInterpreterFactory | None = None
    semantic_request_policy: SemanticRequestPolicy | None = None
    semantic_scoring: SemanticScoringPlan | None = None
    state_estimator: StateEstimator | None = None
    geometric_mapping_pose_lookup: LookupPolicy | None = None
    motion_correction: MotionCorrectionPolicy | None = None
    geometric_mapping_deskew: PoseInterpolationDeskewPolicy | None = None
    geometric_mapping_aggregation: ScanVoxelPolicy | None = None
    point_encoder: PointEncoder | None = None
    center_selection: ObservationSupportAnchorPolicy | None = None
    support_policy: GeometryOverlapSupportPolicy | None = None
    accumulation_policy: BaselineAccumulationPolicy | QualityAwareAccumulationPolicy | None = None
    occlusion_policy: OcclusionPolicy | None = None
    association_tolerances: DiagnosticTolerances | None = None
    association_pose_policy: LookupPolicy | None = None
    association_candidates: CandidateGeometryPolicy | None = None
    association_dense_sampling: InterpolationPolicy | None = None
    association_box_policy: BoxDepthComponentPolicy | None = None
    semantic_mapping_geometry_summary: GeometrySummaryPolicy | None = None
    entity_retrieval_policy: CandidateRetrievalPolicy | None = None
    entity_comparison_channels: ComparisonChannels | None = None
    entity_resolution_policy: ConservativeResolutionPolicy | None = None
    entity_belief_policy: ResolvedBeliefPolicy | None = None
    spatial_relations_policies: RelationsRunPolicies | None = None


def compose(
    effective: EffectiveConfig,
    *,
    providers: Mapping[str, RuntimeProvider] | None = None,
    stages: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    on_provider_override: Callable[[str], None] | None = None,
) -> ComposedRuntime:
    """Construct the implementations an effective configuration selects.

    Args:
        effective: The resolved configuration.
        providers: Model runtimes or clients supplied by the caller, keyed by component
            identity (``"<capability>.<slot>"``), for backends without a bundled loader
            and to override a bundled one. For a component this leaves unset, a
            ``resources.providers`` target declared in ``effective`` is resolved instead,
            lazily, only if that component is actually being composed; see
            :func:`resolve_provider`.
        stages: The stages to compose, or ``None`` for every enabled stage whose
            capability is implemented.
        environ: Environment to read secrets from; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed;
            defaults to an :mod:`importlib` lookup.
        on_provider_override: Called with a component identity when ``providers`` supplies
            an entry for it that wins over a ``resources.providers`` target ``effective``
            also declares for the same component -- the only case where a caller-embedded
            provider silently takes precedence over what the configuration itself names.
            Never called when the two never disagree, in particular on every ordinary run
            through the installed CLI, which never passes ``providers``.

    Returns:
        The composed implementations. No model has been loaded.

    Raises:
        ConfigurationError: If a requested stage is unknown or disabled, or a composed
            stage has a variation point with no backend selected.
        StageUnavailableError: If a requested stage has no implemented capability.
        BackendConfigurationError: If a capability rejects its backend's parameters.
        BackendUnavailableError: If a backend needs a module or secret that is missing.
        BackendRuntimeMissingError: If a backend needs a runtime nobody supplied and the
            configuration declares no ``resources.providers`` target for it either.
        ProviderConfigurationError: If a ``resources.providers`` target this component needs
            is malformed or cannot be resolved into a callable.
    """
    preset = PRESETS[effective.config.pipeline.preset]
    enabled = effective.config.pipeline.stages
    unavailable: dict[str, str] = {}
    selected: list[StageDeclaration]
    if stages is None:
        selected = []
        for stage in preset.stages:
            if not enabled.get(stage.stage_id, False):
                continue
            if stage.available:
                selected.append(stage)
            else:
                unavailable[stage.stage_id] = stage.unavailable_reason
    else:
        requested = list(stages)
        known = {stage.stage_id for stage in preset.stages}
        for stage_id in requested:
            if stage_id not in known:
                raise ConfigurationError.single(
                    f"unknown stage {stage_id!r}; known stages: {', '.join(sorted(known))}",
                    path="stages",
                )
            declaration = preset.stage(stage_id)
            if not declaration.available:
                raise StageUnavailableError(stage_id, declaration.unavailable_reason)
            if not enabled.get(stage_id, False):
                raise ConfigurationError.single(
                    f"stage {stage_id!r} is disabled in the configuration", path="stages"
                )
        selected = [stage for stage in preset.stages if stage.stage_id in requested]

    _require_selection(effective, selected)
    _require_canonical2_selection(effective, selected)
    context = _Context(
        effective=effective,
        providers=dict(providers or {}),
        environ=os.environ if environ is None else environ,
        module_available=module_available,
        on_provider_override=on_provider_override,
    )
    attributes: dict[str, object] = {}
    for stage in selected:
        attributes.update(_STAGE_COMPOSERS[stage.stage_id](context))
    return ComposedRuntime(
        effective=effective,
        stages=tuple(stage.stage_id for stage in selected),
        unavailable_stages=unavailable,
        **cast("dict[str, Any]", attributes),
    )


def composable_backends() -> frozenset[tuple[str, str]]:
    """List every ``(component_id, backend_id)`` this composition root can construct.

    Returns:
        The pairs with a factory. The catalog and this table must agree; a test enforces it.
    """
    return frozenset(
        (component_id, backend_id)
        for component_id, factories in _FACTORIES.items()
        for backend_id in factories
    )


def _require_selection(effective: EffectiveConfig, selected: list[StageDeclaration]) -> None:
    """Fail when a composed stage has a variation point with no backend."""
    owned = {component_id for stage in selected for component_id in stage.components}
    problems = [
        problem
        for problem in check_selection(effective.config)
        if problem.path.removeprefix("components.") in owned
    ]
    if problems:
        raise ConfigurationError(problems)


def _require_canonical2_selection(
    effective: EffectiveConfig, selected: Collection[StageDeclaration]
) -> None:
    """Enforce the frozen discovery choice of the canonical/2 experiment.

    The catalog intentionally keeps SAM3 and Florence-2 selectable for experiments.
    The first context-rich canonical profile is narrower: allowing another backend here
    would make its artifact identity claim a SAM2-first baseline it did not execute.
    """
    if effective.config.pipeline.preset != CANONICAL2_PROFILE_ID:
        return
    if any(stage.stage_id == "region_perception" for stage in selected):
        # Anotado porque o segundo bloco reusa o nome com mais requisitos: sem o tipo
        # aberto, a inferência do primeiro fixaria a aridade da tupla.
        requirements: tuple[tuple[str, str, str], ...] = (
            ("region_perception.discovery", "sam2", "sam2 automatic region discovery"),
        )
        for component_id, backend_id, purpose in requirements:
            selected_backend = effective.config.components[component_id].backend
            if selected_backend != backend_id:
                raise BackendConfigurationError(
                    component_id,
                    selected_backend or "",
                    [f"canonical/2 requires {purpose} ({backend_id})"],
                )
    if any(stage.stage_id == "visual_perception" for stage in selected):
        requirements = (
            ("visual_features.dense", "dinov2", "DINOv2 dense appearance features"),
            ("visual_features.region", "clip", "CLIP region appearance features"),
            ("semantic_interpretation.interpreter", "qwen", "Qwen region interpretation"),
        )
        for component_id, backend_id, purpose in requirements:
            selected_backend = effective.config.components[component_id].backend
            if selected_backend != backend_id:
                raise BackendConfigurationError(
                    component_id,
                    selected_backend or "",
                    [f"canonical/2 requires {purpose} ({backend_id})"],
                )
    if any(stage.stage_id == "semantic_scoring" for stage in selected):
        scorer = effective.config.components["semantic_scoring.scorer"].backend
        if scorer != "clip":
            raise BackendConfigurationError(
                "semantic_scoring.scorer",
                scorer or "",
                ["canonical/2 requires CLIP semantic scoring (clip)"],
            )


@dataclass(frozen=True, kw_only=True)
class _Context:
    """What one composition needs besides the configuration: providers and environment."""

    effective: EffectiveConfig
    providers: Mapping[str, RuntimeProvider]
    environ: Mapping[str, str]
    module_available: Callable[[str], bool] | None
    on_provider_override: Callable[[str], None] | None = None

    def component(self, component_id: str) -> ComponentConfig:
        return self.effective.config.components[component_id]

    def build(
        self,
        component_id: str,
        config_type: type[Any],
        *,
        extras: Mapping[str, type[Any]] | None = None,
        required: Collection[str] = (),
        fixed: Mapping[str, object] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Validate a backend's parameters through the capability's own configuration.

        ``extras`` are reserved parameter groups that build a second object next to the
        configuration (for example a support policy); ``fixed`` are values the slot itself
        decides and a user cannot contradict.
        """
        component = self.component(component_id)
        backend = component.backend or ""
        parameters: dict[str, Any] = dict(component.parameters)
        problems: list[str] = []
        extra_values: dict[str, Any] = {}
        for name, extra_type in (extras or {}).items():
            if name not in parameters:
                if name in required:
                    problems.append(f"missing required parameter: {name}")
                continue
            try:
                extra_values[name] = build_config(extra_type, parameters.pop(name))
            except ParameterError as error:
                problems.extend(f"{name}: {problem}" for problem in error.problems)
        for name, value in (fixed or {}).items():
            if name in parameters and parameters[name] != value:
                problems.append(f"{name}: fixed to {value!r} by the {component_id} slot")
            parameters[name] = value
        config: Any = None
        try:
            config = build_config(config_type, parameters)
        except ParameterError as error:
            problems.extend(error.problems)
        if problems:
            raise BackendConfigurationError(component_id, backend, problems)
        return config, extra_values

    def ensure_available(self, component_id: str) -> None:
        """Fail when a bundled code path needs a module or secret that is missing.

        A component whose runtime a provider supplies -- explicitly, or through a declared
        ``resources.providers`` target -- never needs its own bundled module: that heavy
        import is the provider's business, never ``contextmap.runtime``'s (see the module
        docstring). Only the module check is skipped this way; a missing secret is still
        reported, since a provider is not necessarily what reads it.
        """
        problems = check_component_availability(
            component_id,
            self.component(component_id),
            environ=self.environ,
            module_available=self.module_available,
            check_modules=component_id not in provided_runtimes(self.effective, self.providers),
        )
        if problems:
            raise BackendUnavailableError(problems)

    def secrets(self, component_id: str) -> ResolvedSecrets:
        """Hold only the secrets the selected backend declares."""
        backend = self.component(component_id).backend or ""
        names = COMPONENTS[component_id].backends[backend].secrets
        return ResolvedSecrets(names, {n: self.environ[n] for n in names if self.environ.get(n)})

    def _declared_provider_target(self, component_id: str) -> str | None:
        """Read the ``resources.providers`` target the configuration declares, if any."""
        return self.effective.config.resources.providers.get(component_id)

    def _resolved_provider(self, component_id: str) -> RuntimeProvider | None:
        """Resolve this component's provider, or ``None`` when nobody supplies one.

        Precedence: an explicit ``providers=`` entry always wins, even over a
        ``resources.providers`` target the configuration declares for the same component --
        and that override is reported through ``on_provider_override`` exactly when it
        actually happens (both exist for the same component). Otherwise, a declared target
        is resolved lazily, right here, only because this component is genuinely being
        composed; the target is never resolved for a component nobody asked to build.
        """
        explicit = self.providers.get(component_id)
        target = self._declared_provider_target(component_id)
        if explicit is not None:
            if target is not None and self.on_provider_override is not None:
                self.on_provider_override(component_id)
            return explicit
        if target is not None:
            return resolve_provider(component_id, target)
        return None

    def runtime(self, component_id: str, config: Any, protocol: str) -> Any:
        """Ask the caller's or the configuration's provider for a backend's runtime.

        Raises:
            BackendRuntimeMissingError: If neither an explicit provider nor a declared
                ``resources.providers`` target supplies one.
            ProviderConfigurationError: If a declared target cannot be resolved.
        """
        provider = self._resolved_provider(component_id)
        if provider is None:
            backend = self.component(component_id).backend or ""
            raise BackendRuntimeMissingError(component_id, backend, protocol)
        return provider(config, self.secrets(component_id))

    def optional_runtime(self, component_id: str, config: Any) -> Any:
        """Return the resolved runtime for a backend that also bundles a loader, or ``None``."""
        provider = self._resolved_provider(component_id)
        return None if provider is None else provider(config, self.secrets(component_id))


Factory = Callable[[_Context, str], Any]


# --- ingestion --------------------------------------------------------------------


def _source_adapter(context: _Context, component_id: str) -> SourceAdapterFactory:
    component = context.component(component_id)
    backend = component.backend or ""
    if component.parameters:
        raise BackendConfigurationError(
            component_id,
            backend,
            [
                f"takes no parameters; the source path and topics come with each request: "
                f"{', '.join(sorted(component.parameters))}"
            ],
        )
    context.ensure_available(component_id)

    def build(request: SourceAdapterConfig) -> SourceAdapter:
        if request.source_type != backend:
            raise BackendConfigurationError(
                component_id,
                backend,
                [
                    f"the request asks for source_type {request.source_type!r}; the configuration "
                    f"selects {backend!r} and never substitutes another adapter"
                ],
            )
        if backend == "ros1_bag":
            from contextmap.ingestion.adapters.ros.ros1_bag_source_adapter import (
                Ros1BagSourceAdapter,
            )

            return Ros1BagSourceAdapter(request)
        if backend == "pose_file":
            from contextmap.ingestion.adapters.pose_file_source_adapter import PoseFileSourceAdapter

            return PoseFileSourceAdapter(request)
        from contextmap.ingestion.adapters.ros.ros2_bag_source_adapter import Ros2BagSourceAdapter

        return Ros2BagSourceAdapter(request)

    return build


# --- visual perception: region discovery -------------------------------------------


def _region_extras() -> dict[str, type[Any]]:
    from contextmap.region_perception import (
        DiscoveryPassConfig,
        NormalizationConfig,
    )

    return {"pass_config": DiscoveryPassConfig, "normalization_config": NormalizationConfig}


def _pass_images(prepared_image_root: Path) -> Callable[[DiscoveryInput], object]:
    """Materialize each discovery pass from the prepared images of one run (#661).

    The model runtime a provider supplies never sees the run's directory: the backend built for
    the run hands it the pixels this returns.
    """
    from contextmap.region_perception.adapters.prepared_images import materialize_discovery_pass

    return partial(materialize_discovery_pass, prepared_image_root)


def _region_discovery_plan(
    context: _Context,
    component_id: str,
    factory: RegionDiscoveryFactory,
    backend: BackendProvenance,
    extras: Mapping[str, Any],
) -> RegionDiscoveryPlan:
    """Name the identity a run-scoped discovery backend will report, before any run.

    Raises:
        BackendConfigurationError: If the pass plan has a pass the adapter cannot materialize.
    """
    from contextmap.region_perception import (
        RegionDiscoveryIdentity,
    )
    from contextmap.region_perception.adapters.prepared_images import require_unscaled_passes

    try:
        require_unscaled_passes(extras.get("pass_config"))
    except ValueError as error:
        selected = context.component(component_id).backend or ""
        raise BackendConfigurationError(
            component_id, selected, [f"pass_config: {error}"]
        ) from error
    return RegionDiscoveryPlan(
        factory=factory,
        backend=backend,
        discovery_identity=RegionDiscoveryIdentity.effective(
            extras.get("pass_config"), extras.get("normalization_config")
        ),
    )


def _sam2(context: _Context, component_id: str) -> RegionDiscoveryPlan:
    from contextmap.region_perception.adapters.sam2 import Sam2Config, Sam2RegionDiscovery

    config, extras = context.build(component_id, Sam2Config, extras=_region_extras())
    if context.effective.config.pipeline.preset == CANONICAL2_PROFILE_ID:
        pass_config = extras["pass_config"]
        if (
            not pass_config.include_full_frame
            or pass_config.tiling is not None
            or pass_config.additional_tilings
        ):
            raise BackendConfigurationError(
                component_id,
                "sam2",
                [
                    "canonical/2 requires one full-frame SAM2 pass: "
                    "include_full_frame=true, tiling=null, additional_tilings=()"
                ],
            )
    context.ensure_available(component_id)
    # Um runtime (um carregamento do modelo) por composição serve todo run: ele recebe os
    # pixels de cada pass, nunca a raiz das imagens preparadas, que só existe no run.
    runtime = context.runtime(component_id, config, "Sam2Runtime")

    def build(prepared_image_root: Path) -> AuditedRegionDiscovery:
        return Sam2RegionDiscovery(
            config=config, runtime=runtime, pass_images=_pass_images(prepared_image_root), **extras
        )

    return _region_discovery_plan(
        context, component_id, build, Sam2RegionDiscovery.provenance_for(config), extras
    )


def _sam3(context: _Context, component_id: str) -> RegionDiscoveryPlan:
    from contextmap.region_perception.adapters.sam3 import Sam3Config, Sam3RegionDiscovery

    config, extras = context.build(component_id, Sam3Config, extras=_region_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Sam3Runtime")

    def build(prepared_image_root: Path) -> AuditedRegionDiscovery:
        return Sam3RegionDiscovery(
            config=config, runtime=runtime, pass_images=_pass_images(prepared_image_root), **extras
        )

    return _region_discovery_plan(
        context, component_id, build, Sam3RegionDiscovery.provenance_for(config), extras
    )


def _florence2_regions(context: _Context, component_id: str) -> RegionDiscoveryPlan:
    from contextmap.region_perception.adapters.florence2 import (
        Florence2Config,
        Florence2RegionDiscovery,
    )

    config, extras = context.build(component_id, Florence2Config, extras=_region_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Florence2Runtime")

    def build(prepared_image_root: Path) -> AuditedRegionDiscovery:
        return Florence2RegionDiscovery(
            config=config, runtime=runtime, pass_images=_pass_images(prepared_image_root), **extras
        )

    return _region_discovery_plan(
        context, component_id, build, Florence2RegionDiscovery.provenance_for(config), extras
    )


# --- visual perception: region grounding (run-scoped) ------------------------------


def _locateanything(context: _Context, component_id: str) -> RegionGroundingPlan:
    from contextmap.region_perception import (
        GroundingQuerySet,
        GroundingRequestError,
    )
    from contextmap.region_perception.adapters.locateanything import (
        LocateAnythingConfig,
        LocateAnythingRegionGrounding,
        TransformersLocateAnythingRuntime,
        validate_locateanything_query,
    )

    config, extras = context.build(
        component_id,
        LocateAnythingConfig,
        extras={"query_set": GroundingQuerySet},
        required=("query_set",),
    )
    queries: tuple[GroundingQuery, ...] = extras["query_set"].queries
    problems = []
    for index, query in enumerate(queries):
        try:
            validate_locateanything_query(query)
        except GroundingRequestError as error:
            problems.append(f"query_set.queries[{index}]: {error}")
    if problems:
        raise BackendConfigurationError(component_id, "locateanything", problems)
    context.ensure_available(component_id)

    def build(prepared_image_root: Path) -> RegionGrounding:
        runtime = context.optional_runtime(component_id, config)
        if runtime is None:
            runtime = TransformersLocateAnythingRuntime(
                config=config, prepared_image_root=prepared_image_root
            )
        return LocateAnythingRegionGrounding(config=config, runtime=runtime)

    return RegionGroundingPlan(
        queries=queries,
        factory=build,
        backend=LocateAnythingRegionGrounding.provenance_for(config),
    )


def _sam2_refinement(context: _Context, component_id: str) -> RegionRefinementPlan:
    from contextmap.region_perception.adapters.sam2 import (
        Sam2PromptRefinement,
        Sam2RefinementConfig,
    )

    config, _ = context.build(component_id, Sam2RefinementConfig)
    context.ensure_available(component_id)
    # O runtime recebe os bytes da imagem, não a raiz das imagens preparadas: um único runtime
    # (um carregamento do SAM2) por composição serve o run inteiro.
    runtime = context.runtime(component_id, config, "Sam2PromptRuntime")

    def build(prepared_image_root: Path) -> RegionRefinement:
        return Sam2PromptRefinement(
            config=config, runtime=runtime, prepared_image_root=prepared_image_root
        )

    return RegionRefinementPlan(factory=build, backend=Sam2PromptRefinement.provenance_for(config))


# --- visual perception: features (run-scoped) --------------------------------------


def _dinov2(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.visual_features.adapters.dinov2 import (
        DinoV2Config,
        DinoV2DenseFeatureBackend,
    )

    config, _ = context.build(component_id, DinoV2Config)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return DinoV2DenseFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _dinov3(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.visual_features.adapters.dinov3 import (
        DinoV3Config,
        DinoV3DenseFeatureBackend,
    )

    config, _ = context.build(component_id, DinoV3Config)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return DinoV3DenseFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _siglip2(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.visual_features.adapters.siglip2 import (
        Siglip2Config,
        Siglip2FeatureBackend,
    )

    # O slot de features densas fixa o escopo: um SigLIP2 GLOBAL não produz DenseFeatureMap.
    config, _ = context.build(component_id, Siglip2Config, fixed={"scope": "dense"})
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return Siglip2FeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            source_artifact_id=scope.source_artifact_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _clip(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.visual_features.adapters.clip import (
        ClipConfig,
        ClipVisualFeatureBackend,
    )

    # O slot de features por região fixa o escopo: não é uma escolha do usuário.
    config, _ = context.build(component_id, ClipConfig, fixed={"scope": "region"})
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        return ClipVisualFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
        )

    return build


def _alphaclip(context: _Context, component_id: str) -> FeatureFactory:
    from contextmap.visual_perception.visual_features.adapters.alphaclip import (
        AlphaClipConfig,
        AlphaClipRegionFeatureBackend,
    )

    config, _ = context.build(component_id, AlphaClipConfig)
    context.ensure_available(component_id)

    def build(scope: FeatureBuildScope) -> FeatureExtractor:
        if scope.mask_source is None:
            raise BackendConfigurationError(
                component_id,
                "alphaclip",
                ["needs a mask_source in the FeatureBuildScope: it is conditioned on region masks"],
            )
        return AlphaClipRegionFeatureBackend(
            config=config,
            run_id=scope.run_id,
            feature_stage_id=scope.feature_stage_id,
            mask_source=scope.mask_source,
            payload_sink=scope.payload_sink,
            runtime=context.optional_runtime(component_id, config),
            prepared_image_root=scope.prepared_image_root,
            checkpoint_root=scope.checkpoint_root,
        )

    return build


# --- visual perception: semantic interpretation ------------------------------------


@dataclass(frozen=True, kw_only=True)
class _SemanticInterpretation:
    """A run-scoped semantic interpreter with the resolved request policy its requests follow."""

    interpreter: SemanticInterpreterFactory
    request_policy: SemanticRequestPolicy


def _semantic_interpretation(
    context: _Context,
    component_id: str,
    *,
    interpreter: SemanticInterpreterFactory,
    capabilities: SemanticInterpreterCapabilities,
    request_policy: SemanticRequestPolicy,
) -> _SemanticInterpretation:
    """Pair an interpreter with its request policy once it declares it can consume it.

    Args:
        context: The composition.
        component_id: The semantic interpretation slot.
        interpreter: Builds the interpreter of one run from the directory of its views.
        capabilities: What that interpreter declares, known from its configuration alone.
        request_policy: The resolved request policy of the run.

    Raises:
        BackendConfigurationError: If the interpreter's capabilities refuse a prompted mode, a
            view kind or number of views, or scene-context conditioning. The check reads only
            the declaration, so no model is loaded for it.
    """
    from contextmap.semantic_interpretation import (
        check_request_policy_supported,
    )

    try:
        check_request_policy_supported(request_policy, capabilities)
    except ValueError as error:
        backend = context.component(component_id).backend or ""
        raise BackendConfigurationError(
            component_id, backend, [f"semantic request policy: {error}"]
        ) from error
    return _SemanticInterpretation(interpreter=interpreter, request_policy=request_policy)


_INSTRUCTION_FOLLOWING = frozenset({"qwen", "gemini", "eagle2_5"})
"""Semantic backends that render a configured free-form ``prompt_policy``."""

_SEMANTIC_COMPONENT = "semantic_interpretation.interpreter"


def resolve_semantic_request_policy(effective: EffectiveConfig) -> SemanticRequestPolicy:
    """Resolve the semantic request policy an effective configuration selects (#544).

    Reads only the configuration: no provider is called, no model or backend is built, so an
    experiment manifest can name an arm's treatment by
    :meth:`~contextmap.visual_perception.SemanticRequestPolicy.fingerprint` before anything
    runs. :func:`compose` composes exactly this policy and additionally checks it against the
    interpreter's declared capabilities.

    Args:
        effective: The resolved configuration.

    Returns:
        The prompt of each interpreted mode, the ordered views and the scene-context switch.

    Raises:
        ConfigurationError: If no semantic interpretation backend is selected.
        BackendConfigurationError: If the backend's ``prompt_policy``/``view_policy`` groups
            are missing, invalid or not accepted by that backend.
    """
    component = effective.config.components.get(_SEMANTIC_COMPONENT)
    if component is None or component.backend is None:
        raise ConfigurationError.single(
            "no semantic interpretation backend is selected, so there is no semantic request "
            "policy to resolve",
            path=f"components.{_SEMANTIC_COMPONENT}",
        )
    return _request_policy(_SEMANTIC_COMPONENT, component)


def _request_policy(component_id: str, component: ComponentConfig) -> SemanticRequestPolicy:
    """Build the request policy from a semantic backend's reserved groups, and nothing else."""
    from contextmap.semantic_interpretation import (
        RegionRequestBudget,
        SemanticInterpretationMode,
        SemanticModePrompt,
        SemanticPromptPolicy,
        SemanticRequestPolicy,
        SemanticViewPolicy,
    )

    backend = component.backend or ""
    parameters = dict(component.parameters)
    problems: list[str] = []
    groups: dict[str, Any] = {}
    budget: RegionRequestBudget | None = None
    wanted: dict[str, type[Any]] = {"view_policy": SemanticViewPolicy}
    if backend in _INSTRUCTION_FOLLOWING:
        wanted["prompt_policy"] = SemanticPromptPolicy
    elif "prompt_policy" in parameters:
        # Florence-2 consome o token da própria task; aceitar um prompt livre aqui seria
        # registrá-lo como se tivesse sido consumido, então a limitação é reportada.
        from contextmap.semantic_interpretation.adapters import (
            florence2_semantic,
        )

        task_prompt_policy = florence2_semantic.TASK_PROMPT_POLICY

        problems.append(
            f"prompt_policy: Florence-2 consumes only its task-native prompt policy "
            f"({task_prompt_policy}:<task>); select the task instead"
        )
    for name, group_type in wanted.items():
        if name not in parameters:
            problems.append(f"missing required parameter: {name}")
            continue
        try:
            groups[name] = build_config(group_type, parameters[name])
        except ParameterError as error:
            problems.extend(f"{name}: {problem}" for problem in error.problems)
    if backend in _INSTRUCTION_FOLLOWING and "region_budget" in parameters:
        try:
            budget = build_config(RegionRequestBudget, parameters["region_budget"])
        except ParameterError as error:
            problems.extend(f"region_budget: {problem}" for problem in error.problems)
    if problems:
        raise BackendConfigurationError(component_id, backend, problems)
    if backend in _INSTRUCTION_FOLLOWING:
        policy = SemanticRequestPolicy.from_prompt_policy(
            groups["prompt_policy"], views=groups["view_policy"]
        )
        return policy if budget is None else dataclasses.replace(policy, region_budget=budget)
    from contextmap.semantic_interpretation.adapters.florence2_semantic import (
        TASK_OUTPUT_SCHEMA,
        Florence2SemanticConfig,
        task_prompt_template_id,
    )

    # A task (e o modo que ela serve) é validada pela configuração do próprio Florence-2; o
    # prompt nativo da task vale só para esse modo, e o outro fica sem prompt.
    own = {name: value for name, value in parameters.items() if name not in wanted}
    try:
        config = build_config(Florence2SemanticConfig, own)
    except ParameterError as error:
        raise BackendConfigurationError(component_id, backend, list(error.problems)) from error
    prompt = SemanticModePrompt(
        template_id=task_prompt_template_id(config.task), output_schema=TASK_OUTPUT_SCHEMA
    )
    modes = config.supported_modes
    return SemanticRequestPolicy(
        views=groups["view_policy"],
        scene=prompt if SemanticInterpretationMode.SCENE in modes else None,
        region=prompt if SemanticInterpretationMode.REGION in modes else None,
    )


def _instruction_extras() -> dict[str, type[Any]]:
    """Reserved groups of an instruction-following interpreter.

    ``prompt_policy`` and ``view_policy`` are always required; ``region_budget`` is optional
    and, when present, limits each frame's region requests (see :func:`_request_policy`).
    """
    from contextmap.semantic_interpretation import (
        RegionRequestBudget,
        SemanticPromptPolicy,
        SemanticViewPolicy,
    )

    return {
        "prompt_policy": SemanticPromptPolicy,
        "view_policy": SemanticViewPolicy,
        "region_budget": RegionRequestBudget,
    }


def _qwen(context: _Context, component_id: str) -> _SemanticInterpretation:
    from contextmap.semantic_interpretation.adapters.qwen import (
        QwenSemanticConfig,
        QwenSemanticInterpreter,
    )

    policy = _request_policy(component_id, context.component(component_id))
    config, _ = context.build(component_id, QwenSemanticConfig, extras=_instruction_extras())
    context.ensure_available(component_id)
    # Um runtime por composição serve todo run: ele recebe os bytes verificados de cada view,
    # nunca a raiz das views, que só existe no run.
    runtime = context.runtime(component_id, config, "QwenRuntime")

    def build(view_root: Path) -> SemanticInterpreter:
        return QwenSemanticInterpreter(config=config, runtime=runtime, view_root=view_root)

    return _semantic_interpretation(
        context,
        component_id,
        interpreter=build,
        capabilities=QwenSemanticInterpreter.capabilities_for(config),
        request_policy=policy,
    )


def _gemini(context: _Context, component_id: str) -> _SemanticInterpretation:
    from contextmap.semantic_interpretation.adapters.gemini import (
        GeminiSemanticConfig,
        GeminiSemanticInterpreter,
    )

    policy = _request_policy(component_id, context.component(component_id))
    config, _ = context.build(component_id, GeminiSemanticConfig, extras=_instruction_extras())
    context.ensure_available(component_id)
    client = context.runtime(component_id, config, "GeminiClient")

    def build(view_root: Path) -> SemanticInterpreter:
        return GeminiSemanticInterpreter(config=config, client=client, view_root=view_root)

    return _semantic_interpretation(
        context,
        component_id,
        interpreter=build,
        capabilities=GeminiSemanticInterpreter.capabilities_for(config),
        request_policy=policy,
    )


def _eagle2_5(context: _Context, component_id: str) -> _SemanticInterpretation:
    from contextmap.semantic_interpretation.adapters.eagle2_5 import (
        EagleSemanticConfig,
        EagleSemanticInterpreter,
    )

    policy = _request_policy(component_id, context.component(component_id))
    config, _ = context.build(component_id, EagleSemanticConfig, extras=_instruction_extras())
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "EagleRuntime")

    def build(view_root: Path) -> SemanticInterpreter:
        return EagleSemanticInterpreter(config=config, runtime=runtime, view_root=view_root)

    return _semantic_interpretation(
        context,
        component_id,
        interpreter=build,
        capabilities=EagleSemanticInterpreter.capabilities_for(config),
        request_policy=policy,
    )


def _florence2_semantic(context: _Context, component_id: str) -> _SemanticInterpretation:
    from contextmap.semantic_interpretation import (
        SemanticViewPolicy,
    )
    from contextmap.semantic_interpretation.adapters.florence2_semantic import (
        Florence2SemanticConfig,
        Florence2SemanticInterpreter,
    )

    policy = _request_policy(component_id, context.component(component_id))
    config, _ = context.build(
        component_id, Florence2SemanticConfig, extras={"view_policy": SemanticViewPolicy}
    )
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "Florence2SemanticRuntime")

    def build(view_root: Path) -> SemanticInterpreter:
        return Florence2SemanticInterpreter(config=config, runtime=runtime, view_root=view_root)

    return _semantic_interpretation(
        context,
        component_id,
        interpreter=build,
        capabilities=Florence2SemanticInterpreter.capabilities_for(config),
        request_policy=policy,
    )


# --- visual perception: semantic scoring (over a pinned perception run) -------------


def _scoring_plan(
    *,
    encoder: Any,
    scorer_type: Callable[..., SemanticScorer],
    backend: BackendProvenance,
    text_template: str,
) -> SemanticScoringPlan:
    """Plan a region scorer that compares in exactly the space its text tower declares.

    The encoder loads nothing here: its space, which needs the tower's projection dimension, is
    asked for only when a run builds the scorer over its perception run's feature store.
    """
    from contextmap.visual_perception import embedding_space_fingerprint

    def build(payloads: FeatureStoreReader) -> ComposedSemanticScorer:
        space = encoder.embedding_space()
        scorer = scorer_type(
            provenance=backend,
            embedding_space_id=embedding_space_fingerprint(space),
            payloads=payloads,
            text_encoder=encoder,
        )
        return ComposedSemanticScorer(scorer=scorer, embedding_space=space)

    return SemanticScoringPlan(factory=build, backend=backend, text_template=text_template)


def _clip_scorer(context: _Context, component_id: str) -> SemanticScoringPlan:
    from contextmap.semantic_scoring.adapters import ClipRegionSemanticScorer
    from contextmap.semantic_scoring.adapters.clip_text import (
        ClipTextConfig,
        ClipTextEncoder,
        HuggingFaceClipTextRuntime,
        clip_region_scorer_provenance,
    )

    config, _ = context.build(component_id, ClipTextConfig)
    context.ensure_available(component_id)
    runtime = context.optional_runtime(component_id, config)
    if runtime is None:
        runtime = HuggingFaceClipTextRuntime(config=config)
    return _scoring_plan(
        encoder=ClipTextEncoder(config=config, runtime=runtime),
        scorer_type=ClipRegionSemanticScorer,
        backend=clip_region_scorer_provenance(config),
        text_template=config.text_template,
    )


def _alphaclip_scorer(context: _Context, component_id: str) -> SemanticScoringPlan:
    from contextmap.semantic_scoring.adapters import AlphaClipSemanticScorer
    from contextmap.semantic_scoring.adapters.alphaclip_text import (
        AlphaClipTextConfig,
        AlphaClipTextEncoder,
        alphaclip_region_scorer_provenance,
    )

    config, _ = context.build(component_id, AlphaClipTextConfig)
    context.ensure_available(component_id)
    # Como a feature AlphaCLIP, a torre de texto lê checkpoints locais numa raiz que a
    # configuração não declara (#71): o runtime vem do chamador, nunca de um loader implícito.
    runtime = context.runtime(component_id, config, "AlphaClipTextRuntime")
    return _scoring_plan(
        encoder=AlphaClipTextEncoder(config=config, runtime=runtime),
        scorer_type=AlphaClipSemanticScorer,
        backend=alphaclip_region_scorer_provenance(config),
        text_template=config.text_template,
    )


# --- state estimation --------------------------------------------------------------


def _external_pose(context: _Context, component_id: str) -> StateEstimator:
    from contextmap.state_estimation.backends.external_pose import (
        ExternalPoseConfig,
        ExternalPoseEstimator,
    )

    config, _ = context.build(component_id, ExternalPoseConfig)
    context.ensure_available(component_id)
    return ExternalPoseEstimator(config)


@dataclass(frozen=True, kw_only=True)
class _NoParameters:
    """A backend whose configuration is made only of reserved parameter groups."""


@dataclass(frozen=True, kw_only=True)
class _FastLioRunnerParameters:
    """Parameters of the bundled subprocess runner, as written in configuration."""

    command: tuple[str, ...]
    timeout_s: float
    work_root: Path | None = None


def _fast_lio(context: _Context, component_id: str) -> StateEstimator:
    from contextmap.state_estimation.backends.fast_lio import FastLioConfig, FastLioEstimator
    from contextmap.state_estimation.backends.fast_lio_process import SubprocessFastLioRunner

    supplied = component_id in context.providers
    config, extras = context.build(
        component_id,
        FastLioConfig,
        extras={"runner": _FastLioRunnerParameters},
        required=() if supplied else ("runner",),
    )
    context.ensure_available(component_id)
    if supplied:
        runner = context.runtime(component_id, config, "FastLioRunner")
    else:
        parameters = extras["runner"]
        try:
            runner = SubprocessFastLioRunner(
                command=list(parameters.command),
                timeout_s=parameters.timeout_s,
                work_root=parameters.work_root,
            )
        except ValueError as error:
            raise BackendConfigurationError(
                component_id, "fast_lio", [f"runner: {error}"]
            ) from error
    return FastLioEstimator(config, runner)


# --- geometric mapping ---------------------------------------------------------------


def _lookup_policy(context: _Context, component_id: str) -> Any:
    from contextmap.state_estimation import LookupPolicy

    policy, _ = context.build(component_id, LookupPolicy)
    return policy


def _motion_correction(context: _Context, component_id: str) -> Any:
    from contextmap.geometric_mapping import MotionCorrectionPolicy

    policy, _ = context.build(component_id, MotionCorrectionPolicy)
    return policy


# --- sensor association --------------------------------------------------------------


def _occlusion_policy(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import OcclusionPolicy

    policy, _ = context.build(component_id, OcclusionPolicy)
    return policy


def _box_membership(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import BoxDepthComponentPolicy

    policy, _ = context.build(component_id, BoxDepthComponentPolicy)
    return policy


def _diagnostic_tolerances(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import DiagnosticTolerances

    policy, _ = context.build(component_id, DiagnosticTolerances)
    return policy


@dataclass(frozen=True, kw_only=True)
class _DenseSamplingParameters:
    """Parameters of the dense sampling component, as written in configuration.

    The values an ``interpolation`` may take, and what each means, are Sensor Association's
    :class:`~contextmap.sensor_association.InterpolationPolicy`; there is no default.
    """

    interpolation: str


def _dense_sampling(context: _Context, component_id: str) -> Any:
    from contextmap.sensor_association import InterpolationPolicy

    parameters, _ = context.build(component_id, _DenseSamplingParameters)
    try:
        return InterpolationPolicy(parameters.interpolation)
    except ValueError as error:
        valid = ", ".join(policy.value for policy in InterpolationPolicy)
        raise BackendConfigurationError(
            component_id,
            context.component(component_id).backend or "",
            [f"interpolation: {parameters.interpolation!r} is not one of: {valid}"],
        ) from error


@dataclass(frozen=True, kw_only=True)
class _DeskewParameters:
    """Parameters of the deskew component, as written in configuration.

    The meaning of each value is Geometric Mapping's
    :class:`~contextmap.geometric_mapping.PoseInterpolationDeskewPolicy`; none has a default,
    because the name and unit of a per-point time field belong to one sensor driver.
    """

    time_field: str
    time_unit: str
    reference: str
    time_bin_ns: int


def _deskew(context: _Context, component_id: str) -> Any:
    from contextmap.geometric_mapping import (
        PointTimeReference,
        PointTimeUnit,
        PoseInterpolationDeskewPolicy,
    )

    parameters, _ = context.build(component_id, _DeskewParameters)
    backend = context.component(component_id).backend or ""
    problems = [
        f"{name}: {getattr(parameters, name)!r} is not one of: "
        + ", ".join(member.value for member in enum)
        for name, enum in (("time_unit", PointTimeUnit), ("reference", PointTimeReference))
        if getattr(parameters, name) not in {member.value for member in enum}
    ]
    if problems:
        raise BackendConfigurationError(component_id, backend, problems)
    try:
        return PoseInterpolationDeskewPolicy(
            time_field=parameters.time_field,
            time_unit=PointTimeUnit(parameters.time_unit),
            reference=PointTimeReference(parameters.reference),
            time_bin_ns=parameters.time_bin_ns,
        )
    except ValueError as error:
        raise BackendConfigurationError(component_id, backend, [str(error)]) from error


def _scan_aggregation(context: _Context, component_id: str) -> Any:
    from contextmap.geometric_mapping import ScanVoxelPolicy

    policy, _ = context.build(component_id, ScanVoxelPolicy)
    return policy


# --- entity resolution ---------------------------------------------------------------


def _entity_retrieval_policy(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import CandidateRetrievalPolicy

    policy, _ = context.build(component_id, CandidateRetrievalPolicy)
    return policy


def _entity_belief(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import ResolvedBeliefPolicy

    policy, _ = context.build(component_id, ResolvedBeliefPolicy)
    return policy


def _entity_resolution_policy(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import ConservativeResolutionPolicy

    policy, _ = context.build(component_id, ConservativeResolutionPolicy)
    return policy


def _entity_geometry_comparison(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import GeometryComparisonPolicy

    policy, _ = context.build(component_id, GeometryComparisonPolicy)
    return policy


def _entity_semantic_compatibility(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import SemanticCompatibilityPolicy

    policy, _ = context.build(component_id, SemanticCompatibilityPolicy)
    return policy


def _entity_temporal_compatibility(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import TemporalCompatibilityPolicy

    policy, _ = context.build(component_id, TemporalCompatibilityPolicy)
    return policy


def _entity_appearance(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import AppearanceComparator, AppearanceComparisonPolicy

    config, _ = context.build(component_id, AppearanceComparisonPolicy)
    context.ensure_available(component_id)
    source = context.runtime(component_id, config, "FeatureVectorSource")
    return AppearanceComparator(source=source, policy=config)


def _entity_representation(context: _Context, component_id: str) -> Any:
    from contextmap.entity_resolution import (
        RepresentationComparator,
        RepresentationComparisonPolicy,
    )

    config, _ = context.build(component_id, RepresentationComparisonPolicy)
    context.ensure_available(component_id)
    source = context.runtime(component_id, config, "RepresentationVectorSource")
    return RepresentationComparator(source=source, policy=config)


# --- spatial relations -----------------------------------------------------------------


def _frame_conventions(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import FrameConventions

    policy, _ = context.build(component_id, FrameConventions)
    return policy


def _candidate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import CandidatePolicy

    policy, _ = context.build(component_id, CandidatePolicy)
    return policy


def _geometry_summary_policy(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_mapping import GeometrySummaryPolicy

    policy, _ = context.build(component_id, GeometrySummaryPolicy)
    return policy


def _geometric_predicate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import GeometricPredicatePolicy

    policy, _ = context.build(component_id, GeometricPredicatePolicy)
    return policy


def _contact_predicate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import ContactPredicatePolicy

    policy, _ = context.build(component_id, ContactPredicatePolicy)
    return policy


def _semantic_predicate_policy(context: _Context, component_id: str) -> Any:
    from contextmap.spatial_relations import SemanticPredicatePolicy

    policy, _ = context.build(component_id, SemanticPredicatePolicy)
    return policy


# --- point representation ----------------------------------------------------------


def _observation_support_anchor(context: _Context, component_id: str) -> Any:
    from contextmap.point_representation import ObservationSupportAnchorPolicy

    policy, _ = context.build(component_id, ObservationSupportAnchorPolicy)
    return policy


def _geometric_descriptor(context: _Context, component_id: str) -> PointEncoder:
    from contextmap.point_representation import SupportPolicy
    from contextmap.point_representation.backends.geometric_descriptor import (
        GeometricDescriptorEncoder,
    )

    _, extras = context.build(
        component_id,
        _NoParameters,
        extras={"support_policy": SupportPolicy},
        required=("support_policy",),
    )
    context.ensure_available(component_id)
    try:
        return GeometricDescriptorEncoder(extras["support_policy"])
    except ValueError as error:
        raise BackendConfigurationError(
            component_id, "geometric_descriptor", [f"support_policy: {error}"]
        ) from error


def _ptv3(context: _Context, component_id: str) -> PointEncoder:
    from contextmap.point_representation import SupportPolicy
    from contextmap.point_representation.backends.ptv3 import PTv3Config, PTv3PointEncoder

    config, extras = context.build(
        component_id,
        PTv3Config,
        extras={"support_policy": SupportPolicy},
        required=("support_policy",),
    )
    context.ensure_available(component_id)
    runtime = context.runtime(component_id, config, "PTv3Runtime")
    try:
        return PTv3PointEncoder(
            config=config, support_policy=extras["support_policy"], runtime=runtime
        )
    except ValueError as error:
        raise BackendConfigurationError(
            component_id, "ptv3", [f"support_policy: {error}"]
        ) from error


# --- semantic fusion ---------------------------------------------------------------


def _geometry_overlap_support(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import GeometryOverlapSupportPolicy

    policy, _ = context.build(component_id, GeometryOverlapSupportPolicy)
    return policy


def _baseline_accumulation(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import BaselineAccumulationPolicy

    policy, _ = context.build(component_id, BaselineAccumulationPolicy)
    return policy


def _quality_aware_accumulation(context: _Context, component_id: str) -> Any:
    from contextmap.semantic_fusion import QualityAwareAccumulationPolicy

    policy, _ = context.build(component_id, QualityAwareAccumulationPolicy)
    return policy


_FACTORIES: Mapping[str, Mapping[str, Factory]] = {
    "ingestion.source_adapter": {
        "ros1_bag": _source_adapter,
        "ros2_bag": _source_adapter,
        # Arquivo de poses (TUM) como fonte própria, para o GT ou uma odometria gravada chegarem
        # pelo `contextmap ingest` como sequência de pose, e não só pela API Python.
        "pose_file": _source_adapter,
    },
    "region_perception.discovery": {
        "sam2": _sam2,
        "sam3": _sam3,
        "florence2": _florence2_regions,
    },
    "region_perception.grounding": {"locateanything": _locateanything},
    "region_perception.refinement": {"sam2": _sam2_refinement},
    "visual_features.dense": {
        "dinov2": _dinov2,
        "dinov3": _dinov3,
        "siglip2": _siglip2,
    },
    "visual_features.region": {"clip": _clip, "alphaclip": _alphaclip},
    "semantic_interpretation.interpreter": {
        "qwen": _qwen,
        "gemini": _gemini,
        "florence2": _florence2_semantic,
        "eagle2_5": _eagle2_5,
    },
    "semantic_scoring.scorer": {"clip": _clip_scorer, "alphaclip": _alphaclip_scorer},
    "state_estimation.estimator": {"external_pose": _external_pose, "fast_lio": _fast_lio},
    "geometric_mapping.pose_lookup": {"lookup-policy-v1": _lookup_policy},
    "geometric_mapping.motion_correction": {"motion-correction-v1": _motion_correction},
    "geometric_mapping.deskew": {"pose-interpolation-deskew-v1": _deskew},
    "geometric_mapping.aggregation": {"scan-voxel-centroid-v1": _scan_aggregation},
    "sensor_association.occlusion": {"conservative-depth-support-v1": _occlusion_policy},
    "sensor_association.tolerances": {"diagnostic-tolerances-v1": _diagnostic_tolerances},
    "sensor_association.pose_policy": {"lookup-policy-v1": _lookup_policy},
    "sensor_association.dense_sampling": {"dense-feature-sampling-v1": _dense_sampling},
    "sensor_association.membership": {"box-depth-component-v1": _box_membership},
    "point_representation.center_selection": {
        "observation-support-anchor-v1": _observation_support_anchor
    },
    "point_representation.encoder": {"geometric_descriptor": _geometric_descriptor, "ptv3": _ptv3},
    "semantic_fusion.support": {"geometry-jaccard-support-v1": _geometry_overlap_support},
    "semantic_fusion.accumulation": {
        "baseline-evidence-accumulation-v1": _baseline_accumulation,
        "quality-aware-evidence-accumulation-v1": _quality_aware_accumulation,
    },
    "entity_resolution.retrieval": {"entity-candidate-retrieval-v1": _entity_retrieval_policy},
    "entity_resolution.resolution": {
        "conservative-staged-resolution-v1": _entity_resolution_policy
    },
    "entity_resolution.geometry_comparison": {
        "entity-geometry-comparison-v1": _entity_geometry_comparison
    },
    "entity_resolution.semantic_compatibility": {
        "entity-semantic-compatibility-v1": _entity_semantic_compatibility
    },
    "entity_resolution.temporal_compatibility": {
        "entity-temporal-compatibility-v1": _entity_temporal_compatibility
    },
    "entity_resolution.appearance": {"entity-appearance-comparison-v1": _entity_appearance},
    "entity_resolution.belief": {"resolved-belief-min-support-v1": _entity_belief},
    "entity_resolution.representation": {
        "entity-representation-comparison-v1": _entity_representation
    },
    "semantic_mapping.geometry_summary": {"entity-geometry-summary-v1": _geometry_summary_policy},
    "spatial_relations.frame_conventions": {"map-frame-conventions-v1": _frame_conventions},
    "spatial_relations.candidate": {"bounds-neighborhood-candidates-v1": _candidate_policy},
    "spatial_relations.geometry_summary": {"entity-geometry-summary-v1": _geometry_summary_policy},
    "spatial_relations.geometric_predicate": {
        "bounds-geometric-predicates-v1": _geometric_predicate_policy
    },
    "spatial_relations.contact_predicate": {
        "point-contact-predicates-v1": _contact_predicate_policy
    },
    "spatial_relations.semantic_predicate": {
        "interaction-range-predicate-v1": _semantic_predicate_policy
    },
}


def _construct(context: _Context, component_id: str) -> Any:
    """Build one variation point with the factory of its selected backend."""
    backend = context.component(component_id).backend
    assert backend is not None  # a seleção completa já foi exigida em compose().
    return _FACTORIES[component_id][backend](context, component_id)


def _construct_optional(context: _Context, component_id: str) -> Any | None:
    """Build a genuinely optional variation point, or ``None`` when it was not selected.

    Unlike :func:`_construct`, a missing backend here is not a configuration error: the
    catalog marks the component ``optional`` (see :class:`~contextmap.runtime.components.
    ComponentSpec`), so :func:`~contextmap.runtime.configuration.check_component_selection` never
    requires it, and its absence must never be replaced by a default policy.
    """
    if context.component(component_id).backend is None:
        return None
    return _construct(context, component_id)


# --- stages ------------------------------------------------------------------------


def _compose_ingestion(context: _Context) -> dict[str, object]:
    return {"source_adapter": _construct(context, "ingestion.source_adapter")}


def _compose_region_perception(context: _Context) -> dict[str, object]:
    """Compose discovery and optional edits of the final regional evidence."""
    composed = {
        "region_discovery": _construct(context, "region_perception.discovery"),
        "region_grounding": _construct_optional(context, "region_perception.grounding"),
        "region_refinement": _construct_optional(context, "region_perception.refinement"),
    }
    if composed["region_refinement"] is not None and composed["region_grounding"] is None:
        raise ConfigurationError.single(
            "region refinement refines grounding proposals: select a "
            "region_perception.grounding backend too, or no refinement",
            path="components.region_perception.refinement",
        )
    return composed


def _compose_visual_perception(context: _Context) -> dict[str, object]:
    """Compose feature extractors over selected final regional evidence."""
    return {
        "dense_features": _construct(context, "visual_features.dense"),
        "region_features": _construct(context, "visual_features.region"),
    }


def _compose_semantic_interpretation(context: _Context) -> dict[str, object]:
    """Compose the interpreter and the effective request policy it uses."""
    semantic: _SemanticInterpretation = _construct(context, "semantic_interpretation.interpreter")
    return {
        "semantic_interpreter": semantic.interpreter,
        "semantic_request_policy": semantic.request_policy,
    }


def _compose_semantic_scoring(context: _Context) -> dict[str, object]:
    return {"semantic_scoring": _construct(context, "semantic_scoring.scorer")}


def _compose_state_estimation(context: _Context) -> dict[str, object]:
    return {"state_estimator": _construct(context, "state_estimation.estimator")}


def _compose_geometric_mapping(context: _Context) -> dict[str, object]:
    return {
        "geometric_mapping_pose_lookup": _construct(context, "geometric_mapping.pose_lookup"),
        "motion_correction": _construct(context, "geometric_mapping.motion_correction"),
        "geometric_mapping_deskew": _construct_optional(context, "geometric_mapping.deskew"),
        "geometric_mapping_aggregation": _construct_optional(
            context, "geometric_mapping.aggregation"
        ),
    }


def _compose_sensor_association(context: _Context) -> dict[str, object]:
    from contextmap.sensor_association import CandidateGeometryPolicy

    return {
        "occlusion_policy": _construct(context, "sensor_association.occlusion"),
        "association_tolerances": _construct(context, "sensor_association.tolerances"),
        "association_pose_policy": _construct(context, "sensor_association.pose_policy"),
        # Opcional (#651): sem backend selecionado a associação continua geometry-only.
        "association_dense_sampling": _construct_optional(
            context, "sensor_association.dense_sampling"
        ),
        # Opcional: sem backend, regiões sem máscara continuam puladas (mask-membership-v1).
        "association_box_policy": _construct_optional(context, "sensor_association.membership"),
        # A política de candidatos é um policy cruzado da execução (#562), não um ponto de
        # variação de backend: o padrão None avalia o mapa inteiro, exatamente como antes.
        "association_candidates": CandidateGeometryPolicy(
            max_range_m=context.effective.config.policies.association_max_range_m
        ),
    }


def _compose_point_representation(context: _Context) -> dict[str, object]:
    return {
        "center_selection": _construct(context, "point_representation.center_selection"),
        "point_encoder": _construct(context, "point_representation.encoder"),
    }


def _compose_semantic_fusion(context: _Context) -> dict[str, object]:
    return {
        "support_policy": _construct(context, "semantic_fusion.support"),
        "accumulation_policy": _construct(context, "semantic_fusion.accumulation"),
    }


def _compose_nothing(context: _Context) -> dict[str, object]:
    """Compose a stage with no variation point: its service is stateless capability code."""
    return {}


def _compose_entity_resolution(context: _Context) -> dict[str, object]:
    from contextmap.entity_resolution import ComparisonChannels

    channels = ComparisonChannels(
        geometry=_construct(context, "entity_resolution.geometry_comparison"),
        semantic=_construct_optional(context, "entity_resolution.semantic_compatibility"),
        temporal=_construct_optional(context, "entity_resolution.temporal_compatibility"),
        appearance=_construct_optional(context, "entity_resolution.appearance"),
        representation=_construct_optional(context, "entity_resolution.representation"),
    )
    return {
        "entity_retrieval_policy": _construct(context, "entity_resolution.retrieval"),
        "entity_comparison_channels": channels,
        "entity_resolution_policy": _construct(context, "entity_resolution.resolution"),
        "entity_belief_policy": _construct_optional(context, "entity_resolution.belief"),
    }


def _compose_semantic_mapping(context: _Context) -> dict[str, object]:
    return {
        "semantic_mapping_geometry_summary": _construct(
            context, "semantic_mapping.geometry_summary"
        )
    }


def _compose_spatial_relations(context: _Context) -> dict[str, object]:
    from contextmap.spatial_relations import RelationsRunPolicies

    frame_conventions = _construct(context, "spatial_relations.frame_conventions")
    candidate = _construct(context, "spatial_relations.candidate")
    geometry_summary = _construct(context, "spatial_relations.geometry_summary")
    geometric = _construct_optional(context, "spatial_relations.geometric_predicate")
    contact = _construct_optional(context, "spatial_relations.contact_predicate")
    semantic = _construct_optional(context, "spatial_relations.semantic_predicate")
    try:
        policies = RelationsRunPolicies(
            frame_conventions=frame_conventions,
            candidate=candidate,
            geometry_summary=geometry_summary,
            geometric=geometric,
            contact=contact,
            semantic=semantic,
        )
    except ValueError as error:
        # A regra de coerência é de spatial_relations; aqui a recusa só é atribuída ao
        # componente cujo alcance não cobre as tolerâncias, para o preflight apontá-lo.
        component_id = "spatial_relations.candidate"
        raise BackendConfigurationError(
            component_id,
            context.component(component_id).backend or "",
            [f"proximity_radius_m: {error}"],
        ) from error
    return {"spatial_relations_policies": policies}


_STAGE_COMPOSERS: Mapping[str, Callable[[_Context], dict[str, object]]] = {
    "ingestion": _compose_ingestion,
    "pose_ingestion": _compose_nothing,
    "region_perception": _compose_region_perception,
    "visual_perception": _compose_visual_perception,
    "semantic_interpretation": _compose_semantic_interpretation,
    "semantic_scoring": _compose_semantic_scoring,
    "state_estimation": _compose_state_estimation,
    "geometric_mapping": _compose_geometric_mapping,
    "sensor_association": _compose_sensor_association,
    "point_representation": _compose_point_representation,
    "semantic_fusion": _compose_semantic_fusion,
    "semantic_mapping": _compose_semantic_mapping,
    "entity_resolution": _compose_entity_resolution,
    "spatial_relations": _compose_spatial_relations,
    "context_map": _compose_nothing,
}


def composed_stages() -> frozenset[str]:
    """List the stages this composition root can compose.

    Returns:
        Stage identities with a composer. Every available stage of the catalog must be here.
    """
    return frozenset(_STAGE_COMPOSERS)


def compose_executors(
    effective: EffectiveConfig,
    *,
    providers: Mapping[str, RuntimeProvider] | None = None,
    environ: Mapping[str, str] | None = None,
    module_available: Callable[[str], bool] | None = None,
    on_provider_override: Callable[[str], None] | None = None,
    on_composition_failure: Callable[[str, CompositionError | ConfigurationError], None]
    | None = None,
    semantic_map_id: SemanticMapId | None = None,
    code_digest: str | None = None,
    code_version: str | None = None,
) -> dict[str, StageExecutor]:
    """Compose the real :class:`~contextmap.runtime.pipeline.StageExecutor` a DAG run needs.

    This is the automatic counterpart of :func:`compose`: instead of handing back the
    composed backends and policies, it wraps each one into the concrete executor class of
    ``contextmap.runtime.executors`` its stage needs, keyed by ``stage_id``, exactly as a
    caller previously had to build them by hand. A stage whose variation points are not
    (yet) fully selected in ``effective``, whose selected backend rejects its own
    configured parameters, or whose capability the executor does not support, is left out
    -- never filled with a placeholder or allowed to abort composing the other stages.
    Leaving a stage out only says *that* it has no executor; *why* is the error
    :func:`compose` raised for it, which is handed to ``on_composition_failure`` so the
    caller can pass it to :func:`~contextmap.runtime.pipeline.preflight` as
    ``composition_failures``. Preflight then reports the real cause (a rejected parameter,
    an unresolvable ``resources.providers`` target, a missing model runtime) at the
    component it failed on, instead of a bare "no executor is registered".

    ``state_estimation``, ``geometric_mapping``, ``sensor_association``, ``semantic_fusion``,
    ``region_perception``, ``visual_perception``, ``semantic_scoring``,
    ``point_representation``, ``semantic_mapping``, ``entity_resolution``,
    ``spatial_relations`` and ``context_map`` can be composed this way: each needs only the
    effective configuration and the
    upstream artifacts the DAG already carries -- except the stages whose run records an identity
    no configuration carries (see below). :func:`executor_identities` derives those identities
    from a run's code identity and dataset; a caller that has neither leaves them ``None``.

    - ``ingestion`` and ``pose_ingestion`` (issue #555's opt-in auxiliary pose stage, see
      ``catalog.py``) are not composed here: :class:`~contextmap.runtime.ingestion_service.
      IngestionStageExecutor` needs a concrete ``IngestionRequest`` (source path, topics,
      synchronization tolerance) that is per-invocation input, never part of a resolved
      configuration -- it is what the ``ingest`` command's own flags build. A caller that
      wants either stage to run inside :func:`~contextmap.runtime.pipeline.run_plan`
      still injects its own :class:`~contextmap.runtime.ingestion_service.
      IngestionStageExecutor` explicitly (a second instance, under ``stage_id=
      "pose_ingestion"``, for the auxiliary one); the ordinary canonical path is to run
      ``contextmap ingest`` first and feed its published artifact to ``run``/``stage`` as a
      provided or selected input.
    - ``region_perception`` is composed from its one component, ``region_perception.discovery``:
      the stage runs the discovery and publishes the regions, their masks and the audit of
      each frame. Grounding and refinement are components of ``visual_perception``, which
      executes them, so they are composed there (``docs/perception-dag-decomposition.md``).
    - ``visual_perception`` is composed only when its three required variation points
      (``dense_features``, ``region_features``, ``semantic_interpretation``) are genuinely
      selected and available -- a partially configured capability never gets a partial
      executor (#507). Its optional ``region_grounding`` point, when selected, is part of the
      same executor: a grounding backend that cannot be built leaves the whole stage out,
      never a run without it. It does **not** need ``region_perception`` to compose: it
      replays the regions of the artifact it consumes, so the two stages are tied by the DAG
      edge, and a discovery that cannot be composed is reported at its own stage.
    - ``semantic_scoring`` (#652) is composed only when the configuration enables it: it is
      an optional stage, off by default, so a disabled one is no composition failure. Its
      executor scores the claims of the perception run it is given, pinned or just produced,
      against that run's own region features.
    - ``point_representation`` (#663) is composed only when the configuration enables it (an
      optional stage, off by default) and with a ``code_version``, which its run records. Its
      executor selects the centers with the capability's versioned policy
      (``point_representation.center_selection``) from the association run of the DAG and
      represents them with the selected encoder; ``ptv3`` needs a ``PTv3Runtime`` provider and,
      without one, the stage is left out with that cause, never replaced by the descriptor.
      The run is a derived output: no stage consumes it (#657).
    - ``semantic_fusion`` runs whichever accumulation backend is selected: the baseline
      (``baseline-evidence-accumulation-v1``) or the quality-aware policy
      (``quality-aware-evidence-accumulation-v1``, #655), which
      :class:`~contextmap.runtime.executors.SemanticFusionExecutor` weights with the quality
      its association runs persisted. One never stands in for the other.
    - ``point_representation``, ``semantic_fusion``, ``semantic_mapping``, ``entity_resolution``
      and ``spatial_relations`` record the code that produced their run, which their writers or
      services require (the representation service refuses an empty one, and a ``ContextMap``
      refuses to cite a run from the fusion onward without one): each is composed only with
      a ``code_version`` (#659). ``semantic_mapping`` also records ``semantic_map_id`` (the
      persistent semantic map its entities belong to) and ``code_digest``, and is composed only
      with both. None of these is a configuration value, so this function never invents one:
      without it the stage is left out, with an
      :class:`~contextmap.runtime.errors.ExecutorIdentityMissingError` naming what is missing
      handed to ``on_composition_failure``, and its artifact must be supplied
      (``provided``/``selections``) or the run is blocked by preflight with that cause.
    - ``entity_resolution`` always evaluates the required geometry channel; every other
      channel (``semantic``, ``temporal``, ``appearance``, ``representation``) is evaluated
      only when its own component is selected, and is ``None`` -- never a default policy --
      when it is not.
    - ``spatial_relations`` always applies the required frame conventions, candidate policy
      and geometry summary; the geometric and contact predicate evaluators run only when
      their own component is selected.
    - ``context_map`` has no variation point of its own, so composing it never fails once the
      preset declares it. Its ``up_axis`` comes from the same ``spatial_relations``
      ``FrameConventions`` this function already composed above, when available -- never
      re-derived independently -- and stays unknown otherwise.
    - ``policies.debug_level`` (``"none"``/``"standard"``/``"full"``) is translated here into each
      composed stage's own debug-level type and passed to its executor, which forwards it to the
      stage writer: this is the single place the configured debug level reaches the run
      (``visual_perception`` receives the feature and semantic levels; ``context_map`` and
      ``semantic_scoring`` persist no debug and take none). It never changes contractual output.

    Args:
        effective: The resolved configuration.
        providers: Model runtimes or clients for backends without a bundled loader; see
            :func:`compose`.
        environ: Environment to read secrets from; defaults to ``os.environ``.
        module_available: Predicate telling whether an optional module is installed.
        on_provider_override: Called with a component identity whenever ``providers``
            overrides a ``resources.providers`` target ``effective`` also declares for it;
            see :func:`compose`.
        on_composition_failure: Called with a stage identity and the error :func:`compose`
            raised for it, once for every stage this function tried and could not compose, a
            disabled one included (preflight only looks up the stages that will run). Only
            the documented composition errors (:class:`~contextmap.runtime.errors.
            CompositionError` and :class:`~contextmap.runtime.configuration.ConfigurationError`)
            arrive here; anything else propagates.
        semantic_map_id: Identity of the persistent semantic map ``semantic_mapping``'s
            entities belong to. Required, together with ``code_digest`` and ``code_version``,
            for this function to compose ``semantic_mapping``; see above.
        code_digest: Digest of the code producing the run, exactly as the caller supplies it --
            this function never computes one. Required, together with ``semantic_map_id`` and
            ``code_version``, for this function to compose ``semantic_mapping``; see above.
        code_version: Code revision that produced the run, threaded into every composed
            executor that records one. Required for the stages whose run must record it (see
            above); the others record ``None`` without it -- never a fabricated value.

    Returns:
        One executor per stage that could genuinely be composed from ``effective``. Never
        raises a composition error: a stage this cannot build for any reason (incomplete
        selection, a rejected parameter, a missing module or secret, a missing model runtime,
        an unresolvable declared provider target or an identity its run records and nobody
        supplied) is simply absent from the result, one stage at a time, with its cause handed
        to ``on_composition_failure``, so one broken stage never costs the others their real
        executor.

    Raises:
        ValueError: If a composed executor refuses an identity it was given (an empty
            ``code_version``, for example).
    """
    from contextmap.entity_resolution import MatchEvidenceBuilder, ResolutionDebugLevel
    from contextmap.geometric_mapping import MapDebugLevel
    from contextmap.point_representation import PointRepresentationDebugLevel
    from contextmap.runtime.executors import (
        ContextMapExecutor,
        EntityResolutionExecutor,
        GeometricMappingExecutor,
        PerceptionSchedule,
        PointRepresentationExecutor,
        RegionPerceptionExecutor,
        SemanticFusionExecutor,
        SemanticInterpretationExecutor,
        SemanticMappingExecutor,
        SemanticScoringExecutor,
        SensorAssociationExecutor,
        SpatialRelationsExecutor,
        StateEstimationExecutor,
        VisualPerceptionExecutor,
    )
    from contextmap.runtime.image_masks import resolve_image_masks
    from contextmap.semantic_fusion import SemanticFusionDebugLevel
    from contextmap.semantic_interpretation import (
        SemanticDebugLevel,
    )
    from contextmap.semantic_mapping import EntityMaterializationPolicy, MappingDebugLevel
    from contextmap.sensor_association import SensorAssociationDebugLevel
    from contextmap.spatial_relations import RelationsRunDebugLevel
    from contextmap.state_estimation import StateEstimationDebugLevel
    from contextmap.visual_perception import (
        FeatureDebugLevel,
    )

    # policies.debug_level is one string ("none"/"standard"/"full"); each capability owns its own
    # debug-level type, which shares those member values, so the composition translates the policy
    # into every stage writer's own type here -- the single place a run's debug level is wired.
    debug = effective.config.policies.debug_level
    # As máscaras estáticas por câmera são decodificadas uma vez e entregues às duas etapas que
    # preparam imagens (percepção e associação), para que o rig e a borda do fisheye sejam
    # excluídos de forma consistente. Referências relativas resolvem contra o diretório de
    # trabalho atual.
    image_masks = resolve_image_masks(effective.config.image_preparation)

    def _compose_stage(stage_id: str) -> ComposedRuntime | None:
        """Compose one stage, or hand its composition error over and return ``None``."""
        try:
            return compose(
                effective,
                stages=[stage_id],
                providers=providers,
                environ=environ,
                module_available=module_available,
                on_provider_override=on_provider_override,
            )
        except (ConfigurationError, CompositionError) as error:
            # Seleção incompleta, estágio desabilitado, backend que rejeita seus próprios
            # parâmetros, módulo/segredo/runtime ausente ou alvo de provider irresolvível:
            # ausência honesta, nunca um erro que aborte a composição dos outros estágios. A
            # causa segue para quem chama, que a entrega ao preflight.
            failure = error
        if on_composition_failure is not None:
            on_composition_failure(stage_id, failure)
        return None

    def _identities_given(stage_id: str, **identities: object) -> bool:
        """Tell whether a stage's run gets every identity it records; hand over the missing."""
        missing = tuple(name for name, value in identities.items() if value is None)
        if missing and on_composition_failure is not None:
            on_composition_failure(stage_id, ExecutorIdentityMissingError(stage_id, missing))
        return not missing

    executors: dict[str, StageExecutor] = {}

    region_perception = _compose_stage("region_perception")
    if region_perception is not None:
        assert region_perception.region_discovery is not None
        executors["region_perception"] = RegionPerceptionExecutor(
            region_discovery=region_perception.region_discovery,
            region_grounding=region_perception.region_grounding,
            region_refinement=region_perception.region_refinement,
            image_masks=image_masks,
            code_version=code_version,
            perception_schedule=PerceptionSchedule(effective.config.policies.perception_schedule),
        )

    visual_perception = _compose_stage("visual_perception")
    if visual_perception is not None:
        assert visual_perception.dense_features is not None
        assert visual_perception.region_features is not None
        executors["visual_perception"] = VisualPerceptionExecutor(
            dense_features=visual_perception.dense_features,
            region_features=visual_perception.region_features,
            dense_region_pooling_enabled=(
                effective.config.pipeline.preset == CANONICAL2_PROFILE_ID
            ),
            feature_debug_level=FeatureDebugLevel(debug),
            image_masks=image_masks,
            perception_schedule=PerceptionSchedule(effective.config.policies.perception_schedule),
            code_version=code_version,
        )

    semantic_interpretation = _compose_stage("semantic_interpretation")
    if semantic_interpretation is not None:
        assert semantic_interpretation.semantic_interpreter is not None
        assert semantic_interpretation.semantic_request_policy is not None
        executors["semantic_interpretation"] = SemanticInterpretationExecutor(
            semantic_interpreter=semantic_interpretation.semantic_interpreter,
            semantic_request_policy=semantic_interpretation.semantic_request_policy,
            semantic_debug_level=SemanticDebugLevel(debug),
            image_masks=image_masks,
            perception_schedule=PerceptionSchedule(effective.config.policies.perception_schedule),
            code_version=code_version,
        )

    if effective.config.pipeline.stages.get("semantic_scoring", False):
        semantic_scoring = _compose_stage("semantic_scoring")
        if semantic_scoring is not None:
            assert semantic_scoring.semantic_scoring is not None
            executors["semantic_scoring"] = SemanticScoringExecutor(
                plan=semantic_scoring.semantic_scoring, code_version=code_version
            )

    state_estimation = _compose_stage("state_estimation")
    if state_estimation is not None:
        assert state_estimation.state_estimator is not None
        executors["state_estimation"] = StateEstimationExecutor(
            state_estimation.state_estimator,
            allow_ground_truth_trajectory=(
                effective.config.policies.trajectory_mode == "allow_ground_truth"
            ),
            debug_level=StateEstimationDebugLevel(debug),
        )

    geometric_mapping = _compose_stage("geometric_mapping")
    if geometric_mapping is not None:
        assert geometric_mapping.geometric_mapping_pose_lookup is not None
        assert geometric_mapping.motion_correction is not None
        executors["geometric_mapping"] = GeometricMappingExecutor(
            pose_lookup=geometric_mapping.geometric_mapping_pose_lookup,
            motion_correction=geometric_mapping.motion_correction,
            deskew=geometric_mapping.geometric_mapping_deskew,
            aggregation=geometric_mapping.geometric_mapping_aggregation,
            code_version=code_version,
            debug_level=MapDebugLevel(debug),
        )

    sensor_association = _compose_stage("sensor_association")
    if sensor_association is not None:
        assert sensor_association.occlusion_policy is not None
        assert sensor_association.association_tolerances is not None
        assert sensor_association.association_pose_policy is not None
        assert sensor_association.association_candidates is not None
        executors["sensor_association"] = SensorAssociationExecutor(
            candidates=sensor_association.association_candidates,
            occlusion=sensor_association.occlusion_policy,
            tolerances=sensor_association.association_tolerances,
            pose_policy=sensor_association.association_pose_policy,
            dense_sampling=sensor_association.association_dense_sampling,
            box_policy=sensor_association.association_box_policy,
            code_version=code_version,
            debug_level=SensorAssociationDebugLevel(debug),
            image_masks=image_masks,
        )

    if effective.config.pipeline.stages.get("point_representation", False):
        point_representation = _compose_stage("point_representation")
        if point_representation is not None and _identities_given(
            "point_representation", code_version=code_version
        ):
            assert point_representation.center_selection is not None
            assert point_representation.point_encoder is not None
            assert code_version is not None
            executors["point_representation"] = PointRepresentationExecutor(
                center_selection=point_representation.center_selection,
                encoder=point_representation.point_encoder,
                code_version=code_version,
                debug_level=PointRepresentationDebugLevel(debug),
            )

    semantic_fusion = _compose_stage("semantic_fusion")
    if semantic_fusion is not None and _identities_given(
        "semantic_fusion", code_version=code_version
    ):
        assert semantic_fusion.support_policy is not None
        assert semantic_fusion.accumulation_policy is not None
        assert code_version is not None
        executors["semantic_fusion"] = SemanticFusionExecutor(
            support_policy=semantic_fusion.support_policy,
            accumulation_policy=semantic_fusion.accumulation_policy,
            code_version=code_version,
            debug_level=SemanticFusionDebugLevel(debug),
        )

    semantic_mapping = _compose_stage("semantic_mapping")
    if semantic_mapping is not None and _identities_given(
        "semantic_mapping",
        code_version=code_version,
        semantic_map_id=semantic_map_id,
        code_digest=code_digest,
    ):
        assert semantic_mapping.semantic_mapping_geometry_summary is not None
        assert code_version is not None and semantic_map_id is not None
        assert code_digest is not None
        executors["semantic_mapping"] = SemanticMappingExecutor(
            policy=EntityMaterializationPolicy(
                geometry=semantic_mapping.semantic_mapping_geometry_summary
            ),
            semantic_map_id=semantic_map_id,
            code_digest=code_digest,
            code_version=code_version,
            debug_level=MappingDebugLevel(debug),
        )

    entity_resolution = _compose_stage("entity_resolution")
    if entity_resolution is not None and _identities_given(
        "entity_resolution", code_version=code_version
    ):
        assert entity_resolution.entity_retrieval_policy is not None
        assert entity_resolution.entity_comparison_channels is not None
        assert entity_resolution.entity_resolution_policy is not None
        assert code_version is not None
        executors["entity_resolution"] = EntityResolutionExecutor(
            retrieval=entity_resolution.entity_retrieval_policy,
            builder=MatchEvidenceBuilder(entity_resolution.entity_comparison_channels),
            resolution=entity_resolution.entity_resolution_policy,
            belief_policy=entity_resolution.entity_belief_policy,
            code_version=code_version,
            debug_level=ResolutionDebugLevel(debug),
        )

    spatial_relations = _compose_stage("spatial_relations")
    if spatial_relations is not None and _identities_given(
        "spatial_relations", code_version=code_version
    ):
        assert spatial_relations.spatial_relations_policies is not None
        assert code_version is not None
        executors["spatial_relations"] = SpatialRelationsExecutor(
            policies=spatial_relations.spatial_relations_policies,
            code_version=code_version,
            debug_level=RelationsRunDebugLevel(debug),
        )

    context_map = _compose_stage("context_map")
    if context_map is not None:
        # context_map has no variation point of its own, so composing it never fails once the
        # preset declares it. Its up_axis comes from the same FrameConventions spatial_relations
        # already composed above, when available -- never re-derived independently -- and stays
        # None otherwise (MapFrame allows that).
        up_axis = (
            spatial_relations.spatial_relations_policies.frame_conventions.up_axis
            if spatial_relations is not None and spatial_relations.spatial_relations_policies
            else None
        )
        executors["context_map"] = ContextMapExecutor(up_axis=up_axis, code_version=code_version)

    return executors
