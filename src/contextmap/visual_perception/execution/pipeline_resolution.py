"""Validação e resolução do grafo de Percepção Visual.

Este módulo transforma uma configuração declarativa em estágios executáveis e valida seus
contratos antes de construir backends pesados. Ele não escolhe uma topologia concreta: presets
versionados vivem em :mod:`contextmap.visual_perception.execution.pipeline_presets`; a
execução do grafo pertence a :mod:`contextmap.visual_perception.execution.stage_execution`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from contextmap.region_perception import (
    REGION_SET_ASSEMBLY_POLICY,
    Region2D,
    RegionDiscovery,
    RegionDiscoveryIdentity,
    RegionGroundingExecution,
    RegionRefinementExecution,
    assemble_regions,
)
from contextmap.semantic_interpretation import SceneContext, SemanticClaim, SemanticInterpreter
from contextmap.semantic_scoring import SemanticScore, SemanticScorer

if TYPE_CHECKING:
    from contextmap.semantic_interpretation import SemanticInterpretationExecution
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.execution.stage_execution import StageDefinition
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_features import (
    FeatureExtractor,
    FeatureResolutionEnhancement,
)
from contextmap.visual_perception.visual_features.feature_enhancement import (
    enhance_feature_resolution,
)
from contextmap.visual_perception.visual_features.region_feature_pooling import DenseFeatureMap
from contextmap.visual_perception.visual_features.visual_evidence import FeatureScope, VisualFeature

PIPELINE_SCHEMA_VERSION = "0.2.0"
"""Pipeline preset encoding schema version."""


class PipelineConfigError(Exception):
    """Raised when a pipeline preset is structurally invalid or cannot be resolved."""


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """One declarative stage node within a :class:`PipelinePreset`.

    Attributes:
        stage_id: Unique identity of this stage within its preset.
        capability: Which capability this stage exercises. For a
            backend stage (``backend_id`` is not ``None``), must be one
            of the capabilities in ``_CAPABILITY_ADAPTERS`` — the fixed,
            small set this module knows how to invoke. For a source
            stage (``backend_id`` is ``None``), this is a free-form
            label (e.g. ``"image_preparation"``) used only for
            diagnostics; nothing in this module calls it.
        inputs: Named input -> upstream ``stage_id`` this stage reads
            from once that stage has succeeded, e.g.
            ``{"image": "image_preparation", "regions": "region_discovery"}``.
            A downstream stage never branches on which concrete backend
            produced an upstream input — only on the named input's
            declared contract.
        backend_id: Identity of the backend implementation to resolve
            for this stage (e.g. ``"dinov2"``, ``"fake_region_discovery"``).
            ``None`` marks a stage no backend fills: a source stage whose
            output is supplied externally at
            :meth:`ResolvedPipeline.build_stage_graph` time (e.g. the
            prepared image), or a stage whose capability is a rule this
            module owns (``region_set_assembly``, the only one), computed
            from its inputs.
        optional: Documents that an alternative preset may omit this
            stage without that omission being a mistake. Enforcement is
            structural, not a runtime flag: a preset that omits an
            optional stage must also update whichever downstream
            stage's ``inputs`` referenced it, or validation rejects the
            preset for depending on an unknown stage.
        parameters: Resolved, primitive-valued configuration passed to
            this stage's :data:`StageBackendFactory`, e.g.
            ``{"confidence_threshold": 0.5}``. A stage whose capability is
            a rule of this module declares exactly the version of that
            rule, e.g. ``{"policy_id": "region-set-assembly/1"}``. Never a
            place for secrets/credentials.
        feature_scope: Output scope selected for a ``feature_extractor``
            stage. ``REGION`` makes the ``regions`` input mandatory;
            ``DENSE`` and ``GLOBAL`` accept only the prepared image.
            Must be ``None`` for every other capability.
    """

    stage_id: str
    capability: str
    inputs: Mapping[str, str] = field(default_factory=dict)
    backend_id: str | None = None
    optional: bool = False
    parameters: Mapping[str, object] = field(default_factory=dict)
    feature_scope: FeatureScope | None = None

    @property
    def depends_on(self) -> frozenset[str]:
        """Upstream ``stage_id``s this stage's ``inputs`` reference."""
        return frozenset(self.inputs.values())


@dataclass(frozen=True, kw_only=True)
class PipelinePreset:
    """A versioned, named Visual Perception pipeline configuration.

    Attributes:
        preset_id: Versioned identity, e.g. ``"canonical/1"``. Two
            presets that select different stages/backends/parameters
            must use different ``preset_id``s — a version number is
            never reused for a different topology.
        stages: The declarative stage graph. Order does not need to
            already be dependency-sorted.
        description: Human-readable summary of what this preset is for.
    """

    preset_id: str
    stages: tuple[StageSpec, ...]
    description: str = ""


StageBackendFactory = Callable[[Mapping[str, object]], object]
"""Constructs one stage's backend instance from its resolved ``parameters``.

Never called until :func:`validate_pipeline_preset` has already
succeeded for the preset being resolved — this is what keeps a heavy
model from loading before its graph/config passed validation.
"""


@runtime_checkable
class _ProvenanceCapable(Protocol):
    def backend_provenance(self) -> BackendProvenance: ...


@runtime_checkable
class _DiscoveryIdentityCapable(Protocol):
    """A stage backend that decides regions under an effective discovery identity."""

    def discovery_identity(self) -> RegionDiscoveryIdentity: ...


@runtime_checkable
class _SceneRegionSemanticStageBackend(Protocol):
    """Current canonical/1 scene/region stage shape pending request-stage migration."""

    def interpret_scene(self, image: PreparedImage) -> SceneContext | None: ...

    def interpret_regions(
        self, image: PreparedImage, regions: Sequence[Region2D]
    ) -> Sequence[SemanticClaim]: ...


@runtime_checkable
class _ImageGroundingStageBackend(Protocol):
    """The ``region_grounding`` stage shape: ask one image every configured grounding query.

    The run that owns the evidence and the ordered queries are bound when the backend is
    built, so each image gets one explicit
    :class:`~contextmap.region_perception.grounding.region_grounding.RegionGroundingRequest`
    per query.
    """

    def backend_provenance(self) -> BackendProvenance: ...

    def ground_image(self, image: PreparedImage) -> Sequence[RegionGroundingExecution]: ...


@runtime_checkable
class _GroundingRefinementStageBackend(Protocol):
    """The ``region_refinement`` stage shape: refine the grounding proposals of one image."""

    def backend_provenance(self) -> BackendProvenance: ...

    def refine_groundings(
        self, image: PreparedImage, groundings: Sequence[RegionGroundingExecution]
    ) -> Sequence[RegionRefinementExecution]: ...


@runtime_checkable
class _DenseRegionPoolingStageBackend(Protocol):
    """Stage port that derives mask-weighted region features from dense evidence."""

    def backend_provenance(self) -> BackendProvenance: ...

    def pool(
        self,
        image: PreparedImage,
        regions: Sequence[Region2D],
        dense_features: Sequence[VisualFeature],
    ) -> Sequence[VisualFeature]: ...


def _backend_type_error(capability: str, port: str, backend: object) -> TypeError:
    """Describe a resolved backend that does not implement its stage's capability port.

    The adapters below validate explicitly instead of with ``assert``, which ``python -O``
    strips and which fails without saying what was wrong.
    """
    return TypeError(f"{capability} backend must implement {port}, got {type(backend).__name__}")


def _input_type_error(capability: str, input_name: str, expected: str, value: object) -> TypeError:
    """Describe an upstream output that is not the contract a named input declares."""
    return TypeError(
        f"{capability} input {input_name!r} must be {expected}, got {type(value).__name__}"
    )


def _run_region_discovery(backend: object, inputs: Mapping[str, object]) -> Sequence[Region2D]:
    if not isinstance(backend, RegionDiscovery):
        raise _backend_type_error("region_discovery", "RegionDiscovery", backend)
    image = inputs["image"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("region_discovery", "image", "PreparedImage", image)
    return backend.discover(image)


def _run_region_grounding(
    backend: object, inputs: Mapping[str, object]
) -> tuple[RegionGroundingExecution, ...]:
    if not isinstance(backend, _ImageGroundingStageBackend):
        raise _backend_type_error("region_grounding", "ground_image()", backend)
    image = inputs["image"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("region_grounding", "image", "PreparedImage", image)
    return tuple(backend.ground_image(image))


def _run_region_refinement(
    backend: object, inputs: Mapping[str, object]
) -> tuple[RegionRefinementExecution, ...]:
    if not isinstance(backend, _GroundingRefinementStageBackend):
        raise _backend_type_error("region_refinement", "refine_groundings()", backend)
    image = inputs["image"]
    groundings = inputs["groundings"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("region_refinement", "image", "PreparedImage", image)
    if not isinstance(groundings, Sequence):
        raise _input_type_error("region_refinement", "groundings", "a Sequence", groundings)
    return tuple(backend.refine_groundings(image, groundings))  # type: ignore[arg-type]


def _assemble_region_set(inputs: Mapping[str, object]) -> tuple[Region2D, ...]:
    """Assemble one frame's region set under :data:`REGION_SET_ASSEMBLY_POLICY`.

    Raises:
        TypeError: If an input is not the contract its name declares.
        ValueError: If two regions of the set share one ``region_id``.
    """
    discovered = inputs["discovered"]
    grounded = inputs["grounded"]
    refined = inputs.get("refined", ())
    if not isinstance(discovered, Sequence) or not all(
        isinstance(region, Region2D) for region in discovered
    ):
        raise _input_type_error("region_set_assembly", "discovered", "Region2D values", discovered)
    if not isinstance(grounded, Sequence) or not all(
        isinstance(item, RegionGroundingExecution) for item in grounded
    ):
        raise _input_type_error(
            "region_set_assembly", "grounded", "RegionGroundingExecution values", grounded
        )
    if not isinstance(refined, Sequence) or not all(
        isinstance(item, RegionRefinementExecution) for item in refined
    ):
        raise _input_type_error(
            "region_set_assembly", "refined", "RegionRefinementExecution values", refined
        )
    return assemble_regions(
        discovered=discovered,
        groundings=grounded,
        refinements=refined,
    )


def _run_feature_extractor(
    backend: object, inputs: Mapping[str, object]
) -> Sequence[VisualFeature]:
    if not isinstance(backend, FeatureExtractor):
        raise _backend_type_error("feature_extractor", "FeatureExtractor", backend)
    image = inputs["image"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("feature_extractor", "image", "PreparedImage", image)
    regions = inputs.get("regions", ())
    if not isinstance(regions, Sequence):
        raise _input_type_error("feature_extractor", "regions", "a Sequence", regions)
    return backend.extract(image, regions=regions)  # type: ignore[arg-type]


def _run_feature_resolution_enhancement(
    backend: object, inputs: Mapping[str, object]
) -> DenseFeatureMap:
    if not isinstance(backend, FeatureResolutionEnhancement):
        raise _backend_type_error(
            "feature_resolution_enhancement", "FeatureResolutionEnhancement", backend
        )
    dense_map = inputs["dense_map"]
    if not isinstance(dense_map, DenseFeatureMap):
        raise _input_type_error(
            "feature_resolution_enhancement", "dense_map", "DenseFeatureMap", dense_map
        )
    return enhance_feature_resolution(dense_map, enhancer=backend)


def _run_scene_interpretation(backend: object, inputs: Mapping[str, object]) -> SceneContext | None:
    if not isinstance(backend, _SceneRegionSemanticStageBackend):
        raise _backend_type_error(
            "scene_interpretation", "interpret_scene() and interpret_regions()", backend
        )
    image = inputs["image"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("scene_interpretation", "image", "PreparedImage", image)
    return backend.interpret_scene(image)


def _run_region_interpretation(
    backend: object, inputs: Mapping[str, object]
) -> Sequence[SemanticClaim]:
    if not isinstance(backend, _SceneRegionSemanticStageBackend):
        raise _backend_type_error(
            "region_interpretation", "interpret_scene() and interpret_regions()", backend
        )
    image = inputs["image"]
    regions = inputs["regions"]
    if not isinstance(image, PreparedImage):
        raise _input_type_error("region_interpretation", "image", "PreparedImage", image)
    if not isinstance(regions, Sequence):
        raise _input_type_error("region_interpretation", "regions", "a Sequence", regions)
    return backend.interpret_regions(image, regions)  # type: ignore[arg-type]


def _run_semantic_interpreter(
    backend: object, inputs: Mapping[str, object]
) -> SemanticInterpretationExecution:
    # NOTE: o import é tardio porque o módulo de request importa este pacote de volta.
    # Deixá-lo só sob TYPE_CHECKING fazia o isinstance abaixo levantar NameError em toda
    # execução, e a isolação de falha do grafo registrava o estágio como FAILED sem dizer por
    # quê: nenhuma run com semântica produzia claim.
    from contextmap.semantic_interpretation import (
        SemanticInterpretationRequest as SemanticRequestContract,
    )

    if not isinstance(backend, SemanticInterpreter):
        raise _backend_type_error("semantic_interpreter", "SemanticInterpreter", backend)
    request = inputs["request"]
    if not isinstance(request, SemanticRequestContract):
        raise _input_type_error(
            "semantic_interpreter", "request", "SemanticInterpretationRequest", request
        )
    return backend.interpret(request)


def _run_semantic_scorer(backend: object, inputs: Mapping[str, object]) -> Sequence[SemanticScore]:
    if not isinstance(backend, SemanticScorer):
        raise _backend_type_error("semantic_scorer", "SemanticScorer", backend)
    claims = inputs["claims"]
    features = inputs["features"]
    if not isinstance(claims, Sequence):
        raise _input_type_error("semantic_scorer", "claims", "a Sequence", claims)
    if not isinstance(features, Sequence):
        raise _input_type_error("semantic_scorer", "features", "a Sequence", features)
    return backend.score(claims, features)  # type: ignore[arg-type]


def _run_dense_region_pooling(
    backend: object, inputs: Mapping[str, object]
) -> Sequence[VisualFeature]:
    if not isinstance(backend, _DenseRegionPoolingStageBackend):
        raise _backend_type_error("dense_region_pooling", "DenseRegionPoolingStageBackend", backend)
    image, regions, dense_features = inputs["image"], inputs["regions"], inputs["dense_features"]
    if not isinstance(image, PreparedImage) or not isinstance(regions, Sequence):
        raise TypeError("dense_region_pooling requires a PreparedImage and region sequence")
    if not isinstance(dense_features, Sequence):
        raise _input_type_error(
            "dense_region_pooling", "dense_features", "a Sequence", dense_features
        )
    return backend.pool(image, regions, dense_features)  # type: ignore[arg-type]


_CAPABILITY_ADAPTERS: Mapping[str, Callable[[object, Mapping[str, object]], object]] = {
    "feature_resolution_enhancement": _run_feature_resolution_enhancement,
    "region_discovery": _run_region_discovery,
    "region_grounding": _run_region_grounding,
    "region_refinement": _run_region_refinement,
    "feature_extractor": _run_feature_extractor,
    "scene_interpretation": _run_scene_interpretation,
    "region_interpretation": _run_region_interpretation,
    "semantic_interpreter": _run_semantic_interpreter,
    "semantic_scorer": _run_semantic_scorer,
    "dense_region_pooling": _run_dense_region_pooling,
}

KNOWN_CAPABILITIES = frozenset(_CAPABILITY_ADAPTERS)
"""Capability names this module knows how to invoke on a resolved backend.

Adding a new executable capability means adding one adapter function
to ``_CAPABILITY_ADAPTERS`` — never a generic dispatch/plugin mechanism.
"""

_OWNED_STAGE_RULES: Mapping[str, Callable[[Mapping[str, object]], object]] = {
    "region_set_assembly": _assemble_region_set,
}
"""Capabilities this module computes itself, without a backend, from a stage's inputs."""

_OWNED_STAGE_PARAMETERS: Mapping[str, Mapping[str, object]] = {
    "region_set_assembly": {"policy_id": REGION_SET_ASSEMBLY_POLICY},
}
"""The exact parameters a stage of each owned rule declares: the version this module applies."""

_BASE_REQUIRED_INPUTS: Mapping[str, frozenset[str]] = {
    "region_discovery": frozenset({"image"}),
    "region_grounding": frozenset({"image"}),
    "region_refinement": frozenset({"image", "groundings"}),
    "region_set_assembly": frozenset({"discovered", "grounded"}),
    "feature_extractor": frozenset({"image"}),
    "feature_resolution_enhancement": frozenset({"dense_map"}),
    "scene_interpretation": frozenset({"image"}),
    "region_interpretation": frozenset({"image", "regions"}),
    "semantic_interpreter": frozenset({"request"}),
    "semantic_scorer": frozenset({"claims", "features"}),
    "dense_region_pooling": frozenset({"image", "regions", "dense_features"}),
}

_OPTIONAL_INPUTS: Mapping[str, frozenset[str]] = {
    "region_set_assembly": frozenset({"refined"}),
}

_INPUT_PRODUCER_CAPABILITIES: Mapping[str, frozenset[str]] = {
    "dense_map": frozenset({"dense_feature_map_source", "feature_resolution_enhancement"}),
    "image": frozenset({"image_preparation"}),
    "regions": frozenset({"region_discovery", "region_set_assembly"}),
    "groundings": frozenset({"region_grounding"}),
    "discovered": frozenset({"region_discovery"}),
    "grounded": frozenset({"region_grounding"}),
    "refined": frozenset({"region_refinement"}),
    "request": frozenset({"semantic_request"}),
    "claims": frozenset({"semantic_claims", "region_interpretation"}),
    "features": frozenset({"visual_features", "feature_extractor"}),
    "dense_features": frozenset({"feature_extractor"}),
}


def validate_pipeline_preset(preset: PipelinePreset) -> None:
    """Validate a preset's structure without constructing any backend.

    Checks stage dependency order, named input signatures and producer
    compatibility, that every referenced stage exists, that every
    backend stage declares a known capability, that a stage of a rule this
    module owns declares the version it applies, and that the graph has no
    cycle — before any heavy model is loaded.

    Args:
        preset: The preset to validate.

    Raises:
        PipelineConfigError: If ``preset`` has a duplicate ``stage_id``,
            a stage depending on an unknown ``stage_id``, a backend
            stage with an unrecognized ``capability``, an invalid input
            signature or producer, or a cycle.
    """
    by_id: dict[str, StageSpec] = {}
    for stage in preset.stages:
        if stage.stage_id in by_id:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: duplicate stage_id {stage.stage_id!r}"
            )
        by_id[stage.stage_id] = stage

    for stage in preset.stages:
        unknown_deps = stage.depends_on - by_id.keys()
        if unknown_deps:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: stage {stage.stage_id!r} depends on "
                f"unknown stage(s): {sorted(unknown_deps)}"
            )
        if stage.backend_id is not None and stage.capability not in _CAPABILITY_ADAPTERS:
            raise PipelineConfigError(
                f"preset {preset.preset_id!r}: stage {stage.stage_id!r} has no capability "
                f"adapter for {stage.capability!r}; known capabilities: "
                f"{sorted(_CAPABILITY_ADAPTERS)}"
            )
    _ensure_acyclic(preset.preset_id, by_id)

    for stage in preset.stages:
        if stage.backend_id is not None or _is_owned_rule(stage):
            _validate_stage_inputs(preset.preset_id, stage, by_id)
        if _is_owned_rule(stage):
            expected = _OWNED_STAGE_PARAMETERS[stage.capability]
            if dict(stage.parameters) != dict(expected):
                raise PipelineConfigError(
                    f"preset {preset.preset_id!r}: stage {stage.stage_id!r} must declare the "
                    f"parameters {dict(expected)!r}, the {stage.capability} policy this module "
                    f"applies; got {dict(stage.parameters)!r}"
                )


def _is_owned_rule(stage: StageSpec) -> bool:
    """Whether ``stage`` is computed by a rule of this module rather than seeded or a backend."""
    return stage.backend_id is None and stage.capability in _OWNED_STAGE_RULES


def _validate_stage_inputs(
    preset_id: str, stage: StageSpec, by_id: Mapping[str, StageSpec]
) -> None:
    """Validate one backend stage's named input contract."""
    required_inputs = _BASE_REQUIRED_INPUTS[stage.capability]
    if stage.capability == "feature_extractor":
        if stage.feature_scope is None:
            raise PipelineConfigError(
                f"preset {preset_id!r}: feature_extractor stage {stage.stage_id!r} "
                "must declare feature_scope"
            )
        if stage.feature_scope is FeatureScope.REGION:
            required_inputs |= {"regions"}
    elif stage.feature_scope is not None:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} declares feature_scope "
            "but is not a feature_extractor"
        )

    declared_inputs = frozenset(stage.inputs)
    missing_inputs = required_inputs - declared_inputs
    if missing_inputs:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} is missing required "
            f"input(s): {sorted(missing_inputs)}"
        )

    allowed_inputs = required_inputs | _OPTIONAL_INPUTS.get(stage.capability, frozenset())
    unexpected_inputs = declared_inputs - allowed_inputs
    if unexpected_inputs:
        raise PipelineConfigError(
            f"preset {preset_id!r}: stage {stage.stage_id!r} declares unexpected "
            f"input(s): {sorted(unexpected_inputs)}; allowed inputs: {sorted(allowed_inputs)}"
        )

    for input_name, upstream_id in stage.inputs.items():
        producer_capability = by_id[upstream_id].capability
        compatible_capabilities = _INPUT_PRODUCER_CAPABILITIES[input_name]
        if producer_capability not in compatible_capabilities:
            raise PipelineConfigError(
                f"preset {preset_id!r}: stage {stage.stage_id!r} input {input_name!r} "
                f"must come from capability {sorted(compatible_capabilities)}, not "
                f"{producer_capability!r}"
            )


def _ensure_acyclic(preset_id: str, by_id: Mapping[str, StageSpec]) -> None:
    resolved_ids: set[str] = set()
    remaining = dict(by_id)
    while remaining:
        ready = [
            stage_id for stage_id, stage in remaining.items() if stage.depends_on <= resolved_ids
        ]
        if not ready:
            raise PipelineConfigError(
                f"preset {preset_id!r}: cycle detected among stages: {sorted(remaining)}"
            )
        for stage_id in ready:
            resolved_ids.add(stage_id)
            del remaining[stage_id]


@dataclass(frozen=True, kw_only=True)
class ResolvedPipeline:
    """A preset with every backend stage's backend instance already constructed.

    Resolving happens once per run (heavy models load exactly once
    here); :meth:`build_stage_graph` is then cheap to call once per
    processed observation, only injecting that observation's source
    stage output(s) (e.g. its :class:`PreparedImage`).
    """

    preset: PipelinePreset
    backends: Mapping[str, object]

    def build_stage_graph(self, seed_outputs: Mapping[str, object]) -> tuple[StageDefinition, ...]:
        """Build one executable stage graph, injecting this observation's source output(s).

        Args:
            seed_outputs: ``stage_id`` -> output, one entry per source
                stage in the preset (``backend_id is None``), e.g.
                ``{"image_preparation": prepared_image}``.

        Returns:
            A :class:`StageDefinition` sequence ready for
            :func:`~contextmap.visual_perception.execution.stage_execution.execute_stage_graph`.

        Raises:
            PipelineConfigError: If ``seed_outputs`` is missing an entry
                for one of the preset's source stages.
        """
        source_stage_ids = {
            stage.stage_id
            for stage in self.preset.stages
            if stage.backend_id is None and not _is_owned_rule(stage)
        }
        missing = source_stage_ids - seed_outputs.keys()
        if missing:
            raise PipelineConfigError(
                f"preset {self.preset.preset_id!r}: missing seed output(s) for source "
                f"stage(s): {sorted(missing)}"
            )

        definitions: list[StageDefinition] = []
        for stage in self.preset.stages:
            if _is_owned_rule(stage):
                definitions.append(_owned_rule_stage_definition(stage))
                continue
            if stage.backend_id is None:
                definitions.append(_seeded_stage_definition(stage, seed_outputs[stage.stage_id]))
                continue
            definitions.append(_backend_stage_definition(stage, self.backends[stage.stage_id]))
        return tuple(definitions)

    def backend_provenance(self) -> Mapping[str, BackendProvenance]:
        """Report every resolved backend's provenance, keyed by ``stage_id``.

        Returns:
            One :class:`BackendProvenance` per backend stage, for
            persisting alongside a :class:`~contextmap.visual_perception.models.PerceptionRun`.

        Raises:
            TypeError: If a resolved backend does not report ``backend_provenance()``.
        """
        provenance: dict[str, BackendProvenance] = {}
        for stage_id, backend in self.backends.items():
            if not isinstance(backend, _ProvenanceCapable):
                raise TypeError(
                    f"backend of stage {stage_id!r} does not report backend_provenance(): "
                    f"{type(backend).__name__}"
                )
            provenance[stage_id] = backend.backend_provenance()
        return provenance

    def configuration_digest(self) -> str:
        """Compute a deterministic reproducibility digest for this resolved pipeline.

        Two resolutions produce the same digest if and only if they
        share the same preset content (``preset_id``, stages,
        parameters), the same resolved backend provenance (model
        identity, version, configuration fingerprint) for every backend
        stage and the same effective discovery identity (pass plan,
        normalization and discovery algorithms, defaults resolved) for
        every stage backend that reports one — i.e. the same preset
        resolved against a different model checkpoint, under another
        normalization default or with another mask remap yields a
        different digest.

        Returns:
            ``"sha256:<hex digest>"``.

        Raises:
            TypeError: If a resolved backend does not report ``backend_provenance()``.
        """
        payload = {
            "preset": encode_pipeline_preset(self.preset),
            "backend_provenance": {
                stage_id: asdict(provenance)
                for stage_id, provenance in self.backend_provenance().items()
            },
            "discovery_identity": {
                stage_id: backend.discovery_identity().to_dict()
                for stage_id, backend in self.backends.items()
                if isinstance(backend, _DiscoveryIdentityCapable)
            },
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def resolve_pipeline(
    preset: PipelinePreset, *, backend_factories: Mapping[str, StageBackendFactory]
) -> ResolvedPipeline:
    """Validate a preset, then construct every backend stage's backend exactly once.

    Args:
        preset: The preset to resolve.
        backend_factories: ``stage_id`` -> factory, one entry per
            backend stage in ``preset`` (``backend_id is not None``).
            Never called for a stage that does not survive validation.

    Returns:
        The resolved pipeline, ready for repeated
        :meth:`ResolvedPipeline.build_stage_graph` calls.

    Raises:
        PipelineConfigError: If ``preset`` fails :func:`validate_pipeline_preset`,
            or ``backend_factories`` is missing an entry for one of
            ``preset``'s backend stages.
    """
    validate_pipeline_preset(preset)

    backend_stage_ids = {stage.stage_id for stage in preset.stages if stage.backend_id is not None}
    missing = backend_stage_ids - backend_factories.keys()
    if missing:
        raise PipelineConfigError(f"missing backend factory for stage(s): {sorted(missing)}")

    backends = {
        stage.stage_id: backend_factories[stage.stage_id](stage.parameters)
        for stage in preset.stages
        if stage.backend_id is not None
    }
    return ResolvedPipeline(preset=preset, backends=backends)


def encode_stage_spec(stage: StageSpec) -> dict[str, Any]:
    """Encode a :class:`StageSpec` into a JSON-serializable dict."""
    return {
        "stage_id": stage.stage_id,
        "capability": stage.capability,
        "inputs": dict(stage.inputs),
        "backend_id": stage.backend_id,
        "optional": stage.optional,
        "parameters": dict(stage.parameters),
        "feature_scope": stage.feature_scope.value if stage.feature_scope is not None else None,
    }


def decode_stage_spec(record: Mapping[str, Any]) -> StageSpec:
    """Decode a :class:`StageSpec` from :func:`encode_stage_spec`'s output."""
    return StageSpec(
        stage_id=record["stage_id"],
        capability=record["capability"],
        inputs=dict(record["inputs"]),
        backend_id=record["backend_id"],
        optional=record["optional"],
        parameters=dict(record["parameters"]),
        feature_scope=(
            FeatureScope(record["feature_scope"]) if record["feature_scope"] is not None else None
        ),
    )


def encode_pipeline_preset(preset: PipelinePreset) -> dict[str, Any]:
    """Encode a :class:`PipelinePreset` into a JSON-serializable dict."""
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "preset_id": preset.preset_id,
        "description": preset.description,
        "stages": [encode_stage_spec(stage) for stage in preset.stages],
    }


def decode_pipeline_preset(record: Mapping[str, Any]) -> PipelinePreset:
    """Decode a :class:`PipelinePreset` from :func:`encode_pipeline_preset`'s output.

    Raises:
        PipelineConfigError: If ``record["schema_version"]`` is not
            understood by this module.
    """
    schema_version = record.get("schema_version")
    if schema_version != PIPELINE_SCHEMA_VERSION:
        raise PipelineConfigError(f"unsupported pipeline preset schema_version: {schema_version!r}")
    return PipelinePreset(
        preset_id=record["preset_id"],
        description=record.get("description", ""),
        stages=tuple(decode_stage_spec(stage) for stage in record["stages"]),
    )


def _seeded_stage_definition(stage: StageSpec, value: object) -> StageDefinition:
    def run(_ctx: Mapping[str, object], *, value: object = value) -> object:
        return value

    return StageDefinition(
        stage_id=stage.stage_id,
        capability=stage.capability,
        depends_on=stage.depends_on,
        run=run,
    )


def _owned_rule_stage_definition(stage: StageSpec) -> StageDefinition:
    rule = _OWNED_STAGE_RULES[stage.capability]
    input_names = dict(stage.inputs)

    def run(
        ctx: Mapping[str, object],
        *,
        rule: Callable[[Mapping[str, object]], object] = rule,
        input_names: Mapping[str, str] = input_names,
    ) -> object:
        return rule({name: ctx[upstream_id] for name, upstream_id in input_names.items()})

    return StageDefinition(
        stage_id=stage.stage_id, capability=stage.capability, depends_on=stage.depends_on, run=run
    )


def _backend_stage_definition(stage: StageSpec, backend: object) -> StageDefinition:
    adapter = _CAPABILITY_ADAPTERS[stage.capability]
    input_names = dict(stage.inputs)

    def run(
        ctx: Mapping[str, object],
        *,
        backend: object = backend,
        adapter: Callable[[object, Mapping[str, object]], object] = adapter,
        input_names: Mapping[str, str] = input_names,
    ) -> object:
        resolved_inputs = {name: ctx[upstream_id] for name, upstream_id in input_names.items()}
        return adapter(backend, resolved_inputs)

    return StageDefinition(
        stage_id=stage.stage_id, capability=stage.capability, depends_on=stage.depends_on, run=run
    )
