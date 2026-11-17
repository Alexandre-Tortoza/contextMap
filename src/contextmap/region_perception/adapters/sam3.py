"""SAM3 adapter for the canonical Region Discovery capability."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from contextmap.model_runtime import model_release, names_device, verify_model_placement
from contextmap.shared import BackendProvenance, JsonScalar
from contextmap.visual_perception import PreparedImage

from ..discovery.region_discovery import (
    AuditedRegions,
    BackendDiagnostics,
    DiscoveryInput,
    DiscoveryOutput,
    DiscoveryPassConfig,
    RegionDiscoveryIdentity,
    discover_canonical_regions,
    validate_materialized_discovery_image,
)
from ..discovery.region_normalization import NormalizationConfig
from ..region_evidence import Region2D
from ..region_geometry import (
    BackendScore,
    BoundingBox,
    InlineMask,
    RegionCandidate,
    RegionProvenance,
    mask_bounding_box,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SUPPORTED_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})


class Sam3Strategy(StrEnum):
    """Explicit SAM3 discovery/query strategies supported by the adapter."""

    AUTOMATIC = "automatic"
    TEXT_PROMPT = "text_prompt"
    POINT_GRID = "point_grid"
    TRACKER = "tracker"
    PCS = "pcs"


@dataclass(frozen=True, slots=True)
class Sam3Config:
    """Effective configuration for one SAM3 Region Discovery adapter.

    ``precision`` is the inference precision the official runtime applies: ``float32``
    runs the SDK as loaded, while ``float16`` and ``bfloat16`` run it under
    ``torch.autocast``. The official SAM3 image model requires ``bfloat16``.
    ``model_version`` has no default: it enters provenance and the digest, so a run never
    records a placeholder instead of the checkpoint version that produced it.

    ``prompts`` is the ordered concept set one run queries (#662): ``text_prompt`` asks every
    phrase of every discovery pass, in this order. Each phrase is a query hypothesis, never a
    label. The list, order included, enters provenance and the digest, so changing a phrase or
    reordering them is another run.
    """

    checkpoint: str
    model_version: str
    device: str = "cpu"
    precision: str = "float32"
    strategy: Sam3Strategy = Sam3Strategy.AUTOMATIC
    score_threshold: float = 0.0
    mask_threshold: float = 0.5
    prompts: tuple[str, ...] = ()
    strategy_settings: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate strategy-specific configuration without hidden defaults.

        Raises:
            ValueError: If an identity is empty, the precision is not supported, a threshold
                leaves ``[0, 1]``, ``text_prompt`` has no phrase, ``automatic`` has any, a
                phrase is blank or repeated, or a strategy setting name is empty or repeated.
        """
        if not self.checkpoint or not self.model_version or not self.device or not self.precision:
            raise ValueError("SAM3 checkpoint, version, device, and precision must not be empty")
        if self.precision not in _SUPPORTED_PRECISIONS:
            supported = ", ".join(sorted(_SUPPORTED_PRECISIONS))
            raise ValueError(f"SAM3 precision must be one of: {supported}")
        _validate_unit_threshold(self.score_threshold, "score_threshold")
        _validate_unit_threshold(self.mask_threshold, "mask_threshold")
        if self.strategy is Sam3Strategy.TEXT_PROMPT and not self.prompts:
            raise ValueError("SAM3 text_prompt strategy requires at least one phrase in prompts")
        if self.strategy is Sam3Strategy.AUTOMATIC and self.prompts:
            raise ValueError("SAM3 automatic strategy does not consume text prompts")
        if any(not phrase.strip() for phrase in self.prompts):
            raise ValueError("SAM3 prompts must not be blank")
        if len(set(self.prompts)) != len(self.prompts):
            raise ValueError("SAM3 prompts must be unique")
        keys = [key for key, _ in self.strategy_settings]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("SAM3 strategy setting names must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Return a deterministic digest of the effective SAM3 configuration."""
        payload = {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "device": self.device,
            "precision": self.precision,
            "strategy": self.strategy.value,
            "score_threshold": self.score_threshold,
            "mask_threshold": self.mask_threshold,
            "prompts": list(self.prompts),
            "strategy_settings": list(self.strategy_settings),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(serialized.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Sam3NativeProposal:
    """SDK-isolated SAM3 proposal: Python scalars and the mask as an immutable ``InlineMask``.

    ``prompt`` is the configured phrase whose query produced the proposal, or ``None`` for a
    strategy that queries no phrase. ``proposal_id`` is unique within one discovery pass.
    """

    proposal_id: str
    box: tuple[float, float, float, float]
    mask: InlineMask
    score_name: str
    score: float
    query_id: str
    prompt: str | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        """Validate native scalar output before canonical normalization."""
        if not self.proposal_id or not self.score_name or not self.query_id:
            raise ValueError("SAM3 proposal, score, and query identities must not be empty")
        if not isfinite(self.score):
            raise ValueError("SAM3 native score must be finite")


@dataclass(frozen=True, slots=True)
class Sam3NativeOutput:
    """Structured scalar output returned by the internal SAM3 runtime seam."""

    proposals: tuple[Sam3NativeProposal, ...]
    warnings: tuple[str, ...] = ()
    metadata: tuple[tuple[str, JsonScalar], ...] = ()


class Sam3Runtime(Protocol):
    """Internal seam around a loaded SAM3 model/runtime.

    It receives the exact, already materialized and dimension-checked image of one discovery
    pass, so a provider-supplied runtime never needs to know where the run keeps its prepared
    images (#661).
    """

    def predict(
        self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
    ) -> Sam3NativeOutput:
        """Run the configured SAM3 strategy on ``image`` and return scalar-only proposals.

        Under ``text_prompt`` every proposal names, in ``prompt``, the configured phrase
        whose query produced it.
        """
        ...


class _Sam3ImageProcessor(Protocol):
    """Minimum official SAM3 image processor surface used by the runtime."""

    @property
    def model(self) -> object:
        """Loaded SAM3 model the processor runs, whose placement the runtime verifies."""
        ...

    @property
    def device(self) -> object:
        """Device the processor moves each image to before encoding it (default ``"cuda"``)."""
        ...

    def set_image(self, image: object) -> object:
        """Encode one image and return its inference state."""
        ...

    def set_confidence_threshold(self, threshold: float, state: object = None) -> object:
        """Apply the configured proposal threshold to the processor state."""
        ...

    def reset_all_prompts(self, state: object) -> None:
        """Remove every prompt and result from the state, keeping the encoded image."""
        ...

    def set_text_prompt(self, *, state: object, prompt: str) -> Mapping[str, object]:
        """Run text-conditioned image inference.

        The SDK writes the results into ``state`` and returns that same state, so they are
        only valid until the next prompt or reset on it.
        """
        ...


class Sam3ImageProcessorRuntime:
    """Execute the official SAM3 image processor text-prompt API over a concept set."""

    def __init__(
        self,
        *,
        processor: _Sam3ImageProcessor,
        autocast: Callable[[Sam3Config], AbstractContextManager[object]] | None = None,
    ) -> None:
        """Bind a loaded processor; each pass's image reaches :meth:`predict` materialized.

        Args:
            processor: Official image processor already built around the model.
            autocast: Builds the precision context the SDK runs in for one
                configuration. Defaults to ``torch.autocast`` for ``float16`` and
                ``bfloat16`` and to no context for ``float32``; tests inject a
                recording context. The SDK calls always run inside
                ``torch.inference_mode()`` as well, whatever the precision.
        """
        self._processor: _Sam3ImageProcessor | None = processor
        self._autocast = autocast or _torch_autocast
        self._released = False

    def release(self) -> None:
        """Drop the loaded processor and its model, free CUDA memory; inference is refused."""
        self._processor = None
        self._released = True
        model_release.free_cuda_memory()

    def predict(
        self, discovery_input: DiscoveryInput, image: object, config: Sam3Config
    ) -> Sam3NativeOutput:
        """Run supported official image inference on the pass's image, without a fallback.

        ``image`` is the materialized pass, a Pillow RGB image as the official processor takes.

        The pass image is encoded once (``set_image``) and every configured phrase is queried
        on that state, in order, with ``reset_all_prompts`` between phrases. Each phrase's
        output is detached before the next reset or phrase overwrites it. Proposal and query
        ids carry the phrase index and the proposal index within the phrase.

        Raises:
            ValueError: If the strategy is not ``text_prompt``, if no phrase is configured,
                if the processor moves images to another device than ``config.device``, or
                if the processor's model is not entirely on ``config.device`` or holds
                weights the configured precision does not accept: ``float32`` weights, or
                weights already in the configured autocast dtype.
        """
        if self._released:
            raise RuntimeError("the SAM3 runtime was released; build a new one to run inference")
        processor = self._processor
        assert processor is not None  # só None depois de release(), recusado acima
        if config.strategy is not Sam3Strategy.TEXT_PROMPT:
            raise ValueError("official SAM3 image runtime supports only text_prompt strategy")
        if not config.prompts:
            raise ValueError("official SAM3 image runtime requires text prompts")
        # O autocast realiza float16/bfloat16 sobre pesos float32 (o SDK oficial roda assim sob
        # bfloat16, #338) ou já no próprio dtype; pesos em outro dtype reduzido tornariam falsa
        # a precisão registrada. Em float32 não há autocast, então só pesos float32 servem.
        verify_model_placement(
            processor.model,
            device=config.device,
            precision=config.precision,
            backend="SAM3",
            accepted_dtypes=frozenset({"float32", config.precision}),
        )
        # O processor move cada imagem para o próprio device antes de codificá-la: um device
        # diferente do configurado rodaria a inferência fora do que a provenance registra.
        if not names_device(str(processor.device), config.device):
            raise ValueError(
                f"SAM3 processor moves images to {processor.device}, but the "
                f"configuration declares device {config.device!r}"
            )

        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        proposals: list[Sam3NativeProposal] = []
        with _torch_inference_mode(), self._autocast(config):
            state = processor.set_image(image)
            state = processor.set_confidence_threshold(config.score_threshold, state=state)
            if state is None:
                raise ValueError("SAM3 processor returned no inference state")
            for prompt_index, prompt in enumerate(config.prompts):
                if prompt_index:
                    # O SDK guarda prompt e resultados no próprio estado; o reset apaga os da
                    # frase anterior e mantém a imagem codificada uma vez neste passe.
                    processor.reset_all_prompts(state)
                output = processor.set_text_prompt(state=state, prompt=prompt)
                # A saída é o próprio estado: é destacada antes que a próxima frase a sobrescreva.
                proposals.extend(
                    _parse_image_processor_output(
                        output,
                        width=width,
                        height=height,
                        mask_threshold=config.mask_threshold,
                        prompt=prompt,
                        prompt_index=prompt_index,
                    )
                )
        return Sam3NativeOutput(
            proposals=tuple(proposals),
            metadata=(("runtime_api", "Sam3Processor.set_text_prompt"),),
        )


class Sam3RegionDiscovery:
    """Adapt configured SAM3 output to canonical RegionCandidate values."""

    def __init__(
        self,
        *,
        config: Sam3Config,
        runtime: Sam3Runtime,
        pass_images: Callable[[DiscoveryInput], object],
        pass_config: DiscoveryPassConfig | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        """Build the adapter with explicit configuration and no fallback runtime.

        Args:
            config: Effective SAM3 configuration.
            runtime: Seam of the loaded model; it receives each pass's materialized image.
            pass_images: Materializes the exact image of one discovery pass; the composition
                root binds it to the run's prepared images (``materialize_discovery_pass``).
            pass_config: Discovery passes; ``None`` selects the default full-frame pass.
            normalization_config: Normalization; ``None`` selects the default one.
        """
        self._config = config
        self._runtime = runtime
        self._pass_images = pass_images
        self._pass_config = pass_config
        self._normalization_config = normalization_config

    def release(self) -> None:
        """Release the runtime's model, when the runtime can release it (#689)."""
        model_release.release_runtime(self._runtime)

    def backend_provenance(self) -> BackendProvenance:
        """Return SAM3 identity in the Visual Perception Core contract."""
        return self.provenance_for(self._config)

    @staticmethod
    def provenance_for(config: Sam3Config) -> BackendProvenance:
        """Return the provenance a discovery built on ``config`` reports, without building it.

        The composition root names the discovery backend in the run's identity before the
        run's prepared images exist.
        """
        return BackendProvenance(
            backend_id="sam3",
            capability="region_discovery",
            provider="facebook",
            model=config.checkpoint,
            version=config.model_version,
            configuration_fingerprint=config.digest,
        )

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        """Discover and normalize canonical regions through the public port."""
        return self.discover_audited(image).regions

    def discovery_identity(self) -> RegionDiscoveryIdentity:
        """Return the effective pass plan, normalization and algorithms, defaults resolved."""
        return RegionDiscoveryIdentity.effective(self._pass_config, self._normalization_config)

    def discover_audited(self, image: PreparedImage) -> AuditedRegions:
        """Discover canonical regions together with the audit of every decision behind them."""
        return discover_canonical_regions(
            image,
            self,
            pass_config=self._pass_config,
            normalization_config=self._normalization_config,
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        """Materialize one pass and run exactly the selected SAM3 strategy on it.

        Raises:
            TypeError: If the materialized image exposes no inspectable dimensions.
            ValueError: If the materialized image does not have the pass's input dimensions,
                or a proposal mask does not match them.
        """
        started = perf_counter()
        image = self._pass_images(discovery_input)
        validate_materialized_discovery_image(image, discovery_input)
        native_output = self._runtime.predict(discovery_input, image, self._config)
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        candidates = tuple(
            self._normalize(proposal, discovery_input, width, height)
            for proposal in native_output.proposals
            if proposal.score >= self._config.score_threshold
        )
        duration_ms = (perf_counter() - started) * 1000
        metadata = (
            *native_output.metadata,
            ("raw_proposal_count", len(native_output.proposals)),
            ("filtered_proposal_count", len(native_output.proposals) - len(candidates)),
            ("checkpoint", self._config.checkpoint),
            ("model_version", self._config.model_version),
            ("device", self._config.device),
            ("precision", self._config.precision),
            ("strategy", self._config.strategy.value),
            *((f"prompt_{index:06d}", prompt) for index, prompt in enumerate(self._config.prompts)),
            ("config_digest", self._config.digest),
        )
        return DiscoveryOutput(
            candidates=candidates,
            diagnostics=BackendDiagnostics(
                duration_ms=duration_ms,
                proposal_count=len(candidates),
                warnings=native_output.warnings,
                metadata=metadata,
            ),
        )

    def _normalize(
        self,
        proposal: Sam3NativeProposal,
        discovery_input: DiscoveryInput,
        width: int,
        height: int,
    ) -> RegionCandidate:
        """Convert one scalar SAM3 proposal without semantic promotion.

        A proposal with mask pixels uses the tight half-open box of its mask as
        candidate geometry. The native box comes from a box head independent of
        the mask head, so it may leave the image or fail to contain the mask; it is
        kept as audit metadata instead of geometry. A proposal without mask pixels
        keeps the native box clamped to the pass, when any part of it is inside, and
        is rejected explicitly by normalization.

        The phrase that produced the proposal enters its query provenance and its
        ``text_prompt`` native metadata; it never becomes a label or native text.

        Raises:
            ValueError: If the mask dimensions differ from the pass, or if the proposal
                names no phrase under ``text_prompt`` or a phrase the configuration did
                not ask.
        """
        if (proposal.mask.width, proposal.mask.height) != (width, height):
            raise ValueError("SAM3 proposal mask dimensions must match discovery pass dimensions")
        prompt = proposal.prompt
        if (prompt is None and self._config.strategy is Sam3Strategy.TEXT_PROMPT) or (
            prompt is not None and prompt not in self._config.prompts
        ):
            raise ValueError(
                f"SAM3 proposal {proposal.proposal_id!r} answers phrase {prompt!r}, which is "
                f"not one of the configured phrases {list(self._config.prompts)!r}"
            )
        mask_box = mask_bounding_box(proposal.mask)
        bounding_box = mask_box or _clamp_box(proposal.box, width=width, height=height)
        native_metadata: tuple[tuple[str, JsonScalar], ...] = (
            *proposal.metadata,
            *(() if prompt is None else (("text_prompt", prompt),)),
            ("native_box_x_min", proposal.box[0]),
            ("native_box_y_min", proposal.box[1]),
            ("native_box_x_max", proposal.box[2]),
            ("native_box_y_max", proposal.box[3]),
        )
        if mask_box is not None:
            contains_mask = (
                proposal.box[0] <= mask_box.x_min
                and proposal.box[1] <= mask_box.y_min
                and proposal.box[2] >= mask_box.x_max
                and proposal.box[3] >= mask_box.y_max
            )
            native_metadata = (*native_metadata, ("native_box_contains_mask", contains_mask))
        query_parts = [self._config.strategy.value]
        if prompt is not None:
            query_parts.append(prompt)
        query_parts.append(proposal.query_id)
        return RegionCandidate(
            candidate_id=proposal.proposal_id,
            source_observation_id=discovery_input.prepared_image.source_observation_id,
            perception_run_id=discovery_input.perception_run_id,
            perception_result_id=discovery_input.perception_result_id,
            image_width=width,
            image_height=height,
            bounding_box=bounding_box,
            mask=proposal.mask,
            score=BackendScore(
                name=proposal.score_name,
                value=proposal.score,
                semantics=(
                    f"SAM3-native {proposal.score_name}; not calibrated across discovery backends"
                ),
            ),
            provenance=RegionProvenance(
                backend_id="sam3",
                backend_version=self._config.model_version,
                checkpoint=self._config.checkpoint,
                config_digest=self._config.digest,
                discovery_pass_id=discovery_input.discovery_pass.pass_id,
                native_proposal_id=proposal.proposal_id,
                query=":".join(query_parts),
            ),
            native_metadata=native_metadata,
        )


def _torch_inference_mode() -> AbstractContextManager[object]:
    """Return ``torch.inference_mode()``, so the SDK calls never record autograd state."""
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError("SAM3 inference requires torch inference_mode; install torch") from error
    return cast(AbstractContextManager[object], torch.inference_mode())


def _torch_autocast(config: Sam3Config) -> AbstractContextManager[object]:
    """Return the ``torch.autocast`` context that realizes the configured precision."""
    if config.precision == "float32":
        return nullcontext()
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"SAM3 {config.precision} inference requires torch autocast; install torch"
        ) from error
    return cast(
        AbstractContextManager[object],
        torch.autocast(
            device_type=config.device.split(":", 1)[0],
            dtype=getattr(torch, config.precision),
        ),
    )


def _clamp_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> BoundingBox | None:
    """Clip a native box to the pass, or return ``None`` when nothing of it is inside."""
    x_min, y_min = max(box[0], 0.0), max(box[1], 0.0)
    x_max, y_max = min(box[2], float(width)), min(box[3], float(height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _validate_unit_threshold(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"SAM3 {name} must be between zero and one")


def _parse_image_processor_output(
    output: Mapping[str, object],
    *,
    width: int,
    height: int,
    mask_threshold: float,
    prompt: str,
    prompt_index: int,
) -> tuple[Sam3NativeProposal, ...]:
    """Detach documented boxes, masks, and scores of one phrase's query from SAM3 tensors.

    Proposal and query ids are ``sam3-<phrase index>-<proposal index>`` and
    ``text-prompt-<phrase index>-<proposal index>``, both zero-padded, so they are unique
    within a discovery pass and sort in phrase order.
    """
    boxes = _native_sequence(output.get("boxes"), "boxes")
    scores = _native_sequence(output.get("scores"), "scores")
    masks = _native_array(output.get("masks_logits", output.get("masks")), "masks")
    if masks.ndim == 0:
        raise TypeError("SAM3 masks must be a sequence")
    if not (len(boxes) == len(scores) == len(masks)):
        raise ValueError("SAM3 boxes, scores, and masks must have equal proposal counts")

    proposals: list[Sam3NativeProposal] = []
    for index, (box_value, score_value, mask_value) in enumerate(
        zip(boxes, scores, masks, strict=True)
    ):
        box = _numeric_sequence(box_value, "box", length=4)
        proposals.append(
            Sam3NativeProposal(
                proposal_id=f"sam3-{prompt_index:06d}-{index:06d}",
                box=cast(tuple[float, float, float, float], box),
                mask=_probability_mask(
                    mask_value,
                    width=width,
                    height=height,
                    threshold=mask_threshold,
                ),
                score_name="concept_score",
                score=_finite_number(score_value, "score"),
                query_id=f"text-prompt-{prompt_index:06d}-{index:06d}",
                prompt=prompt,
                metadata=(("mask_threshold", mask_threshold),),
            )
        )
    return tuple(proposals)


def _probability_mask(
    value: NDArray[Any], *, width: int, height: int, threshold: float
) -> InlineMask:
    """Threshold one proposal's probabilities (or boolean mask) into a binary mask.

    Raises:
        TypeError: If the values are not numeric.
        ValueError: If the dimensions differ from the discovery pass, or a value is not finite.
    """
    import numpy as np

    pixels = value
    if pixels.ndim == 3 and pixels.shape[0] == 1 and pixels.shape[1] == height:
        pixels = pixels[0]  # eixo de canal único do SDK
    if pixels.shape != (height, width):
        raise ValueError("SAM3 mask dimensions must match the discovery pass")
    if pixels.dtype.kind not in "biuf":
        raise TypeError("SAM3 mask value must be numeric")
    # Em float64, como a comparação com o float do Python sempre foi: em float32 um limiar como
    # 0.7 arredonda para baixo e o pixel igual a float32(0.7) viraria primeiro plano.
    probabilities = pixels.astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("SAM3 mask value must be finite")
    return InlineMask(probabilities >= threshold)


def _native_array(value: object, name: str) -> NDArray[Any]:
    """Detach an SDK value to a NumPy array, never as a Python object per element.

    A PyTorch tensor may live on the GPU and in ``bfloat16``, which NumPy cannot hold: it is moved
    to the CPU in float64 (exact for every float and boolean dtype), as the other backends detach
    their tensors. Arrays, nested sequences and ``tolist()``-only values are read directly.

    Raises:
        TypeError: If the value is not a sequence.
        ValueError: If its nested sequences are ragged.
    """
    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        return np.asarray(detach().to("cpu").double().numpy())
    native = value
    if not hasattr(value, "__array__") and hasattr(value, "tolist"):
        native = value.tolist()
    if not hasattr(native, "__array__") and (
        not isinstance(native, Sequence) or isinstance(native, (str, bytes))
    ):
        raise TypeError(f"SAM3 {name} must be a sequence")
    try:
        return np.asarray(native)
    except ValueError as error:  # propostas ou linhas de tamanhos diferentes
        raise ValueError("SAM3 mask dimensions must match the discovery pass") from error


def _native_sequence(value: object, name: str) -> Sequence[object]:
    native = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(native, Sequence) or isinstance(native, (str, bytes)):
        raise TypeError(f"SAM3 {name} must be a sequence")
    return cast(Sequence[object], native)


def _numeric_sequence(value: object, name: str, *, length: int) -> tuple[float, ...]:
    native = _native_sequence(value, name)
    if len(native) != length:
        raise ValueError(f"SAM3 {name} must contain {length} numbers")
    return tuple(_finite_number(item, name) for item in native)


def _finite_number(value: object, name: str) -> float:
    # bool é subclasse de int, mas um bool nativo não é score nem coordenada: rejeitado como no
    # SAM2 e no Florence-2 (#617).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"SAM3 {name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"SAM3 {name} must be finite")
    return result
