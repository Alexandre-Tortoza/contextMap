"""CLIP text tower: claim hypotheses projected into the joint space of the CLIP region features.

The adapter satisfies
:class:`~contextmap.semantic_scoring.adapters._scoring_base.TextEmbeddingEncoder` for the
region scorer (:class:`~contextmap.semantic_scoring.adapters._scoring_base.
ClipRegionSemanticScorer`). It renders every hypothesis through an explicit text template,
encodes it with the text tower of the same checkpoint and revision as the image features it
is compared with, and declares exactly their embedding space, as
:func:`~contextmap.visual_perception.visual_features.adapters.clip.clip_embedding_space` names
it: a text tower of another checkpoint or revision lands in another space and is refused by the
scorer, never compared. It never selects a label and never loads the image tower.

Esta torre vive em ``semantic_scoring`` e não entre as features: ela codifica a hipótese de um
claim para a comparação, o que é scoring, e depende da infra de template daqui.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.model_runtime import (
    validate_and_normalize_feature_values,
    validate_huggingface_commit_revision,
)
from contextmap.semantic_scoring.adapters._scoring_base import (
    TextEmbeddingBatch,
    render_text,
    validate_text_template,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    EmbeddingSpace,
    clip_embedding_space,
    embedding_space_fingerprint,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClipTextBackendError(RuntimeError):
    """The text encoder could not execute its configured model."""


class ClipTextDependencyError(ClipTextBackendError):
    """Required text encoder dependencies are unavailable."""


class ClipTextDeviceError(ClipTextBackendError):
    """The configured text encoder device cannot be used."""


class ClipTextModelLoadError(ClipTextBackendError):
    """The configured text checkpoint cannot be loaded."""


class ClipTextInferenceError(ClipTextBackendError):
    """The text encoder returned invalid evidence."""


CLIP_REGION_SCORER_ID = "clip_region_scorer"
"""Backend identity of the CLIP region scorer in :class:`BackendProvenance`."""

_TEXT_ADAPTER_VERSION = "clip-text-projection-l2/1"
_SCORING_RULE = "cosine_similarity:region/1"


@dataclass(frozen=True, kw_only=True)
class ClipTextConfig:
    """Effective configuration of the CLIP text tower that scores region claims.

    Attributes:
        checkpoint: CLIP model/checkpoint identity. It must be the one of the region features
            the scores compare with, or the spaces differ and scoring is refused.
        revision: Immutable repository revision as a full Git commit SHA, the one of the
            region features for the same reason.
        text_template: How one hypothesis becomes the text the tower encodes, holding the
            ``{hypothesis}`` placeholder exactly once, e.g. ``"a photo of a {hypothesis}"``.
            It changes every similarity, so it is required and part of the scorer identity.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference dtype, ``"float32"`` or ``"float16"``. The embeddings are
            normalized in ``float32`` whatever the inference precision.
        local_files_only: Prohibit implicit downloads when true.
        max_batch_size: Largest number of texts sent to the tower in one inference call.
    """

    checkpoint: str
    revision: str
    text_template: str
    device: str = "cpu"
    precision: str = "float32"
    local_files_only: bool = True
    max_batch_size: int = 32

    def __post_init__(self) -> None:
        """Validate the model identity, the template and the execution settings."""
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        validate_huggingface_commit_revision(self.revision)
        validate_text_template(self.text_template)
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")


class ClipTextRuntime(Protocol):
    """Internal replaceable boundary around the CLIP text tower."""

    def projection_dimension(self) -> int:
        """Return the dimension of the tower's projected text embeddings."""
        ...

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        """Encode rendered texts, in order, into rows of projected, unnormalized embeddings."""
        ...


def clip_region_scorer_provenance(config: ClipTextConfig) -> BackendProvenance:
    """Return the identity of the CLIP region scorer that ``config`` configures.

    It is known before any model loads, so a run can be identified by it. The fingerprint
    resolves every setting that changes a score value, including the text template and the
    scoring rule.

    Args:
        config: The text tower configuration.

    Returns:
        The ``semantic_scorer`` provenance every score of this scorer carries.
    """
    return BackendProvenance(
        backend_id=CLIP_REGION_SCORER_ID,
        capability="semantic_scorer",
        provider="huggingface",
        model=config.checkpoint,
        version=config.revision,
        configuration_fingerprint=_configuration_fingerprint(config),
    )


class ClipTextEncoder:
    """Project hypotheses into the joint space of one CLIP checkpoint's region features."""

    def __init__(self, *, config: ClipTextConfig, runtime: ClipTextRuntime | None = None) -> None:
        """Create the encoder; the bundled runtime loads nothing until it is first used."""
        self._config = config
        self._runtime = runtime or HuggingFaceClipTextRuntime(config=config)

    def embedding_space(self) -> EmbeddingSpace:
        """Return the space the encoded hypotheses live in; loads the tower if needed."""
        return clip_embedding_space(
            checkpoint=self._config.checkpoint,
            revision=self._config.revision,
            dimension=self._runtime.projection_dimension(),
            normalization="l2",
        )

    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        """Render, encode and l2-normalize hypotheses in input order.

        Raises:
            ClipTextInferenceError: If there is no hypothesis, or the runtime returns another
                number of rows, another dimension or dtype than declared, or a vector that
                cannot be l2-normalized.
        """
        import numpy as np

        if not hypotheses:
            raise ClipTextInferenceError("at least one hypothesis is required")
        texts = tuple(render_text(self._config.text_template, item) for item in hypotheses)
        native = self._runtime.encode(texts)
        if native.ndim != 2 or native.shape[0] != len(texts):
            raise ClipTextInferenceError(
                f"text runtime returned {native.shape[0] if native.ndim else 0} embeddings for "
                f"{len(texts)} texts"
            )
        dimension = self._runtime.projection_dimension()
        if native.shape[1] != dimension:
            raise ClipTextInferenceError(
                f"text runtime returned dimension {native.shape[1]}, declared {dimension}"
            )
        if str(native.dtype) != self._config.precision:
            raise ClipTextInferenceError(
                f"text runtime dtype {native.dtype!s} does not match precision "
                f"{self._config.precision!r}"
            )
        # A norma é calculada em float32: o scorer exige norma 1 com tolerância de 1e-5.
        values, normalization = validate_and_normalize_feature_values(
            native.astype(np.float32), l2_normalize=True, error_type=ClipTextInferenceError
        )
        space = clip_embedding_space(
            checkpoint=self._config.checkpoint,
            revision=self._config.revision,
            dimension=int(dimension),
            normalization=normalization,
        )
        return TextEmbeddingBatch(
            values=values,
            embedding_space_id=embedding_space_fingerprint(space),
            normalization=normalization,
        )


class HuggingFaceClipTextRuntime:
    """Lazy Transformers runtime of the CLIP text tower and its projection only."""

    def __init__(self, *, config: ClipTextConfig) -> None:
        """Create a runtime without importing SDKs or loading weights."""
        self._config = config
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def projection_dimension(self) -> int:
        """Return the projection dimension the loaded checkpoint declares."""
        self._ensure_loaded()
        return int(self._model.config.projection_dim)

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        """Encode texts in batches of at most ``max_batch_size``, rows in input order.

        Raises:
            ClipTextInferenceError: If a text is longer than the tower accepts (it is refused,
                never truncated) or inference fails.
        """
        import numpy as np

        self._ensure_loaded()
        batch_size = self._config.max_batch_size
        try:
            return np.concatenate(
                [
                    self._encode_batch(texts[start : start + batch_size])
                    for start in range(0, len(texts), batch_size)
                ]
            )
        except ClipTextInferenceError:
            raise
        except Exception as error:
            raise ClipTextInferenceError(f"CLIP text inference failed: {error}") from error

    def _encode_batch(self, texts: Sequence[str]) -> NDArray[Any]:
        """Tokenize one batch without truncation and return its projected embeddings."""
        tokens = self._tokenizer(list(texts), padding=True, truncation=False, return_tensors="pt")
        length = int(tokens["input_ids"].shape[1])
        limit = int(self._model.config.max_position_embeddings)
        if length > limit:
            raise ClipTextInferenceError(
                f"a text of the batch has {length} tokens; the CLIP text tower accepts {limit} "
                "and a hypothesis is never truncated"
            )
        inputs = {name: value.to(self._config.device) for name, value in tokens.items()}
        with self._torch.inference_mode():
            encoded = self._model(**inputs).text_embeds
        array: NDArray[Any] = encoded.detach().to("cpu").numpy()
        return array

    def _ensure_loaded(self) -> None:
        """Import SDKs, validate the device and load the tokenizer and text tower lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise ClipTextDependencyError(
                "the CLIP text tower requires torch and transformers in the runtime environment"
            ) from error
        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise ClipTextDeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise ClipTextDeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise ClipTextDeviceError("float16 CLIP inference is not supported on CPU")
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
            )
            # Só a torre de texto e sua projeção: a torre de imagem nunca é carregada aqui.
            model = transformers.CLIPTextModelWithProjection.from_pretrained(
                self._config.checkpoint,
                revision=self._config.revision,
                local_files_only=self._config.local_files_only,
                dtype=getattr(torch, self._config.precision),
            )
            model = model.to(self._config.device)
            model.eval()
        except Exception as error:
            mode = "local cache" if self._config.local_files_only else "configured model source"
            raise ClipTextModelLoadError(
                f"could not load the text tower of {self._config.checkpoint}@"
                f"{self._config.revision} from {mode}: {error}"
            ) from error
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model


def _configuration_fingerprint(config: ClipTextConfig) -> str:
    """Hash every setting that changes a score, with the adapter and scoring-rule versions."""
    payload = {
        "checkpoint": config.checkpoint,
        "revision": config.revision,
        "text_template": config.text_template,
        "device": config.device,
        "precision": config.precision,
        "local_files_only": config.local_files_only,
        "max_batch_size": config.max_batch_size,
        "text_adapter": _TEXT_ADAPTER_VERSION,
        "scoring_rule": _SCORING_RULE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
