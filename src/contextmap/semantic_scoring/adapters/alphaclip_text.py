"""AlphaCLIP scorer text tower: hypotheses projected into the AlphaCLIP region-feature space.

AlphaCLIP changes only the image encoder: its alpha-conditioned projection was trained against
the text tower of its base CLIP checkpoint. This adapter satisfies
:class:`~contextmap.semantic_scoring.adapters._scoring_base.TextEmbeddingEncoder` with that
base text tower, loaded from the same verified base+alpha checkpoints as the region extractor,
and declares exactly the extractor's space, as
:func:`~contextmap.visual_perception.visual_features.adapters.alphaclip.alphaclip_embedding_space`
names it: other checkpoints, or another extractor ``code_version``, land in another space and
are refused by the scorer, never compared. It never selects a label and never runs the image
encoder.

Esta torre vive em ``semantic_scoring`` e não entre as features: ela codifica a hipótese de um
claim para a comparação, o que é scoring, e depende da infra de template daqui.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.model_runtime import (
    combined_checkpoint_fingerprint,
    resolve_checkpoint_file,
    validate_and_normalize_feature_values,
)
from contextmap.semantic_scoring.adapters._scoring_base import (
    TextEmbeddingBatch,
    render_text,
    validate_text_template,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    EmbeddingSpace,
    alphaclip_embedding_space,
    embedding_space_fingerprint,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AlphaClipTextBackendError(RuntimeError):
    """The text encoder could not execute its configured model."""


class AlphaClipTextDependencyError(AlphaClipTextBackendError):
    """Required text encoder dependencies are unavailable."""


class AlphaClipTextDeviceError(AlphaClipTextBackendError):
    """The configured text encoder device cannot be used."""


class AlphaClipTextModelLoadError(AlphaClipTextBackendError):
    """The configured text checkpoint cannot be loaded."""


class AlphaClipTextInferenceError(AlphaClipTextBackendError):
    """The text encoder returned invalid evidence."""


ALPHACLIP_REGION_SCORER_ID = "alphaclip_region_scorer"
"""Backend identity of the AlphaCLIP region scorer in :class:`BackendProvenance`."""

_TEXT_ADAPTER_VERSION = "alphaclip-base-text-projection-l2/1"
_SCORING_RULE = "cosine_similarity:region/1"


@dataclass(frozen=True, kw_only=True)
class AlphaClipTextConfig:
    """Effective configuration of the AlphaCLIP scorer's text tower.

    Attributes:
        model_name: AlphaCLIP/OpenAI CLIP architecture identity of the region features.
        base_checkpoint_path: Local base CLIP checkpoint under the checkpoint root.
        alpha_checkpoint_path: Local AlphaCLIP checkpoint under the checkpoint root.
        checkpoint_fingerprint: Expected combined SHA-256 identity of both files; the one of
            the region features, or the spaces differ and scoring is refused.
        text_template: How one hypothesis becomes the text the tower encodes, holding the
            ``{hypothesis}`` placeholder exactly once. It is required and part of the scorer
            identity.
        visual_code_version: ``code_version`` of the AlphaCLIP region extractor whose features
            are scored. It takes part in that space's identity
            (:func:`~contextmap.visual_perception.visual_features.adapters.alphaclip.alphaclip_embedding_space`),
            so it must equal the extractor's.
        device: Requested ``"cpu"``, ``"cuda"``, or ``"mps"`` device.
        precision: Inference dtype, ``"float32"`` or ``"float16"``. The embeddings are
            normalized in ``float32`` whatever the inference precision.
        max_batch_size: Largest number of texts sent to the tower in one inference call.
    """

    model_name: str
    base_checkpoint_path: str
    alpha_checkpoint_path: str
    checkpoint_fingerprint: str
    text_template: str
    visual_code_version: str = "1"
    device: str = "cpu"
    precision: str = "float32"
    max_batch_size: int = 32

    def __post_init__(self) -> None:
        """Validate the checkpoint identity, the template and the execution settings."""
        for name, value in (
            ("model_name", self.model_name),
            ("base_checkpoint_path", self.base_checkpoint_path),
            ("alpha_checkpoint_path", self.alpha_checkpoint_path),
            ("checkpoint_fingerprint", self.checkpoint_fingerprint),
            ("visual_code_version", self.visual_code_version),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")
        for name, reference in (
            ("base_checkpoint_path", self.base_checkpoint_path),
            ("alpha_checkpoint_path", self.alpha_checkpoint_path),
        ):
            path = Path(reference)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a path relative to the checkpoint root")
        validate_text_template(self.text_template)
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if self.precision not in {"float32", "float16"}:
            raise ValueError("precision must be one of: float32, float16")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")


class AlphaClipTextRuntime(Protocol):
    """Internal replaceable boundary around the AlphaCLIP base text tower."""

    def projection_dimension(self) -> int:
        """Return the dimension of the tower's projected text embeddings."""
        ...

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        """Encode rendered texts, in order, into rows of projected, unnormalized embeddings."""
        ...


def alphaclip_region_scorer_provenance(config: AlphaClipTextConfig) -> BackendProvenance:
    """Return the identity of the AlphaCLIP region scorer that ``config`` configures.

    Its version is the base+alpha checkpoint fingerprint, the identity of the weights; the
    configuration fingerprint resolves every setting that changes a score value, the text
    template and the scoring rule included. It is known before any model loads.

    Args:
        config: The text tower configuration.

    Returns:
        The ``semantic_scorer`` provenance every score of this scorer carries.
    """
    return BackendProvenance(
        backend_id=ALPHACLIP_REGION_SCORER_ID,
        capability="semantic_scorer",
        provider="SunzeY/AlphaCLIP",
        model=config.model_name,
        version=config.checkpoint_fingerprint,
        configuration_fingerprint=_configuration_fingerprint(config),
    )


class AlphaClipTextEncoder:
    """Project hypotheses into the space of one AlphaCLIP region extractor's features."""

    def __init__(self, *, config: AlphaClipTextConfig, runtime: AlphaClipTextRuntime) -> None:
        """Create the encoder over a text runtime; nothing is loaded here."""
        self._config = config
        self._runtime = runtime

    def embedding_space(self) -> EmbeddingSpace:
        """Return the space the encoded hypotheses live in; loads the tower if needed."""
        return self._space(self._runtime.projection_dimension(), "l2")

    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        """Render, encode and l2-normalize hypotheses in input order.

        Raises:
            AlphaClipTextInferenceError: If there is no hypothesis, or the runtime returns another
                number of rows, another dimension or dtype than declared, or a vector that
                cannot be l2-normalized.
        """
        import numpy as np

        if not hypotheses:
            raise AlphaClipTextInferenceError("at least one hypothesis is required")
        texts = tuple(render_text(self._config.text_template, item) for item in hypotheses)
        native = self._runtime.encode(texts)
        if native.ndim != 2 or native.shape[0] != len(texts):
            raise AlphaClipTextInferenceError(
                f"text runtime returned {native.shape[0] if native.ndim else 0} embeddings for "
                f"{len(texts)} texts"
            )
        dimension = self._runtime.projection_dimension()
        if native.shape[1] != dimension:
            raise AlphaClipTextInferenceError(
                f"text runtime returned dimension {native.shape[1]}, declared {dimension}"
            )
        if str(native.dtype) != self._config.precision:
            raise AlphaClipTextInferenceError(
                f"text runtime dtype {native.dtype!s} does not match precision "
                f"{self._config.precision!r}"
            )
        # A norma é calculada em float32: o scorer exige norma 1 com tolerância de 1e-5.
        values, normalization = validate_and_normalize_feature_values(
            native.astype(np.float32), l2_normalize=True, error_type=AlphaClipTextInferenceError
        )
        return TextEmbeddingBatch(
            values=values,
            embedding_space_id=embedding_space_fingerprint(
                self._space(int(dimension), normalization)
            ),
            normalization=normalization,
        )

    def _space(self, dimension: int, normalization: str) -> EmbeddingSpace:
        return alphaclip_embedding_space(
            model_name=self._config.model_name,
            code_version=self._config.visual_code_version,
            checkpoint_fingerprint=self._config.checkpoint_fingerprint,
            dimension=dimension,
            normalization=normalization,
        )


class OfficialAlphaClipTextRuntime:
    """Lazy wrapper around the official ``alpha_clip`` package's text encoder."""

    def __init__(self, *, config: AlphaClipTextConfig, checkpoint_root: Path) -> None:
        """Create a runtime without importing SDKs or loading checkpoints.

        Args:
            config: The text tower configuration.
            checkpoint_root: Directory the configured checkpoint paths are relative to.
        """
        self._config = config
        self._checkpoint_root = checkpoint_root.resolve()
        self._torch: Any = None
        self._alpha_clip: Any = None
        self._model: Any = None

    def projection_dimension(self) -> int:
        """Return the output dimension of the base text projection."""
        self._ensure_loaded()
        return int(self._model.text_projection.shape[1])

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        """Encode texts in batches of at most ``max_batch_size``, rows in input order.

        Raises:
            AlphaClipTextInferenceError: If a text is longer than the tower's context (it is
                refused, never truncated) or inference fails.
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
        except AlphaClipTextInferenceError:
            raise
        except Exception as error:
            raise AlphaClipTextInferenceError(
                f"AlphaCLIP text inference failed: {error}"
            ) from error

    def _encode_batch(self, texts: Sequence[str]) -> NDArray[Any]:
        """Tokenize one batch without truncation and return its projected embeddings."""
        try:
            tokens = self._alpha_clip.tokenize(list(texts), truncate=False)
        except RuntimeError as error:
            raise AlphaClipTextInferenceError(
                f"a text does not fit the AlphaCLIP text context and is never truncated: {error}"
            ) from error
        with self._torch.inference_mode():
            encoded = self._model.encode_text(tokens.to(self._config.device))
        array: NDArray[Any] = encoded.detach().to("cpu").numpy()
        return array

    def _ensure_loaded(self) -> None:
        """Import SDKs, verify the local checkpoints and load the model lazily."""
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            alpha_clip = importlib.import_module("alpha_clip")
        except ModuleNotFoundError as error:
            raise AlphaClipTextDependencyError(
                "the AlphaCLIP text tower requires torch and alpha_clip in the runtime "
                f"environment (missing module: {error.name or 'unknown'!r}; the official "
                "alpha_clip package also imports loralib and pkg_resources)"
            ) from error
        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise AlphaClipTextDeviceError("configured CUDA device is unavailable")
        if self._config.device == "mps" and not torch.backends.mps.is_available():
            raise AlphaClipTextDeviceError("configured MPS device is unavailable")
        if self._config.device == "cpu" and self._config.precision == "float16":
            raise AlphaClipTextDeviceError("float16 AlphaCLIP inference is not supported on CPU")
        base_path = resolve_checkpoint_file(
            self._checkpoint_root,
            self._config.base_checkpoint_path,
            error_type=AlphaClipTextModelLoadError,
        )
        alpha_path = resolve_checkpoint_file(
            self._checkpoint_root,
            self._config.alpha_checkpoint_path,
            error_type=AlphaClipTextModelLoadError,
        )
        actual = combined_checkpoint_fingerprint(base_path, alpha_path)
        if actual != self._config.checkpoint_fingerprint:
            raise AlphaClipTextModelLoadError(
                f"checkpoint fingerprint mismatch: expected "
                f"{self._config.checkpoint_fingerprint}, found {actual}"
            )
        try:
            # Os mesmos pesos verificados do extrator de região; o texto usa a torre base.
            model, _preprocess = alpha_clip.load(
                str(base_path),
                alpha_vision_ckpt_pth=str(alpha_path),
                device=self._config.device,
            )
            model = model.half() if self._config.precision == "float16" else model.float()
            model.eval()
        except Exception as error:
            raise AlphaClipTextModelLoadError(
                f"could not load local AlphaCLIP checkpoints for {self._config.model_name}: {error}"
            ) from error
        self._torch = torch
        self._alpha_clip = alpha_clip
        self._model = model


def _configuration_fingerprint(config: AlphaClipTextConfig) -> str:
    """Hash every setting that changes a score, with the adapter and scoring-rule versions."""
    payload = {
        "model_name": config.model_name,
        "base_checkpoint_path": config.base_checkpoint_path,
        "alpha_checkpoint_path": config.alpha_checkpoint_path,
        "checkpoint_fingerprint": config.checkpoint_fingerprint,
        "text_template": config.text_template,
        "visual_code_version": config.visual_code_version,
        "device": config.device,
        "precision": config.precision,
        "max_batch_size": config.max_batch_size,
        "text_adapter": _TEXT_ADAPTER_VERSION,
        "scoring_rule": _SCORING_RULE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
