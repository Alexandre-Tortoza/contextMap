"""Visual feature evidence: embeddings and dense maps from feature extraction.

Uma `VisualFeature` é evidência de estrutura visual produzida por extração
(DINOv2, CLIP, SigLIP2, etc). A feature carrega provenance completa, incluindo
espaço de embedding, forma/tipo, e referência ao payload armazenado.

Features são evidência, não crença: elas carregam provenance, espaço de embedding,
escopo, e referência a onde o payload foi armazenado.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NewType

from contextmap.ingestion import SourceObservationId
from contextmap.shared import BackendProvenance

if TYPE_CHECKING:
    from contextmap.region_perception import RegionId


FeatureId = NewType("FeatureId", str)
"""Identity of a VisualFeature, qualified by the producing visual artifact."""


class FeatureScope(Enum):
    """What a VisualFeature is a representation of."""

    DENSE = "dense"
    """Dense spatial map over entire frame."""

    GLOBAL = "global"
    """One vector for the whole frame."""

    REGION = "region"
    """Pooled feature for one specific region."""


class PoolingMethod(Enum):
    """Method used to aggregate dense features into region features."""

    AVERAGE = "average"
    """Average pooling over region mask."""

    MAX = "max"
    """Maximum pooling over region mask."""

    ATTENTION = "attention"
    """Attention-weighted pooling."""


EMBEDDING_SPACE_SCHEMA_VERSION = "0.1.0"
"""Schema version of the embedding-space encoding produced by :func:`encode_embedding_space`."""
# NOTE: Versão congelada pela v0.1.0 (docs/release-v0.1.0.md). Só pode mudar junto com uma
# transição declarada em CHANGELOG.md, verificada por tests/packaging/test_release_scope.py.


@dataclass(frozen=True, kw_only=True)
class EmbeddingSpace:
    """Complete metadata about an embedding space with full scientific provenance.

    Two embeddings are NOT interchangeable just because they have the same
    dimension. They must have the same family, model, version, checkpoint,
    and layer (for multi-layer models). Mixing incompatible embeddings produces
    silent scientific errors.

    Attributes:
        family: Model architecture family (e.g., "clip", "dinov2", "siglip").
        model: Model variant (e.g., "vit-l-14", "base", "large").
        version: Version identifier (e.g., "1", "1.0", "v2.0").
        dimension: Dimensionality of the embedding space.
        checkpoint: Optional specific checkpoint identifier
            (e.g., "openai/clip-vit-l-14", "facebook/dinov2-base").
        layer: Optional layer identifier for multi-layer models
            (e.g., layer number, "patch_tokens", "image_projection").
        normalization: Normalization scheme applied to embeddings
            (e.g., "l2", "none"). ``None`` when not documented or applied.
            Default is "l2" (L2-normalized).
    """

    family: str
    model: str
    version: str
    dimension: int
    checkpoint: str | None = None
    layer: str | None = None
    normalization: str | None = "l2"

    def __post_init__(self) -> None:
        """Validate embedding space metadata."""
        if not str(self.family).strip():
            raise ValueError("family must not be empty")
        if not str(self.model).strip():
            raise ValueError("model must not be empty")
        if not str(self.version).strip():
            raise ValueError("version must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")


class EmbeddingSpaceMismatchError(ValueError):
    """Raised when two incompatible features/embedding spaces are treated as comparable."""


def encode_embedding_space(space: EmbeddingSpace) -> dict[str, Any]:
    """Encode an EmbeddingSpace into a JSON-serializable dict."""
    return {
        "family": space.family,
        "model": space.model,
        "version": space.version,
        "dimension": space.dimension,
        "checkpoint": space.checkpoint,
        "layer": space.layer,
        "normalization": space.normalization,
    }


def decode_embedding_space(record: dict[str, Any]) -> EmbeddingSpace:
    """Decode an EmbeddingSpace from encode_embedding_space's output."""
    return EmbeddingSpace(
        family=record["family"],
        model=record["model"],
        version=record["version"],
        dimension=record["dimension"],
        checkpoint=record.get("checkpoint"),
        layer=record.get("layer"),
        normalization=record.get("normalization", "l2"),
    )


def embedding_space_fingerprint(space: EmbeddingSpace) -> str:
    """Compute the deterministic compatibility fingerprint of an embedding space.

    Two spaces yield the same fingerprint if and only if every field
    is equal.

    Args:
        space: The embedding space to fingerprint.

    Returns:
        "sha256:<hex digest>".
    """
    payload = json.dumps(encode_embedding_space(space), sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def ensure_compatible_embedding_spaces(a: EmbeddingSpace, b: EmbeddingSpace) -> None:
    """Validate two embedding spaces are the exact same identity.

    Args:
        a: First embedding space.
        b: Second embedding space.

    Raises:
        EmbeddingSpaceMismatchError: If a and b are not the same identity.
    """
    fingerprint_a = embedding_space_fingerprint(a)
    fingerprint_b = embedding_space_fingerprint(b)
    if fingerprint_a != fingerprint_b:
        raise EmbeddingSpaceMismatchError(
            f"incompatible embedding spaces: {a.family}/{a.model}@{a.version} ({fingerprint_a}) vs "
            f"{b.family}/{b.model}@{b.version} ({fingerprint_b})"
        )


def ensure_compatible_features(a: VisualFeature, b: VisualFeature) -> None:
    """Validate two features reference the exact same embedding space.

    Args:
        a: First feature.
        b: Second feature.

    Raises:
        EmbeddingSpaceMismatchError: If embedding spaces differ.
    """
    fingerprint_a = embedding_space_fingerprint_from_id(a.embedding_space_id)
    fingerprint_b = embedding_space_fingerprint_from_id(b.embedding_space_id)
    if fingerprint_a != fingerprint_b:
        raise EmbeddingSpaceMismatchError(
            f"incompatible embedding spaces: feature {a.feature_id!r} vs feature {b.feature_id!r}"
        )


def embedding_space_fingerprint_from_id(space_id: str) -> str:
    """Return embedding space fingerprint from VisualFeature's embedding_space_id.

    The embedding_space_id stored in VisualFeature is already the fingerprint
    (computed via embedding_space_fingerprint(space)). This function exists for
    type clarity when comparing features' embedding space IDs.

    Args:
        space_id: The embedding_space_id from a VisualFeature.

    Returns:
        The same space_id (which is already a fingerprint).
    """
    return space_id


@dataclass(frozen=True, kw_only=True)
class VisualFeature:
    """A visual feature vector/map produced by feature extraction.

    The full embedding-space compatibility contract (model family,
    checkpoint, dimension, normalization) is owned by the Feature
    Extraction capability; here, ``embedding_space_id`` is only an opaque
    reference — the same pattern Ingestion used for ``calibration_id``
    before the full calibration contract existed.

    Attributes:
        feature_id: Identity qualified by the producing visual artifact.
        source_observation_id: Physical observation that produced this feature.
        scope: Whether this feature is dense (a spatial map over the
            whole image), global (one vector for the whole image), or
            region (scoped to one ``Region2D``).
        embedding_space_id: Opaque reference to the embedding space this
            feature's values live in. Two features are only comparable
            when this reference matches.
        shape: Tensor shape of the feature payload, e.g. ``(768,)`` for a
            global vector or ``(64, 64, 384)`` for a dense map.
        dtype: Payload element type, e.g. ``"float32"``.
        payload_reference: Reference to the feature payload (e.g. a
            output file path) — never the raw tensor inline.
        provenance: Backend that produced this feature.
        region_id: The ``Region2D`` this feature is scoped to. Required
            when ``scope`` is ``REGION``, ``None`` otherwise.
        normalization: Human-readable normalization applied, e.g.
            ``"l2"``, ``"none"``. ``None`` when not documented.
    """

    feature_id: FeatureId
    source_observation_id: SourceObservationId
    scope: FeatureScope
    embedding_space_id: str
    shape: tuple[int, ...]
    dtype: str
    payload_reference: str
    provenance: BackendProvenance
    region_id: RegionId | None = None
    normalization: str | None = None

    def __post_init__(self) -> None:
        """Validate this feature's identity and that ``region_id`` matches ``scope``.

        Raises:
            ValueError: If ``feature_id`` is empty or blank, if ``scope`` is ``REGION``
                without a ``region_id``, or if ``region_id`` is set while ``scope`` is
                not ``REGION``.
        """
        # Uma feature sem identidade entraria no artifact indistinguível de outra, e o
        # índice de payloads é chaveado por ela.
        if not str(self.feature_id).strip():
            raise ValueError("feature_id must not be empty")
        if not str(self.source_observation_id).strip():
            raise ValueError("source_observation_id must not be empty")
        if self.scope is FeatureScope.REGION and self.region_id is None:
            raise ValueError("region_id is required when scope is REGION")
        if self.scope is not FeatureScope.REGION and self.region_id is not None:
            raise ValueError("region_id must be None unless scope is REGION")


def clip_embedding_space(
    *, checkpoint: str, revision: str, dimension: int, normalization: str
) -> EmbeddingSpace:
    """Return the identity of the joint image-text projection space of one CLIP checkpoint.

    CLIP projects images and texts into one joint space. The image adapter named that space
    after its image projection while it was the only producer; the text tower
    (:mod:`~contextmap.semantic_scoring.adapters.clip_text`) declares this very identity, so
    a text and an image of the same checkpoint and revision compare, and two checkpoints or
    revisions never do.

    Args:
        checkpoint: CLIP model/checkpoint identity.
        revision: Immutable repository revision.
        dimension: Projection dimension.
        normalization: Normalization applied to the vectors, e.g. ``"l2"``.

    Returns:
        The embedding space.
    """
    return EmbeddingSpace(
        family="clip",
        model=checkpoint,
        version=revision,
        checkpoint=f"{checkpoint}@{revision}",
        layer="image_projection",
        dimension=dimension,
        normalization=normalization,
    )


def alphaclip_embedding_space(
    *,
    model_name: str,
    code_version: str,
    checkpoint_fingerprint: str,
    dimension: int,
    normalization: str,
) -> EmbeddingSpace:
    """Return the identity of the space AlphaCLIP projects masked regions into.

    The alpha-conditioned image projection was trained against the base CLIP text tower, so
    the text tower (:mod:`~contextmap.semantic_scoring.adapters.alphaclip_text`) declares this
    very identity. It is distinct from every CLIP space, and it moves with the base+alpha
    checkpoint fingerprint and with the region extractor's ``code_version``.

    Args:
        model_name: AlphaCLIP/OpenAI CLIP architecture identity.
        code_version: ``code_version`` of the region extractor that produced the features.
        checkpoint_fingerprint: Combined SHA-256 identity of the base and alpha checkpoints.
        dimension: Projection dimension.
        normalization: Normalization applied to the vectors, e.g. ``"l2"``.

    Returns:
        The embedding space.
    """
    return EmbeddingSpace(
        family="alphaclip",
        model=model_name,
        version=code_version,
        checkpoint=checkpoint_fingerprint,
        layer="alpha_conditioned_image_projection",
        dimension=dimension,
        normalization=normalization,
    )


def encode_feature(feature: VisualFeature) -> dict[str, Any]:
    """Encode a :class:`VisualFeature` into a JSON-serializable dict."""
    return {
        "feature_id": str(feature.feature_id),
        "source_observation_id": str(feature.source_observation_id),
        "scope": feature.scope.value,
        "embedding_space_id": feature.embedding_space_id,
        "shape": list(feature.shape),
        "dtype": feature.dtype,
        "payload_reference": feature.payload_reference,
        "provenance": {
            name: getattr(feature.provenance, name)
            for name in (
                "backend_id",
                "capability",
                "provider",
                "model",
                "version",
                "configuration_fingerprint",
            )
        },
        "region_id": str(feature.region_id) if feature.region_id is not None else None,
        "normalization": feature.normalization,
    }


def decode_feature(record: dict[str, Any]) -> VisualFeature:
    """Decode a :class:`VisualFeature` from :func:`encode_feature`'s output."""
    from contextmap.region_perception import RegionId

    return VisualFeature(
        feature_id=FeatureId(record["feature_id"]),
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        scope=FeatureScope(record["scope"]),
        embedding_space_id=record["embedding_space_id"],
        shape=tuple(record["shape"]),
        dtype=record["dtype"],
        payload_reference=record["payload_reference"],
        provenance=BackendProvenance(**record["provenance"]),
        region_id=RegionId(record["region_id"]) if record["region_id"] is not None else None,
        normalization=record["normalization"],
    )
