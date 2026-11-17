"""Shared cosine similarity scoring logic for CLIP-family text-visual models."""

from __future__ import annotations

import hashlib
import json
import math
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    SemanticClaim,
)
from contextmap.semantic_scoring.contracts import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    FeatureId,
    FeatureScope,
    VisualFeature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


TEXT_TEMPLATE_FIELD = "hypothesis"
"""The one placeholder a scorer text template holds, replaced by the claim hypothesis."""


class SemanticScoringError(ValueError):
    """Raised when canonical evidence cannot be scored without changing semantics."""


def validate_text_template(template: str) -> None:
    """Require a template that turns one hypothesis into one text, and nothing else.

    The template changes every similarity a text tower yields, so it is part of a scorer's
    identity: it holds the ``{hypothesis}`` placeholder exactly once, with no conversion or
    format specification, and no other field.

    Args:
        template: For example ``"a photo of a {hypothesis}"`` or ``"{hypothesis}"``.

    Raises:
        ValueError: If the template is not of that form.
    """
    problem = "text_template must hold the {hypothesis} placeholder exactly once and no other field"
    try:
        fields = [
            (field, conversion, spec)
            for _, field, spec, conversion in string.Formatter().parse(template)
            if field is not None
        ]
    except ValueError as error:
        raise ValueError(f"{problem}: {error}") from error
    if fields != [(TEXT_TEMPLATE_FIELD, None, "")]:
        raise ValueError(f"{problem}; got {template!r}")


def render_text(template: str, hypothesis: str) -> str:
    """Return the text a scorer's text tower encodes for one hypothesis.

    Args:
        template: A template accepted by :func:`validate_text_template`.
        hypothesis: The claim hypothesis, inserted verbatim.

    Returns:
        The rendered text.
    """
    return template.format(**{TEXT_TEMPLATE_FIELD: hypothesis})


@dataclass(frozen=True, kw_only=True)
class TextEmbeddingBatch:
    """SDK-free text embeddings returned by a scorer-specific runtime."""

    values: NDArray[Any]
    embedding_space_id: str
    normalization: str

    def __post_init__(self) -> None:
        """Validate batch shape and declared metadata."""
        if self.values.ndim != 2 or any(dimension <= 0 for dimension in self.values.shape):
            raise ValueError("text embedding values must have shape (hypothesis_count, dimension)")
        if not self.embedding_space_id:
            raise ValueError("text embedding_space_id must not be empty")
        if not self.normalization:
            raise ValueError("text normalization must not be empty")


class FeaturePayloadSource(Protocol):
    """Minimal lazy feature-payload boundary required by semantic scoring."""

    def load(
        self, source_observation_id: SourceObservationId, feature_id: FeatureId
    ) -> NDArray[Any]:
        """Load one canonical visual feature payload."""
        ...


class TextEmbeddingEncoder(Protocol):
    """Model-specific text projection into the configured visual space."""

    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        """Encode hypotheses in input order."""
        ...


class _CosineSemanticScorer:
    """Shared exact cosine rule for model-specific capability adapters."""

    _scope: FeatureScope

    def __init__(
        self,
        *,
        provenance: BackendProvenance,
        embedding_space_id: str,
        payloads: FeaturePayloadSource,
        text_encoder: TextEmbeddingEncoder,
    ) -> None:
        if provenance.capability != "semantic_scorer":
            raise ValueError("scorer provenance capability must be semantic_scorer")
        for field_name, value in (
            ("backend_id", provenance.backend_id),
            ("provider", provenance.provider),
            ("model", provenance.model),
            ("version", provenance.version),
            ("configuration_fingerprint", provenance.configuration_fingerprint),
        ):
            if value is None or not value.strip():
                raise ValueError(f"scorer provenance {field_name} must not be empty")
        if not embedding_space_id:
            raise ValueError("embedding_space_id must not be empty")
        self._provenance = provenance
        self._embedding_space_id = embedding_space_id
        self._payloads = payloads
        self._text_encoder = text_encoder

    def backend_provenance(self) -> BackendProvenance:
        """Return the immutable scorer/model identity."""
        return self._provenance

    def score(
        self, claims: Sequence[SemanticClaim], features: Sequence[VisualFeature]
    ) -> Sequence[SemanticScore]:
        """Score each claim against the one feature compatible with this adapter.

        A feature is compatible when it shares the adapter's ``FeatureScope``, is anchored to
        the claim's region, and declares the adapter's embedding space. Compatibility is a
        selection filter, not a validation: a feature of another embedding space is evidence
        for another consumer, so it is ignored rather than rejected. A claim with no compatible
        feature stays unscored; more than one compatible feature is an ambiguous configuration
        and is refused, because the adapter must not choose between them silently.
        """
        if not claims:
            return ()
        batch = self._text_encoder.encode(tuple(claim.hypothesis for claim in claims))
        if batch.values.shape[0] != len(claims):
            raise SemanticScoringError("text encoder returned the wrong hypothesis count")
        self._validate_space(batch.embedding_space_id, "text")
        if batch.normalization != "l2":
            raise SemanticScoringError("text embeddings must be declared l2-normalized")

        scores: list[SemanticScore] = []
        for claim_index, claim in enumerate(claims):
            candidates = self._candidate_features(claim, features)
            if len(candidates) > 1:
                raise SemanticScoringError(
                    "semantic scoring found multiple compatible features for claim "
                    f"{claim.claim_id!r} in embedding space {self._embedding_space_id!r}"
                )
            for feature in candidates:
                self._validate_feature(feature)
                visual = self._payloads.load(claim.source_observation_id, feature.feature_id)
                text = batch.values[claim_index]
                value = _normalized_dot(text, visual, expected_shape=feature.shape)
                scores.append(
                    SemanticScore(
                        score_id=_score_id(claim, feature, self._provenance),
                        claim_id=claim.claim_id,
                        feature_id=feature.feature_id,
                        score_type=SemanticScoreType.COSINE_SIMILARITY,
                        value=value,
                        calibrated_probability=None,
                        embedding_space_id=self._embedding_space_id,
                        source_observation_id=claim.source_observation_id,
                        region_id=claim.region_id,
                        provenance=self._provenance,
                    )
                )
        return tuple(scores)

    def _candidate_features(
        self, claim: SemanticClaim, features: Sequence[VisualFeature]
    ) -> tuple[VisualFeature, ...]:
        """Return the features this adapter may compare against ``claim``.

        Selection is by spatial anchoring (``scope`` and region) and representational
        compatibility (embedding space). It never rejects: an incompatible feature is simply
        not returned, so it stays available as evidence for another consumer.
        """
        return tuple(
            feature
            for feature in features
            if feature.scope is self._scope
            and feature.source_observation_id == claim.source_observation_id
            and self._matches_claim_scope(claim, feature)
            and feature.embedding_space_id == self._embedding_space_id
        )

    def _matches_claim_scope(self, claim: SemanticClaim, feature: VisualFeature) -> bool:
        if self._scope is FeatureScope.GLOBAL:
            return claim.region_id is None
        return claim.region_id is not None and feature.region_id == claim.region_id

    def _validate_space(self, embedding_space_id: str, source: str) -> None:
        if embedding_space_id != self._embedding_space_id:
            raise SemanticScoringError(
                f"{source} embedding space {embedding_space_id!r} does not match scorer "
                f"embedding space {self._embedding_space_id!r}"
            )

    def _validate_feature(self, feature: VisualFeature) -> None:
        self._validate_space(feature.embedding_space_id, "visual feature")
        if feature.normalization != "l2":
            raise SemanticScoringError("visual features must be declared l2-normalized")
        if len(feature.shape) != 1:
            raise SemanticScoringError("semantic scoring requires one-dimensional feature vectors")


def _normalized_dot(
    text: NDArray[Any], visual: NDArray[Any], *, expected_shape: tuple[int, ...]
) -> float:
    import numpy as np

    if tuple(visual.shape) != expected_shape or visual.ndim != 1:
        raise SemanticScoringError(
            f"visual payload shape {tuple(visual.shape)} does not match feature shape "
            f"{expected_shape}"
        )
    if text.ndim != 1 or text.shape != visual.shape:
        raise SemanticScoringError("text and visual embedding dimensions do not match")
    if not np.isfinite(text).all() or not np.isfinite(visual).all():
        raise SemanticScoringError("text and visual embeddings must contain only finite values")
    text_norm = float(np.linalg.norm(text))
    visual_norm = float(np.linalg.norm(visual))
    if not math.isclose(text_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise SemanticScoringError("text embedding is not l2-normalized")
    if not math.isclose(visual_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise SemanticScoringError("visual feature is not l2-normalized")
    value = float(np.dot(text, visual))
    return max(-1.0, min(1.0, value))


def _score_id(
    claim: SemanticClaim, feature: VisualFeature, provenance: BackendProvenance
) -> ScoreId:
    payload = json.dumps(
        {
            "claim_id": str(claim.claim_id),
            "feature_id": str(feature.feature_id),
            "semantic_interpretation_run_id": str(claim.semantic_interpretation_run_id),
            "scorer_id": provenance.backend_id,
            "configuration_fingerprint": provenance.configuration_fingerprint,
            "score_type": SemanticScoreType.COSINE_SIMILARITY.value,
        },
        sort_keys=True,
    ).encode("utf-8")
    return ScoreId(f"score-{hashlib.sha256(payload).hexdigest()[:24]}")
