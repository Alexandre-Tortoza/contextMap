"""Tests for auditable semantic scoring over canonical visual features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.semantic_interpretation.serialization import decode_claim, encode_claim
from contextmap.semantic_scoring import (
    ScoreId,
    SemanticScore,
    SemanticScoreType,
    decode_semantic_score,
    encode_semantic_score,
)
from contextmap.semantic_scoring.adapters import (
    AlphaClipSemanticScorer,
    ClipRegionSemanticScorer,
    ClipSemanticScorer,
    SemanticScoringError,
    TextEmbeddingBatch,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import (
    PipelinePreset,
    StageSpec,
    execute_stage_graph,
    resolve_pipeline,
)
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

OBSERVATION_ID = SourceObservationId("frame-0001")
SEMANTIC_ID = SemanticInterpretationRunId("semantic-run-1")
SPACE_ID = "sha256:clip-space"
APPEARANCE_SPACE_ID = "sha256:dino-space"


def _claim(*, region_id: str | None = None) -> SemanticClaim:
    return SemanticClaim(
        claim_id=ClaimId("claim-0001"),
        source_observation_id=OBSERVATION_ID,
        semantic_interpretation_run_id=SEMANTIC_ID,
        hypothesis="wooden chair",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake-interpreter",
                capability="semantic_interpreter",
                provider="fake",
                model="fake",
                version="1",
            ),
            task_identity="region-labeling",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        region_id=None if region_id is None else region_id,  # type: ignore[arg-type]
    )


def _feature(*, scope: FeatureScope, region_id: str | None = None) -> VisualFeature:
    return VisualFeature(
        source_observation_id=OBSERVATION_ID,
        feature_id=FeatureId("feature-0001"),
        scope=scope,
        embedding_space_id=SPACE_ID,
        shape=(3,),
        dtype="float32",
        payload_reference="feature-0001.npy",
        provenance=BackendProvenance(
            backend_id="clip-visual",
            capability="feature_extractor",
            provider="openai",
            model="clip-vit-b-32",
            version="1",
        ),
        region_id=None if region_id is None else region_id,  # type: ignore[arg-type]
        normalization="l2",
    )


class _Payloads:
    def load(self, source_observation_id: SourceObservationId, feature_id: FeatureId) -> np.ndarray:
        assert source_observation_id == OBSERVATION_ID
        assert feature_id == FeatureId("feature-0001")
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class _TextEncoder:
    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        assert tuple(hypotheses) == ("wooden chair",)
        return TextEmbeddingBatch(
            values=np.asarray([[0.6, 0.8, 0.0]], dtype=np.float32),
            embedding_space_id=SPACE_ID,
            normalization="l2",
        )


def _scorer_provenance(backend_id: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        capability="semantic_scorer",
        provider="openai",
        model="clip-vit-b-32",
        version="1",
        configuration_fingerprint="sha256:scorer-config",
    )


def test_clip_scores_global_claim_with_raw_cosine_similarity() -> None:
    claim = _claim()
    feature = _feature(scope=FeatureScope.GLOBAL)
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )

    scores = scorer.score((claim,), (feature,))

    assert len(scores) == 1
    assert scores[0].value == pytest.approx(0.6)
    assert scores[0].score_type is SemanticScoreType.COSINE_SIMILARITY
    assert scores[0].calibrated_probability is None
    assert claim.confidence is None


def test_alphaclip_requires_matching_region_feature() -> None:
    claim = _claim(region_id="region-0001")
    feature = _feature(scope=FeatureScope.REGION, region_id="region-0002")
    scorer = AlphaClipSemanticScorer(
        provenance=_scorer_provenance("alphaclip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )

    assert scorer.score((claim,), (feature,)) == ()


class _TextTable:
    """Return a fixed unit vector per hypothesis, in the order asked."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, hypotheses: Sequence[str]) -> TextEmbeddingBatch:
        self.calls.append(tuple(hypotheses))
        return TextEmbeddingBatch(
            values=np.asarray([self._vectors[item] for item in hypotheses], dtype=np.float32),
            embedding_space_id=SPACE_ID,
            normalization="l2",
        )


class _RegionPayloads:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def load(self, source_observation_id: SourceObservationId, feature_id: FeatureId) -> np.ndarray:
        assert source_observation_id == OBSERVATION_ID
        return np.asarray(self._vectors[str(feature_id)], dtype=np.float32)


def _region_feature(feature_id: str, region_id: str) -> VisualFeature:
    return VisualFeature(
        **{
            **_feature(scope=FeatureScope.REGION, region_id=region_id).__dict__,
            "feature_id": FeatureId(feature_id),
        }
    )


def _region_feature_in_space(
    feature_id: str, region_id: str, embedding_space_id: str
) -> VisualFeature:
    return VisualFeature(
        **{
            **_region_feature(feature_id, region_id).__dict__,
            "embedding_space_id": embedding_space_id,
        }
    )


def test_clip_region_scorer_scores_every_hypothesis_of_a_region_against_its_own_feature() -> None:
    primary = _claim(region_id="region-0001")
    alternative = SemanticClaim(
        **{
            **primary.__dict__,
            "claim_id": ClaimId("claim-0002"),
            "hypothesis": "office chair",
            "role": HypothesisRole.ALTERNATIVE,
        }
    )
    unfeatured = SemanticClaim(
        **{**primary.__dict__, "claim_id": ClaimId("claim-0003"), "region_id": "region-0003"}
    )
    scene = SemanticClaim(**{**_claim().__dict__, "claim_id": ClaimId("claim-0004")})
    text = _TextTable(
        {"wooden chair": [0.6, 0.8, 0.0], "office chair": [0.0, 0.0, 1.0]},
    )
    scorer = ClipRegionSemanticScorer(
        provenance=_scorer_provenance("clip-region-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_RegionPayloads({"feature-a": [1.0, 0.0, 0.0], "feature-b": [0.0, 1.0, 0.0]}),
        text_encoder=text,
    )

    scores = scorer.score(
        (primary, alternative, unfeatured, scene),
        (_region_feature("feature-a", "region-0001"), _region_feature("feature-b", "region-0002")),
    )

    # Uma claim de região é pontuada só contra a feature da própria região; a claim sem feature
    # e a de cena ficam sem score, nunca com um zero fabricado.
    assert [(str(score.claim_id), str(score.feature_id)) for score in scores] == [
        ("claim-0001", "feature-a"),
        ("claim-0002", "feature-a"),
    ]
    assert [score.value for score in scores] == pytest.approx([0.6, 0.0])
    assert all(score.score_type is SemanticScoreType.COSINE_SIMILARITY for score in scores)
    assert all(score.calibrated_probability is None for score in scores)
    assert primary.confidence is None and alternative.role is HypothesisRole.ALTERNATIVE


def test_feature_of_another_embedding_space_is_irrelevant_not_invalid() -> None:
    # Uma feature de outro espaço de embedding é evidência para outro consumidor; para este
    # scorer ela é irrelevante, então a claim fica sem score, sem erro.
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )
    feature = _feature(scope=FeatureScope.GLOBAL)

    assert (
        scorer.score(
            (_claim(),),
            (VisualFeature(**{**feature.__dict__, "embedding_space_id": "other"}),),
        )
        == ()
    )


def test_compatible_feature_with_invalid_normalization_still_fails() -> None:
    # Uma feature que declara o espaço do scorer mas contradiz o contrato (normalização) é
    # candidata e continua sendo evidência inválida: falha explícita.
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )
    feature = _feature(scope=FeatureScope.GLOBAL)

    with pytest.raises(SemanticScoringError, match="l2-normalized"):
        scorer.score((_claim(),), (VisualFeature(**{**feature.__dict__, "normalization": "none"}),))


def test_region_scorer_ignores_region_features_of_other_embedding_spaces() -> None:
    # CLIP crop e DINO pooled dividem escopo REGION e região, mas espaços distintos: o scorer
    # CLIP pontua só a feature CLIP e nunca toca no payload da feature de aparência.
    claim = _claim(region_id="region-0001")
    scorer = ClipRegionSemanticScorer(
        provenance=_scorer_provenance("clip-region-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_RegionPayloads({"clip-region": [1.0, 0.0, 0.0]}),
        text_encoder=_TextTable({"wooden chair": [0.6, 0.8, 0.0]}),
    )

    scores = scorer.score(
        (claim,),
        (
            _region_feature("clip-region", "region-0001"),
            _region_feature_in_space("dino-region", "region-0001", APPEARANCE_SPACE_ID),
        ),
    )

    assert [(str(s.claim_id), str(s.feature_id)) for s in scores] == [("claim-0001", "clip-region")]
    assert scores[0].value == pytest.approx(0.6)


def test_region_scorer_leaves_claim_unscored_when_only_incompatible_features_exist() -> None:
    claim = _claim(region_id="region-0001")
    scorer = ClipRegionSemanticScorer(
        provenance=_scorer_provenance("clip-region-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_RegionPayloads({}),
        text_encoder=_TextTable({"wooden chair": [0.6, 0.8, 0.0]}),
    )

    assert (
        scorer.score(
            (claim,),
            (_region_feature_in_space("dino-region", "region-0001", APPEARANCE_SPACE_ID),),
        )
        == ()
    )


def test_region_scorer_rejects_ambiguous_compatible_features() -> None:
    # Duas features no mesmo espaço para a mesma região são configuração ambígua: o scorer não
    # escolhe silenciosamente entre elas nem pontua ambas.
    claim = _claim(region_id="region-0001")
    scorer = ClipRegionSemanticScorer(
        provenance=_scorer_provenance("clip-region-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_RegionPayloads({"clip-a": [1.0, 0.0, 0.0], "clip-b": [0.0, 1.0, 0.0]}),
        text_encoder=_TextTable({"wooden chair": [0.6, 0.8, 0.0]}),
    )

    with pytest.raises(SemanticScoringError, match="multiple compatible"):
        scorer.score(
            (claim,),
            (
                _region_feature("clip-a", "region-0001"),
                _region_feature("clip-b", "region-0001"),
            ),
        )


def test_compatible_region_feature_with_inconsistent_payload_still_fails() -> None:
    claim = _claim(region_id="region-0001")
    scorer = ClipRegionSemanticScorer(
        provenance=_scorer_provenance("clip-region-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_RegionPayloads({"clip-region": [1.0, 0.0]}),  # declara (3,), payload (2,)
        text_encoder=_TextTable({"wooden chair": [0.6, 0.8, 0.0]}),
    )

    with pytest.raises(SemanticScoringError, match="does not match feature shape"):
        scorer.score((claim,), (_region_feature("clip-region", "region-0001"),))


def test_semantic_score_is_serialized_separately_from_claim() -> None:
    claim = _claim()
    feature = _feature(scope=FeatureScope.GLOBAL)
    score = SemanticScore(
        score_id=ScoreId("score-0001"),
        claim_id=claim.claim_id,
        feature_id=feature.feature_id,
        score_type=SemanticScoreType.COSINE_SIMILARITY,
        value=-0.25,
        calibrated_probability=None,
        embedding_space_id=SPACE_ID,
        source_observation_id=OBSERVATION_ID,
        region_id=None,
        provenance=_scorer_provenance("clip-scorer"),
    )
    restored_claim = decode_claim(encode_claim(claim))

    # O score viaja pelo codec da própria semantic_scoring, nunca dentro do claim nem do
    # PerceptionResult: é essa separação que o teste protege.
    assert decode_semantic_score(encode_semantic_score(score)) == score
    assert restored_claim.confidence is None


def test_semantic_score_rejects_probability_fabrication_and_non_finite_value() -> None:
    values = {
        "score_id": ScoreId("score-0001"),
        "claim_id": ClaimId("claim-0001"),
        "feature_id": FeatureId("feature-0001"),
        "score_type": SemanticScoreType.COSINE_SIMILARITY,
        "value": 0.2,
        "calibrated_probability": None,
        "embedding_space_id": SPACE_ID,
        "source_observation_id": OBSERVATION_ID,
        "region_id": None,
        "provenance": _scorer_provenance("clip-scorer"),
    }
    with pytest.raises(ValueError, match="finite"):
        SemanticScore(**{**values, "value": float("nan")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="calibrated_probability"):
        SemanticScore(**{**values, "calibrated_probability": 1.2})  # type: ignore[arg-type]


def test_semantic_scorer_is_an_executable_stage_graph_capability() -> None:
    scorer = ClipSemanticScorer(
        provenance=_scorer_provenance("clip-scorer"),
        embedding_space_id=SPACE_ID,
        payloads=_Payloads(),
        text_encoder=_TextEncoder(),
    )
    preset = PipelinePreset(
        preset_id="semantic-scoring/1",
        stages=(
            StageSpec(stage_id="claims", capability="semantic_claims"),
            StageSpec(stage_id="features", capability="visual_features"),
            StageSpec(
                stage_id="scores",
                capability="semantic_scorer",
                inputs={"claims": "claims", "features": "features"},
                backend_id="clip-scorer",
            ),
        ),
    )
    resolved = resolve_pipeline(preset, backend_factories={"scores": lambda _parameters: scorer})

    outcomes = execute_stage_graph(
        resolved.build_stage_graph(
            {"claims": (_claim(),), "features": (_feature(scope=FeatureScope.GLOBAL),)}
        )
    )

    assert outcomes[-1].output[0].value == pytest.approx(0.6)  # type: ignore[index]
    outputs = {item.stage_id: item.output for item in outcomes}
    assert len(cast(Any, outputs["claims"])) == 1
