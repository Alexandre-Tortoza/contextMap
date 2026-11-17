"""Semantic scoring run artifact persistence with atomic publication and typed serialization.

Este módulo gerencia a persistência e leitura de artefatos de semantic scoring,
com publicação atômica (staging directory → rename final) e desserialização tipada
de SemanticScore.

Um SemanticScoringRun contém os scores produzidos pela execução de scoring,
com referências explícitas aos upstream runs (SemanticInterpretationRun,
VisualFeaturesRun) dos quais dependem.

Invariantes:
- Publicação atômica: ou o artifact completo é válido ou nada é commitado no path final
- Desserialização tipada: json.loads() → dict é substituído por encode/decode tipado
- Integridade: SHA-256 para cada arquivo, validação na leitura
- Imutabilidade: re-execução cria nova identidade, nunca sobrescreve
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from contextmap.semantic_scoring.contracts import SemanticScore, SemanticScoreType
from contextmap.shared import AtomicRunDirectory, FileEntry, check_file_inventory
from contextmap.visual_perception import (
    EmbeddingSpace,
    FeatureScope,
    decode_embedding_space,
    embedding_space_fingerprint,
    encode_embedding_space,
)

if TYPE_CHECKING:
    from contextmap.shared import BackendProvenance
    from contextmap.visual_perception import PerceptionResult

SemanticScoringRunId = str
"""Identity of one SemanticScoringRun."""


class SemanticScoringRunError(Exception):
    """Raised when semantic scoring artifact cannot be written, published, or read safely."""


@dataclass(frozen=True, kw_only=True)
class SemanticScoringLineage:
    """Lineage of a semantic scoring run: upstreams it depends on.

    Attributes:
        sequence_artifact_id: Canonical sequence this run processed.
        selection_id: Which observations/results were selected for scoring.
        perception_run_id: The perception run whose results were scored.
    """

    sequence_artifact_id: str
    selection_id: str
    region_artifact_id: str
    visual_evidence_artifact_id: str
    semantic_interpretation_artifact_id: str


class _ScoreCounts(NamedTuple):
    """Aggregated counts for a scoring run."""

    result_count: int
    region_claim_count: int
    scene_claim_count: int
    scored_claim_count: int
    unscored_region_claim_count: int
    score_count: int


class _ScoreSemantics(NamedTuple):
    """Semantics documentation for scores in the run.

    ``value_range`` is part of the contract, not decoration: a cosine similarity lives in
    [-1, 1] and is not a probability, and a consumer that thresholds or fuses scores needs
    the range the producer declared rather than one it assumes (AGENTS.md §5).
    """

    score_type: str
    value_range: tuple[float, float]
    calibrated_probability: str | None
    statement: str


def _decode_score_semantics(record: dict[str, Any]) -> _ScoreSemantics:
    """Rebuild the score semantics from the published manifest.

    ``value_range`` comes back from JSON as a list; keeping it one would make a manifest
    read from disk compare unequal to the manifest the writer returned for the same run.
    """
    return _ScoreSemantics(
        score_type=record["score_type"],
        value_range=(float(record["value_range"][0]), float(record["value_range"][1])),
        calibrated_probability=record.get("calibrated_probability"),
        statement=record["statement"],
    )


@dataclass(frozen=True, kw_only=True)
class RunArtifactFileEntry:
    """One file in a run's inventory.

    Attributes:
        path: Path relative to the run directory, with `/` separators.
        size_bytes: Size of the file in bytes.
        content_hash: ``"sha256:<hex digest>"`` of the content.
    """

    path: str
    size_bytes: int
    content_hash: str


@dataclass(frozen=True, kw_only=True)
class SemanticScoringRunManifest:
    """Metadata and lineage for a semantic scoring run.

    Attributes:
        run_id: Unique identity of this scoring run.
        run_index: Monotonic index within sequence scope.
        lineage: Upstream artifact dependencies.
        scorer: BackendProvenance of the scorer used.
        embedding_space: Embedding space identity.
        embedding_space_id: Fingerprint of the embedding space.
        text_template: Text template used for scoring (e.g., "a photo of a {hypothesis}").
        scope: Feature scope (region or global).
        score_type: Type of scores produced (e.g., COSINE_SIMILARITY).
        created_at: ISO 8601 UTC timestamp of run completion.
        code_version: Perception code identity, when available.
        counts: Aggregated counts of results, claims, and scores.
        score_semantics: Documentation of score semantics and calibration.
        file_inventory: The contractual files published, with sizes and hashes.
        schema_version: Semantic scoring artifact schema version.
    """

    run_id: SemanticScoringRunId
    run_index: int
    lineage: SemanticScoringLineage
    scorer: BackendProvenance
    embedding_space: EmbeddingSpace
    embedding_space_id: str
    text_template: str
    scope: FeatureScope
    score_type: SemanticScoreType
    created_at: str
    code_version: str | None = None
    counts: _ScoreCounts | None = None
    score_semantics: _ScoreSemantics | None = None
    file_inventory: tuple[FileEntry, ...] = ()
    schema_version: str = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""
        return {
            "run_id": self.run_id,
            "run_index": self.run_index,
            "lineage": {
                "sequence_artifact_id": self.lineage.sequence_artifact_id,
                "selection_id": self.lineage.selection_id,
                "region_artifact_id": self.lineage.region_artifact_id,
                "visual_evidence_artifact_id": self.lineage.visual_evidence_artifact_id,
                "semantic_interpretation_artifact_id": (
                    self.lineage.semantic_interpretation_artifact_id
                ),
            },
            "scorer": {
                "backend_id": self.scorer.backend_id,
                "capability": self.scorer.capability,
                "provider": self.scorer.provider,
                "model": self.scorer.model,
                "version": self.scorer.version,
                "configuration_fingerprint": self.scorer.configuration_fingerprint,
            },
            "embedding_space": encode_embedding_space(self.embedding_space),
            "embedding_space_id": self.embedding_space_id,
            "text_template": self.text_template,
            "scope": self.scope.value,
            "score_type": self.score_type.value,
            "created_at": self.created_at,
            "code_version": self.code_version,
            "counts": None if self.counts is None else self.counts._asdict(),
            "score_semantics": (
                None if self.score_semantics is None else self.score_semantics._asdict()
            ),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticScoringRunManifest:
        """Reconstruct manifest from JSON-deserialized dict with validation."""
        schema_version = data.get("schema_version", "0.2.0")
        if schema_version != "0.2.0":
            raise SemanticScoringRunError(
                f"unsupported schema_version: {schema_version} (expected 0.2.0)"
            )

        lineage_data = data.get("lineage")
        if not lineage_data:
            raise SemanticScoringRunError("missing lineage in manifest")

        lineage = SemanticScoringLineage(
            sequence_artifact_id=lineage_data.get("sequence_artifact_id", ""),
            selection_id=lineage_data.get("selection_id", ""),
            region_artifact_id=lineage_data["region_artifact_id"],
            visual_evidence_artifact_id=lineage_data["visual_evidence_artifact_id"],
            semantic_interpretation_artifact_id=lineage_data["semantic_interpretation_artifact_id"],
        )

        scorer_data = data.get("scorer")
        if not scorer_data:
            raise SemanticScoringRunError("missing scorer in manifest")

        from contextmap.shared import BackendProvenance

        scorer = BackendProvenance(
            backend_id=scorer_data.get("backend_id", ""),
            capability=scorer_data.get("capability", ""),
            provider=scorer_data.get("provider", ""),
            model=scorer_data.get("model", ""),
            version=scorer_data.get("version", ""),
            configuration_fingerprint=scorer_data.get("configuration_fingerprint"),
        )

        return cls(
            run_id=data.get("run_id", ""),
            run_index=data.get("run_index", 0),
            lineage=lineage,
            scorer=scorer,
            embedding_space=decode_embedding_space(data["embedding_space"]),
            embedding_space_id=data.get("embedding_space_id", ""),
            text_template=data.get("text_template", ""),
            scope=FeatureScope(data["scope"]),
            score_type=SemanticScoreType(data["score_type"]),
            created_at=data.get("created_at", ""),
            code_version=data.get("code_version"),
            counts=(None if data.get("counts") is None else _ScoreCounts(**data["counts"])),
            score_semantics=(
                None
                if data.get("score_semantics") is None
                else _decode_score_semantics(data["score_semantics"])
            ),
            # O inventário é injetado no JSON pela publicação, não por to_dict; lê-lo aqui é o
            # que faz o manifest tipado dizer o mesmo que o artifact. Sem isso, quem deriva a
            # identidade de conteúdo do run dos hashes recebia lista vazia de um run completo.
            file_inventory=tuple(
                FileEntry(
                    path=entry["path"],
                    size_bytes=entry["size_bytes"],
                    content_hash=entry["content_hash"],
                )
                for entry in data.get("file_inventory", ())
            ),
            schema_version=schema_version,
        )


def encode_semantic_score(score: SemanticScore) -> dict[str, Any]:
    """Encode a SemanticScore to a JSON-serializable dict.

    Args:
        score: The SemanticScore to encode.

    Returns:
        A dict with all fields properly serialized.

    Raises:
        ValueError: If the score contains invalid values.
    """
    return {
        "score_id": str(score.score_id),
        "claim_id": str(score.claim_id),
        "feature_id": str(score.feature_id),
        "score_type": score.score_type.value,
        "value": score.value,
        "calibrated_probability": score.calibrated_probability,
        "embedding_space_id": score.embedding_space_id,
        "source_observation_id": str(score.source_observation_id),
        "region_id": None if score.region_id is None else str(score.region_id),
        "provenance": {
            "backend_id": score.provenance.backend_id,
            "capability": score.provenance.capability,
            "provider": score.provenance.provider,
            "model": score.provenance.model,
            "version": score.provenance.version,
            "configuration_fingerprint": score.provenance.configuration_fingerprint,
        },
    }


def decode_semantic_score(record: dict[str, Any]) -> SemanticScore:
    """Decode a SemanticScore from a JSON-deserialized dict with validation.

    Args:
        record: The dict to decode.

    Returns:
        A validated SemanticScore instance.

    Raises:
        SemanticScoringRunError: If required fields are missing or invalid.
    """
    from contextmap.ingestion import SourceObservationId
    from contextmap.region_perception import RegionId
    from contextmap.semantic_interpretation import (
        ClaimId,
    )
    from contextmap.shared import BackendProvenance
    from contextmap.visual_perception import (
        FeatureId,
    )

    required_fields = [
        "score_id",
        "claim_id",
        "feature_id",
        "score_type",
        "value",
        "embedding_space_id",
        "source_observation_id",
        "region_id",
        "provenance",
    ]

    missing = [f for f in required_fields if f not in record]
    if missing:
        raise SemanticScoringRunError(
            f"missing required fields in semantic score record: {missing}"
        )

    try:
        score_type = SemanticScoreType(record["score_type"])
    except (KeyError, ValueError) as e:
        raise SemanticScoringRunError(f"invalid score_type: {record.get('score_type')}") from e

    provenance_data = record.get("provenance", {})
    try:
        provenance = BackendProvenance(
            backend_id=provenance_data.get("backend_id", ""),
            capability=provenance_data.get("capability", ""),
            provider=provenance_data.get("provider", ""),
            model=provenance_data.get("model", ""),
            version=provenance_data.get("version", ""),
            configuration_fingerprint=provenance_data.get("configuration_fingerprint"),
        )
    except (KeyError, ValueError) as e:
        raise SemanticScoringRunError(f"invalid provenance in score record: {e}") from e

    try:
        score = SemanticScore(
            score_id=record["score_id"],
            claim_id=ClaimId(record["claim_id"]),
            feature_id=FeatureId(record["feature_id"]),
            score_type=score_type,
            value=record["value"],
            calibrated_probability=record.get("calibrated_probability"),
            embedding_space_id=record["embedding_space_id"],
            source_observation_id=SourceObservationId(record["source_observation_id"]),
            region_id=None if record["region_id"] is None else RegionId(record["region_id"]),
            provenance=provenance,
        )
        return score
    except (TypeError, ValueError) as e:
        raise SemanticScoringRunError(f"cannot construct SemanticScore: {e}") from e


class SemanticScoringRunWriter:
    """Writes semantic scores and metadata to an artifact directory with atomic publication.

    Preserves score data, lineage, and configuration in a structured, auditable format.
    Writes are atomic: either the full artifact is valid or nothing is committed to the
    final destination. All files are written to a temporary staging directory first,
    then the entire directory is renamed atomically to the final location.
    """

    SCHEMA_VERSION = "0.2.0"
    """Schema version for semantic scoring artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        sequence_name: str,
        run_id: SemanticScoringRunId,
        run_index: int,
        lineage: SemanticScoringLineage,
        scorer: BackendProvenance,
        embedding_space: EmbeddingSpace,
        text_template: str,
        code_version: str | None = None,
    ) -> None:
        """Initialize writer for the given output directory.

        Args:
            output_dir: Directory where the artifact will be written.
                       Must not already exist (atomic publication never replaces).
            sequence_name: Human-readable name of the sequence being processed.
            run_id: Unique identity of this scoring run.
            run_index: Monotonic index within sequence scope.
            lineage: Upstream artifact dependencies.
            scorer: BackendProvenance of the scorer used.
            embedding_space: The one space every score of the run was computed in.
            text_template: Text template used for scoring.
            code_version: Code version identifier, if available.

        Raises:
            SemanticScoringRunError: If output_dir already exists.
        """
        self.output_dir = Path(output_dir)
        self.sequence_name = sequence_name
        self.run_id = run_id
        self.run_index = run_index
        self.lineage = lineage
        self.scorer = scorer
        self.embedding_space = embedding_space
        self.text_template = text_template
        self.code_version = code_version

        # NOTE: a identidade vem da função canônica sobre o contrato, nunca de uma chave
        # acessória: um fingerprint ausente invalidaria a comparação que add_result faz, e
        # inventá-lo marcaria scores de espaços diferentes como se fossem do mesmo.
        self.embedding_space_id = embedding_space_fingerprint(embedding_space)

        self.scores: list[SemanticScore] = []
        self.results_processed: set[str] = set()
        # NOTE: os agregados são contados à medida que os resultados entram, porque o manifest
        # tem de distinguir um claim de região sem score de um score zero — o README da
        # capability diz que nenhum placeholder de valor zero é criado, e sem
        # unscored_region_claim_count essa distinção não chega a quem lê o artifact.
        self._region_claims = 0
        self._scene_claims = 0
        self._scored_claims = 0

    def add_result(self, result: PerceptionResult, scores: Sequence[SemanticScore]) -> None:
        """Process a perception result and its scores.

        Validates that:
        - The result belongs to the expected perception run
        - The result hasn't been processed twice
        - All scores are compatible with the embedding space and scorer
        - All scores reference claims and features in the result

        Args:
            result: The perception result containing claims and features.
            scores: The semantic scores for this result.

        Raises:
            SemanticScoringRunError: If validation fails.
        """
        if (
            result.region_artifact_id != self.lineage.region_artifact_id
            or result.visual_evidence_artifact_id != self.lineage.visual_evidence_artifact_id
            or result.semantic_interpretation_artifact_id
            != self.lineage.semantic_interpretation_artifact_id
            or result.sequence_artifact_id != self.lineage.sequence_artifact_id
        ):
            raise SemanticScoringRunError("view references another selected producer artifact")
        result_key = str(result.source_observation_id)
        if result_key in self.results_processed:
            raise SemanticScoringRunError(f"observation processed twice: {result_key}")
        self.results_processed.add(result_key)

        every_claim = (
            *result.claims,
            *(() if result.scene_context is None else result.scene_context.claims),
        )
        self._region_claims += sum(1 for claim in every_claim if claim.region_id is not None)
        self._scene_claims += sum(1 for claim in every_claim if claim.region_id is None)
        self._scored_claims += len({score.claim_id for score in scores})

        # Build lookup tables for validation
        claims = {str(claim.claim_id): claim for claim in result.claims}
        features = {str(feature.feature_id): feature for feature in result.features}
        claim_ids = set(claims)
        feature_ids = set(features)

        # Validate and accumulate scores
        for score in scores:
            # Check embedding space compatibility
            if score.embedding_space_id != self.embedding_space_id:
                raise SemanticScoringRunError(
                    f"embedding space mismatch in score {score.score_id}: "
                    f"expected {self.embedding_space_id}, got {score.embedding_space_id} (space)"
                )

            # Check scorer compatibility
            if score.provenance.backend_id != self.scorer.backend_id:
                raise SemanticScoringRunError(
                    f"scorer mismatch in score {score.score_id}: "
                    f"expected {self.scorer.backend_id}, got {score.provenance.backend_id} (scorer)"
                )

            # Check claim exists
            if str(score.claim_id) not in claim_ids:
                raise SemanticScoringRunError(
                    f"score references unknown claim_id {score.claim_id} (unknown claim_id)"
                )

            # Check feature exists
            if str(score.feature_id) not in feature_ids:
                raise SemanticScoringRunError(
                    f"score references unknown feature_id {score.feature_id} (region)"
                )

            # A pontuação de região só vale sobre a feature de região daquela própria
            # região: emparelhar uma claim com a feature de outra região, ou com uma feature
            # densa/global, produziria um número que não é sobre nada que a claim afirma.
            claim = claims[str(score.claim_id)]
            feature = features[str(score.feature_id)]
            if (
                feature.scope is not FeatureScope.REGION
                or claim.region_id is None
                or claim.region_id != feature.region_id
            ):
                raise SemanticScoringRunError(
                    f"score {score.score_id} pairs claim {score.claim_id} with feature "
                    f"{score.feature_id}, which is not the region feature of its region "
                    "(region)"
                )

            # Check calibrated_probability constraint
            if score.calibrated_probability is not None:
                raise SemanticScoringRunError(
                    f"score {score.score_id} has calibrated_probability (calibrated), "
                    "which is reserved for external calibration post-hoc"
                )

            if (
                score.source_observation_id != result.source_observation_id
                or score.region_id != claim.region_id
            ):
                raise SemanticScoringRunError(
                    "score anchor does not match its claim and observation"
                )
            from contextmap.semantic_scoring.contracts import ScoreId

            self.scores.append(replace(score, score_id=ScoreId(f"{self.run_id}--{score.score_id}")))

    def finalize(self) -> SemanticScoringRunManifest:
        """Write all accumulated data atomically and return the manifest.

        The entire artifact is written to a temporary staging directory first,
        then renamed atomically to the final output_dir. This ensures the
        final location either contains a complete valid artifact or remains
        non-existent.

        Returns:
            The published manifest with metadata and file inventory.

        Raises:
            SemanticScoringRunError: If output_dir already exists (never replaces)
                                    or if atomic publication fails.
        """
        from datetime import UTC, datetime

        if self.output_dir.exists():
            raise SemanticScoringRunError(
                f"run directory already exists (never replaces): {self.output_dir}"
            )

        now = datetime.now(UTC).isoformat()

        # Use AtomicRunDirectory for atomic publication
        with AtomicRunDirectory(self.output_dir, refuse_existing_early=True) as atomic:
            # Write scores to outputs/scores.jsonl
            scores_path = "outputs/scores.jsonl"
            with atomic.open_binary(scores_path, contractual=True) as f:
                for score in self.scores:
                    score_dict = encode_semantic_score(score)
                    line = json.dumps(score_dict, separators=(",", ":"))
                    f.write((line + "\n").encode("utf-8"))

            # Prepare manifest
            manifest = SemanticScoringRunManifest(
                run_id=self.run_id,
                run_index=self.run_index,
                lineage=self.lineage,
                scorer=self.scorer,
                embedding_space=self.embedding_space,
                embedding_space_id=self.embedding_space_id,
                text_template=self.text_template,
                scope=FeatureScope.REGION,
                score_type=SemanticScoreType.COSINE_SIMILARITY,
                created_at=now,
                code_version=self.code_version,
                counts=_ScoreCounts(
                    result_count=len(self.results_processed),
                    region_claim_count=self._region_claims,
                    scene_claim_count=self._scene_claims,
                    scored_claim_count=self._scored_claims,
                    unscored_region_claim_count=self._region_claims - self._scored_claims,
                    score_count=len(self.scores),
                ),
                score_semantics=_ScoreSemantics(
                    score_type=SemanticScoreType.COSINE_SIMILARITY.value,
                    value_range=(-1.0, 1.0),
                    calibrated_probability=None,
                    statement="cosine_similarity is not a probability; value in [-1, 1]",
                ),
                schema_version=self.SCHEMA_VERSION,
            )

            # NOTE: o inventário publicado entra no manifest devolvido. A publicação o injeta
            # no JSON em disco, e sem ele aqui o manifest tipado divergiria do artifact: quem
            # deriva a identidade do run de file_inventory via inventory_digest receberia uma
            # lista vazia, e dois runs de conteúdos diferentes teriam o mesmo content_hash.
            manifest = replace(manifest, file_inventory=atomic.inventory())
            manifest_dict = manifest.to_dict()
            atomic.publish(
                manifest=manifest_dict,
                readme="# Semantic Scoring Run\n\n"
                f"Run ID: {self.run_id}\n"
                f"Schema Version: {self.SCHEMA_VERSION}\n"
                f"Scores: {len(self.scores)}\n",
            )

        return manifest


class SemanticScoringRunReader:
    """Reads semantic scores from an artifact directory with integrity validation.

    Provides typed access to scores and metadata. Validates file integrity
    against manifest hashes before returning any data.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        """Initialize reader for the given artifact directory.

        Args:
            artifact_dir: Directory containing the semantic scoring artifact.

        Raises:
            SemanticScoringRunError: If the artifact doesn't exist or manifest is invalid.
        """
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise SemanticScoringRunError(f"artifact directory not found: {artifact_dir}")

        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise SemanticScoringRunError(
                f"manifest not found in artifact: {artifact_dir}/manifest.json"
            )

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise SemanticScoringRunError(f"cannot read manifest: {e}") from e

        self._manifest = SemanticScoringRunManifest.from_dict(manifest_data)
        self._manifest_data = manifest_data
        self._scores: list[SemanticScore] | None = None
        self._integrity_issues: list[str] | None = None

    @property
    def manifest(self) -> SemanticScoringRunManifest:
        """Get the run manifest with metadata and lineage."""
        return self._manifest

    def verify_integrity(self) -> list[str]:
        """Verify that all files match their manifest hashes.

        Returns:
            List of problems found (empty means all files are valid).
            Problems include: missing files, size mismatches, hash mismatches,
            invalid paths in manifest.
        """
        if self._integrity_issues is not None:
            return self._integrity_issues

        manifest_data = self._manifest_data
        file_inventory = manifest_data.get("file_inventory", [])

        # Convert manifest entries to FileEntry objects
        entries = []
        for entry_dict in file_inventory:
            entries.append(
                FileEntry(
                    path=entry_dict.get("path", ""),
                    size_bytes=entry_dict.get("size_bytes", 0),
                    content_hash=entry_dict.get("content_hash", ""),
                )
            )

        # Check file inventory
        self._integrity_issues = check_file_inventory(self.artifact_dir, entries)
        return self._integrity_issues

    def list_scores(self) -> Sequence[SemanticScore]:
        """Read all scores from scores.jsonl with typed deserialization.

        Validates integrity before reading. Returns typed SemanticScore objects
        with all fields validated.

        Returns:
            List of parsed, validated SemanticScore objects.

        Raises:
            SemanticScoringRunError: If integrity check fails or deserialization fails.
        """
        if self._scores is not None:
            return self._scores

        # Verify integrity first
        issues = self.verify_integrity()
        if issues:
            raise SemanticScoringRunError(f"artifact integrity check failed: {issues}")

        scores_path = self.artifact_dir / "outputs" / "scores.jsonl"
        if not scores_path.exists():
            self._scores = []
            return self._scores

        scores = []
        try:
            with scores_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise SemanticScoringRunError(
                            f"invalid JSON in scores.jsonl line {line_num}: {e}"
                        ) from e
                    try:
                        score = decode_semantic_score(data)
                        scores.append(score)
                    except SemanticScoringRunError as e:
                        raise SemanticScoringRunError(
                            f"cannot decode score at line {line_num}: {e}"
                        ) from e
        except OSError as e:
            raise SemanticScoringRunError(f"cannot read scores.jsonl: {e}") from e

        self._scores = scores
        return self._scores

    def iter_scores(self) -> Iterator[SemanticScore]:
        """Yield every score, in the order it was recorded, one line at a time.

        A run of a long sequence holds more scores than a consumer needs in memory at
        once, and an evaluation that joins them against a reference walks them in order.
        Integrity is verified once, before the first line is read, so a tampered run is
        refused rather than streamed.

        Yields:
            Each decoded score, in publication order.

        Raises:
            SemanticScoringRunError: If the inventory does not match the files on disk, or
                a line is not a decodable score.
        """
        issues = self.verify_integrity()
        if issues:
            raise SemanticScoringRunError(f"artifact integrity check failed: {issues}")

        scores_path = self.artifact_dir / "outputs" / "scores.jsonl"
        if not scores_path.exists():
            return

        try:
            with scores_path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        yield decode_semantic_score(json.loads(line))
                    except (json.JSONDecodeError, SemanticScoringRunError) as error:
                        raise SemanticScoringRunError(
                            f"cannot decode score at line {line_num}: {error}"
                        ) from error
        except OSError as error:
            raise SemanticScoringRunError(f"cannot read scores.jsonl: {error}") from error

    def read_manifest(self) -> SemanticScoringRunManifest:
        """Read and parse the run manifest.

        This is an alias for the manifest property.

        Returns:
            The parsed manifest.
        """
        return self._manifest

    def read_scores(self) -> Sequence[SemanticScore]:
        """Read all scores from scores.jsonl (alias for list_scores).

        This is an alias for list_scores() for backwards compatibility.

        Returns:
            List of parsed SemanticScore objects.
        """
        return self.list_scores()
