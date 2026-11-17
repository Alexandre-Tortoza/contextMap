"""Independent visual evidence artifacts with lazy payloads and audited sampling geometry.

The writer owns only visual features. Regions, masks and semantic claims stay in their
producer artifacts, selected explicitly through the manifest's upstream identities.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, NewType, TypeVar
from uuid import uuid4

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    IneligibleRegion,
    decode_ineligible_region,
    decode_prepared_image_reference,
    encode_ineligible_region,
    encode_prepared_image_reference,
)
from contextmap.shared import (
    BackendProvenance,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
    file_entry,
    publish_run_directory,
)
from contextmap.visual_perception.execution.stage_execution import StageOutcome
from contextmap.visual_perception.feature_diagnostics import (
    FEATURE_METRICS_PATH,
    DenseFeatureDiagnostic,
    FeatureDebugLevel,
    FeatureDiagnosticPreview,
    FeatureExtractionDiagnostic,
    dense_feature_map_from_record,
    dense_feature_map_of,
    write_feature_diagnostics,
)
from contextmap.visual_perception.models import PreparedImage
from contextmap.visual_perception.visual_features.feature_payload_store import (
    FEATURE_INDEX_FILENAME,
    FeatureStoreReader,
    FeatureStoreWriter,
    write_feature_index,
)
from contextmap.visual_perception.visual_features.region_feature_pooling import DenseFeatureMap
from contextmap.visual_perception.visual_features.visual_evidence import (
    FeatureId,
    FeatureScope,
    VisualFeature,
    decode_feature,
    embedding_space_fingerprint,
    encode_feature,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

VisualEvidenceRunId = NewType("VisualEvidenceRunId", str)

VISUAL_EVIDENCE_SCHEMA_VERSION = "0.1.0"
_RESULTS_FILENAME = "outputs/results.jsonl"
_METRICS_FILENAME = "metrics/stage-timings.jsonl"
_FEATURES_DIRNAME = "outputs/features"
_Record = TypeVar("_Record")


class VisualEvidenceRunError(Exception):
    """Visual evidence cannot be published or read as a valid artifact."""


@dataclass(frozen=True, kw_only=True)
class VisualEvidenceResult:
    """Visual features produced for one physical observation, including empty outcomes."""

    source_observation_id: SourceObservationId
    features: tuple[VisualFeature, ...] = ()
    prepared_image: PreparedImage | None = None

    def __post_init__(self) -> None:
        """Freeze feature metadata and reject ambiguous or foreign observation identities."""
        object.__setattr__(self, "features", tuple(self.features))
        if not str(self.source_observation_id).strip():
            raise ValueError("source_observation_id must not be empty")
        if len({item.feature_id for item in self.features}) != len(self.features):
            raise ValueError("duplicate feature_id in visual evidence")
        if any(item.source_observation_id != self.source_observation_id for item in self.features):
            raise ValueError("feature source_observation_id must match visual evidence")
        if (
            self.prepared_image is not None
            and self.prepared_image.source_observation_id != self.source_observation_id
        ):
            raise ValueError("prepared image source_observation_id must match visual evidence")


@dataclass(frozen=True, kw_only=True)
class VisualEvidenceRunManifest:
    """Identity, selection, producer configuration and inventory of visual evidence."""

    run_id: str
    sequence_artifact_id: str
    region_artifact_id: str
    selection_id: str
    configuration_digest: str
    created_at: str
    source_observation_ids: tuple[str, ...] = ()
    run_index: int = 0
    sequence_name: str = ""
    code_version: str | None = None
    configuration: Mapping[str, Any] | None = None
    backend_provenance: Mapping[str, BackendProvenance] | None = None
    feature_count: int = 0
    file_inventory: tuple[FileEntry, ...] = ()
    contract: str = "VisualEvidenceArtifact"
    schema_version: str = VISUAL_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze observation identities and require explicit upstream lineage."""
        object.__setattr__(self, "source_observation_ids", tuple(self.source_observation_ids))
        for name in (
            "run_id",
            "sequence_artifact_id",
            "region_artifact_id",
            "selection_id",
            "configuration_digest",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.run_index < 0:
            raise ValueError("run_index must be non-negative")
        if len(set(self.source_observation_ids)) != len(self.source_observation_ids):
            raise ValueError("duplicate source_observation_id in visual manifest")

    @property
    def result_count(self) -> int:
        """Return the number of explicitly recorded physical observations."""
        return len(self.source_observation_ids)


@dataclass(frozen=True, kw_only=True)
class RecordedDenseFeatureMap:
    """Persisted extraction stage and the reconstructed image-to-feature geometry."""

    stage_id: str
    dense_map: DenseFeatureMap


class VisualEvidenceRunWriter:
    """Stage payloads immediately and publish visual evidence once, atomically."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        sequence_artifact_id: str,
        region_artifact_id: str,
        selection_id: str,
        configuration_digest: str,
        code_version: str | None = None,
        run_index: int = 0,
        sequence_name: str = "",
        configuration: Mapping[str, Any] | None = None,
        backend_provenance: Mapping[str, BackendProvenance] | None = None,
        feature_debug_level: FeatureDebugLevel = FeatureDebugLevel.NONE,
    ) -> None:
        """Bind this producer to its explicit upstreams and final output directory."""
        self._manifest = VisualEvidenceRunManifest(
            run_id=str(run_id),
            sequence_artifact_id=sequence_artifact_id,
            region_artifact_id=region_artifact_id,
            selection_id=selection_id,
            configuration_digest=configuration_digest,
            created_at="",
            run_index=run_index,
            sequence_name=sequence_name,
            code_version=code_version,
            configuration=configuration,
            backend_provenance=backend_provenance,
        )
        self._run_id = str(run_id)
        self._final_dir = Path(output_dir)
        self._tmp_dir = self._final_dir.parent / f".tmp-{self._final_dir.name}-{uuid4().hex[:8]}"
        self._results: list[VisualEvidenceResult] = []
        self._source_observation_ids: set[SourceObservationId] = set()
        self._feature_ids: set[FeatureId] = set()
        self._ineligible_regions: dict[tuple[str, str, str, str | None], IneligibleRegion] = {}
        self._stage_outcome_records: list[dict[str, Any]] = []
        self._feature_store: FeatureStoreWriter | None = None
        self._feature_payloads: list[tuple[VisualFeature, SourceObservationId]] = []
        self._feature_diagnostics: list[FeatureExtractionDiagnostic] = []
        self._feature_previews: list[FeatureDiagnosticPreview] = []
        self._feature_debug_level = feature_debug_level
        self._finalized = False
        self._discarded = False

    def __enter__(self) -> VisualEvidenceRunWriter:
        """Return the writer without allocating payload memory or creating the final path."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard staging when the caller leaves without successful publication."""
        if not self._finalized:
            self._discard_staging()

    def _ensure_staging(self) -> Path:
        """Create private staging once; a failed writer cannot be resumed implicitly."""
        if self._discarded:
            raise VisualEvidenceRunError("visual evidence staging was discarded")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        return self._tmp_dir

    def _discard_staging(self) -> None:
        """Remove every unpublished payload and prohibit reuse of this failed writer."""
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._feature_store = None
        self._discarded = True

    def add_result(self, result: VisualEvidenceResult) -> None:
        """Record visual metadata for one observation, rejecting duplicate producer IDs."""
        if self._finalized or self._discarded:
            raise VisualEvidenceRunError("cannot add results after finalize or discard")
        if result.source_observation_id in self._source_observation_ids:
            raise VisualEvidenceRunError("duplicate source_observation_id in visual run")
        if any(item.feature_id in self._feature_ids for item in result.features):
            raise VisualEvidenceRunError("duplicate feature_id across visual observations")
        self._source_observation_ids.add(result.source_observation_id)
        self._feature_ids.update(item.feature_id for item in result.features)
        self._results.append(result)

    def add_stage_outcomes(
        self,
        outcomes: Sequence[StageOutcome],
        source_observation_id: SourceObservationId | None = None,
    ) -> None:
        """Retain only lightweight timing/error metadata, never stage output tensors."""
        if self._finalized or self._discarded:
            raise VisualEvidenceRunError("cannot add outcomes after finalize or discard")
        self._stage_outcome_records.extend(
            {
                "source_observation_id": (
                    None if source_observation_id is None else str(source_observation_id)
                ),
                "stage_id": item.stage_id,
                "status": item.status.value,
                "duration_ms": item.duration_ms,
                "error": item.error,
                "error_type": item.error_type,
                "error_traceback": item.error_traceback,
            }
            for item in outcomes
        )

    def add_ineligible_region(self, record: IneligibleRegion) -> None:
        """Record an explicit regional feature skip without copying the upstream region."""
        if self._finalized or self._discarded:
            raise VisualEvidenceRunError("cannot add ineligible regions after finalize or discard")
        key = (
            str(record.source_observation_id),
            str(record.region_id),
            record.stage_id,
            None if record.source_feature_id is None else str(record.source_feature_id),
        )
        if key in self._ineligible_regions:
            raise VisualEvidenceRunError("duplicate ineligible region for this feature stage")
        self._ineligible_regions[key] = record

    def add_feature_payload(
        self,
        feature: VisualFeature,
        source_observation_id: SourceObservationId,
        array: NDArray[Any],
    ) -> None:
        """Queue a feature's numerical payload to be persisted by :meth:`finalize`.

        Persisting a feature's payload is opt-in per feature: a
        ``VisualFeature`` whose payload was never queued here still
        appears in ``outputs/results.jsonl`` with its metadata, just
        without a loadable array in this run's feature store (see
        :mod:`contextmap.visual_perception.artifacts.feature_payload_store`).

        Args:
            feature: The feature this array belongs to. Its ``shape``
                and ``dtype`` must match ``array`` exactly (validated at
                :meth:`finalize` time).
            source_observation_id: The physical observation the feature
                was produced for.
            array: The array to persist.

        Raises:
            VisualEvidenceRunError: If called after :meth:`finalize`.
        """
        if self._finalized or self._discarded:
            raise VisualEvidenceRunError("cannot add feature payloads after finalize or discard")
        if source_observation_id != feature.source_observation_id:
            raise VisualEvidenceRunError("feature payload source_observation_id must match feature")
        try:
            if self._feature_store is None:
                self._feature_store = FeatureStoreWriter(self._ensure_staging() / _FEATURES_DIRNAME)
            self._feature_store.write(feature, source_observation_id, array)
        except BaseException:
            self._discard_staging()
            raise
        # Só a metadata fica retida: as validações de finalize() nunca leem o array.
        self._feature_payloads.append((feature, source_observation_id))

    def staged_feature_payload(
        self, source_observation_id: SourceObservationId, feature_id: FeatureId
    ) -> Any:
        """Load one hash-verified payload during this transaction without retaining its array.

        Derived feature stages use the staging store after a stage-major producer has released
        its numerical state. The unpublished artifact remains inaccessible to downstream readers.
        """
        if self._finalized or self._discarded or self._feature_store is None:
            raise VisualEvidenceRunError("no active staged feature payload store")
        return FeatureStoreReader(
            self._ensure_staging() / _FEATURES_DIRNAME, self._feature_store.entries()
        ).load(source_observation_id, feature_id)

    def add_feature_diagnostic(self, diagnostic: FeatureExtractionDiagnostic) -> None:
        """Queue one structured Feature Extraction audit event.

        Raises:
            VisualEvidenceRunError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise VisualEvidenceRunError("cannot add feature diagnostics after finalize()")
        self._feature_diagnostics.append(diagnostic)

    def add_feature_preview(self, preview: FeatureDiagnosticPreview) -> None:
        """Queue one small human-only preview controlled by the debug level.

        Raises:
            VisualEvidenceRunError: If called after :meth:`finalize`.
        """
        if self._finalized:
            raise VisualEvidenceRunError("cannot add feature previews after finalize()")
        self._feature_previews.append(preview)

    def _result_features_by_key(
        self,
    ) -> dict[tuple[SourceObservationId, FeatureId], list[VisualFeature]]:
        """Index every result feature by ``(source_observation_id, feature_id)``."""
        features_by_key: dict[tuple[SourceObservationId, FeatureId], list[VisualFeature]] = {}
        for result in self._results:
            for feature in result.features:
                key = (result.source_observation_id, feature.feature_id)
                features_by_key.setdefault(key, []).append(feature)
        return features_by_key

    def _validate_feature_payload_references(self) -> None:
        """Require each queued payload to match one feature in its owning result."""
        features_by_key = self._result_features_by_key()

        metadata_fields = (
            "scope",
            "embedding_space_id",
            "shape",
            "dtype",
            "normalization",
            "payload_reference",
            "region_id",
            "provenance",
        )
        for feature, source_observation_id in self._feature_payloads:
            key = (source_observation_id, feature.feature_id)
            matches = features_by_key.get(key, [])
            if len(matches) != 1:
                raise VisualEvidenceRunError(
                    "queued feature payload does not resolve to exactly one result feature: "
                    f"source_observation_id={source_observation_id!r}, "
                    f"feature_id={feature.feature_id!r}, matches={len(matches)}"
                )
            result_feature = matches[0]
            for field_name in metadata_fields:
                queued_value = getattr(feature, field_name)
                result_value = getattr(result_feature, field_name)
                if queued_value != result_value:
                    raise VisualEvidenceRunError(
                        "queued feature payload metadata disagrees with result feature: "
                        f"{field_name} {queued_value!r} != {result_value!r} for "
                        f"source_observation_id={source_observation_id!r}, "
                        f"feature_id={feature.feature_id!r}"
                    )

    def _validate_feature_diagnostics(self) -> None:
        """Require each produced feature diagnostic to describe one persisted result feature.

        ``metrics/feature-extraction.jsonl`` is contractual: its ``dense`` record rebuilds the
        feature-to-image mapping after the run is reopened. A record that only hashes correctly
        could belong to another payload, so every ``SUCCEEDED``/``WARNING`` diagnostic is bound
        to exactly one ``VisualFeature`` of this run (same observation and feature identity),
        and to at most one diagnostic per feature so the geometry is unambiguous.
        """
        features_by_key = self._result_features_by_key()
        described: set[tuple[SourceObservationId, FeatureId]] = set()
        for diagnostic in self._feature_diagnostics:
            feature_id = diagnostic.feature_id
            # FeatureExtractionDiagnostic garante feature_id nos eventos produzidos; o teste
            # de None só estreita o tipo para o índice.
            if not diagnostic.produced_feature or feature_id is None:
                continue
            key = (diagnostic.source_observation_id, feature_id)
            matches = features_by_key.get(key, [])
            if len(matches) != 1:
                raise VisualEvidenceRunError(
                    "feature diagnostic does not resolve to exactly one result feature: "
                    f"event_id={diagnostic.event_id!r}, "
                    f"source_observation_id={diagnostic.source_observation_id!r}, "
                    f"feature_id={feature_id!r}, matches={len(matches)}"
                )
            if key in described:
                raise VisualEvidenceRunError(
                    "more than one feature diagnostic describes the same result feature: "
                    f"source_observation_id={diagnostic.source_observation_id!r}, "
                    f"feature_id={feature_id!r}"
                )
            described.add(key)
            self._validate_diagnostic_matches_feature(diagnostic, matches[0])

    def _validate_diagnostic_matches_feature(
        self, diagnostic: FeatureExtractionDiagnostic, feature: VisualFeature
    ) -> None:
        """Compare a produced diagnostic with its result feature and, for dense, its geometry."""
        space = diagnostic.embedding_space
        comparisons = (
            ("scope", diagnostic.scope, feature.scope),
            ("output_shape", diagnostic.output_shape, feature.shape),
            ("dtype", diagnostic.dtype, feature.dtype),
            ("normalization", diagnostic.normalization, feature.normalization),
            ("payload_reference", diagnostic.payload_reference, feature.payload_reference),
            ("backend provenance", diagnostic.backend, feature.provenance),
            (
                "embedding_space fingerprint",
                embedding_space_fingerprint(space) if space is not None else None,
                feature.embedding_space_id,
            ),
        )
        context = (
            f"event_id={diagnostic.event_id!r}, "
            f"source_observation_id={diagnostic.source_observation_id!r}, "
            f"feature_id={feature.feature_id!r}"
        )
        for name, declared, persisted in comparisons:
            if declared != persisted:
                raise VisualEvidenceRunError(
                    "feature diagnostic disagrees with its result feature: "
                    f"{name} {declared!r} != {persisted!r} for {context}"
                )
        region = diagnostic.region
        if region is not None and region.region_id != feature.region_id:
            raise VisualEvidenceRunError(
                "feature diagnostic disagrees with its result feature: "
                f"region_id {region.region_id!r} != {feature.region_id!r} for {context}"
            )
        if region is not None and region.source_feature_id is not None:
            source_matches = self._result_features_by_key().get(
                (diagnostic.source_observation_id, region.source_feature_id), []
            )
            if len(source_matches) != 1 or source_matches[0].scope is not FeatureScope.DENSE:
                raise VisualEvidenceRunError(
                    "pooled feature diagnostic source_feature_id does not resolve to exactly one "
                    f"dense feature for {context}"
                )
            source = source_matches[0]
            if region.source_artifact_id != str(self._run_id):
                raise VisualEvidenceRunError(
                    "pooled feature diagnostic source_artifact_id does not point to this run "
                    f"for {context}"
                )
            if region.source_payload_reference != source.payload_reference:
                raise VisualEvidenceRunError(
                    "pooled feature diagnostic source_payload_reference does not match its "
                    f"dense feature for {context}"
                )
            if feature.embedding_space_id != source.embedding_space_id:
                raise VisualEvidenceRunError(
                    "pooled feature must preserve its dense source embedding_space_id "
                    f"for {context}"
                )
            if feature.dtype != source.dtype:
                raise VisualEvidenceRunError(
                    f"pooled feature must preserve its dense source dtype for {context}"
                )
            if feature.normalization != source.normalization:
                raise VisualEvidenceRunError(
                    f"pooled feature must preserve its dense source normalization for {context}"
                )
            if len(feature.shape) != 1 or not source.shape or feature.shape[0] != source.shape[-1]:
                raise VisualEvidenceRunError(
                    "pooled feature shape must be one-dimensional and preserve the dense "
                    f"source channel count for {context}"
                )
        if diagnostic.dense is not None:
            self._validate_dense_geometry(diagnostic, diagnostic.dense, feature, context)

    def _validate_dense_geometry(
        self,
        diagnostic: FeatureExtractionDiagnostic,
        dense: DenseFeatureDiagnostic,
        feature: VisualFeature,
        context: str,
    ) -> None:
        """Require the persisted dense geometry to rebuild the feature's ``DenseFeatureMap``."""
        if dense.source_artifact_id != str(self._run_id):
            raise VisualEvidenceRunError(
                "dense feature diagnostic source_artifact_id does not point to the run that "
                f"owns the feature: {dense.source_artifact_id!r} != {str(self._run_id)!r} "
                f"for {context}"
            )
        # Reconstrói o mapa com a mesma regra do leitor (VisualEvidenceRunReader.dense_feature_map):
        # assim shape 3-D e (grid_height, grid_width) == feature.shape[:2] são as regras de
        # DenseFeatureMap, e o que o writer aceita é exatamente o que o leitor reabre.
        try:
            dense_feature_map_of(
                feature,
                dense,
                source_image_width=diagnostic.source_image_width,
                source_image_height=diagnostic.source_image_height,
            )
        except ValueError as error:
            raise VisualEvidenceRunError(
                f"dense feature diagnostic does not rebuild a valid DenseFeatureMap: {error} "
                f"for {context}"
            ) from error

    def finalize(self) -> VisualEvidenceRunManifest:
        """Validate all visual references, inventory payloads and publish without replacement."""
        if self._finalized or self._discarded:
            raise VisualEvidenceRunError("writer already finalized or discarded")
        try:
            if self._final_dir.exists():
                raise VisualEvidenceRunError(
                    f"visual evidence artifact already exists: {self._final_dir}"
                )
            self._validate_feature_payload_references()
            self._validate_feature_diagnostics()
            for item in self._ineligible_regions.values():
                if item.source_observation_id not in self._source_observation_ids:
                    raise VisualEvidenceRunError("ineligible region references unknown observation")
                if item.source_feature_id is not None:
                    matches = self._result_features_by_key().get(
                        (item.source_observation_id, item.source_feature_id), []
                    )
                    if len(matches) != 1 or matches[0].scope is not FeatureScope.DENSE:
                        raise VisualEvidenceRunError(
                            "ineligible region references unknown dense source feature"
                        )
            self._ensure_staging()
            inventory: list[FileEntry] = []
            self._write_stream(
                _RESULTS_FILENAME, (_encode_result(item) for item in self._results), inventory
            )
            self._write_stream(_METRICS_FILENAME, iter(self._stage_outcome_records), inventory)
            if self._ineligible_regions:
                self._write_stream(
                    "outputs/ineligible-regions.jsonl",
                    (encode_ineligible_region(item) for item in self._ineligible_regions.values()),
                    inventory,
                )
            feature_root = self._tmp_dir / _FEATURES_DIRNAME
            feature_root.mkdir(parents=True, exist_ok=True)
            entries = [] if self._feature_store is None else self._feature_store.entries()
            write_feature_index(feature_root, entries)
            for entry in entries:
                inventory.append(
                    FileEntry(
                        path=f"{_FEATURES_DIRNAME}/{entry.payload_reference}",
                        size_bytes=entry.size_bytes,
                        content_hash=entry.content_hash,
                    )
                )
            index_path = f"{_FEATURES_DIRNAME}/{FEATURE_INDEX_FILENAME}"
            inventory.append(file_entry(index_path, (self._tmp_dir / index_path).read_bytes()))
            diagnostic_paths = write_feature_diagnostics(
                run_root=self._tmp_dir,
                diagnostics=self._feature_diagnostics,
                previews=self._feature_previews,
                debug_level=self._feature_debug_level,
            )
            for relative in diagnostic_paths:
                inventory.append(file_entry(relative, (self._tmp_dir / relative).read_bytes()))
            manifest = replace(
                self._manifest,
                created_at=datetime.now(UTC).isoformat(),
                source_observation_ids=tuple(
                    str(item.source_observation_id) for item in self._results
                ),
                feature_count=sum(len(item.features) for item in self._results),
                file_inventory=tuple(sorted(inventory, key=lambda item: item.path)),
            )
            (self._tmp_dir / "manifest.json").write_text(
                json.dumps(_encode_manifest(manifest), sort_keys=True, indent=2), encoding="utf-8"
            )
            (self._tmp_dir / "README.md").write_text(
                f"# Evidência visual\n\nRun: `{manifest.run_id}`.\n\n"
                f"Regiões upstream: `{manifest.region_artifact_id}`.\n",
                encoding="utf-8",
            )
            problems = check_file_inventory(self._tmp_dir, manifest.file_inventory)
            if problems:
                raise VisualEvidenceRunError(f"internal consistency check failed: {problems}")
            publish_run_directory(self._tmp_dir, self._final_dir)
        except BaseException as error:
            self._discard_staging()
            if isinstance(error, RunDirectoryError):
                raise VisualEvidenceRunError(
                    f"cannot publish visual evidence artifact: {error}"
                ) from error
            raise
        self._finalized = True
        return manifest

    def _write_stream(
        self, relative: str, records: Iterator[dict[str, Any]], inventory: list[FileEntry]
    ) -> None:
        """Write one lightweight contractual stream without retaining numerical payloads."""
        content = "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
        path = self._tmp_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        inventory.append(file_entry(relative, content.encode("utf-8")))


class VisualEvidenceRunReader:
    """Read lightweight visual metadata independently of feature tensors and upstreams."""

    def __init__(self, run_dir: Path) -> None:
        """Open a complete artifact and verify its required contractual inventory."""
        self._root = Path(run_dir)
        try:
            record = json.loads((self._root / "manifest.json").read_text(encoding="utf-8"))
            self._manifest = _decode_manifest(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise VisualEvidenceRunError(f"invalid visual evidence manifest: {error}") from error
        required = {
            _RESULTS_FILENAME,
            _METRICS_FILENAME,
            f"{_FEATURES_DIRNAME}/{FEATURE_INDEX_FILENAME}",
        }
        inventoried = {item.path for item in self._manifest.file_inventory}
        if not required <= inventoried:
            raise VisualEvidenceRunError(
                f"missing contractual files in manifest: {required - inventoried}"
            )
        self._feature_store_reader: FeatureStoreReader | None = None
        self._dense_geometry: dict[tuple[str, str], tuple[str, dict[str, Any]]] | None = None

    @property
    def manifest(self) -> VisualEvidenceRunManifest:
        """Return identity, explicit upstream references and persisted inventory."""
        return self._manifest

    def feature_store(self) -> FeatureStoreReader:
        """Open the feature index lazily, without loading numerical arrays."""
        if self._feature_store_reader is None:
            self._feature_store_reader = FeatureStoreReader.open(self._root / _FEATURES_DIRNAME)
        return self._feature_store_reader

    def iter_results(self) -> Iterator[VisualEvidenceResult]:
        """Yield one observation's visual metadata at a time, without loading tensors."""
        seen: set[str] = set()
        for _, _, item in self._read_records(_RESULTS_FILENAME, "visual evidence", _decode_result):
            observation_id = str(item.source_observation_id)
            if observation_id in seen or observation_id not in self.manifest.source_observation_ids:
                raise VisualEvidenceRunError("duplicate or undeclared visual observation")
            seen.add(observation_id)
            yield item
        if seen != set(self.manifest.source_observation_ids):
            raise VisualEvidenceRunError(
                "visual evidence observation coverage disagrees with manifest"
            )

    def list_results(self) -> list[VisualEvidenceResult]:
        """Return every observation's lightweight visual metadata."""
        return list(self.iter_results())

    def result(self, source_observation_id: SourceObservationId) -> VisualEvidenceResult:
        """Read the visual evidence produced for one declared physical observation."""
        for item in self.iter_results():
            if item.source_observation_id == source_observation_id:
                return item
        raise VisualEvidenceRunError(f"no result for observation {source_observation_id!r}")

    def list_ineligible_regions(self) -> list[IneligibleRegion]:
        """Materialize the consumer's lightweight exclusion diagnostics."""
        return list(self.iter_ineligible_regions())

    def iter_ineligible_regions(self) -> Iterator[IneligibleRegion]:
        """Yield explicit regional feature skips without inferring absent evidence."""
        filename = "outputs/ineligible-regions.jsonl"
        if not any(entry.path == filename for entry in self.manifest.file_inventory):
            return
        for _, _, item in self._read_records(
            filename, "ineligible region", decode_ineligible_region
        ):
            if str(item.source_observation_id) not in self.manifest.source_observation_ids:
                raise VisualEvidenceRunError("ineligible region references unknown observation")
            yield item

    def verify_integrity(self) -> list[str]:
        """Verify persisted sizes and hashes without loading numerical arrays into memory."""
        return check_file_inventory(self._root, self.manifest.file_inventory)

    def _read_records(
        self, filename: str, kind: str, decode: Callable[[dict[str, Any]], _Record]
    ) -> Iterator[tuple[str, dict[str, Any], _Record]]:
        """Decode one contractual JSON Lines stream of this run, a line at a time.

        Args:
            filename: The stream, relative to the run directory.
            kind: What one record is, for the error message.
            decode: Turns one JSON object into its contract.

        Yields:
            The ``"<filename>:<line>"`` location, the JSON object and what it decodes to.

        Raises:
            VisualEvidenceRunError: If a line is not JSON, not a JSON object or does
                not decode into its contract; the error names the file and the line, and keeps
                the decoding error as its cause.
        """
        with (self._root / filename).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                location = f"{filename}:{line_number}"
                try:
                    record = json.loads(stripped)
                    if not isinstance(record, dict):
                        raise TypeError(f"expected a JSON object, got {type(record).__name__}")
                    decoded = decode(record)
                # Um valor aninhado de outra forma falha no decoder como AttributeError (um
                # `.get` em uma string, por exemplo): também é uma linha malformada.
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    raise VisualEvidenceRunError(
                        f"invalid {kind} record at {location}: {type(error).__name__}: {error}"
                    ) from error
                yield location, record, decoded

    def dense_feature_map(
        self, source_observation_id: SourceObservationId, feature: VisualFeature
    ) -> RecordedDenseFeatureMap:
        """Rebuild the ``DenseFeatureMap`` of one dense feature of this run.

        The geometry is the feature's contractual ``dense`` record in
        ``metrics/feature-extraction.jsonl``, and the map is rebuilt with
        :func:`~contextmap.visual_perception.feature_diagnostics.dense_feature_map_of`, the
        rule :meth:`VisualEvidenceRunWriter.finalize` validated it with before publishing.
        Nothing under ``__DEBUG__/`` is read and no geometry is ever assumed. The recorded
        geometries are indexed on the first call, one small record per dense feature.

        Args:
            source_observation_id: The observation whose result holds the feature.
            feature: The dense feature, as this run's result holds it (for example from
                :meth:`iter_results`).

        Returns:
            The rebuilt map, with the stage that extracted the feature.

        Raises:
            VisualEvidenceRunError: If ``feature`` is not dense, the run recorded no dense geometry
                for it (its extractor reported none, or the run predates the geometry), or
                the recorded geometry does not rebuild a valid map with it.
        """
        if feature.scope is not FeatureScope.DENSE:
            raise VisualEvidenceRunError(
                f"feature {feature.feature_id!r} is not a dense feature ({feature.scope.value})"
            )
        entry = self._dense_geometries().get((str(source_observation_id), str(feature.feature_id)))
        if entry is None:
            raise VisualEvidenceRunError(
                f"no recorded dense geometry for feature {feature.feature_id!r} of observation "
                f"{source_observation_id!r}: {FEATURE_METRICS_PATH} does not describe it"
            )
        stage_id, geometry = entry
        try:
            dense_map = dense_feature_map_from_record(feature, geometry)
        except (KeyError, TypeError, ValueError) as error:
            raise VisualEvidenceRunError(
                f"the recorded dense geometry of feature {feature.feature_id!r} of observation "
                f"{source_observation_id!r} does not rebuild a DenseFeatureMap: "
                f"{type(error).__name__}: {error}"
            ) from error
        return RecordedDenseFeatureMap(stage_id=stage_id, dense_map=dense_map)

    def _dense_geometries(self) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
        """Index the recorded dense geometry of each produced feature, once per reader.

        Returns:
            ``(stage_id, dense record)`` by ``(source_observation_id, feature_id)``; empty
            when the run has no ``metrics/feature-extraction.jsonl``.

        Raises:
            VisualEvidenceRunError: If a record is malformed or two records describe one feature.
        """
        if self._dense_geometry is not None:
            return self._dense_geometry
        index: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        if (self._root / FEATURE_METRICS_PATH).is_file():
            for location, _, entry in self._read_records(
                FEATURE_METRICS_PATH, "feature extraction metric", _decode_dense_geometry
            ):
                if entry is None:
                    continue
                key, stage_id, geometry = entry
                if key in index:
                    raise VisualEvidenceRunError(
                        f"more than one dense geometry for feature {key[1]!r} of observation "
                        f"{key[0]!r} at {location}"
                    )
                index[key] = (stage_id, geometry)
        self._dense_geometry = index
        return index


def _decode_dense_geometry(
    record: dict[str, Any],
) -> tuple[tuple[str, str], str, dict[str, Any]] | None:
    """Read the dense geometry one ``metrics/feature-extraction.jsonl`` record carries.

    Returns:
        ``((source_observation_id, feature_id), stage_id, dense record)``, or ``None`` for an
        event that produced no feature or carries no dense geometry.

    Raises:
        KeyError: If the record lacks a key the lookup reads.
        TypeError: If its ``dense`` value is not an object.
    """
    if record["status"] not in {"succeeded", "warning"} or record["dense"] is None:
        return None
    geometry = record["dense"]
    if not isinstance(geometry, dict):
        raise TypeError(f"dense must be an object, got {type(geometry).__name__}")
    key = (str(record["source_observation_id"]), str(record["feature_id"]))
    return key, str(record["stage_id"]), geometry


def _encode_result(result: VisualEvidenceResult) -> dict[str, Any]:
    """Persist geometry/preparation metadata without upstream masks or temporary paths."""
    image = result.prepared_image
    if image is not None:
        transformations = tuple(
            replace(
                item,
                output_image=replace(
                    item.output_image,
                    uri=f"source-observation:{result.source_observation_id}:preparation:{index}",
                ),
            )
            for index, item in enumerate(image.transformations)
        )
        image = replace(
            image,
            payload_reference=f"source-observation:{result.source_observation_id}",
            payload_artifact=None,
            transformations=transformations,
            valid_region=None,
            exclusion_regions=(),
        )
    return {
        "source_observation_id": str(result.source_observation_id),
        "features": [encode_feature(item) for item in result.features],
        "prepared_image": None if image is None else encode_prepared_image_reference(image),
    }


def _decode_result(record: dict[str, Any]) -> VisualEvidenceResult:
    """Decode one typed observation envelope without fetching its numerical payloads."""
    return VisualEvidenceResult(
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        features=tuple(decode_feature(item) for item in record["features"]),
        prepared_image=None
        if record["prepared_image"] is None
        else decode_prepared_image_reference(record["prepared_image"]),
    )


def _encode_manifest(manifest: VisualEvidenceRunManifest) -> dict[str, Any]:
    """Render the complete typed manifest, with a contractual artifact discriminator."""
    record = asdict(manifest)
    return record


def _decode_manifest(record: dict[str, Any]) -> VisualEvidenceRunManifest:
    """Reject other artifacts and historical schemas rather than inferring compatibility."""
    if record["contract"] != "VisualEvidenceArtifact":
        raise ValueError("manifest contract must be VisualEvidenceArtifact")
    if record["schema_version"] != VISUAL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported visual evidence schema: {record['schema_version']}")
    decoded = dict(record)
    decoded["file_inventory"] = tuple(FileEntry(**item) for item in record["file_inventory"])
    provenance = record["backend_provenance"]
    decoded["backend_provenance"] = (
        None
        if provenance is None
        else {name: BackendProvenance(**item) for name, item in provenance.items()}
    )
    return VisualEvidenceRunManifest(**decoded)
