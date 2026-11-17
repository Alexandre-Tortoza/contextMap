"""Immutable semantic producer artifacts, including exact inputs and every attempt."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, TracebackType
from typing import Any, TypeVar

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import IneligibleRegion
from contextmap.semantic_interpretation.interpretation_audit import (
    SemanticDebugLevel,
    redact_semantic_secrets,
    write_semantic_audit,
)
from contextmap.semantic_interpretation.interpretation_execution import (
    FailedSemanticBackendCall,
    FailedSemanticInterpretation,
    SemanticInterpretationExecution,
    decode_failed_semantic_backend_call,
    decode_failed_semantic_interpretation,
    decode_semantic_execution,
    encode_failed_semantic_backend_call,
    encode_failed_semantic_interpretation,
    encode_semantic_execution,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationRequest,
    SemanticVisualView,
)
from contextmap.semantic_interpretation.scene_context import SceneContext
from contextmap.semantic_interpretation.semantic_claims import (
    SemanticClaim,
    SemanticInterpretationRunId,
)
from contextmap.semantic_interpretation.serialization import (
    decode_claim,
    decode_scene_context,
    encode_claim,
    encode_scene_context,
)
from contextmap.shared import (
    AtomicRunDirectory,
    BackendProvenance,
    FileEntry,
    RunDirectoryError,
    check_file_inventory,
)

SEMANTIC_INTERPRETATION_SCHEMA_VERSION = "0.1.0"
OBSERVATIONS = "outputs/semantic-observations.jsonl"
EXECUTIONS = "outputs/semantic-interpretations.jsonl"
FAILURES = "outputs/semantic-interpretation-failures.jsonl"
BACKEND_FAILURES = "outputs/semantic-backend-call-failures.jsonl"
T = TypeVar("T")


class SemanticInterpretationRunError(ValueError):
    """Semantic evidence cannot be published or read with its declared lineage."""


class SemanticObservationStatus(Enum):
    """Whether an observation completed, failed, or was explicitly skipped."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, kw_only=True)
class SemanticObservation:
    """Semantic evidence for one physical observation from one semantic producer."""

    source_observation_id: SourceObservationId
    semantic_interpretation_run_id: SemanticInterpretationRunId
    claims: tuple[SemanticClaim, ...] = ()
    scene_context: SceneContext | None = None
    created_at: str = ""
    status: SemanticObservationStatus = SemanticObservationStatus.SUCCEEDED

    def __post_init__(self) -> None:
        """Freeze claims and require every item to name this observation and producer."""
        object.__setattr__(self, "claims", tuple(self.claims))
        evidence = [*self.claims]
        if self.scene_context is not None:
            evidence.extend(self.scene_context.claims)
        owned: list[SemanticClaim | SceneContext] = list(evidence)
        if self.scene_context is not None:
            owned.append(self.scene_context)
        for item in owned:
            if item.source_observation_id != self.source_observation_id:
                raise SemanticInterpretationRunError(
                    "semantic evidence source_observation_id mismatch"
                )
            if item.semantic_interpretation_run_id != self.semantic_interpretation_run_id:
                raise SemanticInterpretationRunError("semantic evidence producer run mismatch")
        if len({claim.claim_id for claim in evidence}) != len(evidence):
            raise SemanticInterpretationRunError("duplicate semantic claim_id in observation")


def _encode_observation(observation: SemanticObservation) -> dict[str, Any]:
    """Encode one semantic observation without upstream region or feature copies."""
    return {
        "source_observation_id": str(observation.source_observation_id),
        "semantic_interpretation_run_id": str(observation.semantic_interpretation_run_id),
        "created_at": observation.created_at,
        "status": observation.status.value,
        "claims": [encode_claim(item) for item in observation.claims],
        "scene_context": None
        if observation.scene_context is None
        else encode_scene_context(observation.scene_context),
    }


def _decode_observation(record: dict[str, Any]) -> SemanticObservation:
    """Decode and validate one semantic observation record."""
    return SemanticObservation(
        source_observation_id=SourceObservationId(record["source_observation_id"]),
        semantic_interpretation_run_id=SemanticInterpretationRunId(
            record["semantic_interpretation_run_id"]
        ),
        created_at=record["created_at"],
        status=SemanticObservationStatus(record["status"]),
        claims=tuple(decode_claim(item) for item in record["claims"]),
        scene_context=None
        if record["scene_context"] is None
        else decode_scene_context(record["scene_context"]),
    )


@dataclass(frozen=True, kw_only=True)
class SemanticInterpretationRunManifest:
    """Identity, immutable upstream selection, code/configuration and coverage of a run."""

    run_id: SemanticInterpretationRunId
    run_index: int
    sequence_name: str
    sequence_artifact_id: str
    selection_id: str
    region_artifact_id: str
    configuration_digest: str
    created_at: str
    source_observation_ids: tuple[str, ...] = ()
    enabled_capabilities: frozenset[str] = frozenset({"semantic_interpreter"})
    backend_provenance: Mapping[str, BackendProvenance] = field(default_factory=dict)
    effective_configuration: Mapping[str, Any] = field(default_factory=dict)
    code_version: str | None = None
    code_digest: str | None = None
    observation_count: int = 0
    claim_count: int = 0
    execution_count: int = 0
    failure_count: int = 0
    file_inventory: tuple[FileEntry, ...] = ()
    schema_version: str = SEMANTIC_INTERPRETATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject unsupported schemas and freeze mutable producer metadata."""
        if self.schema_version != SEMANTIC_INTERPRETATION_SCHEMA_VERSION:
            raise SemanticInterpretationRunError(
                f"unsupported semantic schema_version: {self.schema_version}"
            )
        if self.run_index < 0:
            raise SemanticInterpretationRunError("run_index must be non-negative")
        for name in (
            "run_id",
            "sequence_artifact_id",
            "selection_id",
            "region_artifact_id",
            "configuration_digest",
        ):
            if not str(getattr(self, name)).strip():
                raise SemanticInterpretationRunError(f"{name} must not be empty")
        object.__setattr__(self, "source_observation_ids", tuple(self.source_observation_ids))
        object.__setattr__(self, "enabled_capabilities", frozenset(self.enabled_capabilities))
        object.__setattr__(
            self, "backend_provenance", MappingProxyType(dict(self.backend_provenance))
        )
        object.__setattr__(
            self,
            "effective_configuration",
            MappingProxyType(redact_semantic_secrets(dict(self.effective_configuration))),
        )

    @property
    def artifact_id(self) -> str:
        """Return the immutable identity selected by downstream stages."""
        return str(self.run_id)

    def to_dict(self) -> dict[str, Any]:
        """Encode metadata; publication adds its actual file inventory."""
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "file_inventory"
        }
        result["artifact_kind"] = "SemanticInterpretationArtifact"
        result["run_id"] = str(self.run_id)
        result["source_observation_ids"] = list(self.source_observation_ids)
        result["enabled_capabilities"] = sorted(self.enabled_capabilities)
        result["backend_provenance"] = {
            name: asdict(value) for name, value in self.backend_provenance.items()
        }
        result["effective_configuration"] = dict(self.effective_configuration)
        return result

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> SemanticInterpretationRunManifest:
        """Decode a manifest and reject a different artifact contract."""
        if record.get("artifact_kind") != "SemanticInterpretationArtifact":
            raise SemanticInterpretationRunError("not a SemanticInterpretationArtifact")
        values = dict(record)
        values.pop("artifact_kind")
        values["run_id"] = SemanticInterpretationRunId(values["run_id"])
        values["backend_provenance"] = {
            name: BackendProvenance(**item) for name, item in values["backend_provenance"].items()
        }
        values["file_inventory"] = tuple(FileEntry(**item) for item in values["file_inventory"])
        return cls(**values)


def _execution_with_reference(
    execution: SemanticInterpretationExecution,
) -> SemanticInterpretationExecution:
    """Materialize the contractual raw-response reference on claims and context."""

    def claim(item: SemanticClaim) -> SemanticClaim:
        return replace(item, provenance=replace(item.provenance, raw_response_reference=EXECUTIONS))

    context = execution.parsed.scene_context
    return replace(
        execution,
        parsed=replace(
            execution.parsed,
            claims=tuple(claim(item) for item in execution.parsed.claims),
            scene_context=None
            if context is None
            else replace(
                context,
                provenance=replace(context.provenance, raw_response_reference=EXECUTIONS),
                claims=tuple(claim(item) for item in context.claims),
            ),
        ),
    )


class SemanticInterpretationRunWriter:
    """Publish semantic evidence atomically, preserving views and all attempt outcomes."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_name: str,
        run_id: SemanticInterpretationRunId,
        run_index: int,
        sequence_artifact_id: str,
        selection_id: str,
        region_artifact_id: str,
        configuration_digest: str,
        enabled_capabilities: frozenset[str] = frozenset({"semantic_interpreter"}),
        backend_provenance: Mapping[str, BackendProvenance] | None = None,
        effective_configuration: Mapping[str, Any] | None = None,
        code_version: str | None = None,
        code_digest: str | None = None,
        semantic_debug_level: SemanticDebugLevel = SemanticDebugLevel.FULL,
    ) -> None:
        """Bind explicit producer identity and upstream selection; create no files yet."""
        self._manifest = SemanticInterpretationRunManifest(
            run_id=run_id,
            run_index=run_index,
            sequence_name=sequence_name,
            sequence_artifact_id=sequence_artifact_id,
            selection_id=selection_id,
            region_artifact_id=region_artifact_id,
            configuration_digest=configuration_digest,
            created_at=datetime.now(UTC).isoformat(),
            enabled_capabilities=enabled_capabilities,
            backend_provenance=backend_provenance or {},
            effective_configuration=effective_configuration or {},
            code_version=code_version,
            code_digest=code_digest,
        )
        self._output_dir = Path(output_dir)
        self._directory: AtomicRunDirectory | None = None
        self._observations: dict[SourceObservationId, SemanticObservation] = {}
        self._executions: list[SemanticInterpretationExecution] = []
        self._failures: list[FailedSemanticInterpretation] = []
        self._backend_failures: list[FailedSemanticBackendCall] = []
        self._request_ids: set[str] = set()
        self._views: dict[str, str] = {}
        self._ineligible: list[IneligibleRegion] = []
        self._timings: list[dict[str, Any]] = []
        self._debug_level = semantic_debug_level
        self._finalized = False
        self._discarded = False

    def __enter__(self) -> SemanticInterpretationRunWriter:
        """Return this streaming producer writer."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard every unpublished payload if execution stops or fails."""
        if not self._finalized:
            self._discarded = True
            if self._directory is not None:
                self._directory.discard()

    def _require_open(self) -> None:
        """Refuse changes after publication or aborted context management."""
        if self._finalized or self._discarded:
            raise SemanticInterpretationRunError(
                "cannot modify a finalized or discarded semantic run"
            )

    def _staging(self) -> AtomicRunDirectory:
        """Open publication staging lazily so payloads leave memory immediately."""
        self._require_open()
        if self._directory is None:
            try:
                self._directory = AtomicRunDirectory(self._output_dir)
                self._directory.__enter__()
            except RunDirectoryError as error:
                raise SemanticInterpretationRunError(str(error)) from error
        return self._directory

    def add_observation(self, observation: SemanticObservation) -> None:
        """Record one observation, including a successful empty or failed result."""
        self._require_open()
        if observation.semantic_interpretation_run_id != self._manifest.run_id:
            raise SemanticInterpretationRunError("observation producer run mismatch")
        if observation.source_observation_id in self._observations:
            raise SemanticInterpretationRunError("duplicate semantic source_observation_id")
        self._observations[observation.source_observation_id] = observation

    def _register(self, request: SemanticInterpretationRequest) -> None:
        """Require one producer-local request identity across all outcome streams."""
        self._require_open()
        if request.semantic_interpretation_run_id != self._manifest.run_id:
            raise SemanticInterpretationRunError("request producer run mismatch")
        if str(request.request_id) in self._request_ids:
            raise SemanticInterpretationRunError("duplicate semantic request_id")
        self._request_ids.add(str(request.request_id))

    def add_semantic_execution(self, execution: SemanticInterpretationExecution) -> None:
        """Record parsed output and its exact raw response for publication."""
        self._register(execution.request)
        self._executions.append(execution)

    def add_failed_semantic_interpretation(self, failed: FailedSemanticInterpretation) -> None:
        """Record an observed answer rejected by parsing, keeping its raw response."""
        self._register(failed.request)
        self._failures.append(failed)

    def add_failed_semantic_backend_call(self, failed: FailedSemanticBackendCall) -> None:
        """Record a call that never answered, independently from parsing failures."""
        self._register(failed.request)
        self._backend_failures.append(failed)

    def add_semantic_view_payload(self, view: SemanticVisualView, payload: bytes) -> None:
        """Stage exactly the model input pixels; verify their hash before retaining metadata."""
        self._require_open()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != view.sha256:
            raise SemanticInterpretationRunError("semantic view payload hash mismatch")
        existing = self._views.get(view.payload_reference)
        if existing is not None:
            if existing != digest:
                raise SemanticInterpretationRunError("conflicting semantic view payload")
            return
        self._staging().write_bytes(view.payload_reference, payload)
        self._views[view.payload_reference] = digest

    def add_ineligible_region(self, record: IneligibleRegion) -> None:
        """Preserve the reason a frozen upstream region did not receive a request."""
        self._require_open()
        if record in self._ineligible:
            raise SemanticInterpretationRunError("duplicate semantic ineligible region")
        self._ineligible.append(record)

    def add_stage_outcomes(self, outcomes: Sequence[Any]) -> None:
        """Preserve per-stage timings/status and executions emitted by the stage runner."""
        self._require_open()
        for outcome in outcomes:
            self._timings.append(
                {
                    name: getattr(outcome, name)
                    for name in (
                        "stage_id",
                        "duration_ms",
                        "error",
                        "error_type",
                        "error_traceback",
                    )
                }
                | {"status": outcome.status.value}
            )
            if (
                isinstance(outcome.output, SemanticInterpretationExecution)
                and outcome.status.value == "succeeded"
            ):
                self.add_semantic_execution(outcome.output)

    def _validate(self) -> None:
        """Reject dangling attempts/views, fabricated raw references and lost parsed outputs."""
        attempts: list[
            SemanticInterpretationExecution
            | FailedSemanticInterpretation
            | FailedSemanticBackendCall
        ] = [*self._executions, *self._failures, *self._backend_failures]
        requests = [item.request for item in attempts]
        referenced: dict[str, str] = {}
        for request in requests:
            if request.source_observation_id not in self._observations:
                raise SemanticInterpretationRunError("semantic request has no observation record")
            for view in request.visual_views:
                if (
                    view.payload_reference in referenced
                    and referenced[view.payload_reference] != view.sha256
                ):
                    raise SemanticInterpretationRunError("conflicting semantic view hashes")
                referenced[view.payload_reference] = view.sha256
            if request.scene_context is not None:
                observation = self._observations[request.source_observation_id]
                if request.scene_context != observation.scene_context:
                    raise SemanticInterpretationRunError(
                        "request scene_context was not materialized"
                    )
        if referenced != self._views:
            raise SemanticInterpretationRunError("missing or unreferenced semantic view payload")
        backed_claims: dict[str, SemanticClaim] = {}
        backed_contexts: dict[str, SceneContext] = {}
        for execution in self._executions:
            observation = self._observations[execution.request.source_observation_id]
            for claim in execution.parsed.claims:
                if claim not in observation.claims:
                    raise SemanticInterpretationRunError(
                        "semantic execution claims were not materialized"
                    )
                backed_claims[str(claim.claim_id)] = claim
            if execution.parsed.scene_context is not None:
                if execution.parsed.scene_context != observation.scene_context:
                    raise SemanticInterpretationRunError(
                        "semantic execution scene_context was not materialized"
                    )
                backed_contexts[observation.scene_context.context_id] = observation.scene_context
            for provenance in execution.parsed.provenances():
                if provenance.raw_response_reference not in (None, EXECUTIONS):
                    raise SemanticInterpretationRunError(
                        "raw_response_reference must name contractual execution stream"
                    )
        for observation in self._observations.values():
            if any(str(item.claim_id) not in backed_claims for item in observation.claims):
                raise SemanticInterpretationRunError("semantic claim has no producing execution")
            if (
                observation.scene_context is not None
                and observation.scene_context.context_id not in backed_contexts
            ):
                raise SemanticInterpretationRunError(
                    "semantic scene context has no producing execution"
                )

    def finalize(self) -> SemanticInterpretationRunManifest:
        """Validate and atomically publish every contractual stream, even empty failures."""
        self._require_open()
        self._validate()
        directory = self._staging()
        executions = [_execution_with_reference(item) for item in self._executions]
        persisted_claims = {
            str(claim.claim_id): claim for item in executions for claim in item.parsed.claims
        }
        persisted_contexts = {
            item.parsed.scene_context.context_id: item.parsed.scene_context
            for item in executions
            if item.parsed.scene_context is not None
        }
        observations = [
            replace(
                item,
                claims=tuple(persisted_claims[str(claim.claim_id)] for claim in item.claims),
                scene_context=None
                if item.scene_context is None
                else persisted_contexts[item.scene_context.context_id],
            )
            for item in self._observations.values()
        ]
        failures = [
            replace(item, provenance=replace(item.provenance, raw_response_reference=FAILURES))
            for item in self._failures
        ]
        streams = {
            OBSERVATIONS: [_encode_observation(item) for item in observations],
            EXECUTIONS: [
                encode_semantic_execution(item, raw_response_reference=EXECUTIONS)
                for item in executions
            ],
            FAILURES: [encode_failed_semantic_interpretation(item) for item in failures],
            BACKEND_FAILURES: [
                encode_failed_semantic_backend_call(item) for item in self._backend_failures
            ],
            "metrics/stage-timings.jsonl": self._timings,
            "outputs/ineligible-regions.jsonl": [
                {
                    "source_observation_id": str(item.source_observation_id),
                    "region_id": str(item.region_id),
                    "stage_id": item.stage_id,
                    "reason": item.reason.value,
                    "detail": item.detail,
                }
                for item in self._ineligible
            ],
        }
        for path, records in streams.items():
            directory.write_text(
                path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
            )
        with TemporaryDirectory(
            prefix=".semantic-audit-", dir=self._output_dir.parent
        ) as temporary:
            audit_root = Path(temporary)
            attempts: list[
                SemanticInterpretationExecution
                | FailedSemanticInterpretation
                | FailedSemanticBackendCall
            ] = [*executions, *failures, *self._backend_failures]
            for item in attempts:
                for path in write_semantic_audit(
                    run_root=audit_root, execution=item, debug_level=self._debug_level
                ):
                    directory.write_bytes(path, (audit_root / path).read_bytes(), contractual=False)
        manifest = replace(
            self._manifest,
            source_observation_ids=tuple(str(item.source_observation_id) for item in observations),
            observation_count=len(observations),
            claim_count=sum(
                len(item.claims)
                + (0 if item.scene_context is None else len(item.scene_context.claims))
                for item in observations
            ),
            execution_count=len(executions),
            failure_count=len(failures) + len(self._backend_failures),
            file_inventory=directory.inventory(),
        )
        directory.publish(
            manifest=manifest.to_dict(),
            readme=(
                "# SemanticInterpretationArtifact\n\n"
                "Claims, contexto, requests, views e respostas desta execução.\n"
            ),
        )
        self._finalized = True
        return manifest


class SemanticInterpretationRunReader:
    """Read semantic producer evidence lazily, without loading model runtimes."""

    def __init__(self, artifact_dir: str | Path) -> None:
        """Open a finalized artifact and reject absent or incompatible manifests."""
        self.artifact_dir = Path(artifact_dir)
        try:
            record = json.loads((self.artifact_dir / "manifest.json").read_text())
            self._manifest = SemanticInterpretationRunManifest.from_dict(record)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SemanticInterpretationRunError(
                f"cannot read semantic manifest: {error}"
            ) from error

    @property
    def manifest(self) -> SemanticInterpretationRunManifest:
        """Return the producer's manifest and explicit regional upstream."""
        return self._manifest

    def read_manifest(self) -> SemanticInterpretationRunManifest:
        """Return the typed manifest."""
        return self._manifest

    def verify_integrity(self) -> list[str]:
        """Verify hashes afresh, detecting corruption after an earlier read."""
        return check_file_inventory(self.artifact_dir, self._manifest.file_inventory)

    def _read(self, path: str, decoder: Callable[[dict[str, Any]], T]) -> Iterator[T]:
        """Require and hash-check a contractual stream before decoding its records."""
        matches = [item for item in self.manifest.file_inventory if item.path == path]
        if len(matches) != 1 or not (self.artifact_dir / path).is_file():
            raise SemanticInterpretationRunError(f"missing semantic contractual stream: {path}")
        issues = check_file_inventory(self.artifact_dir, tuple(matches))
        if issues:
            raise SemanticInterpretationRunError("; ".join(issues))
        with (self.artifact_dir / path).open() as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    yield decoder(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise SemanticInterpretationRunError(f"{path}:{line_no}: {error}") from error

    def iter_observations(self) -> Iterator[SemanticObservation]:
        """Stream successful empty, populated, failed and skipped observation records."""
        yield from self._read(OBSERVATIONS, _decode_observation)

    def list_observations(self) -> list[SemanticObservation]:
        """Return all semantic evidence records in publication order."""
        return list(self.iter_observations())

    def iter_semantic_executions(self) -> Iterator[SemanticInterpretationExecution]:
        """Stream parsed attempts with the exact raw response and model diagnostics."""
        yield from self._read(EXECUTIONS, decode_semantic_execution)

    def list_semantic_executions(self) -> list[SemanticInterpretationExecution]:
        """Return all parsed attempts."""
        return list(self.iter_semantic_executions())

    def iter_failed_semantic_interpretations(self) -> Iterator[FailedSemanticInterpretation]:
        """Stream observed answers rejected during parsing."""
        yield from self._read(FAILURES, decode_failed_semantic_interpretation)

    def iter_failed_semantic_backend_calls(self) -> Iterator[FailedSemanticBackendCall]:
        """Stream calls that failed before producing a response."""
        yield from self._read(BACKEND_FAILURES, decode_failed_semantic_backend_call)

    def tracks_semantic_failures(self) -> bool:
        """Require the always-present parsing-failure stream and report full tracking."""
        list(self._read(FAILURES, decode_failed_semantic_interpretation))
        return True

    def records_semantic_backend_call_failures(self) -> bool:
        """Require the always-present backend-failure stream and report full tracking."""
        list(self._read(BACKEND_FAILURES, decode_failed_semantic_backend_call))
        return True
